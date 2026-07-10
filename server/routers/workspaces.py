"""Workspace registry API: list / create / soft-delete / purge / restore, plus the per-workspace
status overview.

These endpoints manage the registry of workspaces themselves (rows in rag_workspaces + the physical
LightRAG data behind each). Shared package state — the DB pool and the cached RAGAnything instances —
is read from the `server` package at call time so the test suite's patches are honoured.
"""

import re
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi import Path as PathParam

import server
from server.deps import _is_valid_slug
from server.schemas import WorkspaceCreate
from server.workspaces import _get_ws_lock

router = APIRouter()

# Graph storage namespace; a workspace `w` gets AGE graph `{w}_chunk_entity_relation`
# (see LightRAG PGGraphStorage._get_workspace_graph_name). The bootstrap `default` workspace,
# whose physical name is POSTGRES_WORKSPACE, uses the bare `chunk_entity_relation` graph.
_AGE_NAMESPACE = "chunk_entity_relation"


def _workspace_public(row) -> dict:
    ca = row["created_at"]
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "created_at": ca.isoformat() if hasattr(ca, "isoformat") else ca,
    }


async def _db_get_workspace_any(pool, workspace_id: str):
    """Fetch a workspace row regardless of soft-delete state (for create/delete/restore)."""
    return await pool.fetchrow(
        "SELECT id, name, description, lightrag_workspace, deleted_at "
        "FROM rag_workspaces WHERE id = $1",
        workspace_id,
    )


async def _purge_workspace_data(pool, physical_workspace: str) -> None:
    """Irreversibly delete one physical workspace's LightRAG data, file metadata, and files.
    Scoped strictly to `physical_workspace`: every DELETE filters on `workspace = $1` and only
    that workspace's dedicated AGE graph is dropped, so purging one workspace never touches
    another's rows."""
    # 1. Delete this workspace's rows from every lightrag_* table that has a workspace column.
    tables = await pool.fetch(
        r"SELECT table_name FROM information_schema.columns "
        r"WHERE table_schema = 'public' AND column_name = 'workspace' "
        r"AND table_name LIKE 'lightrag\_%'"
    )
    for t in tables:
        await pool.execute(
            f'DELETE FROM public."{t["table_name"]}" WHERE workspace = $1', physical_workspace
        )
    # 2. Drop the workspace's dedicated AGE graph, if it was ever created.
    graph_name = f"{re.sub(r'[^a-zA-Z0-9_]', '_', physical_workspace)}_{_AGE_NAMESPACE}"
    if await pool.fetchval("SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1", graph_name):
        async with pool.acquire() as conn:
            await conn.execute("LOAD 'age'")
            await conn.execute("SET search_path = ag_catalog, public")
            await conn.execute(f"SELECT drop_graph('{graph_name}', true)")
    # 3. Delete file metadata rows and the on-disk files directory.
    await pool.execute("DELETE FROM rag_file_metadata WHERE workspace = $1", physical_workspace)
    shutil.rmtree(Path(server.WORKING_DIR) / physical_workspace, ignore_errors=True)


async def _evict_workspace_instance(workspace_id: str) -> None:
    """Drop a workspace's cached instance and finalize its storages (no-op if not built)."""
    lock = await _get_ws_lock(workspace_id)
    async with lock:
        instance = server._rag_instances.pop(workspace_id, None)
    if instance is not None:
        await instance.lightrag.finalize_storages()


@router.get(
    "/all-workspaces/list",
    summary="List workspaces",
    description="List active workspaces, or pass `deleted=true` to list soft-deleted ones instead.",
    responses={503: {"description": "Database not initialised yet"}},
)
async def list_workspaces(
    deleted: bool = Query(
        False, description="If true, return soft-deleted workspaces instead of active ones."
    ),
):
    if server._db_pool is None:
        raise HTTPException(503, "DB not initialised")
    cond = "deleted_at IS NOT NULL" if deleted else "deleted_at IS NULL"
    rows = await server._db_pool.fetch(
        f"SELECT id, name, description, created_at FROM rag_workspaces WHERE {cond} ORDER BY created_at"
    )
    return {"workspaces": [_workspace_public(r) for r in rows]}


@router.post(
    "/all-workspaces/create",
    summary="Create a workspace",
    description=(
        "Create a new isolated workspace. The `id` becomes both the public slug and the storage "
        "namespace (its LightRAG `workspace`). Fails if the id is malformed or already in use."
    ),
    responses={
        409: {"description": "A workspace with this id already exists"},
        422: {"description": "Invalid workspace id (must match ^[a-z][a-z0-9_]{0,47}$)"},
        503: {"description": "Database not initialised yet"},
    },
)
async def create_workspace(body: WorkspaceCreate):
    if server._db_pool is None:
        raise HTTPException(503, "DB not initialised")
    if not _is_valid_slug(body.id):
        raise HTTPException(422, "Invalid workspace id: must match ^[a-z][a-z0-9_]{0,47}$")
    if await _db_get_workspace_any(server._db_pool, body.id) is not None:
        raise HTTPException(409, f"Workspace {body.id!r} already exists")
    await server._db_pool.execute(
        """INSERT INTO rag_workspaces (id, name, description, lightrag_workspace)
               VALUES ($1, $2, $3, $4)""",
        body.id,
        body.name,
        body.description,
        body.id,  # lightrag_workspace := id
    )
    return {"id": body.id, "name": body.name, "description": body.description}


@router.delete(
    "/workspace/{workspace_id}",
    summary="Delete a workspace (soft-delete or purge)",
    description=(
        "By default this is a reversible **soft delete** (hidden from listings, restorable via "
        "`/workspace/{id}/restore`). Pass `purge=true` to **irreversibly** delete the workspace's "
        "graph, vector rows, file metadata, and on-disk files. Any workspace can be deleted."
    ),
    responses={
        404: {"description": "Workspace not found"},
        503: {"description": "Database not initialised yet"},
    },
)
async def delete_workspace(
    workspace_id: str = PathParam(description="Public workspace id (slug) to delete."),
    purge: bool = Query(
        False, description="If true, irreversibly purge all data instead of soft-deleting."
    ),
):
    if server._db_pool is None:
        raise HTTPException(503, "DB not initialised")
    row = await _db_get_workspace_any(server._db_pool, workspace_id)
    if row is None:
        raise HTTPException(404, f"Workspace {workspace_id!r} not found")
    if purge:
        await _evict_workspace_instance(workspace_id)
        await _purge_workspace_data(server._db_pool, row["lightrag_workspace"])
        await server._db_pool.execute("DELETE FROM rag_workspaces WHERE id = $1", workspace_id)
        return {"status": "purged", "id": workspace_id}
    await server._db_pool.execute(
        "UPDATE rag_workspaces SET deleted_at = NOW() WHERE id = $1", workspace_id
    )
    await _evict_workspace_instance(workspace_id)
    return {"status": "soft-deleted", "id": workspace_id}


@router.post(
    "/workspace/{workspace_id}/restore",
    summary="Restore a soft-deleted workspace",
    description="Un-delete a workspace that was previously soft-deleted. No effect on purged workspaces.",
    responses={
        404: {"description": "No soft-deleted workspace with this id to restore"},
        503: {"description": "Database not initialised yet"},
    },
)
async def restore_workspace(
    workspace_id: str = PathParam(description="Public workspace id (slug) to restore."),
):
    if server._db_pool is None:
        raise HTTPException(503, "DB not initialised")
    row = await _db_get_workspace_any(server._db_pool, workspace_id)
    if row is None or row["deleted_at"] is None:
        raise HTTPException(404, f"No soft-deleted workspace {workspace_id!r} to restore")
    await server._db_pool.execute(
        "UPDATE rag_workspaces SET deleted_at = NULL WHERE id = $1", workspace_id
    )
    return {"status": "restored", "id": workspace_id}


async def _count_vdb(phys: str, kind: str) -> int | None:
    """Distinct entity/relationship count from the dedup'd vector tables, discovering the
    embedding-suffixed table name dynamically (robust to the configured embedding model)."""
    try:
        tbl = await server._db_pool.fetchval(
            "SELECT table_name FROM information_schema.tables WHERE table_name LIKE $1 LIMIT 1",
            f"lightrag_vdb_{kind}%",
        )
        if not tbl:
            return None
        return await server._db_pool.fetchval(
            f"SELECT count(*) FROM {tbl} WHERE workspace=$1", phys
        )
    except Exception:
        return None


@router.get(
    "/workspace/{workspace_id}",
    summary="Workspace status (overview + counts)",
    description=(
        "Return a single overview of a workspace: whether it is active (vs soft-deleted), corpus counts "
        "(documents by status, chunks, distinct entities/relationships), and an ingest-job summary. "
        "Read-only and cheap. This is the 'is this workspace healthy, how much is in it?' check — "
        "distinct from the per-job route `/workspace/{id}/status/{job_id}`."
    ),
    responses={
        404: {"description": "Workspace not found (never existed or was purged)"},
        503: {"description": "Database not initialised yet"},
    },
)
async def workspace_status(
    workspace_id: str = PathParam(description="Public workspace id (slug)."),
):
    if server._db_pool is None:
        raise HTTPException(503, "DB not initialised")
    if not _is_valid_slug(workspace_id):
        raise HTTPException(404, f"Workspace {workspace_id!r} not found")
    row = await _db_get_workspace_any(server._db_pool, workspace_id)
    if row is None:
        raise HTTPException(404, f"Workspace {workspace_id!r} not found")
    phys = row["lightrag_workspace"]
    doc_rows = await server._db_pool.fetch(
        "SELECT status, count(*) AS n FROM lightrag_doc_status WHERE workspace=$1 GROUP BY status",
        phys,
    )
    docs_by_status = {r["status"]: r["n"] for r in doc_rows}
    chunks = await server._db_pool.fetchval(
        "SELECT COALESCE(SUM(chunks_count),0) FROM lightrag_doc_status WHERE workspace=$1", phys
    )
    job_rows = await server._db_pool.fetch(
        "SELECT status, count(*) AS n FROM rag_file_metadata WHERE workspace=$1 GROUP BY status",
        phys,
    )
    last_uploaded = await server._db_pool.fetchval(
        "SELECT MAX(uploaded_at) FROM rag_file_metadata WHERE workspace=$1", phys
    )
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "active": row["deleted_at"] is None,
        "documents": {"by_status": docs_by_status, "total": sum(docs_by_status.values())},
        "chunks": int(chunks or 0),
        "entities": await _count_vdb(phys, "entity"),
        "relationships": await _count_vdb(phys, "relation"),
        "ingest": {
            "by_status": {r["status"]: r["n"] for r in job_rows},
            "last_uploaded_at": (
                last_uploaded.isoformat() if hasattr(last_uploaded, "isoformat") else last_uploaded
            ),
        },
    }
