"""Per-workspace RAGAnything/LightRAG instance registry.

Builds and caches one RAGAnything instance per public workspace id, lazily on first use, guarded
by a per-workspace lock. The instance cache, lock table, and asyncpg pool are shared runtime state
owned by the package root (server.*) and read here at call time.
"""

import asyncio
from pathlib import Path

from fastapi import HTTPException
from raganything import RAGAnything

import server
from server import config
from server.graph_store import graph_storage_ready, graph_transaction
from server.llm import _embedding_func, _llm_func, _vision_func


def _workspace_public(row) -> dict:
    ca = row["created_at"]
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "created_at": ca.isoformat() if hasattr(ca, "isoformat") else ca,
    }


async def list_workspace_records(*, deleted: bool = False) -> dict:
    if server._db_pool is None:
        raise HTTPException(503, "DB not initialised")
    cond = "deleted_at IS NOT NULL" if deleted else "deleted_at IS NULL"
    rows = await server._db_pool.fetch(
        f"SELECT id, name, description, created_at FROM rag_workspaces WHERE {cond} ORDER BY created_at"
    )
    return {"workspaces": [_workspace_public(r) for r in rows]}


async def _get_ws_lock(workspace_id: str) -> asyncio.Lock:
    """Get (or lazily create) the per-workspace lock. Guards instance creation and
    serialises inserts for that workspace."""
    async with server._registry_lock:
        return server._ws_locks.setdefault(workspace_id, asyncio.Lock())


async def _lookup_workspace(workspace_id: str):
    """Return the rag_workspaces row for an ACTIVE (not soft-deleted) workspace, else None."""
    if server._db_pool is None:
        return None
    return await server._db_pool.fetchrow(
        "SELECT id, name, description, lightrag_workspace "
        "FROM rag_workspaces WHERE id = $1 AND deleted_at IS NULL",
        workspace_id,
    )


async def _build_workspace_rag(workspace_id: str, physical_workspace: str) -> RAGAnything:
    """Construct + initialize a RAGAnything/LightRAG pair for one workspace.
    `workspace_id` is the public id (names the on-disk working_dir); `physical_workspace`
    is the LightRAG `workspace=` value that namespaces Postgres rows + the AGE graph."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    working_dir = str(Path(config.WORKING_DIR) / workspace_id)
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    embedding_func = EmbeddingFunc(
        embedding_dim=config.EMBEDDING_DIM,
        max_token_size=8192,
        model_name=config.EMBEDDING_MODEL,
        func=_embedding_func,
    )
    lightrag_instance = LightRAG(
        working_dir=working_dir,
        llm_model_func=_llm_func,
        embedding_func=embedding_func,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="PGGraphStorage",
        doc_status_storage="PGDocStatusStorage",
        workspace=physical_workspace,
    )
    await lightrag_instance.initialize_storages()
    if graph_storage_ready():
        # LightRAG still persists its chunks, document status and AGE graph. The service owns
        # vector publication and backfill across both imported facts and manual records.
        async def defer_vector_upsert(data):
            return None

        for storage in (
            lightrag_instance.chunks_vdb,
            lightrag_instance.entities_vdb,
            lightrag_instance.relationships_vdb,
        ):
            storage.upsert = defer_vector_upsert
    return RAGAnything(
        llm_model_func=_llm_func,
        vision_model_func=_vision_func,
        embedding_func=embedding_func,
        lightrag=lightrag_instance,
    )


async def get_workspace_rag(workspace_id: str) -> RAGAnything:
    """Return the cached RAGAnything for a workspace, building it lazily on first use.
    Raises HTTPException(404) if the workspace is unknown or soft-deleted."""
    cached = server._rag_instances.get(workspace_id)
    if cached is not None:
        return cached
    lock = await _get_ws_lock(workspace_id)
    async with lock:
        cached = server._rag_instances.get(workspace_id)
        if cached is not None:
            return cached
        row = await _lookup_workspace(workspace_id)
        if row is None:
            raise HTTPException(404, f"Workspace {workspace_id!r} not found")
        instance = await _build_workspace_rag(workspace_id, row["lightrag_workspace"])
        server._rag_instances[workspace_id] = instance
        return instance


async def _count_vdb(phys: str, kind: str) -> int | None:
    """Distinct entity/relationship count from the dedup'd vector tables, discovering the
    embedding-suffixed table name dynamically (robust to the configured embedding model)."""
    pool = server._db_pool
    if pool is None:
        return None
    try:
        tbl = await pool.fetchval(
            "SELECT table_name FROM information_schema.tables WHERE table_name LIKE $1 LIMIT 1",
            f"lightrag_vdb_{kind}%",
        )
        if not tbl:
            return None
        return await pool.fetchval(f"SELECT count(*) FROM {tbl} WHERE workspace=$1", phys)
    except Exception:
        return None


async def get_workspace_overview(row) -> dict:
    if server._db_pool is None:
        raise HTTPException(503, "DB not initialised")
    phys = row["lightrag_workspace"]
    graph_enabled = graph_storage_ready()
    has_docs = not graph_enabled or await server._db_pool.fetchval(
        "SELECT to_regclass('public.lightrag_doc_status') IS NOT NULL"
    )
    doc_rows = (
        await server._db_pool.fetch(
            "SELECT status, count(*) AS n FROM lightrag_doc_status WHERE workspace=$1 GROUP BY status",
            phys,
        )
        if has_docs
        else []
    )
    docs_by_status = {r["status"]: r["n"] for r in doc_rows}
    chunks = (
        await server._db_pool.fetchval(
            "SELECT COALESCE(SUM(chunks_count),0) FROM lightrag_doc_status WHERE workspace=$1", phys
        )
        if has_docs
        else 0
    )
    job_rows = await server._db_pool.fetch(
        "SELECT status, count(*) AS n FROM rag_file_metadata WHERE workspace=$1 GROUP BY status",
        phys,
    )
    last_uploaded = await server._db_pool.fetchval(
        "SELECT MAX(uploaded_at) FROM rag_file_metadata WHERE workspace=$1", phys
    )
    if graph_enabled and row.get("deleted_at") is None:
        async with graph_transaction(dict(row)) as conn:
            graph_counts = await conn.fetch(
                "SELECT kind,count(*) AS n FROM rag_graph_active WHERE workspace=$1 GROUP BY kind",
                row["id"],
            )
        counts = {r["kind"]: r["n"] for r in graph_counts}
        entities, relationships = counts.get("entity", 0), counts.get("relationship", 0)
        chunks = counts.get("chunk", 0)
    else:
        entities, relationships = await _count_vdb(phys, "entity"), await _count_vdb(
            phys, "relation"
        )
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "active": row.get("deleted_at") is None,
        "documents": {"by_status": docs_by_status, "total": sum(docs_by_status.values())},
        "chunks": int(chunks or 0),
        "entities": entities,
        "relationships": relationships,
        "ingest": {
            "by_status": {r["status"]: r["n"] for r in job_rows},
            "last_uploaded_at": (
                last_uploaded.isoformat() if hasattr(last_uploaded, "isoformat") else last_uploaded
            ),
        },
    }
