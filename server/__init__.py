import asyncio
import base64
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam  # aliased: `Path` is pathlib.Path throughout this module
from fastapi.responses import JSONResponse
from raganything import RAGAnything

# Import config BEFORE the heavy libraries below: importing it runs logging.basicConfig, so
# LightRAG / RAG-Anything don't emit unconfigured logs at import time. Every config name is
# re-exported through this module so `server.<CONST>` — and the test suite's monkeypatches —
# keep working; functions that read config live (e.g. _active_llm_cfg) are defined in
# server.config and patched there.
from server.config import (
    API_TOKENS,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
    PRIMARY_WORKSPACE_ID,
    WORKING_DIR,
)

_job_queue: asyncio.Queue = asyncio.Queue()
_jobs: dict[str, dict] = {}
_batches: dict[str, list] = {}
_db_pool = None  # asyncpg.Pool, set in lifespan

# Per-workspace RAGAnything instance registry. Keyed by PUBLIC workspace id.
_rag_instances: dict[str, RAGAnything] = {}
_ws_locks: dict[str, asyncio.Lock] = (
    {}
)  # one lock per workspace: guards creation AND serialises inserts
_registry_lock = asyncio.Lock()  # guards the dicts above


# --- LLM / embedding shims ---
# Defined in server.llm (they read endpoint config live from server.config); re-exported here
# because _build_workspace_rag wires them into LightRAG / RAG-Anything.
# --- Document processing ---
# Parsing + ingestion live in server.ingest. Re-exported here because upload_batch builds file
# metadata/paths (_build_metadata, _join_path) and the background worker calls _process_file;
# IngestionIncompleteError is re-exported for callers/tests that reference server.<name>.
# --- DB helpers ---
# Schema init + file/job metadata persistence live in server.db (they take the pool explicitly).
# --- Workspace instance registry ---
# Per-workspace RAGAnything instances are built/cached in server.workspaces (reading the shared
# _rag_instances/_ws_locks/_registry_lock/_db_pool from this module). Endpoints use get_workspace_rag;
# require_workspace resolves rows via the module (workspaces._lookup_workspace) so a test patch there
# is seen by both that path and get_workspace_rag's own lookup.
from server.db import _db_init  # noqa: E402
from server.db import (
    _db_set_doc_id as _db_set_doc_id,
)
from server.db import (
    _db_set_lightrag_key as _db_set_lightrag_key,
)
from server.db import (
    _db_update_status as _db_update_status,
)

# --- Document processing ---
# The ingestion pipeline lives in server.ingest; _process_file is re-exported because the
# background worker looks it up as server._process_file (and the test suite patches it there).
from server.ingest import (  # noqa: E402
    _process_file as _process_file,
)

# --- Background worker ---
# The ingestion worker + job reload live in server.worker (a top-level orchestrator that calls the
# re-exported pipeline functions via server.*). Lifespan starts _worker and calls _db_reload_jobs.
from server.worker import (  # noqa: E402
    _db_reload_jobs,
    _worker,
)
from server.worker import (
    _process_job as _process_job,
)
from server.workspaces import _get_ws_lock, get_workspace_rag  # noqa: E402

# --- Startup / shutdown ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_pool
    import asyncpg

    Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)

    _db_pool = await asyncpg.create_pool(
        host=POSTGRES_HOST,
        port=int(POSTGRES_PORT),
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        min_size=2,
        max_size=10,
    )
    await _db_init(_db_pool)
    await _db_reload_jobs(_db_pool)

    # Pre-warm the primary workspace so the existing corpus is ready immediately.
    # Resolve the ACTUAL primary from the DB (the single source of truth) rather than the
    # hardcoded PRIMARY_WORKSPACE_ID — the seeded default can be superseded (e.g. reseeded to a
    # different primary), and loading a stale hardcoded id would 404 and crash startup.
    # Other workspaces are built lazily on first use via get_workspace_rag().
    _primary_row = await _db_pool.fetchrow(
        "SELECT id FROM rag_workspaces WHERE is_primary = TRUE AND deleted_at IS NULL LIMIT 1"
    )
    await get_workspace_rag(_primary_row["id"] if _primary_row else PRIMARY_WORKSPACE_ID)

    worker_task = asyncio.create_task(_worker())
    yield
    worker_task.cancel()
    for instance in list(_rag_instances.values()):
        await instance.lightrag.finalize_storages()
    _rag_instances.clear()
    _ws_locks.clear()
    await _db_pool.close()
    _db_pool = None


app = FastAPI(
    title="RAG-Anything API",
    version="1.0.0",
    description=(
        "Multimodal RAG service over a Postgres/AGE-backed LightRAG knowledge store. "
        "Documents are ingested per **workspace** (an isolated knowledge graph + vector index); "
        "each workspace is addressed by its public slug id under `/workspace/{workspace_id}/...`. "
        "Upload files, query the corpus (with citations), or render the knowledge graph as an "
        "interactive HTML page.\n\n"
        "**Scope:** a single-instance, low-usage service — run exactly one worker/replica "
        "(ingest job state is in-process); it is not designed for high load or horizontal "
        "scaling. **Auth:** disabled by default (ports are loopback-only). Set `API_TOKENS` "
        "to require a token on every endpoint except `/health` — send it as "
        "`Authorization: Bearer <token>`, or in a browser use any username with the token as "
        "the password. Serve over TLS whenever the service is exposed beyond loopback."
    ),
    lifespan=lifespan,
)


# --- Auth (opt-in via API_TOKENS) ---


def _auth_enabled() -> bool:
    """True when at least one API token is configured (read live so tests can toggle it)."""
    return bool(API_TOKENS)


def _token_valid(token: str) -> bool:
    """Constant-time membership check against the configured tokens."""
    return any(secrets.compare_digest(token, t) for t in API_TOKENS)


def _extract_credential(header: str) -> str | None:
    """Pull the presented secret out of an Authorization header, supporting both transports:
    `Bearer <token>` (machines) and `Basic <base64(user:pass)>` (browsers — the token is the
    password, username is ignored). Returns None if the header is absent/malformed."""
    if not header:
        return None
    scheme, _, rest = header.partition(" ")
    scheme = scheme.lower()
    if scheme == "bearer":
        return rest.strip() or None
    if scheme == "basic":
        try:
            decoded = base64.b64decode(rest.strip()).decode("utf-8", "replace")
        except (ValueError, TypeError):
            return None
        _, sep, password = decoded.partition(":")
        return password if sep else None
    return None


@app.middleware("http")
async def _require_auth(request: Request, call_next):
    """Gate every request behind API_TOKENS when auth is enabled. Applied as middleware (not a
    route dependency) so it also covers the auto-generated /docs, /redoc and /openapi.json.
    /health stays open for liveness probes. A 401 carries `WWW-Authenticate: Basic` so browsers
    show a native login prompt and then auto-attach the credentials to same-origin requests."""
    if _auth_enabled() and request.url.path != "/health":
        cred = _extract_credential(request.headers.get("authorization", ""))
        if not (cred and _token_valid(cred)):
            return JSONResponse(
                {"detail": "Unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="PolyGraphRAG"'},
            )
    return await call_next(request)


# --- Shared endpoint deps/helpers (server.deps) ---
from server.deps import _is_valid_slug  # noqa: E402

# --- API ---
# Request models live in server.schemas. Imported by ABSOLUTE name (not `.schemas`) so the
# config-probe test — which re-execs this file under a throwaway module name via
# spec_from_file_location — can still resolve it (a relative import has no package there).
from server.schemas import WorkspaceCreate  # noqa: E402


@app.get(
    "/health",
    summary="Liveness probe",
    description='Returns `{"status": "ok"}` when the service is up. Does not check DB connectivity.',
)
async def health():
    return {"status": "ok"}


# --- Workspace registry API ---


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
        "SELECT id, name, description, lightrag_workspace, is_primary, deleted_at "
        "FROM rag_workspaces WHERE id = $1",
        workspace_id,
    )


# Graph storage namespace; a non-default workspace `w` gets AGE graph `{w}_chunk_entity_relation`
# (see LightRAG PGGraphStorage._get_workspace_graph_name). The default/primary workspace uses the
# bare `chunk_entity_relation` graph and is delete-protected, so purge never touches it.
_AGE_NAMESPACE = "chunk_entity_relation"


async def _purge_workspace_data(pool, physical_workspace: str) -> None:
    """Irreversibly delete one physical workspace's LightRAG data, file metadata, and files.
    Only ever called for non-primary workspaces (primary is delete-protected), so the shared
    `chunk_entity_relation` graph and `default` rows are never affected."""
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
    import shutil

    shutil.rmtree(Path(WORKING_DIR) / physical_workspace, ignore_errors=True)


async def _evict_workspace_instance(workspace_id: str) -> None:
    """Drop a workspace's cached instance and finalize its storages (no-op if not built)."""
    lock = await _get_ws_lock(workspace_id)
    async with lock:
        instance = _rag_instances.pop(workspace_id, None)
    if instance is not None:
        await instance.lightrag.finalize_storages()


@app.get(
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
    if _db_pool is None:
        raise HTTPException(503, "DB not initialised")
    cond = "deleted_at IS NOT NULL" if deleted else "deleted_at IS NULL"
    rows = await _db_pool.fetch(
        f"SELECT id, name, description, created_at FROM rag_workspaces WHERE {cond} ORDER BY created_at"
    )
    return {"workspaces": [_workspace_public(r) for r in rows]}


@app.post(
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
    if _db_pool is None:
        raise HTTPException(503, "DB not initialised")
    if not _is_valid_slug(body.id):
        raise HTTPException(422, "Invalid workspace id: must match ^[a-z][a-z0-9_]{0,47}$")
    if await _db_get_workspace_any(_db_pool, body.id) is not None:
        raise HTTPException(409, f"Workspace {body.id!r} already exists")
    await _db_pool.execute(
        """INSERT INTO rag_workspaces (id, name, description, lightrag_workspace, is_primary)
               VALUES ($1, $2, $3, $4, FALSE)""",
        body.id,
        body.name,
        body.description,
        body.id,  # lightrag_workspace := id
    )
    return {"id": body.id, "name": body.name, "description": body.description}


@app.delete(
    "/workspace/{workspace_id}",
    summary="Delete a workspace (soft-delete or purge)",
    description=(
        "By default this is a reversible **soft delete** (hidden from listings, restorable via "
        "`/workspace/{id}/restore`). Pass `purge=true` to **irreversibly** delete the workspace's "
        "graph, vector rows, file metadata, and on-disk files. The primary workspace cannot be deleted."
    ),
    responses={
        404: {"description": "Workspace not found"},
        409: {"description": "Cannot delete the primary workspace"},
        503: {"description": "Database not initialised yet"},
    },
)
async def delete_workspace(
    workspace_id: str = PathParam(description="Public workspace id (slug) to delete."),
    purge: bool = Query(
        False, description="If true, irreversibly purge all data instead of soft-deleting."
    ),
):
    if _db_pool is None:
        raise HTTPException(503, "DB not initialised")
    row = await _db_get_workspace_any(_db_pool, workspace_id)
    if row is None:
        raise HTTPException(404, f"Workspace {workspace_id!r} not found")
    if row["is_primary"]:
        raise HTTPException(409, "Cannot delete the primary workspace")
    if purge:
        await _evict_workspace_instance(workspace_id)
        await _purge_workspace_data(_db_pool, row["lightrag_workspace"])
        await _db_pool.execute("DELETE FROM rag_workspaces WHERE id = $1", workspace_id)
        return {"status": "purged", "id": workspace_id}
    await _db_pool.execute(
        "UPDATE rag_workspaces SET deleted_at = NOW() WHERE id = $1", workspace_id
    )
    await _evict_workspace_instance(workspace_id)
    return {"status": "soft-deleted", "id": workspace_id}


@app.post(
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
    if _db_pool is None:
        raise HTTPException(503, "DB not initialised")
    row = await _db_get_workspace_any(_db_pool, workspace_id)
    if row is None or row["deleted_at"] is None:
        raise HTTPException(404, f"No soft-deleted workspace {workspace_id!r} to restore")
    await _db_pool.execute(
        "UPDATE rag_workspaces SET deleted_at = NULL WHERE id = $1", workspace_id
    )
    return {"status": "restored", "id": workspace_id}


async def _count_vdb(phys: str, kind: str) -> int | None:
    """Distinct entity/relationship count from the dedup'd vector tables, discovering the
    embedding-suffixed table name dynamically (robust to the configured embedding model)."""
    try:
        tbl = await _db_pool.fetchval(
            "SELECT table_name FROM information_schema.tables WHERE table_name LIKE $1 LIMIT 1",
            f"lightrag_vdb_{kind}%",
        )
        if not tbl:
            return None
        return await _db_pool.fetchval(f"SELECT count(*) FROM {tbl} WHERE workspace=$1", phys)
    except Exception:
        return None


@app.get(
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
    if _db_pool is None:
        raise HTTPException(503, "DB not initialised")
    if not _is_valid_slug(workspace_id):
        raise HTTPException(404, f"Workspace {workspace_id!r} not found")
    row = await _db_get_workspace_any(_db_pool, workspace_id)
    if row is None:
        raise HTTPException(404, f"Workspace {workspace_id!r} not found")
    phys = row["lightrag_workspace"]
    doc_rows = await _db_pool.fetch(
        "SELECT status, count(*) AS n FROM lightrag_doc_status WHERE workspace=$1 GROUP BY status",
        phys,
    )
    docs_by_status = {r["status"]: r["n"] for r in doc_rows}
    chunks = await _db_pool.fetchval(
        "SELECT COALESCE(SUM(chunks_count),0) FROM lightrag_doc_status WHERE workspace=$1", phys
    )
    job_rows = await _db_pool.fetch(
        "SELECT status, count(*) AS n FROM rag_file_metadata WHERE workspace=$1 GROUP BY status",
        phys,
    )
    last_uploaded = await _db_pool.fetchval(
        "SELECT MAX(uploaded_at) FROM rag_file_metadata WHERE workspace=$1", phys
    )
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "is_primary": row["is_primary"],
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


# --- Routers ---
# Endpoints live in server.routers.*; included here. They call the workspace registry via
# server.get_workspace_rag (attribute access at call time) so test patches are honoured.
from server.routers import documents as _documents_router  # noqa: E402
from server.routers import query as _query_router  # noqa: E402

app.include_router(_query_router.router)
app.include_router(_documents_router.router)
