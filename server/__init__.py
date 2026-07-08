import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from raganything import RAGAnything

# Re-exported (not used here): server.auth reads the token set as server.API_TOKENS at call time,
# and the test suite toggles auth by setting this attribute.
from server.config import API_TOKENS as API_TOKENS

# Import config BEFORE the heavy libraries below: importing it runs logging.basicConfig, so
# LightRAG / RAG-Anything don't emit unconfigured logs at import time. Every config name is
# re-exported through this module so `server.<CONST>` — and the test suite's monkeypatches —
# keep working; functions that read config live (e.g. _active_llm_cfg) are defined in
# server.config and patched there.
from server.config import (
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
from server.workspaces import get_workspace_rag  # noqa: E402

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
# Credential parsing + the gate live in server.auth; registered here as HTTP middleware so it also
# covers the auto-generated docs. It reads server.API_TOKENS live, so tests toggle auth by setting
# that attribute.
from server.auth import _require_auth  # noqa: E402

app.middleware("http")(_require_auth)


@app.get(
    "/health",
    summary="Liveness probe",
    description='Returns `{"status": "ok"}` when the service is up. Does not check DB connectivity.',
)
async def health():
    return {"status": "ok"}


# --- Routers ---
# Endpoints live in server.routers.*; included here. They call the workspace registry via
# server.get_workspace_rag (attribute access at call time) so test patches are honoured.
from server.routers import documents as _documents_router  # noqa: E402
from server.routers import query as _query_router  # noqa: E402
from server.routers import workspaces as _workspaces_router  # noqa: E402

app.include_router(_query_router.router)
app.include_router(_documents_router.router)
app.include_router(_workspaces_router.router)
