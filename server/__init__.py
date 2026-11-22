import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from raganything import RAGAnything
from starlette.routing import Route

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
    WORKING_DIR,
)

_job_queue: asyncio.Queue = asyncio.Queue()
_jobs: dict[str, dict] = {}
_batches: dict[str, list] = {}
_db_pool = None  # asyncpg.Pool, set in lifespan
_graph_pool = None  # set after the editable graph schema is initialized

# Per-workspace RAGAnything instance registry. Keyed by PUBLIC workspace id.
_rag_instances: dict[str, RAGAnything] = {}
_ws_locks: dict[str, asyncio.Lock] = (
    {}
)  # one lock per workspace: guards creation AND serialises inserts
_registry_lock = asyncio.Lock()  # guards the dicts above


# --- Re-exported package surface ---
# Each concern lives in its own submodule; the names below are re-exported on `server` because the
# background worker (server/worker.py) looks them up as server.* at call time — so the live object,
# including any test-suite monkeypatch, is always the one used — and lifespan calls some directly.
# _db_init / _db_reload_jobs / _worker are used by lifespan; _process_file, _process_job and the
# _db_set_*/ _db_update_status writers are the worker's server.* call-time targets (and _process_file
# is the suite's patch point); get_workspace_rag is used by lifespan and every router.
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
from server.embeddings import embedding_worker  # noqa: E402
from server.ingest import (  # noqa: E402
    _process_file as _process_file,
)
from server.mcp import MCPTransport, create_session_manager  # noqa: E402
from server.worker import (  # noqa: E402
    _db_reload_jobs,
    _worker,
)
from server.worker import (
    _process_job as _process_job,
)
from server.workspaces import get_workspace_rag as get_workspace_rag  # noqa: E402

# --- Startup / shutdown ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_pool
    import asyncpg

    app.state.mcp_manager = create_session_manager()
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

    worker_task = asyncio.create_task(_worker())
    embedding_task = asyncio.create_task(embedding_worker(_db_pool))
    try:
        async with app.state.mcp_manager.run():
            yield
    finally:
        worker_task.cancel()
        embedding_task.cancel()
        await asyncio.gather(worker_task, embedding_task, return_exceptions=True)
        for instance in list(_rag_instances.values()):
            await instance.lightrag.finalize_storages()
        _rag_instances.clear()
        _ws_locks.clear()
        await _db_pool.close()
        _db_pool = None


app = FastAPI(
    title="PolyGraphRAG",
    # Stateless MCP tools return one JSON response per POST. GET returns 405.
    routes=[Route("/mcp", MCPTransport(), methods=["POST"], include_in_schema=False)],
    version="0.1.0",  # keep in sync with pyproject.toml
    description=(
        "Multimodal RAG service over a Postgres/AGE-backed LightRAG knowledge store. "
        "Documents are ingested per **workspace** (an isolated knowledge graph + vector index); "
        "each workspace is addressed by its public slug id under `/workspace/{workspace_id}/...`. "
        "Upload files, edit the graph directly, query the corpus with citations, or render the "
        "graph as an interactive HTML page.\n\n"
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
from server.compression import NegotiatedGZip  # noqa: E402

app.add_middleware(NegotiatedGZip)
app.middleware("http")(_require_auth)


def _custom_openapi() -> dict:
    """Default FastAPI schema plus the auth advertisement: securitySchemes for the Bearer/Basic
    transports the middleware accepts, marked as a global requirement (with /health documented
    as open). Documentation-only — enforcement stays in the middleware, and behavior is
    unchanged while API_TOKENS is empty. Also gives Swagger UI an Authorize button."""
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title, version=app.version, description=app.description, routes=app.routes
    )
    schema.setdefault("components", {})["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "description": "API token from API_TOKENS (only enforced when API_TOKENS is set).",
        },
        "basicAuth": {
            "type": "http",
            "scheme": "basic",
            "description": "Browser transport: any username, the API token as the password.",
        },
    }
    schema["security"] = [{"bearerAuth": []}, {"basicAuth": []}]
    if "/health" in schema["paths"] and "get" in schema["paths"]["/health"]:
        schema["paths"]["/health"]["get"]["security"] = []  # liveness probe is never gated
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi  # type: ignore[method-assign]


@app.get(
    "/",
    summary="Service discovery card",
    description=(
        "Small service card for humans and agents landing on the root URL: the service name, "
        "version, and where to find the interactive docs and machine-readable spec. Gated by "
        "the same auth as the rest of the API."
    ),
)
async def root():
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "mcp": "/mcp",
        "health": "/health",
    }


from server.schemas import HealthResponse  # noqa: E402


@app.get(
    "/health",
    summary="Liveness probe",
    description='Returns `{"status": "ok"}` when the service is up. Does not check DB connectivity.',
    response_model=HealthResponse,
)
async def health():
    return {"status": "ok"}


# --- Routers ---
# Endpoints live in server.routers.*; included here. They call the workspace registry via
# server.get_workspace_rag (attribute access at call time) so test patches are honoured.
from server.routers import documents as _documents_router  # noqa: E402
from server.routers import graph as _graph_router  # noqa: E402
from server.routers import query as _query_router  # noqa: E402
from server.routers import workspaces as _workspaces_router  # noqa: E402

app.include_router(_query_router.router)
app.include_router(_documents_router.router)
app.include_router(_graph_router.router)
app.include_router(_workspaces_router.router)
