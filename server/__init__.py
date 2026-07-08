import asyncio
import base64
import hashlib
import json
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi import Path as PathParam  # aliased: `Path` is pathlib.Path throughout this module
from fastapi.responses import HTMLResponse, JSONResponse
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
    QUERY_LLM_MODEL,
    RAG_FILTER_TOPK_BOOST,
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
from server.db import (  # noqa: E402
    _db_init,
    _db_insert_job,
)
from server.db import (
    _db_set_doc_id as _db_set_doc_id,
)
from server.db import (
    _db_set_lightrag_key as _db_set_lightrag_key,
)
from server.db import (
    _db_update_status as _db_update_status,
)

# --- References (real-path resolution) ---
# Defined in server.references; re-exported for the query/graph endpoints and upload_batch.
# --- Graph visualisation ---
# Rendering lives in server.graph; the graph.html endpoint calls _build_graph_html.
from server.graph import _build_graph_html  # noqa: E402
from server.ingest import (  # noqa: E402
    _build_metadata,
    _join_path,
)
from server.ingest import (
    _process_file as _process_file,
)
from server.references import (  # noqa: E402
    _build_references,
    _clean_needles,
    _path_matches_any,
    _resolve_block_file_paths,
    _resolve_graph_paths,
    _rewrite_answer_refs,
    _safe_ref_name,
)
from server.workspaces import _get_ws_lock, get_workspace_rag  # noqa: E402


def _query_param(QueryParam, *, mode, include_references=None, top_k=None, chunk_top_k=None):
    """Build a QueryParam, omitting optional knobs so LightRAG's own defaults apply."""
    kwargs = {"mode": mode}
    if include_references is not None:
        kwargs["include_references"] = include_references
    if top_k is not None:
        kwargs["top_k"] = top_k
    if chunk_top_k is not None:
        kwargs["chunk_top_k"] = chunk_top_k
    return QueryParam(**kwargs)


# --- Background worker ---
# The ingestion worker + job reload + on-disk job path live in server.worker (a top-level
# orchestrator that calls the re-exported pipeline functions via server.*). Lifespan starts
# _worker and calls _db_reload_jobs; upload_batch uses _job_path.
from server.worker import (  # noqa: E402
    _db_reload_jobs,
    _job_path,
    _worker,
)
from server.worker import (
    _process_job as _process_job,
)

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
from server.deps import (  # noqa: E402
    _batch_response,
    _internal_error,
    _is_valid_slug,
    require_workspace,
)

# --- API ---
# Request models live in server.schemas. Imported by ABSOLUTE name (not `.schemas`) so the
# config-probe test — which re-execs this file under a throwaway module name via
# spec_from_file_location — can still resolve it (a relative import has no package there).
from server.schemas import (  # noqa: E402
    FileDeleteRequest,
    QueryDataRequest,
    QueryRequest,
    WorkspaceCreate,
)


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


# --- Workspace-scoped data API (everything below lives under /workspace/{workspace_id}) ---


@app.post(
    "/workspace/{workspace_id}/upload/batch",
    summary="Upload one or more files for ingestion",
    description=(
        "Upload a batch of files into the workspace; ingestion runs asynchronously in the background. "
        "Send `files` as `multipart/form-data`. Optionally send `metadata` as a JSON array of objects "
        "**index-aligned with `files`** — each may carry `description`, `source_path`, `path_root`, and "
        "`last_modified_time` (only `description` is embedded into the searchable text). "
        "**Supply both `source_path` and `path_root`** to record the file's real path "
        "(`path_root/source_path`): that path is what `/query` and `/query/data` return in "
        "`references[].file_path`, so citations resolve back to your own source tree. Without them, "
        "references carry the original filename. "
        "Supported types include PDF, images, Office docs, audio, and text/markdown/CSV. "
        "Returns a `batch_id` plus a per-file `job_id`; poll `/workspace/{id}/status/{job_id}` or "
        "`/workspace/{id}/batch/{batch_id}` for progress."
    ),
    responses={404: {"description": "Workspace not found or soft-deleted"}},
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "One or more files to ingest",
                            },
                            "metadata": {
                                "type": "array",
                                "description": "Per-file metadata objects, index-aligned with files. All fields optional.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "description": {
                                            "type": "string",
                                            "description": "Human-readable description of the file",
                                        },
                                        "source_path": {
                                            "type": "string",
                                            "description": "Original relative path or URL of the file",
                                        },
                                        "path_root": {
                                            "type": "string",
                                            "description": "Optional absolute prefix joined to source_path to form the stored file identity",
                                        },
                                        "last_modified_time": {
                                            "type": "string",
                                            "format": "date-time",
                                            "description": "ISO 8601 last-modified timestamp",
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "encoding": {"metadata": {"contentType": "application/json"}},
                }
            },
        }
    },
)
async def upload_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    ws: dict = Depends(require_workspace),
):
    pub = ws["id"]  # public id → routes the worker to the right instance
    phys = ws["lightrag_workspace"]  # physical workspace → storage namespace + metadata tag
    form = await request.form()
    metadata_field = form.get("metadata", "[]")
    if hasattr(metadata_field, "read"):
        # Swagger UI sends metadata with Content-Type: application/json,
        # which makes Starlette parse it as an UploadFile (filename="blob").
        metadata_raw = (await metadata_field.read()).decode()
    else:
        metadata_raw = str(metadata_field) if metadata_field else "[]"
    try:
        meta_list: list = json.loads(metadata_raw)
    except (json.JSONDecodeError, ValueError):
        meta_list = []
    # Swagger UI encodes each array element as a JSON string instead of an object;
    # double-parse any string elements to normalise both representations.
    normalized: list[dict] = []
    for item in meta_list:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            try:
                parsed = json.loads(item)
                normalized.append(parsed if isinstance(parsed, dict) else {})
            except (json.JSONDecodeError, ValueError):
                normalized.append({})
        else:
            normalized.append({})
    meta_list = normalized
    batch_id = uuid.uuid4().hex[:8]
    entries = []
    for i, file in enumerate(files):
        job_id = uuid.uuid4().hex[:8]
        dest = _job_path(phys, job_id, file.filename)
        m = meta_list[i] if i < len(meta_list) else {}
        description = m.get("description", "") or ""
        source_path = m.get("source_path", "") or ""
        last_modified = m.get("last_modified_time", "") or ""
        path_root = m.get("path_root", "") or ""
        try:
            raw = await file.read()
            with dest.open("wb") as fh:
                fh.write(raw)
            content_hash = hashlib.sha256(raw).hexdigest()
            # Two distinct identities per file:
            #  - lightrag_input: what we hand LightRAG. `{job_id}_{basename}` is unique and
            #    slash-free, so LightRAG's basename canonicalization can't drop the job_id and
            #    keys never collide. (Read back as `lightrag_key` after ingest; JOIN-ONLY.)
            #  - display_path: the REAL, openable path shown in references/`/files` — the caller's
            #    path_root/source_path join when given, else source_path, else the plain filename.
            #    Never the on-disk `{job_id}_` token.
            lightrag_input = f"{job_id}_{_safe_ref_name(file.filename)}"
            display_path = (
                _join_path(path_root, source_path)
                if (path_root and source_path)
                else (source_path or _safe_ref_name(file.filename))
            )
            description_text = _build_metadata(description, source_path, last_modified)
            record: dict = {
                "job_id": job_id,
                "file": file.filename,
                "workspace": phys,
                "status": "pending",
                "attempts": 0,
                "error": None,
                "batch_id": batch_id,
                "content_hash": content_hash,
                "file_path": display_path,
            }
            _jobs[job_id] = record
            entries.append(record)
            if _db_pool:
                await _db_insert_job(
                    _db_pool,
                    record,
                    description,
                    source_path,
                    last_modified,
                    content_hash=content_hash,
                    file_path=display_path,
                    lightrag_key=lightrag_input,
                )
            await _job_queue.put((pub, job_id, dest, description_text, lightrag_input))
        except Exception as exc:
            entries.append(
                {
                    "file": file.filename,
                    "workspace": phys,
                    "status": "save_failed",
                    "error": str(exc),
                    "batch_id": batch_id,
                }
            )
    _batches[batch_id] = entries
    return _batch_response(batch_id, entries)


@app.get(
    "/workspace/{workspace_id}/batch/{batch_id}",
    summary="Get the status of an upload batch",
    description="Return the per-file job statuses and a summary count for a batch returned by upload.",
    responses={404: {"description": "Batch not found in this workspace"}},
)
async def get_batch(
    batch_id: str = PathParam(description="Batch id returned by the upload endpoint."),
    ws: dict = Depends(require_workspace),
):
    phys = ws["lightrag_workspace"]
    entries = [e for e in _batches.get(batch_id, []) if e.get("workspace") == phys]
    if not entries:
        raise HTTPException(404, f"Batch {batch_id!r} not found")
    return _batch_response(batch_id, entries)


@app.get(
    "/workspace/{workspace_id}/status/{job_id}",
    summary="Get the status of a single ingestion job",
    description=(
        "Return one file's ingestion job: status (pending/processing/retrying/done/failed), "
        "attempt count, and any error."
    ),
    responses={404: {"description": "Job not found in this workspace"}},
)
async def get_status(
    job_id: str = PathParam(description="Per-file job id returned by the upload endpoint."),
    ws: dict = Depends(require_workspace),
):
    phys = ws["lightrag_workspace"]
    job = _jobs.get(job_id)
    if job is not None and job.get("workspace") == phys:
        return job
    if _db_pool:
        row = await _db_pool.fetchrow(
            "SELECT * FROM rag_file_metadata WHERE job_id = $1 AND workspace = $2", job_id, phys
        )
        if row:
            return dict(row)
    raise HTTPException(404, f"Job {job_id!r} not found")


@app.get(
    "/workspace/{workspace_id}/jobs",
    summary="List recent ingestion jobs",
    description="Return the 100 most recent ingestion jobs for this workspace, newest first.",
    responses={404: {"description": "Workspace not found or soft-deleted"}},
)
async def list_jobs(ws: dict = Depends(require_workspace)):
    phys = ws["lightrag_workspace"]
    if _db_pool:
        rows = await _db_pool.fetch(
            "SELECT * FROM rag_file_metadata WHERE workspace = $1 ORDER BY uploaded_at DESC LIMIT 100",
            phys,
        )
        return {"jobs": [dict(r) for r in rows]}
    jobs = [j for j in _jobs.values() if j.get("workspace") == phys]
    return {"jobs": jobs[-100:][::-1]}


def _file_index_row(row) -> dict:
    ua = row["uploaded_at"]
    return {
        "job_id": row["job_id"],
        "file": row["file"],
        "file_path": row.get("file_path"),
        "source_path": row.get("source_path"),
        "doc_id": row.get("doc_id"),
        "content_hash": row.get("content_hash"),
        "status": row["status"],
        "last_modified_time": row.get("last_modified_time"),
        "uploaded_at": ua.isoformat() if hasattr(ua, "isoformat") else ua,
    }


@app.get(
    "/workspace/{workspace_id}/files",
    summary="List the workspace's ingested-file index",
    description=(
        "Return the durable per-file index for this workspace, sourced entirely from the database — "
        "it does **not** read the filesystem and remains complete even though raw uploaded files are "
        "deleted after a successful ingest. Each entry includes the `content_hash` (SHA-256 of the "
        "ingested bytes) and `doc_id` (LightRAG `doc-<md5>`), which the sync worker uses to detect "
        "changes and to delete precisely. This is the system of record for *what has been ingested*."
    ),
    responses={
        404: {"description": "Workspace not found or soft-deleted"},
        503: {"description": "Database not initialised yet"},
    },
)
async def list_files(ws: dict = Depends(require_workspace)):
    if _db_pool is None:
        raise HTTPException(503, "DB not initialised")
    phys = ws["lightrag_workspace"]
    rows = await _db_pool.fetch(
        "SELECT job_id, file, file_path, source_path, doc_id, content_hash, status, "
        "last_modified_time, uploaded_at FROM rag_file_metadata "
        "WHERE workspace = $1 ORDER BY uploaded_at DESC",
        phys,
    )
    return {"files": [_file_index_row(r) for r in rows]}


async def _resolve_doc_for_delete(
    phys: str, body: FileDeleteRequest
) -> tuple[str | None, dict | None]:
    """Resolve the target (doc_id, metadata_row) for a per-file delete. Order: explicit doc_id →
    metadata match on file_path/source_path → LightRAG doc_status by file_path. Returns (None, None)
    if nothing matches (deleting an absent file is a no-op success)."""
    if body.doc_id:
        row = (
            await _db_pool.fetchrow(
                "SELECT job_id, file, doc_id FROM rag_file_metadata WHERE workspace=$1 AND doc_id=$2 LIMIT 1",
                phys,
                body.doc_id,
            )
            if _db_pool
            else None
        )
        return body.doc_id, (dict(row) if row else None)
    if _db_pool is not None:
        for col, val in (("file_path", body.external_path), ("source_path", body.rel_path)):
            if not val:
                continue
            row = await _db_pool.fetchrow(
                f"SELECT job_id, file, doc_id FROM rag_file_metadata WHERE workspace=$1 AND {col}=$2 "
                "ORDER BY uploaded_at DESC LIMIT 1",
                phys,
                val,
            )
            if row and row["doc_id"]:
                return row["doc_id"], dict(row)
        # Fall back to LightRAG's own doc_status index, keyed by file_path == external_path.
        if body.external_path:
            ds = await _db_pool.fetchrow(
                "SELECT id FROM lightrag_doc_status WHERE workspace=$1 AND file_path=$2 "
                "ORDER BY updated_at DESC LIMIT 1",
                phys,
                body.external_path,
            )
            if ds:
                return ds["id"], None
    return None, None


@app.delete(
    "/workspace/{workspace_id}/file/delete",
    summary="Delete one file's document + entities from the graph",
    description=(
        "Remove a single ingested file from the knowledge graph: its document, chunks, and the "
        "entities/relationships sourced **only** by it. Identify the file by `doc_id` (most precise), "
        "`external_path` (matched against the stored LightRAG file path), or `rel_path` (matched against "
        "the stored source path). The LLM cache for the document is **always cleared** so no outdated "
        "extraction lingers. **Idempotent:** deleting a file that isn't present returns `noop`.\n\n"
        "Note on shared entities: an entity that appears in several files is one merged graph node; this "
        "deletes only entities sourced solely by this file — entities still referenced by other files "
        "correctly survive. The deletion completes before the response returns."
    ),
    responses={
        404: {"description": "Workspace not found or soft-deleted"},
        503: {"description": "Database not initialised yet"},
    },
)
async def delete_file(body: FileDeleteRequest, ws: dict = Depends(require_workspace)):
    if _db_pool is None:
        raise HTTPException(503, "DB not initialised")
    phys = ws["lightrag_workspace"]
    doc_id, meta = await _resolve_doc_for_delete(phys, body)
    if not doc_id:
        return {"status": "noop", "reason": "not_found", "doc_id": None}
    rag_instance = await get_workspace_rag(ws["id"])
    lock = await _get_ws_lock(ws["id"])
    async with lock:
        try:
            await rag_instance.lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        except Exception as exc:
            raise _internal_error(exc, f"delete of doc {doc_id}") from exc
    # Drop the index row(s) and any leftover raw file.
    await _db_pool.execute(
        "DELETE FROM rag_file_metadata WHERE workspace=$1 AND doc_id=$2", phys, doc_id
    )
    if meta and meta.get("job_id") and meta.get("file"):
        _job_path(phys, meta["job_id"], meta["file"]).unlink(missing_ok=True)
    return {"status": "deleted", "doc_id": doc_id}


@app.post(
    "/workspace/{workspace_id}/query",
    summary="Ask a question (LLM answer + citations)",
    description=(
        "Run a RAG query against the workspace and return a synthesised natural-language answer. "
        "When `include_references` is true, the response also lists the source documents used — "
        "each reference's `file_path` is the **real, openable document path** you supplied at upload "
        "(resolved server-side from Postgres), never LightRAG's internal name. "
        "For raw retrieved entities/relationships/chunks instead of a prose answer, use `/query/data`."
    ),
    responses={404: {"description": "Workspace not found or soft-deleted"}},
)
async def query(req: QueryRequest, ws: dict = Depends(require_workspace)):
    rag_instance = await get_workspace_rag(ws["id"])
    try:
        from lightrag import QueryParam

        raw = await rag_instance.lightrag.aquery_llm(
            req.query,
            param=_query_param(
                QueryParam,
                mode=req.mode,
                include_references=req.include_references,
                top_k=req.top_k,
            ),
        )
    except Exception as exc:
        raise _internal_error(exc, "query") from exc

    result = (raw.get("llm_response") or {}).get("content", "")
    raw_refs = (raw.get("data") or {}).get("references")
    # Build the key→row map even to only rewrite the prose; emit the structured refs when asked.
    references, meta_map = await _build_references(
        raw_refs, ws["lightrag_workspace"], answered_model=QUERY_LLM_MODEL
    )
    result = _rewrite_answer_refs(result, raw_refs, meta_map)
    return {"result": result, "references": references if req.include_references else []}


@app.post(
    "/workspace/{workspace_id}/query/data",
    summary="Retrieve raw graph/vector data for a query (no LLM answer)",
    description=(
        "Run retrieval for a query and return the raw matched **entities, relationships, and text "
        "chunks** (plus references when `include_references` is true) without generating a prose "
        "answer. Use this when you want structured context to feed into your own prompt or to "
        "inspect what the corpus contains.\n\n"
        "**Scope to a folder/file with `file_path_contains`**: a list of case-insensitive substrings "
        "matched (OR) against each result's `file_path` — an item is kept if its path contains ANY of "
        "them. Empty/omitted = no filtering. The retrieval budget is auto-boosted when a filter is set, "
        "but matching happens *after* retrieval, so a very narrow folder may return fewer items than "
        "exist in it. Example body: "
        '`{"query":"...","file_path_contains":["/opt/data/workspace/career/"]}`.'
    ),
    responses={404: {"description": "Workspace not found or soft-deleted"}},
)
async def query_data(req: QueryDataRequest, ws: dict = Depends(require_workspace)):
    rag_instance = await get_workspace_rag(ws["id"])
    needles = _clean_needles(req.file_path_contains)  # blank/empty => no filter (all data)
    # Post-filter runs after retrieval; widen the candidate set so a narrow folder still has hits.
    top_k = req.top_k * RAG_FILTER_TOPK_BOOST if needles else req.top_k
    chunk_top_k = None if not needles else req.top_k * RAG_FILTER_TOPK_BOOST
    try:
        from lightrag import QueryParam

        raw = await rag_instance.lightrag.aquery_data(
            req.query,
            param=_query_param(QueryParam, mode=req.mode, top_k=top_k, chunk_top_k=chunk_top_k),
        )
    except Exception as exc:
        raise _internal_error(exc, "query/data") from exc

    data = raw.get("data") or {}
    # Resolve internal LightRAG keys in entity/relationship/chunk file_path -> real Postgres paths
    # BEFORE filtering, so file_path_contains matches the real path consistently with references.
    await _resolve_block_file_paths(data, ws["lightrag_workspace"])
    if needles:
        data["entities"] = [
            e
            for e in (data.get("entities") or [])
            if _path_matches_any(e.get("file_path"), needles)
        ]
        data["relationships"] = [
            r
            for r in (data.get("relationships") or [])
            if _path_matches_any(r.get("file_path"), needles)
        ]
        data["chunks"] = [
            c for c in (data.get("chunks") or []) if _path_matches_any(c.get("file_path"), needles)
        ]
    references, _ = (
        await _build_references(data.get("references"), ws["lightrag_workspace"])
        if req.include_references
        else ([], {})
    )
    if needles:
        references = [r for r in references if _path_matches_any(r.get("file_path"), needles)]
    return {
        "status": raw.get("status"),
        "message": raw.get("message"),
        "data": {
            "entities": data.get("entities") or [],
            "relationships": data.get("relationships") or [],
            "chunks": data.get("chunks") or [],
            "references": references,
        },
        "metadata": raw.get("metadata") or {},
    }


@app.get(
    "/workspace/{workspace_id}/graph.html",
    response_class=HTMLResponse,
    summary="Render this workspace's knowledge graph as an interactive HTML page",
    description=(
        "Returns a **self-contained, offline-capable HTML page** (vis-network JS inlined) "
        "showing the workspace's LightRAG knowledge graph as an interactive force-directed "
        "diagram. Nodes are entities (colored by entity type, sized by their connection "
        "degree); edges are relationships. Hover a node or edge to see its full properties.\n\n"
        "Open the URL directly in a browser, or save the response body to a `.html` file. "
        "This returns rendered HTML, **not** JSON — for machine-readable graph data use "
        "`POST /workspace/{workspace_id}/query/data` instead.\n\n"
        "Tip: lower `max_nodes` / `max_depth` for a faster, less cluttered view of large graphs; "
        "set `physics=false` to freeze the layout once it settles.\n\n"
        "**Scope to a folder/file with `file_path_contains`**: repeat the query param to pass several "
        "case-insensitive substrings; a node is kept if its `file_path` contains ANY of them (OR). "
        "Empty = the whole graph. Filtering runs *after* graph selection (`max_nodes` is auto-boosted "
        "when set), so a very narrow folder may render sparsely. Example: "
        "`?file_path_contains=/opt/data/workspace/career/&file_path_contains=/opt/data/workspace/projects/`."
    ),
    responses={
        200: {"content": {"text/html": {}}, "description": "Interactive graph HTML page"},
        404: {"description": "Workspace not found or soft-deleted"},
    },
)
async def graph_html(
    workspace_id: str = PathParam(description="Public workspace id (slug) whose graph to render."),
    node_label: str = Query(
        "*",
        description="Entity name to center the subgraph on. Use '*' (default) for the entire graph.",
    ),
    max_depth: int = Query(
        3,
        ge=1,
        description="Maximum number of relationship hops to expand out from the starting node(s). Default 3.",
    ),
    max_nodes: int = Query(
        1000,
        ge=1,
        description="Hard cap on nodes returned; closest / highest-degree nodes win when truncated. Default 1000.",
    ),
    physics: bool = Query(
        True,
        description="Animate a force-directed layout (true, default). Set false for a static layout on large graphs.",
    ),
    file_path_contains: list[str] = Query(
        default_factory=list,
        description=(
            "Optional folder/file scope filter: repeatable case-insensitive substrings. Omit or leave "
            "empty to render the WHOLE graph (no filtering, the default). When provided, keep only nodes "
            "whose file_path contains ANY of them (OR; blank strings ignored). Applied after graph "
            "selection, so a very narrow folder may render sparsely."
        ),
    ),
    ws: dict = Depends(require_workspace),
):
    rag_instance = await get_workspace_rag(ws["id"])
    needles = _clean_needles(file_path_contains)  # blank/empty => no filter (whole graph)
    # Post-filter runs after graph selection; widen the fetch so a narrow folder still has nodes.
    fetch_nodes = max_nodes * RAG_FILTER_TOPK_BOOST if needles else max_nodes
    try:
        kg = await rag_instance.lightrag.get_knowledge_graph(
            node_label=node_label,
            max_depth=max_depth,
            max_nodes=fetch_nodes,
        )
    except Exception as exc:
        raise _internal_error(exc, "graph.html") from exc
    # Resolve node + edge file_path (internal keys) -> real Postgres paths before filtering/
    # rendering, so tooltips never show LightRAG's internal name and the folder filter matches the
    # real path. Nodes and edges share one fetch.
    await _resolve_graph_paths(list(kg.nodes) + list(kg.edges), ws["lightrag_workspace"])
    if needles:
        kg.nodes = [
            n for n in kg.nodes if _path_matches_any((n.properties or {}).get("file_path"), needles)
        ]
    return HTMLResponse(_build_graph_html(kg, physics))
