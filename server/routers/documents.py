"""Workspace-scoped document data API: upload/batch, job + batch status, file index, and delete.

All endpoints live under /workspace/{workspace_id}/…. Shared package state (the job/batch maps,
the ingest queue, and the DB pool) and the workspace-instance registry are read from the `server`
package at call time (server._db_pool, server.get_workspace_rag, …) so the test suite's patches on
those names are honoured; stateless helpers import directly from their owning modules.
"""

import hashlib
import json
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi import Path as PathParam

import server
from server.db import _db_insert_job
from server.deps import _batch_response, _internal_error, require_workspace
from server.ingest import _build_metadata, _join_path
from server.references import _safe_ref_name
from server.schemas import FileDeleteRequest
from server.worker import _job_path
from server.workspaces import _get_ws_lock

router = APIRouter()


@router.post(
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
            server._jobs[job_id] = record
            entries.append(record)
            if server._db_pool:
                await _db_insert_job(
                    server._db_pool,
                    record,
                    description,
                    source_path,
                    last_modified,
                    content_hash=content_hash,
                    file_path=display_path,
                    lightrag_key=lightrag_input,
                )
            await server._job_queue.put((pub, job_id, dest, description_text, lightrag_input))
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
    server._batches[batch_id] = entries
    return _batch_response(batch_id, entries)


@router.get(
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
    entries = [e for e in server._batches.get(batch_id, []) if e.get("workspace") == phys]
    if not entries:
        raise HTTPException(404, f"Batch {batch_id!r} not found")
    return _batch_response(batch_id, entries)


@router.get(
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
    job = server._jobs.get(job_id)
    if job is not None and job.get("workspace") == phys:
        return job
    if server._db_pool:
        row = await server._db_pool.fetchrow(
            "SELECT * FROM rag_file_metadata WHERE job_id = $1 AND workspace = $2", job_id, phys
        )
        if row:
            return _job_public(row)
    raise HTTPException(404, f"Job {job_id!r} not found")


@router.get(
    "/workspace/{workspace_id}/jobs",
    summary="List recent ingestion jobs",
    description="Return the 100 most recent ingestion jobs for this workspace, newest first.",
    responses={404: {"description": "Workspace not found or soft-deleted"}},
)
async def list_jobs(ws: dict = Depends(require_workspace)):
    phys = ws["lightrag_workspace"]
    if server._db_pool:
        rows = await server._db_pool.fetch(
            "SELECT * FROM rag_file_metadata WHERE workspace = $1 ORDER BY uploaded_at DESC LIMIT 100",
            phys,
        )
        return {"jobs": [_job_public(r) for r in rows]}
    jobs = [j for j in server._jobs.values() if j.get("workspace") == phys]
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


def _job_public(row) -> dict:
    """Public projection of a rag_file_metadata row for the job endpoints: the file-index
    fields plus job bookkeeping. Never emits lightrag_key (JOIN-ONLY, see db.py) or the
    physical workspace name."""
    return {
        **_file_index_row(row),
        "batch_id": row.get("batch_id"),
        "attempts": row.get("attempts"),
        "error": row.get("error"),
        "description": row.get("description"),
    }


@router.get(
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
    if server._db_pool is None:
        raise HTTPException(503, "DB not initialised")
    phys = ws["lightrag_workspace"]
    rows = await server._db_pool.fetch(
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
            await server._db_pool.fetchrow(
                "SELECT job_id, file, doc_id FROM rag_file_metadata WHERE workspace=$1 AND doc_id=$2 LIMIT 1",
                phys,
                body.doc_id,
            )
            if server._db_pool
            else None
        )
        return body.doc_id, (dict(row) if row else None)
    if server._db_pool is not None:
        for col, val in (("file_path", body.external_path), ("source_path", body.rel_path)):
            if not val:
                continue
            row = await server._db_pool.fetchrow(
                f"SELECT job_id, file, doc_id FROM rag_file_metadata WHERE workspace=$1 AND {col}=$2 "
                "ORDER BY uploaded_at DESC LIMIT 1",
                phys,
                val,
            )
            if row and row["doc_id"]:
                return row["doc_id"], dict(row)
        # Fall back to LightRAG's own doc_status index, keyed by file_path == external_path.
        if body.external_path:
            ds = await server._db_pool.fetchrow(
                "SELECT id FROM lightrag_doc_status WHERE workspace=$1 AND file_path=$2 "
                "ORDER BY updated_at DESC LIMIT 1",
                phys,
                body.external_path,
            )
            if ds:
                return ds["id"], None
    return None, None


@router.delete(
    "/workspace/{workspace_id}/file/delete",
    summary="Delete one file's document + entities from the graph",
    description=(
        "Remove a single ingested file from the knowledge graph: its document, chunks, and the "
        "entities/relationships sourced **only** by it. Identify the file by `doc_id` (most precise), "
        "`external_path` (matched against the stored LightRAG file path), or `rel_path` (matched against "
        "the stored source path). The LLM cache for the document is **always cleared** so no outdated "
        "extraction lingers. **Idempotent:** deleting a file that isn't present returns `noop`; a body "
        "with none of the three identifiers is rejected with 422.\n\n"
        "Note on shared entities: an entity that appears in several files is one merged graph node; this "
        "deletes only entities sourced solely by this file — entities still referenced by other files "
        "correctly survive. The deletion completes before the response returns."
    ),
    responses={
        404: {"description": "Workspace not found or soft-deleted"},
        422: {"description": "No identifier given (doc_id / external_path / rel_path)"},
        503: {"description": "Database not initialised yet"},
    },
)
async def delete_file(body: FileDeleteRequest, ws: dict = Depends(require_workspace)):
    if server._db_pool is None:
        raise HTTPException(503, "DB not initialised")
    phys = ws["lightrag_workspace"]
    doc_id, meta = await _resolve_doc_for_delete(phys, body)
    if not doc_id:
        return {"status": "noop", "reason": "not_found", "doc_id": None}
    rag_instance = await server.get_workspace_rag(ws["id"])
    lock = await _get_ws_lock(ws["id"])
    async with lock:
        try:
            await rag_instance.lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        except Exception as exc:
            raise _internal_error(exc, f"delete of doc {doc_id}") from exc
    # Drop the index row(s) and any leftover raw file.
    await server._db_pool.execute(
        "DELETE FROM rag_file_metadata WHERE workspace=$1 AND doc_id=$2", phys, doc_id
    )
    if meta and meta.get("job_id") and meta.get("file"):
        _job_path(phys, meta["job_id"], meta["file"]).unlink(missing_ok=True)
    return {"status": "deleted", "doc_id": doc_id}
