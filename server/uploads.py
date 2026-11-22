"""Stage raw documents and queue the shared background ingestion worker."""

import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import UploadFile

import server
from server.db import _db_insert_job
from server.deps import _batch_response
from server.graph_store import graph_storage_ready
from server.ingest import _build_metadata, _join_path, ingestion_roles
from server.providers import require_providers
from server.references import _safe_ref_name
from server.worker import _job_path


async def enqueue_uploads(ws: dict, files: list[UploadFile], meta_list: list[dict]) -> dict:
    if graph_storage_ready():
        require_providers(
            "upload.batch",
            *dict.fromkeys(
                role
                for file in files
                for role in ingestion_roles(Path(file.filename or "").suffix.lower())
            ),
        )
    pub = ws["id"]
    phys = ws["lightrag_workspace"]
    batch_id = uuid.uuid4().hex[:8]
    entries = []
    for i, file in enumerate(files):
        job_id = uuid.uuid4().hex[:8]
        filename = file.filename or "upload"
        dest = None
        m = meta_list[i] if i < len(meta_list) else {}
        description = m.get("description", "") or ""
        source_path = m.get("source_path", "") or ""
        last_modified = m.get("last_modified_time", "") or ""
        path_root = m.get("path_root", "") or ""
        try:
            dest = _job_path(phys, job_id, filename)
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
            lightrag_input = f"{job_id}_{_safe_ref_name(filename)}"
            display_path = (
                _join_path(path_root, source_path)
                if (path_root and source_path)
                else (source_path or _safe_ref_name(filename))
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
            server._jobs[job_id] = record
            entries.append(record)
            server._job_queue.put_nowait((pub, job_id, dest, description_text, lightrag_input))
        except Exception as exc:
            if dest is not None:
                try:
                    dest.unlink(missing_ok=True)
                except OSError:
                    logging.exception("Could not clean up failed upload %s", job_id)
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
