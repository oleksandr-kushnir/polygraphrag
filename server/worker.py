"""Background ingestion worker: a single asyncio task that drains the job queue and processes
each uploaded file into its workspace, with retry/backoff bookkeeping, plus the startup job
reload that re-enqueues in-flight jobs after a restart.

The worker is the top-level orchestrator: it ties together the workspace registry, the ingest
pipeline, and the metadata DB. It calls those through the package root (server.*) so the shared
runtime state and the functions the test suite patches are always the live ones; only static
configuration is read from server.config.
"""

from pathlib import Path

import server
from server import config
from server.ingest import _build_metadata
from server.references import _safe_ref_name
from server.workspaces import _get_ws_lock


def _job_path(workspace_id: str, job_id: str, filename: str) -> Path:
    """On-disk path for an uploaded file, namespaced per workspace.

    The filename is basenamed with `_safe_ref_name` (same helper that builds the LightRAG
    key) so a caller-supplied separator can neither create stray subdirectories nor escape
    the workspace dir via traversal (`../..`). The `{job_id}_` prefix keeps it unique.
    """
    d = Path(server.WORKING_DIR) / workspace_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{job_id}_{_safe_ref_name(filename)}"


async def _db_reload_jobs(pool) -> None:
    rows = await pool.fetch(
        "SELECT * FROM rag_file_metadata WHERE status NOT IN ('done', 'failed', 'save_failed')"
    )
    for row in rows:
        row = dict(row)
        job_id = row["job_id"]
        physical = row["workspace"]  # rag_file_metadata stores the physical workspace
        # Resolve the public id the worker uses to route to the right instance.
        pub_row = await pool.fetchrow(
            "SELECT id FROM rag_workspaces WHERE lightrag_workspace = $1 AND deleted_at IS NULL "
            "ORDER BY created_at LIMIT 1",
            physical,
        )
        record = {
            "job_id": job_id,
            "batch_id": row["batch_id"],
            "workspace": physical,
            "file": row["file"],
            "status": "pending",
            "attempts": row["attempts"],
            "error": None,
        }
        server._jobs[job_id] = record
        server._batches.setdefault(row["batch_id"], []).append(record)
        # Rebuild the path exactly as _job_path wrote it at upload time (basenamed filename).
        dest = _job_path(physical, job_id, row["file"])
        if pub_row is not None and dest.exists():
            description_text = _build_metadata(
                row["description"] or "", row["source_path"] or "", row["last_modified_time"] or ""
            )
            # Re-hand LightRAG the same identity (lightrag_input), NOT the real display file_path.
            lightrag_input = f"{job_id}_{_safe_ref_name(row['file'])}"
            await server._db_update_status(pool, job_id, "pending", row["attempts"], None)
            await server._job_queue.put(
                (pub_row["id"], job_id, dest, description_text, lightrag_input)
            )
        else:
            record["status"] = "failed"
            record["error"] = "File missing after restart"
            await server._db_update_status(
                pool, job_id, "failed", row["attempts"], "File missing after restart"
            )


async def _process_job(
    workspace_id: str,
    job_id: str,
    dest: Path,
    description_text: str,
    file_path: str | None = None,
) -> None:
    """Process one queued job into its workspace, with retry/backoff bookkeeping.
    Resolves the workspace's RAGAnything instance and serialises the insert on that
    workspace's lock. Re-enqueues (with workspace) on transient failure until MAX_RETRIES."""
    job = server._jobs[job_id]
    job["status"] = "processing"
    if server._db_pool:
        await server._db_update_status(server._db_pool, job_id, "processing", job["attempts"], None)
    try:
        rag_instance = await server.get_workspace_rag(workspace_id)
        lock = await _get_ws_lock(workspace_id)
        async with lock:
            result = await server._process_file(
                dest, rag_instance, description_text=description_text, file_path=file_path
            )
        # `_process_file` returns (doc_id, lightrag_key); tolerate a bare doc_id from older mocks.
        doc_id, lightrag_key = result if isinstance(result, tuple) else (result, None)
        job["status"] = "done"
        job["doc_id"] = doc_id
        if server._db_pool:
            await server._db_update_status(server._db_pool, job_id, "done", job["attempts"], None)
            if doc_id:
                await server._db_set_doc_id(server._db_pool, job_id, doc_id)
            if lightrag_key:
                await server._db_set_lightrag_key(server._db_pool, job_id, lightrag_key)
        # The DB index is the system of record; the raw bytes are redundant once ingested
        # (the source lives in the workspace), so drop them on success. Kept on retry/failure.
        dest.unlink(missing_ok=True)
    except Exception as exc:
        job["attempts"] += 1
        job["error"] = str(exc)
        if job["attempts"] < config.MAX_RETRIES:
            job["status"] = "retrying"
            if server._db_pool:
                await server._db_update_status(
                    server._db_pool, job_id, "retrying", job["attempts"], str(exc)
                )
            await server._job_queue.put((workspace_id, job_id, dest, description_text, file_path))
        else:
            job["status"] = "failed"
            if server._db_pool:
                await server._db_update_status(
                    server._db_pool, job_id, "failed", job["attempts"], str(exc)
                )
            dest.unlink(missing_ok=True)


async def _worker():
    while True:
        workspace_id, job_id, dest, description_text, file_path = await server._job_queue.get()
        try:
            await server._process_job(workspace_id, job_id, dest, description_text, file_path)
        finally:
            server._job_queue.task_done()
