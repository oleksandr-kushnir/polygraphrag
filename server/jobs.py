"""Workspace-scoped ingest job lookup and public file metadata."""

from fastapi import HTTPException

import server


async def lookup_ingest_job(ws: dict, job_id: str) -> dict:
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
