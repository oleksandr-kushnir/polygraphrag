"""Postgres schema management and file/job metadata persistence.

These helpers take an asyncpg pool/connection explicitly (they are called both from the lifespan
startup and from request handlers), and read only static configuration from server.config. The
rag_file_metadata table is this service's own system-of-record, separate from LightRAG's storage.
"""

from server import config


async def _db_init(pool) -> None:
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS rag_file_metadata (
            job_id             TEXT PRIMARY KEY,
            batch_id           TEXT NOT NULL,
            file               TEXT NOT NULL,
            status             TEXT NOT NULL DEFAULT 'pending',
            attempts           INTEGER NOT NULL DEFAULT 0,
            error              TEXT,
            description        TEXT,
            source_path        TEXT,
            last_modified_time TEXT,
            uploaded_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS rag_workspaces (
            id                 TEXT PRIMARY KEY,
            name               TEXT NOT NULL,
            description        TEXT,
            lightrag_workspace TEXT NOT NULL,
            is_primary         BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at         TIMESTAMPTZ,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    # Add the workspace column WITHOUT a DEFAULT, backfill pre-existing rows with the
    # legacy physical workspace, then enforce NOT NULL. No column default → any future
    # insert that forgets to set workspace fails loudly instead of silently landing in
    # the wrong workspace. Idempotent: re-running these is a no-op.
    await pool.execute("ALTER TABLE rag_file_metadata ADD COLUMN IF NOT EXISTS workspace TEXT")
    await pool.execute(
        "UPDATE rag_file_metadata SET workspace = $1 WHERE workspace IS NULL",
        config.POSTGRES_WORKSPACE,
    )
    await pool.execute("ALTER TABLE rag_file_metadata ALTER COLUMN workspace SET NOT NULL")
    await pool.execute(
        "CREATE INDEX IF NOT EXISTS idx_rag_file_metadata_workspace ON rag_file_metadata (workspace)"
    )
    # Durable per-file index columns (system of record, survive raw-file deletion):
    #   content_hash — SHA-256 of ingested bytes (sync change-detection key)
    #   doc_id       — LightRAG doc-<md5> (precise delete key, captured at ingest)
    #   file_path    — the REAL, openable display path (caller path when provided, else filename)
    #   lightrag_key — the citation key LightRAG actually stored (read back at ingest); JOIN-ONLY,
    #                  never returned to clients. Maps a LightRAG reference back to this row.
    await pool.execute("ALTER TABLE rag_file_metadata ADD COLUMN IF NOT EXISTS content_hash TEXT")
    await pool.execute("ALTER TABLE rag_file_metadata ADD COLUMN IF NOT EXISTS doc_id TEXT")
    await pool.execute("ALTER TABLE rag_file_metadata ADD COLUMN IF NOT EXISTS file_path TEXT")
    await pool.execute("ALTER TABLE rag_file_metadata ADD COLUMN IF NOT EXISTS lightrag_key TEXT")
    # llm_model_extracted — the text LLM (LLM_MODEL) active when this file's entities were
    # extracted. Captured at ingest; surfaced in /query references. Pre-existing rows stay NULL
    # (they were ingested before this was tracked).
    await pool.execute(
        "ALTER TABLE rag_file_metadata ADD COLUMN IF NOT EXISTS llm_model_extracted TEXT"
    )
    # Backfill lightrag_key from the old file_path (which WAS the value passed to LightRAG), so
    # pre-existing rows keep resolving without a re-ingest.
    await pool.execute(
        "UPDATE rag_file_metadata SET lightrag_key = file_path WHERE lightrag_key IS NULL"
    )
    # Clean legacy display paths: rows ingested before the split stored the on-disk
    # `{job_id}_{filename}` token in file_path. Reset those to a real display value (the caller's
    # source_path, else the original filename) so references stop showing the internal token.
    # Real caller paths (path_root/source_path joins) never equal `{job_id}_{file}`, so they are
    # left untouched.
    await pool.execute(
        "UPDATE rag_file_metadata SET file_path = COALESCE(NULLIF(source_path, ''), file) "
        "WHERE file_path = job_id || '_' || file"
    )
    await pool.execute(
        "CREATE INDEX IF NOT EXISTS idx_rag_file_metadata_lightrag_key "
        "ON rag_file_metadata (workspace, lightrag_key)"
    )
    await _db_seed_primary_workspace(pool)


async def _db_seed_primary_workspace(pool) -> None:
    """First-boot seed: register the primary workspace (public id `default`), mapping it to the
    physical LightRAG workspace POSTGRES_WORKSPACE. Skipped once any primary row exists (the DB is
    authoritative thereafter)."""
    existing = await pool.fetchrow("SELECT id FROM rag_workspaces WHERE is_primary = TRUE LIMIT 1")
    if existing is not None:
        return
    await pool.execute(
        """INSERT INTO rag_workspaces (id, name, description, lightrag_workspace, is_primary)
               VALUES ($1, $2, $3, $4, TRUE)
           ON CONFLICT (id) DO NOTHING""",
        config.PRIMARY_WORKSPACE_ID,
        config.PRIMARY_WORKSPACE_NAME,
        config.PRIMARY_WORKSPACE_DESCRIPTION,
        config.POSTGRES_WORKSPACE,
    )


async def _db_insert_job(
    pool,
    record: dict,
    description: str,
    source_path: str,
    last_modified_time: str,
    content_hash: str | None = None,
    file_path: str | None = None,
    lightrag_key: str | None = None,
) -> None:
    await pool.execute(
        """INSERT INTO rag_file_metadata
               (job_id, batch_id, workspace, file, description, source_path, last_modified_time,
                content_hash, file_path, lightrag_key, llm_model_extracted)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
           ON CONFLICT (job_id) DO NOTHING""",
        record["job_id"],
        record["batch_id"],
        record["workspace"],
        record["file"],
        description or None,
        source_path or None,
        last_modified_time or None,
        content_hash or None,
        file_path or None,
        lightrag_key or None,
        config.LLM_MODEL,
    )


async def _db_update_status(
    pool, job_id: str, status: str, attempts: int, error: str | None
) -> None:
    await pool.execute(
        "UPDATE rag_file_metadata SET status=$2, attempts=$3, error=$4 WHERE job_id=$1",
        job_id,
        status,
        attempts,
        error,
    )


async def _db_set_doc_id(pool, job_id: str, doc_id: str) -> None:
    """Persist the LightRAG doc id captured at successful ingest (the precise delete/index key)."""
    await pool.execute(
        "UPDATE rag_file_metadata SET doc_id=$2 WHERE job_id=$1",
        job_id,
        doc_id,
    )


async def _db_set_lightrag_key(pool, job_id: str, lightrag_key: str) -> None:
    """Persist the canonical citation key LightRAG stored (read back at ingest). This is the
    authoritative reference-join key — overwrites the provisional value set at upload time."""
    await pool.execute(
        "UPDATE rag_file_metadata SET lightrag_key=$2 WHERE job_id=$1",
        job_id,
        lightrag_key,
    )
