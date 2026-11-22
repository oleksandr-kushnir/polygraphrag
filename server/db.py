"""Postgres schema management and file/job metadata persistence.

These helpers take an asyncpg pool/connection explicitly (they are called both from the lifespan
startup and from request handlers), and read only static configuration from server.config. The
rag_file_metadata table is this service's own system-of-record, separate from LightRAG's storage.
"""

import server
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
            deleted_at         TIMESTAMPTZ,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    # Migration: drop the legacy is_primary flag. Workspaces are peers now — no primary/special
    # status, no delete-protection. Idempotent; a no-op on registries created after this change.
    await pool.execute("ALTER TABLE rag_workspaces DROP COLUMN IF EXISTS is_primary")
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
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS rag_graph_imports (
            workspace TEXT PRIMARY KEY REFERENCES rag_workspaces(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS rag_graph_merge_previews (
            workspace TEXT NOT NULL REFERENCES rag_workspaces(id) ON DELETE CASCADE,
            token TEXT NOT NULL,
            input JSONB NOT NULL,
            fingerprint TEXT NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL DEFAULT now()+interval '1 hour',
            PRIMARY KEY (workspace,token)
        );
        CREATE TABLE IF NOT EXISTS rag_graph_records (
            workspace TEXT NOT NULL REFERENCES rag_workspaces(id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('entity', 'relationship', 'chunk')),
            id TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1,
            extracted JSONB,
            manual JSONB,
            identity_fallback JSONB,
            suppressed BOOLEAN NOT NULL DEFAULT FALSE,
            redirect TEXT,
            source TEXT GENERATED ALWAYS AS ((COALESCE(extracted, '{}') ||
                                             COALESCE(manual, '{}'))->>'source') STORED,
            target TEXT GENERATED ALWAYS AS ((COALESCE(extracted, '{}') ||
                                             COALESCE(manual, '{}'))->>'target') STORED,
            endpoint_kind TEXT GENERATED ALWAYS AS ('entity'::text) STORED,
            PRIMARY KEY (workspace, kind, id),
            FOREIGN KEY (workspace, endpoint_kind, source)
                REFERENCES rag_graph_records(workspace, kind, id),
            FOREIGN KEY (workspace, endpoint_kind, target)
                REFERENCES rag_graph_records(workspace, kind, id)
        );
        CREATE INDEX IF NOT EXISTS rag_graph_source ON rag_graph_records (workspace, source);
        CREATE INDEX IF NOT EXISTS rag_graph_target ON rag_graph_records (workspace, target);
        CREATE INDEX IF NOT EXISTS rag_graph_name ON rag_graph_records
            (workspace, lower(COALESCE(manual->>'name', extracted->>'name')));
        CREATE INDEX IF NOT EXISTS rag_graph_redirect ON rag_graph_records (workspace, redirect)
            WHERE redirect IS NOT NULL;
        CREATE OR REPLACE VIEW rag_graph_active AS
            SELECT r.*,
                COALESCE(r.identity_fallback, '{}') || COALESCE(r.extracted, '{}') ||
                    COALESCE(r.manual, '{}') AS fields,
                CASE WHEN r.kind='entity' THEN (
                    SELECT jsonb_agg(evidence.source ORDER BY retired.id, evidence.source::text)
                    FROM rag_graph_records retired
                    CROSS JOIN LATERAL jsonb_array_elements(
                        COALESCE(retired.extracted->'sources', '[]') ||
                        CASE WHEN retired.manual->>'source_ref' IS NULL THEN '[]'::jsonb
                             ELSE jsonb_build_array(jsonb_build_object(
                                 'origin', 'manual', 'source_ref', retired.manual->>'source_ref'
                             )) END
                    ) AS evidence(source)
                    WHERE retired.workspace=r.workspace AND retired.kind='entity'
                      AND retired.redirect=r.id
                ) END AS merged_sources
            FROM rag_graph_records r
            WHERE NOT r.suppressed AND r.redirect IS NULL
              AND (r.extracted IS NOT NULL OR r.manual IS NOT NULL OR r.identity_fallback IS NOT NULL)
              AND (r.kind<>'relationship' OR (
                SELECT count(*) FROM rag_graph_records n
                WHERE n.workspace=r.workspace AND n.kind='entity'
                  AND n.id IN (r.source, r.target)
                  AND NOT n.suppressed AND n.redirect IS NULL
                  AND (n.extracted IS NOT NULL OR n.manual IS NOT NULL OR n.identity_fallback IS NOT NULL)
              )=CASE WHEN r.source=r.target THEN 1 ELSE 2 END);
        CREATE OR REPLACE VIEW rag_graph_index_inputs AS
            SELECT workspace, kind, id, fields, origins, content,
                   md5(content) AS content_key
            FROM (
                SELECT r.workspace, r.kind, r.id, r.fields,
                    ARRAY_REMOVE(ARRAY[
                        CASE WHEN r.manual IS NOT NULL
                                OR r.merged_sources @> '[{"origin":"manual"}]'
                             THEN 'manual' END,
                        CASE WHEN r.extracted IS NOT NULL
                                OR r.merged_sources @> '[{"origin":"extracted"}]'
                             THEN 'extracted' END
                    ], NULL) AS origins,
                    CASE WHEN r.kind='chunk' THEN COALESCE(r.fields->>'description', '')
                         ELSE concat_ws(E'\n', r.fields->>'name', r.fields->>'type',
                             CASE WHEN r.kind='relationship' THEN COALESCE(
                                 s.manual->>'name', s.extracted->>'name',s.identity_fallback->>'name') END,
                             CASE WHEN r.kind='relationship' THEN COALESCE(
                                 t.manual->>'name', t.extracted->>'name',t.identity_fallback->>'name') END,
                             r.fields->>'description') END AS content
                FROM rag_graph_active r
                LEFT JOIN rag_graph_records s ON
                    (s.workspace,s.kind,s.id)=(r.workspace,'entity',r.source)
                LEFT JOIN rag_graph_records t ON
                    (t.workspace,t.kind,t.id)=(r.workspace,'entity',r.target)
            ) AS inputs;
        CREATE TABLE IF NOT EXISTS rag_graph_vectors (
            workspace TEXT NOT NULL REFERENCES rag_workspaces(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            id TEXT NOT NULL,
            profile TEXT NOT NULL,
            content_key TEXT NOT NULL,
            embedding vector NOT NULL,
            PRIMARY KEY (workspace,kind,id,profile)
        );
        CREATE TABLE IF NOT EXISTS rag_embedding_jobs (
            workspace TEXT NOT NULL REFERENCES rag_workspaces(id) ON DELETE CASCADE,
            id TEXT NOT NULL,
            profile TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'queued',
            lease_token TEXT,
            lease_until TIMESTAMPTZ,
            ready_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (workspace,id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS rag_embedding_active_job ON
            rag_embedding_jobs(workspace,profile)
            WHERE state IN ('queued','running','waiting_for_provider');
        CREATE TABLE IF NOT EXISTS rag_embedding_items (
            workspace TEXT NOT NULL,
            job TEXT NOT NULL,
            kind TEXT NOT NULL,
            id TEXT NOT NULL,
            content_key TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending',
            PRIMARY KEY (workspace,job,kind,id),
            FOREIGN KEY (workspace,job) REFERENCES rag_embedding_jobs(workspace,id) ON DELETE CASCADE
        );
    """)
    await _db_seed_default_workspace(pool)
    server._graph_pool = pool


async def _db_seed_default_workspace(pool) -> None:
    """Bootstrap seed: on a completely empty registry, register an ordinary `default` workspace
    (mapped to the physical LightRAG workspace POSTGRES_WORKSPACE) so a fresh install is usable
    out of the box. Skipped as soon as ANY workspace row exists — so once real workspaces are in
    play, deleting `default` is permanent and this never re-creates it. The DB is authoritative
    thereafter."""
    existing = await pool.fetchrow("SELECT id FROM rag_workspaces LIMIT 1")
    if existing is not None:
        return
    await pool.execute(
        """INSERT INTO rag_workspaces (id, name, description, lightrag_workspace)
               VALUES ($1, $2, $3, $4)
           ON CONFLICT (id) DO NOTHING""",
        config.SEED_WORKSPACE_ID,
        config.SEED_WORKSPACE_NAME,
        config.SEED_WORKSPACE_DESCRIPTION,
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
