import asyncio
import base64
import hashlib
import json
import logging
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
    _VISION_IS_OPENAI,
    API_TOKENS,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_RETRIES,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_WORKSPACE,
    PRIMARY_WORKSPACE_DESCRIPTION,
    PRIMARY_WORKSPACE_ID,
    PRIMARY_WORKSPACE_NAME,
    QUERY_LLM_MODEL,
    RAG_FILTER_TOPK_BOOST,
    RAG_MIN_CONTENT_FOR_ENTITIES,
    RAG_REQUIRE_GRAPH_EXTRACTION,
    VISION_API_KEY,
    VISION_BASE_URL,
    VISION_MODEL,
    WHISPER_API_KEY,
    WHISPER_BASE_URL,
    WHISPER_MODEL,
    WORKING_DIR,
    _active_llm_cfg,
    _llm_call_kwargs,
    _llm_phase,
)


class IngestionIncompleteError(RuntimeError):
    """Raised when LightRAG stored chunks but did not fully ingest a document (e.g. entity
    extraction failed/timed out). Drives the normal retry/backoff path in _process_job."""


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


async def _llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    import openai

    model, base_url, api_key, is_openai = _active_llm_cfg()
    logging.debug("llm call: phase=%s model=%s", _llm_phase.get(), model)
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        **_llm_call_kwargs(kwargs, is_openai),
    )
    return resp.choices[0].message.content


async def _vision_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    image_data=None,
    messages=None,
    **kwargs,
):
    import openai

    client = openai.AsyncOpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
    if messages is not None:
        final_messages = messages
    elif image_data is not None:
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            {"type": "text", "text": prompt},
        ]
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(history_messages)
        final_messages.append({"role": "user", "content": content})
    else:
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(history_messages)
        final_messages.append({"role": "user", "content": prompt})
    resp = await client.chat.completions.create(
        model=VISION_MODEL,
        messages=final_messages,
        **_llm_call_kwargs(kwargs, is_openai=_VISION_IS_OPENAI),
    )
    return resp.choices[0].message.content


async def _embedding_func(texts: list[str]):
    import numpy as np
    import openai

    client = openai.AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data])


# --- Document processing ---

_VISION_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".webm"}
_OFFICE_SUFFIXES = {".docx", ".xlsx", ".pptx"}
_TEXT_SUFFIXES = {".txt", ".md", ".html", ".csv"}

# Full catalog of file types this service can parse & ingest (22 extensions).
# Unknown types fall through to MinerU but their ingestion is NOT verified.
#   TEXT   (text LLM):                     md, txt, csv, html
#   DOCS   (vision LLM via LibreOffice):   pdf, docx, pptx, xlsx
#   IMAGES (vision LLM, per file):         jpg, jpeg, png, gif, bmp, tiff, webp
#   AUDIO  (whisper transcription):        mp3, wav, m4a, ogg, flac, opus, webm
# (Each role's provider is set per-endpoint in .env — see the endpoint block near the top.)

_EXTRACTION_PROMPT = (
    "Extract all content from this document. "
    "Return all text in reading order. "
    "Render every table as a GitHub-flavoured Markdown table. "
    "For each figure, chart, or image write a concise description enclosed in [Figure: ...] brackets."
)


async def _extract_with_vision(path: Path) -> str:
    import base64

    import openai

    client = openai.AsyncOpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
    b64 = base64.b64encode(path.read_bytes()).decode()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        file_part = {
            "type": "file",
            "file": {"filename": path.name, "file_data": f"data:application/pdf;base64,{b64}"},
        }
    else:
        mime = {"jpg": "jpeg"}.get(suffix.lstrip("."), suffix.lstrip("."))
        file_part = {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}
    resp = await client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": _EXTRACTION_PROMPT}, file_part]}
        ],
        **_llm_call_kwargs({"max_completion_tokens": 16000}, is_openai=_VISION_IS_OPENAI),
    )
    return resp.choices[0].message.content


async def _transcribe_audio(path: Path) -> str:
    import openai

    client = openai.AsyncOpenAI(api_key=WHISPER_API_KEY, base_url=WHISPER_BASE_URL)
    with path.open("rb") as f:
        transcript = await client.audio.transcriptions.create(
            model=WHISPER_MODEL, file=f, response_format="text"
        )
    return transcript


async def _convert_office_to_pdf(path: Path) -> Path:
    import tempfile

    out_dir = Path(tempfile.mkdtemp())
    proc = await asyncio.create_subprocess_exec(
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    pdf_path = out_dir / (path.stem + ".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"LibreOffice conversion failed for {path.name}")
    return pdf_path


async def _count_doc_entities(rag_instance: RAGAnything, doc_id: str) -> int:
    """Number of entities LightRAG extracted for `doc_id` (0 if none).
    Reads the per-document `lightrag_full_entities` index, namespaced by the instance's workspace.
    Returns 1 (i.e. "can't tell, don't block") when no DB pool is available."""
    if _db_pool is None:
        return 1
    workspace = getattr(rag_instance.lightrag, "workspace", None) or POSTGRES_WORKSPACE
    row = await _db_pool.fetchrow(
        "SELECT count FROM lightrag_full_entities WHERE workspace = $1 AND id = $2",
        workspace,
        doc_id,
    )
    return int(row["count"]) if row and row["count"] is not None else 0


def _doc_status_field(status_doc, name, default=None):
    """Read a field from a LightRAG doc-status record, which may be a plain dict (PG storage
    returns dicts from aget_docs_by_ids) or a DocProcessingStatus object."""
    if isinstance(status_doc, dict):
        return status_doc.get(name, default)
    return getattr(status_doc, name, default)


def _content_doc_id(content: str) -> str:
    """The LightRAG doc id for a piece of content. Used both to pin the id at insert time
    (ainsert(ids=[...])) and to look the record back up during verification, so the two always
    agree regardless of LightRAG's internal id derivation."""
    from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

    return compute_mdhash_id(sanitize_text_for_encoding(content), prefix="doc-")


async def _verify_ingestion(
    rag_instance: RAGAnything, content: str
) -> tuple[str, str | None, str, str | None]:
    """Confirm LightRAG fully ingested `content` (chunks AND graph), not just stored chunks.
    Returns (verdict, doc_id, reason, lightrag_key) where verdict is "ok"/"failed". The doc id is
    computed the same way it was pinned at insert (`doc-<md5(sanitized_content)>`). `lightrag_key`
    is the `file_path` LightRAG *actually stored* for this doc (its canonical citation key), read
    straight back from the doc-status — so our reference join uses LightRAG's own value and never
    has to reproduce its canonicalization."""
    doc_id = _content_doc_id(content)
    docs = await rag_instance.lightrag.aget_docs_by_ids(doc_id)
    status_doc = docs.get(doc_id) if isinstance(docs, dict) else None
    if status_doc is None:
        return "failed", doc_id, "no_doc_status", None
    lightrag_key = _doc_status_field(status_doc, "file_path", None)
    raw_status = _doc_status_field(status_doc, "status")
    status = getattr(raw_status, "value", raw_status)  # DocStatus enum → str; str stays str
    if status != "processed":
        return "failed", doc_id, f"doc_status={status}", lightrag_key
    if RAG_REQUIRE_GRAPH_EXTRACTION:
        content_length = _doc_status_field(status_doc, "content_length", None) or len(content)
        if (
            content_length >= RAG_MIN_CONTENT_FOR_ENTITIES
            and await _count_doc_entities(rag_instance, doc_id) == 0
        ):
            return "failed", doc_id, "empty_graph", lightrag_key
    return "ok", doc_id, "processed", lightrag_key


def _join_path(path_root: str, rel_path: str) -> str:
    """Join a caller-supplied root prefix with a workspace-relative path into a single absolute
    identity (e.g. /data/corpus/sub/dir/file.pdf). Stored as the LightRAG file_path so query
    references point at a path the caller's own tooling can resolve back to the source file."""
    return f"{path_root.rstrip('/')}/{rel_path.lstrip('/')}"


async def _process_file(
    path: Path,
    rag_instance: RAGAnything,
    description_text: str = "",
    file_path: str | None = None,
) -> tuple[str | None, str | None]:
    """Parse `path` and insert it into LightRAG, then verify the ingestion actually completed.
    `file_path` is the identity handed to LightRAG (the `{job_id}_{basename}` lightrag_input for
    uploads; falls back to the on-disk basename). Returns `(doc_id, lightrag_key)` — the LightRAG
    doc id and the file_path LightRAG actually stored (its canonical citation key), both None for
    the multimodal fallback path; raises IngestionIncompleteError if graph extraction did not
    complete.

    Runs under the "extract" LLM phase so the entity/relationship extraction calls route to the
    extraction provider (LLM_*), independent of the query-time provider (QUERY_LLM_*)."""
    token = _llm_phase.set("extract")
    logging.info(
        "extraction phase: %s will extract entities for %s (base_url=%s)",
        LLM_MODEL,
        file_path or path.name,
        LLM_BASE_URL or "openai",
    )
    try:
        return await _process_file_impl(path, rag_instance, description_text, file_path)
    finally:
        _llm_phase.reset(token)


async def _process_file_impl(
    path: Path,
    rag_instance: RAGAnything,
    description_text: str = "",
    file_path: str | None = None,
) -> str | None:
    suffix = path.suffix.lower()
    if suffix in _VISION_SUFFIXES:
        text = await _extract_with_vision(path)
    elif suffix in _AUDIO_SUFFIXES:
        text = await _transcribe_audio(path)
    elif suffix in _OFFICE_SUFFIXES:
        if suffix == ".xlsx" and path.stat().st_size > 10 * 1024 * 1024:
            size_mb = path.stat().st_size / 1024 / 1024
            raise ValueError(
                f"{path.name} is {size_mb:.1f} MB — "
                "xlsx files over 10 MB are rejected to avoid silent data truncation "
                "at OpenAI's 100-page/32 MB PDF limit."
            )
        pdf_path = await _convert_office_to_pdf(path)
        try:
            text = await _extract_with_vision(pdf_path)
        finally:
            import shutil

            shutil.rmtree(pdf_path.parent, ignore_errors=True)
    elif suffix in _TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        await rag_instance.process_document_complete(file_path=str(path))
        return None, None
    content = text
    if description_text:
        content += f"\n\nDescription: {description_text}"
    # Pin the doc id so post-insert verification can find the record. LightRAG derives its own
    # id from the (internally normalized) content, and that derivation is not guaranteed stable
    # across releases; passing an explicit `ids=` makes the stored id deterministic and equal to
    # what _verify_ingestion recomputes from the same content.
    doc_id = _content_doc_id(content)
    # `file_path` here is the LightRAG identity (lightrag_input = `{job_id}_{basename}` for
    # uploads). LightRAG canonicalizes it; we read the stored value back below as the join key.
    lightrag_input = file_path or path.name
    await rag_instance.lightrag.ainsert(content, ids=[doc_id], file_paths=[lightrag_input])
    # Don't trust ainsert returning — confirm the graph extraction actually completed, and
    # capture the exact file_path LightRAG stored (its canonical citation key).
    verdict, doc_id, reason, lightrag_key = await _verify_ingestion(rag_instance, content)
    if verdict != "ok":
        # Clean up the partial doc (+ its LLM cache) so the retry re-extracts instead of being
        # skipped as a duplicate or served a stale cached extraction.
        if doc_id:
            try:
                await rag_instance.lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
            except Exception as exc:
                logging.warning("cleanup of partial doc %s failed: %s", doc_id, exc)
        raise IngestionIncompleteError(f"ingestion incomplete ({reason})")
    return doc_id, (lightrag_key or lightrag_input)


def _build_metadata(description: str, source_path: str, last_modified: str) -> str:
    # Only `description` is injected into chunk text (used for vector search
    # and entity extraction). `source_path` and `last_modified` remain in
    # `rag_file_metadata` and are surfaced via /query references; they are
    # not embedded because file paths and ISO timestamps produce noisy
    # entities without improving retrieval.
    return description or ""


# --- DB helpers ---


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
        POSTGRES_WORKSPACE,
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
        PRIMARY_WORKSPACE_ID,
        PRIMARY_WORKSPACE_NAME,
        PRIMARY_WORKSPACE_DESCRIPTION,
        POSTGRES_WORKSPACE,
    )


# --- Workspace instance registry ---


async def _get_ws_lock(workspace_id: str) -> asyncio.Lock:
    """Get (or lazily create) the per-workspace lock. Guards instance creation and
    serialises inserts for that workspace."""
    async with _registry_lock:
        return _ws_locks.setdefault(workspace_id, asyncio.Lock())


async def _lookup_workspace(workspace_id: str):
    """Return the rag_workspaces row for an ACTIVE (not soft-deleted) workspace, else None."""
    if _db_pool is None:
        return None
    return await _db_pool.fetchrow(
        "SELECT id, name, description, lightrag_workspace, is_primary "
        "FROM rag_workspaces WHERE id = $1 AND deleted_at IS NULL",
        workspace_id,
    )


async def _build_workspace_rag(workspace_id: str, physical_workspace: str) -> RAGAnything:
    """Construct + initialize a RAGAnything/LightRAG pair for one workspace.
    `workspace_id` is the public id (names the on-disk working_dir); `physical_workspace`
    is the LightRAG `workspace=` value that namespaces Postgres rows + the AGE graph."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    working_dir = str(Path(WORKING_DIR) / workspace_id)
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        model_name=EMBEDDING_MODEL,
        func=_embedding_func,
    )
    lightrag_instance = LightRAG(
        working_dir=working_dir,
        llm_model_func=_llm_func,
        embedding_func=embedding_func,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="PGGraphStorage",
        doc_status_storage="PGDocStatusStorage",
        workspace=physical_workspace,
    )
    await lightrag_instance.initialize_storages()
    return RAGAnything(
        llm_model_func=_llm_func,
        vision_model_func=_vision_func,
        embedding_func=embedding_func,
        lightrag=lightrag_instance,
    )


async def get_workspace_rag(workspace_id: str) -> RAGAnything:
    """Return the cached RAGAnything for a workspace, building it lazily on first use.
    Raises HTTPException(404) if the workspace is unknown or soft-deleted."""
    cached = _rag_instances.get(workspace_id)
    if cached is not None:
        return cached
    lock = await _get_ws_lock(workspace_id)
    async with lock:
        cached = _rag_instances.get(workspace_id)
        if cached is not None:
            return cached
        row = await _lookup_workspace(workspace_id)
        if row is None:
            raise HTTPException(404, f"Workspace {workspace_id!r} not found")
        instance = await _build_workspace_rag(workspace_id, row["lightrag_workspace"])
        _rag_instances[workspace_id] = instance
        return instance


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
        LLM_MODEL,
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


def _ref_basename(path: str) -> str:
    """Display filename from a stored file_path (absolute path or bare name)."""
    return path.replace("\\", "/").rsplit("/", 1)[-1] if path else path


def _safe_ref_name(name: str | None) -> str:
    """Basename a caller-supplied filename ourselves (strip any '/' or '\\' and directory
    parts). Used to build the LightRAG identity `{job_id}_{safe}`: with no path separator,
    LightRAG's basename canonicalization can never drop the unique `job_id` prefix, so distinct
    documents keep distinct keys by construction."""
    return (name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()


def _clean_needles(needles: list[str] | None) -> list[str]:
    """Drop blank/whitespace-only entries from a file_path_contains list. An all-blank list
    (e.g. Swagger UI's auto-populated `[""]`) therefore means 'no filter → all data', never an
    accidental empty result set."""
    return [n for n in (needles or []) if n and n.strip()]


_JOB_KEY_PREFIX = re.compile(r"^[0-9a-f]{6,}_")


def _strip_job_prefix(name: str) -> str:
    """Strip a leading `{job_id}_` hex prefix from a citation key so no internal token/hex
    surfaces when a reference can't be resolved to a metadata row. Unchanged if absent."""
    return _JOB_KEY_PREFIX.sub("", name) if name else name


def _rewrite_answer_refs(result: str, raw_references, meta_map: dict[str, dict]) -> str:
    """Rewrite LightRAG's internal citation keys embedded in the answer PROSE to the clean
    filename — the same no-internal-name guarantee we give the structured `references[]`.

    LightRAG tags each retrieved chunk with its `file_path` (our internal `{job_id}_{basename}`
    join key) and its answer prompt renders a `### References` list from those values, so the raw
    key leaks into `result`. Each key is a unique, slash-free `{job_id}_{basename}` token, so a
    literal replace is safe (it cannot collide with unrelated prose). A resolved key becomes the
    clean original basename (`_safe_ref_name(row.file)`); an unresolved key falls back to stripping
    its `{job_id}_` prefix so no hex/internal token ever surfaces. Pure string mapping over values
    we mint and store, so it stays correct across LightRAG versions and prompt formats."""
    if not result:
        return result
    replacements: dict[str, str] = {}
    for ref in raw_references or []:
        key = ref.get("file_path") or ""
        if not key or key in replacements:
            continue
        row = meta_map.get(key)
        display = _safe_ref_name(row["file"]) if row else _strip_job_prefix(key)
        if display and display != key:
            replacements[key] = display
    # Longest key first: a key that is a substring of another can't be partially rewritten.
    for key in sorted(replacements, key=len, reverse=True):
        result = result.replace(key, replacements[key])
    return result


def _path_matches_any(value: str | None, needles: list[str] | None) -> bool:
    """OR-filter a stored file_path against a list of case-insensitive substrings.
    `value` may be a GRAPH_FIELD_SEP-joined list of paths (entities/relationships carry the
    joined source-file list); a plain substring test still matches within it. Empty/None/blank
    `needles` => no filter (keep everything); a non-empty needle list with an empty `value`
    => no match."""
    needles = _clean_needles(needles)
    if not needles:
        return True
    if not value:
        return False
    lowered = value.lower()
    return any(n.lower() in lowered for n in needles)


async def _db_fetch_metadata_by_key(pool, keys: list[str], phys: str | None) -> dict[str, dict]:
    """Map each LightRAG citation key -> its rag_file_metadata row. Layered so a real path always
    resolves even across LightRAG canonicalization quirks:
      1. exact match on `lightrag_key` (the value LightRAG stored, read back at ingest);
      2. fallback: the original filename column `file` == key (rescues legacy rows whose
         backfilled key was a full path but LightRAG returned only the basename).
    Newest upload wins on duplicates. `file_path` returned here is the REAL display path."""
    if not keys:
        return {}
    cols = (
        "lightrag_key, file_path, file, job_id, description, source_path, "
        "last_modified_time, uploaded_at, llm_model_extracted"
    )
    if phys is not None:
        rows = await pool.fetch(
            f"SELECT {cols} FROM rag_file_metadata "
            "WHERE workspace = $2 AND (lightrag_key = ANY($1) OR file = ANY($1)) "
            "ORDER BY uploaded_at DESC",
            keys,
            phys,
        )
    else:
        rows = await pool.fetch(
            f"SELECT {cols} FROM rag_file_metadata "
            "WHERE (lightrag_key = ANY($1) OR file = ANY($1)) ORDER BY uploaded_at DESC",
            keys,
        )
    by_key: dict[str, dict] = {}
    by_file: dict[str, dict] = {}
    for r in rows:  # newest-first; setdefault keeps the newest
        d = dict(r)
        if d.get("lightrag_key"):
            by_key.setdefault(d["lightrag_key"], d)
        if d.get("file"):
            by_file.setdefault(d["file"], d)
    resolved = {}
    for k in keys:
        m = by_key.get(k) or by_file.get(k)
        if m is not None:
            resolved[k] = m
    return resolved


def _graph_field_sep() -> str:
    """LightRAG's multi-value delimiter for joined source lists (entities/relationships carry
    several sources in one `file_path`). Read it from LightRAG so we stay in harmony with its
    value; fall back to the stable literal if the import is unavailable (e.g. under test stubs)."""
    try:
        from lightrag.utils import GRAPH_FIELD_SEP as sep

        if isinstance(sep, str) and sep:
            return sep
    except Exception:
        pass
    return "<SEP>"


def _resolve_joined_path(value: str, meta_map: dict[str, dict], sep: str) -> str:
    """Resolve a (possibly GRAPH_FIELD_SEP-joined) list of LightRAG keys to real document paths,
    preserving the SEP structure. Each segment maps to its rag_file_metadata `file_path`; an
    unresolved segment falls back to a prefix-stripped basename so no `{job_id}_` token surfaces."""
    return sep.join(
        ((meta_map.get(s.strip()) or {}).get("file_path") or _strip_job_prefix(s.strip()))
        for s in value.split(sep)
        if s.strip()
    )


async def _resolve_block_file_paths(data: dict, phys: str | None) -> None:
    """Rewrite LightRAG's internal citation keys in entity/relationship/chunk `file_path` fields to
    the REAL document path from Postgres (same `rag_file_metadata` join as references), so the raw
    `{job_id}_{basename}` key never surfaces in `/query/data`. Multi-source fields are GRAPH_FIELD_
    SEP-joined lists; every segment is resolved and the SEP structure preserved. Mutates in place.
    """
    sep = _graph_field_sep()
    blocks = [data.get("entities") or [], data.get("relationships") or [], data.get("chunks") or []]
    keys = {
        s.strip()
        for block in blocks
        for item in block
        for s in (item.get("file_path") or "").split(sep)
        if s.strip()
    }
    if not keys:
        return
    meta_map = await _db_fetch_metadata_by_key(_db_pool, list(keys), phys) if _db_pool else {}
    for block in blocks:
        for item in block:
            fp = item.get("file_path")
            if fp:
                item["file_path"] = _resolve_joined_path(fp, meta_map, sep)


async def _resolve_graph_paths(elements, phys: str | None) -> None:
    """Same real-path resolution as `_resolve_block_file_paths`, for knowledge-graph nodes AND
    edges whose `properties['file_path']` (shown in graph.html tooltips and used by its file_path
    filter) is a GRAPH_FIELD_SEP-joined list of internal keys. Both nodes and relationships carry
    source lists, so pass them together for a single DB fetch. Mutates properties in place."""
    sep = _graph_field_sep()
    keys = {
        s.strip()
        for el in elements
        for s in ((el.properties or {}).get("file_path") or "").split(sep)
        if s.strip()
    }
    if not keys:
        return
    meta_map = await _db_fetch_metadata_by_key(_db_pool, list(keys), phys) if _db_pool else {}
    for el in elements:
        props = el.properties or {}
        fp = props.get("file_path")
        if fp:
            props["file_path"] = _resolve_joined_path(fp, meta_map, sep)
            el.properties = props


async def _build_references(
    raw_references: list[dict] | None, phys: str | None = None, answered_model: str | None = None
) -> tuple[list[dict], dict[str, dict]]:
    """Resolve LightRAG references to their REAL document path + metadata via rag_file_metadata,
    joining on the citation key LightRAG returns (matched against `lightrag_key`). Shared by
    /query and /query/data. Returns `(references, meta_map)` where `meta_map` is the internal
    key→row map (used by /query to rewrite the answer prose; never emitted in any response).

    References expose only the real, openable `file_path` (from Postgres) plus enrichment.
    LightRAG's internal name / our join key are NEVER emitted; when a reference can't be resolved
    to a row, `file_path` is null (we do not echo LightRAG's raw internal value).

    `answered_model` is the text LLM that synthesised the answer for THIS query; added as
    `llm_model_answered` only when supplied (so /query/data, which generates no answer, omits it).
    """
    keys = []
    for ref in raw_references or []:
        keys.append(ref.get("file_path", "") or "")  # LightRAG's citation key (its internal name)
    meta_map: dict[str, dict] = {}
    if any(keys) and _db_pool:
        meta_map = await _db_fetch_metadata_by_key(_db_pool, [k for k in keys if k], phys)
    references = []
    for ref, key in zip(raw_references or [], keys):
        m = meta_map.get(key)
        out = {
            "reference_id": ref.get("reference_id"),
            "file_path": m.get("file_path") if m else None,  # REAL path only; never the key
            "job_id": m.get("job_id") if m else None,
            "file_description": m.get("description") if m else None,
            "last_modified_time": m.get("last_modified_time") if m else None,
        }
        ua = m.get("uploaded_at") if m else None
        out["uploaded_at"] = ua.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(ua, "strftime") else ua
        out["llm_model_extracted"] = m.get("llm_model_extracted") if m else None
        if answered_model is not None:
            out["llm_model_answered"] = answered_model
        references.append(out)
    return references, meta_map


# --- Graph visualisation ---
# Rendering lives in server.graph; the graph.html endpoint calls _build_graph_html.
from server.graph import _build_graph_html  # noqa: E402


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


def _job_path(workspace_id: str, job_id: str, filename: str) -> Path:
    """On-disk path for an uploaded file, namespaced per workspace.

    The filename is basenamed with `_safe_ref_name` (same helper that builds the LightRAG
    key) so a caller-supplied separator can neither create stray subdirectories nor escape
    the workspace dir via traversal (`../..`). The `{job_id}_` prefix keeps it unique.
    """
    d = Path(WORKING_DIR) / workspace_id
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
            "ORDER BY is_primary DESC LIMIT 1",
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
        _jobs[job_id] = record
        _batches.setdefault(row["batch_id"], []).append(record)
        dest = Path(WORKING_DIR) / physical / f"{job_id}_{row['file']}"
        if pub_row is not None and dest.exists():
            description_text = _build_metadata(
                row["description"] or "", row["source_path"] or "", row["last_modified_time"] or ""
            )
            # Re-hand LightRAG the same identity (lightrag_input), NOT the real display file_path.
            lightrag_input = f"{job_id}_{_safe_ref_name(row['file'])}"
            await _db_update_status(pool, job_id, "pending", row["attempts"], None)
            await _job_queue.put((pub_row["id"], job_id, dest, description_text, lightrag_input))
        else:
            record["status"] = "failed"
            record["error"] = "File missing after restart"
            await _db_update_status(
                pool, job_id, "failed", row["attempts"], "File missing after restart"
            )


# --- Background worker ---


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
    job = _jobs[job_id]
    job["status"] = "processing"
    if _db_pool:
        await _db_update_status(_db_pool, job_id, "processing", job["attempts"], None)
    try:
        rag_instance = await get_workspace_rag(workspace_id)
        lock = await _get_ws_lock(workspace_id)
        async with lock:
            result = await _process_file(
                dest, rag_instance, description_text=description_text, file_path=file_path
            )
        # `_process_file` returns (doc_id, lightrag_key); tolerate a bare doc_id from older mocks.
        doc_id, lightrag_key = result if isinstance(result, tuple) else (result, None)
        job["status"] = "done"
        job["doc_id"] = doc_id
        if _db_pool:
            await _db_update_status(_db_pool, job_id, "done", job["attempts"], None)
            if doc_id:
                await _db_set_doc_id(_db_pool, job_id, doc_id)
            if lightrag_key:
                await _db_set_lightrag_key(_db_pool, job_id, lightrag_key)
        # The DB index is the system of record; the raw bytes are redundant once ingested
        # (the source lives in the workspace), so drop them on success. Kept on retry/failure.
        dest.unlink(missing_ok=True)
    except Exception as exc:
        job["attempts"] += 1
        job["error"] = str(exc)
        if job["attempts"] < MAX_RETRIES:
            job["status"] = "retrying"
            if _db_pool:
                await _db_update_status(_db_pool, job_id, "retrying", job["attempts"], str(exc))
            await _job_queue.put((workspace_id, job_id, dest, description_text, file_path))
        else:
            job["status"] = "failed"
            if _db_pool:
                await _db_update_status(_db_pool, job_id, "failed", job["attempts"], str(exc))
            dest.unlink(missing_ok=True)


async def _worker():
    while True:
        workspace_id, job_id, dest, description_text, file_path = await _job_queue.get()
        try:
            await _process_job(workspace_id, job_id, dest, description_text, file_path)
        finally:
            _job_queue.task_done()


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


# --- Helpers ---


def _batch_summary(entries: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    counts["total"] = len(entries)
    return counts


def _batch_response(batch_id: str, entries: list[dict]) -> dict:
    return {"batch_id": batch_id, "summary": _batch_summary(entries), "jobs": entries}


def _internal_error(exc: Exception, context: str) -> HTTPException:
    """Log the real error server-side (with traceback) and return a client-safe generic 500.
    Keeps internal exception text — which can reveal implementation/query details — out of the
    HTTP response. Use as `raise _internal_error(exc, "query")`."""
    logging.exception("%s failed: %s", context, exc)
    return HTTPException(500, "Internal server error")


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

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


def _is_valid_slug(slug: str) -> bool:
    return bool(_SLUG_RE.match(slug or ""))


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


async def require_workspace(workspace_id: str) -> dict:
    """Path dependency: validate the slug and confirm the workspace is active.
    Returns the registry row (with `id` = public id and `lightrag_workspace` = physical
    workspace). 404 if the slug is malformed, unknown, or soft-deleted."""
    if not _is_valid_slug(workspace_id):
        raise HTTPException(404, f"Workspace {workspace_id!r} not found")
    row = await _lookup_workspace(workspace_id)
    if row is None:
        raise HTTPException(404, f"Workspace {workspace_id!r} not found")
    return row


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
