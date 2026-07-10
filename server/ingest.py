"""Document parsing and ingestion into LightRAG.

Turns an uploaded file (PDF/image/office/audio/text) into text, inserts it into LightRAG under
the "extract" LLM phase, and verifies the ingestion actually completed (chunks AND graph) before
reporting success. Endpoint config is read live from server.config; the shared asyncpg pool is
read from the package root (server._db_pool) at call time.
"""

import asyncio
import logging
from pathlib import Path

from raganything import RAGAnything

import server
from server import config


class IngestionIncompleteError(RuntimeError):
    """Raised when LightRAG stored chunks but did not fully ingest a document (e.g. entity
    extraction failed/timed out). Drives the normal retry/backoff path in _process_job."""


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

    client = openai.AsyncOpenAI(api_key=config.VISION_API_KEY, base_url=config.VISION_BASE_URL)
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
        model=config.VISION_MODEL,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": _EXTRACTION_PROMPT}, file_part]}
        ],
        **config._llm_call_kwargs(
            {"max_completion_tokens": 16000}, is_openai=config._VISION_IS_OPENAI
        ),
    )
    return resp.choices[0].message.content


async def _transcribe_audio(path: Path) -> str:
    import openai

    client = openai.AsyncOpenAI(api_key=config.WHISPER_API_KEY, base_url=config.WHISPER_BASE_URL)
    with path.open("rb") as f:
        transcript = await client.audio.transcriptions.create(
            model=config.WHISPER_MODEL, file=f, response_format="text"
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
    if server._db_pool is None:
        return 1
    workspace = getattr(rag_instance.lightrag, "workspace", None) or config.POSTGRES_WORKSPACE
    row = await server._db_pool.fetchrow(
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
    if config.RAG_REQUIRE_GRAPH_EXTRACTION:
        content_length = _doc_status_field(status_doc, "content_length", None) or len(content)
        if (
            content_length >= config.RAG_MIN_CONTENT_FOR_ENTITIES
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
    token = config._llm_phase.set("extract")
    logging.info(
        "extraction phase: %s will extract entities for %s (base_url=%s)",
        config.LLM_MODEL,
        file_path or path.name,
        config.LLM_BASE_URL or "openai",
    )
    try:
        return await _process_file_impl(path, rag_instance, description_text, file_path)
    finally:
        config._llm_phase.reset(token)


async def _process_file_impl(
    path: Path,
    rag_instance: RAGAnything,
    description_text: str = "",
    file_path: str | None = None,
) -> tuple[str | None, str | None]:
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
