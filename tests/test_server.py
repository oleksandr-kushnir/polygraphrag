"""
Tests for server.py — run inside the container:
    pip install pytest pytest-asyncio httpx
    pytest test_server.py -v
"""

import asyncio
import io
import logging

# --------------------------------------------------------------------------- #
# Minimal stubs so server.py can be imported without real Postgres / OpenAI
# --------------------------------------------------------------------------- #
# Stub RAGAnything before the real import happens
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

_DEFAULT_AQUERY_LLM = {
    "status": "success",
    "data": {"references": [], "chunks": [], "entities": [], "relationships": []},
    "llm_response": {"content": "mocked answer", "is_streaming": False},
    "metadata": {},
}

_DEFAULT_AQUERY_DATA = {
    "status": "success",
    "message": "Query completed successfully",
    "data": {"entities": [], "relationships": [], "chunks": [], "references": []},
    "metadata": {},
}

from types import SimpleNamespace


class _AnyKeyDocs(dict):
    """Stand-in for aget_docs_by_ids' return: yields a processed doc-status for any doc id
    (the real doc id is computed from content, which is opaque to tests). LightRAG's PG storage
    returns each doc-status as a plain dict, which is what we mirror here."""

    def __init__(self, doc=None):
        super().__init__()
        self._doc = (
            doc
            if doc is not None
            else {"status": "processed", "content_length": 11, "metadata": {}}
        )

    def get(self, key, default=None):
        return self._doc


rag_stub = MagicMock()
rag_stub.process_document_complete = AsyncMock()
rag_stub.lightrag = MagicMock()
rag_stub.lightrag.ainsert = AsyncMock()
rag_stub.lightrag.aquery = AsyncMock(return_value="mocked answer")
rag_stub.lightrag.aquery_llm = AsyncMock(return_value=_DEFAULT_AQUERY_LLM)
rag_stub.lightrag.aquery_data = AsyncMock(return_value=_DEFAULT_AQUERY_DATA)
# Ingestion-integrity verification (_verify_ingestion) — default: doc processed, small content.
rag_stub.lightrag.aget_docs_by_ids = AsyncMock(return_value=_AnyKeyDocs())
rag_stub.lightrag.adelete_by_doc_id = AsyncMock()

raganything_mod = MagicMock()
raganything_mod.RAGAnything = MagicMock(return_value=rag_stub)
sys.modules.setdefault("raganything", raganything_mod)

# Stub lightrag
lightrag_mod = MagicMock()
lightrag_instance_stub = MagicMock()
lightrag_instance_stub.initialize_storages = AsyncMock()
lightrag_instance_stub.finalize_storages = AsyncMock()
lightrag_instance_stub.ainsert = AsyncMock()
lightrag_instance_stub.aquery = AsyncMock(return_value="mocked answer")
lightrag_instance_stub.aquery_llm = AsyncMock(return_value=_DEFAULT_AQUERY_LLM)
lightrag_mod.LightRAG = MagicMock(return_value=lightrag_instance_stub)
lightrag_mod.QueryParam = MagicMock(return_value=MagicMock())
lightrag_utils_mod = MagicMock()
lightrag_utils_mod.EmbeddingFunc = MagicMock()
lightrag_mod.utils = lightrag_utils_mod
sys.modules.setdefault("lightrag", lightrag_mod)
sys.modules.setdefault("lightrag.utils", lightrag_utils_mod)

import server  # noqa: E402 — must come after stubs

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_mock_pool = MagicMock()
_mock_pool.execute = AsyncMock()
_mock_pool.fetch = AsyncMock(return_value=[])
_mock_pool.fetchrow = AsyncMock(return_value=None)
_mock_pool.close = AsyncMock()


@pytest.fixture(autouse=True)
def reset_server_state():
    """Clear in-memory job/batch dicts and drain the queue between tests."""
    server._jobs.clear()
    server._batches.clear()
    # Drain queue
    while not server._job_queue.empty():
        try:
            server._job_queue.get_nowait()
            server._job_queue.task_done()
        except asyncio.QueueEmpty:
            break
    # No DB pool by default — tests that need it set it explicitly
    server._db_pool = None
    # Clear the per-workspace instance registry + locks
    if hasattr(server, "_rag_instances"):
        server._rag_instances.clear()
    if hasattr(server, "_ws_locks"):
        server._ws_locks.clear()
    # Reset rag stub
    rag_stub.lightrag.ainsert.reset_mock()
    rag_stub.lightrag.aquery.reset_mock()
    rag_stub.lightrag.aquery_llm.reset_mock()
    rag_stub.lightrag.aquery_llm.return_value = _DEFAULT_AQUERY_LLM
    rag_stub.lightrag.aquery_data.reset_mock()
    rag_stub.lightrag.aquery_data.return_value = _DEFAULT_AQUERY_DATA
    rag_stub.lightrag.aget_docs_by_ids.reset_mock()
    rag_stub.lightrag.aget_docs_by_ids.return_value = _AnyKeyDocs()
    rag_stub.lightrag.adelete_by_doc_id.reset_mock()
    rag_stub.process_document_complete.reset_mock()
    # Auth is opt-in; ensure no test leaks an enforced token list into the next.
    server.API_TOKENS = []
    yield
    server._db_pool = None
    server.API_TOKENS = []


# Default public workspace used by data-endpoint tests (maps to physical 'default').
WS = "/workspace/alex"


async def _fake_lookup_workspace(workspace_id):
    """Stub: every workspace looks active and present, mapping id→id physically."""
    return {
        "id": workspace_id,
        "name": workspace_id,
        "description": None,
        "lightrag_workspace": workspace_id,
        "is_primary": workspace_id == "alex",
    }


@pytest_asyncio.fixture
async def client():
    """Async httpx test client that bypasses lifespan. Stubs the workspace registry so
    `/workspace/{id}/...` routes resolve and `get_workspace_rag` returns the shared stub."""
    from httpx import ASGITransport, AsyncClient

    orig_lookup = server.workspaces._lookup_workspace
    orig_get = server.get_workspace_rag
    server.workspaces._lookup_workspace = _fake_lookup_workspace
    server.get_workspace_rag = AsyncMock(return_value=rag_stub)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as c:
            yield c
    finally:
        server.workspaces._lookup_workspace = orig_lookup
        server.get_workspace_rag = orig_get


def _fake_upload(filename: str, content: bytes = b"hello") -> dict:
    return ("files", (filename, io.BytesIO(content), "application/octet-stream"))


# --------------------------------------------------------------------------- #
# Unit — routing (_process_file dispatch)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_routes_pdf_to_vision(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 test")
    with patch.object(server.ingest, "_extract_with_vision", new=AsyncMock(return_value="text")) as m:
        await server._process_file(path, rag_stub)
    m.assert_awaited_once_with(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("ext", [".jpg", ".png", ".webp", ".jpeg", ".gif", ".bmp", ".tiff"])
async def test_routes_images_to_vision(tmp_path, ext):
    path = tmp_path / f"img{ext}"
    path.write_bytes(b"\x89PNG")
    with patch.object(server.ingest, "_extract_with_vision", new=AsyncMock(return_value="text")) as m:
        await server._process_file(path, rag_stub)
    m.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("ext", [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".webm"])
async def test_routes_audio_to_whisper(tmp_path, ext):
    path = tmp_path / f"audio{ext}"
    path.write_bytes(b"RIFF")
    with patch.object(server.ingest, "_transcribe_audio", new=AsyncMock(return_value="transcript")) as m:
        await server._process_file(path, rag_stub)
    m.assert_awaited_once_with(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("ext", [".docx", ".xlsx", ".pptx"])
async def test_routes_office_converts_then_vision(tmp_path, ext):
    path = tmp_path / f"office{ext}"
    path.write_bytes(b"PK\x03\x04")
    pdf_dir = Path(tempfile.mkdtemp())
    fake_pdf = pdf_dir / "office.pdf"
    fake_pdf.write_bytes(b"%PDF")
    with (
        patch.object(server.ingest, "_convert_office_to_pdf", new=AsyncMock(return_value=fake_pdf)),
        patch.object(server.ingest, "_extract_with_vision", new=AsyncMock(return_value="text")) as mv,
    ):
        await server._process_file(path, rag_stub)
    mv.assert_awaited_once_with(fake_pdf)


@pytest.mark.asyncio
async def test_office_temp_pdf_deleted_on_success(tmp_path):
    path = tmp_path / "slides.pptx"
    path.write_bytes(b"PK\x03\x04")
    pdf_dir = Path(tempfile.mkdtemp())
    fake_pdf = pdf_dir / "slides.pdf"
    fake_pdf.write_bytes(b"%PDF")
    with (
        patch.object(server.ingest, "_convert_office_to_pdf", new=AsyncMock(return_value=fake_pdf)),
        patch.object(server.ingest, "_extract_with_vision", new=AsyncMock(return_value="text")),
    ):
        await server._process_file(path, rag_stub)
    assert not fake_pdf.exists()
    assert not pdf_dir.exists()


@pytest.mark.asyncio
async def test_office_temp_pdf_deleted_on_vision_error(tmp_path):
    path = tmp_path / "slides.pptx"
    path.write_bytes(b"PK\x03\x04")
    pdf_dir = Path(tempfile.mkdtemp())
    fake_pdf = pdf_dir / "slides.pdf"
    fake_pdf.write_bytes(b"%PDF")
    with (
        patch.object(server.ingest, "_convert_office_to_pdf", new=AsyncMock(return_value=fake_pdf)),
        patch.object(
            server.ingest, "_extract_with_vision", new=AsyncMock(side_effect=RuntimeError("vision fail"))
        ),
    ):
        with pytest.raises(RuntimeError):
            await server._process_file(path, rag_stub)
    assert not fake_pdf.exists()
    assert not pdf_dir.exists()


@pytest.mark.asyncio
async def test_xlsx_over_10mb_raises(tmp_path):
    path = tmp_path / "big.xlsx"
    path.write_bytes(b"x" * (11 * 1024 * 1024))
    with pytest.raises(ValueError, match="10 MB"):
        await server._process_file(path, rag_stub)


@pytest.mark.asyncio
async def test_xlsx_under_10mb_proceeds(tmp_path):
    path = tmp_path / "small.xlsx"
    path.write_bytes(b"PK\x03\x04")
    pdf_dir = Path(tempfile.mkdtemp())
    fake_pdf = pdf_dir / "small.pdf"
    fake_pdf.write_bytes(b"%PDF")
    with (
        patch.object(server.ingest, "_convert_office_to_pdf", new=AsyncMock(return_value=fake_pdf)),
        patch.object(server.ingest, "_extract_with_vision", new=AsyncMock(return_value="text")) as mv,
    ):
        await server._process_file(path, rag_stub)
    mv.assert_awaited_once()


@pytest.mark.asyncio
async def test_xlsx_size_limit_only_applies_to_xlsx(tmp_path):
    path = tmp_path / "big.docx"
    path.write_bytes(b"x" * (11 * 1024 * 1024))
    pdf_dir = Path(tempfile.mkdtemp())
    fake_pdf = pdf_dir / "big.pdf"
    fake_pdf.write_bytes(b"%PDF")
    with (
        patch.object(server.ingest, "_convert_office_to_pdf", new=AsyncMock(return_value=fake_pdf)),
        patch.object(server.ingest, "_extract_with_vision", new=AsyncMock(return_value="text")) as mv,
    ):
        await server._process_file(path, rag_stub)
    mv.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("ext", [".txt", ".md", ".html", ".csv"])
async def test_routes_text_to_direct_read(tmp_path, ext):
    path = tmp_path / f"file{ext}"
    path.write_text("hello world", encoding="utf-8")
    await server._process_file(path, rag_stub)
    rag_stub.lightrag.ainsert.assert_awaited_once()
    args = rag_stub.lightrag.ainsert.call_args[0][0]
    assert "hello world" in args


@pytest.mark.asyncio
async def test_routes_unknown_extension_to_fallback(tmp_path):
    path = tmp_path / "file.xyz"
    path.write_bytes(b"data")
    await server._process_file(path, rag_stub)
    rag_stub.process_document_complete.assert_awaited_once()


# --------------------------------------------------------------------------- #
# Unit — file storage
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_saved_path_uses_job_id_prefix(tmp_path, client):
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        resp = await client.post(f"{WS}/upload/batch", files=[_fake_upload("report.pdf")])
    data = resp.json()
    job_id = data["jobs"][0]["job_id"]
    expected = tmp_path / "alex" / f"{job_id}_report.pdf"
    assert expected.exists()


@pytest.mark.asyncio
async def test_same_filename_no_collision(tmp_path, client):
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        r1 = await client.post(f"{WS}/upload/batch", files=[_fake_upload("report.pdf", b"v1")])
        r2 = await client.post(f"{WS}/upload/batch", files=[_fake_upload("report.pdf", b"v2")])
    j1 = r1.json()["jobs"][0]["job_id"]
    j2 = r2.json()["jobs"][0]["job_id"]
    assert j1 != j2
    assert (tmp_path / "alex" / f"{j1}_report.pdf").exists()
    assert (tmp_path / "alex" / f"{j2}_report.pdf").exists()


@pytest.mark.asyncio
async def test_failed_permanently_file_deleted(tmp_path):
    path = tmp_path / "aaa_file.txt"
    path.write_text("content")
    job_id = "aaa"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "file.txt",
        "workspace": "alex",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }

    original_max = server.config.MAX_RETRIES
    server.config.MAX_RETRIES = 1
    with (
        patch.object(server, "_process_file", new=AsyncMock(side_effect=RuntimeError("fail"))),
        patch.object(server, "get_workspace_rag", new=AsyncMock(return_value=rag_stub)),
    ):
        await server._process_job("alex", job_id, path, "")

    server.config.MAX_RETRIES = original_max
    assert not path.exists()
    assert server._jobs[job_id]["status"] == "failed"


@pytest.mark.asyncio
async def test_successful_job_file_deleted(tmp_path):
    # A2: the DB index is the system of record; raw bytes are dropped on successful ingest.
    path = tmp_path / "aaa_file.txt"
    path.write_text("content")
    job_id = "aaa_ok"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "file.txt",
        "workspace": "alex",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }
    with (
        patch.object(server, "_process_file", new=AsyncMock(return_value="doc-x")),
        patch.object(server, "get_workspace_rag", new=AsyncMock(return_value=rag_stub)),
    ):
        await server._process_job("alex", job_id, path, "")
    assert not path.exists()
    assert server._jobs[job_id]["status"] == "done"


# --------------------------------------------------------------------------- #
# Unit — retry logic
# --------------------------------------------------------------------------- #


async def _run_worker_until_terminal(
    job_id: str, dest: Path, metadata: str, process_mock: AsyncMock
):
    """Drive the worker loop manually until the job reaches a terminal state, using the
    real _process_job (with _process_file + get_workspace_rag stubbed)."""
    with (
        patch.object(server, "_process_file", process_mock),
        patch.object(server, "get_workspace_rag", AsyncMock(return_value=rag_stub)),
    ):
        while server._jobs[job_id]["status"] not in ("done", "failed"):
            if server._job_queue.empty():
                break
            ws, j, d, m, fp = await server._job_queue.get()
            try:
                await server._process_job(ws, j, d, m, fp)
            finally:
                server._job_queue.task_done()


@pytest.mark.asyncio
async def test_retry_on_transient_error(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("x")
    job_id = "retry1"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "file.txt",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }
    await server._job_queue.put(("alex", job_id, path, "", None))

    call_count = 0

    async def _fail_once(p, rag_instance=None, description_text="", file_path=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient")

    await _run_worker_until_terminal(job_id, path, "", AsyncMock(side_effect=_fail_once))
    assert server._jobs[job_id]["status"] == "done"
    assert server._jobs[job_id]["attempts"] == 1


@pytest.mark.asyncio
async def test_retry_up_to_max_retries_5(tmp_path):
    path = tmp_path / "fail.txt"
    path.write_text("x")
    job_id = "always_fail"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "fail.txt",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }
    await server._job_queue.put(("alex", job_id, path, "", None))
    original = server.config.MAX_RETRIES
    server.config.MAX_RETRIES = 5
    mock = AsyncMock(side_effect=RuntimeError("permanent"))
    await _run_worker_until_terminal(job_id, path, "", mock)
    server.config.MAX_RETRIES = original
    assert server._jobs[job_id]["status"] == "failed"
    assert server._jobs[job_id]["attempts"] == 5
    assert not path.exists()


@pytest.mark.asyncio
async def test_success_on_second_attempt(tmp_path):
    path = tmp_path / "retry2.txt"
    path.write_text("x")
    job_id = "retry2"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "retry2.txt",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }
    await server._job_queue.put(("alex", job_id, path, "", None))

    call_count = 0

    async def _fail_once(p, rag_instance=None, description_text="", file_path=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("first fail")

    await _run_worker_until_terminal(job_id, path, "", AsyncMock(side_effect=_fail_once))
    assert server._jobs[job_id]["status"] == "done"
    assert server._jobs[job_id]["attempts"] == 1


@pytest.mark.asyncio
async def test_max_retries_env_var_override(tmp_path, monkeypatch):
    path = tmp_path / "env_retry.txt"
    path.write_text("x")
    job_id = "env_fail"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "env_retry.txt",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }
    await server._job_queue.put(("alex", job_id, path, "", None))
    original = server.config.MAX_RETRIES
    server.config.MAX_RETRIES = 2
    mock = AsyncMock(side_effect=RuntimeError("perm"))
    await _run_worker_until_terminal(job_id, path, "", mock)
    server.config.MAX_RETRIES = original
    assert server._jobs[job_id]["status"] == "failed"
    assert server._jobs[job_id]["attempts"] == 2


@pytest.mark.asyncio
async def test_error_field_reflects_last_attempt(tmp_path):
    path = tmp_path / "errmsg.txt"
    path.write_text("x")
    job_id = "errmsg"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "errmsg.txt",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }
    await server._job_queue.put(("alex", job_id, path, "", None))
    original = server.config.MAX_RETRIES
    server.config.MAX_RETRIES = 2

    attempt = 0

    async def _variable_error(p, rag_instance=None, description_text="", file_path=None):
        nonlocal attempt
        attempt += 1
        raise RuntimeError(f"error attempt {attempt}")

    await _run_worker_until_terminal(job_id, path, "", AsyncMock(side_effect=_variable_error))
    server.config.MAX_RETRIES = original
    assert "attempt 2" in server._jobs[job_id]["error"]


# --------------------------------------------------------------------------- #
# Unit — state machine helpers
# --------------------------------------------------------------------------- #


def test_build_metadata_returns_description_only():
    result = server._build_metadata("My desc", "/path/to/file", "2026-01-01")
    assert result == "My desc"


def test_build_metadata_ignores_source_and_timestamp():
    # source_path and last_modified are persisted in rag_file_metadata but
    # must not be injected into chunk text.
    result = server._build_metadata("", "/path", "2026-01-01")
    assert result == ""


def test_build_metadata_empty_when_no_description():
    assert server._build_metadata("", "", "") == ""


def test_batch_summary_counts():
    entries = [
        {"status": "done"},
        {"status": "done"},
        {"status": "failed"},
        {"status": "save_failed"},
    ]
    summary = server.deps._batch_summary(entries)
    assert summary["done"] == 2
    assert summary["failed"] == 1
    assert summary["save_failed"] == 1
    assert summary["total"] == 4


# --------------------------------------------------------------------------- #
# Integration — HTTP endpoints
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_health_unchanged(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_query_unchanged(client):
    resp = await client.post(f"{WS}/query", json={"query": "test", "mode": "hybrid"})
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    assert "references" in data


@pytest.mark.asyncio
async def test_batch_single_file_returns_batch_id(tmp_path, client):
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        resp = await client.post(f"{WS}/upload/batch", files=[_fake_upload("test.pdf")])
    assert resp.status_code == 200
    data = resp.json()
    assert "batch_id" in data
    assert len(data["jobs"]) == 1
    assert data["summary"]["total"] == 1


@pytest.mark.asyncio
async def test_batch_20_files_all_enqueued(tmp_path, client):
    files = [_fake_upload(f"file{i}.txt", b"x") for i in range(20)]
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        resp = await client.post(f"{WS}/upload/batch", files=files)
    data = resp.json()
    assert len(data["jobs"]) == 20
    assert data["summary"]["total"] == 20
    pending = sum(1 for j in data["jobs"] if j["status"] == "pending")
    assert pending == 20


@pytest.mark.asyncio
async def test_batch_partial_save_failure_no_abort(tmp_path, client):
    call_count = 0

    async def _fail_third_read():
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise OSError("disk full")
        return b"content"

    files = [_fake_upload(f"f{i}.txt") for i in range(5)]

    # Simulate a mid-batch failure by patching Path.open to fail on the 3rd file write
    open_call = 0
    real_open = Path.open

    def _counting_open(self, mode="r", **kw):
        nonlocal open_call
        if "wb" in mode:
            open_call += 1
            if open_call == 3:
                raise OSError("disk full")
        return real_open(self, mode, **kw)

    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
        patch.object(Path, "open", _counting_open),
    ):
        resp = await client.post(f"{WS}/upload/batch", files=files)
    data = resp.json()
    assert resp.status_code == 200
    assert data["summary"]["total"] == 5
    statuses = [j["status"] for j in data["jobs"]]
    assert statuses.count("save_failed") == 1
    assert statuses.count("pending") == 4


@pytest.mark.asyncio
async def test_batch_response_has_summary_counts(tmp_path, client):
    files = [_fake_upload(f"f{i}.txt") for i in range(3)]
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        resp = await client.post(f"{WS}/upload/batch", files=files)
    s = resp.json()["summary"]
    assert s["total"] == 3
    assert s.get("pending", 0) == 3


@pytest.mark.asyncio
async def test_get_batch_404_unknown(client):
    resp = await client.get(f"{WS}/batch/doesnotexist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_status_pending_immediate(tmp_path, client):
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        resp = await client.post(f"{WS}/upload/batch", files=[_fake_upload("f.txt")])
    job_id = resp.json()["jobs"][0]["job_id"]
    sr = await client.get(f"{WS}/status/{job_id}")
    assert sr.status_code == 200
    assert sr.json()["status"] == "pending"


@pytest.mark.asyncio
async def test_status_404_unknown_job(client):
    resp = await client.get(f"{WS}/status/notexist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_jobs_newest_first(tmp_path, client):
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        r1 = await client.post(f"{WS}/upload/batch", files=[_fake_upload("first.txt")])
        r2 = await client.post(f"{WS}/upload/batch", files=[_fake_upload("second.txt")])
    jid1 = r1.json()["jobs"][0]["job_id"]
    jid2 = r2.json()["jobs"][0]["job_id"]
    resp = await client.get(f"{WS}/jobs")
    ids = [j["job_id"] for j in resp.json()["jobs"]]
    assert ids.index(jid2) < ids.index(jid1)


@pytest.mark.asyncio
async def test_jobs_capped_at_100(tmp_path, client):
    files = [_fake_upload(f"f{i}.txt") for i in range(110)]
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        await client.post(f"{WS}/upload/batch", files=files)
    resp = await client.get(f"{WS}/jobs")
    assert len(resp.json()["jobs"]) == 100


# --------------------------------------------------------------------------- #
# Integration — worker drives jobs to terminal states
# --------------------------------------------------------------------------- #


async def _drain_queue(process_mock: AsyncMock):
    """Process all queued jobs via the real _process_job (with _process_file +
    get_workspace_rag stubbed)."""
    with (
        patch.object(server, "_process_file", process_mock),
        patch.object(server, "get_workspace_rag", AsyncMock(return_value=rag_stub)),
    ):
        while not server._job_queue.empty():
            ws, job_id, dest, metadata, fp = await server._job_queue.get()
            try:
                await server._process_job(ws, job_id, dest, metadata, fp)
            finally:
                server._job_queue.task_done()


@pytest.mark.asyncio
async def test_status_done_after_drain(tmp_path, client):
    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        resp = await client.post(f"{WS}/upload/batch", files=[_fake_upload("f.txt")])
    job_id = resp.json()["jobs"][0]["job_id"]
    await _drain_queue(AsyncMock(return_value=None))
    sr = await client.get(f"{WS}/status/{job_id}")
    assert sr.json()["status"] == "done"


@pytest.mark.asyncio
async def test_status_failed_after_max_retries(tmp_path, client):
    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        resp = await client.post(f"{WS}/upload/batch", files=[_fake_upload("bad.txt")])
    job_id = resp.json()["jobs"][0]["job_id"]
    original = server.config.MAX_RETRIES
    server.config.MAX_RETRIES = 2
    await _drain_queue(AsyncMock(side_effect=RuntimeError("always fails")))
    server.config.MAX_RETRIES = original
    sr = await client.get(f"{WS}/status/{job_id}")
    data = sr.json()
    assert data["status"] == "failed"
    assert "always fails" in data["error"]


@pytest.mark.asyncio
async def test_get_batch_live_status(tmp_path, client):
    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        resp = await client.post(f"{WS}/upload/batch", files=[_fake_upload("doc.txt")])
    batch_id = resp.json()["batch_id"]
    # Before drain — pending
    br = await client.get(f"{WS}/batch/{batch_id}")
    assert br.json()["summary"].get("pending", 0) == 1
    # After drain — done
    await _drain_queue(AsyncMock(return_value=None))
    br2 = await client.get(f"{WS}/batch/{batch_id}")
    assert br2.json()["summary"].get("done", 0) == 1


@pytest.mark.asyncio
async def test_status_retrying_intermediate(tmp_path, client):
    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        resp = await client.post(f"{WS}/upload/batch", files=[_fake_upload("retry.txt")])
    job_id = resp.json()["jobs"][0]["job_id"]

    original = server.config.MAX_RETRIES
    server.config.MAX_RETRIES = 3
    call_count = 0

    async def _fail_first(p, rag_instance=None, description_text="", file_path=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient")

    # Process just one item — leaves it as retrying
    ws, j, d, m, fp = await server._job_queue.get()
    with (
        patch.object(server, "_process_file", AsyncMock(side_effect=_fail_first)),
        patch.object(server, "get_workspace_rag", AsyncMock(return_value=rag_stub)),
    ):
        try:
            await server._process_job(ws, j, d, m, fp)
        finally:
            server._job_queue.task_done()

    sr = await client.get(f"{WS}/status/{job_id}")
    assert sr.json()["status"] == "retrying"

    # Finish it
    await _drain_queue(AsyncMock(return_value=None))
    server.config.MAX_RETRIES = original
    sr2 = await client.get(f"{WS}/status/{job_id}")
    assert sr2.json()["status"] == "done"


# --------------------------------------------------------------------------- #
# Concurrency / stress
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_10_concurrent_batches_all_done(tmp_path, client):
    async def _upload_one(i):
        with patch.object(server, "WORKING_DIR", str(tmp_path)):
            return await client.post(f"{WS}/upload/batch", files=[_fake_upload(f"f{i}.txt")])

    resps = await asyncio.gather(*[_upload_one(i) for i in range(10)])
    job_ids = [r.json()["jobs"][0]["job_id"] for r in resps]

    await _drain_queue(AsyncMock(return_value=None))

    for jid in job_ids:
        assert server._jobs[jid]["status"] == "done"


@pytest.mark.asyncio
async def test_ainsert_called_exactly_once_per_file(tmp_path, client):
    n = 5
    files = [_fake_upload(f"f{i}.txt", f"content {i}".encode()) for i in range(n)]
    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        await client.post(f"{WS}/upload/batch", files=files)

    rag_stub.lightrag.ainsert.reset_mock()

    async def _side_effect(p, rag_instance=None, description_text="", file_path=None):
        await rag_stub.lightrag.ainsert(f"content from {p.name}", file_paths=[p.name])

    await _drain_queue(AsyncMock(side_effect=_side_effect))
    assert rag_stub.lightrag.ainsert.await_count == n


@pytest.mark.asyncio
async def test_lock_prevents_overlap(tmp_path, client):
    """Verify _insert_lock serialises calls — no two _process_file calls overlap."""
    active = 0
    max_active = 0

    async def _track_overlap(p, rag_instance=None, description_text="", file_path=None):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)  # yield
        active -= 1

    files = [_fake_upload(f"f{i}.txt") for i in range(5)]
    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        await client.post(f"{WS}/upload/batch", files=files)

    await _drain_queue(AsyncMock(side_effect=_track_overlap))
    assert max_active == 1


# --------------------------------------------------------------------------- #
# Query — structured references
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_query_references_returned_by_default(client):
    resp = await client.post(f"{WS}/query", json={"query": "test"})
    assert resp.status_code == 200
    assert "references" in resp.json()
    assert isinstance(resp.json()["references"], list)


@pytest.mark.asyncio
async def test_query_references_parsed_from_aquery_llm(client):
    server._db_pool = _mock_pool
    _mock_pool.fetch.reset_mock()
    _mock_pool.fetch.return_value = []  # no metadata rows -> basename fallback, null metadata
    rag_stub.lightrag.aquery_llm.return_value = {
        "status": "success",
        "data": {
            "references": [
                {"reference_id": "1", "file_path": "/opt/data/workspace/Tag_1.pdf"},
                {"reference_id": "2", "file_path": "/opt/data/workspace/sub/Tag_2.pdf"},
            ]
        },
        "llm_response": {"content": "answer text", "is_streaming": False},
        "metadata": {},
    }
    resp = await client.post(f"{WS}/query", json={"query": "KI-Strategie", "mode": "hybrid"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"] == "answer text"
    refs = data["references"]
    assert len(refs) == 2
    # Unresolvable references (no metadata row): file_path is null — we NEVER echo
    # LightRAG's raw internal value. reference_id + answering model are still present.
    assert refs[0]["reference_id"] == "1"
    assert refs[0]["file_path"] is None
    assert "file_name" not in refs[0]  # file_name is no longer emitted at all
    assert refs[0]["file_description"] is None and refs[0]["job_id"] is None
    assert "source_path" not in refs[0]
    assert refs[0]["llm_model_extracted"] is None  # no DB row -> unknown extractor
    assert (
        refs[0]["llm_model_answered"] == server.config.QUERY_LLM_MODEL
    )  # /query reports the query-time answering model
    # The internal LightRAG key must not appear anywhere in the reference payload.
    assert "lightrag_key" not in refs[0]
    assert "Tag_1.pdf" not in str(refs[0])


@pytest.mark.asyncio
async def test_query_references_empty_when_none(client):
    rag_stub.lightrag.aquery_llm.return_value = {
        "status": "success",
        "data": {"references": []},
        "llm_response": {"content": "no refs", "is_streaming": False},
        "metadata": {},
    }
    resp = await client.post(f"{WS}/query", json={"query": "test"})
    assert resp.json()["references"] == []


@pytest.mark.asyncio
async def test_query_include_references_false(client):
    rag_stub.lightrag.aquery_llm.return_value = {
        "status": "success",
        "data": {"references": [{"reference_id": "1", "file_path": "20ed9a7c_Tag_1.pdf"}]},
        "llm_response": {"content": "answer", "is_streaming": False},
        "metadata": {},
    }
    resp = await client.post(f"{WS}/query", json={"query": "test", "include_references": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["references"] == []
    query_param_call = lightrag_mod.QueryParam.call_args
    assert query_param_call.kwargs.get("include_references") is False


@pytest.mark.asyncio
async def test_query_references_unknown_format(client):
    rag_stub.lightrag.aquery_llm.return_value = {
        "status": "success",
        "data": {"references": [{"reference_id": "99", "file_path": "nodash.pdf"}]},
        "llm_response": {"content": "answer", "is_streaming": False},
        "metadata": {},
    }
    resp = await client.post(f"{WS}/query", json={"query": "test"})
    ref = resp.json()["references"][0]
    assert ref["job_id"] is None
    # No metadata row -> unresolved; never surface LightRAG's internal name.
    assert "file_name" not in ref  # file_name is no longer emitted at all
    assert ref["file_path"] is None


@pytest.mark.asyncio
async def test_query_result_field_is_llm_content(client):
    rag_stub.lightrag.aquery_llm.return_value = {
        "status": "success",
        "data": {"references": []},
        "llm_response": {"content": "specific answer string", "is_streaming": False},
        "metadata": {},
    }
    resp = await client.post(f"{WS}/query", json={"query": "anything"})
    assert resp.json()["result"] == "specific answer string"


# --------------------------------------------------------------------------- #
# DB persistence
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_upload_calls_db_insert(tmp_path, client):
    server._db_pool = _mock_pool
    _mock_pool.execute.reset_mock()
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        resp = await client.post(
            f"{WS}/upload/batch",
            files=[_fake_upload("report.pdf")],
            data={
                "metadata": '[{"description":"Q1 doc","source_path":"/docs/report.pdf","last_modified_time":"2026-04-01T10:00:00"}]'
            },
        )
    assert resp.status_code == 200
    # INSERT should have been called
    insert_calls = [str(c) for c in _mock_pool.execute.call_args_list]
    assert any("INSERT" in s for s in insert_calls)


@pytest.mark.asyncio
async def test_worker_updates_db_on_done(tmp_path):
    server._db_pool = _mock_pool
    _mock_pool.execute.reset_mock()
    path = tmp_path / "aaa_file.txt"
    path.write_text("content")
    job_id = "aaa"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "file.txt",
        "workspace": "alex",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }
    with (
        patch.object(server, "_process_file", new=AsyncMock()),
        patch.object(server, "get_workspace_rag", new=AsyncMock(return_value=rag_stub)),
    ):
        await server._process_job("alex", job_id, path, "")

    update_calls = [str(c) for c in _mock_pool.execute.call_args_list]
    done_calls = [c for c in update_calls if "done" in c]
    assert done_calls, "Expected a DB update with status='done'"


@pytest.mark.asyncio
async def test_worker_updates_db_on_failed(tmp_path):
    server._db_pool = _mock_pool
    _mock_pool.execute.reset_mock()
    path = tmp_path / "bbb_fail.txt"
    path.write_text("x")
    job_id = "bbb"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "fail.txt",
        "workspace": "alex",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }

    original = server.config.MAX_RETRIES
    server.config.MAX_RETRIES = 1
    with (
        patch.object(
            server, "_process_file", new=AsyncMock(side_effect=RuntimeError("permanent failure"))
        ),
        patch.object(server, "get_workspace_rag", new=AsyncMock(return_value=rag_stub)),
    ):
        await server._process_job("alex", job_id, path, "")
    server.config.MAX_RETRIES = original

    update_calls = [str(c) for c in _mock_pool.execute.call_args_list]
    failed_calls = [c for c in update_calls if "failed" in c]
    assert failed_calls, "Expected a DB update with status='failed'"


@pytest.mark.asyncio
async def test_status_falls_back_to_db(client):
    server._db_pool = _mock_pool
    _mock_pool.fetchrow.reset_mock()
    _mock_pool.fetchrow.return_value = {
        "job_id": "dbonly01",
        "batch_id": "batch1",
        "file": "remote.pdf",
        "status": "done",
        "attempts": 0,
        "error": None,
        "description": "remote doc",
        "source_path": "/remote.pdf",
        "last_modified_time": "2026-01-01T00:00:00",
        "uploaded_at": "2026-01-01T00:00:00",
    }
    resp = await client.get(f"{WS}/status/dbonly01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "dbonly01"
    assert data["status"] == "done"
    _mock_pool.fetchrow.assert_awaited_once()
    _mock_pool.fetchrow.return_value = None  # restore default


@pytest.mark.asyncio
async def test_query_references_enriched_from_db(client):
    server._db_pool = _mock_pool
    from datetime import datetime, timezone

    uploaded = datetime(2026, 5, 8, 12, 34, 56, tzinfo=timezone.utc)
    _mock_pool.fetch.reset_mock()
    _mock_pool.fetch.return_value = [
        {
            "lightrag_key": "20ed9a7c_Tag_1.pdf",  # what LightRAG returns; JOIN key
            "file_path": "/opt/data/workspace/Tag_1.pdf",  # real display path
            "file": "Tag_1.pdf",
            "job_id": "20ed9a7c",
            "description": "Q1 strategy",
            "source_path": "Tag_1.pdf",
            "last_modified_time": "2026-04-01T10:00:00",
            "uploaded_at": uploaded,
            "llm_model_extracted": "deepseek/deepseek-v4-flash",
        }
    ]
    rag_stub.lightrag.aquery_llm.return_value = {
        "status": "success",
        # LightRAG returns its internal citation key (the basename); we resolve it to the real path.
        "data": {"references": [{"reference_id": "1", "file_path": "20ed9a7c_Tag_1.pdf"}]},
        "llm_response": {"content": "answer", "is_streaming": False},
        "metadata": {},
    }
    resp = await client.post(f"{WS}/query", json={"query": "test"})
    assert resp.status_code == 200
    ref = resp.json()["references"][0]
    assert ref["file_path"] == "/opt/data/workspace/Tag_1.pdf"  # resolved real, openable path
    assert "file_name" not in ref  # file_name is no longer emitted at all
    assert ref["file_description"] == "Q1 strategy"
    assert "source_path" not in ref  # dropped from references
    assert ref["last_modified_time"] == "2026-04-01T10:00:00"
    assert ref["uploaded_at"] == "2026-05-08T12:34:56"
    assert ref["llm_model_extracted"] == "deepseek/deepseek-v4-flash"  # per-file extractor
    assert ref["llm_model_answered"] == server.config.QUERY_LLM_MODEL  # current query-time answering model
    _mock_pool.fetch.return_value = []  # restore default


@pytest.mark.asyncio
async def test_query_references_null_when_no_db_record(client):
    server._db_pool = _mock_pool
    _mock_pool.fetch.reset_mock()
    _mock_pool.fetch.return_value = []  # no rows for this job_id
    rag_stub.lightrag.aquery_llm.return_value = {
        "status": "success",
        "data": {"references": [{"reference_id": "1", "file_path": "20ed9a7c_Tag_1.pdf"}]},
        "llm_response": {"content": "answer", "is_streaming": False},
        "metadata": {},
    }
    resp = await client.post(f"{WS}/query", json={"query": "test"})
    assert resp.status_code == 200
    ref = resp.json()["references"][0]
    assert ref["file_description"] is None
    assert "source_path" not in ref
    assert ref["last_modified_time"] is None
    assert ref["uploaded_at"] is None
    assert ref["llm_model_extracted"] is None  # no DB row -> unknown extractor
    assert (
        ref["llm_model_answered"] == server.config.QUERY_LLM_MODEL
    )  # answering model present regardless of DB row


@pytest.mark.asyncio
async def test_query_answer_prose_rewrites_internal_keys(client):
    """LightRAG embeds its internal citation key in the answer's `### References` prose; we must
    rewrite it to the clean filename (resolved) or a prefix-stripped basename (unresolved) so no
    `{job_id}_` token / hex ever surfaces in the answer text — mirroring the structured refs."""
    server._db_pool = _mock_pool
    _mock_pool.fetch.reset_mock()
    _mock_pool.fetch.return_value = [
        {
            "lightrag_key": "20ed9a7c_Tag_1.pdf",
            "file_path": "/opt/data/workspace/reports/Tag_1.pdf",
            "file": "sub/Tag_1.pdf",  # original had a dir part; prose shows the basename
            "job_id": "20ed9a7c",
            "description": "Q1",
            "source_path": "sub/Tag_1.pdf",
            "last_modified_time": None,
            "uploaded_at": None,
            "llm_model_extracted": "m",
        }
    ]
    rag_stub.lightrag.aquery_llm.return_value = {
        "status": "success",
        "data": {
            "references": [
                {"reference_id": "1", "file_path": "20ed9a7c_Tag_1.pdf"},  # resolved
                {"reference_id": "2", "file_path": "beefcafe_notes.txt"},  # unresolved (no row)
            ]
        },
        "llm_response": {
            "content": "See sources.\n\n### References\n- [1] 20ed9a7c_Tag_1.pdf\n- [2] beefcafe_notes.txt",
            "is_streaming": False,
        },
        "metadata": {},
    }
    resp = await client.post(f"{WS}/query", json={"query": "test"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "20ed9a7c_Tag_1.pdf" not in result  # resolved internal key gone
    assert "beefcafe_notes.txt" not in result  # unresolved key's hex prefix gone
    assert "[1] Tag_1.pdf" in result  # resolved -> clean basename
    assert "[2] notes.txt" in result  # unresolved -> {job_id}_ prefix stripped
    _mock_pool.fetch.return_value = []


@pytest.mark.asyncio
async def test_query_top_k_forwarded(client):
    await client.post(f"{WS}/query", json={"query": "test", "top_k": 7})
    assert lightrag_mod.QueryParam.call_args.kwargs.get("top_k") == 7


@pytest.mark.asyncio
async def test_query_default_mode_is_mix(client):
    await client.post(f"{WS}/query", json={"query": "test"})
    assert lightrag_mod.QueryParam.call_args.kwargs.get("mode") == "mix"


@pytest.mark.asyncio
async def test_query_default_top_k_is_40(client):
    await client.post(f"{WS}/query", json={"query": "test"})
    assert lightrag_mod.QueryParam.call_args.kwargs.get("top_k") == 40


@pytest.mark.asyncio
async def test_query_data_default_top_k_is_40(client):
    await client.post(f"{WS}/query/data", json={"query": "test"})
    assert lightrag_mod.QueryParam.call_args.kwargs.get("top_k") == 40


# --------------------------------------------------------------------------- #
# /query/data — structured retrieval context (no LLM synthesis)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_query_data_passthrough(client):
    rag_stub.lightrag.aquery_data.return_value = {
        "status": "success",
        "message": "Query completed successfully",
        "data": {
            "entities": [{"entity_name": "KI", "entity_type": "concept"}],
            "relationships": [{"src_id": "KI", "tgt_id": "Strategie"}],
            "chunks": [{"content": "chunk text", "chunk_id": "c1"}],
            "references": [],
        },
        "metadata": {"query_mode": "mix"},
    }
    resp = await client.post(f"{WS}/query/data", json={"query": "KI-Strategie"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["message"] == "Query completed successfully"
    assert body["data"]["entities"] == [{"entity_name": "KI", "entity_type": "concept"}]
    assert body["data"]["relationships"] == [{"src_id": "KI", "tgt_id": "Strategie"}]
    assert body["data"]["chunks"] == [{"content": "chunk text", "chunk_id": "c1"}]
    assert body["metadata"] == {"query_mode": "mix"}


@pytest.mark.asyncio
async def test_query_data_references_parsed_and_enriched(client):
    server._db_pool = _mock_pool
    from datetime import datetime, timezone

    uploaded = datetime(2026, 5, 8, 12, 34, 56, tzinfo=timezone.utc)
    _mock_pool.fetch.reset_mock()
    _mock_pool.fetch.return_value = [
        {
            "lightrag_key": "20ed9a7c_Tag_1.pdf",  # what LightRAG returns; JOIN key
            "file_path": "/opt/data/workspace/Tag_1.pdf",  # real display path
            "file": "Tag_1.pdf",
            "job_id": "20ed9a7c",
            "description": "Q1 strategy",
            "source_path": "Tag_1.pdf",
            "last_modified_time": "2026-04-01T10:00:00",
            "uploaded_at": uploaded,
            "llm_model_extracted": "deepseek/deepseek-v4-flash",
        }
    ]
    rag_stub.lightrag.aquery_data.return_value = {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [],
            "relationships": [],
            "chunks": [],
            "references": [{"reference_id": "1", "file_path": "20ed9a7c_Tag_1.pdf"}],
        },
        "metadata": {},
    }
    resp = await client.post(f"{WS}/query/data", json={"query": "test"})
    assert resp.status_code == 200
    ref = resp.json()["data"]["references"][0]
    assert "file_name" not in ref  # file_name is no longer emitted at all
    assert ref["file_path"] == "/opt/data/workspace/Tag_1.pdf"
    assert ref["file_description"] == "Q1 strategy"
    assert "source_path" not in ref  # dropped from references
    assert ref["uploaded_at"] == "2026-05-08T12:34:56"
    assert ref["llm_model_extracted"] == "deepseek/deepseek-v4-flash"
    # /query/data performs no synthesis -> no answering model on its references
    assert "llm_model_answered" not in ref
    _mock_pool.fetch.return_value = []  # restore default


@pytest.mark.asyncio
async def test_query_data_block_file_paths_resolved_to_real(client):
    """Entities/relationships/chunks show the REAL Postgres path, not LightRAG's internal key.
    Multi-source <SEP>-joined lists resolve segment-by-segment (SEP preserved); an unresolved
    segment falls back to a prefix-stripped basename so no {job_id}_ token surfaces. References
    carry the real path and no file_name."""
    server._db_pool = _mock_pool
    _mock_pool.fetch.reset_mock()
    _mock_pool.fetch.return_value = [
        _meta_row("aa11beef_overview.txt", "/corpus/helix/docs/overview.txt", "overview.txt"),
        _meta_row("bb22cafe_memo.txt", "/corpus/helix/docs/memo.txt", "memo.txt"),
    ]
    rag_stub.lightrag.aquery_data.return_value = {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [
                {
                    "entity_name": "Helix",
                    "file_path": "aa11beef_overview.txt<SEP>bb22cafe_memo.txt",
                },
                {"entity_name": "Orphan", "file_path": "cc33dead_ghost.txt"},  # no row -> fallback
            ],
            "relationships": [{"src_id": "A", "tgt_id": "B", "file_path": "bb22cafe_memo.txt"}],
            "chunks": [{"content": "c1", "file_path": "aa11beef_overview.txt"}],
            "references": [{"reference_id": "1", "file_path": "aa11beef_overview.txt"}],
        },
        "metadata": {},
    }
    data = (await client.post(f"{WS}/query/data", json={"query": "t"})).json()["data"]
    # multi-source entity -> both real paths, <SEP> structure preserved
    assert (
        data["entities"][0]["file_path"]
        == "/corpus/helix/docs/overview.txt<SEP>/corpus/helix/docs/memo.txt"
    )
    # unresolved segment -> {job_id}_ prefix stripped (no hex token surfaces)
    assert data["entities"][1]["file_path"] == "ghost.txt"
    assert data["relationships"][0]["file_path"] == "/corpus/helix/docs/memo.txt"
    assert data["chunks"][0]["file_path"] == "/corpus/helix/docs/overview.txt"
    blob = str(data["entities"] + data["relationships"] + data["chunks"])
    assert "aa11beef_" not in blob and "bb22cafe_" not in blob and "cc33dead_" not in blob
    ref = data["references"][0]
    assert ref["file_path"] == "/corpus/helix/docs/overview.txt" and "file_name" not in ref
    _mock_pool.fetch.return_value = []  # restore default


@pytest.mark.asyncio
async def test_query_data_include_references_false(client):
    server._db_pool = _mock_pool
    _mock_pool.fetch.reset_mock()
    rag_stub.lightrag.aquery_data.return_value = {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [],
            "relationships": [],
            "chunks": [],
            "references": [{"reference_id": "1", "file_path": "20ed9a7c_Tag_1.pdf"}],
        },
        "metadata": {},
    }
    resp = await client.post(
        f"{WS}/query/data", json={"query": "test", "include_references": False}
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["references"] == []
    _mock_pool.fetch.assert_not_called()


@pytest.mark.asyncio
async def test_query_data_top_k_and_default_mode_forwarded(client):
    await client.post(f"{WS}/query/data", json={"query": "test", "top_k": 12})
    kwargs = lightrag_mod.QueryParam.call_args.kwargs
    assert kwargs.get("top_k") == 12
    assert kwargs.get("mode") == "mix"
    # aquery_data never gets an include_references QueryParam knob
    assert "include_references" not in kwargs


@pytest.mark.asyncio
async def test_query_unknown_workspace_404(client):
    # Hard cutover: no global "not initialised" state. An unknown/soft-deleted workspace 404s.
    async def _no_such(workspace_id):
        return None

    server.workspaces._lookup_workspace = _no_such
    resp = await client.post("/workspace/ghost/query/data", json={"query": "test"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_query_data_exception_returns_500(client):
    rag_stub.lightrag.aquery_data.side_effect = RuntimeError("boom")
    try:
        resp = await client.post(f"{WS}/query/data", json={"query": "test"})
    finally:
        rag_stub.lightrag.aquery_data.side_effect = None
    assert resp.status_code == 500


# --------------------------------------------------------------------------- #
# file_path_contains — folder/file scope post-filter
# --------------------------------------------------------------------------- #


def test_path_matches_any_empty_needles_keeps_all():
    # Empty / None needle list = no filter, everything passes (even empty values).
    assert server.references._path_matches_any("/opt/data/workspace/career/cv.pdf", []) is True
    assert server.references._path_matches_any("/opt/data/workspace/career/cv.pdf", None) is True
    assert server.references._path_matches_any("", []) is True
    assert server.references._path_matches_any(None, []) is True


def test_path_matches_any_single_and_or_semantics():
    p = "/opt/data/workspace/career/cv.pdf"
    assert server.references._path_matches_any(p, ["/career/"]) is True
    assert server.references._path_matches_any(p, ["/projects/"]) is False
    # OR: matches if ANY needle is a substring.
    assert server.references._path_matches_any(p, ["/projects/", "/career/"]) is True


def test_path_matches_any_case_insensitive():
    assert server.references._path_matches_any("/opt/Data/Workspace/Career/CV.pdf", ["/career/"]) is True
    assert server.references._path_matches_any("/opt/data/workspace/career/cv.pdf", ["/CAREER/"]) is True


def test_path_matches_any_empty_value_with_needles_is_no_match():
    assert server.references._path_matches_any("", ["/career/"]) is False
    assert server.references._path_matches_any(None, ["/career/"]) is False


def test_path_matches_any_sep_joined_list():
    # Entities/relationships carry a GRAPH_FIELD_SEP-joined list of source files; a substring test
    # still matches within the joined string.
    joined = "/opt/data/workspace/notes/a.md<SEP>/opt/data/workspace/career/cv.pdf"
    assert server.references._path_matches_any(joined, ["/career/"]) is True
    assert server.references._path_matches_any(joined, ["/finance/"]) is False


@pytest.mark.asyncio
async def test_query_data_file_path_filter_or(client):
    # References resolve to their real path via Postgres; provide rows so the two refs map to
    # real paths (the reference post-filter runs on the resolved real path).
    server._db_pool = _mock_pool
    _mock_pool.fetch.reset_mock()
    _mock_pool.fetch.return_value = [
        {
            "lightrag_key": "/opt/data/workspace/career/cv.pdf",
            "file_path": "/opt/data/workspace/career/cv.pdf",
            "file": "cv.pdf",
            "job_id": None,
            "description": None,
            "source_path": None,
            "last_modified_time": None,
            "uploaded_at": None,
            "llm_model_extracted": None,
        },
        {
            "lightrag_key": "/opt/data/workspace/finance/tax.pdf",
            "file_path": "/opt/data/workspace/finance/tax.pdf",
            "file": "tax.pdf",
            "job_id": None,
            "description": None,
            "source_path": None,
            "last_modified_time": None,
            "uploaded_at": None,
            "llm_model_extracted": None,
        },
    ]
    rag_stub.lightrag.aquery_data.return_value = {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [
                {"entity_name": "A", "file_path": "/opt/data/workspace/career/cv.pdf"},
                {"entity_name": "B", "file_path": "/opt/data/workspace/finance/tax.pdf"},
            ],
            "relationships": [
                {"src_id": "A", "tgt_id": "B", "file_path": "/opt/data/workspace/projects/x.md"},
                {"src_id": "A", "tgt_id": "C", "file_path": "/opt/data/workspace/finance/tax.pdf"},
            ],
            "chunks": [
                {"content": "c1", "file_path": "/opt/data/workspace/career/cv.pdf"},
                {"content": "c2", "file_path": "/opt/data/workspace/finance/tax.pdf"},
            ],
            "references": [
                {"reference_id": "1", "file_path": "/opt/data/workspace/career/cv.pdf"},
                {"reference_id": "2", "file_path": "/opt/data/workspace/finance/tax.pdf"},
            ],
        },
        "metadata": {},
    }
    resp = await client.post(
        f"{WS}/query/data",
        json={
            "query": "test",
            "file_path_contains": ["/career/", "/projects/"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert [e["entity_name"] for e in data["entities"]] == ["A"]
    assert [r["tgt_id"] for r in data["relationships"]] == ["B"]  # the /projects/ rel
    assert [c["content"] for c in data["chunks"]] == ["c1"]
    assert [r["file_path"] for r in data["references"]] == ["/opt/data/workspace/career/cv.pdf"]
    _mock_pool.fetch.return_value = []  # restore default


@pytest.mark.asyncio
async def test_query_data_file_path_filter_boosts_top_k(client):
    await client.post(
        f"{WS}/query/data",
        json={
            "query": "test",
            "top_k": 12,
            "file_path_contains": ["/career/"],
        },
    )
    kwargs = lightrag_mod.QueryParam.call_args.kwargs
    boost = server.config.RAG_FILTER_TOPK_BOOST
    assert kwargs.get("top_k") == 12 * boost
    assert kwargs.get("chunk_top_k") == 12 * boost


@pytest.mark.asyncio
async def test_query_data_no_filter_does_not_boost_or_drop(client):
    rag_stub.lightrag.aquery_data.return_value = {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [{"entity_name": "A", "file_path": "/opt/data/workspace/finance/tax.pdf"}],
            "relationships": [],
            "chunks": [],
            "references": [],
        },
        "metadata": {},
    }
    resp = await client.post(f"{WS}/query/data", json={"query": "test", "top_k": 12})
    assert resp.status_code == 200
    # Nothing dropped even though no path matches a (non-existent) filter.
    assert len(resp.json()["data"]["entities"]) == 1
    kwargs = lightrag_mod.QueryParam.call_args.kwargs
    assert kwargs.get("top_k") == 12  # no boost
    assert "chunk_top_k" not in kwargs  # omitted when no filter


@pytest.mark.asyncio
async def test_graph_html_file_path_filter(client):
    def _node(nid, fp):
        return SimpleNamespace(id=nid, properties={"file_path": fp, "entity_id": nid}, labels=["E"])

    kg = SimpleNamespace(
        nodes=[
            _node("career_node", "/opt/data/workspace/career/cv.pdf"),
            _node("finance_node", "/opt/data/workspace/finance/tax.pdf"),
        ],
        edges=[SimpleNamespace(source="career_node", target="finance_node", properties={})],
    )
    gkg = AsyncMock(return_value=kg)
    orig = rag_stub.lightrag.get_knowledge_graph
    rag_stub.lightrag.get_knowledge_graph = gkg
    try:
        resp = await client.get(f"{WS}/graph.html", params={"file_path_contains": "/career/"})
    finally:
        rag_stub.lightrag.get_knowledge_graph = orig
    assert resp.status_code == 200
    # Only the career node survives the filter (its label is rendered; the finance one is gone).
    assert "career_node" in resp.text
    assert "finance_node" not in resp.text
    # max_nodes was boosted for the filtered fetch (default 1000 * boost).
    assert gkg.call_args.kwargs["max_nodes"] == 1000 * server.config.RAG_FILTER_TOPK_BOOST


@pytest.mark.asyncio
async def test_graph_html_node_file_paths_resolved_no_key_leak(client):
    """Graph node file_path (shown in tooltips) must be resolved to the REAL Postgres path, and
    the folder filter must match that real path — never LightRAG's internal {job_id}_ key."""
    server._db_pool = _mock_pool
    _mock_pool.fetch.reset_mock()
    _mock_pool.fetch.return_value = [
        _meta_row("aa11beef_overview.txt", "/corpus/helix/docs/overview.txt", "overview.txt"),
    ]
    _mock_pool.fetch.return_value = [
        _meta_row("aa11beef_overview.txt", "/corpus/helix/docs/overview.txt", "overview.txt"),
        _meta_row("bb22cafe_memo.txt", "/corpus/helix/docs/memo.txt", "memo.txt"),
    ]
    kg = SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="helix_node",
                properties={"file_path": "aa11beef_overview.txt", "entity_id": "helix_node"},
                labels=["E"],
            ),
            SimpleNamespace(
                id="memo_node",
                properties={"file_path": "bb22cafe_memo.txt", "entity_id": "memo_node"},
                labels=["E"],
            ),
        ],
        # edge tooltip carries a multi-source <SEP>-joined internal-key file_path too
        edges=[
            SimpleNamespace(
                source="helix_node",
                target="memo_node",
                properties={"file_path": "aa11beef_overview.txt<SEP>bb22cafe_memo.txt"},
            )
        ],
    )
    gkg = AsyncMock(return_value=kg)
    orig = rag_stub.lightrag.get_knowledge_graph
    rag_stub.lightrag.get_knowledge_graph = gkg
    try:
        resp = await client.get(f"{WS}/graph.html")
    finally:
        rag_stub.lightrag.get_knowledge_graph = orig
    assert resp.status_code == 200
    assert "helix_node" in resp.text  # rendered
    assert "/corpus/helix/docs/overview.txt" in resp.text  # real path in a tooltip
    # internal {job_id}_ keys never rendered anywhere (nodes OR edges)
    assert "aa11beef_" not in resp.text and "bb22cafe_" not in resp.text
    _mock_pool.fetch.return_value = []


@pytest.mark.asyncio
async def test_graph_html_no_filter_no_boost(client):
    kg = SimpleNamespace(
        nodes=[SimpleNamespace(id="n1", properties={"entity_id": "n1"}, labels=["E"])],
        edges=[],
    )
    gkg = AsyncMock(return_value=kg)
    orig = rag_stub.lightrag.get_knowledge_graph
    rag_stub.lightrag.get_knowledge_graph = gkg
    try:
        resp = await client.get(f"{WS}/graph.html")
    finally:
        rag_stub.lightrag.get_knowledge_graph = orig
    assert resp.status_code == 200
    assert gkg.call_args.kwargs["max_nodes"] == 1000  # unboosted default


# --------------------------------------------------------------------------- #
# _split_if_csv — per-file metadata from comma-separated form fields
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_batch_json_metadata_per_file(tmp_path, client):
    server._db_pool = _mock_pool
    _mock_pool.execute.reset_mock()
    file1 = tmp_path / "A.txt"
    file2 = tmp_path / "B.txt"
    file1.write_text("aaa")
    file2.write_text("bbb")
    import json as _json

    meta = _json.dumps(
        [
            {
                "description": "Desc A",
                "source_path": "/src/A.txt",
                "last_modified_time": "2026-01-01T00:00:00",
            },
            {
                "description": "Desc B",
                "source_path": "/src/B.txt",
                "last_modified_time": "2026-02-01T00:00:00",
            },
        ]
    )
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        resp = await client.post(
            f"{WS}/upload/batch",
            files=[
                ("files", ("A.txt", file1.read_bytes(), "text/plain")),
                ("files", ("B.txt", file2.read_bytes(), "text/plain")),
            ],
            data={"metadata": meta},
        )
    assert resp.status_code == 200
    # INSERT args: (sql, job_id, batch_id, workspace, file, description, source_path,
    #               last_modified_time, content_hash, file_path, lightrag_key, llm_model_extracted)
    insert_calls = [
        c for c in _mock_pool.execute.call_args_list if "INSERT INTO rag_file_metadata" in str(c)
    ]
    assert len(insert_calls) == 2
    args0 = insert_calls[0].args
    args1 = insert_calls[1].args
    assert args0[3] == "alex" and args1[3] == "alex"  # workspace
    descs = {args0[5], args1[5]}
    assert descs == {"Desc A", "Desc B"}
    srcs = {args0[6], args1[6]}
    assert srcs == {"/src/A.txt", "/src/B.txt"}
    # lightrag_key (arg 10) is the {job_id}_{basename} identity; llm_model_extracted moved to arg 11.
    assert args0[10] == f"{args0[1]}_{args0[4]}" and args1[10] == f"{args1[1]}_{args1[4]}"
    assert (
        args0[11] == server.config.LLM_MODEL and args1[11] == server.config.LLM_MODEL
    )  # extractor captured at ingest


# --------------------------------------------------------------------------- #
# Workspace registry — migration & seed (Step 1)
# --------------------------------------------------------------------------- #


def _fresh_pool(primary_row=None):
    pool = MagicMock()
    pool.execute = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=primary_row)
    pool.fetch = AsyncMock(return_value=[])
    return pool


@pytest.mark.asyncio
async def test_db_init_creates_both_tables():
    pool = _fresh_pool()
    await server._db_init(pool)
    sql = " ".join(str(c) for c in pool.execute.call_args_list)
    assert "rag_file_metadata" in sql
    assert "rag_workspaces" in sql


@pytest.mark.asyncio
async def test_db_init_adds_workspace_column_backfills_and_enforces_not_null(monkeypatch):
    monkeypatch.setattr(server.config, "POSTGRES_WORKSPACE", "default")
    pool = _fresh_pool()
    await server._db_init(pool)
    stmts = list(pool.execute.call_args_list)
    joined = " ".join(str(c) for c in stmts)
    assert "ADD COLUMN IF NOT EXISTS workspace" in joined
    assert "SET NOT NULL" in joined
    # backfill uses the physical workspace value, not a hardcoded literal
    update_calls = [
        c for c in stmts if "UPDATE rag_file_metadata" in str(c) and "workspace" in str(c)
    ]
    assert update_calls, "expected a backfill UPDATE for existing rows"
    assert "default" in update_calls[0].args


@pytest.mark.asyncio
async def test_db_init_workspace_column_has_no_default():
    # The added column must NOT carry a DEFAULT — future inserts must set workspace explicitly.
    pool = _fresh_pool()
    await server._db_init(pool)
    add_col = [
        str(c)
        for c in pool.execute.call_args_list
        if "ADD COLUMN IF NOT EXISTS workspace" in str(c)
    ]
    assert add_col and "DEFAULT" not in add_col[0]


@pytest.mark.asyncio
async def test_db_init_seeds_primary_default_when_absent(monkeypatch):
    monkeypatch.setattr(server.config, "POSTGRES_WORKSPACE", "default")
    pool = _fresh_pool(primary_row=None)
    await server._db_init(pool)
    inserts = [c for c in pool.execute.call_args_list if "INSERT INTO rag_workspaces" in str(c)]
    # The primary workspace seed runs when no primary row exists yet.
    assert len(inserts) >= 1
    args = inserts[0].args
    assert server.config.PRIMARY_WORKSPACE_ID in args  # "default"
    assert server.config.PRIMARY_WORKSPACE_DESCRIPTION in args
    assert "default" in args  # physical lightrag_workspace == POSTGRES_WORKSPACE


@pytest.mark.asyncio
async def test_db_init_does_not_reseed_when_primary_exists():
    pool = _fresh_pool(primary_row={"id": "alex"})
    await server._db_init(pool)
    inserts = [c for c in pool.execute.call_args_list if "INSERT INTO rag_workspaces" in str(c)]
    assert not inserts


def test_primary_workspace_constants():
    assert server.config.PRIMARY_WORKSPACE_ID == "default"
    assert server.config.PRIMARY_WORKSPACE_DESCRIPTION == "Default workspace."


# --------------------------------------------------------------------------- #
# Instance registry — get_workspace_rag (Step 2)
# --------------------------------------------------------------------------- #


def _ws_pool(row):
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value=row)
    return pool


_ALEX_ROW = {"id": "alex", "lightrag_workspace": "default", "is_primary": True}


@pytest.mark.asyncio
async def test_get_workspace_rag_builds_and_caches(monkeypatch):
    server._db_pool = _ws_pool(_ALEX_ROW)
    built = []

    async def _fake_build(wid, physical):
        built.append((wid, physical))
        return MagicMock(name=f"rag-{wid}")

    monkeypatch.setattr(server.workspaces, "_build_workspace_rag", _fake_build)
    inst1 = await server.get_workspace_rag("alex")
    inst2 = await server.get_workspace_rag("alex")
    assert inst1 is inst2
    assert built == [("alex", "default")]  # built exactly once, public→physical resolved


@pytest.mark.asyncio
async def test_get_workspace_rag_unknown_404(monkeypatch):
    server._db_pool = _ws_pool(None)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await server.get_workspace_rag("ghost")
    assert ei.value.status_code == 404


@pytest.mark.asyncio
async def test_get_workspace_rag_soft_deleted_404(monkeypatch):
    # _lookup_workspace filters deleted_at IS NULL, so a soft-deleted workspace returns None → 404
    server._db_pool = _ws_pool(None)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await server.get_workspace_rag("career")
    assert ei.value.status_code == 404


@pytest.mark.asyncio
async def test_get_workspace_rag_concurrent_builds_once(monkeypatch):
    server._db_pool = _ws_pool(_ALEX_ROW)
    calls = 0

    async def _fake_build(wid, physical):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return MagicMock()

    monkeypatch.setattr(server.workspaces, "_build_workspace_rag", _fake_build)
    results = await asyncio.gather(*[server.get_workspace_rag("alex") for _ in range(8)])
    assert calls == 1
    assert all(r is results[0] for r in results)


@pytest.mark.asyncio
async def test_lookup_workspace_filters_deleted():
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value=None)
    server._db_pool = pool
    await server.workspaces._lookup_workspace("career")
    sql = str(pool.fetchrow.call_args)
    assert "deleted_at IS NULL" in sql


# --------------------------------------------------------------------------- #
# Workspace registry API (Step 3)
# --------------------------------------------------------------------------- #


def _registry_pool(existing=None, rows=None):
    pool = MagicMock()
    pool.execute = AsyncMock()
    pool.fetch = AsyncMock(return_value=rows or [])
    pool.fetchrow = AsyncMock(return_value=existing)
    return pool


@pytest.mark.parametrize("slug", ["business", "ai_base_1", "career", "a", "a_b_c", "x" * 48])
def test_valid_slugs(slug):
    assert server._is_valid_slug(slug) is True


@pytest.mark.parametrize(
    "slug", ["Business", "1abc", "a b", "", "x" * 49, "a-b", "ab!", "_ab", "café"]
)
def test_invalid_slugs(slug):
    assert server._is_valid_slug(slug) is False


@pytest.mark.asyncio
async def test_create_workspace_sets_lightrag_workspace_to_id(client):
    pool = _registry_pool(existing=None)
    server._db_pool = pool
    resp = await client.post(
        "/all-workspaces/create",
        json={"id": "career", "name": "Career", "description": "x", "lightrag_workspace": "HACK"},
    )
    assert resp.status_code == 200
    inserts = [c for c in pool.execute.call_args_list if "INSERT INTO rag_workspaces" in str(c)]
    assert len(inserts) == 1
    args = inserts[0].args
    assert args.count("career") >= 2  # id AND lightrag_workspace both 'career'
    assert "HACK" not in args  # client-supplied lightrag_workspace ignored


@pytest.mark.asyncio
async def test_create_workspace_invalid_slug_422(client):
    server._db_pool = _registry_pool(existing=None)
    resp = await client.post("/all-workspaces/create", json={"id": "Bad Slug", "name": "x"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_workspace_duplicate_409(client):
    pool = _registry_pool(existing={"id": "career", "is_primary": False, "deleted_at": None})
    server._db_pool = pool
    resp = await client.post("/all-workspaces/create", json={"id": "career", "name": "x"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_list_workspaces_active(client):
    from datetime import datetime, timezone

    pool = _registry_pool(
        rows=[
            {
                "id": "alex",
                "name": "alex",
                "description": "Alex's personal workspace.",
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            },
        ]
    )
    server._db_pool = pool
    resp = await client.get("/all-workspaces/list")
    assert resp.status_code == 200
    data = resp.json()["workspaces"]
    assert data[0]["id"] == "alex"
    assert "deleted_at IS NULL" in str(pool.fetch.call_args)


@pytest.mark.asyncio
async def test_list_workspaces_deleted_filter(client):
    pool = _registry_pool()
    server._db_pool = pool
    await client.get("/all-workspaces/list?deleted=true")
    assert "deleted_at IS NOT NULL" in str(pool.fetch.call_args)


@pytest.mark.asyncio
async def test_soft_delete_sets_deleted_at(client):
    pool = _registry_pool(existing={"id": "career", "is_primary": False, "deleted_at": None})
    server._db_pool = pool
    resp = await client.delete("/workspace/career")
    assert resp.status_code == 200
    upd = [c for c in pool.execute.call_args_list if "deleted_at = NOW()" in str(c)]
    assert upd


@pytest.mark.asyncio
async def test_delete_primary_409(client):
    pool = _registry_pool(existing={"id": "alex", "is_primary": True, "deleted_at": None})
    server._db_pool = pool
    resp = await client.delete("/workspace/alex")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_delete_unknown_404(client):
    pool = _registry_pool(existing=None)
    server._db_pool = pool
    resp = await client.delete("/workspace/ghost")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_restore_clears_deleted_at(client):
    pool = _registry_pool(
        existing={"id": "career", "is_primary": False, "deleted_at": "2026-01-01T00:00:00"}
    )
    server._db_pool = pool
    resp = await client.post("/workspace/career/restore")
    assert resp.status_code == 200
    upd = [c for c in pool.execute.call_args_list if "deleted_at = NULL" in str(c)]
    assert upd


@pytest.mark.asyncio
async def test_restore_not_deleted_404(client):
    pool = _registry_pool(existing={"id": "career", "is_primary": False, "deleted_at": None})
    server._db_pool = pool
    resp = await client.post("/workspace/career/restore")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Hard purge (Step 5)
# --------------------------------------------------------------------------- #


def _purge_pool(graph_exists=False, lightrag_tables=None):
    pool = MagicMock()
    pool.execute = AsyncMock()
    pool.fetch = AsyncMock(return_value=[{"table_name": t} for t in (lightrag_tables or [])])
    pool.fetchval = AsyncMock(return_value=1 if graph_exists else None)
    pool.fetchrow = AsyncMock(
        return_value={
            "id": "career",
            "name": "Career",
            "description": None,
            "is_primary": False,
            "deleted_at": None,
            "lightrag_workspace": "career",
        }
    )
    conn = MagicMock()
    conn.execute = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=cm)
    pool._conn = conn  # expose for assertions
    return pool


@pytest.mark.asyncio
async def test_purge_deletes_rows_files_and_registry(client, tmp_path):
    pool = _purge_pool(lightrag_tables=["lightrag_doc_status", "lightrag_doc_full"])
    server._db_pool = pool
    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        (tmp_path / "career").mkdir()
        (tmp_path / "career" / "x.txt").write_text("data")
        resp = await client.delete("/workspace/career?purge=true")
    assert resp.status_code == 200
    assert resp.json()["status"] == "purged"
    executed = [str(c) for c in pool.execute.call_args_list]
    assert any("DELETE FROM" in s and "lightrag_doc_status" in s for s in executed)
    assert any("DELETE FROM rag_file_metadata" in s for s in executed)
    assert any("DELETE FROM rag_workspaces" in s for s in executed)
    assert not (tmp_path / "career").exists()  # files dir removed


@pytest.mark.asyncio
async def test_purge_drops_age_graph_when_present(client, tmp_path):
    pool = _purge_pool(graph_exists=True)
    server._db_pool = pool
    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        resp = await client.delete("/workspace/career?purge=true")
    assert resp.status_code == 200
    drops = [str(c) for c in pool._conn.execute.call_args_list if "drop_graph" in str(c)]
    assert drops and "career_chunk_entity_relation" in drops[0]


@pytest.mark.asyncio
async def test_purge_skips_graph_drop_when_absent(client, tmp_path):
    pool = _purge_pool(graph_exists=False)
    server._db_pool = pool
    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        resp = await client.delete("/workspace/career?purge=true")
    assert resp.status_code == 200
    drops = [str(c) for c in pool._conn.execute.call_args_list if "drop_graph" in str(c)]
    assert not drops  # no graph → no drop attempted


@pytest.mark.asyncio
async def test_purge_primary_409(client):
    pool = _purge_pool()
    pool.fetchrow = AsyncMock(
        return_value={
            "id": "alex",
            "name": "alex",
            "description": None,
            "is_primary": True,
            "deleted_at": None,
            "lightrag_workspace": "default",
        }
    )
    server._db_pool = pool
    resp = await client.delete("/workspace/alex?purge=true")
    assert resp.status_code == 409
    # no destructive SQL ran
    assert not any("drop_graph" in str(c) for c in pool._conn.execute.call_args_list)
    assert not any("DELETE FROM rag_workspaces" in str(c) for c in pool.execute.call_args_list)


# --------------------------------------------------------------------------- #
# Workspace-scoped routing & isolation (Step 4)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/query"),
        ("post", "/query/data"),
        ("post", "/upload/batch"),
        ("get", "/jobs"),
        ("get", "/batch/x"),
        ("get", "/status/x"),
    ],
)
async def test_old_unprefixed_routes_gone(client, method, path):
    kwargs = {"json": {"query": "t"}} if method == "post" else {}
    resp = await getattr(client, method)(path, **kwargs)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_require_workspace_invalid_slug_404(client):
    resp = await client.post("/workspace/Bad_Slug!/query", json={"query": "t"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_upload_routes_to_named_workspace(tmp_path, client):
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        resp = await client.post("/workspace/career/upload/batch", files=[_fake_upload("c.txt")])
    job_id = resp.json()["jobs"][0]["job_id"]
    assert server._jobs[job_id]["workspace"] == "career"
    # file saved under the career subdir; queued item carries the workspace
    assert (tmp_path / "career" / f"{job_id}_c.txt").exists()
    ws, qid, _dest, _meta, _fp = await server._job_queue.get()
    assert ws == "career" and qid == job_id


@pytest.mark.asyncio
async def test_status_scoped_to_workspace(tmp_path, client):
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        resp = await client.post(f"{WS}/upload/batch", files=[_fake_upload("a.txt")])
    job_id = resp.json()["jobs"][0]["job_id"]
    # visible under its own workspace…
    assert (await client.get(f"{WS}/status/{job_id}")).status_code == 200
    # …but not under a different workspace (no DB fallback configured)
    assert (await client.get(f"/workspace/career/status/{job_id}")).status_code == 404


@pytest.mark.asyncio
async def test_batch_scoped_to_workspace(tmp_path, client):
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock()),
    ):
        resp = await client.post(f"{WS}/upload/batch", files=[_fake_upload("a.txt")])
    batch_id = resp.json()["batch_id"]
    assert (await client.get(f"{WS}/batch/{batch_id}")).status_code == 200
    assert (await client.get(f"/workspace/career/batch/{batch_id}")).status_code == 404


@pytest.mark.asyncio
async def test_two_workspaces_get_distinct_instances(monkeypatch):
    # Real registry: each public id maps to its own cached instance.
    server._db_pool = _ws_pool(None)

    async def _fake_lookup(wid):
        return {
            "id": wid,
            "name": wid,
            "description": None,
            "lightrag_workspace": wid,
            "is_primary": False,
        }

    async def _fake_build(wid, physical):
        return MagicMock(name=f"rag-{wid}")

    monkeypatch.setattr(server.workspaces, "_lookup_workspace", _fake_lookup)
    monkeypatch.setattr(server.workspaces, "_build_workspace_rag", _fake_build)
    a = await server.get_workspace_rag("business")
    b = await server.get_workspace_rag("career")
    assert a is not b
    assert (await server.get_workspace_rag("business")) is a


# --------------------------------------------------------------------------- #
# Part 0 — ingestion-integrity guard (_verify_ingestion)
# --------------------------------------------------------------------------- #


def _docs(status="processed", content_length=11):
    return _AnyKeyDocs({"status": status, "content_length": content_length, "metadata": {}})


@pytest.mark.asyncio
async def test_ingestion_ok_when_processed(tmp_path):
    """Happy path: doc_status processed + small content → no raise, ainsert + verify ran."""
    path = tmp_path / "ok.txt"
    path.write_text("hello world")
    rag_stub.lightrag.aget_docs_by_ids.return_value = _docs(status="processed")
    await server._process_file(path, rag_stub)
    rag_stub.lightrag.ainsert.assert_awaited_once()
    rag_stub.lightrag.aget_docs_by_ids.assert_awaited_once()
    rag_stub.lightrag.adelete_by_doc_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingestion_raises_when_doc_failed(tmp_path):
    """doc_status FAILED → IngestionIncompleteError + partial doc cleaned up with cache."""
    path = tmp_path / "bad.txt"
    path.write_text("hello world")
    rag_stub.lightrag.aget_docs_by_ids.return_value = _docs(status="failed")
    with pytest.raises(server.ingest.IngestionIncompleteError, match="doc_status=failed"):
        await server._process_file(path, rag_stub)
    rag_stub.lightrag.adelete_by_doc_id.assert_awaited_once()
    assert rag_stub.lightrag.adelete_by_doc_id.call_args.kwargs.get("delete_llm_cache") is True


@pytest.mark.asyncio
async def test_ingestion_raises_when_no_doc_status(tmp_path):
    """No doc_status row at all → treated as failure (and cleaned up)."""
    path = tmp_path / "missing.txt"
    path.write_text("hello world")
    rag_stub.lightrag.aget_docs_by_ids.return_value = {}  # no row for the computed doc id
    with pytest.raises(server.ingest.IngestionIncompleteError, match="no_doc_status"):
        await server._process_file(path, rag_stub)
    rag_stub.lightrag.adelete_by_doc_id.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_graph_guard_fails_when_no_entities(tmp_path):
    """processed but zero entities on non-trivial content → failure (empty_graph)."""
    server._db_pool = _mock_pool
    _mock_pool.fetchrow.return_value = {"count": 0}
    path = tmp_path / "long.txt"
    path.write_text("x" * 300)
    rag_stub.lightrag.aget_docs_by_ids.return_value = _docs(status="processed", content_length=None)
    try:
        with pytest.raises(server.ingest.IngestionIncompleteError, match="empty_graph"):
            await server._process_file(path, rag_stub)
    finally:
        _mock_pool.fetchrow.return_value = None
    rag_stub.lightrag.adelete_by_doc_id.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_graph_guard_passes_when_entities_present(tmp_path):
    server._db_pool = _mock_pool
    _mock_pool.fetchrow.return_value = {"count": 5}
    path = tmp_path / "long.txt"
    path.write_text("x" * 300)
    rag_stub.lightrag.aget_docs_by_ids.return_value = _docs(status="processed", content_length=None)
    try:
        await server._process_file(path, rag_stub)
    finally:
        _mock_pool.fetchrow.return_value = None
    rag_stub.lightrag.adelete_by_doc_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_graph_guard_disabled(tmp_path, monkeypatch):
    """With the guard off, zero entities is allowed (corpora that yield no entities)."""
    monkeypatch.setattr(server.config, "RAG_REQUIRE_GRAPH_EXTRACTION", False)
    server._db_pool = _mock_pool
    _mock_pool.fetchrow.return_value = {"count": 0}
    path = tmp_path / "long.txt"
    path.write_text("x" * 300)
    rag_stub.lightrag.aget_docs_by_ids.return_value = _docs(status="processed", content_length=None)
    try:
        await server._process_file(path, rag_stub)
    finally:
        _mock_pool.fetchrow.return_value = None
    rag_stub.lightrag.adelete_by_doc_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_ingest_retries_then_fails_via_process_job(tmp_path):
    """End-to-end: a partial ingest surfaces as a job failure (not silent 'done'), records the
    reason, and cleaned up the partial doc — so the file isn't left searchable-but-unlinked."""
    path = tmp_path / "p0.txt"
    path.write_text("content")
    job_id = "p0fail"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "p0.txt",
        "workspace": "alex",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }
    rag_stub.lightrag.aget_docs_by_ids.return_value = _docs(status="failed")
    original = server.config.MAX_RETRIES
    server.config.MAX_RETRIES = 1
    with patch.object(server, "get_workspace_rag", new=AsyncMock(return_value=rag_stub)):
        await server._process_job("alex", job_id, path, "")
    server.config.MAX_RETRIES = original
    assert server._jobs[job_id]["status"] == "failed"
    assert "incomplete" in server._jobs[job_id]["error"]
    rag_stub.lightrag.adelete_by_doc_id.assert_awaited()
    assert not path.exists()


# --------------------------------------------------------------------------- #
# Part A0 — endpoint URL consistency rename
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/workspaces/list"),
        ("post", "/workspaces"),
        ("delete", "/workspaces/career"),
        ("post", "/workspaces/career/restore"),
    ],
)
async def test_old_workspace_registry_routes_gone(client, method, path):
    kwargs = {"json": {"id": "career", "name": "x"}} if method == "post" else {}
    resp = await getattr(client, method)(path, **kwargs)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_new_registry_routes_present(client):
    server._db_pool = _registry_pool(rows=[])
    assert (await client.get("/all-workspaces/list")).status_code == 200
    pool = _registry_pool(existing={"id": "career", "is_primary": False, "deleted_at": None})
    server._db_pool = pool
    assert (await client.delete("/workspace/career")).status_code == 200


# --------------------------------------------------------------------------- #
# Part A1/A3/A4 — caller-supplied absolute file_path, content_hash, doc_id index
# --------------------------------------------------------------------------- #


def test_join_path():
    assert server._join_path("/data/corpus", "sub/dir/f.pdf") == "/data/corpus/sub/dir/f.pdf"
    assert server._join_path("/data/corpus/", "/sub/f.pdf") == "/data/corpus/sub/f.pdf"


@pytest.mark.asyncio
async def test_process_file_uses_given_file_path(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("hello world")
    rag_stub.lightrag.aget_docs_by_ids.return_value = _docs(status="processed")
    await server._process_file(path, rag_stub, file_path="/opt/data/workspace/sub/x.txt")
    assert rag_stub.lightrag.ainsert.call_args.kwargs["file_paths"] == [
        "/opt/data/workspace/sub/x.txt"
    ]
    # The doc id is pinned at insert (ids=) so post-insert verification can find the record.
    assert "ids" in rag_stub.lightrag.ainsert.call_args.kwargs


@pytest.mark.asyncio
async def test_upload_threads_external_path_and_hash(tmp_path, client):
    import json as _json

    meta = _json.dumps([{"source_path": "sub/report.pdf", "path_root": "/data/corpus"}])
    with (
        patch.object(server, "WORKING_DIR", str(tmp_path)),
        patch.object(server, "_process_file", new=AsyncMock(return_value="doc-x")),
    ):
        resp = await client.post(
            f"{WS}/upload/batch",
            files=[_fake_upload("report.pdf", b"hello")],
            data={"metadata": meta},
        )
    job_id = resp.json()["jobs"][0]["job_id"]
    # The queue carries the LightRAG identity (lightrag_input = {job_id}_basename), NOT the
    # display path; the real caller path is recorded as the display file_path.
    ws, j, d, m, fp = await server._job_queue.get()
    assert fp == f"{job_id}_report.pdf"
    assert server._jobs[job_id]["file_path"] == "/data/corpus/sub/report.pdf"
    # content_hash = sha256("hello")
    import hashlib

    assert server._jobs[job_id]["content_hash"] == hashlib.sha256(b"hello").hexdigest()


@pytest.mark.asyncio
async def test_doc_id_persisted_on_done(tmp_path):
    server._db_pool = _mock_pool
    _mock_pool.execute.reset_mock()
    path = tmp_path / "aaa_f.txt"
    path.write_text("content")
    job_id = "docidjob"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "f.txt",
        "workspace": "alex",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }
    with (
        patch.object(server, "_process_file", new=AsyncMock(return_value="doc-abc")),
        patch.object(server, "get_workspace_rag", new=AsyncMock(return_value=rag_stub)),
    ):
        await server._process_job("alex", job_id, path, "", None)
    calls = [str(c) for c in _mock_pool.execute.call_args_list]
    assert any("SET doc_id" in c for c in calls)
    assert server._jobs[job_id]["doc_id"] == "doc-abc"


@pytest.mark.asyncio
async def test_files_index_endpoint(client):
    from datetime import datetime, timezone

    pool = MagicMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                "job_id": "j1",
                "file": "report.pdf",
                "file_path": "/opt/data/workspace/report.pdf",
                "source_path": "report.pdf",
                "doc_id": "doc-1",
                "content_hash": "abc123",
                "status": "done",
                "last_modified_time": "2026-01-01T00:00:00",
                "uploaded_at": datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc),
            }
        ]
    )
    server._db_pool = pool
    resp = await client.get(f"{WS}/files")
    assert resp.status_code == 200
    f = resp.json()["files"][0]
    assert f["doc_id"] == "doc-1" and f["content_hash"] == "abc123"
    assert f["file_path"] == "/opt/data/workspace/report.pdf"
    assert f["uploaded_at"] == "2026-05-08T12:00:00+00:00"


# --------------------------------------------------------------------------- #
# Part A5 — per-file delete (with cache invalidation)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_delete_file_by_doc_id_clears_cache(client):
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value={"job_id": "j1", "file": "f.txt", "doc_id": "doc-1"})
    pool.execute = AsyncMock()
    server._db_pool = pool
    resp = await client.request("DELETE", f"{WS}/file/delete", json={"doc_id": "doc-1"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "deleted", "doc_id": "doc-1"}
    rag_stub.lightrag.adelete_by_doc_id.assert_awaited_once()
    assert rag_stub.lightrag.adelete_by_doc_id.call_args.kwargs.get("delete_llm_cache") is True
    assert any("DELETE FROM rag_file_metadata" in str(c) for c in pool.execute.call_args_list)


@pytest.mark.asyncio
async def test_delete_file_noop_when_not_found(client):
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value=None)
    pool.execute = AsyncMock()
    server._db_pool = pool
    resp = await client.request("DELETE", f"{WS}/file/delete", json={"rel_path": "nope.txt"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "noop"
    rag_stub.lightrag.adelete_by_doc_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_file_by_rel_path_resolves(client):
    pool = MagicMock()
    # first lookup (file_path) misses, second (source_path) hits
    pool.fetchrow = AsyncMock(side_effect=[{"job_id": "j2", "file": "g.txt", "doc_id": "doc-2"}])
    pool.execute = AsyncMock()
    server._db_pool = pool
    resp = await client.request("DELETE", f"{WS}/file/delete", json={"rel_path": "g.txt"})
    assert resp.status_code == 200
    assert resp.json()["doc_id"] == "doc-2"


# --------------------------------------------------------------------------- #
# Part A7 — workspace status endpoint
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_workspace_status(client):
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "default",
                "name": "Default",
                "description": None,
                "lightrag_workspace": "default",
                "is_primary": True,
                "deleted_at": None,
            },  # _db_get_workspace_any
        ]
    )
    pool.fetch = AsyncMock(
        side_effect=[
            [{"status": "processed", "n": 3}, {"status": "failed", "n": 1}],  # doc statuses
            [{"status": "done", "n": 3}],  # job statuses
        ]
    )
    pool.fetchval = AsyncMock(
        side_effect=[
            12,
            "lightrag_vdb_entity_x",
            40,
            "lightrag_vdb_relation_x",
            55,
            "2026-05-08T12:00:00",
        ]
    )
    server._db_pool = pool
    resp = await client.get("/workspace/default")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active"] is True
    assert data["documents"]["by_status"] == {"processed": 3, "failed": 1}
    assert data["documents"]["total"] == 4
    assert "lightrag_workspace" not in data  # internal storage namespace never leaks to the API


@pytest.mark.asyncio
async def test_workspace_status_soft_deleted_shows_inactive(client):
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "career",
                "name": "Career",
                "description": None,
                "lightrag_workspace": "career",
                "is_primary": False,
                "deleted_at": "2026-01-01T00:00:00",
            },
            None,
        ]
    )
    pool.fetch = AsyncMock(side_effect=[[], []])
    pool.fetchval = AsyncMock(side_effect=[0, None, None, None, None])
    server._db_pool = pool
    resp = await client.get("/workspace/career")
    assert resp.status_code == 200
    assert resp.json()["active"] is False


# --------------------------------------------------------------------------- #
# LLM provider routing: _llm_call_kwargs (OpenAI vs OpenRouter/DeepSeek)
# --------------------------------------------------------------------------- #


@pytest.fixture
def _restore_llm_flag():
    """Save/restore the module-level _LLM_IS_OPENAI flag mutated per test."""
    original = server.config._LLM_IS_OPENAI
    yield
    server.config._LLM_IS_OPENAI = original


def test_llm_call_kwargs_openai_path_keeps_max_completion_tokens(_restore_llm_flag):
    server.config._LLM_IS_OPENAI = True
    out = server.config._llm_call_kwargs(
        {"temperature": 0.2, "max_completion_tokens": 16000, "max_tokens": 999, "stream": True}
    )
    # OpenAI path: keep temperature + max_completion_tokens, drop everything else (incl. max_tokens).
    assert out == {"temperature": 0.2, "max_completion_tokens": 16000}


def test_llm_call_kwargs_thirdparty_translates_to_max_tokens(_restore_llm_flag):
    server.config._LLM_IS_OPENAI = False
    out = server.config._llm_call_kwargs(
        {"temperature": 0.2, "max_completion_tokens": 16000, "stream": True}
    )
    # Third-party (OpenRouter/DeepSeek) path: translate max_completion_tokens -> max_tokens.
    assert out == {"temperature": 0.2, "max_tokens": 16000}


def test_llm_call_kwargs_thirdparty_prefers_explicit_max_tokens(_restore_llm_flag):
    server.config._LLM_IS_OPENAI = False
    out = server.config._llm_call_kwargs({"max_tokens": 8000, "max_completion_tokens": 16000})
    # An explicit max_tokens is honoured as-is (no translation/overwrite).
    assert out == {"max_tokens": 8000}


def test_llm_call_kwargs_drops_unknown_kwargs(_restore_llm_flag):
    server.config._LLM_IS_OPENAI = False
    out = server.config._llm_call_kwargs({"foo": "bar", "n": 3})
    assert out == {}


# --------------------------------------------------------------------------- #
# Phase-split LLM routing: extraction (ingest) vs query-time provider
# --------------------------------------------------------------------------- #


def test_llm_phase_defaults_to_query():
    # Outside an ingest, the active phase is the hot query path.
    assert server.config._llm_phase.get() == "query"


def _reimport_server_with_env(monkeypatch, env: dict):
    """Load a fresh, isolated copy of server.py under a throwaway module name with a
    controlled env, so import-time config resolution can be asserted without mutating
    the shared `server` module (its constants are baked from env at import). Import is
    side-effect-free (only os.getenv + FastAPI app/route definitions), and the RAGAnything
    stub in sys.modules is reused, so this is safe to run repeatedly."""
    import importlib.util

    for k in (
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "QUERY_LLM_MODEL",
        "QUERY_LLM_BASE_URL",
        "QUERY_LLM_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location("_server_cfg_probe", server.config.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_query_llm_config_defaults_to_extraction_config(monkeypatch):
    # With no QUERY_LLM_* overrides set, the query config mirrors the extraction config —
    # preserving the historical single-model behaviour. Config is resolved from env at
    # import, so re-derive it in an isolated module instance from a CLEARED env (QUERY_LLM_*
    # unset) to assert the fallback regardless of the ambient/production env (which sets the split).
    mod = _reimport_server_with_env(
        monkeypatch,
        {
            "OPENAI_API_KEY": "oai-key",
            "LLM_MODEL": "extract-only-model",
            "LLM_BASE_URL": "https://openrouter.ai/api/v1",
            "LLM_API_KEY": "or-key",
        },
    )
    assert mod.QUERY_LLM_MODEL == mod.LLM_MODEL == "extract-only-model"
    assert mod.QUERY_LLM_BASE_URL == mod.LLM_BASE_URL == "https://openrouter.ai/api/v1"
    assert mod.QUERY_LLM_API_KEY == mod.LLM_API_KEY == "or-key"
    assert mod.QUERY_LLM_IS_OPENAI == mod._LLM_IS_OPENAI is False


def test_query_llm_config_blank_base_url_reuses_extraction(monkeypatch):
    # docker-compose passes unset vars through as empty strings, so an empty (not just absent)
    # QUERY_LLM_BASE_URL must still reuse the extraction endpoint — otherwise the DeepSeek model id
    # would be sent to OpenAI ("invalid model ID"). This is the exact production/compose shape.
    mod = _reimport_server_with_env(
        monkeypatch,
        {
            "OPENAI_API_KEY": "oai-key",
            "LLM_MODEL": "deepseek/deepseek-chat",
            "LLM_BASE_URL": "https://openrouter.ai/api/v1",
            "LLM_API_KEY": "or-key",
            "QUERY_LLM_MODEL": "",  # blank -> reuse LLM_MODEL
            "QUERY_LLM_BASE_URL": "",  # blank -> reuse LLM_BASE_URL (NOT OpenAI)
            "QUERY_LLM_API_KEY": "",
        },
    )
    assert mod.QUERY_LLM_MODEL == "deepseek/deepseek-chat"
    assert mod.QUERY_LLM_BASE_URL == "https://openrouter.ai/api/v1"
    assert mod.QUERY_LLM_API_KEY == "or-key"
    assert mod.QUERY_LLM_IS_OPENAI is False


def test_query_llm_config_overrides_extraction_config(monkeypatch):
    # With QUERY_LLM_* set (the split production shape), the query config is independent of the
    # extraction config. To send query-time work to a different provider than extraction, set
    # QUERY_LLM_BASE_URL explicitly (here: force OpenAI while extraction runs on OpenRouter).
    mod = _reimport_server_with_env(
        monkeypatch,
        {
            "OPENAI_API_KEY": "oai-key",
            "LLM_MODEL": "deepseek/deepseek-v4-flash",
            "LLM_BASE_URL": "https://openrouter.ai/api/v1",
            "LLM_API_KEY": "or-key",
            "QUERY_LLM_MODEL": "gpt-5.4-mini",
            "QUERY_LLM_BASE_URL": "https://api.openai.com/v1",  # explicit -> OpenAI
        },
    )
    assert mod.LLM_MODEL == "deepseek/deepseek-v4-flash" and mod._LLM_IS_OPENAI is False
    assert mod.QUERY_LLM_MODEL == "gpt-5.4-mini"
    assert mod.QUERY_LLM_BASE_URL == "https://api.openai.com/v1"
    assert (
        mod.QUERY_LLM_API_KEY == "oai-key"
    )  # different endpoint than extraction -> OPENAI_API_KEY
    assert mod.QUERY_LLM_IS_OPENAI is True


def test_active_llm_cfg_switches_on_phase(monkeypatch):
    monkeypatch.setattr(server.config, "LLM_MODEL", "extract-model")
    monkeypatch.setattr(server.config, "LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(server.config, "LLM_API_KEY", "or-key")
    monkeypatch.setattr(server.config, "_LLM_IS_OPENAI", False)
    monkeypatch.setattr(server.config, "QUERY_LLM_MODEL", "query-model")
    monkeypatch.setattr(server.config, "QUERY_LLM_BASE_URL", None)
    monkeypatch.setattr(server.config, "QUERY_LLM_API_KEY", "oai-key")
    monkeypatch.setattr(server.config, "QUERY_LLM_IS_OPENAI", True)

    # Default (query) phase
    assert server.config._active_llm_cfg() == ("query-model", None, "oai-key", True)
    # Extract phase
    token = server.config._llm_phase.set("extract")
    try:
        assert server.config._active_llm_cfg() == (
            "extract-model",
            "https://openrouter.ai/api/v1",
            "or-key",
            False,
        )
    finally:
        server.config._llm_phase.reset(token)


@pytest.mark.asyncio
async def test_process_file_runs_under_extract_phase(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("hello world", encoding="utf-8")
    seen = {}

    async def _capture(content, ids=None, file_paths=None):
        seen["phase"] = server.config._llm_phase.get()

    rag_stub.lightrag.ainsert.side_effect = _capture
    try:
        await server._process_file(path, rag_stub)
    finally:
        rag_stub.lightrag.ainsert.side_effect = None
    assert seen["phase"] == "extract"
    # Phase is restored to the default after ingest completes.
    assert server.config._llm_phase.get() == "query"


@pytest.mark.asyncio
async def test_process_file_logs_extraction_model(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(server.config, "LLM_MODEL", "extract-model")
    path = tmp_path / "note.txt"
    path.write_text("hello world", encoding="utf-8")

    async def _noop(content, ids=None, file_paths=None):
        return None

    rag_stub.lightrag.ainsert.side_effect = _noop
    try:
        with caplog.at_level(logging.INFO):
            await server._process_file(path, rag_stub, file_path="note.txt")
    finally:
        rag_stub.lightrag.ainsert.side_effect = None
    # An INFO record names the extraction model and the file being ingested.
    msgs = [r.getMessage() for r in caplog.records]
    assert any("extract-model" in m and "note.txt" in m for m in msgs), msgs


def _install_fake_openai(monkeypatch, captured):
    class _FakeResp:
        def __init__(self):
            self.choices = [SimpleNamespace(message=SimpleNamespace(content="ok"))]

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            captured.append(("client", api_key, base_url))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        async def _create(self, model=None, messages=None, **kw):
            captured.append(("create", model))
            return _FakeResp()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=_FakeClient))


def _set_split_cfg(monkeypatch):
    monkeypatch.setattr(server.config, "LLM_MODEL", "extract-model")
    monkeypatch.setattr(server.config, "LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(server.config, "LLM_API_KEY", "or-key")
    monkeypatch.setattr(server.config, "_LLM_IS_OPENAI", False)
    monkeypatch.setattr(server.config, "QUERY_LLM_MODEL", "query-model")
    monkeypatch.setattr(server.config, "QUERY_LLM_BASE_URL", None)
    monkeypatch.setattr(server.config, "QUERY_LLM_API_KEY", "oai-key")
    monkeypatch.setattr(server.config, "QUERY_LLM_IS_OPENAI", True)


@pytest.mark.asyncio
async def test_llm_func_uses_query_cfg_by_default(monkeypatch):
    captured = []
    _install_fake_openai(monkeypatch, captured)
    _set_split_cfg(monkeypatch)
    await server.llm._llm_func("hi")
    assert ("client", "oai-key", None) in captured
    assert ("create", "query-model") in captured


@pytest.mark.asyncio
async def test_llm_func_uses_extract_cfg_in_extract_phase(monkeypatch):
    captured = []
    _install_fake_openai(monkeypatch, captured)
    _set_split_cfg(monkeypatch)
    token = server.config._llm_phase.set("extract")
    try:
        await server.llm._llm_func("hi")
    finally:
        server.config._llm_phase.reset(token)
    assert ("client", "or-key", "https://openrouter.ai/api/v1") in captured
    assert ("create", "extract-model") in captured


# --------------------------------------------------------------------------- #
# Smoke — per-role endpoint routing (embeddings / vision / whisper)
# Proves the .env-configurable endpoints (blank -> OpenAI; set -> that endpoint) actually
# reach the client constructor and the right model/token kwargs, for every role. Nothing
# is hardcoded to OpenAI. Companion to the phase-split LLM tests above.
# --------------------------------------------------------------------------- #


def _install_fake_openai_full(monkeypatch, captured):
    """Fake `openai` exposing embeddings + chat + audio, recording constructor kwargs
    and each method's model/kwargs into `captured`."""

    class _EmbResp:
        def __init__(self):
            self.data = [SimpleNamespace(embedding=[0.1, 0.2, 0.3])]

    class _ChatResp:
        def __init__(self):
            self.choices = [SimpleNamespace(message=SimpleNamespace(content="ok"))]

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            captured["init"] = {"api_key": api_key, "base_url": base_url}
            self.embeddings = SimpleNamespace(create=self._emb)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
            self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._transcribe))

        async def _emb(self, model=None, input=None, **kw):
            captured["emb"] = {"model": model, "input": input, **kw}
            return _EmbResp()

        async def _chat(self, model=None, messages=None, **kw):
            captured["chat"] = {"model": model, **kw}
            return _ChatResp()

        async def _transcribe(self, model=None, file=None, **kw):
            captured["transcribe"] = {"model": model, **kw}
            return "transcript text"

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=_FakeClient))


@pytest.mark.asyncio
async def test_embedding_func_routes_to_configured_endpoint(monkeypatch):
    import numpy as np

    captured = {}
    _install_fake_openai_full(monkeypatch, captured)
    monkeypatch.setattr(server.config, "EMBEDDING_BASE_URL", "http://local-embed:1234/v1")
    monkeypatch.setattr(server.config, "EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setattr(server.config, "EMBEDDING_MODEL", "bge-m3")
    out = await server.llm._embedding_func(["a", "b"])
    assert captured["init"] == {"api_key": "embed-key", "base_url": "http://local-embed:1234/v1"}
    assert captured["emb"]["model"] == "bge-m3"
    assert isinstance(out, np.ndarray)  # LightRAG calls .size on the result


@pytest.mark.asyncio
async def test_embedding_func_blank_base_url_is_openai(monkeypatch):
    captured = {}
    _install_fake_openai_full(monkeypatch, captured)
    monkeypatch.setattr(server.config, "EMBEDDING_BASE_URL", None)
    monkeypatch.setattr(server.config, "EMBEDDING_API_KEY", "oai-key")
    await server.llm._embedding_func(["x"])
    assert captured["init"] == {"api_key": "oai-key", "base_url": None}


@pytest.mark.asyncio
async def test_vision_func_routes_and_keeps_openai_tokens(monkeypatch):
    captured = {}
    _install_fake_openai_full(monkeypatch, captured)
    monkeypatch.setattr(server.config, "VISION_BASE_URL", None)
    monkeypatch.setattr(server.config, "VISION_API_KEY", "oai-key")
    monkeypatch.setattr(server.config, "VISION_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr(server.config, "_VISION_IS_OPENAI", True)
    await server.llm._vision_func("describe", max_completion_tokens=50)
    assert captured["init"] == {"api_key": "oai-key", "base_url": None}
    assert captured["chat"]["model"] == "gpt-5.4-mini"
    assert captured["chat"].get("max_completion_tokens") == 50
    assert "max_tokens" not in captured["chat"]


@pytest.mark.asyncio
async def test_vision_func_translates_tokens_for_non_openai(monkeypatch):
    captured = {}
    _install_fake_openai_full(monkeypatch, captured)
    monkeypatch.setattr(server.config, "VISION_BASE_URL", "http://local-vlm:1234/v1")
    monkeypatch.setattr(server.config, "VISION_API_KEY", "vlm-key")
    monkeypatch.setattr(server.config, "VISION_MODEL", "qwen2-vl")
    monkeypatch.setattr(server.config, "_VISION_IS_OPENAI", False)
    await server.llm._vision_func("describe", max_completion_tokens=50)
    assert captured["init"] == {"api_key": "vlm-key", "base_url": "http://local-vlm:1234/v1"}
    assert captured["chat"].get("max_tokens") == 50  # translated for non-OpenAI
    assert "max_completion_tokens" not in captured["chat"]


@pytest.mark.asyncio
async def test_extract_with_vision_routes_to_configured_endpoint(tmp_path, monkeypatch):
    captured = {}
    _install_fake_openai_full(monkeypatch, captured)
    monkeypatch.setattr(server.config, "VISION_BASE_URL", "http://local-vlm:1234/v1")
    monkeypatch.setattr(server.config, "VISION_API_KEY", "vlm-key")
    monkeypatch.setattr(server.config, "VISION_MODEL", "qwen2-vl")
    monkeypatch.setattr(server.config, "_VISION_IS_OPENAI", False)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    await server.ingest._extract_with_vision(pdf)
    assert captured["init"] == {"api_key": "vlm-key", "base_url": "http://local-vlm:1234/v1"}
    assert captured["chat"]["model"] == "qwen2-vl"
    assert captured["chat"].get("max_tokens") == 16000  # translated from the hardcoded cap


@pytest.mark.asyncio
async def test_transcribe_audio_routes_and_uses_configured_model(tmp_path, monkeypatch):
    captured = {}
    _install_fake_openai_full(monkeypatch, captured)
    monkeypatch.setattr(server.config, "WHISPER_BASE_URL", "http://local-whisper:9000/v1")
    monkeypatch.setattr(server.config, "WHISPER_API_KEY", "whisper-key")
    monkeypatch.setattr(server.config, "WHISPER_MODEL", "faster-whisper-large-v3")
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"RIFF....")
    out = await server.ingest._transcribe_audio(audio)
    assert captured["init"] == {
        "api_key": "whisper-key",
        "base_url": "http://local-whisper:9000/v1",
    }
    assert captured["transcribe"]["model"] == "faster-whisper-large-v3"
    assert out == "transcript text"


# --------------------------------------------------------------------------- #
# Token auth (API_TOKENS) — opt-in, Bearer for machines / Basic for browsers
# --------------------------------------------------------------------------- #


def _basic_header(username: str, password: str) -> str:
    import base64

    return "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()


@pytest.mark.asyncio
async def test_auth_disabled_by_default_allows(client):
    # No API_TOKENS configured (the default) => no auth, unchanged behaviour.
    resp = await client.post(f"{WS}/query", json={"query": "test"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_missing_credentials_401(client):
    server.API_TOKENS = ["testtok"]
    resp = await client.post(f"{WS}/query", json={"query": "test"})
    assert resp.status_code == 401
    # Browsers rely on this to show a native login prompt.
    assert resp.headers.get("www-authenticate", "").startswith("Basic")


@pytest.mark.asyncio
async def test_valid_bearer_200(client):
    server.API_TOKENS = ["testtok"]
    resp = await client.post(
        f"{WS}/query",
        json={"query": "test"},
        headers={"Authorization": "Bearer testtok"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_valid_basic_token_as_password_200(client):
    # Humans in a browser: any username + the token as the password.
    server.API_TOKENS = ["testtok"]
    resp = await client.post(
        f"{WS}/query",
        json={"query": "test"},
        headers={"Authorization": _basic_header("anyone", "testtok")},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_wrong_token_401(client):
    server.API_TOKENS = ["testtok"]
    resp = await client.post(
        f"{WS}/query",
        json={"query": "test"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_health_open_when_auth_enabled(client):
    server.API_TOKENS = ["testtok"]
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_docs_protected_when_auth_enabled(client):
    # The whole API is gated, including the auto-generated OpenAPI schema/docs.
    server.API_TOKENS = ["testtok"]
    resp = await client.get("/openapi.json")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_multiple_tokens_each_valid(client):
    server.API_TOKENS = ["tok1", "tok2"]
    for tok in ("tok1", "tok2"):
        resp = await client.post(
            f"{WS}/query",
            json={"query": "test"},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Low-hanging hardening — error hygiene, log level, input bounds
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_internal_error_not_leaked_to_client(client):
    # A raw upstream exception must not appear in the 500 body.
    rag_stub.lightrag.aquery_data.side_effect = RuntimeError("secret internal detail")
    try:
        resp = await client.post(f"{WS}/query/data", json={"query": "test"})
    finally:
        rag_stub.lightrag.aquery_data.side_effect = None
    assert resp.status_code == 500
    body = resp.text
    assert "secret internal detail" not in body


@pytest.mark.parametrize(
    "value,expected",
    [
        ("DEBUG", logging.DEBUG),
        ("info", logging.INFO),
        (None, logging.INFO),
        ("", logging.INFO),
        ("bogus", logging.INFO),
        ("WARNING", logging.WARNING),
        ("  Error  ", logging.ERROR),
    ],
)
def test_log_level_from_env(value, expected):
    assert server.config._log_level_from_env(value) == expected


@pytest.mark.asyncio
async def test_query_invalid_mode_422(client):
    resp = await client.post(f"{WS}/query", json={"query": "test", "mode": "bogus"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_top_k_over_ceiling_422(client):
    resp = await client.post(f"{WS}/query", json={"query": "test", "top_k": 100000})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_top_k_at_ceiling_ok(client):
    resp = await client.post(f"{WS}/query", json={"query": "test", "top_k": 1000})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_query_data_invalid_mode_422(client):
    resp = await client.post(f"{WS}/query/data", json={"query": "test", "mode": "bogus"})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# Reference real-path resolution (lightrag_key) + no-internal-leak
# --------------------------------------------------------------------------- #


def _meta_row(lightrag_key, file_path, file, **over):
    row = {
        "lightrag_key": lightrag_key,
        "file_path": file_path,
        "file": file,
        "job_id": None,
        "description": None,
        "source_path": None,
        "last_modified_time": None,
        "uploaded_at": None,
        "llm_model_extracted": None,
    }
    row.update(over)
    return row


@pytest.mark.asyncio
async def test_process_file_passes_lightrag_input_not_display(tmp_path):
    # We hand LightRAG the {job_id}_basename identity, never the caller's real path.
    path = tmp_path / "job1_report.txt"
    path.write_text("hello world content here")
    rag_stub.lightrag.ainsert.reset_mock()
    await server._process_file(path, rag_stub, file_path="job1_report.txt")
    assert rag_stub.lightrag.ainsert.call_args.kwargs.get("file_paths") == ["job1_report.txt"]


@pytest.mark.asyncio
async def test_process_file_captures_stored_key_from_docstatus(tmp_path):
    # The join key is READ BACK from LightRAG's doc-status (its own stored value), not assumed.
    path = tmp_path / "j_r.txt"
    path.write_text("hello world content here")

    class _Docs(dict):
        def get(self, k, d=None):
            return {"status": "processed", "content_length": 5, "file_path": "canon_r.txt"}

    rag_stub.lightrag.aget_docs_by_ids.return_value = _Docs()
    try:
        doc_id, key = await server._process_file(path, rag_stub, file_path="j_r.txt")
    finally:
        rag_stub.lightrag.aget_docs_by_ids.return_value = _AnyKeyDocs()
    assert key == "canon_r.txt"  # LightRAG's stored value, not our input "j_r.txt"


@pytest.mark.asyncio
async def test_lightrag_key_persisted_on_done(tmp_path):
    server._db_pool = _mock_pool
    _mock_pool.execute.reset_mock()
    path = tmp_path / "aaa_f.txt"
    path.write_text("content")
    job_id = "keyjob"
    server._jobs[job_id] = {
        "job_id": job_id,
        "file": "f.txt",
        "workspace": "alex",
        "status": "pending",
        "attempts": 0,
        "error": None,
        "batch_id": "b",
    }
    with (
        patch.object(
            server, "_process_file", new=AsyncMock(return_value=("doc-1", "keyjob_f.txt"))
        ),
        patch.object(server, "get_workspace_rag", new=AsyncMock(return_value=rag_stub)),
    ):
        await server._process_job("alex", job_id, path, "", None)
    key_calls = [c for c in _mock_pool.execute.call_args_list if "SET lightrag_key" in str(c)]
    assert key_calls, "expected an UPDATE ... SET lightrag_key"
    assert key_calls[0].args[2] == "keyjob_f.txt"


@pytest.mark.asyncio
async def test_reference_resolves_basename_key_to_real_path(client):
    # LightRAG returns its internal basename key; we resolve it to the real path + enrichment.
    server._db_pool = _mock_pool
    _mock_pool.fetch.reset_mock()
    _mock_pool.fetch.return_value = [
        _meta_row("ab12_cv.pdf", "/corpus/career/cv.pdf", "cv.pdf", job_id="ab12", description="CV")
    ]
    rag_stub.lightrag.aquery_llm.return_value = {
        "status": "success",
        "data": {"references": [{"reference_id": "1", "file_path": "ab12_cv.pdf"}]},
        "llm_response": {"content": "a", "is_streaming": False},
        "metadata": {},
    }
    ref = (await client.post(f"{WS}/query", json={"query": "t"})).json()["references"][0]
    assert ref["file_path"] == "/corpus/career/cv.pdf"
    assert "file_name" not in ref  # file_name is no longer emitted at all
    assert ref["job_id"] == "ab12" and ref["file_description"] == "CV"
    assert "lightrag_key" not in ref and "ab12_cv.pdf" not in str(ref)
    _mock_pool.fetch.return_value = []


@pytest.mark.asyncio
async def test_reference_basename_collision_distinct_paths(client):
    # Two files with the same basename resolve to their OWN real paths (unique-key by design).
    server._db_pool = _mock_pool
    _mock_pool.fetch.reset_mock()
    _mock_pool.fetch.return_value = [
        _meta_row("job1_report.pdf", "/corpus/2024/report.pdf", "report.pdf", job_id="job1"),
        _meta_row("job2_report.pdf", "/corpus/2025/report.pdf", "report.pdf", job_id="job2"),
    ]
    rag_stub.lightrag.aquery_llm.return_value = {
        "status": "success",
        "data": {
            "references": [
                {"reference_id": "1", "file_path": "job1_report.pdf"},
                {"reference_id": "2", "file_path": "job2_report.pdf"},
            ]
        },
        "llm_response": {"content": "a", "is_streaming": False},
        "metadata": {},
    }
    refs = (await client.post(f"{WS}/query", json={"query": "t"})).json()["references"]
    assert refs[0]["file_path"] == "/corpus/2024/report.pdf"
    assert refs[1]["file_path"] == "/corpus/2025/report.pdf"
    _mock_pool.fetch.return_value = []


@pytest.mark.asyncio
async def test_files_endpoint_hides_lightrag_key(client):
    server._db_pool = _mock_pool
    _mock_pool.fetch.reset_mock()
    _mock_pool.fetch.return_value = [
        {
            "job_id": "j",
            "file": "f.txt",
            "file_path": "/corpus/f.txt",
            "source_path": None,
            "doc_id": "doc-1",
            "content_hash": "h",
            "status": "done",
            "last_modified_time": None,
            "uploaded_at": None,
        }
    ]
    files = (await client.get(f"{WS}/files")).json()["files"]
    assert files and "lightrag_key" not in files[0]
    assert files[0]["file_path"] == "/corpus/f.txt"
    _mock_pool.fetch.return_value = []


def test_safe_ref_name_strips_directories():
    assert server._safe_ref_name("a/b/c.txt") == "c.txt"
    assert server._safe_ref_name("..\\x\\y.pdf") == "y.pdf"
    assert server._safe_ref_name("plain.txt") == "plain.txt"
    assert server._safe_ref_name(None) == ""


def test_strip_job_prefix():
    assert server.references._strip_job_prefix("20ed9a7c_Tag_1.pdf") == "Tag_1.pdf"
    assert server.references._strip_job_prefix("beefcafe_notes.txt") == "notes.txt"
    assert (
        server.references._strip_job_prefix("no_prefix_here.txt") == "no_prefix_here.txt"
    )  # 'no' isn't hex-run
    assert server.references._strip_job_prefix("plain.txt") == "plain.txt"
    assert server.references._strip_job_prefix("") == ""


def test_rewrite_answer_refs_resolved_and_unresolved():
    meta = {"20ed9a7c_Tag_1.pdf": {"file": "sub/Tag_1.pdf"}}
    raw = [{"file_path": "20ed9a7c_Tag_1.pdf"}, {"file_path": "beefcafe_notes.txt"}]
    prose = "Body.\n### References\n- [1] 20ed9a7c_Tag_1.pdf\n- [2] beefcafe_notes.txt"
    out = server.references._rewrite_answer_refs(prose, raw, meta)
    assert "20ed9a7c_Tag_1.pdf" not in out and "beefcafe_notes.txt" not in out
    assert "[1] Tag_1.pdf" in out  # resolved -> clean basename (dir part dropped)
    assert "[2] notes.txt" in out  # unresolved -> hex prefix stripped
    # empty / no-refs are no-ops
    assert server.references._rewrite_answer_refs("", raw, meta) == ""
    assert server.references._rewrite_answer_refs("text", [], {}) == "text"


def test_rewrite_answer_refs_no_partial_substring_hit():
    # A key that is a substring of a longer key must not be rewritten inside the longer one.
    meta = {}
    raw = [{"file_path": "aa11beef_x.txt"}, {"file_path": "aa11beef_x.txt.bak_extra"}]
    prose = "[1] aa11beef_x.txt.bak_extra and [2] aa11beef_x.txt"
    out = server.references._rewrite_answer_refs(prose, raw, meta)
    # longest-first replacement keeps the longer token intact as its own stripped form
    assert "aa11beef_x.txt.bak_extra" not in out
    assert "x.txt.bak_extra" in out  # longer key stripped whole
    assert out.count("x.txt") >= 2


def test_job_path_sanitizes_separators_no_traversal(tmp_path, monkeypatch):
    # A filename carrying a path separator must not create subdirs or escape the
    # workspace dir — the on-disk name is basenamed just like the LightRAG key.
    monkeypatch.setattr(server, "WORKING_DIR", str(tmp_path))
    p = server._job_path("test", "deadbeef", "nav/watchtower[spec].txt")
    assert p.parent == tmp_path / "test"  # stays inside the workspace dir
    assert p.name == "deadbeef_watchtower[spec].txt"  # separator stripped, job_id intact
    # a traversal attempt is neutralised to a plain basename under the workspace dir
    ev = server._job_path("test", "deadbeef", "../../etc/passwd")
    assert ev.parent == tmp_path / "test"
    assert ev.name == "deadbeef_passwd"


# --------------------------------------------------------------------------- #
# file_path_contains — blank/empty means "no filter → all data"
# --------------------------------------------------------------------------- #


def test_path_matches_any_blank_needles_keep_all():
    assert server.references._path_matches_any("/x/y.pdf", [""]) is True
    assert server.references._path_matches_any("/x/y.pdf", ["  "]) is True
    assert server.references._path_matches_any("/x/y.pdf", ["", "  "]) is True
    assert server.references._path_matches_any(None, [""]) is True
    # a real needle still filters; blanks alongside a real needle are ignored
    assert server.references._path_matches_any("/x/y.pdf", ["/z/"]) is False
    assert server.references._path_matches_any("/x/y.pdf", ["", "/x/"]) is True


@pytest.mark.asyncio
async def test_query_data_blank_filter_returns_all(client):
    rag_stub.lightrag.aquery_data.return_value = {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [
                {"entity_name": "A", "file_path": "/a"},
                {"entity_name": "B", "file_path": "/b"},
            ],
            "relationships": [],
            "chunks": [],
            "references": [],
        },
        "metadata": {},
    }
    # Swagger's auto-populated [""] must return ALL data, not an empty set.
    resp = await client.post(f"{WS}/query/data", json={"query": "t", "file_path_contains": [""]})
    ents = [e["entity_name"] for e in resp.json()["data"]["entities"]]
    assert ents == ["A", "B"]
    # and the boost is not triggered by a blank-only filter
    assert lightrag_mod.QueryParam.call_args.kwargs.get("top_k") == 40
