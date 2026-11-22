"""End-to-end journeys through the PolyGraphRAG HTTP API.

Where ``test_server.py`` verifies each endpoint in isolation, this module drives the *real*
ASGI app through complete, multi-step journeys — the flows a real client actually performs —
asserting that the output of one call correctly feeds the next:

  * ingest → track → query lifecycle (upload → background worker → status/batch → query/query-data)
  * workspace registry lifecycle (create → list → status → soft-delete → restore) against a
    stateful in-memory stand-in for the ``rag_workspaces`` registry
  * an auth-gated journey: with ``API_TOKENS`` set, the same journey is blocked without a
    Bearer token and succeeds with one

RAGAnything, LightRAG and Postgres are stubbed exactly as the unit suite stubs them, so these
run fast with no container, DB, or API keys. The live-container equivalent is
``scripts/smoke_test_docker.sh``.
"""

import asyncio
import base64
import gzip
import hashlib
import io
import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest
import pytest_asyncio

# Import the unit-test module first so its raganything/lightrag stubs are the ones registered in
# sys.modules, and reuse the very same RAGAnything stub. Registering a *second*, competing set of
# stubs here would shadow the QueryParam object test_server introspects (whichever module imports
# first wins), so we deliberately share one set instead of duplicating it.
import test_server as _ts  # noqa: E402
from httpx import ASGITransport, AsyncClient
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

import server  # noqa: E402
from server import config, graph_viewer, mcp_viewer

rag_stub = _ts.rag_stub
_AnyKeyDocs = _ts._AnyKeyDocs
_AQUERY_LLM = _ts._DEFAULT_AQUERY_LLM
_AQUERY_DATA = _ts._DEFAULT_AQUERY_DATA

# --------------------------------------------------------------------------- #
# Fixtures & helpers
# --------------------------------------------------------------------------- #
WS = "/workspace/alex"  # primary public workspace → physical "alex" via the lookup stub
MCP_URL = "http://polygraphrag:9622"
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}
READ_TOOLS = {
    "ask_workspace",
    "find_graph_entities",
    "get_embedding_consistency",
    "get_graph_entity",
    "get_graph_neighbors",
    "get_graph_relationship",
    "get_ingest_job",
    "get_workspace_overview",
    "list_workspaces",
    "retrieve_graph",
    "search_workspace",
    "show_workspace_graph",
}
WRITE_TOOLS = {"write_graph_batch", "patch_graph_entity", "patch_graph_relationship"}


@asynccontextmanager
async def _mcp_client(headers=None):
    manager = server.create_session_manager()
    server.app.state.mcp_manager = manager
    async with (
        manager.run(),
        AsyncClient(
            transport=ASGITransport(app=server.app), base_url=MCP_URL, headers=headers
        ) as http,
        streamable_http_client(f"{MCP_URL}/mcp", http_client=http) as (reader, writer, _),
        ClientSession(reader, writer) as agent,
    ):
        await agent.initialize()
        yield http, agent


def _mcp_error(result):
    assert result.isError
    return json.loads(result.content[0].text)


def _mcp_request(method, **params):
    return {"jsonrpc": "2.0", "id": 5, "method": method, "params": params}


async def _fake_lookup_workspace(workspace_id):
    """Every workspace resolves as present, active, and physically named after its id."""
    return {
        "id": workspace_id,
        "name": workspace_id,
        "description": None,
        "lightrag_workspace": workspace_id,
    }


@pytest.fixture(autouse=True)
def reset_server_state():
    """Isolate package-level mutable state between tests (this module has its own copy of the
    autouse reset that test_server.py applies to itself)."""
    server._jobs.clear()
    server._batches.clear()
    while not server._job_queue.empty():
        try:
            server._job_queue.get_nowait()
            server._job_queue.task_done()
        except asyncio.QueueEmpty:
            break
    server._db_pool = None
    if hasattr(server, "_rag_instances"):
        server._rag_instances.clear()
    if hasattr(server, "_ws_locks"):
        server._ws_locks.clear()
    for m in (
        rag_stub.lightrag.aquery_llm,
        rag_stub.lightrag.aquery_data,
        rag_stub.lightrag.aget_docs_by_ids,
        rag_stub.lightrag.adelete_by_doc_id,
    ):
        m.reset_mock()
    rag_stub.lightrag.aquery_llm.return_value = _AQUERY_LLM
    rag_stub.lightrag.aquery_data.return_value = _AQUERY_DATA
    rag_stub.lightrag.aget_docs_by_ids.return_value = _AnyKeyDocs()
    server.API_TOKENS = []
    yield
    server._db_pool = None
    server.API_TOKENS = []


@pytest_asyncio.fixture
async def client():
    """Async httpx client over the real ASGI app, with the workspace registry lookup and the
    per-workspace RAG instance stubbed so ``/workspace/{id}/...`` routes resolve without a DB."""
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


def _upload(filename: str, content: bytes = b"hello") -> tuple:
    return ("files", (filename, io.BytesIO(content), "application/octet-stream"))


def _basic(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


async def _drain_queue(process_result=None):
    """Run the real worker over every queued job, with ingestion + instance lookup stubbed.
    ``process_result`` is what the stubbed ``_process_file`` returns (a ``(doc_id, key)`` tuple
    on success, or an ``Exception`` side-effect to force failure)."""
    pf = (
        AsyncMock(side_effect=process_result)
        if isinstance(process_result, Exception)
        else AsyncMock(return_value=process_result)
    )
    with (
        patch.object(server, "_process_file", pf),
        patch.object(server, "get_workspace_rag", AsyncMock(return_value=rag_stub)),
    ):
        while not server._job_queue.empty():
            ws, job_id, dest, metadata, fp = await server._job_queue.get()
            try:
                await server._process_job(ws, job_id, dest, metadata, fp)
            finally:
                server._job_queue.task_done()


class _StatefulRegistryPool:
    """A minimal, stateful stand-in for the asyncpg pool backing ``rag_workspaces``.

    It tracks created workspace rows in memory and routes the registry endpoints' SQL by
    substring, so a create→list→status→delete→restore journey observes its own effects — the
    thing purpose-built single-answer mocks in the unit suite deliberately don't do. Only the
    ``rag_workspaces`` surface is modelled; corpus-count queries return empty/zero.
    """

    def __init__(self):
        self._rows: dict[str, dict] = {}

    async def execute(self, sql: str, *args):
        if "SET deleted_at = NOW()" in sql:
            self._rows[args[0]]["deleted_at"] = datetime.now(timezone.utc)
        elif "SET deleted_at = NULL" in sql:
            self._rows[args[0]]["deleted_at"] = None
        elif "DELETE FROM rag_workspaces" in sql:
            self._rows.pop(args[0], None)

    async def fetchrow(self, sql: str, *args):
        if "INSERT INTO rag_workspaces" in sql:
            wid, name, description, physical = args
            self._rows[wid] = {
                "id": wid,
                "name": name,
                "description": description,
                "lightrag_workspace": physical,
                "is_primary": False,
                "deleted_at": None,
                "created_at": datetime.now(timezone.utc),
            }
            return dict(self._rows[wid])
        if "FROM rag_workspaces WHERE id = $1" in sql:
            row = self._rows.get(args[0])
            if row and "deleted_at IS NULL" in sql and row["deleted_at"] is not None:
                return None
            return dict(row) if row else None
        return None

    async def fetch(self, sql: str, *args):
        if "FROM rag_workspaces" in sql:
            deleted = "deleted_at IS NOT NULL" in sql
            return [
                dict(r) for r in self._rows.values() if (r["deleted_at"] is not None) == deleted
            ]
        return []  # doc_status / file_metadata group-by queries: nothing ingested

    async def fetchval(self, sql: str, *args):
        return None  # chunk sums, last-upload, and vdb table discovery all resolve to empty


async def test_mcp_agent_discovery_search_and_cited_answer(monkeypatch, tmp_path):
    pool = _StatefulRegistryPool()
    pool.close = AsyncMock()
    monkeypatch.setitem(
        sys.modules, "asyncpg", SimpleNamespace(create_pool=AsyncMock(return_value=pool))
    )
    monkeypatch.setattr(server, "WORKING_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_db_init", AsyncMock())
    monkeypatch.setattr(server, "_db_reload_jobs", AsyncMock())
    monkeypatch.setattr(server, "_worker", AsyncMock())
    monkeypatch.setattr(server, "API_TOKENS", ["agent-token"])
    monkeypatch.setattr(config, "MCP_GRAPH_WRITES", False)

    # Both workspaces use the same internal citation key; only the scoped metadata differs.
    key = "abc123_refunds.md"

    async def scoped_metadata(pool, keys, phys):
        return {key: {"file": "refunds.md", "file_path": f"/{phys}/refunds.md"}}

    monkeypatch.setattr(server.references, "_db_fetch_metadata_by_key", scoped_metadata)

    async def get_instance(wid):
        rag = MagicMock()
        references = [{"reference_id": "1", "file_path": key}]
        rag.lightrag.aquery_data = AsyncMock(
            return_value={
                "data": {"references": references, "chunks": [{"content": wid, "file_path": key}]}
            }
        )
        rag.lightrag.aquery_llm = AsyncMock(
            return_value={
                "data": {"references": references},
                "llm_response": {"content": f"{wid} cites {key}"},
            }
        )
        return rag

    monkeypatch.setattr(server, "get_workspace_rag", get_instance)
    url = "http://polygraphrag:9622"
    async with (
        server.lifespan(server.app),
        AsyncClient(
            transport=ASGITransport(app=server.app),
            base_url=url,
            headers={"Authorization": "Bearer agent-token"},
        ) as http,
        streamable_http_client(f"{url}/mcp", http_client=http) as (reader, writer, _),
        ClientSession(reader, writer) as agent,
    ):
        for wid in ("alpha", "beta", "retired"):
            assert (
                await http.post("/all-workspaces/create", json={"id": wid, "name": wid})
            ).is_success
        assert (await http.delete("/workspace/retired")).is_success
        await agent.initialize()
        tools = (await agent.list_tools()).tools
        assert {t.name for t in tools} == READ_TOOLS
        assert all(t.annotations.readOnlyHint for t in tools)
        assert _mcp_error(await agent.call_tool("write_graph_batch", {}))["code"] == "unknown_tool"
        listed = await agent.call_tool("list_workspaces", {})
        assert listed.structuredContent == (await http.get("/all-workspaces/list")).json()
        assert [w["id"] for w in listed.structuredContent["workspaces"]] == ["alpha", "beta"]
        for wid in ("alpha", "beta"):
            overview = await agent.call_tool("get_workspace_overview", {"workspace_id": wid})
            assert overview.structuredContent == (await http.get(f"/workspace/{wid}")).json()
            question = {"query": "What are the refund terms?"}
            search = {**question, "file_path_contains": [f"/{wid}/"]}
            evidence = await agent.call_tool("search_workspace", {"workspace_id": wid, **search})
            rest_search = await http.post(f"/workspace/{wid}/query/data", json=search)
            assert evidence.structuredContent == rest_search.json()
            assert evidence.structuredContent["data"]["chunks"] == [
                {"content": wid, "file_path": f"/{wid}/refunds.md"}
            ]
            answer = await agent.call_tool("ask_workspace", {"workspace_id": wid, **question})
            rest_answer = await http.post(f"/workspace/{wid}/query", json=question)
            assert answer.structuredContent == rest_answer.json()
            assert answer.structuredContent["result"] == f"{wid} cites refunds.md"
            assert answer.structuredContent["references"][0]["file_path"] == f"/{wid}/refunds.md"
        for invalid_args in (
            {"workspace_id": "retired"},
            {"workspace_id": "unknown"},
            {"workspace_id": "alpha", "top_k": 0},
            {},
        ):
            failure = await agent.call_tool("ask_workspace", {"query": "refunds", **invalid_args})
            assert failure.isError
        for wid in ("retired", "unknown"):
            for delivery in ("auto", "html", "link"):
                failure = await agent.call_tool(
                    "show_workspace_graph", {"workspace_id": wid, "delivery": delivery}
                )
                assert _mcp_error(failure)["code"] == "workspace_not_found"


async def _raw_response(http, method, path, coding, **kwargs):
    async with http.stream(
        method, path, headers=MCP_HEADERS | {"Accept-Encoding": coding}, **kwargs
    ) as response:
        response.raise_for_status()
        wire = b"".join([chunk async for chunk in response.aiter_raw()])
        assert int(response.headers["Content-Length"]) == len(wire)
        return response.headers, wire


async def test_mcp_ingest_job_is_workspace_scoped_and_omits_private_error(monkeypatch, client):
    def failed_job(workspace):
        return {
            "workspace": workspace,
            "status": "failed",
            "attempts": 2,
            "error": "private database exception",
            "job_id": "persisted",
            "file": "original.md",
            "uploaded_at": datetime.now(timezone.utc),
        }

    async def fetchrow(_sql, job_id, workspace):
        job = failed_job("alex")
        return job if (job_id, workspace) == ("persisted", job["workspace"]) else None

    monkeypatch.setattr(server, "_db_pool", SimpleNamespace(fetchrow=fetchrow))
    server._jobs["memory"] = failed_job("alex")
    async with _mcp_client() as (_, agent):
        for job_id in ("memory", "persisted"):
            result = await agent.call_tool(
                "get_ingest_job", {"workspace_id": "alex", "job_id": job_id}
            )
            assert not result.isError
            assert result.structuredContent == {"job_id": job_id, "status": "failed", "attempts": 2}
            assert "private database exception" not in result.model_dump_json()
            failure = await agent.call_tool(
                "get_ingest_job", {"workspace_id": "other", "job_id": job_id}
            )
            assert _mcp_error(failure)["code"] == "job_not_found"


async def test_mcp_upload_uses_shared_ingestion_and_citation_path(monkeypatch, tmp_path, client):
    monkeypatch.setattr(config, "MCP_FILE_UPLOADS", True)
    monkeypatch.setattr(config, "MCP_GRAPH_WRITES", False)
    monkeypatch.setattr(config, "MCP_UPLOAD_MAX_BYTES", 1024)
    monkeypatch.setattr(server, "WORKING_DIR", str(tmp_path))
    monkeypatch.setattr(server, "API_TOKENS", ["agent-token"])
    raw_file = b"x" * 1024
    file = {
        "filename": "field-notes.MD",
        "content_base64": base64.b64encode(raw_file).decode(),
        "source_path": "journals/field-notes.MD",
        "path_root": "https://example.org/archives",
    }
    args = {"workspace_id": "alex", "files": [file]}
    async with _mcp_client({"Authorization": "Bearer agent-token"}) as (http, agent):
        tools = {t.name: t for t in (await agent.list_tools()).tools}
        assert set(tools) == READ_TOOLS | {"upload_documents"}
        definition = tools["upload_documents"]
        assert definition.annotations.readOnlyHint is False
        assert "1024 bytes" in definition.description
        denied = await http.post(
            "/mcp",
            json=_mcp_request("tools/call", name="upload_documents", arguments=args),
            headers=MCP_HEADERS | {"Authorization": ""},
        )
        assert denied.status_code == 401
        assert server._job_queue.empty()

        result = await agent.call_tool("upload_documents", args)
        assert not result.isError
        batch = result.structuredContent
        assert batch["summary"] == {"pending": 1, "total": 1}
        job = batch["jobs"][0]
        job_args = {"workspace_id": "alex", "job_id": job["job_id"]}
        assert job["file_path"] == "https://example.org/archives/journals/field-notes.MD"
        assert job["status"] == "pending"
        assert file["content_base64"] not in result.model_dump_json()
        assert next(tmp_path.rglob("*.MD")).read_bytes() == raw_file
        rest = (await http.get(f"{WS}/batch/{batch['batch_id']}")).json()["jobs"][0]
        assert rest["job_id"] == job["job_id"]
        assert rest["content_hash"] == hashlib.sha256(raw_file).hexdigest()
        await _drain_queue(process_result=("doc-1", "key-1"))
        tracked = await agent.call_tool("get_ingest_job", job_args)
        assert tracked.structuredContent == {
            "job_id": job["job_id"],
            "status": "done",
            "attempts": 0,
        }
        denied = await agent.call_tool("get_ingest_job", job_args | {"workspace_id": "other"})
        assert _mcp_error(denied)["code"] == "job_not_found"


async def test_mcp_upload_limits_and_disabled_catalog(monkeypatch, tmp_path, client):
    monkeypatch.setattr(config, "MCP_FILE_UPLOADS", False)
    monkeypatch.setattr(config, "MCP_UPLOAD_MAX_BYTES", 1024)
    monkeypatch.setattr(server, "WORKING_DIR", str(tmp_path))

    def document(size=1, **fields):
        return {
            "filename": "example.txt",
            "content_base64": base64.b64encode(b"a" * size).decode(),
        } | fields

    async with _mcp_client() as (_, agent):
        assert "upload_documents" not in {t.name for t in (await agent.list_tools()).tools}
        result = await agent.call_tool(
            "upload_documents", {"workspace_id": "alex", "files": [document()]}
        )
        assert _mcp_error(result)["code"] == "unknown_tool"

    monkeypatch.setattr(config, "MCP_FILE_UPLOADS", True)
    async with _mcp_client() as (_, agent):
        for files, code in (
            ([document(1025)], "document_upload_too_large"),
            ([document(999), document(26)], "document_upload_too_large"),
            ([document(2048)], "document_upload_too_large"),
            ([document()] * 9, "too_many_documents"),
            ([document(0)], "empty_document"),
            ([document(), document(content_base64="bad!")], "invalid_document_base64"),
            ([document(filename="../other.txt")], "invalid_document_filename"),
            ([document(filename="backup.zip")], "unsupported_document_type"),
        ):
            result = await agent.call_tool(
                "upload_documents", {"workspace_id": "alex", "files": files}
            )
            error = _mcp_error(result)
            assert error["code"] == code
            if error["status"] == 413:
                assert error["limit_bytes"] == 1024
    assert not server._jobs and server._job_queue.empty() and not list(tmp_path.iterdir())


async def test_mcp_upload_reports_only_persisted_jobs(monkeypatch, tmp_path, client):
    monkeypatch.setattr(config, "MCP_FILE_UPLOADS", True)
    monkeypatch.setattr(server, "WORKING_DIR", str(tmp_path))
    server._db_pool = SimpleNamespace(
        execute=AsyncMock(side_effect=[RuntimeError("internal database error"), None])
    )
    files = [
        {"filename": name, "content_base64": "aGVsbG8="} for name in ("failed.txt", "queued.txt")
    ]
    async with _mcp_client() as (_, agent):
        result = await agent.call_tool("upload_documents", {"workspace_id": "alex", "files": files})
    assert not result.isError
    jobs = result.structuredContent["jobs"]
    assert jobs[0]["status"] == "save_failed" and jobs[0]["job_id"] is None
    assert jobs[1]["status"] == "pending" and jobs[1]["job_id"] in server._jobs
    assert set(server._jobs) == {jobs[1]["job_id"]}
    assert server._job_queue.qsize() == 1
    assert [p.name for p in tmp_path.rglob("*.txt")] == [f"{jobs[1]['job_id']}_queued.txt"]
    assert "internal database error" not in result.model_dump_json()


async def test_mcp_gzip_upload_and_transport_errors(monkeypatch, tmp_path, client):
    monkeypatch.setattr(config, "MCP_FILE_UPLOADS", True)
    monkeypatch.setattr(config, "MCP_UPLOAD_MAX_BYTES", 4 * 1024 * 1024)
    monkeypatch.setattr(server, "WORKING_DIR", str(tmp_path))
    raw_file = b"t" * (4 * 1024 * 1024)
    args = {
        "workspace_id": "alex",
        "files": [{"filename": "notes.txt", "content_base64": base64.b64encode(raw_file).decode()}],
    }
    request = _mcp_request("tools/call", name="upload_documents", arguments=args)
    compressed = gzip.compress(json.dumps(request).encode())

    async with _mcp_client() as (http, agent):
        headers = MCP_HEADERS | {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "Accept-Encoding": "identity",
        }
        for wire, encoding, status, code in (
            (b"malformed", "gzip", 400, "invalid_gzip_request"),
            (compressed[:-1], "gzip", 400, "invalid_gzip_request"),
            (compressed + compressed, "gzip", 400, "invalid_gzip_request"),
            (compressed, "br", 415, "unsupported_content_encoding"),
            (gzip.compress(b" " * (7 * 1024 * 1024)), "gzip", 413, "mcp_request_too_large"),
        ):
            response = await http.post(
                "/mcp", headers=headers | {"Content-Encoding": encoding}, content=wire
            )
            assert response.status_code == status
            assert response.json()["detail"]["code"] == code
        assert server._job_queue.empty()

        async def chunks():
            for pos in range(0, len(compressed), 7):
                yield compressed[pos : pos + 7]

        response = await http.post("/mcp", headers=headers, content=chunks())
        assert response.status_code == 200
        assert "Content-Encoding" not in response.headers
        job_id = response.json()["result"]["structuredContent"]["jobs"][0]["job_id"]
        assert (tmp_path / "alex" / f"{job_id}_notes.txt").read_bytes() == raw_file
        job = await agent.call_tool("get_ingest_job", {"workspace_id": "alex", "job_id": job_id})
        assert job.structuredContent["status"] == "pending"


async def test_mcp_writes_flat_patch_contract_and_auth(monkeypatch, client):
    monkeypatch.setattr(config, "MCP_GRAPH_WRITES", True)
    monkeypatch.setattr(server, "API_TOKENS", ["agent-token"])
    common = {
        "description": "same",
        "type": "Fact",
        "revision": 2,
        "origins": ["manual"],
        "sources": [],
    }
    entity = {"id": "clocktower", "name": "Wren Clocktower", **common}
    edge = {
        "id": "storage",
        "source": "clocktower",
        "target": "map",
        "direction": "directed",
        **common,
    }
    record = AsyncMock(side_effect=[(entity, False), (entity, False), (edge, False), (edge, False)])
    monkeypatch.setattr(server.mcp, "write_graph_record", record)
    monkeypatch.setattr(server.routers.graph, "write_graph_record", record)
    manager = server.create_session_manager()
    server.app.state.mcp_manager = manager
    async with (
        manager.run(),
        AsyncClient(
            transport=ASGITransport(app=server.app), base_url=MCP_URL, headers=MCP_HEADERS
        ) as http,
    ):
        listing = _mcp_request("tools/list")
        headers = {"Authorization": "Bearer agent-token"}
        listed = await http.post("/mcp", json=listing, headers=headers)
        definitions = {tool["name"]: tool for tool in listed.json()["result"]["tools"]}
        assert set(definitions) == READ_TOOLS | WRITE_TOOLS
        for name in WRITE_TOOLS:
            assert definitions[name]["annotations"]["readOnlyHint"] is False
            assert definitions[name]["annotations"]["destructiveHint"] is True

        for name, id_field, item, path in (
            ("patch_graph_entity", "entity_id", entity, "entities"),
            ("patch_graph_relationship", "edge_id", edge, "relationships"),
        ):
            assert "expected_revision" in definitions[name]["inputSchema"]["required"]
            arguments = {"workspace_id": "alex", id_field: item["id"], "expected_revision": 2}
            for edit in ({}, {"unknown": False}, {"description": None}):
                reply = await http.post(
                    "/mcp",
                    json=_mcp_request("tools/call", name=name, arguments=arguments | edit),
                    headers=headers,
                )
                assert reply.json()["result"]["isError"]

            changes = {"description": "same", "source_ref": None}
            request = _mcp_request("tools/call", name=name, arguments=arguments | changes)
            before = record.await_count
            exclude = {"Accept-Encoding": "identity;q=0, gzip;q=0"}
            assert (await http.post("/mcp", json=request, headers=exclude)).status_code == 401
            rejected = await http.post("/mcp", json=request, headers=headers | exclude)
            assert rejected.status_code == 406
            assert rejected.content == b""
            assert rejected.headers["Vary"] == "Accept-Encoding"
            assert record.await_count == before

            reply = await http.post("/mcp", json=request, headers=headers)
            rest = await http.patch(
                f"{WS}/graph/{path}/{item['id']}",
                json={"expected_revision": 2, **changes},
                headers=headers,
            )
            assert reply.json()["result"]["structuredContent"] == rest.json()
            for call in record.await_args_list[before:]:
                assert call.args[3].model_dump(exclude_unset=True) == {
                    "expected_revision": 2,
                    **changes,
                }


async def test_disabled_graph_viewer_reports_configuration_in_rest_and_mcp(monkeypatch, client):
    monkeypatch.setattr(config, "GRAPH_VIEWER_ENABLED", False)
    monkeypatch.setattr(config, "MCP_VIEWER_BASE_URL", "https://example.org/rag")

    async with _mcp_client() as (http, agent):
        assert "show_workspace_graph" in {tool.name for tool in (await agent.list_tools()).tools}
        with monkeypatch.context() as disabled:
            lookup = AsyncMock(side_effect=AssertionError("Viewer should skip workspace lookup"))
            disabled.setattr(server.workspaces, "_lookup_workspace", lookup)
            rest = await http.get(f"{WS}/graph.html")
            assert rest.status_code == 403
            detail = rest.json()["detail"]
            assert detail["code"] == "graph_viewer_disabled"
            assert "GRAPH_VIEWER_ENABLED=false" in detail["message"]

            for delivery in ("auto", "html", "link"):
                result = await agent.call_tool(
                    "show_workspace_graph", {"workspace_id": "alex", "delivery": delivery}
                )
                assert _mcp_error(result) == {"status": 403, **detail}
                assert len(result.content) == 1
                assert result.structuredContent is None
                assert result.meta is None
            lookup.assert_not_awaited()

        search = {"query": "refunds"}
        evidence = await agent.call_tool("search_workspace", {"workspace_id": "alex", **search})
        rest_search = await http.post(f"{WS}/query/data", json=search)
        assert not evidence.isError and rest_search.status_code == 200
        assert evidence.structuredContent == rest_search.json()


def _viewer_fixture():
    return SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id="original",
                labels=["Chart"],
                properties={"entity_id": "Original </script>", "file_path": "/café/original.md"},
            )
        ],
        edges=[],
    )


@pytest.mark.parametrize("editable", [False, True])
async def test_mcp_graph_app_modes_and_rest_snapshot_parity(monkeypatch, client, editable):
    selector = AsyncMock(side_effect=lambda *args, **kwargs: _viewer_fixture())
    monkeypatch.setattr(graph_viewer, "graph_storage_ready", lambda: editable)
    monkeypatch.setattr(graph_viewer, "effective_knowledge_graph", selector)
    monkeypatch.setattr(rag_stub.lightrag, "get_knowledge_graph", selector)
    monkeypatch.setattr(config, "MCP_VIEWER_BASE_URL", "https://example.org/rag/")
    async with _mcp_client() as (http, agent):
        # A second stateless client advertises Apps and fetches the declared UI resource.
        init = _mcp_request(
            "initialize",
            protocolVersion="2025-11-25",
            clientInfo={"name": "test-app-host", "version": "1"},
            capabilities={
                "extensions": {
                    "io.modelcontextprotocol/ui": {"mimeTypes": [mcp_viewer.VIEWER_APP_MIME]}
                }
            },
        )
        assert (await http.post("/mcp", json=init, headers=MCP_HEADERS)).is_success
        tools = {tool.name: tool for tool in (await agent.list_tools()).tools}
        uri = tools["show_workspace_graph"].meta["ui"]["resourceUri"]
        resource = (await agent.read_resource(mcp_viewer.VIEWER_APP_URI)).contents[0]
        assert str(resource.uri) == uri
        assert resource.mimeType == mcp_viewer.VIEWER_APP_MIME
        assert "ui/notifications/tool-result" in resource.text
        assert "Original" not in resource.text
        assert "forceSimulation" in resource.text
        filters = ["/café/", "a&b"]
        args = {
            "workspace_id": "alex",
            "node_label": "Original",
            "max_nodes": 1,
            "physics": False,
            "file_path_contains": filters,
        }
        linked = await agent.call_tool("show_workspace_graph", args | {"delivery": "link"})
        selector.assert_not_awaited()
        url = urlsplit(linked.structuredContent["viewer_url"])
        assert url.path == "/rag/workspace/alex/graph.html"
        assert parse_qs(url.query)["file_path_contains"] == filters
        assert parse_qs(url.query)["physics"] == ["false"]

        raw = _mcp_request("tools/call", name="show_workspace_graph", arguments=args)
        with_app, plain = await asyncio.gather(
            http.post("/mcp", json=raw, headers=MCP_HEADERS),
            agent.call_tool("show_workspace_graph", args),
        )
        assert with_app.json()["result"] == plain.model_dump(by_alias=True, exclude_none=True)
        assert plain.meta["polygraphrag/graph"]["nodes"][0]["label"] == "Original </script>"
        assert len(plain.content) == 1
        assert "Interactive rendering requires an MCP Apps host" in plain.content[0].text
        assert linked.structuredContent["viewer_url"] in plain.content[0].text

        html = await agent.call_tool("show_workspace_graph", args | {"delivery": "html"})
        rest = await http.get(f"{WS}/graph.html", params=url.query)
        assert html.meta is None
        assert "If your client supports saving embedded resources" in html.content[0].text
        assert html.content[1].resource.mimeType == "text/html"
        assert html.content[1].resource.text == rest.text
        assert "Original <\\/script>" in rest.text
        assert "const PHYSICS = false" in rest.text
        for method, endpoint, media_type, options in (
            ("GET", f"{WS}/graph.html", "text/html", {"params": url.query}),
            ("POST", "/mcp", "application/json", {"json": _mcp_request("tools/list")}),
        ):
            identity, plain_bytes = await _raw_response(
                http, method, endpoint, "identity", **options
            )
            gzipped, wire = await _raw_response(http, method, endpoint, "gzip", **options)
            assert "Content-Encoding" not in identity
            assert gzipped["Content-Encoding"] == "gzip"
            assert identity["Vary"] == gzipped["Vary"] == "Accept-Encoding"
            assert identity["Content-Type"] == gzipped["Content-Type"]
            assert gzipped["Content-Type"].startswith(media_type)
            assert gzip.decompress(wire) == plain_bytes

        expected_length = len(url.geturl()) + len("&file_path_contains=")
        long_filter = "x" * (mcp_viewer.VIEWER_URL_BYTES - expected_length)
        exact = args | {"delivery": "link", "file_path_contains": filters + [long_filter]}
        at_limit = await agent.call_tool("show_workspace_graph", exact)
        assert (
            len(at_limit.structuredContent["viewer_url"].encode("ascii"))
            == mcp_viewer.VIEWER_URL_BYTES
        )
        too_long = exact | {"file_path_contains": filters + [long_filter + "x"]}
        failure = await agent.call_tool("show_workspace_graph", too_long)
        assert _mcp_error(failure)["code"] == "viewer_url_too_long"
        inline = await agent.call_tool("show_workspace_graph", too_long | {"delivery": "auto"})
        assert "viewer_url" not in inline.structuredContent
        assert 'delivery="html"' in inline.content[0].text

        monkeypatch.setattr(server, "API_TOKENS", ["viewer-token"])
        unauthenticated = await http.get(f"{WS}/graph.html", params=url.query)
        assert unauthenticated.status_code == 401
        assert unauthenticated.headers["WWW-Authenticate"].startswith("Basic ")
        browser = await http.get(
            f"{WS}/graph.html",
            params=url.query,
            headers={"Authorization": _basic("viewer", "viewer-token")},
        )
        assert browser.text == rest.text


async def test_graph_result_budget_measures_uncompressed_serialized_result(monkeypatch, client):
    monkeypatch.setattr(config, "MCP_VIEWER_BASE_URL", "")
    monkeypatch.setattr(
        mcp_viewer, "select_viewer_graph", AsyncMock(return_value=_viewer_fixture())
    )
    renderer = MagicMock(return_value="é")
    monkeypatch.setattr(mcp_viewer, "render_graph_snapshot", renderer)
    async with _mcp_client() as (http, agent):
        inline = await agent.call_tool("show_workspace_graph", {"workspace_id": "alex"})
        assert 'delivery="html"' in inline.content[0].text
        link = await agent.call_tool(
            "show_workspace_graph", {"workspace_id": "alex", "delivery": "link"}
        )
        error = _mcp_error(link)
        assert error["code"] == "viewer_link_unavailable"
        assert "MCP_VIEWER_BASE_URL" in error["message"]

        args = {"workspace_id": "alex", "delivery": "html"}
        minimal = await agent.call_tool("show_workspace_graph", args)
        length = len(minimal.model_dump_json(by_alias=True, exclude_none=True).encode())
        renderer.return_value = "é" + "a" * (mcp_viewer.VIEWER_RESULT_BYTES - length)
        request = _mcp_request("tools/call", name="show_workspace_graph", arguments=args)
        async with http.stream(
            "POST", "/mcp", json=request, headers=MCP_HEADERS | {"Accept-Encoding": "gzip"}
        ) as res:
            raw = b"".join([chunk async for chunk in res.aiter_raw()])
            assert res.headers["Content-Encoding"] == "gzip"
            assert int(res.headers["Content-Length"]) == len(raw)
            decoded = gzip.decompress(raw)
            assert len(decoded) > mcp_viewer.VIEWER_RESULT_BYTES
            result = types.CallToolResult.model_validate(json.loads(decoded)["result"])
            assert not result.isError
            assert (
                len(result.model_dump_json(by_alias=True, exclude_none=True).encode())
                == mcp_viewer.VIEWER_RESULT_BYTES
            )

        renderer.return_value += "a"
        failures = []
        for coding in ("gzip", "identity"):
            res = await http.post(
                "/mcp", json=request, headers=MCP_HEADERS | {"Accept-Encoding": coding}
            )
            failures.append(_mcp_error(types.CallToolResult.model_validate(res.json()["result"])))
        assert failures[0] == failures[1]
        assert failures[0]["code"] == "viewer_result_too_large"
        assert failures[0]["limit"] == mcp_viewer.VIEWER_RESULT_BYTES


@pytest.mark.parametrize("base_url", ["", "https://example.org/rag"])
async def test_oversized_mcp_inline_result_falls_back_to_configured_link(
    monkeypatch, client, base_url
):
    monkeypatch.setattr(config, "MCP_VIEWER_BASE_URL", base_url)
    monkeypatch.setattr(
        mcp_viewer, "select_viewer_graph", AsyncMock(return_value=_viewer_fixture())
    )
    monkeypatch.setattr(mcp_viewer, "VIEWER_RESULT_BYTES", 1)
    async with _mcp_client() as (_, agent):
        inline = await agent.call_tool("show_workspace_graph", {"workspace_id": "alex"})
        if base_url:
            assert inline.structuredContent["representation"] == "link"
            assert inline.structuredContent["viewer_url"].startswith(base_url)
            assert inline.meta is None
        else:
            assert _mcp_error(inline)["code"] == "viewer_result_too_large"
        html = await agent.call_tool(
            "show_workspace_graph", {"workspace_id": "alex", "delivery": "html"}
        )
        failure = _mcp_error(html)
        assert failure["code"] == "viewer_result_too_large"
        assert ("viewer_url" in failure) is bool(base_url)


# --------------------------------------------------------------------------- #
# Journey 1 — ingest → track → query lifecycle (in-memory job/batch path)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_e2e_ingest_track_and_query_lifecycle(tmp_path, client):
    """Upload a 2-file batch, watch it move pending→done through the real worker, then query
    the workspace — the whole happy path a client walks, with each step's ids threaded into
    the next."""
    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        up = await client.post(
            f"{WS}/upload/batch",
            files=[_upload("a.txt", b"alpha"), _upload("b.txt", b"beta")],
        )
    assert up.status_code == 200
    body = up.json()
    batch_id = body["batch_id"]
    job_ids = [j["job_id"] for j in body["jobs"]]
    assert len(job_ids) == 2
    assert body["summary"]["total"] == 2
    assert all(j["status"] == "pending" for j in body["jobs"])

    # The batch and jobs listings reflect what upload just enqueued.
    batch_before = await client.get(f"{WS}/batch/{batch_id}")
    assert batch_before.json()["summary"].get("pending", 0) == 2
    jobs_listed = await client.get(f"{WS}/jobs")
    assert {j["job_id"] for j in jobs_listed.json()["jobs"]} == set(job_ids)

    # Run the background worker to completion.
    await _drain_queue(process_result=("doc-1", "key-1"))

    # Every job is now terminal-done, individually and in the batch summary.
    for jid in job_ids:
        st = await client.get(f"{WS}/status/{jid}")
        assert st.status_code == 200
        assert st.json()["status"] == "done"
    batch_after = await client.get(f"{WS}/batch/{batch_id}")
    assert batch_after.json()["summary"].get("done", 0) == 2

    # With the corpus ingested, the query surface answers and the data surface returns evidence.
    q = await client.post(f"{WS}/query", json={"query": "what is alpha?"})
    assert q.status_code == 200
    assert q.json()["result"] == "mocked answer"
    assert q.json()["references"] == []  # include_references defaults off

    qd = await client.post(f"{WS}/query/data", json={"query": "what is alpha?"})
    assert qd.status_code == 200
    data = qd.json()["data"]
    assert set(data) == {"entities", "relationships", "chunks", "references"}


@pytest.mark.asyncio
async def test_e2e_failed_ingest_surfaces_error(tmp_path, client):
    """A file whose ingestion keeps raising ends up 'failed' with the error surfaced on the
    job status — the unhappy path of the same lifecycle."""
    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        up = await client.post(f"{WS}/upload/batch", files=[_upload("boom.txt")])
    job_id = up.json()["jobs"][0]["job_id"]

    original = server.config.MAX_RETRIES
    server.config.MAX_RETRIES = 2
    try:
        await _drain_queue(process_result=RuntimeError("ingest exploded"))
    finally:
        server.config.MAX_RETRIES = original

    st = await client.get(f"{WS}/status/{job_id}")
    assert st.json()["status"] == "failed"
    assert "ingest exploded" in st.json()["error"]


# --------------------------------------------------------------------------- #
# Journey 2 — workspace registry lifecycle (stateful registry pool)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_e2e_workspace_registry_lifecycle(client):
    """Create a workspace, see it in the listing and its status, soft-delete it (drops out of
    the active listing), then restore it (reappears) — driven against a pool that remembers
    its own writes."""
    server._db_pool = _StatefulRegistryPool()

    # Absent before creation.
    listed = await client.get("/all-workspaces/list")
    assert "projects" not in [w["id"] for w in listed.json()["workspaces"]]

    created = await client.post(
        "/all-workspaces/create",
        json={"id": "projects", "name": "Projects", "description": "work"},
    )
    assert created.status_code == 200
    assert created.json()["created_at"]  # same public shape as the list endpoint

    # Re-creating the same id now conflicts (the pool remembers it).
    dup = await client.post("/all-workspaces/create", json={"id": "projects", "name": "x"})
    assert dup.status_code == 409

    # It shows up in the active listing and has a readable status.
    listed = await client.get("/all-workspaces/list")
    entry = [w for w in listed.json()["workspaces"] if w["id"] == "projects"]
    assert entry and entry[0]["name"] == "Projects"

    status = await client.get("/workspace/projects")
    assert status.status_code == 200
    assert status.json()["active"] is True
    assert "is_primary" not in status.json()  # primary concept removed; workspaces are peers
    assert status.json()["documents"]["total"] == 0

    # Soft-delete hides it from the active listing but keeps it restorable.
    deleted = await client.delete("/workspace/projects")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "soft-deleted"
    active_ids = [w["id"] for w in (await client.get("/all-workspaces/list")).json()["workspaces"]]
    assert "projects" not in active_ids
    deleted_ids = [
        w["id"]
        for w in (await client.get("/all-workspaces/list?deleted=true")).json()["workspaces"]
    ]
    assert "projects" in deleted_ids

    # Restore brings it back to the active listing.
    restored = await client.post("/workspace/projects/restore")
    assert restored.status_code == 200
    active_ids = [w["id"] for w in (await client.get("/all-workspaces/list")).json()["workspaces"]]
    assert "projects" in active_ids


@pytest.mark.asyncio
async def test_e2e_delete_default_workspace_allowed(client):
    """The seeded `default` is an ordinary peer now (no primary concept), so it soft-deletes
    like any other workspace and drops out of the active listing."""
    pool = _StatefulRegistryPool()
    pool._rows["default"] = {
        "id": "default",
        "name": "Default",
        "description": None,
        "lightrag_workspace": "default",
        "deleted_at": None,
        "created_at": datetime.now(timezone.utc),
    }
    server._db_pool = pool
    resp = await client.delete("/workspace/default")
    assert resp.status_code == 200
    assert resp.json()["status"] == "soft-deleted"
    active_ids = [w["id"] for w in (await client.get("/all-workspaces/list")).json()["workspaces"]]
    assert "default" not in active_ids


@pytest.mark.asyncio
async def test_e2e_workspace_data_isolation(client):
    """Two peer workspaces route to independent RAG instances: a query to one returns only its
    own content, never the other's. The API threads ``workspace_id`` end-to-end with no shared or
    leaked instance. (Real Postgres row-level namespacing is exercised by the live
    ``scripts/e2e_isolation.py``; here we prove the routing layer keeps workspaces separate.)"""

    def _answer(ws: str) -> dict:
        return {
            "status": "success",
            "data": {"references": [], "chunks": [], "entities": [], "relationships": []},
            "llm_response": {"content": f"answer-from-{ws}", "is_streaming": False},
            "metadata": {},
        }

    instances: dict[str, MagicMock] = {}

    def _make(ws: str) -> MagicMock:
        m = MagicMock(name=f"rag-{ws}")
        m.lightrag.aquery_llm = AsyncMock(return_value=_answer(ws))
        return m

    async def _dispatch(ws: str) -> MagicMock:
        # One distinct instance per workspace id — the isolation boundary under test.
        return instances.setdefault(ws, _make(ws))

    server.get_workspace_rag = _dispatch

    a = await client.post("/workspace/alpha/query", json={"query": "who are you?"})
    b = await client.post("/workspace/beta/query", json={"query": "who are you?"})
    assert a.status_code == 200 and b.status_code == 200

    # Each workspace answers with only its own content …
    assert a.json()["result"] == "answer-from-alpha"
    assert b.json()["result"] == "answer-from-beta"
    # … and never the other's (no cross-workspace leakage through the routing layer).
    assert "beta" not in a.json()["result"]
    assert "alpha" not in b.json()["result"]
    # Two separate instances were built — nothing is shared across the two workspaces.
    assert set(instances) == {"alpha", "beta"}
    assert instances["alpha"] is not instances["beta"]


# --------------------------------------------------------------------------- #
# Journey 3 — the same journey, gated by API tokens
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_e2e_auth_gates_full_journey(tmp_path, client):
    """With ``API_TOKENS`` set, health stays open but every step of a real journey is refused
    without a Bearer token and succeeds with one."""
    token = "e2e-secret"
    server.API_TOKENS = [token]
    server._db_pool = _StatefulRegistryPool()
    auth = {"Authorization": f"Bearer {token}"}

    # Health is always public.
    assert (await client.get("/health")).status_code == 200

    # No creds → every gated step 401.
    assert (await client.get("/all-workspaces/list")).status_code == 401
    assert (await client.post(f"{WS}/query", json={"query": "hi"})).status_code == 401
    assert (
        await client.post("/all-workspaces/create", json={"id": "team", "name": "T"})
    ).status_code == 401

    # A wrong token is still 401 (defends against a leaked-but-stale value).
    assert (
        await client.get("/all-workspaces/list", headers={"Authorization": "Bearer nope"})
    ).status_code == 401

    # Basic auth (any user + token as password) also works, for browsers.
    assert (
        await client.get("/all-workspaces/list", headers={"Authorization": _basic("u", token)})
    ).status_code == 200

    # With the Bearer token the journey proceeds: create → upload → drain → query.
    created = await client.post(
        "/all-workspaces/create", json={"id": "team", "name": "T"}, headers=auth
    )
    assert created.status_code == 200

    with patch.object(server, "WORKING_DIR", str(tmp_path)):
        up = await client.post(f"{WS}/upload/batch", files=[_upload("d.txt")], headers=auth)
    assert up.status_code == 200
    await _drain_queue(process_result=("doc-x", "key-x"))
    job_id = up.json()["jobs"][0]["job_id"]
    st = await client.get(f"{WS}/status/{job_id}", headers=auth)
    assert st.status_code == 200 and st.json()["status"] == "done"

    q = await client.post(f"{WS}/query", json={"query": "hi"}, headers=auth)
    assert q.status_code == 200
