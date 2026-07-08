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
import io
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

# Import the unit-test module first so its raganything/lightrag stubs are the ones registered in
# sys.modules, and reuse the very same RAGAnything stub. Registering a *second*, competing set of
# stubs here would shadow the QueryParam object test_server introspects (whichever module imports
# first wins), so we deliberately share one set instead of duplicating it.
import test_server as _ts  # noqa: E402

import server  # noqa: E402

rag_stub = _ts.rag_stub
_AnyKeyDocs = _ts._AnyKeyDocs
_AQUERY_LLM = _ts._DEFAULT_AQUERY_LLM
_AQUERY_DATA = _ts._DEFAULT_AQUERY_DATA

# --------------------------------------------------------------------------- #
# Fixtures & helpers
# --------------------------------------------------------------------------- #
WS = "/workspace/alex"  # primary public workspace → physical "alex" via the lookup stub


async def _fake_lookup_workspace(workspace_id):
    """Every workspace resolves as present, active, and physically named after its id."""
    return {
        "id": workspace_id,
        "name": workspace_id,
        "description": None,
        "lightrag_workspace": workspace_id,
        "is_primary": workspace_id == "alex",
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
        elif "SET deleted_at = NOW()" in sql:
            self._rows[args[0]]["deleted_at"] = datetime.now(timezone.utc)
        elif "SET deleted_at = NULL" in sql:
            self._rows[args[0]]["deleted_at"] = None
        elif "DELETE FROM rag_workspaces" in sql:
            self._rows.pop(args[0], None)

    async def fetchrow(self, sql: str, *args):
        if "FROM rag_workspaces WHERE id = $1" in sql:
            row = self._rows.get(args[0])
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
    assert status.json()["is_primary"] is False
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
async def test_e2e_delete_primary_workspace_blocked(client):
    """The primary workspace is delete-protected end-to-end, even though it exists."""
    pool = _StatefulRegistryPool()
    pool._rows["default"] = {
        "id": "default",
        "name": "Default",
        "description": None,
        "lightrag_workspace": "default",
        "is_primary": True,
        "deleted_at": None,
        "created_at": datetime.now(timezone.utc),
    }
    server._db_pool = pool
    resp = await client.delete("/workspace/default")
    assert resp.status_code == 409


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
