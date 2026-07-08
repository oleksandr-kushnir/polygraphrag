"""
Comprehensive in-process smoke test for the PolyGraphRAG API.

Drives the *real* ASGI app (auth middleware, request validation, error handling and all)
through httpx's in-memory transport — no Postgres, no OpenAI, no running container. RAGAnything
and LightRAG are stubbed the same way the unit suite stubs them, so this is a fast end-to-end
sanity check of the HTTP surface, focused on the auth + hardening work.

Run:   python smoke_test.py
Exit:  0 if every check passes, 1 otherwise (suitable for CI / pre-deploy gating).
"""

import asyncio
import base64
import sys
from unittest.mock import AsyncMock, MagicMock

# --------------------------------------------------------------------------- #
# Stub the heavy deps BEFORE importing server (mirrors tests/test_server.py).
# --------------------------------------------------------------------------- #
_AQUERY_LLM = {
    "status": "success",
    "data": {"references": []},
    "llm_response": {"content": "smoke answer", "is_streaming": False},
    "metadata": {},
}
_AQUERY_DATA = {
    "status": "success",
    "message": "ok",
    "data": {"entities": [], "relationships": [], "chunks": [], "references": []},
    "metadata": {},
}

rag_stub = MagicMock()
rag_stub.lightrag = MagicMock()
rag_stub.lightrag.aquery_llm = AsyncMock(return_value=_AQUERY_LLM)
rag_stub.lightrag.aquery_data = AsyncMock(return_value=_AQUERY_DATA)

raganything_mod = MagicMock()
raganything_mod.RAGAnything = MagicMock(return_value=rag_stub)
sys.modules.setdefault("raganything", raganything_mod)

lightrag_mod = MagicMock()
lightrag_mod.QueryParam = MagicMock(return_value=MagicMock())
lightrag_utils_mod = MagicMock()
lightrag_utils_mod.EmbeddingFunc = MagicMock()
lightrag_mod.utils = lightrag_utils_mod
sys.modules.setdefault("lightrag", lightrag_mod)
sys.modules.setdefault("lightrag.utils", lightrag_utils_mod)

from httpx import ASGITransport, AsyncClient  # noqa: E402

import server  # noqa: E402

WS = "/workspace/alex"
TOKENS = ["smoketok", "alt-token"]


def _basic(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


# --------------------------------------------------------------------------- #
# Tiny check harness
# --------------------------------------------------------------------------- #
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))


async def main() -> int:
    # Make workspace routes resolve without a DB, and hand out the stub instance.
    async def _fake_lookup(workspace_id):
        return {
            "id": workspace_id,
            "name": workspace_id,
            "description": None,
            "lightrag_workspace": workspace_id,
            "is_primary": workspace_id == "alex",
        }

    server._lookup_workspace = _fake_lookup
    server.get_workspace_rag = AsyncMock(return_value=rag_stub)
    server._db_pool = None

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://smoke") as c:

        # ---- Pure helper: LOG_LEVEL resolver -----------------------------
        import logging

        check("log_level DEBUG->10", server._log_level_from_env("DEBUG") == logging.DEBUG)
        check("log_level 'info'->20", server._log_level_from_env("info") == logging.INFO)
        check("log_level None->INFO", server._log_level_from_env(None) == logging.INFO)
        check("log_level 'bogus'->INFO", server._log_level_from_env("bogus") == logging.INFO)
        check("log_level '  Error '->40", server._log_level_from_env("  Error ") == logging.ERROR)

        # ---- Auth DISABLED (default) -------------------------------------
        server.API_TOKENS = []
        r = await c.get("/health")
        check("auth-off: /health 200", r.status_code == 200 and r.json() == {"status": "ok"})
        r = await c.get("/openapi.json")
        check("auth-off: /openapi.json 200", r.status_code == 200)
        r = await c.post(f"{WS}/query", json={"query": "hi"})
        check("auth-off: query 200 (no creds)", r.status_code == 200)

        # ---- Auth ENABLED ------------------------------------------------
        server.API_TOKENS = list(TOKENS)

        r = await c.get("/health")
        check("auth-on: /health open (200)", r.status_code == 200)

        r = await c.post(f"{WS}/query", json={"query": "hi"})
        check("auth-on: no creds -> 401", r.status_code == 401)
        check(
            "auth-on: 401 sends WWW-Authenticate: Basic",
            r.headers.get("www-authenticate", "").startswith("Basic"),
        )
        check("auth-on: 401 body has no internals", "smoke answer" not in r.text)

        r = await c.get("/openapi.json")
        check("auth-on: /openapi.json gated -> 401", r.status_code == 401)

        r = await c.post(
            f"{WS}/query", json={"query": "hi"}, headers={"Authorization": f"Bearer {TOKENS[0]}"}
        )
        check("auth-on: valid Bearer (token 1) -> 200", r.status_code == 200)

        r = await c.post(
            f"{WS}/query", json={"query": "hi"}, headers={"Authorization": f"Bearer {TOKENS[1]}"}
        )
        check("auth-on: valid Bearer (token 2) -> 200", r.status_code == 200)

        r = await c.post(
            f"{WS}/query", json={"query": "hi"}, headers={"Authorization": "Bearer nope"}
        )
        check("auth-on: wrong Bearer -> 401", r.status_code == 401)

        r = await c.post(
            f"{WS}/query",
            json={"query": "hi"},
            headers={"Authorization": _basic("anyone", TOKENS[0])},
        )
        check("auth-on: Basic (any user + token) -> 200", r.status_code == 200)

        r = await c.post(
            f"{WS}/query",
            json={"query": "hi"},
            headers={"Authorization": _basic("anyone", "wrongpass")},
        )
        check("auth-on: Basic wrong password -> 401", r.status_code == 401)

        r = await c.post(
            f"{WS}/query", json={"query": "hi"}, headers={"Authorization": "Token abc"}
        )
        check("auth-on: unknown scheme -> 401", r.status_code == 401)

        r = await c.post(f"{WS}/query", json={"query": "hi"}, headers={"Authorization": "Bearer "})
        check("auth-on: empty Bearer -> 401", r.status_code == 401)

        # ---- Every gated route rejects missing creds ---------------------
        gated = [
            ("GET", "/all-workspaces/list"),
            ("POST", "/all-workspaces/create"),
            ("GET", f"{WS}"),
            ("POST", f"{WS}/query/data"),
            ("GET", f"{WS}/graph.html"),
            ("GET", f"{WS}/files"),
            ("GET", f"{WS}/jobs"),
            ("GET", f"{WS}/batch/x"),
            ("GET", f"{WS}/status/x"),
        ]
        for method, path in gated:
            resp = await c.request(method, path, json={} if method == "POST" else None)
            check(
                f"auth-on: {method} {path} -> 401",
                resp.status_code == 401,
                f"got {resp.status_code}",
            )

        auth = {"Authorization": f"Bearer {TOKENS[0]}"}

        # ---- Input validation (authenticated) ----------------------------
        r = await c.post(f"{WS}/query", json={"query": "hi", "mode": "bogus"}, headers=auth)
        check("validation: invalid mode -> 422", r.status_code == 422)

        r = await c.post(f"{WS}/query", json={"query": "hi", "top_k": 100000}, headers=auth)
        check("validation: top_k over ceiling -> 422", r.status_code == 422)

        r = await c.post(f"{WS}/query", json={"query": "hi", "top_k": 0}, headers=auth)
        check("validation: top_k below 1 -> 422", r.status_code == 422)

        r = await c.post(f"{WS}/query", json={"query": "hi", "top_k": 1000}, headers=auth)
        check("validation: top_k at ceiling -> 200", r.status_code == 200)

        r = await c.post(f"{WS}/query/data", json={"query": "hi", "mode": "bogus"}, headers=auth)
        check("validation: query/data invalid mode -> 422", r.status_code == 422)

        # ---- Error hygiene: no internal leak on 500 ----------------------
        rag_stub.lightrag.aquery_data.side_effect = RuntimeError("SECRET-INTERNAL-DETAIL")
        try:
            r = await c.post(f"{WS}/query/data", json={"query": "hi"}, headers=auth)
        finally:
            rag_stub.lightrag.aquery_data.side_effect = None
        check("error-hygiene: upstream failure -> 500", r.status_code == 500)
        check("error-hygiene: 500 body hides internals", "SECRET-INTERNAL-DETAIL" not in r.text)

    # ---- Report ----------------------------------------------------------
    server.API_TOKENS = []
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print("\nPolyGraphRAG smoke test\n" + "=" * 60)
    for name, ok, detail in _results:
        line = f"[{'PASS' if ok else 'FAIL'}] {name}"
        if not ok and detail:
            line += f"  ({detail})"
        print(line)
    print("=" * 60)
    print(f"{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
