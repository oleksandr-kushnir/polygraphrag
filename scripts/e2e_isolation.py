#!/usr/bin/env python3
"""Live end-to-end **workspace-isolation** test against a RUNNING PolyGraphRAG deployment.

Where ``scripts/smoke_test.py`` runs in-process with LightRAG/Postgres stubbed, and
``scripts/smoke_test_docker.sh`` drives a single workspace over HTTP, this script proves the
property that actually matters for multi-project use: **data ingested into one workspace never
leaks into another.** It drives the real container over HTTP against the live Postgres + pgvector
+ Apache AGE backend, so it exercises the real per-workspace row/graph namespacing.

What it does:
  1. Creates two fresh, peer workspaces (unique ids per run so reruns never collide).
  2. Uploads **four different file types** (txt, md, csv, html) to each, where every file carries
     a distinct, workspace-unique sentinel token (workspace A ⇒ ZEPHYRION, workspace B ⇒ QUORVAX).
  3. Waits for background ingestion of both batches to reach a terminal state.
  4. Queries each workspace and asserts the isolation invariant:
       - a workspace CAN retrieve its own sentinel, and
       - a workspace NEVER surfaces the other workspace's sentinel — even when asked directly
         about the other's topic (the hard cross-contamination check).
  5. Confirms each workspace's status counts reflect only its own uploads.
  6. Purges both workspaces it created (best-effort cleanup, even on failure).

It deliberately does NOT touch any pre-existing workspace (e.g. ``default``).

Usage (from the repo root, with the stack up):
    python scripts/e2e_isolation.py                 # reads RAG_PORT / API_TOKENS from .env
    RAG_PORT=9632 API_TOKEN=xxx python scripts/e2e_isolation.py

Exit 0 if every isolation check passes, 1 otherwise (suitable for CI / post-deploy gating).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx

# --------------------------------------------------------------------------- #
# Config: prefer env, fall back to .env, then defaults
# --------------------------------------------------------------------------- #


def _env_get(key: str) -> str | None:
    """Read KEY from a repo-root .env, if present (mirrors smoke_test_docker.sh)."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return None


RAG_PORT = os.getenv("RAG_PORT") or _env_get("RAG_PORT") or "9632"
_TOKENS = os.getenv("API_TOKEN") or (_env_get("API_TOKENS") or "").split(",")[0]
BASE = f"http://127.0.0.1:{RAG_PORT}"
HEADERS = {"Authorization": f"Bearer {_TOKENS}"} if _TOKENS else {}

RUN = f"{int(time.time()) % 100000:05d}"  # short, unique-per-run suffix
WS_A = f"e2e_iso_a_{RUN}"
WS_B = f"e2e_iso_b_{RUN}"

# Distinct, workspace-unique sentinels + the fictional entity each corpus is about.
SENT_A, TOPIC_A = "ZEPHYRIONXQ", "the Zephyrion Consortium"
SENT_B, TOPIC_B = "QUORVAXWZ", "the Quorvax Institute"

INGEST_TIMEOUT_S = 600  # CPU-only extraction of a few small text files
POLL_EVERY_S = 5

_pass = 0
_fail = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _pass, _fail
    if ok:
        print(f"[PASS] {name}")
        _pass += 1
    else:
        print(f"[FAIL] {name}" + (f"  ({detail})" if detail else ""))
        _fail += 1


def _files_for(sentinel: str, topic: str) -> list[tuple]:
    """Four DIFFERENT file types, each embedding the workspace's unique sentinel + topic."""
    body = (
        f"{topic} (internal code name {sentinel}) is a research organization. "
        f"{sentinel} was founded to study knowledge graphs. Contact {sentinel} for details."
    )
    return [
        ("files", (f"notes_{sentinel}.txt", body.encode(), "text/plain")),
        ("files", (f"readme_{sentinel}.md", f"# {topic}\n\n{body}\n".encode(), "text/markdown")),
        ("files", (f"data_{sentinel}.csv", f"org,code\n{topic},{sentinel}\n".encode(), "text/csv")),
        (
            "files",
            (
                f"page_{sentinel}.html",
                f"<html><body><h1>{topic}</h1><p>{body}</p></body></html>".encode(),
                "text/html",
            ),
        ),
    ]


def _create_workspace(client: httpx.Client, ws: str) -> bool:
    r = client.post(
        f"{BASE}/all-workspaces/create",
        json={"id": ws, "name": ws, "description": "e2e isolation run"},
    )
    check(f"create workspace {ws} -> 200", r.status_code == 200, f"got {r.status_code}: {r.text}")
    return r.status_code == 200


def _upload(client: httpx.Client, ws: str, sentinel: str, topic: str) -> str | None:
    r = client.post(f"{BASE}/workspace/{ws}/upload/batch", files=_files_for(sentinel, topic))
    ok = r.status_code == 200
    check(f"upload 4 file types to {ws} -> 200", ok, f"got {r.status_code}: {r.text[:200]}")
    if not ok:
        return None
    body = r.json()
    check(f"{ws}: 4 jobs enqueued", body["summary"]["total"] == 4, str(body["summary"]))
    return body["batch_id"]


def _wait_batch(client: httpx.Client, ws: str, batch_id: str) -> bool:
    """Poll the batch until every job is terminal (done/failed) or we time out."""
    deadline = time.time() + INGEST_TIMEOUT_S
    while time.time() < deadline:
        r = client.get(f"{BASE}/workspace/{ws}/batch/{batch_id}")
        if r.status_code != 200:
            time.sleep(POLL_EVERY_S)
            continue
        summary = r.json().get("summary", {})
        in_flight = sum(summary.get(s, 0) for s in ("pending", "processing", "retrying"))
        if in_flight == 0:
            done = summary.get("done", 0)
            check(f"{ws}: all 4 files ingested (done)", done == 4, f"summary={summary}")
            return done == 4
        time.sleep(POLL_EVERY_S)
    check(f"{ws}: ingestion finished before timeout", False, "timed out")
    return False


def _retrieved_text(client: httpx.Client, ws: str, query: str) -> str:
    """Return ONLY the retrieved `data` payload (entities/relationships/chunks/references) of
    /query/data, serialized for substring sentinel checks.

    We deliberately exclude the response's top-level `status`/`message`/`metadata`, because the
    service echoes the query text there — so a query that names the *other* workspace's sentinel
    would appear at the top level even with perfect isolation. The isolation invariant is strictly
    about *retrieved corpus data*, which lives under `data`.
    """
    r = client.post(
        f"{BASE}/workspace/{ws}/query/data",
        json={"query": query, "mode": "mix", "top_k": 40},
    )
    if r.status_code != 200:
        return ""
    return json.dumps(r.json().get("data", {}))


def _assert_isolation(client: httpx.Client) -> None:
    """The core invariant: each workspace retrieves ONLY its own data, never the other's."""
    # Ask each workspace about ITS OWN topic → its own sentinel should be retrievable.
    a_own = _retrieved_text(client, WS_A, f"Tell me about {TOPIC_A}")
    b_own = _retrieved_text(client, WS_B, f"Tell me about {TOPIC_B}")
    check(f"{WS_A} retrieves its own sentinel {SENT_A}", SENT_A in a_own)
    check(f"{WS_B} retrieves its own sentinel {SENT_B}", SENT_B in b_own)

    # Hard cross-contamination check: ask each workspace directly about the OTHER's topic AND
    # sentinel by name. If isolation holds, the other's data can never be *retrieved* here — the
    # sentinel is absent from the retrieved `data` even under this leading query.
    a_cross = _retrieved_text(client, WS_A, f"Tell me about {TOPIC_B} and {SENT_B}")
    b_cross = _retrieved_text(client, WS_B, f"Tell me about {TOPIC_A} and {SENT_A}")
    check(f"{WS_A} does NOT leak {WS_B}'s sentinel {SENT_B}", SENT_B not in a_cross, "cross-talk!")
    check(f"{WS_B} does NOT leak {WS_A}'s sentinel {SENT_A}", SENT_A not in b_cross, "cross-talk!")

    # Belt-and-braces: neither own-topic response carries the other workspace's sentinel either.
    check(f"{WS_A} own-topic answer clean of {SENT_B}", SENT_B not in a_own, "cross-talk!")
    check(f"{WS_B} own-topic answer clean of {SENT_A}", SENT_A not in b_own, "cross-talk!")


def _assert_status_counts(client: httpx.Client) -> None:
    for ws in (WS_A, WS_B):
        r = client.get(f"{BASE}/workspace/{ws}")
        if r.status_code != 200:
            check(f"{ws}: status -> 200", False, f"got {r.status_code}")
            continue
        total = r.json().get("documents", {}).get("total", 0)
        check(f"{ws}: exactly 4 documents", total == 4, f"total={total}")


def _purge(client: httpx.Client, ws: str) -> None:
    try:
        client.delete(f"{BASE}/workspace/{ws}", params={"purge": "true"})
    except Exception as exc:  # cleanup is best-effort
        print(f"[warn] purge {ws} failed: {exc}")


def main() -> int:
    print(f"PolyGraphRAG live ISOLATION test  ->  {BASE}")
    print(f"workspaces: {WS_A} (-> {SENT_A})  vs  {WS_B} (-> {SENT_B})")
    print("=" * 60)

    with httpx.Client(headers=HEADERS, timeout=60.0) as client:
        # Readiness.
        ready = False
        for _ in range(30):
            try:
                if client.get(f"{BASE}/health").status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(3)
        check("app is healthy", ready)
        if not ready:
            print("app never became healthy; aborting")
            return 1

        try:
            if not (_create_workspace(client, WS_A) and _create_workspace(client, WS_B)):
                return 1
            batch_a = _upload(client, WS_A, SENT_A, TOPIC_A)
            batch_b = _upload(client, WS_B, SENT_B, TOPIC_B)
            if not batch_a or not batch_b:
                return 1
            ok_a = _wait_batch(client, WS_A, batch_a)
            ok_b = _wait_batch(client, WS_B, batch_b)
            if ok_a and ok_b:
                _assert_status_counts(client)
                _assert_isolation(client)
        finally:
            _purge(client, WS_A)
            _purge(client, WS_B)
            print(f"[cleanup] purged {WS_A} and {WS_B}")

    print("=" * 60)
    print(f"{_pass}/{_pass + _fail} checks passed")
    return 0 if _fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
