#!/usr/bin/env bash
# Live end-to-end smoke test against a RUNNING PolyGraphRAG deployment.
#
# Unlike smoke_test.py (in-process, LightRAG/Postgres stubbed out), this drives the
# real container over HTTP: it boots a fresh workspace and runs a query + graph fetch,
# which forces LightRAG's PG*Storage.initialize_storages() over asyncpg against the
# live Postgres + pgvector + Apache AGE backend. That is the path that would break if a
# required Postgres driver (e.g. psycopg2) were missing.
#
# Usage:
#   docker compose up -d
#   ./smoke_test_docker.sh                 # reads RAG_PORT / API_TOKENS from .env
#   RAG_PORT=9632 API_TOKEN=xxx ./smoke_test_docker.sh
#
# Exit 0 if every check passes, 1 otherwise (suitable for CI / post-deploy gating).
set -uo pipefail

# --- Config: prefer env, fall back to .env, then defaults ---------------------
env_get() { grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2-; }
RAG_PORT="${RAG_PORT:-$(env_get RAG_PORT)}"; RAG_PORT="${RAG_PORT:-9622}"
# First token from API_TOKENS (comma-separated) unless API_TOKEN is given explicitly.
if [ -z "${API_TOKEN:-}" ]; then API_TOKEN="$(env_get API_TOKENS | cut -d, -f1)"; fi
BASE="http://127.0.0.1:${RAG_PORT}"
WS="smoke_$$"                       # unique per run so reruns never collide
AUTH=(); [ -n "${API_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer ${API_TOKEN}")

pass=0; fail=0
check() { # check "<name>" <actual> <expected>
  if [ "$2" = "$3" ]; then echo "[PASS] $1"; pass=$((pass+1));
  else echo "[FAIL] $1  (got '$2', want '$3')"; fail=$((fail+1)); fi
}
code() { curl -s -o /dev/null -w "%{http_code}" "${AUTH[@]}" "$@"; }

echo "PolyGraphRAG live smoke test  ->  ${BASE}  (workspace ${WS})"
echo "============================================================"

# --- Readiness: wait up to ~90s for the app to finish startup -----------------
ready=""
for _ in $(seq 1 30); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/health")" = "200" ] && { ready=1; break; }
  sleep 3
done
check "app becomes healthy" "${ready:-timeout}" "1"
[ -z "$ready" ] && { echo "app never became healthy; aborting"; exit 1; }

# --- Health is public (never gated) -------------------------------------------
check "GET /health -> 200"            "$(curl -s -o /dev/null -w '%{http_code}' ${BASE}/health)" "200"

# --- Auth gate (only meaningful when API_TOKENS is set) -----------------------
if [ -n "${API_TOKEN:-}" ]; then
  check "no creds -> 401"             "$(curl -s -o /dev/null -w '%{http_code}' ${BASE}/all-workspaces/list)" "401"
fi

# --- Workspace lifecycle + real PG storage init -------------------------------
check "create workspace -> 200"       "$(code -X POST ${BASE}/all-workspaces/create -H 'Content-Type: application/json' -d "{\"id\":\"${WS}\",\"name\":\"smoke\",\"description\":\"live smoke\"}")" "200"
check "list workspaces -> 200"        "$(code ${BASE}/all-workspaces/list)" "200"
check "workspace listed"              "$(curl -s "${AUTH[@]}" ${BASE}/all-workspaces/list | grep -c "\"${WS}\"")" "1"
check "GET /workspace/{id} -> 200"    "$(code ${BASE}/workspace/${WS})" "200"
check "GET files -> 200"              "$(code ${BASE}/workspace/${WS}/files)" "200"
# query builds a LightRAG instance -> PG*Storage.initialize_storages() over asyncpg
check "POST query -> 200"             "$(code -X POST ${BASE}/workspace/${WS}/query -H 'Content-Type: application/json' -d '{"query":"hello","mode":"local"}')" "200"
check "GET graph.html -> 200"         "$(code ${BASE}/workspace/${WS}/graph.html)" "200"

# --- Input validation ---------------------------------------------------------
check "invalid mode -> 422"           "$(code -X POST ${BASE}/workspace/${WS}/query -H 'Content-Type: application/json' -d '{"query":"hi","mode":"bogus"}')" "422"

# --- Cleanup ------------------------------------------------------------------
check "delete workspace -> 200"       "$(code -X DELETE ${BASE}/workspace/${WS})" "200"

echo "============================================================"
echo "$pass/$((pass+fail)) checks passed"
[ "$fail" -eq 0 ]
