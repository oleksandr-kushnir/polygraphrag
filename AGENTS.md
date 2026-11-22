# Repository Guidelines

## Project Structure

PolyGraphRAG is a Python 3.11+ FastAPI service. `server/__init__.py` assembles the app and manages its lifespan. Put HTTP handlers in `server/routers/` and Pydantic request and response models in `server/schemas.py`. Document parsing belongs in `server/ingest.py`, background jobs in `server/worker.py`, and workspace lookup in `server/workspaces.py`.

`db/` builds PostgreSQL and enables pgvector and Apache AGE. `server/db.py` manages application metadata. The graph viewer lives in `server/graph.py` and uses D3 from `server/vendor/`. API and configuration guides live in `docs/`, with screenshots in `docs/images/`.

## Build and Run Locally

- In an active Python virtual environment, run `python -m pip install -r requirements-dev.txt` to install the test and style tools.
- For a new local deployment, copy `.env.example` to `.env`, set `POSTGRES_PASSWORD`, and configure credentials for your model providers. Keep secrets in the local `.env`.
- Run `docker compose up -d --build` from the repository root to build and start the API and database from your checkout. Check `http://localhost:9622/health` for readiness and open `/docs` to explore the API.
- Leave `POSTGRES_WORKSPACE` unset so each workspace retains its own LightRAG namespace.

## Coding Style

Use four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Format with `black server tests scripts` (100-column target), check lint and import order with `ruff check server tests scripts`, and type-check with `mypy server`.

## Testing

Run `python -m pytest -q` for the full suite. Pytest and pytest-asyncio exercise the app with stubbed RAG libraries and PostgreSQL, so tests run without containers or API keys. Name files `test_*.py` and functions `test_*`. Add focused endpoint cases to `tests/test_server.py` and multi-request workflows to `tests/test_e2e.py`. The project sets no numeric coverage threshold. For live HTTP checks against a running development stack, use `bash scripts/smoke_test_docker.sh`.

## Commits and Pull Requests

Use short imperative commit subjects, as in `Add entity-type color legend to graph.html`. In a pull request, describe the behavior change, link any related issue, and list the checks you ran. Include a screenshot for visible graph viewer changes and update `docs/api-reference.md` when an endpoint contract changes.
