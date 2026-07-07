# API Reference

Base URL in the default deployment: `http://localhost:9622`. Every endpoint is also documented interactively at **`/docs`** (Swagger UI) and **`/redoc`**.

All request/response bodies are JSON unless noted. File uploads use `multipart/form-data`.

## Authentication

Auth is opt-in via the `API_TOKENS` env var. When it is empty (default), no credentials are
required (the ports are loopback-only). When set, **every endpoint except `GET /health`** requires
a token, sent either as `Authorization: Bearer <token>` (machines) or via HTTP Basic with any
username and the token as the password (browsers, for `/docs` and `graph.html`). A missing/invalid
token returns `401` with a `WWW-Authenticate: Basic` challenge. Use TLS when exposed beyond
loopback.

---

## Health

### `GET /health`
Liveness probe. Returns `{"status":"ok"}`. Does not check the database.

---

## Workspaces (projects / graphs)

A **workspace** is an isolated knowledge graph + vector namespace. The slug must match `^[a-z][a-z0-9_]{0,47}$`.

### `POST /all-workspaces/create`
Create a new isolated workspace.

```json
{ "id": "acme", "name": "Acme Corp", "description": "optional" }
```
`id` doubles as the storage namespace. Returns the created workspace.

### `GET /all-workspaces/list`
List all active workspaces.

### `GET /workspace/{id}`
Overview of one workspace: active/soft-deleted state, document counts by status, chunk count, distinct entity/relationship counts, and an ingest-job summary.

### `DELETE /workspace/{id}`
Soft-delete a workspace (recoverable). The primary/`default` workspace is delete-protected.

### `POST /workspace/{id}/restore`
Restore a soft-deleted workspace.

---

## Ingestion

### `POST /workspace/{id}/upload/batch`
Upload one or more files; ingestion runs asynchronously in the background.

- **Form fields:** `files` (one or more binary parts). Optional `metadata` — a JSON array **index-aligned with `files`**, each object may carry:
  - `description` — embedded into the searchable text.
  - `source_path` — original relative path/URL.
  - `path_root` — optional absolute prefix; when combined with `source_path`, the file's stored identity becomes `path_root/source_path` so query references resolve back to your own source tree.
  - `last_modified_time` — ISO-8601 timestamp.
- **Returns:** a `batch_id` and a per-file `job_id`.

Supported types (22): text `md,txt,csv,html`; docs `pdf,docx,pptx,xlsx`; images `jpg,jpeg,png,gif,bmp,tiff,webp`; audio `mp3,wav,m4a,ogg,flac,opus,webm`.

```bash
curl -X POST localhost:9622/workspace/acme/upload/batch \
  -F 'files=@handbook.pdf' \
  -F 'metadata=[{"description":"employee handbook"}]'
```

### `GET /workspace/{id}/status/{job_id}`
Status of one ingest job: `pending` → `processing` → `processed` / `failed` (with error and attempt count).

### `GET /workspace/{id}/batch/{batch_id}`
Status of every job in a batch.

### `GET /workspace/{id}/jobs`
List recent ingest jobs for the workspace.

---

## Files

### `GET /workspace/{id}/files`
List ingested files with their stored metadata (path, content hash, doc id, extracting model, timestamps).

### `DELETE /workspace/{id}/file/delete`
Remove one file's document, chunks, and the entities/relationships sourced **only** by it. Identify the file by one of:

```json
{ "doc_id": "doc-<md5>" }          // most precise
{ "external_path": "/data/corpus/sub/handbook.pdf" }
{ "rel_path": "sub/handbook.pdf" }
```

Entities shared with other documents are preserved (only solely-sourced entities are removed). The document's LLM cache is always cleared. **Idempotent** — deleting an absent file returns a `noop`.

---

## Query

### `POST /workspace/{id}/query`
Ask a natural-language question and get a synthesized answer.

```json
{
  "query": "How do we handle refunds?",
  "mode": "mix",
  "include_references": true,
  "top_k": 40
}
```

**Modes:** `mix` (default — graph + vector, recommended), `local` (entity-centric), `global` (relationship/theme-centric), `hybrid` (local + global), `naive` (plain vector search).

Returns the answer plus, when `include_references` is true, enriched source-document citations.

### `POST /workspace/{id}/query/data`
Structured retrieval **without** LLM answer generation — returns the entities, relationships, chunks, and references that retrieval selected. Same fields as `/query`, plus:

- `file_path_contains` — a list of case-insensitive substrings; keep only results whose `file_path` contains any of them (folder/file scoping). Applied after retrieval, with an auto-boosted budget.

Useful for building your own UI, debugging retrieval, or feeding another pipeline.

---

## Visualization

### `GET /workspace/{id}/graph.html`
Returns a self-contained interactive HTML page (built with [pyvis](https://pyvis.readthedocs.io/)) rendering the workspace's knowledge graph — open it directly in a browser. Supports optional filtering by file-path substring so you can focus on one folder's subgraph.
