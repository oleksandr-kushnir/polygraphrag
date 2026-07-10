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

## Discovery & health

### `GET /`
Service card: `{"name","version","docs":"/docs","openapi":"/openapi.json","health":"/health"}`. Gated by the same auth as the rest of the API.

### `GET /health`
Liveness probe. Returns `{"status":"ok"}`. Does not check the database. Never requires auth.

---

## Workspaces (projects / graphs)

A **workspace** is an isolated knowledge graph + vector namespace. The slug must match `^[a-z][a-z0-9_]{0,47}$`.

### `POST /all-workspaces/create`
Create a new isolated workspace.

```json
{ "id": "acme", "name": "Acme Corp", "description": "optional" }
```
`id` doubles as the storage namespace. Returns the created workspace in the same shape as the list endpoint:

```json
{ "id": "acme", "name": "Acme Corp", "description": "optional", "created_at": "2026-07-10T09:00:00+00:00" }
```

### `GET /all-workspaces/list`
List all active workspaces. Pass `?deleted=true` to list soft-deleted workspaces instead.

### `GET /workspace/{id}`
Overview of one workspace: active/soft-deleted state, document counts by status, chunk count, distinct entity/relationship counts, and an ingest-job summary.

### `DELETE /workspace/{id}`
Soft-delete a workspace (recoverable). Any workspace can be deleted — all workspaces are peers (there is no delete-protected "primary"). Pass `purge=true` to irreversibly delete its data instead.

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

Response:

```json
{
  "batch_id": "9f3a1c2b",
  "summary": { "pending": 1, "total": 1 },
  "jobs": [
    {
      "job_id": "ab12cd34",
      "file": "handbook.pdf",
      "file_path": "handbook.pdf",
      "status": "pending",
      "attempts": 0,
      "error": null,
      "batch_id": "9f3a1c2b",
      "content_hash": "9c56cc51…"
    }
  ]
}
```

### Ingest-job statuses (canonical list)

An ingest job's `status` is one of:

`pending` → `processing` → (`retrying`) → **`done`** | **`failed`**, plus **`save_failed`** when the uploaded bytes couldn't be written before processing started.

> **Note:** this is the *job* status. LightRAG's internal per-*document* status
> (`lightrag_doc_status`, surfaced in the `GET /workspace/{id}` overview under
> `documents.by_status`) is a different vocabulary that includes `processed` — don't
> poll a job for `processed`; a finished job is `done`.

### `GET /workspace/{id}/status/{job_id}`
Status of one ingest job (see the canonical status list above), with error and attempt count:

```json
{
  "job_id": "ab12cd34",
  "batch_id": "9f3a1c2b",
  "file": "handbook.pdf",
  "file_path": "/data/corpus/hr/handbook.pdf",
  "source_path": "hr/handbook.pdf",
  "doc_id": "doc-1a2b3c…",
  "content_hash": "9c56cc51…",
  "status": "done",
  "attempts": 0,
  "error": null,
  "description": "employee handbook",
  "last_modified_time": "2026-01-05T09:00:00",
  "uploaded_at": "2026-01-06T10:00:00"
}
```

### `GET /workspace/{id}/batch/{batch_id}`
Status of every job in a batch.

### `GET /workspace/{id}/jobs`
List the 100 most recent ingest jobs for the workspace, newest first.

---

## Files

### `GET /workspace/{id}/files`
List ingested files with their stored metadata:

```json
{
  "files": [
    {
      "job_id": "ab12cd34",
      "file": "handbook.pdf",
      "file_path": "/data/corpus/hr/handbook.pdf",
      "source_path": "hr/handbook.pdf",
      "doc_id": "doc-1a2b3c…",
      "content_hash": "9c56cc51…",
      "status": "done",
      "last_modified_time": "2026-01-05T09:00:00",
      "uploaded_at": "2026-01-06T10:00:00"
    }
  ]
}
```

### `DELETE /workspace/{id}/file/delete`
Remove one file's document, chunks, and the entities/relationships sourced **only** by it. Identify the file by one of:

```json
{ "doc_id": "doc-<md5>" }          // most precise
{ "external_path": "/data/corpus/sub/handbook.pdf" }   // the real path recorded at upload (path_root/source_path)
{ "rel_path": "sub/handbook.pdf" }
```

Entities shared with other documents are preserved (only solely-sourced entities are removed). The document's LLM cache is always cleared. **Idempotent** — deleting an absent file returns a `noop`; a body with none of the three identifiers is rejected with `422`.

```json
{ "status": "deleted", "doc_id": "doc-1a2b3c…" }     // success
{ "status": "noop", "reason": "not_found" }          // file wasn't present
```

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

Response:

```json
{
  "result": "Refunds are granted within 30 days… [1]",
  "references": [
    {
      "reference_id": "1",
      "file_path": "/data/corpus/policies/refunds.pdf",
      "job_id": "ab12cd34",
      "file_description": "Refund policy 2026",
      "last_modified_time": "2026-01-05T09:00:00",
      "uploaded_at": "2026-01-06T10:00:00",
      "llm_model_extracted": "gpt-5.4-mini",
      "llm_model_answered": "gpt-5.4-mini"
    }
  ]
}
```

**Reference paths are real, not internal.** Each `references[].file_path` is the **openable document path you supplied at upload** (via `path_root` + `source_path`), resolved server-side from Postgres — **not** LightRAG's internal canonical name. If a reference can't be resolved to a stored file, `file_path` is `null` (the internal name is never exposed). LightRAG 1.5.x canonicalizes its own `file_path` to a basename for dedup; PolyGraphRAG keeps the authoritative path in its own metadata and maps citations back to it, so this stays correct across LightRAG versions.

### `POST /workspace/{id}/query/data`
Structured retrieval **without** LLM answer generation — returns the entities, relationships, chunks, and references that retrieval selected:

```json
{
  "status": "success",
  "message": "Query completed successfully",
  "data": {
    "entities":      [ { "entity_name": "Refund Policy", "entity_type": "concept", "description": "…", "file_path": "/data/corpus/policies/refunds.pdf" } ],
    "relationships": [ { "src_id": "Refund Policy", "tgt_id": "Billing Team", "description": "…", "file_path": "/data/corpus/policies/refunds.pdf" } ],
    "chunks":        [ { "content": "Refunds are granted within 30 days…", "file_path": "/data/corpus/policies/refunds.pdf" } ],
    "references":    [ { "reference_id": "1", "file_path": "/data/corpus/policies/refunds.pdf", "job_id": "ab12cd34", "…": "…" } ]
  },
  "metadata": {}
}
```

Same request fields as `/query`, plus:

Every `file_path` in the response — on entities, relationships, chunks, **and** references — is the **real, openable path** resolved from Postgres, never LightRAG's internal name. Entities and relationships drawn from several documents carry their sources as a `<SEP>`-joined list of real paths.

- `file_path_contains` — folder/file scope filter. **Omit it or leave it empty to get ALL data (no filtering) — this is the default.** When provided, it is a case-insensitive **OR** substring filter: an entity/relationship/chunk/reference is kept if its (real) `file_path` contains ANY of the strings (blank strings are ignored, so an all-blank list still returns everything). Applied after retrieval with an auto-boosted budget, so a very narrow scope may return fewer items than exist.

Useful for building your own UI, debugging retrieval, or feeding another pipeline.

---

## Visualization

### `GET /workspace/{id}/graph.html`
Returns a self-contained interactive HTML page (built with [D3.js v7](https://d3js.org/), force layout drawn on an HTML canvas, D3 inlined) rendering the workspace's knowledge graph — open it directly in a browser. Supports optional filtering by file-path substring so you can focus on one folder's subgraph.

Query parameters:

| Param | Default | Meaning |
|---|---|---|
| `node_label` | `*` | Entity name to center the subgraph on; `*` renders the entire graph. |
| `max_depth` | `3` | Maximum relationship hops expanded out from the starting node(s). |
| `max_nodes` | `5000` | Hard cap on rendered nodes; closest / highest-degree nodes win when truncated. |
| `physics` | `true` | Animated force-directed layout; set `false` for a static layout on large graphs. |
| `file_path_contains` | _(empty)_ | Repeatable case-insensitive substring filter on node `file_path` (OR semantics); empty = whole graph. |
