# API Reference

Base URL in the default deployment: `http://localhost:9622`. Browse the REST endpoints at **`/docs`** (Swagger UI) and **`/redoc`**. MCP clients discover tools at **`/mcp`**.

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
The service card includes the service name, version, and paths for `docs`, `openapi`, `health`, and `mcp`. Gated by the same auth as the rest of the API.

### `GET /health`
Liveness probe. Returns `{"status":"ok"}`. Does not check the database. Never requires auth.

---

## MCP

Connect an agent to the knowledge graph at `http://localhost:9622/mcp` using **Streamable HTTP**.
From a container on the same Compose network, use `http://polygraphrag:9622/mcp`. If
`API_TOKENS` is set, send `Authorization: Bearer <token>`. The token grants service-wide access
across REST and MCP.

### Discover and call tools

The client initializes a stateless MCP connection, discovers the catalog and JSON Schemas with
`tools/list`, then invokes a tool by name with `tools/call`. All three operations go to
`POST /mcp`. Swagger UI and `/openapi.json` document the REST endpoints. Clients can select
individual tools to expose to an agent. That selection applies inside the client. The server
enforces its own catalog and access rules.

To try discovery and list active workspaces from Python, install the MCP SDK with
`python -m pip install "mcp>=1.30,<2"`. Set `POLYGRAPHRAG_MCP_URL` to override the endpoint
and `POLYGRAPHRAG_TOKEN` if authentication is enabled.

```python
import asyncio
import os

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main():
    url = os.getenv("POLYGRAPHRAG_MCP_URL", "http://localhost:9622/mcp")
    token = os.getenv("POLYGRAPHRAG_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(headers=headers) as http:
        async with streamable_http_client(url, http_client=http) as (reader, writer, _):
            async with ClientSession(reader, writer) as mcp:
                await mcp.initialize()
                catalog = await mcp.list_tools()
                print([tool.name for tool in catalog.tools])
                result = await mcp.call_tool("list_workspaces", {})
                if result.isError:
                    raise RuntimeError(result.content)
                print(result.structuredContent)


asyncio.run(main())
```

### Read tools

| Tool | Inputs | Result |
|---|---|---|
| `list_workspaces` | None | Active workspaces with public IDs, names and descriptions |
| `search_workspace` | `workspace_id`, `query`, optional retrieval settings and `file_path_contains` | Entities, relationships, chunks and source references from the selected workspace |
| `ask_workspace` | `workspace_id`, `query`, optional retrieval settings | A synthesized answer in `result` and its document `references` |
| `get_workspace_overview` | `workspace_id` | Corpus, graph and ingest counts |
| `get_ingest_job` | `workspace_id`, `job_id` | Job ID, status and attempt count |
| `find_graph_entities` | `workspace_id`, optional filters, `limit` and `cursor` | Entity candidates and a continuation cursor |
| `get_graph_entity` | `workspace_id`, `entity_id` | Entity fields, revision and provenance |
| `get_graph_relationship` | `workspace_id`, `edge_id` | Relationship fields, direction, revision and provenance |
| `get_graph_neighbors` | `workspace_id`, `entity_id`, optional traversal bounds and filters | Bounded entities and edges with a truncation flag |
| `retrieve_graph` | `workspace_id`, `q` or `text`, optional seed and traversal settings | Ranked seeds, bounded subgraph, sources and embedding coverage |
| `get_embedding_consistency` | `workspace_id` | Counts of indexed, missing and stale embeddings |
| `show_workspace_graph` | `workspace_id`, optional viewer filters and `delivery` | Interactive graph, standalone HTML or browser link |

The two corpus query tools accept the same `mode`, `top_k` and `include_references` settings as REST.
Graph ID lookup, keyword search, traversal, the viewer and the consistency report work with model
providers disabled. Use `find_graph_entities` for entity lookup or `retrieve_graph` for a
subgraph with sources. In keyword mode, send plain terms or phrases in `text`. Use `q` for
Boolean expressions with uppercase `AND` and `OR` and double quotes around phrases, such as
`"star chart" AND locker`.
An MCP tool error carries `isError: true`. Domain errors include JSON text with `status`, `code`
and `message`, plus public details such as the current revision for a conflict. HTTP authentication
failures return `401` before tool discovery or execution.

### Opt-in graph edits

Set `MCP_GRAPH_WRITES=true` and restart the API with the new setting to expose three additional
tools to clients with API access. This switch applies to MCP graph edits across the deployment.
Workspace management, file operations, graph deletion, merges, and embedding backfill are available
through REST.

| Tool | Inputs |
|---|---|
| `write_graph_batch` | `workspace_id` and the `entities` and `relationships` arrays from the REST batch contract |
| `patch_graph_entity` | `workspace_id`, `entity_id`, `expected_revision` and fields to change |
| `patch_graph_relationship` | `workspace_id`, `edge_id`, `expected_revision` and fields to change |

Use a stable caller-owned ID for each new record. Batch writes are atomic. Patches keep omitted
fields and accept null for clearing `source_ref`. On `revision_conflict`, reread the record before
retrying. See the [manual graph guide](manual-graph-guide.md) for examples and provenance.

### Upload documents through MCP

Set `MCP_FILE_UPLOADS=true` and recreate the API container to expose `upload_documents`. It accepts
the original bytes of up to eight supported files as standard base64. The client should encode the
files directly. The tool uses the same extraction worker and provider settings as the REST upload.

```json
{
  "workspace_id": "default",
  "files": [
    {
      "filename": "notes.txt",
      "content_base64": "SGVsbG8sIHdvcmxkIQo=",
      "source_path": "archive/notes.txt"
    }
  ]
}
```

Per-file metadata also accepts `description`, `path_root`, and `last_modified_time`. The tool
returns a `batch_id` and per-file results. Poll `get_ingest_job` with the workspace and each
`job_id` until the status is `done` or `failed`. A `save_failed` entry means the file was
never queued and has no job ID.

The default decoded limit is **32 MiB (33,554,432 bytes) per call**, shared by all files in the
call. A single file can use the full allowance. Set `MCP_UPLOAD_MAX_BYTES` to change it. The
JSON body is larger before gzip, and the client, reverse proxy, or extraction provider may impose
its own limits. For a larger upload, split a batch across calls or use the
[multipart REST endpoint](#post-workspaceiduploadbatch).

Clients that implement gzip requests can compress the entire MCP JSON-RPC body and send
`Content-Encoding: gzip` with `Content-Type: application/json`. Plain JSON also works. Both the
compressed and inflated JSON sizes are bounded for gzip requests, and gzip leaves the decoded
file limit unchanged. `Accept-Encoding` controls response compression independently.

### Graph viewer delivery

`show_workspace_graph` accepts `node_label`, `max_depth`, `max_nodes`, `physics` and
`file_path_contains` with the same defaults as the REST viewer.

With `GRAPH_VIEWER_ENABLED=false`, `show_workspace_graph` stays discoverable and returns
`isError: true` for every delivery mode. The error has `status: 403` and code
`graph_viewer_disabled`, with a message identifying the setting that disabled it.

- `delivery="auto"` (default) returns a summary and the graph data for MCP Apps hosts. For
  text clients, the summary includes a browser link when configured or explains how to request HTML.
- `delivery="html"` returns the complete self-contained HTML as an embedded `text/html` resource
  which clients can save as `graph.html`.
- `delivery="link"` returns the live REST viewer URL. Set `MCP_VIEWER_BASE_URL` to an address the
  user's browser can reach. In local Docker Compose, for example, set it to `http://localhost:9622`
  or use the port you publish with `RAG_PORT`, then recreate the API container. For text-only chat,
  ask the agent to use `delivery="link"` and share the returned `viewer_url`. The browser
  authenticates separately through the usual REST mechanism.

Inline and HTML deliveries allow up to 5000 nodes and 1 MiB for the uncompressed MCP tool result.
If the inline result is too large, the tool can return a configured link. Narrow the filters or
lower `max_nodes` when an export exceeds the limit. Viewer URLs have an 8192-byte limit.

### HTTP response compression

REST and MCP negotiate gzip for buffered JSON and HTML of at least 1024 bytes. To request an
uncompressed response, omit `Accept-Encoding` or send `Accept-Encoding: identity`. A gzip-only
request can compress smaller bodies. If the request excludes both codings, the API returns an empty
`406` before executing the operation. SSE and pre-encoded responses pass through. The graph viewer
result limit is measured before HTTP compression.

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

## Manual graph and embedding operations

All routes below start with `/workspace/{id}` and use the same authentication as the rest of the API. Each workspace exposes one graph. Connect a new manual relationship to an extracted entity by looking up the entity and using its returned ID. Exact, Boolean and keyword queries, graph writes and embedding inventories work with PostgreSQL alone. [Manual graph guide](manual-graph-guide.md)

| Method and path | Behavior |
|---|---|
| `GET /graph/entities` | Search by `name`, Boolean `q`, and/or free `text`. Filter by `type` or `origin`. Use `match=exact` for literal equality and `sort=degree` for high-degree browsing. Returns `items`, `total`, and `next_cursor`. |
| `GET/PUT/PATCH/DELETE /graph/entities/{entity_id}` | Read, create, replace, edit, or suppress an entity. Use the current `expected_revision` for edits and deletions. |
| `GET/PUT/PATCH/DELETE /graph/relationships/{edge_id}` | Use endpoint IDs in `source` and `target`. Distinct relationship IDs preserve parallel and reverse edges. |
| `POST /graph/batch` | Apply an atomic payload containing `entities` and/or `relationships` arrays. Each item has an `id`. |
| `GET /graph/entities/{entity_id}/neighbors` | Traverse by ID with `hops`, `direction`, `relationship_type`, and `max_nodes`. |
| `POST /graph/retrieve` | Retrieve ranked entity or relationship seeds plus a bounded neighborhood with sources and indexing coverage. `mode` accepts `keyword`, `semantic`, or `hybrid`. The latter two require `text` and an embedding provider. |
| `POST /graph/entities/merge/preview`, `POST /graph/entities/merge/apply` | Merge explicitly selected duplicates, review conflicts and rewiring, and apply with a preview token. |
| `GET /embeddings/missing` | Page through records with `state=missing\|stale\|needs_embedding`. Filter by `kind` or `origin`. The total counts all current matches before pagination. |
| `POST /embeddings/backfill`, `GET /embeddings/backfill/{job_id}` | Trigger or poll indexing for the configured profile. The trigger returns `202`. |
| `GET /embeddings/consistency` | Report ready, missing, stale and orphaned vectors by record kind. |

Boolean expressions accept uppercase `AND` and `OR`, parentheses, and quoted phrases. For example, `(type:Team OR name:"Alice Chen") AND description:payments`. Supply the same filters with the returned cursor to read the next page. The graph retrieval response reports coverage for the selected seed kind across the workspace. With semantic or hybrid ranking, degree acts as a tie-breaker when requested.

Validation errors return `422`. Revision conflicts return `409`. Provider-dependent requests return `503` with `detail.code` set to `provider_not_configured` or `provider_unavailable` and `detail.roles` identifying the required capability.

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

Hover a node or relationship to see its description and evidence. Click to pin the panel and
expand **Technical details** for record and provenance IDs.

With `GRAPH_VIEWER_ENABLED=false`, this endpoint returns HTTP `403` with JSON containing
`detail.code: graph_viewer_disabled` and a `detail.message` explaining the configuration.

Query parameters:

| Param | Default | Meaning |
|---|---|---|
| `node_label` | `*` | Entity name to center the subgraph on; `*` renders the entire graph. |
| `max_depth` | `3` | Maximum relationship hops expanded out from the starting node(s). |
| `max_nodes` | `5000` | Hard cap on rendered nodes; closest / highest-degree nodes win when truncated. |
| `physics` | `true` | Animated force-directed layout; set `false` for a static layout on large graphs. |
| `file_path_contains` | _(empty)_ | Repeatable case-insensitive substring filter on node and relationship source paths (OR semantics). An empty filter returns the whole graph. |
