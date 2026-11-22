# PolyGraphRAG

**Give your AI agents a knowledge graph with one MCP endpoint.** Connect at `/mcp` to search workspace evidence, follow relationships, and show interactive graphs. [Connect an MCP client →](#connect-your-agent-with-mcp)

<p align="center">
  <img alt="PolyGraphRAG workflow from multimodal ingestion to knowledge graphs and MCP tools" src="docs/images/hero.png" width="100%">
</p>

**A production FastAPI service for multimodal, multi-project knowledge-graph RAG** — ingest PDFs, Office docs, images, and audio; extract an entity/relationship **knowledge graph**; and query it with dual-level retrieval. Provider-agnostic, backed by **Postgres + pgvector + Apache AGE**.

<p>
  <img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-async-009688">
  <img alt="Postgres" src="https://img.shields.io/badge/Postgres-pgvector%20%2B%20Apache%20AGE-336791">
  <img alt="Tests" src="https://img.shields.io/badge/tests-passing-brightgreen">
  <a href="https://github.com/oleksandr-kushnir/polygraphrag/pkgs/container/polygraphrag"><img alt="Pull from GHCR" src="https://img.shields.io/badge/ghcr.io-pull%20image-2496ED?logo=docker&logoColor=white"></a>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-black">
  <a href="https://www.linkedin.com/in/oleksandr-kushnir-ai/"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-Oleksandr%20Kushnir-0A66C2?logo=linkedin&logoColor=white"></a>
</p>

PolyGraphRAG wraps [RAG-Anything](https://github.com/HKUDS/RAG-Anything) and [LightRAG](https://github.com/HKUDS/LightRAG) in a deployable HTTP API and adds the things you need to actually run graph-RAG as a service: **many isolated projects on one instance**, **asynchronous ingest jobs**, **pluggable model providers per role**, file lifecycle management, and an **interactive graph viewer**.

## Connect your agent with MCP

Point any **Streamable HTTP** MCP client at **`http://localhost:9622/mcp`**. Your client discovers the tools and their inputs automatically. Give a research agent just the tools it needs, or add graph editing to a workflow that maintains the knowledge base.

- **Search with sources.** Retrieve entities, relationships, text chunks, and document references with `search_workspace`, or ask PolyGraphRAG to synthesize a cited answer with `ask_workspace`.
- **Follow the graph.** Find an entity by name, inspect its source evidence, and traverse the actual relationships. Keyword lookup and traversal work with all model providers disabled.
- **Show the graph.** Use `show_workspace_graph` to render an interactive graph in an MCP Apps host, export a self-contained HTML file, or return a browser link.
- **Write facts directly.** Opt in with `MCP_GRAPH_WRITES=true` to insert entities and relationships in one atomic batch or patch records by revision. Manual and extracted facts live in the same graph.
- **Upload original documents.** Opt in with `MCP_FILE_UPLOADS=true` to send files for background extraction and track each job through MCP. The default limit is 32 MiB per call, with optional gzip for the request. [Upload tool reference](docs/api-reference.md#upload-documents-through-mcp)

The server exposes **12 read tools by default**. Set `API_TOKENS` to require the same bearer authentication as REST. For setup, a runnable client example, and the full catalog, see the [MCP tool reference](docs/api-reference.md#mcp).

---

## 📥 Drop in any file → a GraphRAG database (embeddings + knowledge graph)

**No separate OCR service. No parsing pipeline to wire up. No extraction stack to babysit.** One upload endpoint swallows documents, images, and audio alike — OCR, layout parsing, transcription, and entity extraction all happen **inside this one service**. Whatever you send comes out the other side as a **GraphRAG store in Postgres: vector embeddings in pgvector *and* an entity/relationship knowledge graph in Apache AGE**, both queryable together. **22 first-class, verified file types**, and unknown extensions are still attempted (routed through MinerU, ingestion unverified).

| Category | How it's parsed | Extensions |
|----------|-----------------|------------|
| 📄 **Documents** | Vision model (Office → PDF via LibreOffice) | `pdf` · `docx` · `pptx` · `xlsx` |
| 🖼️ **Images** | Vision model, per file | `jpg` · `jpeg` · `png` · `gif` · `bmp` · `tiff` · `webp` |
| 🔊 **Audio** | Whisper transcription | `mp3` · `wav` · `m4a` · `ogg` · `flac` · `opus` · `webm` |
| 📝 **Text** | Ingested directly | `md` · `txt` · `csv` · `html` |

**One pipeline, one store, one query surface.** The extracted text is embedded into **pgvector** for semantic retrieval, and its entities/relationships are lifted into an **Apache AGE knowledge graph** — so queries traverse both the vectors *and* the graph, not just chunk similarity. Point it at a PDF, a screenshot, a spreadsheet, or a voice memo; the plumbing is already done.

<p align="center">
  <img alt="Interactive knowledge-graph viewer for a workspace" src="docs/images/workspace_graph.png" width="100%">
  <br>
  <em>The extracted knowledge graph for a workspace, rendered by the built-in interactive viewer.</em>
</p>

---

## 🔀🤖 A knowledge backend for workflows and AI agents

**Upload from a workflow or an MCP client, then explore the same graph in a browser.** A workflow can send documents from a drive, inbox, or webhook to `POST /workspace/{id}/upload/batch`. An MCP client can send files with `upload_documents`. PolyGraphRAG handles parsing and extraction in the background, then agents can retrieve the evidence through MCP or REST.

Give each project its own workspace on a single deployment. Use **Swagger UI** at `/docs` to try the REST endpoints and `/openapi.json` to build typed integrations. If your source system already knows the facts, write them directly and start querying the graph without model credentials. [Manual graph guide](docs/manual-graph-guide.md)

---

## Why it's interesting

- 🗂️ **Multi-project — multiple isolated knowledge graphs.** Every *workspace* is its own graph **and** its own vector namespace. Spin up as many projects as you like on a single deployment and ingest, query, and visualize each one completely independently. No cross-talk between corpora.
- 🔌 **Provider-agnostic, per role.** The extraction LLM, the query LLM, the vision model, embeddings, and Whisper are each independently pointed at OpenAI **or any OpenAI-compatible endpoint** (OpenRouter, DeepSeek, Ollama, vLLM, LM Studio, TEI…). Run cheap high-volume extraction on one provider and fast synthesis on another — configured entirely by environment variables.
- 🧩 **Genuinely multimodal.** PDFs and images go through a vision model, Office docs via LibreOffice, audio via Whisper, and text/markdown/CSV directly — 22 file types, one upload endpoint.
- 🕸️ **A real knowledge graph, not just chunks.** LightRAG extracts entities and relationships and stores them in **Apache AGE** (graph) + **pgvector** (embeddings) inside one Postgres. Query modes span local, global, hybrid, and naive retrieval, and `query/data` returns the structured evidence without an LLM answer.
- ⚙️ **Built to operate.** Async batch ingestion with per-file job tracking and integrity verification, soft-delete + restore of workspaces, per-file delete that correctly preserves entities shared with other documents, and a container `HEALTHCHECK`.
- ✅ **A fully mocked test suite.** The suite stubs RAG-Anything, LightRAG, and the database, so you can test REST and MCP with **no Postgres and no API keys**.

## Architecture

```mermaid
flowchart LR
    client([REST client]) -->|HTTP| api[FastAPI service<br/>PolyGraphRAG]
    agent([MCP client]) -->|Streamable HTTP| api

    subgraph providers[Pluggable model providers &#40;per role&#41;]
      llm[Extraction + Query LLM]
      vis[Vision]
      emb[Embeddings]
      whis[Whisper]
    end

    api --> raga[RAG-Anything<br/>parsing + ingest pipeline]
    api --> lr[LightRAG<br/>graph build + retrieval]
    raga --> lr
    lr <--> llm
    raga <--> vis
    lr <--> emb
    raga <--> whis

    subgraph pg[Postgres]
      age[(Apache AGE<br/>knowledge graph)]
      vec[(pgvector<br/>embeddings)]
      meta[(file / job / workspace<br/>metadata)]
    end
    lr --> age
    lr --> vec
    api --> meta
```

Each workspace maps to a physical LightRAG namespace: the bootstrap `default` workspace (seeded only into an empty registry) uses the shared `chunk_entity_relation` AGE graph, and every workspace created via the API gets its own `{workspace}_chunk_entity_relation` graph plus workspace-scoped vector rows — that's what keeps projects isolated. Workspaces are peers; none is delete-protected.

## Requires Postgres to run

PolyGraphRAG's storage backend **is** Postgres (pgvector + Apache AGE + pg_trgm) — it can't serve requests without it. You don't need to set that up by hand: `docker compose up` builds and starts a correctly-extended Postgres alongside the API. The only thing that runs without a database is the **test suite**, which is fully mocked.

## Quickstart

```bash
cp .env.example .env          # set POSTGRES_PASSWORD
docker compose pull           # fetch prebuilt images from GHCR (fast path)
docker compose up -d          # start Postgres(+AGE) and the API
curl localhost:9622/health    # -> {"status":"ok"}
```

Prefer to build the images yourself (e.g. after changing the code)? Swap the middle
two commands for `docker compose up -d --build`.

Open **http://localhost:9622/docs** for the full interactive Swagger UI.

For a first MCP call, connect at **`http://localhost:9622/mcp`** and run `list_workspaces`.
A new database has a `default` workspace. You can start with manual graph data using only
Postgres. Configure model providers in `.env` for document extraction, vector search, and
generated answers. [Configuration guide](docs/configuration.md)

> **Auth is off by default** (the ports are loopback-only). To require credentials, set
> `API_TOKENS` in `.env` (comma-separated). Every endpoint except `/health` then needs a token:
> machines send `-H "Authorization: Bearer <token>"`; in a browser enter any username with the
> token as the password. Serve over TLS whenever you expose the service beyond loopback.
>
> **Scope:** single-instance, low-usage service — run **one worker/replica** (ingest job state
> is in-process). Not designed for high load or horizontal scaling.

### Ingest and query (end to end)

```bash
# 1. Create an isolated project/graph
curl -X POST localhost:9622/all-workspaces/create \
  -H 'content-type: application/json' \
  -d '{"id":"acme","name":"Acme Corp"}'

# 2. Upload one or more files (ingestion runs in the background)
curl -X POST localhost:9622/workspace/acme/upload/batch \
  -F 'files=@handbook.pdf'
# -> {"batch_id":"...","jobs":[{"job_id":"ab12cd34", ...}]}

# 3. Poll a job until it's "done" (statuses: pending/processing/retrying/done/failed)
curl localhost:9622/workspace/acme/status/ab12cd34

# 4. Query the knowledge graph
curl -X POST localhost:9622/workspace/acme/query \
  -H 'content-type: application/json' \
  -d '{"query":"How do we handle refunds?","mode":"mix"}'

# 5. Open the interactive graph in a browser
#    http://localhost:9622/workspace/acme/graph.html
```

**Citations point at your real files.** Upload with `source_path` + `path_root` metadata and every `/query` reference's `file_path` is the **real, openable document path** you provided (resolved from Postgres, not LightRAG's internal name) — so answers cite files your own tooling can open.

## API at a glance

| Area | Endpoints |
|------|-----------|
| Discovery & health | `GET /` (service card) · `GET /health` |
| Workspaces | `GET /all-workspaces/list` · `POST /all-workspaces/create` · `GET /workspace/{id}` · `DELETE /workspace/{id}` · `POST /workspace/{id}/restore` |
| Ingestion | `POST /workspace/{id}/upload/batch` · `GET /workspace/{id}/batch/{batch_id}` · `GET /workspace/{id}/status/{job_id}` · `GET /workspace/{id}/jobs` |
| Files | `GET /workspace/{id}/files` · `DELETE /workspace/{id}/file/delete` |
| Query | `POST /workspace/{id}/query` · `POST /workspace/{id}/query/data` |
| Manual graph | Atomic writes, revisioned edits, lookup, traversal, and merges under [`/workspace/{id}/graph`](docs/api-reference.md#manual-graph-and-embedding-operations) |
| Embeddings | Consistency, missing vectors, and backfill under [`/workspace/{id}/embeddings`](docs/api-reference.md#manual-graph-and-embedding-operations) |
| Visualization | `GET /workspace/{id}/graph.html` |
| MCP | [`POST /mcp`](docs/api-reference.md#mcp) for tool discovery and invocation over Streamable HTTP |

Full request/response details, query modes, and examples live in **[docs/api-reference.md](docs/api-reference.md)**.

## Configuration

Everything is environment-driven. The headline knob is **per-role provider routing**: for each of `LLM`, `QUERY_LLM`, `VISION`, `EMBEDDING`, and `WHISPER`, an empty `*_BASE_URL` means OpenAI, and a set one points at any OpenAI-compatible endpoint authenticated with the matching `*_API_KEY`.

A cost-optimized example — **DeepSeek (via OpenRouter) for the LLM, OpenAI for embeddings**:

```bash
LLM_MODEL=deepseek/deepseek-chat
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...
EMBEDDING_MODEL=text-embedding-3-small     # blank EMBEDDING_BASE_URL -> OpenAI
OPENAI_API_KEY=sk-...
```

See **[.env.example](.env.example)** for deployment settings and **[docs/configuration.md](docs/configuration.md)** for provider routing and embedding backfill after a model change.

## Testing

```bash
python -m venv .venv && . .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                          # all green — no Postgres, no keys
```

The suite mocks RAG-Anything, LightRAG, and the Postgres pool — no external services, no keys.

Two smoke tests live in **[scripts/](scripts/)**:

```bash
python scripts/smoke_test.py          # in-process ASGI smoke (stubs, no DB/keys)
./scripts/smoke_test_docker.sh        # live HTTP smoke against a running stack
```

## Project layout

```
.
├── server/              # the FastAPI service (package: app + routers + services)
│   ├── __init__.py      #   app assembly, lifespan, /health, router wiring
│   ├── config.py        #   env parsing + provider routing
│   ├── schemas.py       #   Pydantic request models
│   ├── db.py            #   schema init + metadata persistence
│   ├── workspaces.py    #   per-workspace RAGAnything instance registry
│   ├── ingest.py        #   document parsing + ingestion pipeline
│   ├── worker.py        #   background ingest worker
│   ├── llm.py           #   LLM / embedding shims
│   ├── references.py    #   citation + real-path resolution
│   ├── graph.py         #   knowledge-graph HTML rendering
│   ├── mcp.py           #   MCP tool catalog and Streamable HTTP transport
│   ├── auth.py          #   opt-in API-token middleware
│   ├── deps.py          #   shared endpoint dependencies
│   └── routers/         #   query, documents, and workspace-registry endpoints
├── Dockerfile           # app image
├── requirements.txt     # runtime deps (lightrag-hku pinned to a release with the AGE fix)
├── docker-compose.yml   # postgres (pgvector + AGE) + polygraphrag services
├── db/                  # Postgres image build + init.sql (extensions)
├── scripts/             # in-process + live-docker smoke tests
├── tests/               # fully mocked tests (unit + end-to-end)
└── docs/                # architecture, API reference, configuration, LightRAG internals
```

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — services, request flow, workspace isolation, and the storage model.
- **[MCP tool reference](docs/api-reference.md#mcp)** — connect clients, discover tools, and deliver interactive graphs.
- **[Manual graph guide](docs/manual-graph-guide.md)** — load, edit, and traverse your own facts, with or without model providers.
- **[REST API reference](docs/api-reference.md)** — endpoints, parameters, and examples.
- **[docs/configuration.md](docs/configuration.md)** — all environment variables and the provider-routing matrix.
- **[docs/lightrag-internals.md](docs/lightrag-internals.md)** — how ingestion builds the graph and how retrieval works under the hood.

## Upstream contribution

The one bug that used to force this project to vendor a 6,700-line fork of LightRAG's Postgres layer — Apache AGE silently dropping edge properties — was **fixed upstream by [Oleksandr Kushnir](https://github.com/oleksandr-kushnir)** and merged into LightRAG as [HKUDS/LightRAG#3052](https://github.com/HKUDS/LightRAG/pull/3052). PolyGraphRAG now runs **stock, current LightRAG** with no patch and no version-lock. The live smoke test confirms edges persist their full property maps (`weight`, `keywords`, `description`, `source_id`, `file_path`) on `lightrag-hku==1.5.4`.

## Author

**Oleksandr Kushnir** — [GitHub](https://github.com/oleksandr-kushnir) · [LinkedIn](https://www.linkedin.com/in/oleksandr-kushnir-ai/)

## Credits & license

PolyGraphRAG is a service layer over [RAG-Anything](https://github.com/HKUDS/RAG-Anything) and [LightRAG](https://github.com/HKUDS/LightRAG) (both MIT) by HKUDS. **Document parsing is powered by [MinerU](https://github.com/opendatalab/MinerU)** ([MinerU Open Source License](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md), Apache-2.0 based) by OpenDataLab.

Full attribution and the third-party license inventory are in [NOTICE](NOTICE) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). PolyGraphRAG's own code is licensed under the [MIT License](LICENSE).
