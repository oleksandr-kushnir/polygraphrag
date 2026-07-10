# Architecture

PolyGraphRAG is a thin, operational service layer over two libraries — **RAG-Anything** (multimodal parsing + ingestion) and **LightRAG** (graph construction + retrieval) — with **Postgres** as the single source of truth.

## Components

| Component | Role |
|---|---|
| **`polygraphrag` (FastAPI)** | The HTTP API. Owns workspaces, ingest jobs, file metadata, and model-provider routing. Structured as the `server/` package (app assembly + `routers/` by resource + service modules). |
| **RAG-Anything** | Parses each uploaded file into text/structure (LibreOffice for Office docs, a vision model for PDFs/images, Whisper for audio) and drives ingestion. |
| **LightRAG** | Chunks text, extracts entities/relationships with the LLM, merges them into a knowledge graph, and runs dual-level retrieval at query time. |
| **Postgres** | Stores everything: the knowledge graph (**Apache AGE**), embeddings (**pgvector**), fuzzy-text indexes (**pg_trgm**), and PolyGraphRAG's own workspace/job/file metadata tables. |

External model providers (LLM, vision, embeddings, Whisper) are reached over HTTP and are independently configurable per role — see [configuration.md](configuration.md).

## Request flow

**Ingestion** (`POST /workspace/{id}/upload/batch`):

1. Files are written to `WORKING_DIR` and a `job` row is created per file (`pending`).
2. A background worker pulls each job, runs RAG-Anything parsing, then LightRAG ingestion under the **extraction** LLM phase (routes to `LLM_*`).
3. After `ainsert` returns, the service **verifies** the document actually reached `processed` and — when the content is non-trivial and `RAG_REQUIRE_GRAPH_EXTRACTION=true` — that it produced graph entities. Only then is the job marked `processed`.
4. File identity, content hash, doc id, and the extracting model are recorded for later listing/deletion.

**Query** (`POST /workspace/{id}/query`):

1. Runs under the **query** LLM phase (routes to `QUERY_LLM_*`).
2. LightRAG performs keyword extraction + dual-level retrieval over the workspace's AGE graph and pgvector tables, then synthesizes an answer.
3. References are enriched from the file-metadata table before returning.

`POST /workspace/{id}/query/data` stops after step 2 and returns the structured evidence with no LLM answer.

## Workspace isolation (multi-project)

Isolation is the core design property: **each workspace is a separate graph and a separate vector namespace.**

- Workspaces are **peers** — there is no special "primary" workspace and none is delete-protected. On a *fresh, empty* registry a single ordinary `default` workspace is seeded (mapped to the physical LightRAG workspace `POSTGRES_WORKSPACE`, using the bare `chunk_entity_relation` AGE graph) so the install is usable out of the box; once any workspace exists the seed never runs again, so a deleted `default` stays deleted.
- Every workspace `w` created via the API gets its own AGE graph `{w}_chunk_entity_relation` and workspace-scoped rows in the shared `lightrag_*` tables (keyed by a `workspace` column).
- Deleting a workspace (with `purge=true`) drops *only* its own dedicated graph, deletes *only* its workspace-scoped rows from every `lightrag_*` table, removes its file metadata, and clears its on-disk files — every DELETE is filtered on that workspace's physical name, so another workspace's data is never touched.

A small registry table (`rag_workspaces`) maps **public slug → physical namespace** and tracks soft-delete state; a per-workspace lock serializes instance creation and inserts.

## Storage model

Within one Postgres database:

- **Apache AGE** — the property graph of entities and relationships (`*_chunk_entity_relation` graphs).
- **pgvector** — entity, relationship, and chunk embeddings in `lightrag_vdb_*_<model>_<dim>d` tables (the model+dim are in the table name, which is why they must stay stable).
- **pg_trgm** — trigram indexes supporting fuzzy text matching.
- **PolyGraphRAG tables** — `rag_workspaces` (registry) and `rag_file_metadata` (per-file system of record: content hash, doc id, stored path, extracting model, timestamps).

The Postgres image (`db/Dockerfile`) is `pgvector/pgvector:pg16` with Apache AGE compiled from source; `db/init.sql` enables the three extensions on first boot.

## Why the vendored patch is gone

Older deployments of this service shipped a 6,700-line vendored fork of LightRAG's `postgres_impl.py` to fix one bug: Apache AGE silently dropped edge properties written via `SET r += {map}`. That fix was **contributed upstream by [Oleksandr Kushnir](https://github.com/oleksandr-kushnir)** and merged into LightRAG as [HKUDS/LightRAG#3052](https://github.com/HKUDS/LightRAG/pull/3052), so PolyGraphRAG simply pins `lightrag-hku` to a release that contains it — no fork, and no version-lock. See [lightrag-internals.md](lightrag-internals.md#edge-properties-on-apache-age) for detail.
