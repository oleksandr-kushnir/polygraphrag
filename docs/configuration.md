# Configuration

Everything is configured through environment variables (loaded from `.env` by Docker Compose). This page documents every variable and the per-role provider-routing model.

## Provider routing (the important part)

PolyGraphRAG never hardcodes OpenAI. Each **model role** is routed independently using the same rule:

| If `<ROLE>_BASE_URL` is… | The role calls… | Authenticated with… |
|---|---|---|
| **empty** | OpenAI | `OPENAI_API_KEY` |
| **set** | that OpenAI-compatible endpoint | `<ROLE>_API_KEY` (falls back to `OPENAI_API_KEY` if blank) |

The roles are `LLM` (extraction), `QUERY_LLM` (query synthesis), `VISION`, `EMBEDDING`, and `WHISPER`.

This means you can mix providers freely — for example run document extraction on a cheap high-volume model, answer synthesis on a fast one, and embeddings on OpenAI, all at once.

### OpenAI vs. compatible token params

OpenAI's `gpt-5.x` models require `max_completion_tokens`; classic OpenAI-compatible providers (OpenRouter, DeepSeek, …) expect `max_tokens`. PolyGraphRAG detects this from the base URL (`openai.com` ⇒ OpenAI dialect) and sends the correct parameter automatically — you don't configure anything.

## Text LLM: extraction vs. query

LightRAG calls one text LLM for two very different jobs:

- **Extraction** (`LLM_*`) — reads whole documents at ingest to pull out entities/relationships. High token volume; route it somewhere cheap.
- **Query** (`QUERY_LLM_*`) — keyword extraction + prose answer synthesis at query time. Latency-sensitive; route it somewhere fast.

Any `QUERY_LLM_*` variable left blank falls back to the corresponding `LLM_*` value, so leaving all three blank gives you a single-model setup.

## Full variable reference

### Postgres

| Variable | Default | Notes |
|---|---|---|
| `POSTGRES_PASSWORD` | — | **Required.** Compose refuses to start without it. |
| `POSTGRES_DB` | `ragdb` | Database name. |
| `POSTGRES_USER` | `raguser` | Role name. |
| `POSTGRES_PORT` | `5432` | Host port (bound to loopback). |
| `POSTGRES_HOST` | `db` | Set by compose to the DB service name. |
| `POSTGRES_WORKSPACE` | `default` | Physical LightRAG namespace backing the primary workspace. |

### Text LLM

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `gpt-5.4-mini` | Extraction model. |
| `LLM_BASE_URL` | _(empty ⇒ OpenAI)_ | Extraction endpoint. |
| `LLM_API_KEY` | _(→ `OPENAI_API_KEY`)_ | Extraction key. |
| `QUERY_LLM_MODEL` | _(→ `LLM_MODEL`)_ | Query-time model. |
| `QUERY_LLM_BASE_URL` | _(→ `LLM_BASE_URL`)_ | Query-time endpoint. |
| `QUERY_LLM_API_KEY` | _(→ `LLM_API_KEY`/`OPENAI_API_KEY`)_ | Query-time key. |

### Embeddings

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | ⚠️ Names the vector tables. |
| `EMBEDDING_DIM` | `1536` | ⚠️ Names the vector tables. |
| `EMBEDDING_BASE_URL` | _(empty ⇒ OpenAI)_ | Embeddings endpoint. |
| `EMBEDDING_API_KEY` | _(→ `OPENAI_API_KEY`)_ | Embeddings key. |

> **⚠️ Don't change `EMBEDDING_MODEL`/`EMBEDDING_DIM` after ingesting.** They are baked into the pgvector table names (`lightrag_vdb_*_<model>_<dim>d`). Changing either points the service at **new, empty** tables — the existing corpus must be re-ingested — and the value must stay consistent across a deployment.

### Vision

| Variable | Default | Notes |
|---|---|---|
| `VISION_MODEL` | `gpt-5.4-mini` | **Must be multimodal.** PDFs are sent as OpenAI-style `file` parts, so endpoints lacking that will fail on `.pdf` (images are more portable). |
| `VISION_BASE_URL` | _(empty ⇒ OpenAI)_ | Vision endpoint. |
| `VISION_API_KEY` | _(→ `OPENAI_API_KEY`)_ | Vision key. |

### Whisper

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `whisper-1` | Audio transcription model. |
| `WHISPER_BASE_URL` | _(empty ⇒ OpenAI)_ | Must expose `/v1/audio/transcriptions`. |
| `WHISPER_API_KEY` | _(→ `OPENAI_API_KEY`)_ | Whisper key. |

### Behavior

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | Fallback key for any role whose own key is blank. |
| `WORKING_DIR` | `/app/data` | On-disk root for uploaded files (a Docker volume). |
| `MAX_RETRIES` | `5` | Ingestion retry budget. |
| `RAG_REQUIRE_GRAPH_EXTRACTION` | `true` | Fail an ingest if a non-trivial document produced zero graph entities. |
| `RAG_PORT` | `9622` | Host port for the API (loopback). |

## Worked example — DeepSeek + OpenAI

Cheap extraction on DeepSeek (via OpenRouter), high-quality embeddings on OpenAI:

```bash
LLM_MODEL=deepseek/deepseek-chat
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-...          # OpenRouter key
# QUERY_LLM_* blank -> reuse DeepSeek for query synthesis
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=            # blank -> OpenAI
OPENAI_API_KEY=sk-...          # embeddings (and vision, for PDFs/images)
```

To run entirely locally, point the LLM/embedding roles at Ollama or vLLM instead (e.g. `LLM_BASE_URL=http://host.docker.internal:11434/v1`).
