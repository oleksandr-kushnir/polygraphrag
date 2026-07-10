"""Environment-derived configuration and LLM-phase routing.

All runtime configuration is resolved from environment variables at import time and exposed as
module-level constants. Importing this module also configures the root logger (before the heavy
libraries in server/__init__.py emit anything at import). Keeping it dependency-free (only the
stdlib) lets the test suite re-execute this file under a controlled env to assert config
resolution, and lets every other server submodule import config without a circular dependency.
"""

import contextvars
import logging
import os


def _log_level_from_env(value: str | None) -> int:
    """Map a LOG_LEVEL env string to a logging constant; unknown/empty -> INFO (never crash
    the process on a typo). Case-insensitive; surrounding whitespace is ignored."""
    return getattr(logging, (value or "INFO").strip().upper(), logging.INFO)


# One knob for all logging. Default INFO keeps LightRAG / RAG-Anything DEBUG-level output —
# which can include prompt and document content — out of the logs; set LOG_LEVEL=DEBUG to opt
# into full verbosity for troubleshooting. The noisy third-party loggers are intentionally NOT
# pinned here: without an explicit level they inherit root's LOG_LEVEL (single source of truth).
LOG_LEVEL = _log_level_from_env(os.getenv("LOG_LEVEL"))
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

WORKING_DIR = os.getenv("WORKING_DIR", "/app/data")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5.4-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Per-role endpoints (embeddings / vision / whisper) ---
# Every model role is independently routable to any OpenAI-compatible endpoint (incl. a local
# Ollama / vLLM / LM Studio / TEI / faster-whisper server), same idiom as LLM_BASE_URL below:
#   <ROLE>_BASE_URL empty -> OpenAI (uses OPENAI_API_KEY)
#   <ROLE>_BASE_URL set   -> that endpoint, authed with <ROLE>_API_KEY (falls back to OPENAI_API_KEY)
# Nothing is hardcoded to OpenAI anymore. See .env.example for each endpoint's specific requirements.

# Embeddings. NOTE: EMBEDDING_MODEL + EMBEDDING_DIM name the pgvector tables
# (lightrag_vdb_*_<model>_<dim>d). Changing the model/dim points at NEW empty tables — the
# existing corpus must be re-ingested, and the value must stay consistent across the workspace.
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").strip() or None
EMBEDDING_API_KEY = (
    (os.getenv("EMBEDDING_API_KEY", "").strip() or OPENAI_API_KEY)
    if EMBEDDING_BASE_URL
    else OPENAI_API_KEY
)

# Vision (PDF/image extraction). Must be a MULTIMODAL model; PDFs are sent as an OpenAI-style
# {"type":"file"} part, so endpoints lacking that will fail on .pdf (images are more portable).
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "").strip() or None
VISION_API_KEY = (
    (os.getenv("VISION_API_KEY", "").strip() or OPENAI_API_KEY)
    if VISION_BASE_URL
    else OPENAI_API_KEY
)
_VISION_IS_OPENAI = (VISION_BASE_URL is None) or ("openai.com" in VISION_BASE_URL)

# Whisper (audio transcription). Endpoint must expose the OpenAI /v1/audio/transcriptions shape.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
WHISPER_BASE_URL = os.getenv("WHISPER_BASE_URL", "").strip() or None
WHISPER_API_KEY = (
    (os.getenv("WHISPER_API_KEY", "").strip() or OPENAI_API_KEY)
    if WHISPER_BASE_URL
    else OPENAI_API_KEY
)

# --- LLM endpoint (entity/relationship extraction + query synthesis) ---
# The *text* LLM (_llm_func) is routable to any third-party / local OpenAI-compatible endpoint,
# same as embeddings/vision/whisper above — no role is hardcoded to OpenAI.
#   LLM_BASE_URL empty  -> OpenAI (legacy behaviour, uses OPENAI_API_KEY)
#   LLM_BASE_URL set    -> that endpoint, authenticated with LLM_API_KEY
# e.g. OpenRouter: LLM_BASE_URL=https://openrouter.ai/api/v1, LLM_MODEL=deepseek/deepseek-v4-flash
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None
LLM_API_KEY = (
    (os.getenv("LLM_API_KEY", "").strip() or OPENAI_API_KEY) if LLM_BASE_URL else OPENAI_API_KEY
)
# OpenAI's gpt-5.x reject `max_tokens` and require `max_completion_tokens`; classic
# OpenAI-compatible providers (OpenRouter, DeepSeek, ...) want `max_tokens`.
_LLM_IS_OPENAI = (LLM_BASE_URL is None) or ("openai.com" in LLM_BASE_URL)

# --- Phase-split text LLM (extraction vs query) ---
# The LLM_* vars above are the EXTRACTION config: the model that reads whole documents
# to pull out entities/relationships at ingest time (high token volume — route it to a
# cheap provider like OpenRouter). Query-time LLM work (keyword extraction + prose
# synthesis) is latency-sensitive and best served by a fast provider; benchmarks show
# synthesis on gpt-5.4-mini is 2-33x faster than on deepseek. Override it independently:
#   QUERY_LLM_MODEL / QUERY_LLM_BASE_URL / QUERY_LLM_API_KEY
# Any unset QUERY_* var falls back to the corresponding extraction value, so leaving all
# three blank preserves the historical single-model behaviour. Empty QUERY_LLM_BASE_URL
# forces OpenAI (uses OPENAI_API_KEY), matching the LLM_BASE_URL convention above.
QUERY_LLM_MODEL = os.getenv("QUERY_LLM_MODEL", "").strip() or LLM_MODEL
# Blank (or unset) QUERY_LLM_BASE_URL reuses the extraction endpoint (LLM_BASE_URL) — this is what
# "leave QUERY_* blank to reuse the extraction LLM" means, and it holds even under docker-compose,
# which passes unset vars through as empty strings. To send query-time work to a *different*
# provider (e.g. force OpenAI while extraction runs on OpenRouter), set QUERY_LLM_BASE_URL
# explicitly (e.g. https://api.openai.com/v1).
QUERY_LLM_BASE_URL = os.getenv("QUERY_LLM_BASE_URL", "").strip() or LLM_BASE_URL
if QUERY_LLM_BASE_URL is None:
    QUERY_LLM_API_KEY = OPENAI_API_KEY
else:
    QUERY_LLM_API_KEY = os.getenv("QUERY_LLM_API_KEY", "").strip() or (
        LLM_API_KEY if QUERY_LLM_BASE_URL == LLM_BASE_URL else OPENAI_API_KEY
    )
QUERY_LLM_IS_OPENAI = (QUERY_LLM_BASE_URL is None) or ("openai.com" in QUERY_LLM_BASE_URL)

# Which LLM config _llm_func should use. LightRAG calls the same llm_model_func for both
# extraction (during ainsert) and querying (during aquery); this contextvar lets one
# process route the two phases to different providers. Default is the hot query path;
# _process_file flips it to "extract" for the duration of an ingest.
_llm_phase: contextvars.ContextVar[str] = contextvars.ContextVar("llm_phase", default="query")


def _active_llm_cfg() -> tuple[str, str | None, str, bool]:
    """(model, base_url, api_key, is_openai) for the current phase — read live so tests
    (and any future runtime reconfig) can monkeypatch the module-level config vars."""
    if _llm_phase.get() == "extract":
        return LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, _LLM_IS_OPENAI
    return QUERY_LLM_MODEL, QUERY_LLM_BASE_URL, QUERY_LLM_API_KEY, QUERY_LLM_IS_OPENAI


def _llm_call_kwargs(kwargs: dict, is_openai: bool | None = None) -> dict:
    """Pick the token/temperature kwargs valid for the active LLM endpoint.

    OpenAI path: pass temperature + max_completion_tokens through unchanged.
    Third-party path: pass temperature + max_tokens, translating LightRAG's
    max_completion_tokens -> max_tokens so the output cap is still honoured.
    `is_openai` defaults to the extraction flag for backward-compatible callers.
    """
    if is_openai is None:
        is_openai = _LLM_IS_OPENAI
    if is_openai:
        return {k: v for k, v in kwargs.items() if k in ("temperature", "max_completion_tokens")}
    out = {k: v for k, v in kwargs.items() if k in ("temperature", "max_tokens")}
    if "max_tokens" not in out and "max_completion_tokens" in kwargs:
        out["max_tokens"] = kwargs["max_completion_tokens"]
    return out


POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ragdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "raguser")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_WORKSPACE = os.getenv("POSTGRES_WORKSPACE", "default")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

# --- API auth (opt-in) ---
# Comma-separated bearer tokens. Blank => auth disabled (the loopback/dev default, so the
# existing local workflow and the test suite need no credentials). When set, every endpoint
# except /health requires a matching token, sent either as `Authorization: Bearer <token>`
# (machines) or via HTTP Basic with the token as the password (browsers). Read live so tests
# can monkeypatch it, mirroring the _active_llm_cfg() idiom above.
API_TOKENS = [t.strip() for t in os.getenv("API_TOKENS", "").split(",") if t.strip()]

# Ingestion-integrity guard: after ainsert returns, confirm LightRAG actually finished
# (doc_status == 'processed', and — when content is non-trivial — that it produced graph
# entities). The pipeline marks a doc FAILED on extraction error but ainsert does NOT re-raise,
# so without this a transient LLM timeout leaves a doc searchable-but-unlinked yet marked done.
RAG_REQUIRE_GRAPH_EXTRACTION = os.getenv("RAG_REQUIRE_GRAPH_EXTRACTION", "true").lower() == "true"
RAG_MIN_CONTENT_FOR_ENTITIES = int(os.getenv("RAG_MIN_CONTENT_FOR_ENTITIES", "200"))
# file_path_contains post-filter (/query/data, /graph.html): matching runs AFTER retrieval, so
# when a filter is active we retrieve a larger candidate set (top_k/chunk_top_k/max_nodes * boost)
# to reduce the chance a narrow folder is crowded out before filtering.
RAG_FILTER_TOPK_BOOST = int(os.getenv("RAG_FILTER_TOPK_BOOST", "5"))

# Bootstrap workspace. Seeded ONCE, only into a completely empty registry, so a fresh
# install is usable out of the box. It is an ordinary, deletable workspace — there is no
# "primary"/special status. Its data physically lives under POSTGRES_WORKSPACE (the legacy
# single-workspace value, read ONLY to seed this row on first boot); afterwards the
# rag_workspaces table is the single source of truth and this seed never runs again.
SEED_WORKSPACE_ID = "default"
SEED_WORKSPACE_NAME = "Default"
SEED_WORKSPACE_DESCRIPTION = "Default workspace."
