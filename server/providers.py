"""Provider capability checks shared by manual graph and the existing API."""

from server import config
from server.graph_store import graph_error


def available(role: str) -> bool:
    model, endpoint, secret = {
        "embedding": (config.EMBEDDING_MODEL, config.EMBEDDING_BASE_URL, config.EMBEDDING_API_KEY),
        "extraction": (config.LLM_MODEL, config.LLM_BASE_URL, config.LLM_API_KEY),
        "query": (config.QUERY_LLM_MODEL, config.QUERY_LLM_BASE_URL, config.QUERY_LLM_API_KEY),
        "vision": (config.VISION_MODEL, config.VISION_BASE_URL, config.VISION_API_KEY),
        "transcription": (config.WHISPER_MODEL, config.WHISPER_BASE_URL, config.WHISPER_API_KEY),
    }[role]
    return bool(model and (endpoint or secret))


def require_providers(operation: str, *roles: str) -> None:
    missing = [role for role in roles if not available(role)]
    if missing:
        raise graph_error(
            503,
            "provider_not_configured",
            "Configure the required model provider",
            operation=operation,
            roles=missing,
        )


def provider_unavailable(operation: str, *roles: str):
    return graph_error(
        503,
        "provider_unavailable",
        "The configured model provider could not complete the operation",
        operation=operation,
        roles=list(roles),
    )
