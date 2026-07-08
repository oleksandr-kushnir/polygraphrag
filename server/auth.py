"""API-token authentication: credential parsing plus the opt-in gate applied as HTTP middleware.

Auth is disabled unless API_TOKENS is configured. The token set is read from the `server` package
at call time (server.API_TOKENS) so tests — and a live reconfigure — can toggle it without
re-importing this module. Registered on the app by server/__init__.py via app.middleware("http").
"""

import base64
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse

import server


def _auth_enabled() -> bool:
    """True when at least one API token is configured (read live so tests can toggle it)."""
    return bool(server.API_TOKENS)


def _token_valid(token: str) -> bool:
    """Constant-time membership check against the configured tokens."""
    return any(secrets.compare_digest(token, t) for t in server.API_TOKENS)


def _extract_credential(header: str) -> str | None:
    """Pull the presented secret out of an Authorization header, supporting both transports:
    `Bearer <token>` (machines) and `Basic <base64(user:pass)>` (browsers — the token is the
    password, username is ignored). Returns None if the header is absent/malformed."""
    if not header:
        return None
    scheme, _, rest = header.partition(" ")
    scheme = scheme.lower()
    if scheme == "bearer":
        return rest.strip() or None
    if scheme == "basic":
        try:
            decoded = base64.b64decode(rest.strip()).decode("utf-8", "replace")
        except (ValueError, TypeError):
            return None
        _, sep, password = decoded.partition(":")
        return password if sep else None
    return None


async def _require_auth(request: Request, call_next):
    """Gate every request behind API_TOKENS when auth is enabled. Applied as middleware (not a
    route dependency) so it also covers the auto-generated /docs, /redoc and /openapi.json.
    /health stays open for liveness probes. A 401 carries `WWW-Authenticate: Basic` so browsers
    show a native login prompt and then auto-attach the credentials to same-origin requests."""
    if _auth_enabled() and request.url.path != "/health":
        cred = _extract_credential(request.headers.get("authorization", ""))
        if not (cred and _token_valid(cred)):
            return JSONResponse(
                {"detail": "Unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="PolyGraphRAG"'},
            )
    return await call_next(request)
