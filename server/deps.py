"""Shared endpoint dependencies and small HTTP helpers used across the routers.

Kept in a leaf module (importable by every router without importing the app package back) so the
routers can depend on it without a circular import through server/__init__.py.
"""

import logging
import re

from fastapi import HTTPException

from server import workspaces

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


def _is_valid_slug(slug: str) -> bool:
    return bool(_SLUG_RE.match(slug or ""))


async def require_workspace(workspace_id: str) -> dict:
    """Path dependency: validate the slug and confirm the workspace is active.
    Returns the registry row (with `id` = public id and `lightrag_workspace` = physical
    workspace). 404 if the slug is malformed, unknown, or soft-deleted."""
    if not _is_valid_slug(workspace_id):
        raise HTTPException(404, f"Workspace {workspace_id!r} not found")
    row = await workspaces._lookup_workspace(workspace_id)
    if row is None:
        raise HTTPException(404, f"Workspace {workspace_id!r} not found")
    return row


def _batch_summary(entries: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    counts["total"] = len(entries)
    return counts


def _batch_response(batch_id: str, entries: list[dict]) -> dict:
    return {"batch_id": batch_id, "summary": _batch_summary(entries), "jobs": entries}


def _internal_error(exc: Exception, context: str) -> HTTPException:
    """Log the real error server-side (with traceback) and return a client-safe generic 500.
    Keeps internal exception text — which can reveal implementation/query details — out of the
    HTTP response. Use as `raise _internal_error(exc, "query")`."""
    logging.exception("%s failed: %s", context, exc)
    return HTTPException(500, "Internal server error")
