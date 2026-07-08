"""Per-workspace RAGAnything/LightRAG instance registry.

Builds and caches one RAGAnything instance per public workspace id, lazily on first use, guarded
by a per-workspace lock. The instance cache, lock table, and asyncpg pool are shared runtime state
owned by the package root (server.*) and read here at call time.
"""

import asyncio
from pathlib import Path

from fastapi import HTTPException
from raganything import RAGAnything

import server
from server import config
from server.llm import _embedding_func, _llm_func, _vision_func


async def _get_ws_lock(workspace_id: str) -> asyncio.Lock:
    """Get (or lazily create) the per-workspace lock. Guards instance creation and
    serialises inserts for that workspace."""
    async with server._registry_lock:
        return server._ws_locks.setdefault(workspace_id, asyncio.Lock())


async def _lookup_workspace(workspace_id: str):
    """Return the rag_workspaces row for an ACTIVE (not soft-deleted) workspace, else None."""
    if server._db_pool is None:
        return None
    return await server._db_pool.fetchrow(
        "SELECT id, name, description, lightrag_workspace, is_primary "
        "FROM rag_workspaces WHERE id = $1 AND deleted_at IS NULL",
        workspace_id,
    )


async def _build_workspace_rag(workspace_id: str, physical_workspace: str) -> RAGAnything:
    """Construct + initialize a RAGAnything/LightRAG pair for one workspace.
    `workspace_id` is the public id (names the on-disk working_dir); `physical_workspace`
    is the LightRAG `workspace=` value that namespaces Postgres rows + the AGE graph."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    working_dir = str(Path(config.WORKING_DIR) / workspace_id)
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    embedding_func = EmbeddingFunc(
        embedding_dim=config.EMBEDDING_DIM,
        max_token_size=8192,
        model_name=config.EMBEDDING_MODEL,
        func=_embedding_func,
    )
    lightrag_instance = LightRAG(
        working_dir=working_dir,
        llm_model_func=_llm_func,
        embedding_func=embedding_func,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="PGGraphStorage",
        doc_status_storage="PGDocStatusStorage",
        workspace=physical_workspace,
    )
    await lightrag_instance.initialize_storages()
    return RAGAnything(
        llm_model_func=_llm_func,
        vision_model_func=_vision_func,
        embedding_func=embedding_func,
        lightrag=lightrag_instance,
    )


async def get_workspace_rag(workspace_id: str) -> RAGAnything:
    """Return the cached RAGAnything for a workspace, building it lazily on first use.
    Raises HTTPException(404) if the workspace is unknown or soft-deleted."""
    cached = server._rag_instances.get(workspace_id)
    if cached is not None:
        return cached
    lock = await _get_ws_lock(workspace_id)
    async with lock:
        cached = server._rag_instances.get(workspace_id)
        if cached is not None:
            return cached
        row = await _lookup_workspace(workspace_id)
        if row is None:
            raise HTTPException(404, f"Workspace {workspace_id!r} not found")
        instance = await _build_workspace_rag(workspace_id, row["lightrag_workspace"])
        server._rag_instances[workspace_id] = instance
        return instance
