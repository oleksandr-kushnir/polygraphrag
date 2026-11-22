"""Workspace query operations shared by the REST and MCP interfaces."""

import server
from server import config
from server.deps import _internal_error
from server.graph_queries import effective_answer, effective_query_data
from server.graph_store import graph_storage_ready
from server.references import (
    _build_references,
    _clean_needles,
    _path_matches_any,
    _resolve_block_file_paths,
    _rewrite_answer_refs,
)
from server.schemas import QueryDataRequest, QueryRequest


def _query_param(QueryParam, *, mode, include_references=None, top_k=None, chunk_top_k=None):
    """Build a QueryParam, omitting optional knobs so LightRAG's own defaults apply."""
    kwargs = {"mode": mode}
    if include_references is not None:
        kwargs["include_references"] = include_references
    if top_k is not None:
        kwargs["top_k"] = top_k
    if chunk_top_k is not None:
        kwargs["chunk_top_k"] = chunk_top_k
    return QueryParam(**kwargs)


async def ask_workspace(req: QueryRequest, ws: dict):
    if graph_storage_ready():
        return await effective_answer(req, ws)
    rag_instance = await server.get_workspace_rag(ws["id"])
    try:
        from lightrag import QueryParam

        raw = await rag_instance.lightrag.aquery_llm(
            req.query,
            param=_query_param(
                QueryParam,
                mode=req.mode,
                include_references=req.include_references,
                top_k=req.top_k,
            ),
        )
    except Exception as exc:
        raise _internal_error(exc, "query") from exc

    result = (raw.get("llm_response") or {}).get("content", "")
    raw_refs = (raw.get("data") or {}).get("references")
    # Build the key→row map even to only rewrite the prose; emit the structured refs when asked.
    references, meta_map = await _build_references(
        raw_refs, ws["lightrag_workspace"], answered_model=config.QUERY_LLM_MODEL
    )
    result = _rewrite_answer_refs(result, raw_refs, meta_map)
    return {"result": result, "references": references if req.include_references else []}


async def search_workspace(req: QueryDataRequest, ws: dict):
    if graph_storage_ready():
        return await effective_query_data(req, ws)
    rag_instance = await server.get_workspace_rag(ws["id"])
    needles = _clean_needles(req.file_path_contains)  # blank/empty => no filter (all data)
    # Post-filter runs after retrieval; widen the candidate set so a narrow folder still has hits.
    top_k = req.top_k * config.RAG_FILTER_TOPK_BOOST if needles else req.top_k
    chunk_top_k = None if not needles else req.top_k * config.RAG_FILTER_TOPK_BOOST
    try:
        from lightrag import QueryParam

        raw = await rag_instance.lightrag.aquery_data(
            req.query,
            param=_query_param(QueryParam, mode=req.mode, top_k=top_k, chunk_top_k=chunk_top_k),
        )
    except Exception as exc:
        raise _internal_error(exc, "query/data") from exc

    data = raw.get("data") or {}
    # Resolve internal LightRAG keys in entity/relationship/chunk file_path -> real Postgres paths
    # BEFORE filtering, so file_path_contains matches the real path consistently with references.
    await _resolve_block_file_paths(data, ws["lightrag_workspace"])
    if needles:
        data["entities"] = [
            e
            for e in (data.get("entities") or [])
            if _path_matches_any(e.get("file_path"), needles)
        ]
        data["relationships"] = [
            r
            for r in (data.get("relationships") or [])
            if _path_matches_any(r.get("file_path"), needles)
        ]
        data["chunks"] = [
            c for c in (data.get("chunks") or []) if _path_matches_any(c.get("file_path"), needles)
        ]
    references, _ = (
        await _build_references(data.get("references"), ws["lightrag_workspace"])
        if req.include_references
        else ([], {})
    )
    if needles:
        references = [r for r in references if _path_matches_any(r.get("file_path"), needles)]
    return {
        "status": raw.get("status"),
        "message": raw.get("message"),
        "data": {
            "entities": data.get("entities") or [],
            "relationships": data.get("relationships") or [],
            "chunks": data.get("chunks") or [],
            "references": references,
        },
        "metadata": raw.get("metadata") or {},
    }
