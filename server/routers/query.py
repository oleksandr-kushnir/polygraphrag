"""Query + knowledge-graph endpoints: /workspace/{id}/query, /query/data, and graph.html."""

from fastapi import APIRouter, Depends, Query
from fastapi import Path as PathParam
from fastapi.responses import HTMLResponse

import server
from server import config
from server.deps import _internal_error, require_workspace
from server.graph import _build_graph_html
from server.references import (
    _build_references,
    _clean_needles,
    _path_matches_any,
    _resolve_block_file_paths,
    _resolve_graph_paths,
    _rewrite_answer_refs,
)
from server.schemas import QueryDataRequest, QueryRequest, QueryResponse

router = APIRouter()


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


@router.post(
    "/workspace/{workspace_id}/query",
    summary="Ask a question (LLM answer + citations)",
    description=(
        "Run a RAG query against the workspace and return a synthesised natural-language answer. "
        "When `include_references` is true, the response also lists the source documents used — "
        "each reference's `file_path` is the **real, openable document path** you supplied at upload "
        "(resolved server-side from Postgres), never LightRAG's internal name. "
        "For raw retrieved entities/relationships/chunks instead of a prose answer, use `/query/data`."
    ),
    response_model=QueryResponse,
    responses={404: {"description": "Workspace not found or soft-deleted"}},
)
async def query(req: QueryRequest, ws: dict = Depends(require_workspace)):
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


@router.post(
    "/workspace/{workspace_id}/query/data",
    summary="Retrieve raw graph/vector data for a query (no LLM answer)",
    description=(
        "Run retrieval for a query and return the raw matched **entities, relationships, and text "
        "chunks** (plus references when `include_references` is true) without generating a prose "
        "answer. Use this when you want structured context to feed into your own prompt or to "
        "inspect what the corpus contains.\n\n"
        "**Scope to a folder/file with `file_path_contains`**: a list of case-insensitive substrings "
        "matched (OR) against each result's `file_path` — an item is kept if its path contains ANY of "
        "them. Empty/omitted = no filtering. The retrieval budget is auto-boosted when a filter is set, "
        "but matching happens *after* retrieval, so a very narrow folder may return fewer items than "
        "exist in it. Example body: "
        '`{"query":"...","file_path_contains":["/opt/data/workspace/career/"]}`.'
    ),
    responses={
        # The data blocks are LightRAG pass-through (loosely shaped), so the contract is
        # documented with a static example rather than a model that could drop fields.
        200: {
            "description": "Raw retrieved data with resolved real file paths",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Query completed successfully",
                        "data": {
                            "entities": [
                                {
                                    "entity_name": "Refund Policy",
                                    "entity_type": "concept",
                                    "description": "30-day refund window for unused licences.",
                                    "file_path": "/data/corpus/policies/refunds.pdf",
                                }
                            ],
                            "relationships": [
                                {
                                    "src_id": "Refund Policy",
                                    "tgt_id": "Billing Team",
                                    "description": "Refunds are executed by the billing team.",
                                    "file_path": "/data/corpus/policies/refunds.pdf",
                                }
                            ],
                            "chunks": [
                                {
                                    "content": "Refunds are granted within 30 days…",
                                    "file_path": "/data/corpus/policies/refunds.pdf",
                                }
                            ],
                            "references": [
                                {
                                    "reference_id": "1",
                                    "file_path": "/data/corpus/policies/refunds.pdf",
                                    "job_id": "ab12cd34",
                                    "file_description": "Refund policy 2026",
                                    "last_modified_time": "2026-01-05T09:00:00",
                                    "uploaded_at": "2026-01-06T10:00:00",
                                    "llm_model_extracted": "gpt-5.4-mini",
                                }
                            ],
                        },
                        "metadata": {},
                    }
                }
            },
        },
        404: {"description": "Workspace not found or soft-deleted"},
    },
)
async def query_data(req: QueryDataRequest, ws: dict = Depends(require_workspace)):
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


@router.get(
    "/workspace/{workspace_id}/graph.html",
    response_class=HTMLResponse,
    summary="Render this workspace's knowledge graph as an interactive HTML page",
    description=(
        "Returns a **self-contained, offline-capable HTML page** (D3.js v7 inlined, drawn on "
        "an HTML canvas) showing the workspace's LightRAG knowledge graph as an interactive "
        "force-directed diagram. Nodes are entities (colored by entity type, sized by their connection "
        "degree); edges are relationships. Hover a node or edge to see its full properties.\n\n"
        "Open the URL directly in a browser, or save the response body to a `.html` file. "
        "This returns rendered HTML, **not** JSON — for machine-readable graph data use "
        "`POST /workspace/{workspace_id}/query/data` instead.\n\n"
        "Tip: lower `max_nodes` / `max_depth` for a faster, less cluttered view of large graphs; "
        "set `physics=false` to freeze the layout once it settles.\n\n"
        "**Scope to a folder/file with `file_path_contains`**: repeat the query param to pass several "
        "case-insensitive substrings; a node is kept if its `file_path` contains ANY of them (OR). "
        "Empty = the whole graph. Filtering runs *after* graph selection (`max_nodes` is auto-boosted "
        "when set), so a very narrow folder may render sparsely. Example: "
        "`?file_path_contains=/opt/data/workspace/career/&file_path_contains=/opt/data/workspace/projects/`."
    ),
    responses={
        200: {"content": {"text/html": {}}, "description": "Interactive graph HTML page"},
        404: {"description": "Workspace not found or soft-deleted"},
    },
)
async def graph_html(
    workspace_id: str = PathParam(description="Public workspace id (slug) whose graph to render."),
    node_label: str = Query(
        "*",
        description="Entity name to center the subgraph on. Use '*' (default) for the entire graph.",
    ),
    max_depth: int = Query(
        3,
        ge=1,
        description="Maximum number of relationship hops to expand out from the starting node(s). Default 3.",
    ),
    max_nodes: int = Query(
        5000,
        ge=1,
        description="Hard cap on nodes returned; closest / highest-degree nodes win when truncated. Default 5000.",
    ),
    physics: bool = Query(
        True,
        description="Animate a force-directed layout (true, default). Set false for a static layout on large graphs.",
    ),
    file_path_contains: list[str] = Query(
        default_factory=list,
        description=(
            "Optional folder/file scope filter: repeatable case-insensitive substrings. Omit or leave "
            "empty to render the WHOLE graph (no filtering, the default). When provided, keep only nodes "
            "whose file_path contains ANY of them (OR; blank strings ignored). Applied after graph "
            "selection, so a very narrow folder may render sparsely."
        ),
    ),
    ws: dict = Depends(require_workspace),
):
    rag_instance = await server.get_workspace_rag(ws["id"])
    needles = _clean_needles(file_path_contains)  # blank/empty => no filter (whole graph)
    # Post-filter runs after graph selection; widen the fetch so a narrow folder still has nodes.
    fetch_nodes = max_nodes * config.RAG_FILTER_TOPK_BOOST if needles else max_nodes
    try:
        kg = await rag_instance.lightrag.get_knowledge_graph(
            node_label=node_label,
            max_depth=max_depth,
            max_nodes=fetch_nodes,
        )
    except Exception as exc:
        raise _internal_error(exc, "graph.html") from exc
    # Resolve node + edge file_path (internal keys) -> real Postgres paths before filtering/
    # rendering, so tooltips never show LightRAG's internal name and the folder filter matches the
    # real path. Nodes and edges share one fetch.
    await _resolve_graph_paths(list(kg.nodes) + list(kg.edges), ws["lightrag_workspace"])
    if needles:
        kg.nodes = [
            n for n in kg.nodes if _path_matches_any((n.properties or {}).get("file_path"), needles)
        ][:max_nodes]  # re-apply the hard cap after the boosted fetch; priority order preserved
    return HTMLResponse(_build_graph_html(kg, physics))
