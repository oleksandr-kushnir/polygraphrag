"""Query + knowledge-graph endpoints: /workspace/{id}/query, /query/data, and graph.html."""

from fastapi import APIRouter, Depends, Query
from fastapi import Path as PathParam
from fastapi.responses import HTMLResponse

from server.deps import require_workspace
from server.graph import _build_graph_html
from server.graph_viewer import select_viewer_graph
from server.queries import ask_workspace, search_workspace
from server.schemas import GraphViewerOptions, QueryDataRequest, QueryRequest, QueryResponse

router = APIRouter()


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
    return await ask_workspace(req, ws)


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
    return await search_workspace(req, ws)


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
    options = GraphViewerOptions(
        node_label=node_label,
        max_depth=max_depth,
        max_nodes=max_nodes,
        physics=physics,
        file_path_contains=file_path_contains,
    )
    return HTMLResponse(_build_graph_html(await select_viewer_graph(ws, options), physics))
