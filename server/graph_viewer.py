"""Select the graph shown by the REST and MCP viewers."""

import server
from server import config
from server.deps import _internal_error
from server.graph_queries import effective_knowledge_graph
from server.graph_store import graph_error, graph_storage_ready
from server.references import _clean_needles, _path_matches_any, _resolve_graph_paths
from server.schemas import GraphViewerOptions


async def require_graph_viewer() -> None:
    if not config.GRAPH_VIEWER_ENABLED:
        raise graph_error(
            403,
            "graph_viewer_disabled",
            "Graph viewing is disabled by server configuration (GRAPH_VIEWER_ENABLED=false).",
        )


async def select_viewer_graph(ws: dict, options: GraphViewerOptions):
    needles = _clean_needles(options.file_path_contains)  # blank/empty => no filter (whole graph)
    # Post-filter runs after graph selection; widen the fetch so a narrow folder still has nodes.
    fetch_nodes = options.max_nodes * config.RAG_FILTER_TOPK_BOOST if needles else options.max_nodes
    if graph_storage_ready():
        kg = await effective_knowledge_graph(
            ws, options.node_label, options.max_depth, min(fetch_nodes, 5000), needles
        )
        kg.nodes = kg.nodes[: options.max_nodes]
        return kg
    rag_instance = await server.get_workspace_rag(ws["id"])
    try:
        kg = await rag_instance.lightrag.get_knowledge_graph(
            node_label=options.node_label,
            max_depth=options.max_depth,
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
        ]
    kg.nodes = kg.nodes[: options.max_nodes]
    return kg
