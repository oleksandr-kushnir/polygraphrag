"""MCP delivery modes for the shared graph viewer."""

import re
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

from mcp import types
from pydantic import AnyHttpUrl, AnyUrl, TypeAdapter

from server.graph import graph_viewer_payload, render_graph_snapshot
from server.graph_store import graph_error
from server.graph_viewer import select_viewer_graph
from server.schemas import GraphViewerResult, ShowGraphToolInput

VIEWER_APP_URI = AnyUrl("ui://polygraphrag/graph.html")
VIEWER_APP_MIME = "text/html;profile=mcp-app"
VIEWER_RESULT_BYTES = 1024 * 1024
VIEWER_URL_BYTES = 8192


def validate_viewer_base_url(configured: str) -> str | None:
    if not configured:
        return None
    try:
        # Reject delimiters even if they introduce an empty userinfo, query or fragment.
        if re.search(r"[\\?#\s\x00-\x1f\x7f]", configured) or "@" in urlsplit(configured).netloc:
            raise ValueError
        if not re.match(r"^https?://[^/]", configured, re.IGNORECASE):
            raise ValueError
        return str(TypeAdapter(AnyHttpUrl).validate_python(configured)).rstrip("/")
    except ValueError as exc:
        raise ValueError(
            "MCP_VIEWER_BASE_URL must be an absolute HTTP(S) URL without userinfo, query or fragment"
        ) from exc


def _viewer_url(base_url: str | None, workspace_id: str, options: ShowGraphToolInput) -> str | None:
    if base_url is None:
        return None
    params = [
        ("node_label", options.node_label),
        ("max_depth", str(options.max_depth)),
        ("max_nodes", str(options.max_nodes)),
        ("physics", str(options.physics).lower()),
        *(("file_path_contains", path) for path in options.file_path_contains),
    ]
    return f"{base_url}/workspace/{workspace_id}/graph.html?{urlencode(params)}"


def _result(
    metadata: GraphViewerResult,
    text: str,
    resource: types.EmbeddedResource | None = None,
    graph: dict | None = None,
) -> types.CallToolResult:
    content: list[types.ContentBlock] = [types.TextContent(type="text", text=text)]
    if resource:
        content.append(resource)
    return types.CallToolResult(
        content=content,
        structuredContent=metadata.model_dump(mode="json", exclude_none=True),
        _meta={"polygraphrag/graph": graph} if graph is not None else None,
    )


async def show_workspace_graph(
    ws: dict, options: ShowGraphToolInput, base_url: str | None
) -> types.CallToolResult:
    workspace_id = ws["id"]
    url = _viewer_url(base_url, workspace_id, options)
    too_long = url is not None and len(url.encode("ascii")) > VIEWER_URL_BYTES
    if too_long:
        url = None
    title = f"Knowledge graph for {workspace_id}"
    if options.delivery == "link":
        if too_long:
            raise graph_error(
                422, "viewer_url_too_long", "Shorten the graph filters and try again."
            )
        if url is None:
            raise graph_error(
                422, "viewer_link_unavailable", 'Request delivery="html" for an offline snapshot.'
            )
        return _result(
            GraphViewerResult(
                workspace_id=workspace_id, title=title, representation="link", viewer_url=url
            ),
            f"{title}\n{url}",
        )

    kg = await select_viewer_graph(ws, options)
    graph = graph_viewer_payload(kg, options.physics)
    if options.delivery == "html":
        resource = types.EmbeddedResource(
            type="resource",
            resource=types.TextResourceContents(
                uri=AnyUrl(f"polygraphrag://snapshot/{uuid4()}/graph.html"),
                mimeType="text/html",
                text=render_graph_snapshot(graph),
            ),
        )
        result = _result(
            GraphViewerResult(
                workspace_id=workspace_id, title=title, representation="html", viewer_url=url
            ),
            f"Save the attached HTML as graph.html to view {workspace_id} offline.",
            resource=resource,
        )
    else:
        fallback = url or 'Request delivery="html" for an offline snapshot.'
        result = _result(
            GraphViewerResult(
                workspace_id=workspace_id, title=title, representation="inline", viewer_url=url
            ),
            f"{title} ({len(graph['nodes'])} nodes, {len(graph['links'])} relationships).\n{fallback}",
            graph=graph,
        )

    if len(result.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")) <= (
        VIEWER_RESULT_BYTES
    ):
        return result
    if options.delivery == "auto" and url:
        return _result(
            GraphViewerResult(
                workspace_id=workspace_id, title=title, representation="link", viewer_url=url
            ),
            f"Open {title.lower()} in your browser.\n{url}",
        )
    details = {"viewer_url": url} if url else {}
    raise graph_error(
        422,
        "viewer_result_too_large",
        "Narrow the graph with a smaller max_nodes or more specific filters.",
        limit=VIEWER_RESULT_BYTES,
        **details,
    )
