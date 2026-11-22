"""PolyGraphRAG tools over stateless MCP Streamable HTTP."""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from fastapi import HTTPException
from mcp import McpError, types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyUrl, BaseModel, ValidationError
from starlette.types import Receive, Scope, Send

from server import config
from server.deps import require_workspace
from server.graph import graph_app_html
from server.graph_operations import (
    find_graph_entities,
    get_embedding_consistency,
    get_graph_neighbors,
    get_graph_record,
    retrieve_graph,
    write_graph_batch,
    write_graph_record,
)
from server.graph_store import graph_error
from server.jobs import lookup_ingest_job
from server.mcp_viewer import (
    VIEWER_APP_MIME,
    VIEWER_APP_URI,
    show_workspace_graph,
    validate_viewer_base_url,
)
from server.queries import ask_workspace, search_workspace
from server.schemas import (
    AskWorkspaceInput,
    EmbeddingConsistency,
    EmptyToolInput,
    EntityPage,
    EntityPatch,
    EntityPatchToolInput,
    EntityToolInput,
    FindEntitiesToolInput,
    GraphBatchToolInput,
    GraphContents,
    GraphEntity,
    GraphNeighborhood,
    GraphRelationship,
    GraphRetrieval,
    GraphViewerResult,
    IngestJobStatus,
    JobToolInput,
    NeighborsToolInput,
    QueryResponse,
    RelationshipPatch,
    RelationshipPatchToolInput,
    RelationshipToolInput,
    RetrieveGraphToolInput,
    SearchWorkspaceInput,
    ShowGraphToolInput,
    WorkspaceListResponse,
    WorkspaceOverview,
    WorkspaceToolInput,
)
from server.workspaces import get_workspace_overview, list_workspace_records

_Input = TypeVar("_Input", bound=BaseModel)
_Result = dict[str, Any] | types.CallToolResult


@dataclass(frozen=True)
class _Tool:
    definition: types.Tool
    execute: Callable[[dict], Awaitable[_Result]]


def _tool(
    name: str,
    description: str,
    input_model: type[_Input],
    output_model: type[BaseModel] | None,
    run: Callable[[_Input], Awaitable[Any]],
    *,
    write: bool = False,
    ui: bool = False,
) -> _Tool:
    definition = types.Tool(
        name=name,
        description=description,
        inputSchema=input_model.model_json_schema(),
        outputSchema=output_model.model_json_schema() if output_model else None,
        annotations=(
            types.ToolAnnotations(readOnlyHint=False, destructiveHint=True)
            if write
            else types.ToolAnnotations(readOnlyHint=True)
        ),
        _meta={"ui": {"resourceUri": str(VIEWER_APP_URI)}} if ui else None,
    )

    async def execute(arguments: dict) -> _Result:
        try:
            result = await run(input_model.model_validate(arguments))
        except ValidationError as exc:
            issues = exc.errors(include_input=False, include_context=False, include_url=False)
            message = "; ".join(
                f"{'.'.join(map(str, issue['loc'])) or 'arguments'}: {issue['msg']}"
                for issue in issues
            )
            raise graph_error(422, "invalid_arguments", message) from exc
        if output_model and not isinstance(result, types.CallToolResult):
            return output_model.model_validate(result).model_dump(mode="json")
        return result

    return _Tool(definition, execute)


async def _ingest_job_status(ws: dict, job_id: str) -> dict:
    job = await lookup_ingest_job(ws, job_id)
    return {"job_id": job_id, "status": job["status"], "attempts": job.get("attempts")}


async def _patch_entity(ws: dict, args: EntityPatchToolInput) -> dict:
    patch = EntityPatch.model_validate(
        args.model_dump(exclude_unset=True, exclude={"workspace_id", "entity_id"})
    )
    return (await write_graph_record(ws, "entity", args.entity_id, patch, partial=True))[0]


async def _patch_relationship(ws: dict, args: RelationshipPatchToolInput) -> dict:
    patch = RelationshipPatch.model_validate(
        args.model_dump(exclude_unset=True, exclude={"workspace_id", "edge_id"})
    )
    return (await write_graph_record(ws, "relationship", args.edge_id, patch, partial=True))[0]


_WorkspaceInput = TypeVar(
    "_WorkspaceInput",
    bound=WorkspaceToolInput | AskWorkspaceInput | SearchWorkspaceInput,
)


def _workspace_tool(
    name: str,
    description: str,
    input_model: type[_WorkspaceInput],
    output_model: type[BaseModel] | None,
    run: Callable[[dict, _WorkspaceInput], Awaitable[Any]],
    *,
    write: bool = False,
    ui: bool = False,
) -> _Tool:
    async def in_workspace(args: _WorkspaceInput):
        return await run(await require_workspace(args.workspace_id), args)

    return _tool(name, description, input_model, output_model, in_workspace, write=write, ui=ui)


def _catalog(viewer_base_url: str | None, writes: bool) -> dict[str, _Tool]:
    tools = [
        _tool(
            "list_workspaces",
            "List active workspaces to find the public ID of the corpus to query.",
            EmptyToolInput,
            WorkspaceListResponse,
            lambda _: list_workspace_records(),
        ),
        _workspace_tool(
            "search_workspace",
            "Retrieve entities, relationships, text chunks and sources to compose your own answer.",
            SearchWorkspaceInput,
            None,
            lambda ws, args: search_workspace(args, ws),
        ),
        _workspace_tool(
            "ask_workspace",
            "Ask PolyGraphRAG to synthesize an answer, with source references by default.",
            AskWorkspaceInput,
            QueryResponse,
            lambda ws, args: ask_workspace(args, ws),
        ),
        _workspace_tool(
            "get_workspace_overview",
            "Get document, chunk, entity, relationship and ingest counts for an active workspace.",
            WorkspaceToolInput,
            WorkspaceOverview,
            lambda ws, _: get_workspace_overview(ws),
        ),
        _workspace_tool(
            "get_ingest_job",
            "Check a known ingest job's status and attempt count.",
            JobToolInput,
            IngestJobStatus,
            lambda ws, args: _ingest_job_status(ws, args.job_id),
        ),
        _workspace_tool(
            "find_graph_entities",
            "Find graph entities by label, type, origin or search expression. Pass next_cursor "
            "with the same filters to continue. Keyword mode works without providers.",
            FindEntitiesToolInput,
            EntityPage,
            find_graph_entities,
        ),
        _workspace_tool(
            "get_graph_entity",
            "Read an entity by its public ID, including revision and provenance.",
            EntityToolInput,
            GraphEntity,
            lambda ws, args: get_graph_record(ws, "entity", args.entity_id),
        ),
        _workspace_tool(
            "get_graph_relationship",
            "Read a relationship by its public ID, including direction, revision and provenance.",
            RelationshipToolInput,
            GraphRelationship,
            lambda ws, args: get_graph_record(ws, "relationship", args.edge_id),
        ),
        _workspace_tool(
            "get_graph_neighbors",
            "Follow edges from an entity ID with explicit direction and bounds.",
            NeighborsToolInput,
            GraphNeighborhood,
            lambda ws, args: get_graph_neighbors(ws, args.entity_id, args),
        ),
        _workspace_tool(
            "retrieve_graph",
            "Find ranked entity or relationship seeds and return their bounded graph and sources. "
            "Supply q or text. Semantic and hybrid modes require text and embeddings.",
            RetrieveGraphToolInput,
            GraphRetrieval,
            lambda ws, args: retrieve_graph(args, ws),
        ),
        _workspace_tool(
            "get_embedding_consistency",
            "Check counts of indexed, missing and stale embeddings without running indexing.",
            WorkspaceToolInput,
            EmbeddingConsistency,
            lambda ws, _: get_embedding_consistency(ws),
        ),
        _workspace_tool(
            "show_workspace_graph",
            "Show the interactive graph in a compatible host. Use html for an offline snapshot "
            "or link for the live REST viewer when a browser URL is configured.",
            ShowGraphToolInput,
            GraphViewerResult,
            lambda ws, args: show_workspace_graph(ws, args, viewer_base_url),
            ui=True,
        ),
    ]
    if writes:
        tools.extend(
            [
                _workspace_tool(
                    "write_graph_batch",
                    "Insert or replace manual graph facts atomically. Use stable IDs and provide "
                    "expected_revision for any changed existing record. Supply source_ref for evidence.",
                    GraphBatchToolInput,
                    GraphContents,
                    lambda ws, args: write_graph_batch(args, ws),
                    write=True,
                ),
                _workspace_tool(
                    "patch_graph_entity",
                    "Edit the supplied entity fields using its observed expected_revision. "
                    "Use null only to clear source_ref.",
                    EntityPatchToolInput,
                    GraphEntity,
                    _patch_entity,
                    write=True,
                ),
                _workspace_tool(
                    "patch_graph_relationship",
                    "Edit the supplied relationship fields using its observed expected_revision. "
                    "Use null only to clear source_ref.",
                    RelationshipPatchToolInput,
                    GraphRelationship,
                    _patch_relationship,
                    write=True,
                ),
            ]
        )
    return {tool.definition.name: tool for tool in tools}


def _tool_error(status: int, detail: dict | str) -> types.CallToolResult:
    if isinstance(detail, str):
        code = "request_failed"
        if status == 404 and detail.startswith("Workspace "):
            code = "workspace_not_found"
        elif status == 404 and detail.startswith("Job "):
            code = "job_not_found"
        elif status == 503:
            code = "database_unavailable"
        elif status == 500:
            code = "internal_error"
        detail = {"code": code, "message": detail}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps({"status": status, **detail}))],
        isError=True,
    )


def create_session_manager() -> StreamableHTTPSessionManager:
    viewer_base_url = validate_viewer_base_url(config.MCP_VIEWER_BASE_URL)
    catalog = _catalog(viewer_base_url, config.MCP_GRAPH_WRITES)
    app: Server = Server("PolyGraphRAG")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [tool.definition for tool in catalog.values()]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> _Result:
        tool = catalog.get(name)
        if tool is None:
            return _tool_error(404, {"code": "unknown_tool", "message": "Tool is not available."})
        try:
            return await tool.execute(arguments)
        except HTTPException as exc:
            return _tool_error(exc.status_code, exc.detail)
        except Exception:
            logging.exception("MCP tool %s failed", name)
            return _tool_error(500, "Internal server error")

    @app.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                name="PolyGraphRAG graph viewer",
                uri=VIEWER_APP_URI,
                mimeType=VIEWER_APP_MIME,
            )
        ]

    @app.read_resource()
    async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        if uri != VIEWER_APP_URI:
            raise McpError(types.ErrorData(code=types.INVALID_PARAMS, message="Unknown resource"))
        return [ReadResourceContents(content=graph_app_html(), mime_type=VIEWER_APP_MIME)]

    return StreamableHTTPSessionManager(
        app=app,
        stateless=True,
        json_response=True,
        security_settings=TransportSecuritySettings(
            allowed_hosts=["polygraphrag:9622", "localhost:*", "127.0.0.1:*", "[::1]:*"],
            allowed_origins=[
                "http://polygraphrag:9622",
                "http://localhost:*",
                "http://127.0.0.1:*",
                "http://[::1]:*",
            ],
        ),
    )


class MCPTransport:
    """Serve the exact /mcp route with the session manager for this app lifespan."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await scope["app"].state.mcp_manager.handle_request(scope, receive, send)
