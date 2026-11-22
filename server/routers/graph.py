"""REST endpoints for manual graph editing, search and embedding maintenance."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Response

from server import embeddings
from server.deps import require_workspace
from server.graph_operations import (
    find_graph_entities,
    get_embedding_consistency,
    get_graph_neighbors,
    get_graph_record,
    retrieve_graph,
    write_graph_batch,
    write_graph_record,
)
from server.graph_search import cursor_offset, next_cursor
from server.graph_store import (
    apply_merge,
    delete_record,
    graph_transaction,
    preview_merge,
)
from server.providers import require_providers
from server.schemas import (
    Direction,
    EmbeddingConsistency,
    EmbeddingJob,
    EmbeddingState,
    EntityFilters,
    EntityPage,
    EntityPatch,
    EntityWrite,
    GraphBatch,
    GraphContents,
    GraphDeletion,
    GraphEntity,
    GraphID,
    GraphLabel,
    GraphNeighborhood,
    GraphOrigin,
    GraphRelationship,
    GraphRetrieval,
    GraphRetrieve,
    IndexKind,
    MergeApplyRequest,
    MergePreview,
    MergePreviewRequest,
    MergeResult,
    MissingEmbeddingsPage,
    NeighborOptions,
    RelationshipPatch,
    RelationshipWrite,
    Revision,
    SearchMode,
)

router = APIRouter(prefix="/workspace/{workspace_id}")
Workspace = Annotated[dict, Depends(require_workspace)]
PageSize = Annotated[int, Query(ge=1, le=200)]
Cursor = Annotated[str | None, Query(max_length=512)]
Expression = Annotated[str | None, Query(max_length=2000)]


@router.get("/graph/entities", response_model=EntityPage, tags=["graph"])
async def list_entities(
    ws: Workspace,
    name: GraphLabel | None = None,
    match: Literal["exact", "contains"] = "contains",
    type: GraphLabel | None = None,
    origin: GraphOrigin | None = None,
    sort: Literal["degree"] | None = None,
    q: Expression = None,
    text: GraphLabel | None = None,
    mode: SearchMode = "keyword",
    limit: PageSize = 50,
    cursor: Cursor = None,
):
    return await find_graph_entities(
        ws,
        EntityFilters(
            name=name,
            match=match,
            type=type,
            origin=origin,
            sort=sort,
            q=q,
            text=text,
            mode=mode,
            limit=limit,
            cursor=cursor,
        ),
    )


@router.post("/graph/batch", response_model=GraphContents, tags=["graph"])
async def graph_batch(body: GraphBatch, ws: Workspace):
    return await write_graph_batch(body, ws)


@router.get("/graph/entities/{entity_id}", response_model=GraphEntity, tags=["graph"])
async def get_entity(entity_id: GraphID, ws: Workspace):
    return await get_graph_record(ws, "entity", entity_id)


@router.put(
    "/graph/entities/{entity_id}",
    response_model=GraphEntity,
    tags=["graph"],
    responses={201: {"model": GraphEntity, "description": "Entity created"}},
)
async def put_entity(entity_id: GraphID, body: EntityWrite, response: Response, ws: Workspace):
    record, created = await write_graph_record(ws, "entity", entity_id, body)
    response.status_code = 201 if created else 200
    return record


@router.patch("/graph/entities/{entity_id}", response_model=GraphEntity, tags=["graph"])
async def patch_entity(entity_id: GraphID, body: EntityPatch, ws: Workspace):
    return (await write_graph_record(ws, "entity", entity_id, body, partial=True))[0]


@router.delete("/graph/entities/{entity_id}", response_model=GraphDeletion, tags=["graph"])
async def delete_entity(entity_id: GraphID, expected_revision: Revision, ws: Workspace):
    async with graph_transaction(ws, write=True) as conn:
        return await delete_record(conn, ws["id"], "entity", entity_id, expected_revision)


@router.get("/graph/relationships/{edge_id}", response_model=GraphRelationship, tags=["graph"])
async def get_relationship(edge_id: GraphID, ws: Workspace):
    return await get_graph_record(ws, "relationship", edge_id)


@router.put(
    "/graph/relationships/{edge_id}",
    response_model=GraphRelationship,
    tags=["graph"],
    responses={201: {"model": GraphRelationship, "description": "Relationship created"}},
)
async def put_relationship(
    edge_id: GraphID, body: RelationshipWrite, response: Response, ws: Workspace
):
    record, created = await write_graph_record(ws, "relationship", edge_id, body)
    response.status_code = 201 if created else 200
    return record


@router.patch("/graph/relationships/{edge_id}", response_model=GraphRelationship, tags=["graph"])
async def patch_relationship(edge_id: GraphID, body: RelationshipPatch, ws: Workspace):
    return (await write_graph_record(ws, "relationship", edge_id, body, partial=True))[0]


@router.delete("/graph/relationships/{edge_id}", response_model=GraphDeletion, tags=["graph"])
async def delete_relationship(edge_id: GraphID, expected_revision: Revision, ws: Workspace):
    async with graph_transaction(ws, write=True) as conn:
        return await delete_record(conn, ws["id"], "relationship", edge_id, expected_revision)


@router.get(
    "/graph/entities/{entity_id}/neighbors", response_model=GraphNeighborhood, tags=["graph"]
)
async def neighbors(
    entity_id: GraphID,
    ws: Workspace,
    hops: Annotated[int, Query(ge=0, le=3)] = 1,
    direction: Direction = "both",
    relationship_type: GraphLabel | None = None,
    max_nodes: Annotated[int, Query(ge=1, le=1000)] = 100,
):
    return await get_graph_neighbors(
        ws,
        entity_id,
        NeighborOptions(
            hops=hops, direction=direction, relationship_type=relationship_type, max_nodes=max_nodes
        ),
    )


@router.post("/graph/retrieve", response_model=GraphRetrieval, tags=["graph"])
async def retrieve(body: GraphRetrieve, ws: Workspace):
    return await retrieve_graph(body, ws)


@router.post("/graph/entities/merge/preview", response_model=MergePreview, tags=["graph"])
async def merge_preview(body: MergePreviewRequest, ws: Workspace):
    async with graph_transaction(ws, write=True) as conn:
        return await preview_merge(conn, ws["id"], body)


@router.post("/graph/entities/merge/apply", response_model=MergeResult, tags=["graph"])
async def merge_apply(body: MergeApplyRequest, ws: Workspace):
    async with graph_transaction(ws, write=True) as conn:
        return await apply_merge(conn, ws["id"], body)


@router.get("/embeddings/missing", response_model=MissingEmbeddingsPage, tags=["embeddings"])
async def list_missing(
    ws: Workspace,
    kind: IndexKind | None = None,
    origin: GraphOrigin | None = None,
    state: EmbeddingState = "missing",
    limit: PageSize = 50,
    cursor: Cursor = None,
):
    wid = ws["id"]
    criteria = (wid, "embeddings", kind, origin, state, embeddings.embedding_profile())
    offset = cursor_offset(cursor, criteria)
    async with graph_transaction(ws) as conn:
        items, total = await embeddings.missing_embeddings(
            conn, wid, kind, origin, state, limit, offset
        )
    return {
        "items": items,
        "total": total,
        "next_cursor": next_cursor(criteria, offset + limit, total),
    }


@router.post(
    "/embeddings/backfill", response_model=EmbeddingJob, status_code=202, tags=["embeddings"]
)
async def trigger_backfill(ws: Workspace):
    require_providers("embeddings.backfill", "embedding")
    async with graph_transaction(ws, write=True) as conn:
        job = await embeddings.enqueue(conn, ws["id"], embeddings.embedding_profile())
        return await embeddings.job_status(conn, ws["id"], job)


@router.get("/embeddings/backfill/{job_id}", response_model=EmbeddingJob, tags=["embeddings"])
async def backfill_status(job_id: GraphID, ws: Workspace):
    async with graph_transaction(ws) as conn:
        return await embeddings.job_status(conn, ws["id"], job_id)


@router.get("/embeddings/consistency", response_model=EmbeddingConsistency, tags=["embeddings"])
async def check_embeddings(ws: Workspace):
    return await get_embedding_consistency(ws)
