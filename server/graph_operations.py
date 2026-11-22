"""Shared graph operations for REST and MCP."""

from server import embeddings
from server.graph_search import cursor_offset, find_records, neighborhood, next_cursor
from server.graph_store import graph_error, graph_transaction, read_record, write_record
from server.schemas import (
    EntityFilters,
    GraphBatch,
    GraphKind,
    GraphRetrieve,
    GraphWrite,
    NeighborOptions,
)


async def find_graph_entities(ws: dict, filters: EntityFilters):
    wid = ws["id"]
    profile = embeddings.embedding_profile() if filters.mode != "keyword" else None
    criteria = (
        wid,
        "entities",
        filters.name,
        filters.match,
        filters.type,
        filters.origin,
        filters.sort,
        filters.q,
        filters.text,
        filters.mode,
        profile,
    )
    offset = cursor_offset(filters.cursor, criteria)
    vector = None
    if filters.mode != "keyword":
        if not filters.text:
            raise graph_error(422, "text_required", "Semantic and hybrid search require text")
        vector = await embeddings.query_embedding(filters.text, "graph.entities")
    async with graph_transaction(ws) as conn:
        items, total = await find_records(
            conn,
            wid,
            "entity",
            name=filters.name,
            match=filters.match,
            entity_type=filters.type,
            origin=filters.origin,
            sort=filters.sort,
            q=filters.q,
            text=filters.text,
            mode=filters.mode,
            limit=filters.limit,
            offset=offset,
            query_vector=vector,
            profile=profile,
        )
    return {
        "items": items,
        "total": total,
        "next_cursor": next_cursor(criteria, offset + filters.limit, total),
    }


async def write_graph_batch(body: GraphBatch, ws: dict):
    wid = ws["id"]
    async with graph_transaction(ws, write=True) as conn:
        entities = [
            (await write_record(conn, wid, "entity", item.id, item))[0] for item in body.entities
        ]
        relationships = [
            (await write_record(conn, wid, "relationship", item.id, item))[0]
            for item in body.relationships
        ]
    return {"entities": entities, "relationships": relationships}


async def get_graph_record(ws: dict, kind: GraphKind, record_id: str):
    async with graph_transaction(ws) as conn:
        return await read_record(conn, ws["id"], kind, record_id)


async def write_graph_record(
    ws: dict, kind: GraphKind, record_id: str, body: GraphWrite, *, partial: bool = False
):
    async with graph_transaction(ws, write=True) as conn:
        return await write_record(conn, ws["id"], kind, record_id, body, partial=partial)


async def get_graph_neighbors(ws: dict, entity_id: str, options: NeighborOptions):
    async with graph_transaction(ws) as conn:
        record = await read_record(conn, ws["id"], "entity", entity_id)
        return await neighborhood(
            conn,
            ws["id"],
            [record["id"]],
            hops=options.hops,
            max_nodes=options.max_nodes,
            direction=options.direction,
            relationship_type=options.relationship_type,
        )


async def retrieve_graph(body: GraphRetrieve, ws: dict):
    vector = None
    if body.mode != "keyword" and body.text:
        vector = await embeddings.query_embedding(body.text, "graph.retrieve")
    wid = ws["id"]
    async with graph_transaction(ws) as conn:
        records, _ = await find_records(
            conn,
            wid,
            body.seed_kind,
            q=body.q,
            text=body.text,
            entity_type=body.type,
            origin=body.origin,
            mode=body.mode,
            limit=body.seed_limit,
            query_vector=vector,
            profile=embeddings.embedding_profile(),
        )
        edge_seeds = records if body.seed_kind == "relationship" else []
        entity_ids = (
            [node for e in edge_seeds for node in (e["source"], e["target"])]
            if edge_seeds
            else [r["id"] for r in records]
        )
        graph = await neighborhood(
            conn,
            wid,
            entity_ids,
            hops=body.hops,
            max_nodes=body.max_nodes,
            seed_edges=edge_seeds,
        )
        coverage = await embeddings.embedding_coverage(conn, wid, body.seed_kind)
    sources = {}
    for record in [*graph["entities"], *graph["relationships"]]:
        for source in record["sources"]:
            sources[tuple(sorted(source.items()))] = source
    return {
        **graph,
        "seeds": [{"id": row["id"], "kind": body.seed_kind} for row in records],
        "sources": list(sources.values()),
        "coverage": coverage,
    }


async def get_embedding_consistency(ws: dict):
    async with graph_transaction(ws) as conn:
        return await embeddings.consistency_report(conn, ws["id"])
