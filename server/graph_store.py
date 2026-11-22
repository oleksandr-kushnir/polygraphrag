"""The workspace's editable graph and its projection from LightRAG."""

import hashlib
import json
import re
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import HTTPException

import server
from server.references import _resolve_joined_path
from server.schemas import GraphWrite, MergeApplyRequest, MergePreviewRequest

VISIBLE = """NOT suppressed AND redirect IS NULL
             AND (manual IS NOT NULL OR extracted IS NOT NULL OR identity_fallback IS NOT NULL)"""
EXTRACTED_PREFIX = "lr."


def graph_storage_ready() -> bool:
    return server._db_pool is not None and server._graph_pool is server._db_pool


def graph_error(status: int, code: str, message: str, **extra) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, **extra})


def age_graph_name(physical: str) -> str:
    if not physical.strip() or physical.strip().lower() == "default":
        return "chunk_entity_relation"
    return re.sub(r"[^a-zA-Z0-9_]", "_", physical.strip()) + "_chunk_entity_relation"


def extracted_id(kind: str, *keys: str) -> str:
    digest = hashlib.sha256(json.dumps(keys, ensure_ascii=True).encode()).hexdigest()
    return f"{EXTRACTED_PREFIX}{kind}.{digest}"


def _properties(value):
    return json.loads(value) if isinstance(value, str) else value


def _source_paths(props, chunks, file_paths):
    path = props.get("file_path") or ""
    fallback = [p for p in path.split("<SEP>") if p and p != "unknown_source"]
    evidence = []
    for chunk_id in (props.get("source_id") or "").split("<SEP>"):
        if chunk_id and chunk_id in chunks:
            evidence.append(chunks[chunk_id])
    known_paths = {s["file_path"] for s in evidence}
    for key in sorted(fallback):
        file_path = _resolve_joined_path(key, file_paths, "<SEP>")
        if file_path not in known_paths:
            evidence.append({"origin": "extracted", "file_path": file_path})
            known_paths.add(file_path)
    return evidence


async def _extracted_records(conn, physical: str):
    graph = age_graph_name(physical)
    paths = await conn.fetch(
        """SELECT lightrag_key, file_path, file
           FROM rag_file_metadata WHERE workspace=$1
           ORDER BY uploaded_at DESC, job_id""",
        physical,
    )
    file_paths: dict[str, dict] = {}
    for key in ("lightrag_key", "file"):
        for row in paths:
            if row[key]:
                file_paths.setdefault(row[key], {"file_path": row["file_path"]})
    chunks: dict[str, dict] = {}
    if await conn.fetchval("SELECT to_regclass('public.lightrag_doc_chunks') IS NOT NULL"):
        for row in await conn.fetch(
            """SELECT id, content, full_doc_id, file_path
               FROM lightrag_doc_chunks WHERE workspace=$1 ORDER BY id""",
            physical,
        ):
            key = row["file_path"]
            source = {
                "origin": "extracted",
                "chunk_id": row["id"],
                "document_id": row["full_doc_id"],
                "file_path": _resolve_joined_path(key or "", file_paths, "<SEP>") or None,
            }
            chunks[row["id"]] = source
            yield "chunk", row["id"], {"description": row["content"] or "", "sources": [source]}
    if not await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM ag_catalog.ag_graph WHERE name=$1)", graph
    ):
        return
    await conn.execute("LOAD 'age'")
    # The schema name comes from age_graph_name(), which only emits ASCII identifier characters.
    nodes = await conn.fetch(
        f"SELECT id::text AS id, ag_catalog.agtype_out(properties)::text AS props "
        f'FROM "{graph}"._ag_label_vertex'
    )
    vertex_keys = {}
    for row in nodes:
        props = _properties(row["props"])
        key = props["entity_id"]
        vertex_keys[row["id"]] = key
        yield "entity", extracted_id("entity", key), {
            "name": key,
            "type": props.get("entity_type") or "UNKNOWN",
            "description": props.get("description") or "",
            "created_at": props.get("created_at"),
            "sources": _source_paths(props, chunks, file_paths),
        }
    edges = await conn.fetch(
        "SELECT start_id::text AS source, end_id::text AS target, "
        "ag_catalog.agtype_out(properties)::text AS props "
        f'FROM "{graph}"._ag_label_edge'
    )
    for row in edges:
        props = _properties(row["props"])
        source, target = sorted((vertex_keys[row["source"]], vertex_keys[row["target"]]))
        yield "relationship", extracted_id("relationship", source, target), {
            "source": extracted_id("entity", source),
            "target": extracted_id("entity", target),
            "type": props.get("keywords") or "RELATED_TO",
            "description": props.get("description") or "",
            "created_at": props.get("created_at"),
            "direction": "undirected",
            "sources": _source_paths(props, chunks, file_paths),
        }


async def refresh_extraction(conn, ws: dict) -> None:
    """Publish a complete upstream snapshot while holding the workspace registry row lock."""
    wid = ws["id"]
    records = [record async for record in _extracted_records(conn, ws["lightrag_workspace"])]
    # Register nodes before edges. Keep the original extracted IDs even after a merge.
    redirects = {
        row["id"]: row["redirect"]
        for row in await conn.fetch(
            "SELECT id, redirect FROM rag_graph_records WHERE workspace=$1 AND redirect IS NOT NULL",
            wid,
        )
    }
    for kind, record_id, fields in records:
        if kind == "relationship":
            for key in ("source", "target"):
                fields[key] = redirects.get(fields[key], fields[key])
    await conn.executemany(
        """INSERT INTO rag_graph_records (workspace, kind, id, extracted)
           VALUES ($1, $2, $3, $4::jsonb)
           ON CONFLICT (workspace, kind, id) DO UPDATE
           SET extracted=excluded.extracted, identity_fallback=NULL,
               revision=rag_graph_records.revision+1
           WHERE rag_graph_records.extracted IS DISTINCT FROM excluded.extracted""",
        [(wid, kind, record_id, json.dumps(fields)) for kind, record_id, fields in records],
    )
    await conn.execute(
        """UPDATE rag_graph_records r
           SET manual=CASE WHEN r.kind='relationship' AND r.manual IS NOT NULL
                THEN (r.extracted - 'sources' || '{"description": ""}'::jsonb) || r.manual
                ELSE r.manual END,
               identity_fallback=CASE WHEN r.kind='entity' AND
                (r.manual IS NOT NULL OR EXISTS (
                    SELECT 1 FROM rag_graph_records e WHERE e.workspace=r.workspace
                      AND NOT e.suppressed AND e.manual IS NOT NULL
                      AND (e.source=r.id OR e.target=r.id)
                )) THEN jsonb_build_object('name', r.extracted->'name',
                     'type', r.extracted->'type', 'description', '')
                END,
               extracted=NULL, revision=revision+1
           WHERE workspace=$1 AND extracted IS NOT NULL
             AND NOT EXISTS (
                 SELECT 1 FROM unnest($2::text[], $3::text[]) AS live(kind, id)
                 WHERE live.kind=r.kind AND live.id=r.id
             )""",
        wid,
        [kind for kind, _, _ in records],
        [record_id for _, record_id, _ in records],
    )
    await conn.execute("INSERT INTO rag_graph_imports VALUES ($1) ON CONFLICT DO NOTHING", wid)
    await _prune_unused_identities(conn, wid)


async def _prune_unused_identities(conn, wid: str) -> None:
    # Keep the extracted label only for as long as a visible manual assertion needs it.
    await conn.execute(
        """UPDATE rag_graph_records n SET identity_fallback=NULL,revision=revision+1
           WHERE n.workspace=$1 AND n.kind='entity' AND n.identity_fallback IS NOT NULL
             AND n.extracted IS NULL AND n.manual IS NULL AND NOT n.suppressed AND n.redirect IS NULL
             AND NOT EXISTS (
                 SELECT 1 FROM rag_graph_active e WHERE e.workspace=n.workspace
                   AND e.kind='relationship' AND e.manual IS NOT NULL
                   AND (e.source=n.id OR e.target=n.id)
             )""",
        wid,
    )


@asynccontextmanager
async def graph_transaction(ws: dict, *, write: bool = False):
    pool = server._db_pool
    if pool is None:
        raise graph_error(503, "database_unavailable", "Graph storage is unavailable")
    async with pool.acquire() as conn, conn.transaction():
        wid = ws["id"]
        initialized = await conn.fetchval("SELECT 1 FROM rag_graph_imports WHERE workspace=$1", wid)
        lock = "UPDATE" if write or not initialized else "SHARE"
        active = await conn.fetchval(
            f"SELECT 1 FROM rag_workspaces WHERE id=$1 AND deleted_at IS NULL FOR {lock}", wid
        )
        if not active:
            raise graph_error(404, "workspace_not_found", "Workspace is unavailable")
        if not initialized and not await conn.fetchval(
            "SELECT 1 FROM rag_graph_imports WHERE workspace=$1", wid
        ):
            await refresh_extraction(conn, ws)
        yield conn


def public_record(row, *, requested_id: str | None = None) -> dict:
    base = _properties(row["extracted"]) or {}
    manual = _properties(row["manual"])
    merged_sources = _properties(row.get("merged_sources")) or []
    fields = (_properties(row.get("identity_fallback")) or {}) | base | (manual or {})
    sources = [*base.get("sources", []), *merged_sources]
    if manual and manual.get("source_ref"):
        sources.append({"origin": "manual", "source_ref": manual["source_ref"]})
    origins = [
        origin
        for origin, present in (
            ("manual", manual is not None or any(s["origin"] == "manual" for s in merged_sources)),
            ("extracted", bool(base) or any(s["origin"] == "extracted" for s in merged_sources)),
        )
        if present
    ]
    result = {
        **{k: v for k, v in fields.items() if k not in {"sources", "source_ref"}},
        "id": row["id"],
        "revision": row["revision"],
        "origins": origins,
        "sources": list(
            {json.dumps(source, sort_keys=True): source for source in sources}.values()
        ),
    }
    if requested_id is not None:
        result["requested_id"] = requested_id
    if "degree" in row:
        result["degree"] = row["degree"]
    return result


async def get_row(conn, wid: str, kind: str, record_id: str):
    return await conn.fetchrow(
        "SELECT * FROM rag_graph_records WHERE workspace=$1 AND kind=$2 AND id=$3",
        wid,
        kind,
        record_id,
    )


def is_visible(row) -> bool:
    return bool(
        row
        and not row["suppressed"]
        and not row["redirect"]
        and (
            row["manual"] is not None
            or row["extracted"] is not None
            or row["identity_fallback"] is not None
        )
    )


async def require_record(conn, wid: str, kind: str, record_id: str):
    row = await conn.fetchrow(
        "SELECT * FROM rag_graph_active WHERE workspace=$1 AND kind=$2 AND id=$3",
        wid,
        kind,
        record_id,
    )
    if row is None:
        raise graph_error(
            404, "record_not_found", f"{kind.capitalize()} {record_id!r} is unavailable"
        )
    return row


def check_revision(row, expected: int | None) -> None:
    if expected != (row["revision"] if row else None):
        raise graph_error(
            409,
            "revision_conflict",
            "The record revision has changed or is required",
            revision=row["revision"] if row else None,
        )


async def write_record(
    conn, wid: str, kind: str, record_id: str, body: GraphWrite, *, partial: bool = False
) -> tuple[dict, bool]:
    row = await get_row(conn, wid, kind, record_id)
    if partial and not is_visible(row):
        raise graph_error(
            404, "record_not_found", f"{kind.capitalize()} {record_id!r} is unavailable"
        )
    if row is None and record_id.startswith(EXTRACTED_PREFIX):
        raise graph_error(422, "reserved_id", "The lr. prefix is reserved for extracted graph IDs")
    if row and (row["suppressed"] or row["redirect"]):
        raise graph_error(409, "retired_id", "The ID is suppressed or has been merged")
    previous = (_properties(row["manual"]) or {}) if row else {}
    changes = body.model_dump(exclude_unset=partial, exclude={"id", "expected_revision"})
    manual = previous | changes if partial else changes
    base = (_properties(row["extracted"]) or {}) if row else {}
    if kind == "relationship":
        if not partial:
            manual["direction"] = base.get("direction", "directed")
        fields = base | manual
        endpoints = {fields["source"], fields["target"]}
        valid = await conn.fetchval(
            f"""SELECT count(*) FROM rag_graph_records WHERE workspace=$1 AND kind='entity'
                AND id=ANY($2::text[]) AND {VISIBLE}""",
            wid,
            sorted(endpoints),
        )
        if valid != len(endpoints):
            raise graph_error(
                422, "invalid_endpoint", "Both endpoints must be active entities in the workspace"
            )
    if row and row["manual"] is not None and manual == previous and body.expected_revision is None:
        return public_record(row), False
    check_revision(row, body.expected_revision)
    if row and row["manual"] is not None and manual == previous:
        return public_record(row), False
    stored = await conn.fetchrow(
        """INSERT INTO rag_graph_records (workspace, kind, id, manual) VALUES ($1, $2, $3, $4)
           ON CONFLICT (workspace, kind, id) DO UPDATE
           SET manual=excluded.manual, revision=rag_graph_records.revision+1 RETURNING *""",
        wid,
        kind,
        record_id,
        json.dumps(manual),
    )
    if (
        kind == "relationship"
        and row
        and (row["source"], row["target"]) != (stored["source"], stored["target"])
    ):
        await _prune_unused_identities(conn, wid)
    return public_record(await require_record(conn, wid, kind, stored["id"])), row is None


async def delete_record(conn, wid: str, kind: str, record_id: str, revision: int) -> dict:
    row = await require_record(conn, wid, kind, record_id)
    check_revision(row, revision)
    affected = []
    if kind == "entity":
        affected = await conn.fetch(
            f"""UPDATE rag_graph_records SET suppressed=TRUE, revision=revision+1
                WHERE workspace=$1 AND kind='relationship' AND {VISIBLE}
                AND (source=$2 OR target=$2) RETURNING id""",
            wid,
            record_id,
        )
    await conn.execute(
        """UPDATE rag_graph_records SET suppressed=TRUE, revision=revision+1
           WHERE workspace=$1 AND kind=$2 AND id=$3""",
        wid,
        kind,
        record_id,
    )
    await _prune_unused_identities(conn, wid)
    return {"id": record_id, "revision": revision + 1, "affected_relationships": len(affected)}


async def read_record(conn, wid: str, kind: str, record_id: str) -> dict:
    row = await get_row(conn, wid, kind, record_id)
    canonical = row["redirect"] if row and row["redirect"] else record_id
    result = await require_record(conn, wid, kind, canonical)
    return public_record(result, requested_id=record_id if canonical != record_id else None)


def _merge_fingerprint(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


async def _merge_state(conn, wid: str, request: MergePreviewRequest) -> dict:
    survivor = public_record(await require_record(conn, wid, "entity", request.survivor_id))
    retired = [
        public_record(await require_record(conn, wid, "entity", record_id))
        for record_id in sorted(request.retire_ids)
    ]
    edges = await conn.fetch(
        """SELECT * FROM rag_graph_active WHERE workspace=$1 AND kind='relationship'
           AND (source=ANY($2) OR target=ANY($2)) ORDER BY id""",
        wid,
        request.retire_ids,
    )
    rewired = []
    self_loops = []
    for row in edges:
        edge = public_record(row)
        for endpoint in ("source", "target"):
            if edge[endpoint] in request.retire_ids:
                edge[endpoint] = request.survivor_id
        rewired.append(edge)
        if edge["source"] == edge["target"]:
            self_loops.append(edge["id"])
    conflicts = {}
    for field in ("name", "type", "description"):
        values = sorted({r[field] for r in [survivor, *retired] if r[field]})
        if len(values) > 1:
            conflicts[field] = values
    return {
        "survivor": survivor,
        "retired": retired,
        "rewired_edges": rewired,
        "conflicts": conflicts,
        "self_loops": self_loops,
    }


async def preview_merge(conn, wid: str, request: MergePreviewRequest) -> dict:
    state = await _merge_state(conn, wid, request)
    token = uuid4().hex
    await conn.execute(
        "DELETE FROM rag_graph_merge_previews WHERE workspace=$1 AND expires_at<now()", wid
    )
    await conn.execute(
        """INSERT INTO rag_graph_merge_previews (workspace,token,input,fingerprint)
           VALUES ($1,$2,$3,$4)""",
        wid,
        token,
        request.model_dump_json(),
        _merge_fingerprint(state),
    )
    return {"preview_token": token, **state}


async def apply_merge(conn, wid: str, request: MergeApplyRequest) -> dict:
    preview = await conn.fetchrow(
        """SELECT input,fingerprint FROM rag_graph_merge_previews
           WHERE workspace=$1 AND token=$2 AND expires_at>now()""",
        wid,
        request.preview_token,
    )
    stale = graph_error(409, "stale_merge_preview", "Preview the merge again before applying it")
    if not preview:
        raise stale
    original = MergePreviewRequest.model_validate(_properties(preview["input"]))
    try:
        state = await _merge_state(conn, wid, original)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise stale from exc
        raise
    if preview["fingerprint"] != _merge_fingerprint(state):
        raise stale
    if set(state["conflicts"]) - request.resolutions.keys() or (
        state["self_loops"] and not request.allow_self_loops
    ):
        raise graph_error(
            422, "merge_resolution_required", "Resolve field conflicts and approve self-loops"
        )
    for key in ("name", "type"):
        if key in request.resolutions:
            value = request.resolutions[key].strip()
            if not value or len(value) > 512:
                raise graph_error(422, "invalid_resolution", f"{key} must be 1 to 512 characters")
            request.resolutions[key] = value
    survivor = state["survivor"]
    fields = {k: survivor[k] for k in ("name", "type", "description")}
    for retired in state["retired"]:
        if not fields["description"] and retired["description"]:
            fields["description"] = retired["description"]
    fields.update({key: value for key, value in request.resolutions.items()})
    await conn.execute(
        """UPDATE rag_graph_records SET manual=COALESCE(manual,'{}') || $3::jsonb,
              revision=revision+1 WHERE workspace=$1 AND kind='entity' AND id=$2""",
        wid,
        original.survivor_id,
        json.dumps(fields),
    )
    for edge in state["rewired_edges"]:
        # A pure extracted edge keeps its ownership. Future snapshots resolve the same redirect.
        await conn.execute(
            """UPDATE rag_graph_records
               SET extracted=CASE WHEN extracted IS NOT NULL THEN extracted || $3::jsonb END,
                   manual=CASE WHEN manual IS NOT NULL THEN manual || $3::jsonb END,
                   revision=revision+1 WHERE workspace=$1 AND kind='relationship' AND id=$2""",
            wid,
            edge["id"],
            json.dumps({k: edge[k] for k in ("source", "target")}),
        )
    await conn.execute(
        """UPDATE rag_graph_records SET redirect=$2,revision=revision+1
           WHERE workspace=$1 AND kind='entity' AND (id=ANY($3) OR redirect=ANY($3))""",
        wid,
        original.survivor_id,
        original.retire_ids,
    )
    await conn.execute(
        "DELETE FROM rag_graph_merge_previews WHERE workspace=$1 AND token=$2",
        wid,
        request.preview_token,
    )
    return {
        "survivor": await read_record(conn, wid, "entity", original.survivor_id),
        "retired_ids": original.retire_ids,
        "rewired_relationships": len(state["rewired_edges"]),
    }
