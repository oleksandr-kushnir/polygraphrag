"""Bounded search and traversal of the effective graph."""

import base64
import binascii
import hashlib
import json
import re
from typing import Literal

from server.graph_store import graph_error, public_record

_TOKEN = re.compile(r'\s+|"(?:[^"\\]|\\.)*"|[():]|[^\s":()]+')
_FIELDS = {"entity": ("name", "type", "description"), "relationship": ("type", "description")}


class SearchExpression:
    """Parse the Boolean grammar and append bound parameters as terms are emitted."""

    def __init__(self, query: str, kind: str, args: list):
        self.query = query
        self.fields = _FIELDS[kind]
        self.args = args
        self.tokens: list[tuple[str, int]] = []
        end = 0
        for token in _TOKEN.finditer(query):
            if token.start() != end:
                self.invalid(end)
            if not token[0].isspace():
                self.tokens.append((token[0], token.start()))
            end = token.end()
        if end != len(query) or len(self.tokens) > 128:
            self.invalid(end if end != len(query) else self.tokens[128][1])
        self.tokens.append(("", len(query)))
        self.pos = 0

    @staticmethod
    def invalid(position: int):
        raise graph_error(422, "invalid_search", "Invalid Boolean expression", position=position)

    def peek(self) -> str:
        return self.tokens[self.pos][0]

    def accept(self, value: str) -> bool:
        if self.peek() != value:
            return False
        self.pos += 1
        return True

    def parse(self) -> str:
        result = self.expression()
        if self.peek():
            self.invalid(self.tokens[self.pos][1])
        return result

    def expression(self) -> str:
        terms = [self.conjunction()]
        while self.accept("OR"):
            terms.append(self.conjunction())
        return "(" + " OR ".join(terms) + ")"

    def conjunction(self) -> str:
        terms = [self.term()]
        while self.accept("AND"):
            terms.append(self.term())
        return "(" + " AND ".join(terms) + ")"

    def term(self) -> str:
        if self.accept("("):
            result = self.expression()
            if not self.accept(")"):
                self.invalid(self.tokens[self.pos][1])
            return result
        token, position = self.tokens[self.pos]
        fields = self.fields
        if self.tokens[self.pos + 1 : self.pos + 2] and self.tokens[self.pos + 1][0] == ":":
            if token not in fields:
                self.invalid(position)
            fields = (token,)
            self.pos += 2
            token, position = self.tokens[self.pos]
        if token in {"", "AND", "OR", ")", "(", ":"}:
            self.invalid(position)
        self.pos += 1
        if token.startswith('"'):
            try:
                token = json.loads(token)
            except json.JSONDecodeError as exc:
                self.invalid(position + exc.pos)
        if not token or "\x00" in token or any(0xD800 <= ord(c) <= 0xDFFF for c in token):
            self.invalid(position)
        self.args.append(token)
        value = f"lower(${len(self.args)}::text)"
        conditions = []
        for field in fields:
            column = f"lower(r.fields->>'{field}')"
            conditions.append(
                f"{column}={value}" if field == "type" else f"strpos({column}, {value})>0"
            )
        return "(" + " OR ".join(conditions) + ")"


def _fingerprint(criteria: tuple) -> str:
    return hashlib.sha256(json.dumps(criteria, separators=(",", ":")).encode()).hexdigest()


def cursor_offset(cursor: str | None, criteria: tuple) -> int:
    if cursor is None:
        return 0
    try:
        fingerprint, offset = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
        if fingerprint == _fingerprint(criteria) and type(offset) is int and 0 < offset < 2**63:
            return offset
    except (ValueError, TypeError, binascii.Error):
        pass
    raise graph_error(422, "invalid_cursor", "Cursor does not match the workspace or search")


def next_cursor(criteria: tuple, offset: int, total: int) -> str | None:
    if offset >= total:
        return None
    return base64.urlsafe_b64encode(json.dumps([_fingerprint(criteria), offset]).encode()).decode()


async def find_records(
    conn,
    wid: str,
    kind: str,
    *,
    q=None,
    text=None,
    name=None,
    match="contains",
    origin=None,
    entity_type=None,
    sort=None,
    limit=50,
    offset=0,
    mode="keyword",
    query_vector=None,
    profile=None,
):
    args: list = [wid, kind]
    where = ["r.workspace=$1", "r.kind=$2"]

    def param(value):
        args.append(value)
        return f"${len(args)}"

    if q is not None:
        where.append(SearchExpression(q, kind, args).parse())
    if origin:
        column = {"manual": "manual", "extracted": "extracted"}[origin]
        value = param(origin)
        where.append(
            f"(r.{column} IS NOT NULL OR r.merged_sources @> "
            f"jsonb_build_array(jsonb_build_object('origin', {value}::text)))"
        )
    if entity_type is not None:
        where.append(f"lower(r.fields->>'type')=lower({param(entity_type)}::text)")
    name_order = ""
    if name is not None:
        p = f"lower({param(name)}::text)"
        label = "lower(r.fields->>'name')"
        if match == "exact":
            where.append(f"{label}={p}")
        else:
            where.append(f"strpos({label}, {p})>0")
            name_order = (
                f"CASE WHEN {label}={p} THEN 0 WHEN strpos({label}, {p})=1 THEN 1 ELSE 2 END"
            )
    degree = (
        """(SELECT count(*) FROM rag_graph_active e
                  WHERE e.workspace=r.workspace AND e.kind='relationship'
                    AND (e.source=r.id OR e.target=r.id))"""
        if sort == "degree"
        else "NULL::bigint"
    )
    keyword_match = "TRUE"
    keyword_score = "0::real"
    if text is not None and mode != "semantic":
        value = f"websearch_to_tsquery('simple', {param(text)}::text)"
        document = "to_tsvector('simple', concat_ws(' ', r.fields->>'name', r.fields->>'type', r.fields->>'description'))"
        keyword_match = f"{document} @@ {value}"
        keyword_score = f"ts_rank({document}, {value})"
    similarity = "NULL::double precision"
    joins = ""
    if query_vector is not None:
        joins = (
            " JOIN rag_graph_index_inputs inp USING (workspace, kind, id)"
            " LEFT JOIN rag_graph_vectors v ON (v.workspace,v.kind,v.id,v.profile,v.content_key)="
            f"(r.workspace,r.kind,r.id,{param(profile)}::text,inp.content_key)"
        )
        similarity = f"v.embedding <=> {param(query_vector)}::vector"
    order = ", ".join(
        part
        for part in (
            "score DESC",
            "degree DESC" if sort == "degree" else "",
            name_order and "name_rank",
            "id",
        )
        if part
    )
    rank_score = {
        "keyword": "CASE WHEN keyword_match THEN keyword_score END",
        "semantic": "-distance",
        "hybrid": (
            "CASE WHEN keyword_match OR distance IS NOT NULL THEN "
            "CASE WHEN keyword_match THEN 1.0/(60+keyword_rank) ELSE 0 END + "
            "CASE WHEN distance IS NOT NULL THEN 1.0/(60+vector_rank) ELSE 0 END END"
        ),
    }[mode]
    data = await conn.fetchrow(
        f"""WITH candidates AS (
                SELECT r.*, {degree} AS degree, {name_order or '0'} AS name_rank,
                       {keyword_match} AS keyword_match, {keyword_score} AS keyword_score,
                       {similarity} AS distance
                FROM rag_graph_active r {joins} WHERE {' AND '.join(where)}
            ), ranked AS (
                SELECT *, row_number() OVER (ORDER BY keyword_match DESC, keyword_score DESC, id)
                             AS keyword_rank,
                          row_number() OVER (ORDER BY distance NULLS LAST, id) AS vector_rank
                FROM candidates
            ), scored AS (
                SELECT *, {rank_score} AS score FROM ranked
            ), page AS (
                SELECT * FROM scored WHERE score IS NOT NULL
                ORDER BY {order} LIMIT {param(limit)} OFFSET {param(offset)}
            )
            SELECT (SELECT count(*) FROM scored WHERE score IS NOT NULL) AS total,
                   (SELECT jsonb_agg(to_jsonb(page) ORDER BY {order}) FROM page) AS items""",
        *args,
    )
    rows = json.loads(data["items"]) if data["items"] else []
    return [public_record(row) for row in rows], data["total"]


async def neighborhood(
    conn,
    wid: str,
    seed_ids: list[str],
    *,
    hops: int,
    max_nodes: int,
    direction: Literal["in", "out", "both"] = "both",
    relationship_type=None,
    seed_edges: list[dict] | None = None,
) -> dict:
    edge_budget = min(5000, max_nodes * 10)
    ids = list(dict.fromkeys(seed_ids))
    chosen = set(ids[:max_nodes])
    frontier = sorted(chosen)
    truncated = len(ids) > max_nodes
    edges = {e["id"]: e for e in seed_edges or [] if {e["source"], e["target"]} <= chosen}
    direction_filter = {
        "out": "source=ANY($2) OR (fields->>'direction'='undirected' AND target=ANY($2))",
        "in": "target=ANY($2) OR (fields->>'direction'='undirected' AND source=ANY($2))",
        "both": "source=ANY($2) OR target=ANY($2)",
    }[direction]
    for _ in range(hops):
        if not frontier:
            break
        # Exclude already emitted edges so returning to visited nodes cannot exhaust the budget.
        rows = await conn.fetch(
            f"""SELECT * FROM rag_graph_active WHERE workspace=$1 AND kind='relationship'
                AND ({direction_filter}) AND id<>ALL($3::text[])
                AND ($4::text IS NULL OR lower(fields->>'type')=lower($4))
                ORDER BY id LIMIT $5""",
            wid,
            frontier,
            list(edges),
            relationship_type,
            edge_budget - len(edges) + 1,
        )
        new = set()
        for row in rows:
            unseen = {row["source"], row["target"]} - chosen
            if len(chosen) + len(unseen) > max_nodes or len(edges) == edge_budget:
                truncated = True
                continue
            chosen.update(unseen)
            new.update(unseen)
            edges[row["id"]] = public_record(row)
        frontier = sorted(new)
    entities = await conn.fetch(
        """SELECT * FROM rag_graph_active
           WHERE workspace=$1 AND kind='entity' AND id=ANY($2::text[]) ORDER BY id""",
        wid,
        sorted(chosen),
    )
    return {
        "entities": [public_record(row) for row in entities],
        "relationships": list(edges.values()),
        "truncated": truncated,
    }
