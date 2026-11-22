"""Revision-aware embedding jobs for the effective workspace graph."""

import asyncio
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from uuid import uuid4

from server import config, llm
from server.graph_store import graph_error, graph_storage_ready, graph_transaction
from server.providers import available, provider_unavailable, require_providers

_KINDS = ("entity", "relationship", "chunk")
_ACTIVE_STATES = "('queued','running','waiting_for_provider')"
_STATES = """
    SELECT i.*,
        CASE WHEN v.content_key=i.content_key THEN 'ready'
             WHEN EXISTS(SELECT 1 FROM rag_graph_vectors previous
                         WHERE (previous.workspace,previous.kind,previous.id)=
                               (i.workspace,i.kind,i.id)) THEN 'stale'
             ELSE 'missing' END AS state,
        CASE WHEN v.content_key IS NULL THEN 'profile_changed' ELSE 'content_changed' END AS reason
    FROM rag_graph_index_inputs i
    LEFT JOIN rag_graph_vectors v
      ON (v.workspace,v.kind,v.id,v.profile)=(i.workspace,i.kind,i.id,$2)
    WHERE i.workspace=$1
"""


def embedding_profile() -> str:
    identity = json.dumps(
        [
            config.EMBEDDING_MODEL,
            config.EMBEDDING_DIM,
            config.EMBEDDING_BASE_URL,
            config.EMBEDDING_PROFILE,
        ]
    )
    return "emb." + hashlib.sha256(identity.encode()).hexdigest()[:32]


class InvalidEmbedding(ValueError):
    """The provider returned an invalid vector or rejected an input."""


async def embed(texts: list[str]) -> list[str]:
    vectors = await asyncio.wait_for(llm._embedding_func(texts), timeout=45)
    if len(vectors) != len(texts):
        raise InvalidEmbedding("Embedding provider returned an unexpected number of vectors")
    result = []
    for vector in vectors:
        floats = [float(value) for value in vector]
        if (
            len(floats) != config.EMBEDDING_DIM
            or not all(math.isfinite(value) and abs(value) <= 3.4e38 for value in floats)
            or not any(abs(value) >= 1e-38 for value in floats)
        ):
            raise InvalidEmbedding(
                "Embedding dimension or values do not match the configured profile"
            )
        result.append(json.dumps(floats, separators=(",", ":")))
    return result


async def query_embedding(text: str, operation: str) -> str:
    require_providers(operation, "embedding")
    try:
        return (await embed([text]))[0]
    except Exception as exc:
        raise provider_unavailable(operation, "embedding") from exc


async def embedding_coverage(conn, wid: str, kind: str) -> dict:
    row = await conn.fetchrow(
        f"""SELECT count(*) AS eligible, count(*) FILTER (WHERE state='ready') AS indexed
            FROM ({_STATES}) AS coverage WHERE kind=$3""",
        wid,
        embedding_profile(),
        kind,
    )
    return dict(row)


async def missing_embeddings(conn, wid, kind, origin, state, limit, offset):
    row = await conn.fetchrow(
        f"""WITH matching AS (
            SELECT ix.id, ix.kind, ix.fields->>'name' AS name,
                ix.fields->'sources'->0->>'chunk_id' AS chunk_ref,
                ix.state, CASE WHEN ix.state='missing' THEN 'never_indexed' ELSE ix.reason END AS reason,
                (SELECT j.state FROM rag_embedding_jobs j
                 JOIN rag_embedding_items p ON (p.workspace,p.job)=(j.workspace,j.id)
                 WHERE j.workspace=ix.workspace AND j.profile=$2
                   AND (p.kind,p.id,p.content_key)=(ix.kind,ix.id,ix.content_key)
                 ORDER BY j.created_at DESC LIMIT 1) AS job_state
            FROM ({_STATES}) ix
            WHERE ($3::text IS NULL OR kind=$3) AND ($4::text IS NULL OR $4=ANY(origins))
              AND (state=$5 OR ($5='needs_embedding' AND state IN ('missing','stale')))
        ), page AS (SELECT * FROM matching ORDER BY kind,id LIMIT $6 OFFSET $7)
        SELECT (SELECT count(*) FROM matching) AS total,
               (SELECT jsonb_agg(to_jsonb(page) ORDER BY kind,id) FROM page) AS items""",
        wid,
        embedding_profile(),
        kind,
        origin,
        state,
        limit,
        offset,
    )
    return json.loads(row["items"]) if row["items"] else [], row["total"]


async def consistency_report(conn, wid: str) -> dict:
    profile = embedding_profile()
    rows = await conn.fetch(
        f"""SELECT kind,state,count(*) AS n FROM ({_STATES}) ix GROUP BY kind,state
            UNION ALL
            SELECT v.kind,'orphaned',count(*) FROM rag_graph_vectors v
            WHERE v.workspace=$1 AND v.profile=$2 AND NOT EXISTS (
                SELECT 1 FROM rag_graph_index_inputs i
                WHERE (i.workspace,i.kind,i.id)=(v.workspace,v.kind,v.id)
            ) GROUP BY v.kind""",
        wid,
        profile,
    )
    counts = {kind: dict.fromkeys(("ready", "missing", "stale", "orphaned"), 0) for kind in _KINDS}
    for row in rows:
        counts[row["kind"]][row["state"]] = row["n"]
    return {
        "profile": profile,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
    }


async def job_status(conn, wid: str, job_id: str) -> dict:
    row = await conn.fetchrow(
        "SELECT profile,state FROM rag_embedding_jobs WHERE workspace=$1 AND id=$2", wid, job_id
    )
    if not row:
        raise graph_error(404, "job_not_found", "Embedding job is unavailable")
    items = await conn.fetch(
        """SELECT kind,state,count(*) AS n FROM rag_embedding_items
           WHERE workspace=$1 AND job=$2 GROUP BY kind,state""",
        wid,
        job_id,
    )
    counts = {
        kind: dict.fromkeys(("pending", "indexed", "skipped", "failed"), 0) for kind in _KINDS
    }
    for item in items:
        counts[item["kind"]][item["state"]] = item["n"]
    return {"job_id": job_id, **dict(row), "counts": counts}


async def enqueue(conn, wid: str, profile: str, *, automatic: bool = False) -> str:
    existing = await conn.fetchval(
        f"""SELECT id FROM rag_embedding_jobs
            WHERE workspace=$1 AND profile=$2 AND state IN {_ACTIVE_STATES}""",
        wid,
        profile,
    )
    if existing:
        return existing
    job_id = uuid4().hex
    await conn.execute(
        "INSERT INTO rag_embedding_jobs (workspace,id,profile) VALUES ($1,$2,$3)",
        wid,
        job_id,
        profile,
    )
    # Automatic work avoids repeatedly selecting an unchanged input rejected by the provider.
    # An explicit backfill can retry it after the provider or configuration is fixed.
    failure_filter = (
        """
        AND NOT EXISTS (
            SELECT 1 FROM rag_embedding_items p
            JOIN rag_embedding_jobs j ON (j.workspace,j.id)=(p.workspace,p.job)
            WHERE (p.workspace,p.kind,p.id,p.content_key)=(ix.workspace,ix.kind,ix.id,ix.content_key)
              AND j.profile=$2 AND p.state='failed'
        )"""
        if automatic
        else ""
    )
    await conn.execute(
        f"""INSERT INTO rag_embedding_items (workspace,job,kind,id,content_key)
            SELECT workspace,$3,kind,id,content_key FROM ({_STATES}) ix
            WHERE state<>'ready' {failure_filter}""",
        wid,
        profile,
        job_id,
    )
    await conn.execute(
        """UPDATE rag_embedding_jobs SET state='completed' WHERE workspace=$1 AND id=$2
           AND NOT EXISTS(SELECT 1 FROM rag_embedding_items WHERE workspace=$1 AND job=$2)""",
        wid,
        job_id,
    )
    return job_id


async def _auto_enqueue(conn, profile: str) -> None:
    if config.EMBEDDING_INDEX_MODE != "auto":
        return
    # Choose workspaces with eligible inputs that are not already in an active job.
    rows = await conn.fetch(
        f"""SELECT w.id FROM rag_workspaces w WHERE w.deleted_at IS NULL
            AND EXISTS (
                SELECT 1 FROM rag_graph_index_inputs ix WHERE ix.workspace=w.id
                AND NOT EXISTS (
                    SELECT 1 FROM rag_graph_vectors v WHERE
                    (v.workspace,v.kind,v.id,v.content_key,v.profile)=
                    (ix.workspace,ix.kind,ix.id,ix.content_key,$1)
                )
                AND NOT EXISTS (
                    SELECT 1 FROM rag_embedding_items p
                    JOIN rag_embedding_jobs j ON (p.workspace,p.job)=(j.workspace,j.id)
                    WHERE (p.workspace,p.kind,p.id,p.content_key)=
                          (ix.workspace,ix.kind,ix.id,ix.content_key)
                      AND j.profile=$1 AND p.state='failed'
                )
            ) AND NOT EXISTS (
                SELECT 1 FROM rag_embedding_jobs j WHERE j.workspace=w.id
                AND j.profile=$1 AND j.state IN {_ACTIVE_STATES}
            ) ORDER BY w.id LIMIT 4 FOR UPDATE OF w SKIP LOCKED""",
        profile,
    )
    for row in rows:
        await enqueue(conn, row["id"], profile, automatic=True)


async def _claim(conn, profile: str):
    row = await conn.fetchrow(
        f"""SELECT j.workspace,j.id FROM rag_embedding_jobs j
            JOIN rag_workspaces w ON w.id=j.workspace AND w.deleted_at IS NULL
            WHERE j.profile=$1 AND j.state IN {_ACTIVE_STATES} AND j.ready_at<=now()
              AND (j.lease_until IS NULL OR j.lease_until<=now())
            ORDER BY j.ready_at,j.created_at LIMIT 1 FOR UPDATE OF j SKIP LOCKED""",
        profile,
    )
    if not row:
        return None, []
    wid, job_id = row["workspace"], row["id"]
    token = uuid4().hex
    await conn.execute(
        """UPDATE rag_embedding_jobs SET state='running',lease_token=$3,
               lease_until=now()+interval '3 minutes' WHERE workspace=$1 AND id=$2""",
        wid,
        job_id,
        token,
    )
    items = await conn.fetch(
        """SELECT p.kind,p.id,p.content_key,i.content FROM rag_embedding_items p
           LEFT JOIN rag_graph_index_inputs i USING (workspace,kind,id,content_key)
           WHERE p.workspace=$1 AND p.job=$2 AND p.state='pending' ORDER BY p.kind,p.id LIMIT 16""",
        wid,
        job_id,
    )
    return (wid, job_id, token), items


async def _index_items(items):
    texts = [item["content"] for item in items if item["content"] is not None]
    if not texts:
        return []
    try:
        return await embed(texts)
    except Exception as exc:
        if not isinstance(exc, InvalidEmbedding) and getattr(exc, "status_code", None) not in {
            400,
            413,
            422,
        }:
            raise
    # A permanent input error should not starve the rest of the batch.
    results = []
    for text in texts:
        try:
            results.append((await embed([text]))[0])
        except Exception as exc:
            if not isinstance(exc, InvalidEmbedding) and getattr(exc, "status_code", None) not in {
                400,
                413,
                422,
            }:
                raise
            results.append(None)
    return results


async def index_once(pool) -> bool:
    if not available("embedding"):
        return False
    profile = embedding_profile()
    async with pool.acquire() as conn, conn.transaction():
        await _auto_enqueue(conn, profile)
        claim, items = await _claim(conn, profile)
    if claim is None:
        return False
    wid, job_id, token = claim
    try:
        vectors = iter(await asyncio.wait_for(_index_items(items), timeout=120))
    except Exception:
        await pool.execute(
            """UPDATE rag_embedding_jobs SET state='waiting_for_provider',lease_token=NULL,
                   lease_until=NULL,ready_at=now()+interval '10 seconds'
               WHERE workspace=$1 AND id=$2 AND lease_token=$3""",
            wid,
            job_id,
            token,
        )
        return True
    async with pool.acquire() as conn, conn.transaction():
        # Follow the same workspace-first lock order as graph writers and backfill creation.
        active = await conn.fetchval(
            "SELECT 1 FROM rag_workspaces WHERE id=$1 AND deleted_at IS NULL FOR UPDATE",
            wid,
        )
        current = await conn.fetchval(
            """SELECT 1 FROM rag_embedding_jobs
               WHERE workspace=$1 AND id=$2 AND lease_token=$3 FOR UPDATE""",
            wid,
            job_id,
            token,
        )
        if not current:
            return True
        if not active or profile != embedding_profile():
            await conn.execute(
                """UPDATE rag_embedding_jobs SET state='queued',lease_token=NULL,lease_until=NULL
                   WHERE workspace=$1 AND id=$2""",
                wid,
                job_id,
            )
            return True
        for item in items:
            state = "skipped"
            if item["content"] is not None:
                vector = next(vectors)
                if vector is None:
                    state = "failed"
                else:
                    stored = await conn.fetchval(
                        """INSERT INTO rag_graph_vectors (workspace,kind,id,profile,content_key,embedding)
                           SELECT workspace,kind,id,$5,content_key,$6::vector
                           FROM rag_graph_index_inputs
                           WHERE (workspace,kind,id,content_key)=($1,$2,$3,$4)
                           ON CONFLICT (workspace,kind,id,profile) DO UPDATE
                           SET content_key=excluded.content_key,embedding=excluded.embedding RETURNING 1""",
                        wid,
                        item["kind"],
                        item["id"],
                        item["content_key"],
                        profile,
                        vector,
                    )
                    if stored:
                        state = "indexed"
            await conn.execute(
                """UPDATE rag_embedding_items SET state=$5
                   WHERE (workspace,job,kind,id)=($1,$2,$3,$4)""",
                wid,
                job_id,
                item["kind"],
                item["id"],
                state,
            )
        await conn.execute(
            """UPDATE rag_embedding_jobs SET lease_token=NULL,lease_until=NULL,ready_at=now(),
                   state=CASE WHEN EXISTS(SELECT 1 FROM rag_embedding_items
                              WHERE workspace=$1 AND job=$2 AND state='pending') THEN 'queued'
                              WHEN EXISTS(SELECT 1 FROM rag_embedding_items
                              WHERE workspace=$1 AND job=$2 AND state='failed') THEN 'failed'
                              ELSE 'completed' END
               WHERE workspace=$1 AND id=$2""",
            wid,
            job_id,
        )
    return True


async def embedding_worker(pool) -> None:
    while True:
        try:
            if graph_storage_ready() and available("embedding"):
                # A workspace can predate this schema and have no public graph records yet.
                # Lazy reads handle it immediately. The worker also discovers untouched workspaces.
                pending = await pool.fetch("""SELECT w.id,w.lightrag_workspace FROM rag_workspaces w
                       LEFT JOIN rag_graph_imports i ON i.workspace=w.id
                       WHERE w.deleted_at IS NULL AND i.workspace IS NULL
                       ORDER BY w.id LIMIT 4""")
                for ws in pending:
                    try:
                        async with graph_transaction(dict(ws)):
                            pass
                    except Exception:
                        logging.exception("Could not import graph for workspace %s", ws["id"])
            if await index_once(pool):
                continue
        except Exception:
            logging.exception("Graph embedding worker failed")
        await asyncio.sleep(2)
