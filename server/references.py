"""Citation/reference resolution: map LightRAG's internal `{job_id}_{basename}` citation keys to
the real, openable document paths stored in Postgres (rag_file_metadata), for /query, /query/data
and graph.html. The shared asyncpg pool is read from the package root (server._db_pool) at call time.
"""

import re

import server


def _ref_basename(path: str) -> str:
    """Display filename from a stored file_path (absolute path or bare name)."""
    return path.replace("\\", "/").rsplit("/", 1)[-1] if path else path


def _safe_ref_name(name: str | None) -> str:
    """Basename a caller-supplied filename ourselves (strip any '/' or '\\' and directory
    parts). Used to build the LightRAG identity `{job_id}_{safe}`: with no path separator,
    LightRAG's basename canonicalization can never drop the unique `job_id` prefix, so distinct
    documents keep distinct keys by construction."""
    return (name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()


def _clean_needles(needles: list[str] | None) -> list[str]:
    """Drop blank/whitespace-only entries from a file_path_contains list. An all-blank list
    (e.g. Swagger UI's auto-populated `[""]`) therefore means 'no filter → all data', never an
    accidental empty result set."""
    return [n for n in (needles or []) if n and n.strip()]


_JOB_KEY_PREFIX = re.compile(r"^[0-9a-f]{6,}_")


def _strip_job_prefix(name: str) -> str:
    """Strip a leading `{job_id}_` hex prefix from a citation key so no internal token/hex
    surfaces when a reference can't be resolved to a metadata row. Unchanged if absent."""
    return _JOB_KEY_PREFIX.sub("", name) if name else name


def _rewrite_answer_refs(result: str, raw_references, meta_map: dict[str, dict]) -> str:
    """Rewrite LightRAG's internal citation keys embedded in the answer PROSE to the clean
    filename — the same no-internal-name guarantee we give the structured `references[]`.

    LightRAG tags each retrieved chunk with its `file_path` (our internal `{job_id}_{basename}`
    join key) and its answer prompt renders a `### References` list from those values, so the raw
    key leaks into `result`. Each key is a unique, slash-free `{job_id}_{basename}` token, so a
    literal replace is safe (it cannot collide with unrelated prose). A resolved key becomes the
    clean original basename (`_safe_ref_name(row.file)`); an unresolved key falls back to stripping
    its `{job_id}_` prefix so no hex/internal token ever surfaces. Pure string mapping over values
    we mint and store, so it stays correct across LightRAG versions and prompt formats."""
    if not result:
        return result
    replacements: dict[str, str] = {}
    for ref in raw_references or []:
        key = ref.get("file_path") or ""
        if not key or key in replacements:
            continue
        row = meta_map.get(key)
        display = _safe_ref_name(row["file"]) if row else _strip_job_prefix(key)
        if display and display != key:
            replacements[key] = display
    # Longest key first: a key that is a substring of another can't be partially rewritten.
    for key in sorted(replacements, key=len, reverse=True):
        result = result.replace(key, replacements[key])
    return result


def _path_matches_any(value: str | None, needles: list[str] | None) -> bool:
    """OR-filter a stored file_path against a list of case-insensitive substrings.
    `value` may be a GRAPH_FIELD_SEP-joined list of paths (entities/relationships carry the
    joined source-file list); a plain substring test still matches within it. Empty/None/blank
    `needles` => no filter (keep everything); a non-empty needle list with an empty `value`
    => no match."""
    needles = _clean_needles(needles)
    if not needles:
        return True
    if not value:
        return False
    lowered = value.lower()
    return any(n.lower() in lowered for n in needles)


async def _db_fetch_metadata_by_key(pool, keys: list[str], phys: str | None) -> dict[str, dict]:
    """Map each LightRAG citation key -> its rag_file_metadata row. Layered so a real path always
    resolves even across LightRAG canonicalization quirks:
      1. exact match on `lightrag_key` (the value LightRAG stored, read back at ingest);
      2. fallback: the original filename column `file` == key (rescues legacy rows whose
         backfilled key was a full path but LightRAG returned only the basename).
    Newest upload wins on duplicates. `file_path` returned here is the REAL display path."""
    if not keys:
        return {}
    cols = (
        "lightrag_key, file_path, file, job_id, description, source_path, "
        "last_modified_time, uploaded_at, llm_model_extracted"
    )
    if phys is not None:
        rows = await pool.fetch(
            f"SELECT {cols} FROM rag_file_metadata "
            "WHERE workspace = $2 AND (lightrag_key = ANY($1) OR file = ANY($1)) "
            "ORDER BY uploaded_at DESC",
            keys,
            phys,
        )
    else:
        rows = await pool.fetch(
            f"SELECT {cols} FROM rag_file_metadata "
            "WHERE (lightrag_key = ANY($1) OR file = ANY($1)) ORDER BY uploaded_at DESC",
            keys,
        )
    by_key: dict[str, dict] = {}
    by_file: dict[str, dict] = {}
    for r in rows:  # newest-first; setdefault keeps the newest
        d = dict(r)
        if d.get("lightrag_key"):
            by_key.setdefault(d["lightrag_key"], d)
        if d.get("file"):
            by_file.setdefault(d["file"], d)
    resolved = {}
    for k in keys:
        m = by_key.get(k) or by_file.get(k)
        if m is not None:
            resolved[k] = m
    return resolved


def _graph_field_sep() -> str:
    """LightRAG's multi-value delimiter for joined source lists (entities/relationships carry
    several sources in one `file_path`). Read it from LightRAG so we stay in harmony with its
    value; fall back to the stable literal if the import is unavailable (e.g. under test stubs)."""
    try:
        from lightrag.utils import GRAPH_FIELD_SEP as sep

        if isinstance(sep, str) and sep:
            return sep
    except Exception:
        pass
    return "<SEP>"


def _resolve_joined_path(value: str, meta_map: dict[str, dict], sep: str) -> str:
    """Resolve a (possibly GRAPH_FIELD_SEP-joined) list of LightRAG keys to real document paths,
    preserving the SEP structure. Each segment maps to its rag_file_metadata `file_path`; an
    unresolved segment falls back to a prefix-stripped basename so no `{job_id}_` token surfaces."""
    return sep.join(
        ((meta_map.get(s.strip()) or {}).get("file_path") or _strip_job_prefix(s.strip()))
        for s in value.split(sep)
        if s.strip()
    )


async def _resolve_block_file_paths(data: dict, phys: str | None) -> None:
    """Rewrite LightRAG's internal citation keys in entity/relationship/chunk `file_path` fields to
    the REAL document path from Postgres (same `rag_file_metadata` join as references), so the raw
    `{job_id}_{basename}` key never surfaces in `/query/data`. Multi-source fields are GRAPH_FIELD_
    SEP-joined lists; every segment is resolved and the SEP structure preserved. Mutates in place.
    """
    sep = _graph_field_sep()
    blocks = [data.get("entities") or [], data.get("relationships") or [], data.get("chunks") or []]
    keys = {
        s.strip()
        for block in blocks
        for item in block
        for s in (item.get("file_path") or "").split(sep)
        if s.strip()
    }
    if not keys:
        return
    meta_map = (
        await _db_fetch_metadata_by_key(server._db_pool, list(keys), phys)
        if server._db_pool
        else {}
    )
    for block in blocks:
        for item in block:
            fp = item.get("file_path")
            if fp:
                item["file_path"] = _resolve_joined_path(fp, meta_map, sep)


async def _resolve_graph_paths(elements, phys: str | None) -> None:
    """Same real-path resolution as `_resolve_block_file_paths`, for knowledge-graph nodes AND
    edges whose `properties['file_path']` (shown in graph.html tooltips and used by its file_path
    filter) is a GRAPH_FIELD_SEP-joined list of internal keys. Both nodes and relationships carry
    source lists, so pass them together for a single DB fetch. Mutates properties in place."""
    sep = _graph_field_sep()
    keys = {
        s.strip()
        for el in elements
        for s in ((el.properties or {}).get("file_path") or "").split(sep)
        if s.strip()
    }
    if not keys:
        return
    meta_map = (
        await _db_fetch_metadata_by_key(server._db_pool, list(keys), phys)
        if server._db_pool
        else {}
    )
    for el in elements:
        props = el.properties or {}
        fp = props.get("file_path")
        if fp:
            props["file_path"] = _resolve_joined_path(fp, meta_map, sep)
            el.properties = props


async def _build_references(
    raw_references: list[dict] | None, phys: str | None = None, answered_model: str | None = None
) -> tuple[list[dict], dict[str, dict]]:
    """Resolve LightRAG references to their REAL document path + metadata via rag_file_metadata,
    joining on the citation key LightRAG returns (matched against `lightrag_key`). Shared by
    /query and /query/data. Returns `(references, meta_map)` where `meta_map` is the internal
    key→row map (used by /query to rewrite the answer prose; never emitted in any response).

    References expose only the real, openable `file_path` (from Postgres) plus enrichment.
    LightRAG's internal name / our join key are NEVER emitted; when a reference can't be resolved
    to a row, `file_path` is null (we do not echo LightRAG's raw internal value).

    `answered_model` is the text LLM that synthesised the answer for THIS query; added as
    `llm_model_answered` only when supplied (so /query/data, which generates no answer, omits it).
    """
    keys = []
    for ref in raw_references or []:
        keys.append(ref.get("file_path", "") or "")  # LightRAG's citation key (its internal name)
    meta_map: dict[str, dict] = {}
    if any(keys) and server._db_pool:
        meta_map = await _db_fetch_metadata_by_key(server._db_pool, [k for k in keys if k], phys)
    references = []
    for ref, key in zip(raw_references or [], keys):
        m = meta_map.get(key)
        out = {
            "reference_id": ref.get("reference_id"),
            "file_path": m.get("file_path") if m else None,  # REAL path only; never the key
            "job_id": m.get("job_id") if m else None,
            "file_description": m.get("description") if m else None,
            "last_modified_time": m.get("last_modified_time") if m else None,
        }
        ua = m.get("uploaded_at") if m else None
        out["uploaded_at"] = ua.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(ua, "strftime") else ua
        out["llm_model_extracted"] = m.get("llm_model_extracted") if m else None
        if answered_model is not None:
            out["llm_model_answered"] = answered_model
        references.append(out)
    return references, meta_map
