# LightRAG internals

PolyGraphRAG delegates the actual RAG work to [LightRAG](https://github.com/HKUDS/LightRAG). This page summarizes what LightRAG does under the hood so you can reason about ingestion cost, retrieval quality, and the Postgres tables involved. It reflects the behavior relied on by this service; the upstream repo is the source of truth for exact details.

## Ingestion: from document to graph

When a document is inserted (`ainsert`), LightRAG runs roughly:

1. **Chunking.** The document text is split into token-bounded chunks (with overlap). Each chunk is stored and embedded.
2. **Entity & relationship extraction.** For each chunk, the **text LLM** (routed to `LLM_*` — the *extraction* phase) is prompted to emit entities (name, type, description) and relationships (source, target, description, keywords, weight). This is the token-heavy step — route it at a cheap provider.
3. **Merging / dedup.** Entities and relationships with the same identity across chunks are merged; descriptions are combined and summarized. This produces a single graph node per real-world entity, even when it appears in many files.
4. **Persistence.** Nodes and edges are written to the **Apache AGE** graph; entity/relationship/chunk embeddings are written to **pgvector**.

PolyGraphRAG adds an integrity gate after this: it confirms the document reached `processed` and (for non-trivial content) that entities were actually produced before marking the ingest job done.

## Retrieval: dual-level

At query time LightRAG extracts keywords from the question and retrieves at two levels, combined per the requested **mode**:

- **Local** — entity-centric. Finds the most relevant entities and expands to their neighborhood (attached relationships and source chunks). Best for specific, factual questions.
- **Global** — relationship/theme-centric. Finds the most relevant relationships and reasons over broader themes. Best for "how do these connect / what's the big picture" questions.
- **Hybrid** — union of local + global.
- **Mix** (PolyGraphRAG default) — graph retrieval blended with plain vector search over chunks.
- **Naive** — plain vector similarity over chunks, no graph.

The retrieved entities, relationships, and chunks become the context the **query LLM** (routed to `QUERY_LLM_*`) synthesizes the answer from. `top_k` controls how many entities/relationships are retrieved per keyword set.

`/query/data` returns exactly this retrieved evidence with no synthesis step — handy for inspection and for building custom experiences.

## What lives in Postgres

| Data | Storage | Table(s) |
|---|---|---|
| Knowledge graph (entities, relationships) | Apache AGE | `chunk_entity_relation` and `{workspace}_chunk_entity_relation` graphs |
| Entity / relationship / chunk embeddings | pgvector | `lightrag_vdb_*_<model>_<dim>d` |
| Document status & full-doc records | KV / doc-status | `lightrag_doc_status`, `lightrag_*` KV tables |
| Fuzzy text matching | pg_trgm | trigram indexes |

All LightRAG tables carry a `workspace` column, which is how one Postgres serves many isolated projects (see [architecture.md](architecture.md#workspace-isolation-multi-project)).

## Edge properties on Apache AGE

A subtle AGE behavior shaped this project's history. When writing edge properties, the intuitive Cypher `SET r += {map}` (and `ON CREATE/ON MATCH SET`) **silently drops the properties on AGE** — the edge is created but `description`, `keywords`, `weight`, `source_id`, and `file_path` are never persisted, leaving relations with empty `{}` property maps. Nodes are unaffected; the bug is edge-specific and only manifests on the AGE graph backend (file-backed/NetworkX storage is fine).

The reliable fix is to write edge properties **inline in a `CREATE`**, deleting any existing edge first to keep the upsert idempotent (`OPTIONAL MATCH … DELETE … CREATE (…)-[r:REL {props}]->(…)`).

This service previously carried that fix as a large vendored copy of LightRAG's `postgres_impl.py`, which locked it to one LightRAG version. The fix was **authored and contributed upstream by [Oleksandr Kushnir](https://github.com/oleksandr-kushnir)**, merged into LightRAG as [HKUDS/LightRAG#3052](https://github.com/HKUDS/LightRAG/pull/3052), so PolyGraphRAG runs stock LightRAG (pinned in `requirements.txt`) with no patch. If you ever run a graph query and see all-empty edge property maps, you're on a LightRAG build that predates the fix.

### Verifying edges persist

After ingesting a document, edges should carry real properties:

```sql
SELECT * FROM cypher('chunk_entity_relation',
  $$ MATCH ()-[r]->() RETURN properties(r) LIMIT 5 $$) AS (props agtype);
-- Expect populated maps (description/keywords/weight/…), not {}
```
