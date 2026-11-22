# Build a knowledge graph from facts you already have

Start with the data you already trust. Write structured facts directly through REST or MCP, then enrich the same graph by uploading documents for automatic extraction. Each workspace holds both kinds of facts in one editable graph. [API reference](api-reference.md)

The examples build a small operations graph. Alice leads a platform team and the team operates a service. You can populate the entire graph from your own data or attach Alice to a team already extracted from a file. Keyword lookup, edits and traversal require only the database. Embeddings add a way to enter the graph when you know a fact's meaning but do not know its name.

## Connect and check the server

For a fresh provider-free deployment, copy `.env.example` to `.env`, set `POSTGRES_PASSWORD`, and leave provider keys and base URLs empty, including the shared `OPENAI_API_KEY`. Leave `POSTGRES_WORKSPACE` unset. Start the checkout with `docker compose up -d --build`. Check `/health` for process liveness, then inspect `/openapi.json` for feature availability. On a shared installation use an existing workspace or your own new workspace. [Deployment settings](configuration.md)

The REST examples use the Python 3 standard library and run in order in one interpreter. Set `POLYGRAPHRAG_URL` if your server uses another address. If the server enables `API_TOKENS`, supply an authorized token through the client environment variable `POLYGRAPHRAG_TOKEN`. Keep the token out of example payloads and logs. You can also make the REST calls in Swagger UI at `/docs`. [MCP client setup](api-reference.md#mcp)

```python
import json
import os
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

base = os.getenv("POLYGRAPHRAG_URL", "http://localhost:9622").rstrip("/")
token = os.getenv("POLYGRAPHRAG_TOKEN")

def api(method, path, body=None, **params):
    url = base + path
    if params:
        url += "?" + urlencode(params, doseq=True)
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(body).encode()
    request = Request(url, data=payload, method=method, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

spec = api("GET", "/openapi.json")
if not any(path.endswith("/graph/batch") for path in spec["paths"]):
    raise RuntimeError("This deployment does not advertise manual graph writes.")
```

Choose an existing workspace from `GET /all-workspaces/list` or create a new one. The example creates `graph_demo` only when the active list lacks that ID. If creation returns a conflict, check the deleted-workspace list. Restoring an existing workspace is a separate decision.

```python
workspace = "graph_demo"
active = api("GET", "/all-workspaces/list")["workspaces"]
if not any(row["id"] == workspace for row in active):
    api("POST", "/all-workspaces/create", {"id": workspace, "name": "Graph demo"})
ws = "/workspace/" + quote(workspace, safe="")
```

## Choose identities and write useful descriptions

Give a new record a durable ID from the system that owns it. For example, `person-184` can remain stable when Alice changes her surname. If you generate IDs locally, persist them alongside the source records. Give each relationship its own ID too. Alice can both lead and have founded the same team. Write each fact under its own edge ID and use the direction in which the sentence reads.

Give descriptions enough context for a human to disambiguate the result and for an embedding search to find it. “Coordinates the production on-call rotation and leads incident response for the payments service” is more informative than “Platform.” Put the explicit fact that Alice leads the team in a `LEADS` edge. The description helps discovery. The edge makes traversal deterministic.

Keep provenance at the level you can substantiate. The optional `source_ref` below is a caller-owned reference, such as an entry in a reviewed service catalog. Omit it for unsourced assertions. Use the real workspace entity ID when attaching to an extracted fact. The service manages file and chunk evidence from extraction.

## Populate an empty graph in one transaction

Use this example on a new demo workspace. A file upload, model credential and embedding backfill are all optional for this step. The batch uses entity IDs in its relationships, so the server can validate the whole graph before committing.

```python
alice_id = "person-184"
team_id = "team-platform"
service_id = "service-payments"
batch = {
    "entities": [
        {"id": alice_id, "name": "Alice Chen", "type": "Person",
         "description": "Engineering lead for production reliability."},
        {"id": team_id, "name": "Platform Reliability", "type": "Team",
         "description": "Coordinates the production on-call rotation and incident response."},
        {"id": service_id, "name": "Payments API", "type": "Service",
         "description": "Authorizes card payments for the storefront."},
    ],
    "relationships": [
        {"id": "role-184-leads-platform", "source": alice_id, "target": team_id,
         "type": "LEADS", "description": "Alice directs the on-call team.",
         "source_ref": "service-catalog/roles/184"},
        {"id": "platform-operates-payments", "source": team_id, "target": service_id,
         "type": "OPERATES", "description": "The team operates Payments API."},
    ],
}
created = api("POST", ws + "/graph/batch", batch)
print([(entity["id"], entity["revision"]) for entity in created["entities"]])
```

Expect three entities and two edges on an initially empty workspace. An identical replay should produce the same IDs and counts. If even one endpoint is unknown, the entire batch fails. Fix the source payload and retry. A later change to an existing record uses its current revision.

### Use the same batch from an MCP agent

With `MCP_GRAPH_WRITES=true`, call `write_graph_batch` with the same batch and a top-level `workspace_id`. In the [Python MCP client example](api-reference.md#discover-and-call-tools), replace the `list_workspaces` call with `await mcp.call_tool("write_graph_batch", {"workspace_id": workspace, **batch})`. Create or select the workspace through REST first.

Use `get_graph_entity` or `get_graph_relationship` to read the revision before calling `patch_graph_entity` or `patch_graph_relationship`. Supply `workspace_id`, the record ID, `expected_revision`, and the fields to change. Use `show_workspace_graph` to share the result as an interactive graph, HTML file, or browser link. [MCP tool reference](api-reference.md#mcp)

## Attach a manual fact to an extracted entity

This is an alternative starting point for an existing workspace. Set `ws` to its public workspace path. If you just uploaded a document, poll its job through `GET /workspace/{id}/status/{job_id}` until it reaches `done`. For a workspace populated earlier, go straight to lookup.

```python
matches = api("GET", ws + "/graph/entities",
              name="Platform Reliability", match="exact", type="Team")
for candidate in matches["items"]:
    print(candidate["id"], candidate["description"], candidate.get("sources", []))
```

Check the workspace, name, type and source evidence. If two records are plausible, read each by ID and inspect its neighbors. An agent should ask its caller to resolve any remaining ambiguity. An exact match on the display name alone is insufficient. Once the target is established, reuse the **returned** ID. The following example requires exactly one candidate across all pages.

```python
if matches["total"] != 1 or "extracted" not in matches["items"][0]["origins"]:
    raise ValueError("Select the intended team from its source evidence before writing.")
team_id = matches["items"][0]["id"]
alice_id = "person-184"  # known ID from the caller's personnel system
api("POST", ws + "/graph/batch", {
    "entities": [{"id": alice_id, "name": "Alice Chen", "type": "Person",
                  "description": "Engineering lead for production reliability."}],
    "relationships": [{"id": "role-184-leads-platform", "source": alice_id,
                       "target": team_id, "type": "LEADS",
                       "description": "Alice directs the on-call team.",
                       "source_ref": "service-catalog/roles/184"}],
})
```

The extracted team retains its file evidence. The edge has manual provenance. If a person from the source system already corresponds to an extracted Alice under another ID, resolve Alice the same way and use her returned ID in the edge. You can submit a batch containing only relationships. This avoids creating a duplicate person.

## Find records and follow edges with all providers off

```python
people = api("GET", ws + "/graph/entities",
             q='(name:Alice OR name:Bob) AND type:Person')
popular = api("GET", ws + "/graph/entities", sort="degree", limit=10)
path = api("GET", ws + "/graph/entities/" + quote(alice_id, safe="") + "/neighbors",
           hops=2, direction="out", max_nodes=100)
print([(edge["source"], edge["type"], edge["target"]) for edge in path["relationships"]])
```

In the empty-graph example you should see Alice → Platform Reliability → Payments API. An extracted undirected association can be explored using `direction=both`. `truncated=true` means the response reached its budget. For a focused follow-up, start at a returned entity ID or request one relationship type.

| Intent | Input |
|---|---|
| Find the literal name “Research AND Development” | `name="Research AND Development"` as a query parameter value |
| Find a phrase in descriptions | `q='description:"incident response" AND type:Team'` |
| Find either of two people | `q='(name:Alice OR name:Bob) AND type:Person'` |
| Identify two distinct people named Alice | `name=Alice`, `match=exact`, `type=Person`, then compare IDs, descriptions and evidence |
| Inspect exactly one known entity | `GET /workspace/{id}/graph/entities/{entity_id}` |

`AND` binds more tightly than `OR`. Quote literal phrases containing operators. Pass parameters through a URL encoder like the helper above. The graph viewer at `/workspace/{id}/graph.html` and the workspace overview should reflect the same committed edits and counts.

## Correct a fact, merge duplicates or remove an edge

Read immediately before editing so that the precondition describes the state you reviewed.

```python
entity_path = ws + "/graph/entities/" + quote(team_id, safe="")
team = api("GET", entity_path)
updated = api("PATCH", entity_path, {
    "expected_revision": team["revision"],
    "description": "Coordinates production on-call rotations and leads outage response.",
})
print(updated["id"], updated["revision"])
```

The public ID and incident edges survive the edit. The new description is available immediately to keyword search. Until a replacement vector is ready, semantic search excludes the outdated vector. Re-ingestion may contribute new evidence while preserving the manually corrected field.

- **Correct a relationship.** Read `GET /graph/relationships/{edge_id}` under the workspace, then PATCH its description or type with the observed `expected_revision`. Verify the edge in both endpoint neighborhoods.
- **Merge duplicate entities.** POST `{"survivor_id":"team-platform","retire_ids":["team-platform-duplicate"]}` to `/graph/entities/merge/preview`. Inspect the evidence, edges, field conflicts and self-loops. Apply the returned `preview_token` to `/graph/entities/merge/apply` only when the survivor and resolutions are authorized. Re-preview on a revision conflict.
- **Remove a wrong edge.** DELETE `/graph/relationships/{edge_id}?expected_revision=N` with the revision you read. An explicit entity deletion also hides incident edges. Deleting an uploaded file withdraws that file's contributions. Choose the operation matching the intended scope.
- **Recover from a timeout.** Read by stable ID before retrying a write. An identical creation batch is safe to replay. On `409`, compare current fields and revisions before resubmitting a change.

A merge apply body can specify the final values for the conflicting fields and consent to self-loops. For example, with a conflict on `type`, send `{"preview_token":"<preview_token>","resolutions":{"type":"Team"},"allow_self_loops":false}`. A lookup by retired ID returns the survivor and includes the old ID in `requested_id`. New IDs are unique within their workspace and record kind. The `lr.` prefix identifies records imported from LightRAG.

## Find a starting point by meaning

With an embedding provider configured and indexing complete, “who handles outages?” can lead to the team through its description. Traversing the incoming `LEADS` edge then identifies Alice. The graph supplies that relationship even when Alice has no vector. The endpoint returns data a client can inspect directly.

```python
result = api("POST", ws + "/graph/retrieve", {
    "text": "who handles outages?", "mode": "hybrid", "q": "type:Team",
    "seed_kind": "entity", "seed_limit": 3, "hops": 1, "max_nodes": 50,
})
print(result["seeds"], result["coverage"])
print([(edge["source"], edge["type"], edge["target"]) for edge in result["relationships"]])
```

`type:Team` restricts the starting points. The returned neighbors may include people or services. Set `hops=0` to see matches alone. Use `seed_kind="relationship"` with `text="responsibility for running payments"` and optionally `q="type:OPERATES"` to start from an edge description and recover both endpoints. `semantic` ranks by vector similarity. `hybrid` also considers keyword matches. Ranking is data- and model-dependent. For a known relationship, ID lookup and exact traversal give a deterministic answer.

Clients using `POST /workspace/{id}/query/data` can supply keywords. Send the following body with an embedding provider configured. The response uses `data.entities`, `data.relationships`, `data.chunks` and `data.references`. Local mode with supplied entity terms retrieves context without a query LLM.

```json
{
  "query": "Who leads Platform Reliability?",
  "mode": "local",
  "ll_keywords": ["Platform Reliability", "Alice Chen"],
  "hl_keywords": [],
  "include_references": true
}
```

## Inspect missing embeddings and trigger backfill

List records before deciding whether to spend embedding capacity. The list and consistency report work while the provider is offline.

```python
filters = {"kind": "entity", "origin": "manual", "state": "needs_embedding", "limit": 50}
page = api("GET", ws + "/embeddings/missing", **filters)
print("Total matching entities:", page["total"])
while True:
    for item in page["items"]:
        print(item["id"], item["state"])
    cursor = page["next_cursor"]
    if cursor is None:
        break
    page = api("GET", ws + "/embeddings/missing", cursor=cursor, **filters)
```

`missing` means the record has never received a vector. `stale` means its content or embedding profile has changed. `needs_embedding` includes both. `total` is the count across all pages for the selected filters. For a count alone, request `limit=1` and read `total`. A cursor is an opaque bookmark. Return it unchanged with the same filters. Concurrent edits or backfill can change the live list. After a scan, read the current total again if the final count matters.

In automatic mode, eligible work is picked up when the provider is available. In on-demand mode, call the authenticated backfill endpoint after an offline import. It works as a hook for an external automation. Configure the embedding model, dimension and endpoint, restart the service to load deployment configuration, then submit the request.

Set `EMBEDDING_INDEX_MODE=on_demand` to trigger indexing explicitly. Use `auto` for background catch-up. To enable embeddings alone, set `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM` and the embedding endpoint's credential if required. A role-specific endpoint and credential let the extraction and query LLM settings stay empty. [Provider routing](configuration.md)

```python
job = api("POST", ws + "/embeddings/backfill")
job_path = ws + "/embeddings/backfill/" + quote(job["job_id"], safe="")
status = api("GET", job_path)
print(job["job_id"], status["state"])
```

Poll that URL with backoff until `completed` or `failed`. If an accepted job says `waiting_for_provider`, repair the provider configuration or connectivity and follow the same job. Repeated triggers reuse active work. After completion, check the consistency report and remaining missing/stale counts for each kind. A job skips inputs edited after it was queued. Automatic mode picks up the new text. In on-demand mode, trigger another backfill. A model change requires fresh vectors even when dimensions match. Graph IDs and links survive reindexing.

## Handle capability and validation errors

| Result | Client action |
|---|---|
| `401` | Supply the service token. Provider credentials belong to the service, separate from the client token. |
| `404` | Check workspace, public ID and deletion state. Inspect OpenAPI if the route itself is missing. |
| `409` | Read the resource or merge preview again. Retry a writer conflict after backoff. Reconcile a revision conflict before changing facts. |
| `422` | Correct the Boolean syntax, bounds, cursor or batch endpoint IDs. A rejected batch commits zero items. |
| `503 provider_not_configured` | Read the missing role(s) from `detail.roles`. Configure that capability or explicitly choose a provider-free method. |
| `503 provider_unavailable` | A configured provider could not serve the request. Retry with backoff as appropriate. An accepted background operation has its own job status. |

For example, an explicit semantic search on an offline-only deployment should return a body like the following.

```json
{
  "detail": {
    "code": "provider_not_configured",
    "operation": "semantic_search",
    "roles": ["embedding"],
    "message": "Configure the required model provider"
  }
}
```

When all providers are disabled, finish the workflow by reading the stored IDs, counting the visible entities, traversing the expected path and checking the missing-vector total. When embeddings are enabled later, run the same exact reads, then add a semantic retrieval check. Successful graph writes and fully indexed semantic search are separate outcomes.
