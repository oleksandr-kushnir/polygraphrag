# Guide: Interactive Knowledge-Graph Visualization in Python

A self-contained, project-agnostic guide to rendering a graph of nodes and
edges as an interactive HTML page — using **D3.js v7** with the force layout
drawn on an HTML `<canvas>`.

## 1. What to install

Nothing from pip. The visualization is pure client-side JavaScript: you vendor
one file, [`d3.v7.min.js`](https://d3js.org/) (~280 KB, the full D3 v7 bundle),
and inline it into the page. Python's only job is to build a small JSON blob and
drop it, plus the D3 source, into an HTML template string.

```bash
# one-time: fetch the D3 v7 bundle and commit it next to your code
curl -L https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js -o vendor/d3.v7.min.js
```

Inlining D3 (rather than loading it from a CDN) is what makes the exported page
**self-contained and offline** — openable by double-click, emailable, no
internet required.

## 2. The core idea

You give the page:

- **nodes** — each with an `id`, a `label`, a `color`, a `degree` (drives size),
  and a plain-text `tooltip`
- **links** — pairs of node ids, each with a `tooltip`

D3's [`forceSimulation`](https://github.com/d3/d3-force) computes an
(x, y) position for every node by simulating repulsion + link springs, and you
repaint the whole graph on a single `<canvas>` each tick. Canvas draws the entire
scene with one 2D context — there is no per-node DOM element — so it stays light
and smooth even for thousands of nodes, where an SVG/DOM approach bogs down.

## 3. Minimal working example

```html
<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<canvas id="graph"></canvas>
<script>/* contents of vendor/d3.v7.min.js inlined here */</script>
<script>
const nodes = [{id: "a", label: "Alice"}, {id: "b", label: "Acme"}];
const links = [{source: "a", target: "b"}];
const canvas = document.getElementById("graph");
const ctx = canvas.getContext("2d");
canvas.width = innerWidth; canvas.height = innerHeight;

const sim = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(100))
  .force("charge", d3.forceManyBody().strength(-250))
  .force("center", d3.forceCenter(innerWidth / 2, innerHeight / 2))
  .on("tick", draw);

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#999";
  for (const l of links) { ctx.beginPath(); ctx.moveTo(l.source.x, l.source.y); ctx.lineTo(l.target.x, l.target.y); ctx.stroke(); }
  for (const n of nodes) { ctx.beginPath(); ctx.arc(n.x, n.y, 8, 0, 2 * Math.PI); ctx.fillStyle = "#4e79a7"; ctx.fill(); }
}
</script>
</body></html>
```

`d3.forceLink(...).id(d => d.id)` resolves each link's string `source`/`target`
into references to the actual node objects, so after the first tick `l.source.x`
is a real coordinate.

## 4. A production-quality builder

This is the pattern most real knowledge-graph viewers use: **color nodes by
category, size them by connectivity, show full metadata on hover, and pan/zoom.**
Python prepares the data and fills an HTML template; all interactivity is D3.

```python
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

# A stable categorical palette (colorblind-friendly). Canvas accepts CSS color strings.
PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1f77b4", "#d62728",
]

D3_PATH = Path(__file__).parent / "vendor" / "d3.v7.min.js"


@lru_cache(maxsize=1)
def d3_source() -> str:
    """Read the vendored D3 bundle once (cached across requests)."""
    return D3_PATH.read_text(encoding="utf-8")


def format_value(key, value):
    """Pretty-print a property; render epoch timestamps as readable UTC."""
    if key.endswith("_at") and isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = str(value)
    return text[:800] + "…" if len(text) > 800 else text


def props_tooltip(props, key_order=()):
    """Render a properties dict as multi-line 'Key: value' plain text.
    Empty values dropped, keys title-cased, known fields ordered first."""
    props = props or {}
    keys = [k for k in key_order if k in props] + [k for k in props if k not in key_order]
    lines = []
    for k in keys:
        v = props[k]
        if v is None or str(v).strip() == "":
            continue
        lines.append(f"{k.replace('_', ' ').title()}: {format_value(k, v)}")
    return "\n".join(lines)


def build_graph_html(nodes, edges, *, physics=True):
    """
    nodes: list of dicts like {"id": "n1", "label": "Alice", "type": "Person", "props": {...}}
    edges: list of dicts like {"source": "n1", "target": "n2", "props": {...}}
    Returns a self-contained HTML string.
    """
    # 1. Degree = how many edges touch each node (drives node size).
    degree = {}
    for e in edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1

    # 2. Deterministic type -> color (sorted so a type keeps its color across renders).
    types = sorted({n.get("type", "unknown") for n in nodes})
    color_of = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types)}

    # 3. Build the {nodes, links} blob the browser will lay out and draw.
    out_nodes = [
        {
            "id": n["id"],
            "label": str(n.get("label", n["id"])),
            "color": color_of[n.get("type", "unknown")],
            "degree": degree.get(n["id"], 0),
            "tooltip": props_tooltip({"type": n.get("type", "unknown"), **n.get("props", {})}),
        }
        for n in nodes
    ]
    node_ids = {n["id"] for n in nodes}
    out_links = [
        {"source": e["source"], "target": e["target"], "tooltip": props_tooltip(e.get("props", {}))}
        for e in edges
        # Guard against edges pointing at nodes you didn't add (e.g. after truncation).
        if e["source"] in node_ids and e["target"] in node_ids
    ]

    # 4. Script-safe embedding: neutralize any "</..." that could close <script> early.
    data_json = json.dumps({"nodes": out_nodes, "links": out_links}).replace("</", "<\\/")

    return (
        HTML_TEMPLATE
        .replace("__PHYSICS__", "true" if physics else "false")
        .replace("__DATA_JSON__", data_json)
        .replace("__D3_SOURCE__", d3_source())
    )
```

`HTML_TEMPLATE` is a string holding the page skeleton (see §5): a `<canvas>`, an
overlay tooltip `<div>`, the inlined `__D3_SOURCE__`, the embedded `__DATA_JSON__`,
and a script that runs the simulation + canvas draw loop.

Usage:

```python
nodes = [
    {"id": "1", "label": "Alice", "type": "Person",
     "props": {"description": "Engineer", "created_at": 1700000000}},
    {"id": "2", "label": "Acme",  "type": "Company", "props": {"industry": "Software"}},
]
edges = [{"source": "1", "target": "2", "props": {"relation": "works_at"}}]

with open("graph.html", "w", encoding="utf-8") as f:
    f.write(build_graph_html(nodes, edges, physics=True))
```

## 5. The techniques that matter

| Technique | Why |
|---|---|
| Vendor + inline `d3.v7.min.js` | Makes the HTML a single offline file — no external JS, no CDN. |
| Canvas 2D instead of SVG/DOM nodes | One draw call paints the whole scene; scales to thousands of nodes with low memory and smooth pan/zoom. |
| Color by category, sorted deterministically | Same category → same color on every render; easy visual grouping. |
| Size by **degree** (`radius = 8 + 2*deg`) | Important/hub nodes pop out visually. |
| Tooltip as `\n`-joined `Key: value` in an overlay `<div>` with `white-space: pre-wrap` | Canvas has no hoverable elements, so hit-test the pointer against node radius / segment distance and show the text in a positioned `<div>`. |
| `physics=false` → `tick()` in a loop, then `stop()` | The live force simulation is expensive; on large graphs settle it synchronously and draw once (a static layout). |
| Apply the `d3.zoom` transform via `ctx.translate/scale`, and `devicePixelRatio` on resize | Crisp rendering on HiDPI screens; pan/zoom without touching node coordinates. |
| Guard edges against missing node ids | Prevents broken/invisible edges when nodes are filtered or capped. |

The heart of the client script:

```js
const DATA = __DATA_JSON__, PHYSICS = __PHYSICS__;
const sim = d3.forceSimulation(DATA.nodes)
  .force("link", d3.forceLink(DATA.links).id(d => d.id).distance(100))
  .force("charge", d3.forceManyBody().strength(-250))
  .force("center", d3.forceCenter(innerWidth / 2, innerHeight / 2))
  .force("collision", d3.forceCollide().radius(d => 8 + 2 * d.degree + 4));

// zoom/pan applied to the canvas; the draw() loop uses the current transform
d3.select(canvas).call(d3.zoom().scaleExtent([0.1, 5]).on("zoom", e => { transform = e.transform; draw(); }));

if (PHYSICS) sim.on("tick", draw);                       // animate live
else { sim.stop(); for (let i = 0; i < 300; i++) sim.tick(); draw(); }  // settle then draw once
```

## 6. Serving it (optional)

To serve it from a web framework instead of saving a file, return the HTML
string with the right content type. Example with FastAPI:

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/graph.html", response_class=HTMLResponse)
def graph():
    return HTMLResponse(build_graph_html(nodes, edges))
```

## 7. Alternatives worth knowing

- **D3.js v7 + canvas** — interactive, self-contained, and light on the client
  even for large graphs (recommended here).
- **`pyvis` / vis-network** — quickest to wire up from Python, but renders via
  heavier per-node objects; slower on big graphs.
- **`networkx` + `matplotlib`** — static PNG images, good for reports, no interactivity.
- **`plotly`** — interactive, integrates with dashboards, more setup for graph layouts.
- **`ipysigma` / `graphviz`** — Jupyter-native or publication-grade static layouts respectively.

For an interactive, self-contained, shareable knowledge-graph page that stays
responsive on large graphs, **D3 v7 on a canvas is the sweet spot.**
