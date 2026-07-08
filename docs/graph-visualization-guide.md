# Guide: Interactive Knowledge-Graph Visualization in Python

A self-contained, project-agnostic guide to rendering a graph of nodes and
edges as an interactive HTML page.

## 1. What to install

```bash
pip install pyvis
```

`pyvis` is a Python wrapper around **vis-network** (a JavaScript graph library).
You build the graph in Python; it emits an HTML/JS page that renders an
interactive, draggable, force-directed diagram in any browser. That's the only
required dependency — no server, no CDN needed.

## 2. The core idea

You give pyvis:

- **nodes** — each with an id, a label, a color, a size, and an optional hover
  tooltip (`title`)
- **edges** — pairs of node ids, optionally with a tooltip

pyvis produces a single `.html` file. With `cdn_resources="in_line"` the
vis-network JS/CSS is embedded directly, so the file is **self-contained and
works offline** (openable by double-click, emailable, no internet).

## 3. Minimal working example

```python
from pyvis.network import Network

net = Network(height="100vh", width="100%", directed=True,
              bgcolor="#1a1a1a", font_color="#eaeaea",
              cdn_resources="in_line")

net.add_node("a", label="Alice", title="Person")
net.add_node("b", label="Acme", title="Company")
net.add_edge("a", "b", title="works_at")

net.save_graph("graph.html")   # or: html = net.generate_html()
```

Open `graph.html` in a browser — you can drag nodes, zoom, and pan.

## 4. A production-quality builder

This is the pattern most real knowledge-graph viewers use: **color nodes by
category, size them by connectivity, and show full metadata on hover.**

```python
from datetime import datetime, timezone
from pyvis.network import Network

# A stable categorical palette (colorblind-friendly). vis-network accepts CSS color strings.
PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1f77b4", "#d62728",
]

# Tooltips are rendered by vis-network as ESCAPED PLAIN TEXT, so HTML tags show
# literally. Emit "Key: value" lines joined by "\n" and let CSS wrap them.
TOOLTIP_CSS = """
<style>
.vis-tooltip {
  white-space: pre-wrap !important;
  max-width: 380px;
  font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif !important;
  font-size: 12px !important; line-height: 1.45 !important;
  padding: 8px 11px !important; border-radius: 6px !important;
  background-color: #2b2b2b !important; color: #eaeaea !important;
  border: 1px solid #555 !important;
  box-shadow: 0 2px 10px rgba(0,0,0,0.45) !important;
}
</style>
"""


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

    net = Network(height="100vh", width="100%", directed=True,
                  bgcolor="#1a1a1a", font_color="#eaeaea",
                  cdn_resources="in_line")
    net.toggle_physics(physics)

    node_ids = {n["id"] for n in nodes}
    for n in nodes:
        t = n.get("type", "unknown")
        net.add_node(
            n["id"],
            label=str(n.get("label", n["id"])),
            title=props_tooltip({"type": t, **n.get("props", {})}),
            color=color_of[t],
            size=12 + 3 * degree.get(n["id"], 0),   # bigger = more connected
        )

    for e in edges:
        # Guard against edges pointing at nodes you didn't add (e.g. after truncation).
        if e["source"] in node_ids and e["target"] in node_ids:
            net.add_edge(e["source"], e["target"], title=props_tooltip(e.get("props", {})))

    # Inject the tooltip CSS so the "\n"-separated lines wrap legibly.
    html = net.generate_html()
    return html.replace("</head>", TOOLTIP_CSS + "</head>", 1)
```

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
| `cdn_resources="in_line"` | Makes the HTML a single offline file — no external JS. |
| Color by category, sorted deterministically | Same category → same color on every render; easy visual grouping. |
| Size by **degree** (`12 + 3*deg`) | Important/hub nodes pop out visually. |
| Tooltip as `\n`-joined `Key: value` + `white-space: pre-wrap` CSS | vis-network escapes HTML in tooltips, so you can't use `<br>`; this is the reliable way to get multi-line hover cards. |
| `physics=False` for large graphs | The force simulation is expensive; freeze the layout once it's big. |
| Guard edges against missing node ids | Prevents broken/invisible edges when nodes are filtered or capped. |

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

- **`pyvis`** — easiest for interactive HTML (recommended here).
- **`networkx` + `matplotlib`** — static PNG images, good for reports, no interactivity.
- **`plotly`** — interactive, integrates with dashboards, more setup for graph layouts.
- **`ipysigma` / `graphviz`** — Jupyter-native or publication-grade static layouts respectively.

For an interactive, self-contained, shareable knowledge-graph page, **pyvis is
the shortest path.**
