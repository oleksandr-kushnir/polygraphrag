"""Knowledge-graph visualization: render a LightRAG KnowledgeGraph into a self-contained
interactive HTML page (D3.js v7, force layout drawn on an HTML <canvas>).

The graph.html endpoint calls `_build_graph_html`; the rest are internal helpers for node
coloring, property tooltips, and inlining the vendored D3 source.
"""

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

# Stable categorical palette (CSS color strings). Entity types are mapped to colors
# deterministically by sorted type name, so a given type keeps its color across renders.
_GRAPH_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
    "#1f77b4",
    "#d62728",
]

_BGCOLOR = "#1a1a1a"

# Vendored D3 v7 bundle, inlined into the page so the output is fully offline/self-contained.
_D3_PATH = Path(__file__).parent / "vendor" / "d3.v7.min.js"


@lru_cache(maxsize=1)
def _d3_source() -> str:
    """Read the vendored D3 v7 minified bundle once (cached across requests)."""
    return _D3_PATH.read_text(encoding="utf-8")


def _node_entity_type(node) -> str:
    """Best-effort entity type for coloring: properties.entity_type, then first label, else 'unknown'."""
    et = (node.properties or {}).get("entity_type")
    if et:
        return str(et)
    if node.labels:
        return str(node.labels[0])
    return "unknown"


# Show the most useful fields first; any remaining (non-empty) fields follow in insertion order.
_TOOLTIP_KEY_ORDER = [
    "entity_type",
    "description",
    "keywords",
    "weight",
    "file_path",
    "source_id",
    "created_at",
]


def _format_tooltip_value(key: str, value) -> str:
    """Stringify a property value for display; render epoch timestamps as readable UTC datetimes."""
    if key.endswith("_at") and isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = str(value)
    if len(text) > 800:
        text = text[:800] + "…"
    return text


def _props_tooltip(props: dict) -> str:
    """Render node/edge properties as well-formatted multi-line plain text (one 'Key: value' per
    line). Empty values are dropped (removes LightRAG's empty `truncate` artifact); keys are
    title-cased; known fields are ordered first. The page styles the tooltip <div> with
    `white-space: pre-wrap` so these newlines render."""
    props = props or {}
    ordered_keys = [k for k in _TOOLTIP_KEY_ORDER if k in props]
    ordered_keys += [k for k in props if k not in _TOOLTIP_KEY_ORDER]
    lines = []
    for k in ordered_keys:
        v = props[k]
        if v is None or str(v).strip() == "":
            continue
        label = k.replace("_", " ").title()
        lines.append(f"{label}: {_format_tooltip_value(k, v)}")
    return "\n".join(lines)


# Self-contained page: inlined D3, an embedded {nodes, links} blob, and a canvas force-graph
# script. Placeholders (__D3_SOURCE__ / __DATA_JSON__ / __PHYSICS__ / __BGCOLOR__) are filled by
# str.replace so the JS braces need no escaping.
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge Graph</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: __BGCOLOR__; }
  #graph { display: block; position: fixed; top: 0; left: 0; cursor: grab; }
  #graph:active { cursor: grabbing; }
  #tooltip {
    position: fixed;
    display: none;
    pointer-events: none;
    z-index: 10;
    white-space: pre-wrap;
    max-width: 380px;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 12px;
    line-height: 1.45;
    padding: 8px 11px;
    border-radius: 6px;
    background-color: #2b2b2b;
    color: #eaeaea;
    border: 1px solid #555;
    box-shadow: 0 2px 10px rgba(0,0,0,0.45);
  }
  #hint {
    position: fixed; bottom: 10px; left: 12px; z-index: 10;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 11px; color: #888; pointer-events: none;
  }
</style>
</head>
<body>
<canvas id="graph"></canvas>
<div id="tooltip"></div>
<div id="hint">scroll to zoom · drag background to pan · drag a node to move it · hover for details</div>
<script>__D3_SOURCE__</script>
<script>
const DATA = __DATA_JSON__;
const PHYSICS = __PHYSICS__;
const nodes = DATA.nodes;
const links = DATA.links;

const canvas = document.getElementById("graph");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

let dpr = window.devicePixelRatio || 1;
let width = window.innerWidth;
let height = window.innerHeight;
let transform = d3.zoomIdentity;

function radius(d) { return 8 + 2 * (d.degree || 0); }
function showLabels() { return nodes.length <= 300 || transform.k > 1.3; }

const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(100))
  .force("charge", d3.forceManyBody().strength(-250))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(d => radius(d) + 4));

function resize() {
  dpr = window.devicePixelRatio || 1;
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = width + "px";
  canvas.style.height = height + "px";
  simulation.force("center", d3.forceCenter(width / 2, height / 2));
  draw();
}
window.addEventListener("resize", resize);

function draw() {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.scale(dpr, dpr);
  ctx.translate(transform.x, transform.y);
  ctx.scale(transform.k, transform.k);

  // Edges (directed): line + arrowhead at the target boundary.
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(180,180,180,0.35)";
  ctx.fillStyle = "rgba(180,180,180,0.55)";
  for (const l of links) {
    const s = l.source, t = l.target;
    if (s.x == null || t.x == null) continue;
    const dx = t.x - s.x, dy = t.y - s.y;
    const dist = Math.hypot(dx, dy) || 1;
    const ux = dx / dist, uy = dy / dist;
    const tr = radius(t);
    const ex = t.x - ux * tr, ey = t.y - uy * tr;  // stop at the target circle's edge
    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    // Arrowhead
    const a = 5, spread = 0.5;
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex - a * Math.cos(Math.atan2(uy, ux) - spread), ey - a * Math.sin(Math.atan2(uy, ux) - spread));
    ctx.lineTo(ex - a * Math.cos(Math.atan2(uy, ux) + spread), ey - a * Math.sin(Math.atan2(uy, ux) + spread));
    ctx.closePath();
    ctx.fill();
  }

  // Nodes
  for (const n of nodes) {
    if (n.x == null) continue;
    const r = radius(n);
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = n.color;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(0,0,0,0.4)";
    ctx.stroke();
  }

  // Labels (skipped on large graphs when zoomed out, to keep repaints cheap)
  if (showLabels()) {
    ctx.fillStyle = "#eaeaea";
    ctx.font = "11px -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
    ctx.textBaseline = "middle";
    for (const n of nodes) {
      if (n.x == null) continue;
      ctx.fillText(n.label, n.x + radius(n) + 3, n.y);
    }
  }
  ctx.restore();
}

// --- Hit-testing (pointer -> graph coordinates via the inverse zoom transform) ---
function nodeAt(mx, my) {
  const [px, py] = transform.invert([mx, my]);
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n = nodes[i];
    if (n.x == null) continue;
    const dx = px - n.x, dy = py - n.y, r = radius(n);
    if (dx * dx + dy * dy <= r * r) return n;
  }
  return null;
}
function distToSegment(px, py, a, b) {
  const vx = b.x - a.x, vy = b.y - a.y;
  const wx = px - a.x, wy = py - a.y;
  const len2 = vx * vx + vy * vy;
  let t = len2 ? (wx * vx + wy * vy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  const cx = a.x + t * vx, cy = a.y + t * vy;
  return Math.hypot(px - cx, py - cy);
}
function linkAt(mx, my) {
  const [px, py] = transform.invert([mx, my]);
  const tol = 4 / transform.k;
  for (const l of links) {
    if (l.source.x == null || l.target.x == null) continue;
    if (distToSegment(px, py, l.source, l.target) <= tol) return l;
  }
  return null;
}

canvas.addEventListener("mousemove", (event) => {
  const rect = canvas.getBoundingClientRect();
  const mx = event.clientX - rect.left, my = event.clientY - rect.top;
  const hit = nodeAt(mx, my) || linkAt(mx, my);
  if (hit && hit.tooltip) {
    tooltip.textContent = hit.tooltip;
    tooltip.style.display = "block";
    tooltip.style.left = (event.clientX + 12) + "px";
    tooltip.style.top = (event.clientY + 12) + "px";
  } else {
    tooltip.style.display = "none";
  }
});
canvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });

// --- Zoom / pan (background) ---
d3.select(canvas).call(
  d3.zoom()
    .scaleExtent([0.1, 5])
    .filter((event) => {
      // Wheel/dblclick always zoom; a mousedown on a node is left to the drag handler.
      if (event.type === "wheel" || event.type === "dblclick") return true;
      const rect = canvas.getBoundingClientRect();
      return !nodeAt(event.clientX - rect.left, event.clientY - rect.top);
    })
    .on("zoom", (event) => { transform = event.transform; draw(); })
);

// --- Node dragging ---
d3.select(canvas).call(
  d3.drag()
    .subject((event) => nodeAt(event.x, event.y))
    .on("start", (event) => {
      if (!event.subject) return;
      if (PHYSICS) simulation.alphaTarget(0.3).restart();
      const [gx, gy] = transform.invert([event.x, event.y]);
      event.subject.fx = gx;
      event.subject.fy = gy;
    })
    .on("drag", (event) => {
      if (!event.subject) return;
      const [gx, gy] = transform.invert([event.x, event.y]);
      event.subject.fx = gx;
      event.subject.fy = gy;
      if (!PHYSICS) draw();
    })
    .on("end", (event) => {
      if (!event.subject) return;
      if (PHYSICS) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    })
);

// --- Run the layout ---
if (PHYSICS) {
  simulation.on("tick", draw);
} else {
  simulation.stop();
  for (let i = 0; i < 300; i++) simulation.tick();
  draw();
}
resize();
</script>
</body>
</html>
"""


def _build_graph_html(kg, physics: bool) -> str:
    """Render a KnowledgeGraph (LightRAG) into a self-contained interactive HTML page using D3.js
    v7 force layout drawn on an HTML <canvas>.

    Nodes are colored by entity type and sized by their connection degree; hovering a node or edge
    reveals its full properties via an overlay tooltip. The vendored `d3.v7.min.js` is inlined, so
    the returned HTML is a single self-contained, offline-capable document. `physics=False` settles
    the layout synchronously and renders a static graph; `physics=True` animates it live."""
    # Degree from the edge list (undirected count — both endpoints).
    degree: dict[str, int] = {}
    for e in kg.edges:
        degree[e.source] = degree.get(e.source, 0) + 1
        degree[e.target] = degree.get(e.target, 0) + 1

    # Deterministic type → color mapping.
    types = sorted({_node_entity_type(n) for n in kg.nodes})
    color_of = {t: _GRAPH_PALETTE[i % len(_GRAPH_PALETTE)] for i, t in enumerate(types)}

    nodes = []
    for n in kg.nodes:
        et = _node_entity_type(n)
        props = n.properties or {}
        nodes.append(
            {
                "id": n.id,
                "label": str(props.get("entity_id") or n.id),
                "color": color_of[et],
                "degree": degree.get(n.id, 0),
                "tooltip": _props_tooltip({"entity_type": et, **props}),
            }
        )

    node_ids = {n.id for n in kg.nodes}
    links = []
    for e in kg.edges:
        # Guard against edges referencing nodes trimmed by max_nodes truncation.
        if e.source in node_ids and e.target in node_ids:
            links.append(
                {
                    "source": e.source,
                    "target": e.target,
                    "tooltip": _props_tooltip(e.properties or {}),
                }
            )

    # Script-safe embedding: neutralize any "</..." that could close the <script> element early.
    data_json = json.dumps({"nodes": nodes, "links": links}).replace("</", "<\\/")

    # Fill the small placeholders first, then insert the large D3/data blobs, so a later
    # replace can never match a token that happens to occur inside embedded content.
    return (
        _HTML_TEMPLATE.replace("__PHYSICS__", "true" if physics else "false")
        .replace("__BGCOLOR__", _BGCOLOR)
        .replace("__DATA_JSON__", data_json)
        .replace("__D3_SOURCE__", _d3_source())
    )
