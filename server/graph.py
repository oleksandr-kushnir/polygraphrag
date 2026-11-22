"""Knowledge-graph visualization: render a LightRAG KnowledgeGraph into a self-contained
interactive HTML page (D3.js v7, force layout drawn on an HTML <canvas>).

The graph.html endpoint calls `_build_graph_html`; the rest are internal helpers for node
coloring, property tooltips, and inlining the vendored D3 source.
"""

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from server.references import _graph_field_sep

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


def _format_tooltip_value(key: str, value) -> str:
    """Format timestamps and cap long text for display."""
    if key.endswith("_at") and isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = str(value).strip()
    if len(text) > 800:
        text = text[:800] + "…"
    return text


def _props_tooltip(props: dict, sep: str) -> dict:
    """Project graph properties into readable summary, evidence, and technical details."""
    sources = [source for source in props.get("sources") or [] if isinstance(source, dict)]
    evidence = (props.get("file_path") or "").split(sep)
    for source in sources:
        evidence.extend(source.get(field) or "" for field in ("file_path", "source_ref"))
    evidence = list(dict.fromkeys(value.strip() for value in evidence if value.strip()))
    info: dict = {}
    description = _format_tooltip_value(
        "description", str(props.get("description") or "").replace(sep, "\n\n")
    )
    if description:
        info["description"] = description
    if evidence:
        info["evidence"] = evidence

    created = props.get("created_at")
    metadata = {
        "Origin": ", ".join(origin.capitalize() for origin in props.get("origins") or []),
        "Created": _format_tooltip_value("created_at", created) if created is not None else None,
        "Revision": props.get("revision"),
    }
    metadata = {label: value for label, value in metadata.items() if value not in (None, "")}
    if metadata:
        info["metadata"] = metadata

    technical = {}
    for key, label in (
        ("id", "Record ID"),
        ("requested_id", "Requested ID"),
        ("keywords", "Keywords"),
        ("weight", "Weight"),
        ("source_id", "Source IDs"),
    ):
        if props.get(key) is not None:
            value = _format_tooltip_value(key, props[key]).replace(sep, "\n")
            if value:
                technical[label] = value
    for key, label in (("document_id", "Document IDs"), ("chunk_id", "Chunk IDs")):
        ids = dict.fromkeys(source[key] for source in sources if source.get(key))
        if ids:
            technical[label] = _format_tooltip_value(key, "\n".join(ids))
    if technical:
        info["technical"] = technical
    return info


# Standalone exports and the MCP App use the same D3 renderer.
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
    z-index: 20;
    box-sizing: border-box;
    width: min(360px, calc(100vw - 24px));
    max-height: calc(100vh - 24px);
    overflow: auto;
    overflow-wrap: anywhere;
    font: 13px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    padding: 15px 16px;
    border-radius: 10px;
    background: #23262b;
    color: #f0f2f5;
    border: 1px solid #454a53;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  }
  #tooltip:not(.pinned) { pointer-events: none; }
  #tooltip.pinned { top: 12px; right: 12px; }
  #tooltip header { position: relative; }
  #tooltip.pinned header { padding-right: 36px; }
  #tooltip h2, #tooltip h3, #tooltip p, #tooltip dl, #tooltip dd { margin: 0; }
  #tooltip h2 { font-size: 17px; line-height: 1.3; }
  #tooltip .info-category {
    color: #b8c4d4;
    font-size: 11px;
    font-weight: 650;
    letter-spacing: .055em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  #tooltip .info-close {
    position: absolute; top: 0; right: 0;
    width: 28px; height: 28px;
    padding: 0; border: 1px solid #505965; border-radius: 6px;
    background: transparent; color: inherit; font-size: 18px; cursor: pointer;
  }
  #tooltip .info-close:hover { background: #363d48; }
  #tooltip button:focus-visible, #tooltip summary:focus-visible {
    outline: 2px solid #a9caff; outline-offset: 2px;
  }
  #tooltip .info-endpoints {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
    align-items: center;
    gap: 9px;
    padding: 8px 10px;
    background: #2e3540;
    border-radius: 6px;
    font-weight: 600;
  }
  #tooltip .info-endpoints span:nth-child(2) { color: #bbc9db; }
  #tooltip .info-endpoints span:last-child { text-align: right; }
  #tooltip .info-endpoints, #tooltip .info-description, #tooltip section { margin-top: 12px; }
  #tooltip .info-description { white-space: pre-line; }
  #tooltip:not(.pinned) .info-description {
    display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 4; overflow: hidden;
  }
  #tooltip h3 { color: #c5cfdb; font-size: 12px; font-weight: 650; }
  #tooltip ul { margin: 4px 0 0; padding-left: 17px; }
  #tooltip li + li { margin-top: 3px; }
  #tooltip dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 4px 12px; }
  #tooltip dt { color: #bbc5d3; }
  #tooltip .info-metadata, #tooltip details {
    margin-top: 11px;
    border-top: 1px solid #454a53;
    padding-top: 9px;
    font-size: 12px;
  }
  #tooltip .info-technical { margin-top: 7px; }
  #tooltip .info-technical dd { white-space: pre-wrap; font: 11px/1.5 ui-monospace, monospace; }
  #tooltip summary { color: #c5cfdb; cursor: pointer; }
  #tooltip .info-more { margin-top: 4px; color: #b8c4d4; font-size: 12px; }
  #hint {
    position: fixed; bottom: 10px; left: 12px; z-index: 10;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 11px; color: #888; pointer-events: none;
  }
  #legend {
    position: fixed; bottom: 12px; right: 12px; z-index: 10;
    max-height: 45vh; overflow-y: auto;
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 12px; color: #eaeaea;
    padding: 9px 12px;
    border-radius: 6px;
    background-color: rgba(43,43,43,0.9);
    border: 1px solid #555;
    box-shadow: 0 2px 10px rgba(0,0,0,0.45);
  }
  #legend .legend-title {
    font-size: 11px; color: #999; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 6px;
  }
  #legend .legend-item {
    display: flex; align-items: center; gap: 7px; line-height: 1.7;
  }
  #legend .legend-swatch {
    width: 11px; height: 11px; border-radius: 50%;
    flex: 0 0 auto; border: 1px solid rgba(0,0,0,0.4);
  }
__EXTRA_STYLE__
</style>
</head>
<body>
<canvas id="graph"></canvas>
<aside id="tooltip" aria-label="Graph details" hidden></aside>
<div id="hint">Scroll to zoom · Drag to pan or move a node · Hover for details · Click to pin</div>
<div id="legend"></div>
__STATUS__
<script>__D3_SOURCE__</script>
<script>
function renderGraph(DATA, PHYSICS, LEGEND) {
  const nodes = DATA.nodes;
  const links = DATA.links;

  const canvas = document.getElementById("graph");
  const ctx = canvas.getContext("2d");
  const tooltip = document.getElementById("tooltip");
  const legend = document.getElementById("legend");
  const events = new AbortController();
  const listening = { signal: events.signal };

  // --- Legend (entity type -> node color) ---
  (function buildLegend() {
    const el = legend;
    el.replaceChildren();
    el.style.display = "";
    if (!LEGEND || !LEGEND.length) { el.style.display = "none"; return; }
    const title = document.createElement("div");
    title.className = "legend-title";
    title.textContent = "Entity type";
    el.appendChild(title);
    for (const entry of LEGEND) {
      const row = document.createElement("div");
      row.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.backgroundColor = entry.color;
      const label = document.createElement("span");
      label.textContent = entry.label;
      row.appendChild(swatch);
      row.appendChild(label);
      el.appendChild(row);
    }
  })();

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
  window.addEventListener("resize", resize, listening);

  function draw() {
    if (events.signal.aborted) return;
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.scale(dpr, dpr);
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.k, transform.k);

    // Draw each edge and add a target arrowhead for directed relationships.
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
      if (l.direction !== "directed") continue;
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

  let inspected;
  let pinned = false;

  function itemAt(event) {
    return nodeAt(event.offsetX, event.offsetY) || linkAt(event.offsetX, event.offsetY);
  }

  function textElement(tag, value, className = "") {
    const element = document.createElement(tag);
    element.textContent = value;
    element.className = className;
    return element;
  }

  function appendFields(list, fields) {
    for (const [label, value] of Object.entries(fields)) {
      list.append(textElement("dt", label), textElement("dd", value));
    }
  }

  function hideDetails() {
    inspected = undefined;
    pinned = false;
    tooltip.hidden = true;
    tooltip.replaceChildren();
  }

  function showDetails(item) {
    inspected = item;
    const info = item.tooltip || {};
    const isLink = Boolean(item.source);
    const directed = item.direction === "directed";
    const category = isLink ? (directed ? "Directed relationship" : "Undirected relationship")
      : (info.type && info.type !== "unknown" ? info.type : "Entity");
    const title = isLink ? (info.type || "Relationship") : item.label;
    const header = document.createElement("header");
    header.append(textElement("p", category, "info-category"), textElement("h2", title));
    tooltip.replaceChildren(header);
    tooltip.hidden = false;
    tooltip.classList.toggle("pinned", pinned);
    tooltip.scrollTop = 0;

    if (isLink) {
      const endpoints = document.createElement("p");
      endpoints.className = "info-endpoints";
      const connector = textElement("span", directed ? "→" : "—");
      connector.setAttribute("aria-label", directed ? "to" : "connected to");
      endpoints.append(textElement("span", item.source.label), connector,
        textElement("span", item.target.label));
      tooltip.append(endpoints);
    }
    if (info.description) tooltip.append(textElement("p", info.description, "info-description"));
    if (info.evidence?.length) {
      const section = document.createElement("section");
      const list = document.createElement("ul");
      const visible = pinned ? info.evidence : info.evidence.slice(0, 3);
      for (const source of visible) list.append(textElement("li", source));
      section.append(textElement("h3", "Evidence"), list);
      const remaining = info.evidence.length - visible.length;
      if (remaining) section.append(textElement("p", `+ ${remaining} more`, "info-more"));
      tooltip.append(section);
    }
    if (info.metadata) {
      const list = document.createElement("dl");
      list.className = "info-metadata";
      appendFields(list, info.metadata);
      tooltip.append(list);
    }
    if (pinned) {
      tooltip.style.removeProperty("left");
      tooltip.style.removeProperty("top");
      const details = document.createElement("details");
      const list = document.createElement("dl");
      list.className = "info-technical";
      const fields = { ...info.technical };
      if (isLink) {
        fields["Endpoint ID 1"] = item.source.id;
        fields["Endpoint ID 2"] = item.target.id;
      } else {
        fields["Record ID"] ??= item.id;
      }
      appendFields(list, fields);
      details.append(textElement("summary", "Technical details"), list);
      tooltip.append(details);
      const close = textElement("button", "×", "info-close");
      close.type = "button";
      close.setAttribute("aria-label", "Close graph details");
      close.addEventListener("click", hideDetails);
      header.append(close);
      close.focus({ preventScroll: true });
    }
  }

  canvas.addEventListener("mousemove", (event) => {
    if (pinned) return;
    const item = itemAt(event);
    if (!item) {
      if (inspected) hideDetails();
      return;
    }
    if (item !== inspected) showDetails(item);
    const gap = 12;
    const { offsetWidth: cardWidth, offsetHeight: cardHeight } = tooltip;
    const x = event.clientX + gap + cardWidth <= width - gap ? event.clientX + gap
      : Math.max(gap, event.clientX - gap - cardWidth);
    tooltip.style.left = x + "px";
    tooltip.style.top = Math.max(gap, Math.min(event.clientY + gap, height - gap - cardHeight)) + "px";
  }, listening);
  canvas.addEventListener("mouseleave", () => { if (!pinned) hideDetails(); }, listening);
  canvas.addEventListener("click", (event) => {
    const item = itemAt(event);
    if (!item) {
      hideDetails();
      return;
    }
    pinned = true;
    showDetails(item);
  }, listening);
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideDetails();
  }, listening);

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
      .subject((event) => {
        const node = nodeAt(event.x, event.y);
        if (!node) return null;
        const [x, y] = transform.apply([node.x, node.y]);
        return {node, x, y};
      })
      .on("start", (event) => {
        const {node} = event.subject;
        node.fx = node.x;
        node.fy = node.y;
        if (PHYSICS) simulation.alphaTarget(0.3).restart();
      })
      .on("drag", (event) => {
        const {node} = event.subject;
        const [gx, gy] = transform.invert([event.x, event.y]);
        node.fx = gx;
        node.fy = gy;
        if (!PHYSICS) {
          node.x = gx;
          node.y = gy;
          draw();
        }
      })
      .on("end", (event) => {
        if (PHYSICS) simulation.alphaTarget(0);
        event.subject.node.fx = null;
        event.subject.node.fy = null;
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

  return () => {
    events.abort();
    simulation.stop();
    d3.select(canvas).on(".zoom", null).on(".drag", null).property("__zoom", null);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    hideDetails();
    legend.replaceChildren();
    legend.style.display = "none";
  };
}
__INITIALIZE__
</script>
</body>
</html>
"""


def graph_viewer_payload(kg, physics: bool) -> dict:
    """Project the selected graph into the shared canvas renderer's input."""
    sep = _graph_field_sep()
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
                "tooltip": {"type": et, **_props_tooltip(props, sep)},
            }
        )

    node_ids = {n.id for n in kg.nodes}
    links = []
    for e in kg.edges:
        # Guard against edges referencing nodes trimmed by max_nodes truncation.
        if e.source in node_ids and e.target in node_ids:
            props = e.properties or {}
            links.append(
                {
                    "source": e.source,
                    "target": e.target,
                    "direction": props.get("direction") or "undirected",
                    "tooltip": {"type": props.get("type") or "", **_props_tooltip(props, sep)},
                }
            )

    # Legend entries (entity type -> color), in the same sorted order as the color mapping.
    legend = [{"label": t, "color": color_of[t]} for t in types]

    return {"nodes": nodes, "links": links, "legend": legend, "physics": physics}


def _viewer_document(initializer: str, *, style: str = "", status: str = "") -> str:
    values = {
        "__BGCOLOR__": _BGCOLOR,
        "__D3_SOURCE__": _d3_source(),
        "__INITIALIZE__": initializer,
        "__EXTRA_STYLE__": style,
        "__STATUS__": status,
    }
    return re.sub(r"__[A-Z0-9_]+__", lambda m: values.get(m[0], m[0]), _HTML_TEMPLATE)


def _script_json(value) -> str:
    # Prevent closing the containing script element through graph data.
    return json.dumps(value).replace("</", "<\\/")


def render_graph_snapshot(payload: dict) -> str:
    data_json = _script_json({"nodes": payload["nodes"], "links": payload["links"]})
    physics_json = _script_json(payload["physics"])
    legend_json = _script_json(payload["legend"])
    return _viewer_document(
        f"const DATA = {data_json};\n"
        f"const PHYSICS = {physics_json};\n"
        f"const LEGEND = {legend_json};\n"
        "renderGraph(DATA, PHYSICS, LEGEND);"
    )


def _build_graph_html(kg, physics: bool) -> str:
    return render_graph_snapshot(graph_viewer_payload(kg, physics))


@lru_cache(maxsize=1)
def graph_app_html() -> str:
    bridge = Path(__file__).with_name("graph_app.js").read_text(encoding="utf-8")
    return _viewer_document(
        bridge,
        style="""
  #hint { display: none; }
  #status {
    position: fixed; inset: 12px; height: fit-content;
    color: #eaeaea; background: #1a1a1a;
    font: 13px system-ui, sans-serif; overflow-wrap: anywhere;
  }
  #status button { display: block; margin-top: 8px; }
""",
        status='<div id="status" role="status">Waiting for the workspace graph.</div>',
    )
