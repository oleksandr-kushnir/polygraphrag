"""Knowledge-graph visualization: render a LightRAG KnowledgeGraph into a self-contained
interactive HTML page (pyvis / vis-network).

The graph.html endpoint calls `build_graph_html`; the rest are internal helpers for node
coloring and property tooltips.
"""

from datetime import datetime, timezone

# Stable categorical palette (vis-network reads CSS color strings). Entity types are mapped to
# colors deterministically by sorted type name, so a given type keeps its color across renders.
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


def _node_entity_type(node) -> str:
    """Best-effort entity type for coloring: properties.entity_type, then first label, else 'unknown'."""
    et = (node.properties or {}).get("entity_type")
    if et:
        return str(et)
    if node.labels:
        return str(node.labels[0])
    return "unknown"


# Tooltip CSS — vis-network renders a string `title` as escaped plain text inside `.vis-tooltip`,
# so HTML tags would show literally. Instead we emit clean "Key: value" lines joined by "\n" and
# style the tooltip with `white-space: pre-wrap` so the newlines render. Injected into <head>.
_TOOLTIP_CSS = """
<style>
.vis-tooltip {
  white-space: pre-wrap !important;
  max-width: 380px;
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
  font-size: 12px !important;
  line-height: 1.45 !important;
  padding: 8px 11px !important;
  border-radius: 6px !important;
  background-color: #2b2b2b !important;
  color: #eaeaea !important;
  border: 1px solid #555 !important;
  box-shadow: 0 2px 10px rgba(0,0,0,0.45) !important;
}
</style>
"""

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
    title-cased; known fields are ordered first. Paired with `_TOOLTIP_CSS` for line wrapping."""
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


def _build_graph_html(kg, physics: bool) -> str:
    """Render a KnowledgeGraph (LightRAG) into a self-contained interactive HTML page via pyvis.

    Nodes are colored by entity type and sized by their connection degree; hovering a node or edge
    reveals its full properties. `cdn_resources="in_line"` inlines the vis-network JS/CSS so the
    returned HTML is a single self-contained, offline-capable document."""
    from pyvis.network import Network

    # Degree from the edge list (undirected count — both endpoints).
    degree: dict[str, int] = {}
    for e in kg.edges:
        degree[e.source] = degree.get(e.source, 0) + 1
        degree[e.target] = degree.get(e.target, 0) + 1

    # Deterministic type → color mapping.
    types = sorted({_node_entity_type(n) for n in kg.nodes})
    color_of = {t: _GRAPH_PALETTE[i % len(_GRAPH_PALETTE)] for i, t in enumerate(types)}

    net = Network(
        height="100vh",
        width="100%",
        directed=True,
        bgcolor="#1a1a1a",
        font_color="#eaeaea",
        cdn_resources="in_line",
    )
    net.toggle_physics(physics)

    for n in kg.nodes:
        et = _node_entity_type(n)
        label = str((n.properties or {}).get("entity_id") or n.id)
        net.add_node(
            n.id,
            label=label,
            title=_props_tooltip({"entity_type": et, **(n.properties or {})}),
            color=color_of[et],
            size=12 + 3 * degree.get(n.id, 0),
        )

    node_ids = {n.id for n in kg.nodes}
    for e in kg.edges:
        # Guard against edges referencing nodes trimmed by max_nodes truncation.
        if e.source in node_ids and e.target in node_ids:
            net.add_edge(e.source, e.target, title=_props_tooltip(e.properties or {}))

    html = net.generate_html()
    # Inject tooltip styling so the "\n"-separated property lines wrap and render legibly.
    return html.replace("</head>", _TOOLTIP_CSS + "</head>", 1)
