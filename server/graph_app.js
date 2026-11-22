(() => {
  const status = document.getElementById("status");
  const hint = document.getElementById("hint");
  const host = window.parent;
  let clearGraph;
  let capabilities;
  let dimensions;
  let requestId = 1;
  let openLinkId;

  function send(message) {
    host.postMessage({ jsonrpc: "2.0", ...message }, "*");
  }

  function notifySize(height) {
    if (dimensions?.height != null) return;
    send({
      method: "ui/notifications/size-changed",
      params: { height: Math.min(height, dimensions?.maxHeight ?? height) },
    });
  }

  function showResult(result) {
    clearGraph?.();
    clearGraph = undefined;
    openLinkId = undefined;
    status.hidden = false;
    hint.style.display = "none";
    const viewer = result.structuredContent;
    const graph = result._meta?.["polygraphrag/graph"];
    if (!result.isError && viewer?.representation === "inline" && graph?.nodes.length) {
      status.hidden = true;
      hint.style.display = "block";
      clearGraph = renderGraph(graph, graph.physics, graph.legend);
      notifySize(480);
      return;
    }

    const text = result.content?.find(item => item.type === "text")?.text;
    status.textContent = text || "No graph result was returned.";
    if (result.isError && text) {
      try { status.textContent = JSON.parse(text).message || text; } catch { /* Plain text error. */ }
    } else if (viewer?.representation === "inline" && graph?.nodes.length === 0) {
      status.textContent = "No nodes match these graph filters.";
    }
    if (viewer?.viewer_url && /^https?:\/\//i.test(viewer.viewer_url)) {
      const link = document.createElement("div");
      link.textContent = viewer.viewer_url;
      status.append(link);
      if (capabilities.openLinks) {
        const button = document.createElement("button");
        button.textContent = "Open graph in browser";
        button.onclick = () => {
          openLinkId = ++requestId;
          send({ id: openLinkId, method: "ui/open-link", params: { url: viewer.viewer_url } });
        };
        status.append(button);
      }
    }
    notifySize(status.offsetHeight + 24);
  }

  function receive(event) {
    if (event.source !== host || event.data?.jsonrpc !== "2.0") return;
    const message = event.data;
    if (message.id === 1 && !message.method) {
      if (!message.result) {
        status.textContent = "The host could not initialize the graph viewer.";
        return;
      }
      capabilities = message.result.hostCapabilities ?? {};
      dimensions = message.result.hostContext?.containerDimensions;
      send({ method: "ui/notifications/initialized" });
      notifySize(120);
      return;
    }
    if (!capabilities) return;
    if (message.method === "ui/notifications/tool-result") {
      showResult(message.params);
    } else if (message.method === "ui/notifications/tool-cancelled") {
      showResult({ content: [{ type: "text", text: "Graph request cancelled." }] });
    } else if (message.method === "ui/resource-teardown") {
      clearGraph?.();
      window.removeEventListener("message", receive);
      send({ id: message.id, result: {} });
    } else if (message.method === "ping") {
      send({ id: message.id, result: {} });
    } else if (message.id === openLinkId && message.error) {
      const notice = document.createElement("div");
      notice.textContent = "Copy the URL to open the graph in your browser.";
      status.append(notice);
      notifySize(status.offsetHeight + 24);
    }
  }

  window.addEventListener("message", receive);
  send({
    id: requestId,
    method: "ui/initialize",
    params: {
      protocolVersion: "2026-01-26",
      appInfo: { name: "PolyGraphRAG graph viewer", version: "0.1.0" },
      appCapabilities: { availableDisplayModes: ["inline"] },
    },
  });
})();
