import React, { useEffect, useRef, useState } from "react";
import FeatureInsightWindow from "./FeatureInsightWindow";

export default function LiveExpandingGraph({
  incomingBatch,
  focusedAlert,
  evidenceMask,
  highlightedRefs = [],
  baselines = {},
}) {
  const [viewMode, setViewMode] = useState("live");
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [generation, setGeneration] = useState(0);
  const [selectedItem, setSelectedItem] = useState(null);
  const [zoom, setZoom] = useState(1.0);
  const [draggingId, setDraggingId] = useState(null);
  const svgRef = useRef(null);
  const lastBatchIdRef = useRef(null);

  useEffect(() => {
    if (!incomingBatch || incomingBatch.batch_id === lastBatchIdRef.current) return;
    lastBatchIdRef.current = incomingBatch.batch_id;
    const nextGen = generation + 1;
    setGeneration(nextGen);

    setNodes((prev) => {
      const map = new Map(prev.map((n) => [n.node_id, n]));
      incomingBatch.nodes.forEach((n, idx) => {
        if (!map.has(n.node_id)) {
          const parentEdge = incomingBatch.edges.find(
            (e) => (e.target_id === n.node_id && map.has(e.source_id)) ||
                   (e.source_id === n.node_id && map.has(e.target_id))
          );
          const parentId = parentEdge
            ? (map.has(parentEdge.source_id) ? parentEdge.source_id : parentEdge.target_id)
            : null;
          const parent = parentId ? map.get(parentId) : null;
          const angle = ((nextGen * 53 + idx * 137.5) * Math.PI) / 180;
          const radius = parent ? 76 : 125;
          const cx = parent ? parent.x : 390;
          const cy = parent ? parent.y : 260;

          map.set(n.node_id, {
            ...n,
            bornGen: nextGen,
            x: Math.max(48, Math.min(732, cx + Math.cos(angle) * radius)),
            y: Math.max(48, Math.min(472, cy + Math.sin(angle) * radius)),
          });
        } else {
          const old = map.get(n.node_id);
          map.set(n.node_id, { ...old, ...n, bornGen: nextGen, x: old.x, y: old.y });
        }
      });
      return Array.from(map.values()).filter((n) => nextGen - n.bornGen <= 7);
    });

    setEdges((prev) => {
      const map = new Map(prev.map((e) => [e.edge_id, e]));
      incomingBatch.edges.forEach((e) => map.set(e.edge_id, { ...e, bornGen: nextGen }));
      return Array.from(map.values()).filter((e) => nextGen - e.bornGen <= 7);
    });
  }, [incomingBatch]);

  const displayNodes = React.useMemo(() => {
    if (viewMode === "alert" && focusedAlert) {
      const sub = focusedAlert.subgraph_nodes || [];
      return sub.map((n, idx) => {
        const ring = n.node_type === "transaction" ? 95 : 185;
        const angle = (2 * Math.PI * idx) / Math.max(1, sub.length);
        return { ...n, bornGen: generation, x: 390 + Math.cos(angle) * ring, y: 260 + Math.sin(angle) * ring };
      });
    }
    return nodes;
  }, [viewMode, focusedAlert, nodes, generation]);

  const displayEdges = viewMode === "alert" && focusedAlert ? (focusedAlert.subgraph_edges || []) : edges;
  const minNodes = new Set(evidenceMask?.minimal_nodes || []);
  const minEdges = new Set(evidenceMask?.minimal_edges || []);
  const claimRefs = new Set(highlightedRefs || []);
  const lookup = Object.fromEntries(displayNodes.map((n) => [n.node_id, n]));

  const handleSvgMouseMove = (e) => {
    if (!draggingId || viewMode !== "live" || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 780;
    const y = ((e.clientY - rect.top) / rect.height) * 520;
    setNodes((prev) => prev.map((n) => (n.node_id === draggingId ? { ...n, x, y } : n)));
  };

  return (
    <div style={{ position: "relative", background: "#11141c", border: "1px solid #1e232e", borderRadius: 12, padding: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10, gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <button
            onClick={() => setViewMode("live")}
            style={{
              background: viewMode === "live" ? "#1e293b" : "transparent",
              color: viewMode === "live" ? "#f8fafc" : "#64748b",
              border: "1px solid #262b36", borderRadius: 6, padding: "4px 10px", fontSize: 11.5, fontWeight: 500, cursor: "pointer",
            }}
          >
            Live Interlaced Stream
          </button>
          <button
            disabled={!focusedAlert}
            onClick={() => setViewMode("alert")}
            style={{
              background: viewMode === "alert" ? "#1e293b" : "transparent",
              color: !focusedAlert ? "#334155" : viewMode === "alert" ? "#f8fafc" : "#64748b",
              border: "1px solid #262b36", borderRadius: 6, padding: "4px 10px", fontSize: 11.5, fontWeight: 500,
              cursor: focusedAlert ? "pointer" : "not-allowed",
            }}
          >
            Focused Alert Topology {focusedAlert ? `(${focusedAlert.alert_id})` : ""}
          </button>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 11, color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace" }}>
          <span>GEN #{generation}</span>
          <div style={{ display: "flex", gap: 4 }}>
            <button onClick={() => setZoom((z) => Math.min(1.6, +(z + 0.15).toFixed(2)))} style={{ background: "#181c26", color: "#cbd5e1", border: "1px solid #262b36", borderRadius: 5, width: 24, height: 22, cursor: "pointer" }}>+</button>
            <button onClick={() => setZoom((z) => Math.max(0.7, +(z - 0.15).toFixed(2)))} style={{ background: "#181c26", color: "#cbd5e1", border: "1px solid #262b36", borderRadius: 5, width: 24, height: 22, cursor: "pointer" }}>−</button>
            <button onClick={() => setZoom(1.0)} style={{ background: "#181c26", color: "#cbd5e1", border: "1px solid #262b36", borderRadius: 5, padding: "0 6px", height: 22, fontSize: 10, cursor: "pointer" }}>1:1</button>
          </div>
        </div>
      </div>

      <svg
        ref={svgRef}
        viewBox="0 0 780 520"
        onMouseMove={handleSvgMouseMove}
        onMouseUp={() => setDraggingId(null)}
        onMouseLeave={() => setDraggingId(null)}
        style={{ width: "100%", height: 500, background: "#0a0c10", borderRadius: 8, border: "1px solid #181c26", userSelect: "none" }}
      >
        <defs>
          <radialGradient id="flagGlow">
            <stop offset="0%" stopColor="rgba(244,63,94,0.45)" />
            <stop offset="100%" stopColor="rgba(244,63,94,0)" />
          </radialGradient>
        </defs>

        <g transform={`translate(${390 * (1 - zoom)}, ${260 * (1 - zoom)}) scale(${zoom})`}>
          {displayEdges.map((e) => {
            const src = lookup[e.source_id];
            const dst = lookup[e.target_id];
            if (!src || !dst) return null;
            const age = viewMode === "alert" ? 0 : Math.max(0, generation - (e.bornGen ?? generation));
            const isClaim = claimRefs.has(e.edge_id);
            const isFlag = minEdges.has(e.edge_id);
            const isSel = selectedItem?.kind === "edge" && selectedItem.data.edge_id === e.edge_id;
            const opacity = isFlag || isClaim || isSel ? 0.96 : Math.max(0.14, 0.85 - age * 0.12);
            return (
              <g key={e.edge_id} style={{ cursor: "pointer" }} onClick={() => setSelectedItem({ kind: "edge", data: e })}>
                <line x1={src.x} y1={src.y} x2={dst.x} y2={dst.y} stroke="transparent" strokeWidth={12} />
                <line
                  x1={src.x} y1={src.y} x2={dst.x} y2={dst.y}
                  stroke={isSel ? "#ffffff" : isClaim ? "#fbbf24" : isFlag ? "#f43f5e" : "#334155"}
                  strokeWidth={isSel || isClaim ? 3.2 : isFlag ? 2.2 : 1.15}
                  strokeOpacity={opacity}
                />
              </g>
            );
          })}

          {displayNodes.map((n) => {
            const age = viewMode === "alert" ? 0 : Math.max(0, generation - (n.bornGen ?? generation));
            const isClaim = claimRefs.has(n.node_id);
            const isFlag = minNodes.has(n.node_id);
            const isSel = selectedItem?.kind === "node" && selectedItem.data.node_id === n.node_id;
            const opacity = isFlag || isClaim || isSel ? 1.0 : Math.max(0.16, 0.94 - age * 0.12);
            const fill = isClaim ? "#fbbf24" : isFlag ? "#f43f5e" : n.node_type === "transaction" ? "#38bdf8" : "#a78bfa";

            return (
              <g
                key={n.node_id}
                transform={`translate(${n.x}, ${n.y})`}
                style={{ cursor: "pointer", opacity, transition: "opacity 0.45s ease" }}
                onMouseDown={() => viewMode === "live" && setDraggingId(n.node_id)}
                onClick={(ev) => { ev.stopPropagation(); setSelectedItem({ kind: "node", data: n }); }}
              >
                {(isFlag || isClaim || isSel) && <circle r={18} fill="url(#flagGlow)" />}
                <circle r={isSel ? 8.5 : isFlag || isClaim ? 7 : 5} fill={fill} stroke={isSel ? "#ffffff" : "#e2e8f0"} strokeWidth={isSel ? 2.2 : 1.0} />
                <text x={10} y={3.5} fill="#cbd5e1" fontSize="8.8" fontFamily="'JetBrains Mono', monospace">
                  {n.node_id.length > 18 ? n.node_id.slice(0, 17) + "…" : n.node_id}
                </text>
              </g>
            );
          })}
        </g>
      </svg>

      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10.5, color: "#64748b", marginTop: 8 }}>
        <span>Click any node or edge to open its timestamped feature insight profile. Drag nodes in Live mode.</span>
        <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>
          <span style={{ color: "#f43f5e" }}>● Explainer</span> &nbsp;
          <span style={{ color: "#38bdf8" }}>● Tx</span> &nbsp;
          <span style={{ color: "#a78bfa" }}>● Wallet</span>
        </span>
      </div>

      <FeatureInsightWindow item={selectedItem} evidenceMask={evidenceMask} baselines={baselines} onClose={() => setSelectedItem(null)} />
    </div>
  );
}
