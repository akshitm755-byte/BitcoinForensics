import React, { useEffect, useRef, useState } from "react";

/**
 * Bureau-Grade Continuous Expanding Temporal Graph:
 * - New subgraph generations sprout radially outward from parent perimeter nodes.
 * - Older generations progressively dim and fade out cleanly rather than vanishing abruptly.
 * - GNNExplainer minimal nodes & edges pulse crimson; clicking a claim highlights its exact refs.
 */
export default function LiveExpandingGraph({ incomingBatch, evidenceMask, highlightedRefs = [] }) {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [generation, setGeneration] = useState(0);
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
          const angle = ((nextGen * 47 + idx * 137.5) * Math.PI) / 180;
          const radius = parent ? 68 : 110;
          const cx = parent ? parent.x : 380;
          const cy = parent ? parent.y : 250;

          map.set(n.node_id, {
            ...n,
            bornGen: nextGen,
            x: Math.max(40, Math.min(720, cx + Math.cos(angle) * radius)),
            y: Math.max(40, Math.min(460, cy + Math.sin(angle) * radius)),
          });
        } else {
          // Refresh generation slightly if node is actively participating again
          map.set(n.node_id, { ...map.get(n.node_id), bornGen: nextGen });
        }
      });
      return Array.from(map.values()).filter((n) => nextGen - n.bornGen <= 6);
    });

    setEdges((prev) => {
      const map = new Map(prev.map((e) => [e.edge_id, e]));
      incomingBatch.edges.forEach((e) => {
        map.set(e.edge_id, { ...e, bornGen: nextGen });
      });
      return Array.from(map.values()).filter((e) => nextGen - e.bornGen <= 6);
    });
  }, [incomingBatch]);

  const minNodes = new Set(evidenceMask?.minimal_nodes || []);
  const minEdges = new Set(evidenceMask?.minimal_edges || []);
  const claimRefs = new Set(highlightedRefs || []);
  const lookup = Object.fromEntries(nodes.map((n) => [n.node_id, n]));

  return (
    <div style={{ background: "#090f1d", border: "1px solid #1e293b", borderRadius: 10, padding: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94a3b8", fontFamily: "monospace", marginBottom: 8 }}>
        <span>TEMPORAL HETEROGENEOUS GRAPH // GEN #{generation} // ACTIVE NODES: {nodes.length}</span>
        <span>
          <span style={{ color: "#ef4444" }}>● GNNExplainer Minimal</span> &nbsp;
          <span style={{ color: "#38bdf8" }}>● Transaction</span> &nbsp;
          <span style={{ color: "#a855f7" }}>● Wallet Address</span>
        </span>
      </div>
      <svg viewBox="0 0 760 500" style={{ width: "100%", height: 470, background: "#050811", borderRadius: 6 }}>
        <defs>
          <radialGradient id="crimsonHalo">
            <stop offset="0%" stopColor="rgba(239,68,68,0.55)" />
            <stop offset="100%" stopColor="rgba(239,68,68,0)" />
          </radialGradient>
        </defs>
        {edges.map((e) => {
          const src = lookup[e.source_id];
          const dst = lookup[e.target_id];
          if (!src || !dst) return null;
          const age = Math.max(0, generation - e.bornGen);
          const isClaimRef = claimRefs.has(e.edge_id);
          const isFlagged = minEdges.has(e.edge_id);
          const opacity = isFlagged || isClaimRef ? 0.95 : Math.max(0.12, 0.85 - age * 0.14);
          return (
            <line
              key={e.edge_id}
              x1={src.x}
              y1={src.y}
              x2={dst.x}
              y2={dst.y}
              stroke={isClaimRef ? "#facc15" : isFlagged ? "#ef4444" : "#334155"}
              strokeWidth={isClaimRef ? 3.4 : isFlagged ? 2.4 : 1.1}
              strokeOpacity={opacity}
            />
          );
        })}
        {nodes.map((n) => {
          const age = Math.max(0, generation - n.bornGen);
          const isClaimRef = claimRefs.has(n.node_id);
          const isFlagged = minNodes.has(n.node_id);
          const opacity = isFlagged || isClaimRef ? 1.0 : Math.max(0.14, 0.92 - age * 0.14);
          const fill = isClaimRef
            ? "#facc15"
            : isFlagged
            ? "#ef4444"
            : n.node_type === "transaction"
            ? "#38bdf8"
            : "#a855f7";
          return (
            <g key={n.node_id} transform={`translate(${n.x}, ${n.y})`} style={{ opacity, transition: "opacity 0.5s ease" }}>
              {(isFlagged || isClaimRef) && <circle r={18} fill="url(#crimsonHalo)" />}
              <circle r={isFlagged || isClaimRef ? 7.5 : 5} fill={fill} stroke="#f8fafc" strokeWidth={1.1} />
              <text x={10} y={3} fill="#cbd5e1" fontSize="9" fontFamily="monospace">
                {n.node_id.length > 18 ? n.node_id.slice(0, 17) + "…" : n.node_id}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
