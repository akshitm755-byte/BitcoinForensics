import React from "react";

const FEATURE_LABELS = {
  in_degree: { label: "In-Degree", unit: "edges", desc: "Inbound counterparties inside the active temporal window." },
  out_degree: { label: "Out-Degree", unit: "edges", desc: "Outbound splits; spikes sharply during fan-out and peeling hops." },
  total_in_val: { label: "Log Inbound Value", unit: "ln(1+BTC)", desc: "Log-normalized incoming value flowing into this node." },
  total_out_val: { label: "Log Outbound Value", unit: "ln(1+BTC)", desc: "Log-normalized outgoing value leaving this node." },
  time_delta_prev: { label: "Inter-Event Delta", unit: "min", desc: "Time since prior interaction; compressed deltas indicate scripted laundering." },
  fan_out_ratio: { label: "Fan-Out Ratio", unit: "ratio", desc: "out_degree / (in_degree + 1); primary structural driver for peeling chains." },
  hop2_neighbor_count: { label: "2-Hop Density", unit: "ln(nodes)", desc: "Local cluster density across 2-hop heterogeneous neighborhoods." },
};

function formatTs(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

/**
 * Timestamped Feature Insight Window:
 * Appears whenever an investigator clicks a node or an edge on the dynamic graph.
 */
export default function FeatureInsightWindow({ item, evidenceMask, baselines = {}, onClose }) {
  if (!item) return null;

  const isNode = item.kind === "node";
  const data = item.data;
  const importance = isNode
    ? evidenceMask?.node_importance_mask?.[data.node_id]
    : evidenceMask?.edge_importance_mask?.[data.edge_id];
  const isFlagged = isNode
    ? (evidenceMask?.minimal_nodes || []).includes(data.node_id)
    : (evidenceMask?.minimal_edges || []).includes(data.edge_id);

  return (
    <div
      style={{
        position: "absolute",
        bottom: 14,
        left: 14,
        width: 345,
        background: "rgba(15, 18, 25, 0.96)",
        backdropFilter: "blur(12px)",
        border: `1px solid ${isFlagged ? "rgba(244,63,94,0.55)" : "rgba(148,163,184,0.22)"}`,
        borderRadius: 12,
        padding: "14px 16px",
        boxShadow: "0 16px 40px rgba(0,0,0,0.65)",
        zIndex: 20,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
            <span
              style={{
                fontSize: 10,
                fontFamily: "'JetBrains Mono', monospace",
                padding: "2px 6px",
                borderRadius: 4,
                background: isFlagged ? "rgba(244,63,94,0.16)" : "rgba(56,189,248,0.14)",
                color: isFlagged ? "#fb7185" : "#38bdf8",
                fontWeight: 600,
              }}
            >
              {isNode ? data.node_type.toUpperCase() : data.edge_type.toUpperCase()}
            </span>
            {isFlagged && (
              <span style={{ fontSize: 10, color: "#fb7185", fontFamily: "'JetBrains Mono', monospace" }}>
                ● EXPLAINER CORE
              </span>
            )}
          </div>
          <div style={{ fontSize: 12.5, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: "#f8fafc", wordBreak: "break-all" }}>
            {isNode ? data.node_id : data.edge_id}
          </div>
          <div style={{ fontSize: 10.5, color: "#64748b", fontFamily: "'JetBrains Mono', monospace", marginTop: 2 }}>
            {formatTs(data.timestamp)}
          </div>
        </div>
        <button
          onClick={onClose}
          style={{
            background: "transparent", border: "none", color: "#94a3b8",
            fontSize: 16, cursor: "pointer", padding: "0 4px", lineHeight: 1,
          }}
        >
          ×
        </button>
      </div>

      {/* Key Metrics Strip */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 8,
          marginTop: 10,
          marginBottom: 10,
          padding: "8px 10px",
          background: "#0a0c10",
          borderRadius: 8,
          border: "1px solid #1e232e",
        }}
      >
        <div>
          <div style={{ fontSize: 9.5, color: "#64748b", textTransform: "uppercase" }}>Explainer Mask</div>
          <div style={{ fontSize: 13, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: isFlagged ? "#fb7185" : "#e2e8f0" }}>
            {importance !== undefined ? `${(importance * 100).toFixed(1)}%` : "Unmasked"}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 9.5, color: "#64748b", textTransform: "uppercase" }}>
            {isNode ? "In-Graph Memory ||m||" : "Value / Spacing"}
          </div>
          <div style={{ fontSize: 13, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: "#38bdf8" }}>
            {isNode
              ? (data.memory_norm ?? 0).toFixed(3)
              : `${data.value} BTC · ${data.temporal_spacing_sec || 0}s`}
          </div>
        </div>
      </div>

      {isNode ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 205, overflowY: "auto" }}>
          {Object.entries(data.features || {}).map(([k, v]) => {
            const meta = FEATURE_LABELS[k] || { label: k, unit: "", desc: "" };
            const base = baselines[k] ?? 1.0;
            const deltaPct = base > 0 ? ((v - base) / base) * 100 : 0;
            const isAnomalous = k === "time_delta_prev" ? v > 0 && v < 0.75 : deltaPct > 45;
            return (
              <div key={k} style={{ paddingBottom: 5, borderBottom: "1px solid #1a1f2b" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5 }}>
                  <span style={{ color: "#cbd5e1", fontWeight: 500 }}>{meta.label}</span>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", color: isAnomalous ? "#fb7185" : "#f8fafc" }}>
                    {Number(v).toFixed(2)} <span style={{ fontSize: 9.5, color: "#64748b" }}>{meta.unit}</span>
                  </span>
                </div>
                <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>{meta.desc}</div>
              </div>
            );
          })}
        </div>
      ) : (
        <div style={{ fontSize: 11.5, color: "#cbd5e1", lineHeight: 1.5 }}>
          <div><span style={{ color: "#64748b" }}>Source:</span> <code style={{ fontFamily: "'JetBrains Mono', monospace" }}>{data.source_id}</code></div>
          <div><span style={{ color: "#64748b" }}>Target:</span> <code style={{ fontFamily: "'JetBrains Mono', monospace" }}>{data.target_id}</code></div>
          <div style={{ marginTop: 6, fontSize: 10.5, color: "#94a3b8" }}>
            {data.temporal_spacing_sec > 0 && data.temporal_spacing_sec < 35
              ? "Compressed inter-hop spacing (<35s) strongly correlates with scripted laundering relays."
              : "Edge cadence sits within normal mempool confirmation bounds."}
          </div>
        </div>
      )}
    </div>
  );
}
