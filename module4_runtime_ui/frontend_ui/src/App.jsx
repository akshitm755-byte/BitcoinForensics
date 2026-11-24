import React, { useEffect, useState } from "react";
import LiveExpandingGraph from "./components/LiveExpandingGraph";
import DossierViewer from "./components/DossierViewer";

export default function App() {
  const [state, setState] = useState(null);
  const [selectedAlertId, setSelectedAlertId] = useState(null);
  const [alertDetail, setAlertDetail] = useState(null);
  const [selectedClaim, setSelectedClaim] = useState(null);

  const pollState = async () => {
    try {
      const res = await fetch("/api/dashboard/state");
      if (res.ok) setState(await res.json());
    } catch {}
  };

  const loadAlertDetail = async (id) => {
    if (!id) return;
    try {
      const res = await fetch(`/api/alerts/${id}/full`);
      if (res.ok) setAlertDetail(await res.json());
    } catch {}
  };

  useEffect(() => {
    pollState();
    const timer = setInterval(pollState, 1200);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!selectedAlertId) return;
    loadAlertDetail(selectedAlertId);
    const timer = setInterval(() => loadAlertDetail(selectedAlertId), 2000);
    return () => clearInterval(timer);
  }, [selectedAlertId]);

  const controlStream = async (action) => {
    await fetch(`/api/stream/control?action=${action}`, { method: "POST" });
    pollState();
  };

  const injectTypology = async (typ) => {
    await fetch(`/api/stream/inject/${typ}`, { method: "POST" });
    pollState();
  };

  const toggleFP = async (alertId) => {
    await fetch(`/api/alerts/${alertId}/false-positive`, { method: "POST" });
    loadAlertDetail(alertId);
    pollState();
  };

  const streamRunning = state?.stream?.running || false;
  const alerts = state?.alerts || [];

  return (
    <div style={{ minHeight: "100vh", padding: "14px 18px", display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Top Bureau Command Bar */}
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12, background: "#090f1d", border: "1px solid #1e293b", padding: "12px 16px", borderRadius: 10 }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "0.04em", color: "#f8fafc" }}>
            AUTONOMOUS BLOCKCHAIN INVESTIGATOR
          </div>
          <div style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace" }}>
            TEMPORAL BIT-CHETG + AMATRICIANA // MODEL: {state?.stream?.checkpoint_status || "INITIALIZING"} // EVENTS: {state?.stream?.events_processed || 0}
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <button onClick={() => controlStream(streamRunning ? "pause" : "start")} style={{ background: streamRunning ? "#dc2626" : "#16a34a", color: "#fff", border: "none", borderRadius: 6, padding: "7px 14px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>
            {streamRunning ? "Pause Live Stream" : "Start Live AMLSim Stream"}
          </button>
          <button onClick={() => injectTypology("peeling_chain")} style={{ background: "#1e293b", color: "#f8fafc", border: "1px solid #334155", borderRadius: 6, padding: "7px 10px", fontSize: 11, cursor: "pointer" }}>
            + Inject Peeling Chain
          </button>
          <button onClick={() => injectTypology("fan_out_fan_in")} style={{ background: "#1e293b", color: "#f8fafc", border: "1px solid #334155", borderRadius: 6, padding: "7px 10px", fontSize: 11, cursor: "pointer" }}>
            + Inject Fan-Out/In
          </button>
          <button onClick={() => injectTypology("cyclic_flow")} style={{ background: "#1e293b", color: "#f8fafc", border: "1px solid #334155", borderRadius: 6, padding: "7px 10px", fontSize: 11, cursor: "pointer" }}>
            + Inject Cyclic Ring
          </button>
          <button onClick={() => injectTypology("smurfing")} style={{ background: "#1e293b", color: "#f8fafc", border: "1px solid #334155", borderRadius: 6, padding: "7px 10px", fontSize: 11, cursor: "pointer" }}>
            + Inject Smurfing
          </button>
        </div>

        <div style={{ fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
          DETECT p50: <strong style={{ color: "#38bdf8" }}>{state?.latency?.p50_ms || 0}ms</strong> | p95: <strong>{state?.latency?.p95_ms || 0}ms</strong>
        </div>
      </header>

      {/* Main 3-Column Bureau Workspace */}
      <div style={{ display: "grid", gridTemplateColumns: "270px 1fr 410px", gap: 14, alignItems: "start" }}>
        {/* Left: Ranked Alert Queue */}
        <aside style={{ background: "#090f1d", border: "1px solid #1e293b", borderRadius: 10, padding: 12, maxHeight: 540, overflowY: "auto" }}>
          <div style={{ fontSize: 11, fontFamily: "monospace", color: "#94a3b8", marginBottom: 10 }}>
            RANKED ALERT QUEUE ({alerts.length})
          </div>
          {alerts.length === 0 && (
            <div style={{ fontSize: 12, color: "#64748b" }}>
              No alerts yet. Click <b>Start Live AMLSim Stream</b> or inject a typology above.
            </div>
          )}
          {alerts.map((a) => (
            <div
              key={a.alert_id}
              onClick={() => { setSelectedAlertId(a.alert_id); setSelectedClaim(null); }}
              style={{
                padding: 10, marginBottom: 8, borderRadius: 6, cursor: "pointer",
                background: selectedAlertId === a.alert_id ? "#172554" : "#0f172a",
                borderLeft: a.is_false_positive ? "3px solid #eab308" : "3px solid #ef4444",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, fontFamily: "monospace" }}>
                <strong style={{ color: "#f8fafc" }}>{a.alert_id}</strong>
                <span style={{ color: "#ef4444", fontWeight: 700 }}>{(a.risk_score * 100).toFixed(1)}%</span>
              </div>
              <div style={{ fontSize: 12, color: "#cbd5e1", marginTop: 3 }}>
                {a.predicted_typology.replace(/_/g, " ").toUpperCase()}
              </div>
              <div style={{ fontSize: 10, color: a.has_report ? "#22c55e" : "#38bdf8", fontFamily: "monospace", marginTop: 4 }}>
                {a.has_report ? "● DOSSIER SEALED (PDF READY)" : "◌ EXPLAINING SUBGRAPH…"}
              </div>
            </div>
          ))}
        </aside>

        {/* Center: Live Expanding & Fading Temporal Graph */}
        <main>
          <LiveExpandingGraph
            incomingBatch={state?.latest_batch}
            evidenceMask={alertDetail?.report?.evidence}
            highlightedRefs={selectedClaim?.evidence_refs || []}
          />
        </main>

        {/* Right: Evidence-Grounded Dossier & Provenance Viewer */}
        <section>
          <DossierViewer
            detail={alertDetail}
            selectedClaimId={selectedClaim?.claim_id}
            onSelectClaim={setSelectedClaim}
            onToggleFP={toggleFP}
          />
        </section>
      </div>
    </div>
  );
}
