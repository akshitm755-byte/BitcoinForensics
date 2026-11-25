import React, { useEffect, useState } from "react";
import LiveExpandingGraph from "./components/LiveExpandingGraph";
import DossierViewer from "./components/DossierViewer";

export default function App() {
  const [state, setState] = useState(null);
  const [selectedAlertId, setSelectedAlertId] = useState(null);
  const [alertDetail, setAlertDetail] = useState(null);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [reloading, setReloading] = useState(false);

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

  const reloadModels = async () => {
    setReloading(true);
    try {
      await fetch("/api/models/reload", { method: "POST" });
      await pollState();
    } finally {
      setReloading(false);
    }
  };

  const toggleFP = async (alertId) => {
    await fetch(`/api/alerts/${alertId}/false-positive`, { method: "POST" });
    loadAlertDetail(alertId);
    pollState();
  };

  const streamRunning = state?.stream?.running || false;
  const alerts = state?.alerts || [];
  const trained = String(state?.stream?.checkpoint_status || "").startsWith("TRAINED");

  return (
    <div style={{ minHeight: "100vh", padding: "16px 20px", display: "flex", flexDirection: "column", gap: 14, maxWidth: 1680, margin: "0 auto" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12, background: "#11141c", border: "1px solid #1e232e", padding: "12px 18px", borderRadius: 12 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 15, fontWeight: 700, letterSpacing: "-0.01em", color: "#f8fafc" }}>
              ABI Forensics v4
            </span>
            <span
              style={{
                fontSize: 10, fontFamily: "'JetBrains Mono', monospace", padding: "2px 7px", borderRadius: 99,
                background: trained ? "rgba(34,197,94,0.14)" : "rgba(245,158,11,0.14)",
                color: trained ? "#4ade80" : "#fbbf24",
              }}
            >
              {state?.stream?.checkpoint_status || "CONNECTING"}
            </span>
            <span
              style={{
                fontSize: 10, fontFamily: "'JetBrains Mono', monospace", padding: "2px 7px", borderRadius: 99,
                background: "rgba(56,189,248,0.12)", color: "#38bdf8",
              }}
            >
              {state?.stream?.reporting_dispatch || "SERVERLESS"}
            </span>
          </div>
          <div style={{ fontSize: 11, color: "#64748b", fontFamily: "'JetBrains Mono', monospace", marginTop: 3 }}>
            Bit-CHetG + Amatriciana + In-Graph Memory · Events: {state?.stream?.events_processed || 0} · Interlaced Queue: {state?.stream?.interlaced_pending_hops || 0} hops
          </div>
        </div>

        <div style={{ display: "flex", gap: 7, flexWrap: "wrap", alignItems: "center" }}>
          <button
            onClick={() => controlStream(streamRunning ? "pause" : "start")}
            style={{
              background: streamRunning ? "#e11d48" : "#f8fafc",
              color: streamRunning ? "#fff" : "#0a0c10",
              border: "none", borderRadius: 7, padding: "7px 14px", fontSize: 12, fontWeight: 600, cursor: "pointer",
            }}
          >
            {streamRunning ? "Pause Stream" : "Start AMLSim Stream"}
          </button>
          {[
            ["peeling_chain", "Peeling Chain"],
            ["fan_out_fan_in", "Fan-Out / In"],
            ["cyclic_flow", "Cyclic Ring"],
            ["smurfing", "Smurfing"],
          ].map(([key, label]) => (
            <button
              key={key}
              onClick={() => injectTypology(key)}
              style={{
                background: "#181c26", color: "#cbd5e1", border: "1px solid #262b36",
                borderRadius: 7, padding: "7px 10px", fontSize: 11.5, cursor: "pointer",
              }}
            >
              + {label}
            </button>
          ))}
          <button
            onClick={reloadModels}
            disabled={reloading}
            style={{
              background: "transparent", color: "#38bdf8", border: "1px solid rgba(56,189,248,0.35)",
              borderRadius: 7, padding: "7px 10px", fontSize: 11.5, cursor: "pointer",
            }}
          >
            {reloading ? "Syncing…" : "Sync GCS Model"}
          </button>
        </div>

        <div style={{ fontSize: 11, fontFamily: "'JetBrains Mono', monospace", color: "#94a3b8" }}>
          p50 <strong style={{ color: "#f8fafc" }}>{state?.latency?.p50_ms || 0}ms</strong> · p95 <strong>{state?.latency?.p95_ms || 0}ms</strong>
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "265px 1fr 430px", gap: 14, alignItems: "start" }}>
        <aside style={{ background: "#11141c", border: "1px solid #1e232e", borderRadius: 12, padding: 14, maxHeight: 595, overflowY: "auto" }}>
          <div style={{ fontSize: 10.5, fontFamily: "'JetBrains Mono', monospace", color: "#64748b", marginBottom: 10 }}>
            ALERT QUEUE ({alerts.length})
          </div>
          {alerts.length === 0 && (
            <div style={{ fontSize: 12, color: "#64748b", lineHeight: 1.55 }}>
              Stream idle. Start the interlaced AMLSim feed or inject a laundering campaign to trigger serverless multi-agent analysis.
            </div>
          )}
          {alerts.map((a) => {
            const active = selectedAlertId === a.alert_id;
            return (
              <div
                key={a.alert_id}
                onClick={() => { setSelectedAlertId(a.alert_id); setSelectedClaim(null); }}
                style={{
                  padding: "10px 11px", marginBottom: 8, borderRadius: 8, cursor: "pointer",
                  background: active ? "#181f2e" : "#0a0c10",
                  border: `1px solid ${active ? "#38bdf8" : "#1e232e"}`,
                  borderLeft: `3px solid ${a.is_false_positive ? "#eab308" : "#f43f5e"}`,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, fontFamily: "'JetBrains Mono', monospace" }}>
                  <strong style={{ color: "#f8fafc" }}>{a.alert_id}</strong>
                  <span style={{ color: "#fb7185", fontWeight: 600 }}>{(a.risk_score * 100).toFixed(1)}%</span>
                </div>
                <div style={{ fontSize: 12, color: "#cbd5e1", fontWeight: 500, marginTop: 3 }}>
                  {a.predicted_typology.replace(/_/g, " ").toUpperCase()}
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, fontFamily: "'JetBrains Mono', monospace", marginTop: 5 }}>
                  <span style={{ color: a.has_report ? "#4ade80" : "#38bdf8" }}>
                    {a.has_report ? "● PDF READY" : "◌ SERVERLESS…"}
                  </span>
                  {a.top_similarity && <span style={{ color: "#94a3b8" }}>SIM {a.top_similarity.toFixed(2)}</span>}
                </div>
              </div>
            );
          })}
        </aside>

        <main>
          <LiveExpandingGraph
            incomingBatch={state?.latest_batch}
            focusedAlert={alertDetail?.alert}
            evidenceMask={alertDetail?.report?.evidence}
            highlightedRefs={selectedClaim?.evidence_refs || []}
            baselines={state?.baselines || {}}
          />
        </main>

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
