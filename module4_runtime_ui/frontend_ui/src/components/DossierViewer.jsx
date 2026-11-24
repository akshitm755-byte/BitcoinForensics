import React from "react";

export default function DossierViewer({ detail, selectedClaimId, onSelectClaim, onToggleFP }) {
  if (!detail) {
    return (
      <div style={{ background: "#090f1d", border: "1px solid #1e293b", borderRadius: 10, padding: 16, color: "#64748b", fontSize: 13 }}>
        Select a flagged alert from the queue—or inject a laundering typology—to inspect its GNNExplainer evidence artifact, provenance-linked claims, and downloadable PDF dossier.
      </div>
    );
  }

  const { alert, report, audit_trace } = detail;

  return (
    <div style={{ background: "#090f1d", border: "1px solid #1e293b", borderRadius: 10, padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
        <div>
          <div style={{ fontSize: 11, color: "#94a3b8", fontFamily: "monospace" }}>CASE FILE // {alert.alert_id}</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#f8fafc" }}>
            {alert.predicted_typology.replace(/_/g, " ").toUpperCase()} ({(alert.risk_score * 100).toFixed(1)}%)
          </div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            onClick={() => onToggleFP(alert.alert_id)}
            style={{
              background: alert.is_false_positive ? "#854d0e" : "#1e293b",
              color: "#f8fafc", border: "1px solid #334155", borderRadius: 6,
              padding: "6px 10px", fontSize: 11, cursor: "pointer",
            }}
          >
            {alert.is_false_positive ? "Marked False Positive" : "Mark False Positive"}
          </button>
          {report && (
            <a
              href={`/api/reports/${report.report_id}/export?format=pdf`}
              target="_blank"
              rel="noreferrer"
              style={{
                background: "#2563eb", color: "#fff", borderRadius: 6,
                padding: "6px 12px", fontSize: 11, fontWeight: 600, textDecoration: "none",
              }}
            >
              Download .PDF Dossier
            </a>
          )}
        </div>
      </div>

      {!report ? (
        <div style={{ fontSize: 12, color: "#38bdf8", fontFamily: "monospace" }}>
          Running PyTorch GNNExplainer + OSINT RAG + Compliance Critic loop...
        </div>
      ) : (
        <>
          <div style={{ display: "flex", gap: 12, fontSize: 11, fontFamily: "monospace", background: "#0f172a", padding: "8px 10px", borderRadius: 6 }}>
            <span>VERDICT: <strong style={{ color: "#22c55e" }}>{report.critic_verdict}</strong></span>
            <span>FAITHFULNESS: <strong style={{ color: "#38bdf8" }}>{(report.faithfulness_score * 100).toFixed(1)}%</strong></span>
            <span>OSINT: <strong style={{ color: report.evidence.attribution_available ? "#f59e0b" : "#94a3b8" }}>
              {report.evidence.attribution_available ? "MATCHED" : "ANONYMIZED"}
            </strong></span>
          </div>

          <div>
            <div style={{ fontSize: 11, color: "#94a3b8", fontFamily: "monospace", marginBottom: 6 }}>
              GROUNDED CLAIMS (CLICK TO HIGHLIGHT PROVENANCE ON LIVE GRAPH)
            </div>
            {report.claims.map((c) => {
              const active = selectedClaimId === c.claim_id;
              return (
                <div
                  key={c.claim_id}
                  onClick={() => onSelectClaim(active ? null : c)}
                  style={{
                    padding: 10, marginBottom: 6, borderRadius: 6, cursor: "pointer", fontSize: 12.5, lineHeight: 1.45,
                    background: active ? "#1e3a8a" : "#0f172a",
                    borderLeft: c.is_grounded ? "3px solid #22c55e" : "3px solid #ef4444",
                  }}
                >
                  <div>{c.sentence}</div>
                  <div style={{ marginTop: 4, fontSize: 10.5, color: "#93c5fd", fontFamily: "monospace" }}>
                    Citations: {c.evidence_refs.join(" · ")}
                  </div>
                </div>
              );
            })}
          </div>

          <div>
            <div style={{ fontSize: 11, color: "#94a3b8", fontFamily: "monospace", marginBottom: 6 }}>
              FEATURE ATTRIBUTIONS (GNNEXPLAINER)
            </div>
            {Object.entries(report.evidence.feature_attributions)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 5)
              .map(([k, v]) => (
                <div key={k} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, fontSize: 11 }}>
                  <span style={{ width: 130, fontFamily: "monospace", color: "#cbd5e1" }}>{k.replace("feat:", "")}</span>
                  <div style={{ flex: 1, height: 6, background: "#1e293b", borderRadius: 3 }}>
                    <div style={{ width: `${Math.min(100, v * 220)}%`, height: "100%", background: "#38bdf8", borderRadius: 3 }} />
                  </div>
                  <span style={{ width: 38, textAlign: "right", fontFamily: "monospace" }}>{(v * 100).toFixed(1)}%</span>
                </div>
              ))}
          </div>

          <div>
            <div style={{ fontSize: 11, color: "#94a3b8", fontFamily: "monospace", marginBottom: 4 }}>
              TAMPER-EVIDENT AUDIT TRACE (SHA-256)
            </div>
            <div style={{ maxHeight: 85, overflowY: "auto", fontSize: 10, fontFamily: "monospace", color: "#64748b", background: "#050811", padding: 8, borderRadius: 6 }}>
              {audit_trace.map((t, i) => (
                <div key={i}>{t.stage} :: {t.sha256_hash.slice(0, 28)}…</div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
