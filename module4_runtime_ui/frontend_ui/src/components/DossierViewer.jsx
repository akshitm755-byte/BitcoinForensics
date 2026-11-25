import React from "react";

export default function DossierViewer({ detail, selectedClaimId, onSelectClaim, onToggleFP }) {
  if (!detail) {
    return (
      <div style={{ background: "#11141c", border: "1px solid #1e232e", borderRadius: 12, padding: 20, color: "#64748b", fontSize: 13, lineHeight: 1.6 }}>
        Select a flagged case from the alert queue—or inject an interlaced laundering campaign—to inspect its PyTorch GNNExplainer masks, multimodal figure-by-figure analysis, OSINT historic case similarity index, and downloadable PDF dossier.
      </div>
    );
  }

  const { alert, report, audit_trace } = detail;
  const topCase = report?.evidence?.historic_case_matches?.[0];

  return (
    <div style={{ background: "#11141c", border: "1px solid #1e232e", borderRadius: 12, padding: 16, display: "flex", flexDirection: "column", gap: 13, maxHeight: 595, overflowY: "auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <div>
          <div style={{ fontSize: 10.5, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
            CASE // {alert.alert_id} · MOTIF SLOT #{alert.matched_motif_slot}
          </div>
          <div style={{ fontSize: 16, fontWeight: 600, color: "#f8fafc", marginTop: 2 }}>
            {alert.predicted_typology.replace(/_/g, " ").toUpperCase()}{" "}
            <span style={{ color: "#fb7185", fontFamily: "'JetBrains Mono', monospace", fontSize: 14 }}>
              {(alert.risk_score * 100).toFixed(1)}%
            </span>
          </div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            onClick={() => onToggleFP(alert.alert_id)}
            style={{
              background: alert.is_false_positive ? "rgba(234,179,8,0.18)" : "#181c26",
              color: alert.is_false_positive ? "#fde047" : "#cbd5e1",
              border: "1px solid #262b36", borderRadius: 6, padding: "6px 9px", fontSize: 11, cursor: "pointer",
            }}
          >
            {alert.is_false_positive ? "False Positive" : "Flag FP"}
          </button>
          {report && (
            <a
              href={`/api/reports/${report.report_id}/export?format=pdf`}
              target="_blank"
              rel="noreferrer"
              style={{
                background: "#f8fafc", color: "#0a0c10", borderRadius: 6,
                padding: "6px 11px", fontSize: 11, fontWeight: 600, textDecoration: "none",
              }}
            >
              Export .PDF
            </a>
          )}
        </div>
      </div>

      {!report ? (
        <div style={{ fontSize: 12, color: "#38bdf8", fontFamily: "'JetBrains Mono', monospace", padding: "12px 0" }}>
          Serverless Pub/Sub triggered -> Rendering Figures 1-5 + Multimodal Gemini Documentation Agent...
        </div>
      ) : (
        <>
          {report.executive_summary && (
            <div style={{ background: "rgba(56,189,248,0.06)", border: "1px solid rgba(56,189,248,0.28)", borderRadius: 8, padding: "10px 12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9.5, fontFamily: "'JetBrains Mono', monospace", color: "#38bdf8", marginBottom: 4 }}>
                <span>EXECUTIVE INVESTIGATOR BRIEF</span>
                <span>{report.narrative_engine.toUpperCase()}</span>
              </div>
              <div style={{ fontSize: 12.2, lineHeight: 1.55, color: "#e2e8f0" }}>
                {report.executive_summary}
              </div>
            </div>
          )}

          {topCase && (
            <div style={{ background: "#0a0c10", border: "1px solid #1e232e", borderRadius: 8, padding: "10px 12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                <span style={{ fontSize: 10, color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace" }}>
                  OSINT PRECEDENT ({topCase.source_domain})
                </span>
                <span style={{ fontSize: 11, fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: "#38bdf8" }}>
                  SIM {topCase.similarity_index.toFixed(2)} · ESC {(topCase.escalation_rate * 100).toFixed(0)}%
                </span>
              </div>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: "#f1f5f9" }}>
                {topCase.case_name} ({topCase.year})
              </div>
              <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 3, lineHeight: 1.45 }}>
                {topCase.temporal_spacing_pattern} · <span style={{ color: "#cbd5e1" }}>{topCase.pseudorandomization_signature}</span>
              </div>
            </div>
          )}

          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: "#94a3b8", padding: "6px 10px", background: "#0a0c10", borderRadius: 6, border: "1px solid #1e232e" }}>
            <span>CRITIC: <strong style={{ color: "#4ade80" }}>{report.critic_verdict}</strong></span>
            <span>FAITHFUL: <strong style={{ color: "#38bdf8" }}>{(report.faithfulness_score * 100).toFixed(0)}%</strong></span>
            <span>TRIGGER: <strong>{report.trigger_mode.toUpperCase()}</strong></span>
          </div>

          <div>
            <div style={{ fontSize: 10, color: "#64748b", fontFamily: "'JetBrains Mono', monospace", marginBottom: 6 }}>
              GROUNDED INVESTIGATIVE CLAIMS (CLICK TO HIGHLIGHT ON GRAPH)
            </div>
            {report.claims.map((c) => {
              const active = selectedClaimId === c.claim_id;
              return (
                <div
                  key={c.claim_id}
                  onClick={() => onSelectClaim(active ? null : c)}
                  style={{
                    padding: "9px 11px", marginBottom: 6, borderRadius: 7, cursor: "pointer",
                    fontSize: 12, lineHeight: 1.5, color: "#e2e8f0",
                    background: active ? "rgba(56,189,248,0.12)" : "#0a0c10",
                    border: `1px solid ${active ? "rgba(56,189,248,0.5)" : "#1e232e"}`,
                    borderLeft: `3px solid ${c.is_grounded ? "#22c55e" : "#f43f5e"}`,
                  }}
                >
                  <div>{c.sentence}</div>
                  <div style={{ marginTop: 4, fontSize: 10, color: "#64748b", fontFamily: "'JetBrains Mono', monospace" }}>
                    {c.evidence_refs.join(" · ")}
                  </div>
                </div>
              );
            })}
          </div>

          {report.figure_explanations && report.figure_explanations.length > 0 && (
            <div>
              <div style={{ fontSize: 10, color: "#64748b", fontFamily: "'JetBrains Mono', monospace", marginBottom: 6 }}>
                MULTIMODAL FIGURE-BY-FIGURE BREAKDOWN (PDF FIGURES 1–5)
              </div>
              {report.figure_explanations.map((f) => {
                const active = selectedClaimId === f.figure_id;
                return (
                  <div
                    key={f.figure_id}
                    onClick={() => onSelectClaim(active ? null : { claim_id: f.figure_id, evidence_refs: f.evidence_refs })}
                    style={{
                      padding: "8px 10px", marginBottom: 6, borderRadius: 7, cursor: "pointer",
                      background: active ? "rgba(56,189,248,0.12)" : "#0a0c10",
                      border: `1px solid ${active ? "rgba(56,189,248,0.5)" : "#1e232e"}`,
                    }}
                  >
                    <div style={{ fontSize: 10.5, fontWeight: 600, color: "#38bdf8", fontFamily: "'JetBrains Mono', monospace", marginBottom: 3 }}>
                      {f.figure_id} — {f.title}
                    </div>
                    <div style={{ fontSize: 11.5, lineHeight: 1.48, color: "#cbd5e1" }}>{f.explanation}</div>
                  </div>
                );
              })}
            </div>
          )}

          <div>
            <div style={{ fontSize: 10, color: "#64748b", fontFamily: "'JetBrains Mono', monospace", marginBottom: 6 }}>
              GNNEXPLAINER FEATURE ATTRIBUTIONS
            </div>
            {Object.entries(report.evidence.feature_attributions)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 5)
              .map(([k, v]) => (
                <div key={k} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5, fontSize: 11 }}>
                  <span style={{ width: 122, fontFamily: "'JetBrains Mono', monospace", color: "#94a3b8" }}>{k.replace("feat:", "")}</span>
                  <div style={{ flex: 1, height: 5, background: "#181c26", borderRadius: 3 }}>
                    <div style={{ width: `${Math.min(100, v * 230)}%`, height: "100%", background: "#38bdf8", borderRadius: 3 }} />
                  </div>
                  <span style={{ width: 36, textAlign: "right", fontFamily: "'JetBrains Mono', monospace", color: "#e2e8f0" }}>{(v * 100).toFixed(0)}%</span>
                </div>
              ))}
          </div>

          <div>
            <div style={{ fontSize: 10, color: "#64748b", fontFamily: "'JetBrains Mono', monospace", marginBottom: 4 }}>
              SHA-256 PROVENANCE AUDIT TRAIL
            </div>
            <div style={{ fontSize: 9.5, fontFamily: "'JetBrains Mono', monospace", color: "#64748b", background: "#0a0c10", padding: 8, borderRadius: 6, border: "1px solid #1e232e" }}>
              {audit_trace.map((t, i) => (
                <div key={i}>{t.stage} :: {t.sha256_hash.slice(0, 32)}…</div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
