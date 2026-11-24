import json
import os
import uuid
from typing import List
from contracts.schemas import Alert, Evidence, Claim, Report

class AgenticForensicOrchestrator:
    """
    Cooperating Forensic Agents:
      1. Narrative Agent (uses live Gemini API if GEMINI_API_KEY is configured, else grounded deterministic writer)
      2. Compliance Critic (parses claims, enforces Evidence.resolve(ref), rejects & retries up to 3x)
      3. Faithfulness & Completeness Evaluator
    """
    MAX_RETRIES = 3

    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

    def generate_verified_report(self, alert: Alert, evidence: Evidence) -> Report:
        draft_claims = self._narrative_agent_draft(alert, evidence)

        for _ in range(self.MAX_RETRIES):
            all_valid, verified = self._compliance_critic_verify(draft_claims, evidence)
            if all_valid and verified:
                faithfulness = self._compute_faithfulness(verified)
                completeness = self._compute_completeness(verified, evidence)
                return Report(
                    report_id=f"DOS-{uuid.uuid4().hex[:8].upper()}",
                    alert_id=alert.alert_id,
                    correlation_id=alert.correlation_id,
                    faithfulness_score=faithfulness,
                    completeness_score=completeness,
                    critic_verdict="APPROVED",
                    claims=verified,
                    evidence=evidence,
                )
            # Rejection loop: strip ungrounded claims and retry
            draft_claims = [c for c in verified if c.is_grounded]
            if not draft_claims:
                draft_claims = self._deterministic_grounded_draft(alert, evidence)

        return Report(
            report_id=f"DOS-{uuid.uuid4().hex[:8].upper()}",
            alert_id=alert.alert_id,
            correlation_id=alert.correlation_id,
            faithfulness_score=0.0,
            completeness_score=0.0,
            critic_verdict="ESCALATED_TO_HUMAN",
            claims=[],
            evidence=evidence,
        )

    def _narrative_agent_draft(self, alert: Alert, evidence: Evidence) -> List[Claim]:
        if self.gemini_api_key:
            try:
                return self._call_gemini_grounded_writer(alert, evidence)
            except Exception as exc:
                print(f"[NarrativeAgent] Gemini call fell back to deterministic writer: {exc}")
        return self._deterministic_grounded_draft(alert, evidence)

    def _call_gemini_grounded_writer(self, alert: Alert, evidence: Evidence) -> List[Claim]:
        import httpx
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
        )
        prompt = (
            "You are a forensic AML investigator. Draft 4 concise, defensible investigative claims "
            "ONLY using the JSON evidence below. Every claim MUST cite valid IDs from minimal_nodes, "
            "minimal_edges, feature_attributions keys, or doc_refs doc_id. Return a pure JSON array of "
            '{"claim_id": "CLM-1", "sentence": "...", "evidence_refs": ["..."]}.\n'
            f"ALERT: risk={alert.risk_score}, typology={alert.predicted_typology.value}\n"
            f"EVIDENCE: {evidence.model_dump_json()}"
        )
        resp = httpx.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=12.0)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        raw_claims = json.loads(text)
        return [Claim(**c) for c in raw_claims]

    def _deterministic_grounded_draft(self, alert: Alert, evidence: Evidence) -> List[Claim]:
        top_nodes = evidence.minimal_nodes[:2]
        top_edges = evidence.minimal_edges[:2]
        ranked_feats = sorted(evidence.feature_attributions.items(), key=lambda kv: kv[1], reverse=True)
        f1_name, f1_val = ranked_feats[0]
        f2_name, f2_val = ranked_feats[1]

        claims = [
            Claim(
                claim_id="CLM-01",
                sentence=(
                    f"Temporal Bit-CHetG flagged a {len(alert.subgraph_nodes)}-node subgraph centered on "
                    f"{', '.join(top_nodes)} with an illicit risk score of {alert.risk_score:.2%} "
                    f"consistent with a {alert.predicted_typology.value.replace('_', ' ')} typology."
                ),
                evidence_refs=top_nodes + top_edges[:1],
            ),
            Claim(
                claim_id="CLM-02",
                sentence=(
                    f"PyTorch GNNExplainer isolated {len(evidence.minimal_nodes)} critical nodes and "
                    f"{len(evidence.minimal_edges)} edges ({', '.join(top_edges)}) as the minimal structural "
                    f"explanation sustaining the classification."
                ),
                evidence_refs=top_nodes + top_edges,
            ),
            Claim(
                claim_id="CLM-03",
                sentence=(
                    f"Feature attribution analysis ranked {f1_name.replace('feat:', '')} ({f1_val:.2%}) and "
                    f"{f2_name.replace('feat:', '')} ({f2_val:.2%}) as the primary quantitative drivers "
                    f"within window [{alert.window_bounds[0]}, {alert.window_bounds[1]}]."
                ),
                evidence_refs=[f1_name, f2_name],
            ),
        ]

        if evidence.attribution_available and evidence.doc_refs:
            doc = evidence.doc_refs[0]
            refs = [doc.doc_id] + ([doc.entity_id] if evidence.resolve(doc.entity_id) else top_nodes[:1])
            claims.append(
                Claim(
                    claim_id="CLM-04",
                    sentence=f"OSINT attribution matched {doc.entity_id} in '{doc.source_title}': {doc.excerpt}",
                    evidence_refs=refs,
                )
            )
        else:
            claims.append(
                Claim(
                    claim_id="CLM-04",
                    sentence=(
                        "External OSINT wallet attribution is unavailable for this anonymized cluster; "
                        "investigative priority is strictly grounded in temporal graph topology."
                    ),
                    evidence_refs=top_nodes[:1],
                )
            )
        return claims

    def _compliance_critic_verify(self, claims: List[Claim], evidence: Evidence):
        if not claims:
            return False, []
        all_ok = True
        for c in claims:
            if c.evidence_refs and all(evidence.resolve(ref) for ref in c.evidence_refs):
                c.is_grounded = True
            else:
                c.is_grounded = False
                all_ok = False
        return all_ok, claims

    @staticmethod
    def _compute_faithfulness(claims: List[Claim]) -> float:
        return round(sum(1 for c in claims if c.is_grounded) / max(1, len(claims)), 4)

    @staticmethod
    def _compute_completeness(claims: List[Claim], evidence: Evidence) -> float:
        cited = {ref for c in claims for ref in c.evidence_refs}
        important_targets = set(evidence.minimal_nodes[:2]) | set(evidence.minimal_edges[:1])
        if not important_targets:
            return 1.0
        return round(len(cited & important_targets) / len(important_targets), 4)
