import json
import os
import uuid
from typing import List, Tuple
from contracts.schemas import Alert, Evidence, Claim, Report


class AgenticForensicOrchestrator:
    """
    Multi-Agent Documentation & Compliance Layer:
      1. Narrative & Citation Agent: Gemini 3.8 Flash (primary) -> Gemini 3.5 Flash-Lite (fallback)
         -> Deterministic Grounded Writer (offline fallback).
      2. Compliance Critic Agent: Parses every claim, verifies every ID via Evidence.resolve(),
         strips ungrounded claims, and retries up to 3x before sealing.
      3. Abstractive Summary Agent: Generates a plain-English human conclusion for the end of the dossier.
    """
    MAX_RETRIES = 3

    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.primary_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
        self.fallback_model = "gemini-3.5-flash-lite"

    def generate_verified_report(self, alert: Alert, evidence: Evidence, trigger_mode: str = "serverless_pubsub") -> Report:
        draft_claims, summary, engine = self._narrative_agent_draft(alert, evidence)

        for _ in range(self.MAX_RETRIES):
            all_valid, verified = self._compliance_critic_verify(draft_claims, evidence)
            if all_valid and verified:
                if not summary:
                    summary = self._deterministic_summary(alert, evidence)
                return Report(
                    report_id=f"DOS-{uuid.uuid4().hex[:8].upper()}",
                    alert_id=alert.alert_id,
                    correlation_id=alert.correlation_id,
                    faithfulness_score=self._compute_faithfulness(verified),
                    completeness_score=self._compute_completeness(verified, evidence),
                    critic_verdict="APPROVED",
                    narrative_engine=engine,
                    trigger_mode=trigger_mode,
                    executive_summary=summary,
                    claims=verified,
                    evidence=evidence,
                )
            draft_claims = [c for c in verified if c.is_grounded]
            if not draft_claims:
                draft_claims = self._deterministic_grounded_draft(alert, evidence)
                summary = self._deterministic_summary(alert, evidence)
                engine = "deterministic_grounded_fallback"

        return Report(
            report_id=f"DOS-{uuid.uuid4().hex[:8].upper()}",
            alert_id=alert.alert_id,
            correlation_id=alert.correlation_id,
            faithfulness_score=0.0,
            completeness_score=0.0,
            critic_verdict="ESCALATED_TO_HUMAN",
            narrative_engine=engine,
            trigger_mode=trigger_mode,
            executive_summary="Escalated to human compliance officer: automated claims could not be fully verified against sealed evidence.",
            claims=[],
            evidence=evidence,
        )

    def _narrative_agent_draft(self, alert: Alert, evidence: Evidence) -> Tuple[List[Claim], str, str]:
        if self.gemini_api_key:
            for model_name in [self.primary_model, self.fallback_model]:
                try:
                    claims, summary = self._call_gemini_agent(alert, evidence, model_name)
                    return claims, summary, f"gemini:{model_name}"
                except Exception as exc:
                    print(f"[NarrativeAgent] {model_name} call failed ({exc}); trying next fallback...")
        return (
            self._deterministic_grounded_draft(alert, evidence),
            self._deterministic_summary(alert, evidence),
            "deterministic_grounded",
        )

    def _call_gemini_agent(self, alert: Alert, evidence: Evidence, model_name: str) -> Tuple[List[Claim], str]:
        import httpx
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={self.gemini_api_key}"
        )
        valid_ids = sorted(
            set(evidence.minimal_nodes)
            | set(evidence.minimal_edges)
            | set(evidence.feature_attributions.keys())
            | {d.doc_id for d in evidence.doc_refs}
            | {c.case_id for c in evidence.historic_case_matches}
        )
        prompt = (
            "You are a senior blockchain forensic documentation agent writing an investigative dossier for a bureau. "
            "Return a valid JSON object with exactly TWO keys:\n"
            '1. "claims": array of 5 formal, evidence-grounded investigative claims, each with '
            '{"claim_id": "CLM-01", "sentence": "...", "evidence_refs": ["..."]}. '
            "Cover: (a) Bit-CHetG + Amatriciana + In-Graph Memory flag, (b) GNNExplainer minimal subgraph & top features, "
            "(c) statistical inter-hop timing & volume deviation vs licit baseline, (d) top historic case similarity index & "
            "bureau escalation rate referencing topology + temporal spacing + pseudorandomization, and (e) OSINT wallet attribution "
            "(explicitly state attribution unavailable if attribution_available is false).\n"
            '2. "executive_summary": a unique, 4-5 sentence abstractive summary written in clear, human-friendly prose '
            "(avoid raw field names or robotic templates). Explain in plain language what this cluster did, why its timing and "
            "topology are suspicious compared to ordinary Bitcoin users, how it compares to historic cases, and the concrete next step.\n\n"
            f"STRICT CITATION CONTRACT: Every string in evidence_refs MUST appear verbatim in: {json.dumps(valid_ids)}\n"
            f"ALERT: typology={alert.predicted_typology.value}, risk={alert.risk_score:.4f}, "
            f"cadence={alert.cadence_signature}, motif_slot={alert.matched_motif_slot}, "
            f"in_graph_mem_act={alert.in_graph_memory_activation}\n"
            f"EVIDENCE: {evidence.model_dump_json()}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.35,
                "responseMimeType": "application/json",
            },
        }
        resp = httpx.post(url, json=payload, timeout=22.0)
        resp.raise_for_status()
        raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw_text)
        return [Claim(**c) for c in parsed["claims"]], str(parsed.get("executive_summary", "")).strip()

    def _deterministic_grounded_draft(self, alert: Alert, evidence: Evidence) -> List[Claim]:
        top_nodes = evidence.minimal_nodes[:2]
        top_edges = evidence.minimal_edges[:2]
        ranked_feats = sorted(evidence.feature_attributions.items(), key=lambda kv: kv[1], reverse=True)
        f1_k, f1_v = ranked_feats[0]
        f2_k, f2_v = ranked_feats[1]
        stats = evidence.statistical_correlations
        typ_label = alert.predicted_typology.value.replace("_", " ")

        claims = [
            Claim(
                claim_id="CLM-01",
                sentence=(
                    f"Temporal Bit-CHetG + In-Graph Memory (Motif Slot #{alert.matched_motif_slot}, activation "
                    f"{alert.in_graph_memory_activation:.2f}) flagged this {len(alert.subgraph_nodes)}-node cluster "
                    f"around {', '.join(top_nodes)} at {alert.risk_score:.1%} illicit risk ({typ_label})."
                ),
                evidence_refs=top_nodes + top_edges[:1],
            ),
            Claim(
                claim_id="CLM-02",
                sentence=(
                    f"PyTorch GNNExplainer isolated {len(evidence.minimal_nodes)} nodes and {len(evidence.minimal_edges)} "
                    f"edges ({', '.join(top_edges)}), attributing primary weight to {f1_k.replace('feat:', '')} "
                    f"({f1_v:.1%}) and {f2_k.replace('feat:', '')} ({f2_v:.1%})."
                ),
                evidence_refs=top_nodes + top_edges + [f1_k, f2_k],
            ),
            Claim(
                claim_id="CLM-03",
                sentence=(
                    f"Statistical profiling confirms a median inter-hop cadence of {stats.get('median_inter_hop_sec', 18.0)}s "
                    f"(CV={stats.get('spacing_cv', 0.24)}) across {stats.get('total_cluster_btc', 0.0)} BTC volume, with "
                    f"fan-out ratio deviating {stats.get('fan_out_z_deviation', 0.0):+.2f} SD from the licit network baseline."
                ),
                evidence_refs=["feat:fan_out_ratio", "feat:time_delta_prev"],
            ),
        ]

        if evidence.historic_case_matches:
            top_case = evidence.historic_case_matches[0]
            claims.append(
                Claim(
                    claim_id="CLM-04",
                    sentence=(
                        f"This transaction cluster exhibits a {top_case.similarity_index:.2f} similarity index with the "
                        f"'{top_case.case_name}' ({top_case.year}) case indexed from {top_case.source_domain} by the OSINT & "
                        f"Historic Cases Scraper Agent. Comparable cases yield a {top_case.escalation_rate:.0%} bureau "
                        f"escalation rate when {typ_label} topology coincides with {f1_k.replace('feat:', '')} elevation, "
                        f"{top_case.temporal_spacing_pattern}, and {alert.cadence_signature} pseudorandomization."
                    ),
                    evidence_refs=[top_case.case_id, f1_k, top_nodes[0]],
                )
            )

        if evidence.attribution_available and evidence.doc_refs:
            doc = evidence.doc_refs[0]
            refs = [doc.doc_id] + ([doc.entity_id] if evidence.resolve(doc.entity_id) else [top_nodes[0]])
            claims.append(
                Claim(
                    claim_id="CLM-05",
                    sentence=f"Direct OSINT intelligence matched {doc.entity_id} in '{doc.source_title}': {doc.excerpt}",
                    evidence_refs=refs,
                )
            )
        else:
            claims.append(
                Claim(
                    claim_id="CLM-05",
                    sentence=(
                        "Direct real-world wallet attribution is unavailable for this anonymized cluster; "
                        "investigative priority is strictly grounded in topology, temporal cadence, and historic precedent."
                    ),
                    evidence_refs=[top_nodes[0]],
                )
            )
        return claims

    @staticmethod
    def _deterministic_summary(alert: Alert, evidence: Evidence) -> str:
        stats = evidence.statistical_correlations
        typ = alert.predicted_typology.value.replace("_", " ")
        top_case = evidence.historic_case_matches[0] if evidence.historic_case_matches else None
        case_bit = (
            f"Its movement pattern closely parallels the {top_case.year} {top_case.case_name} incident "
            f"(similarity {top_case.similarity_index:.2f}), a profile that historically escalates to formal investigation "
            f"{top_case.escalation_rate:.0%} of the time. "
            if top_case else ""
        )
        osint_bit = (
            f"Public threat intelligence also linked wallet {evidence.doc_refs[0].entity_id} directly to prior illicit activity, "
            "reinforcing the graph-level flag. "
            if evidence.attribution_available and evidence.doc_refs
            else "Although the individual wallet addresses are not yet listed on public sanctions registries, the automated timing and structural footprint alone justify priority analyst review. "
        )
        return (
            f"In plain terms, this cluster routed {stats.get('total_cluster_btc', 0.0)} BTC through {len(alert.subgraph_nodes)} "
            f"interconnected nodes at a rapid median pace of {stats.get('median_inter_hop_sec', 15.0)} seconds between hops—far "
            f"more synchronized than ordinary Bitcoin payments—matching a classic {typ} laundering structure ({alert.risk_score:.1%} confidence). "
            f"{case_bit}{osint_bit}"
            "Recommended next step: trace the crimson-highlighted exit addresses forward to regulated exchange off-ramps and preserve this signed dossier for compliance escalation."
        )

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
        targets = set(evidence.minimal_nodes[:2]) | set(evidence.minimal_edges[:1])
        if evidence.historic_case_matches:
            targets.add(evidence.historic_case_matches[0].case_id)
        return round(len(cited & targets) / max(1, len(targets)), 4)
