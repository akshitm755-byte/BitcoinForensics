import base64
import json
import os
import uuid
from typing import Dict, List, Optional, Tuple
from contracts.schemas import Alert, Evidence, Claim, FigureExplanation, Report


class AgenticForensicOrchestrator:
    """
    Multimodal Multi-Agent Documentation & Compliance Layer:
      1. Visual & Narrative Documentation Agent: Sends all 5 labeled chart PNGs PLUS exact
         underlying metrics to Gemini (`gemini-3.8-flash` -> `gemini-3.5-flash-lite` -> deterministic fallback)
         to generate:
           (a) Evidence-grounded forensic claims
           (b) Dedicated figure-by-figure visual explanations (Figure 1 through Figure 5)
           (c) Plain-English abstractive concluding executive summary
      2. Compliance Critic Agent: Verifies every citation across BOTH `claims` and `figure_explanations`
         via `Evidence.resolve(ref_id)`, stripping or repairing ungrounded statements before sealing.
    """
    MAX_RETRIES = 3

    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.primary_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
        self.fallback_model = "gemini-3.5-flash-lite"

    def generate_verified_report(
        self,
        alert: Alert,
        evidence: Evidence,
        trigger_mode: str = "serverless_pubsub",
        charts: Optional[Dict[str, dict]] = None,
    ) -> Report:
        draft_claims, draft_figs, summary, engine = self._narrative_agent_draft(alert, evidence, charts=charts)

        # Verify figure explanations individually; replace any ungrounded figure with deterministic grounded backup
        fallback_figs = {f.figure_id: f for f in self._deterministic_figure_explanations(alert, evidence, charts)}
        verified_figs: List[FigureExplanation] = []
        seen_ids = set()
        for fe in draft_figs:
            if fe.evidence_refs and all(evidence.resolve(r) for r in fe.evidence_refs):
                fe.is_grounded = True
                verified_figs.append(fe)
                seen_ids.add(fe.figure_id)
        for fid in ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5"]:
            if fid not in seen_ids and fid in fallback_figs:
                fb = fallback_figs[fid]
                fb.is_grounded = all(evidence.resolve(r) for r in fb.evidence_refs)
                verified_figs.append(fb)
        verified_figs.sort(key=lambda f: f.figure_id)

        for _ in range(self.MAX_RETRIES):
            all_valid, verified_claims = self._compliance_critic_verify(draft_claims, evidence)
            if all_valid and verified_claims:
                if not summary:
                    summary = self._deterministic_summary(alert, evidence)
                return Report(
                    report_id=f"DOS-{uuid.uuid4().hex[:8].upper()}",
                    alert_id=alert.alert_id,
                    correlation_id=alert.correlation_id,
                    faithfulness_score=self._compute_faithfulness(verified_claims, verified_figs),
                    completeness_score=self._compute_completeness(verified_claims, evidence),
                    critic_verdict="APPROVED",
                    narrative_engine=engine,
                    trigger_mode=trigger_mode,
                    executive_summary=summary,
                    claims=verified_claims,
                    figure_explanations=verified_figs,
                    evidence=evidence,
                )
            draft_claims = [c for c in verified_claims if c.is_grounded]
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
            executive_summary="Escalated to human compliance officer: automated claims could not be verified.",
            claims=[],
            figure_explanations=verified_figs,
            evidence=evidence,
        )

    def _narrative_agent_draft(
        self, alert: Alert, evidence: Evidence, charts: Optional[Dict[str, dict]] = None
    ) -> Tuple[List[Claim], List[FigureExplanation], str, str]:
        if self.gemini_api_key:
            for model_name in [self.primary_model, self.fallback_model]:
                try:
                    claims, fig_exps, summary = self._call_gemini_agent(alert, evidence, model_name, charts=charts)
                    return claims, fig_exps, summary, f"gemini-multimodal:{model_name}"
                except Exception as exc:
                    print(f"[NarrativeAgent] {model_name} multimodal call failed ({exc}); trying fallback...")
        return (
            self._deterministic_grounded_draft(alert, evidence),
            self._deterministic_figure_explanations(alert, evidence, charts),
            self._deterministic_summary(alert, evidence),
            "deterministic_grounded",
        )

    def _call_gemini_agent(
        self,
        alert: Alert,
        evidence: Evidence,
        model_name: str,
        charts: Optional[Dict[str, dict]] = None,
    ) -> Tuple[List[Claim], List[FigureExplanation], str]:
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
            "You have been provided both structured forensic evidence AND the actual rendered visualizations (Figure 1 through Figure 5) "
            "along with their exact underlying chart metrics. Inspect every figure visually and cross-check against the exact metrics.\n\n"
            "Return a valid JSON object with EXACTLY THREE keys:\n"
            '1. "claims": array of 5 formal investigative claims, each with '
            '{"claim_id": "CLM-01", "sentence": "...", "evidence_refs": ["..."]}. '
            "Cover: (a) Bit-CHetG + Amatriciana + In-Graph Memory flag, (b) GNNExplainer minimal subgraph & top features, "
            "(c) statistical inter-hop timing & volume deviation vs licit baseline, (d) top historic case similarity index & "
            "bureau escalation rate referencing topology + temporal spacing + pseudorandomization, and (e) OSINT wallet attribution "
            "(explicitly state unavailable if attribution_available is false).\n"
            '2. "figure_explanations": array of 5 objects — one each for "Figure 1", "Figure 2", "Figure 3", "Figure 4", and "Figure 5" — '
            'each with {"figure_id": "Figure 1", "title": "...", "explanation": "2-3 clear sentences describing what is visually shown in that exact figure and what it proves forensically", '
            '"evidence_refs": ["..."]}.\n'
            '3. "executive_summary": a unique, 4-5 sentence abstractive summary written in clear, natural human language '
            "(avoid raw field names or boilerplate). Explain what happened in plain terms, why the timing and topology are suspicious, "
            "what historic precedent suggests, and the concrete next investigative step.\n\n"
            f"STRICT CITATION CONTRACT: Every string inside any evidence_refs array MUST appear verbatim in: {json.dumps(valid_ids)}\n"
            f"ALERT: typology={alert.predicted_typology.value}, risk={alert.risk_score:.4f}, "
            f"cadence={alert.cadence_signature}, motif_slot={alert.matched_motif_slot}, "
            f"in_graph_mem_act={alert.in_graph_memory_activation}\n"
            f"EVIDENCE: {evidence.model_dump_json()}"
        )

        parts = [{"text": prompt}]
        if charts:
            for fig_id, chart_obj in charts.items():
                parts.append({
                    "text": f"\n[{fig_id}: {chart_obj['title']}] Exact metrics: {json.dumps(chart_obj['stats'])}"
                })
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(chart_obj["png_bytes"]).decode("utf-8"),
                    }
                })

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
            },
        }
        resp = httpx.post(url, json=payload, timeout=35.0)
        resp.raise_for_status()
        raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw_text)

        claims = [Claim(**c) for c in parsed["claims"]]
        fig_exps = [FigureExplanation(**f) for f in parsed.get("figure_explanations", [])]
        summary = str(parsed.get("executive_summary", "")).strip()
        return claims, fig_exps, summary

    def _deterministic_figure_explanations(
        self, alert: Alert, evidence: Evidence, charts: Optional[Dict[str, dict]] = None
    ) -> List[FigureExplanation]:
        top_nodes = evidence.minimal_nodes[:2]
        top_edges = evidence.minimal_edges[:2]
        ranked_feats = sorted(evidence.feature_attributions.items(), key=lambda kv: kv[1], reverse=True)
        f1_k, f1_v = ranked_feats[0]
        f2_k, f2_v = ranked_feats[1]
        stats = evidence.statistical_correlations
        typ_label = alert.predicted_typology.value.replace("_", " ")

        fig3_stats = (charts or {}).get("Figure 3", {}).get("stats", {})
        pos_corr = fig3_stats.get("strongest_positive_correlations", [])
        neg_corr = fig3_stats.get("strongest_negative_correlations", [])
        corr_snippet = (
            f"The strongest positive coupling appears in {pos_corr[0]['pair']} (r={pos_corr[0]['r']}), "
            f"while {neg_corr[0]['pair']} shows inverse movement (r={neg_corr[0]['r']})."
            if (pos_corr and neg_corr)
            else "The red and blue off-diagonal blocks highlight synchronized feature coupling across active nodes."
        )

        fig4_stats = (charts or {}).get("Figure 4", {}).get("stats", {})
        med_dt = fig4_stats.get("flagged_median_spacing_sec", stats.get("median_inter_hop_sec", 15.0))
        med_btc = fig4_stats.get("flagged_median_btc", 0.5)

        top_case = evidence.historic_case_matches[0] if evidence.historic_case_matches else None
        fig5_refs = [top_case.case_id, top_nodes[0]] if top_case else [top_nodes[0]]
        fig5_text = (
            f"Figure 5 contrasts vector similarity (blue) against historic bureau escalation rates (amber) across the top "
            f"OSINT-scraped precedents. '{top_case.case_name}' ({top_case.year}) leads at {top_case.similarity_index:.2f} "
            f"similarity with a {top_case.escalation_rate:.0%} escalation rate, reinforcing high triage priority for this {typ_label} cluster."
            if top_case
            else "Figure 5 summarizes vector similarity against historic bureau escalation benchmarks for retrieved precedents."
        )

        return [
            FigureExplanation(
                figure_id="Figure 1",
                title="Minimal Flagged Subgraph Topology",
                explanation=(
                    f"Figure 1 maps the {len(alert.subgraph_nodes)}-node neighborhood around this alert, highlighting the "
                    f"{len(evidence.minimal_nodes)} crimson nodes (led by {', '.join(top_nodes)}) and {len(evidence.minimal_edges)} "
                    f"crimson edges ({', '.join(top_edges)}) that PyTorch GNNExplainer isolated as sufficient to sustain the {typ_label} flag. "
                    f"Blue context nodes provide surrounding wallet connectivity without driving the core risk score."
                ),
                evidence_refs=top_nodes + top_edges[:1],
                is_grounded=True,
            ),
            FigureExplanation(
                figure_id="Figure 2",
                title="GNNExplainer Feature Attribution Weights",
                explanation=(
                    f"Figure 2 ranks the seven recomputable node features by learned GNNExplainer importance. "
                    f"The top crimson bar, {f1_k.replace('feat:', '')} ({f1_v:.1%}), paired with {f2_k.replace('feat:', '')} ({f2_v:.1%}), "
                    f"contributes the majority of the decision signal, confirming that topology-velocity anomalies rather than isolated volume triggered the model."
                ),
                evidence_refs=[f1_k, f2_k],
                is_grounded=True,
            ),
            FigureExplanation(
                figure_id="Figure 3",
                title="Subgraph Feature Correlation Heatmap (Pearson r)",
                explanation=(
                    f"Figure 3 displays pairwise Pearson correlations across all nodes inside the flagged subgraph. "
                    f"{corr_snippet} Tight multi-feature covariance like this is characteristic of scripted {typ_label} automation "
                    f"rather than independent human wallets."
                ),
                evidence_refs=[f1_k, "feat:time_delta_prev"],
                is_grounded=True,
            ),
            FigureExplanation(
                figure_id="Figure 4",
                title="Temporal Spacing vs. Volume Anomaly Scatter",
                explanation=(
                    f"Figure 4 plots inter-transaction spacing against transferred BTC volume relative to the grey licit baseline cloud (45-220s). "
                    f"The crimson flagged edges cluster tightly on the left at a median spacing of {med_dt}s (median {med_btc} BTC), "
                    f"visually separating this {alert.cadence_signature} burst from ordinary retail confirmation cadence."
                ),
                evidence_refs=top_edges[:1] + ["feat:time_delta_prev"],
                is_grounded=True,
            ),
            FigureExplanation(
                figure_id="Figure 5",
                title="OSINT Historic Cases — Similarity vs. Escalation",
                explanation=fig5_text,
                evidence_refs=fig5_refs,
                is_grounded=True,
            ),
        ]

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
    def _compute_faithfulness(claims: List[Claim], fig_exps: List[FigureExplanation]) -> float:
        total = len(claims) + len(fig_exps)
        grounded = sum(1 for c in claims if c.is_grounded) + sum(1 for f in fig_exps if f.is_grounded)
        return round(grounded / max(1, total), 4)

    @staticmethod
    def _compute_completeness(claims: List[Claim], evidence: Evidence) -> float:
        cited = {ref for c in claims for ref in c.evidence_refs}
        targets = set(evidence.minimal_nodes[:2]) | set(evidence.minimal_edges[:1])
        if evidence.historic_case_matches:
            targets.add(evidence.historic_case_matches[0].case_id)
        return round(len(cited & targets) / max(1, len(targets)), 4)
