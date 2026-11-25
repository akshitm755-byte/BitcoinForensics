import base64
import json
import os
import sys
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./contracts"))

from contracts.schemas import Alert, Typology, Claim
from module1_data_pipeline.src.amlsim_stream import AMLSimStreamGenerator
from module1_data_pipeline.src.splitter import TemporalSplitter
from module2_detection_gnn.src.graph_store import IncrementalGraphStore
from module2_detection_gnn.src.torch_model import TemporalBitCHetGNet, subgraph_to_tensors
from module3_reporting_agents.src.evidence_builder import EvidenceArtifactBuilder
from module3_reporting_agents.src.orchestrator import AgenticForensicOrchestrator
from module3_reporting_agents.src.pdf_exporter import ForensicDossierPDFExporter


def test_terraform_hcl2_has_no_semicolons():
    with open("infra/main.tf", "r", encoding="utf-8") as f:
        tf_text = f.read()
    assert ";" not in tf_text, "Semicolons are invalid in Terraform HCL2 multiline blocks!"
    assert "google_pubsub_subscription" in tf_text
    assert "google_storage_bucket_iam_member" in tf_text


def test_v4_full_serverless_and_multimodal_figures_pipeline():
    # 1. Interlaced AMLSim stream verification
    gen = AMLSimStreamGenerator(seed=11, illicit_campaign_prob=0.0)
    gen.enqueue_interlaced_typology(Typology.PEELING_CHAIN, immediate_first_hop=True)
    stream_seq = [gen.next_event() for _ in range(16)]
    peel_idx = [i for i, e in enumerate(stream_seq) if e.ground_truth_typology == Typology.PEELING_CHAIN]
    assert len(peel_idx) >= 3
    assert any((peel_idx[i + 1] - peel_idx[i]) > 1 for i in range(len(peel_idx) - 1))

    # 2. Temporal leakage gate
    c_gen = AMLSimStreamGenerator(seed=21, illicit_campaign_prob=0.08)
    nodes, edges, _ = c_gen.generate_training_corpus(total_events=60, illicit_campaign_prob=0.08)
    ts_sorted = sorted(n.timestamp for n in nodes)
    splitter = TemporalSplitter(ts_sorted[int(len(ts_sorted) * 0.5)], ts_sorted[int(len(ts_sorted) * 0.75)], ts_sorted[-1])
    manifest, _ = splitter.split_and_verify(nodes, edges)
    assert manifest.leakage_verified is True

    # 3. Bit-CHetG + Amatriciana + In-Graph Memory forward pass
    store = IncrementalGraphStore(memory_dim=32)
    model = TemporalBitCHetGNet()
    model.eval()
    target_sub = None
    for ev in stream_seq:
        sub_nodes, sub_edges = store.apply_event(ev)
        x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, _ = subgraph_to_tensors(
            sub_nodes, sub_edges, store.memory, memory_dim=32
        )
        with torch.no_grad():
            logit, _, _, upd_mem, motif_act, motif_slot = model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx,
                enable_memory=True, enable_in_graph_memory=True
            )
        assert torch.isfinite(logit).all()
        for i, n in enumerate(sub_nodes):
            store.memory[n.node_id] = upd_mem[i]
        if ev.ground_truth_typology == Typology.PEELING_CHAIN:
            target_sub = (ev, sub_nodes, sub_edges, motif_act, motif_slot)

    ev, sub_nodes, sub_edges, motif_act, motif_slot = target_sub
    alert = Alert(
        alert_id="ALT-V4FIGTEST",
        correlation_id=ev.correlation_id,
        timestamp=ev.timestamp,
        risk_score=0.88,
        model_version="Temporal-BitCHetG-InGraphMem-v4.0",
        in_graph_memory_activation=round(motif_act, 4),
        matched_motif_slot=motif_slot,
        predicted_typology=Typology.PEELING_CHAIN,
        cadence_signature=ev.cadence_algorithm or "lognormal_peel_jitter_v2",
        window_bounds=(sub_nodes[0].timestamp, ev.timestamp),
        subgraph_nodes=sub_nodes,
        subgraph_edges=sub_edges,
    )

    # 4. Simulate Pub/Sub base64 push round-trip
    encoded = base64.b64encode(alert.model_dump_json().encode("utf-8")).decode("utf-8")
    decoded_alert = Alert(**json.loads(base64.b64decode(encoded).decode("utf-8")))
    assert decoded_alert.alert_id == alert.alert_id

    # 5. Chart-First Pipeline: Evidence -> Render Figures 1-5 -> Multimodal Orchestrator -> PDF
    builder = EvidenceArtifactBuilder(model, enable_live_scrape=False)
    evidence = builder.build_sealed_evidence(decoded_alert)
    assert len(evidence.historic_case_matches) == 3

    charts = ForensicDossierPDFExporter.render_labeled_charts(decoded_alert, evidence)
    assert list(charts.keys()) == ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5"]
    for fig_id, ch in charts.items():
        assert len(ch["png_bytes"]) > 5000, f"{fig_id} PNG looks empty!"
        assert isinstance(ch["stats"], dict) and len(ch["stats"]) > 0

    orchestrator = AgenticForensicOrchestrator()
    ok, _ = orchestrator._compliance_critic_verify(
        [Claim(claim_id="BAD", sentence="Hallucinated claim", evidence_refs=["FAKE_ID"])], evidence
    )
    assert ok is False

    report = orchestrator.generate_verified_report(
        decoded_alert, evidence, trigger_mode="serverless_pubsub", charts=charts
    )
    assert report.critic_verdict == "APPROVED"
    assert report.faithfulness_score == 1.0
    assert len(report.executive_summary) > 100
    assert len(report.figure_explanations) == 5
    assert all(fe.is_grounded and len(fe.explanation) > 40 for fe in report.figure_explanations)

    pdf_path = ForensicDossierPDFExporter.export_pdf(
        decoded_alert, report, output_dir="./test_dossiers", charts=charts
    )
    assert os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 60000


if __name__ == "__main__":
    test_terraform_hcl2_has_no_semicolons()
    test_v4_full_serverless_and_multimodal_figures_pipeline()
    print("ALL MULTIMODAL FIGURE-BY-FIGURE PATCH TESTS PASSED!")
