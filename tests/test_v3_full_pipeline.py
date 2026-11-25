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


def test_v3_interlaced_stream_in_graph_memory_and_rich_pdf():
    # 1. Verify realistic interlacing: licit transactions appear BETWEEN hops of the same illicit campaign
    gen = AMLSimStreamGenerator(seed=11, illicit_campaign_prob=0.0)
    gen.enqueue_interlaced_typology(Typology.PEELING_CHAIN, immediate_first_hop=True)
    stream_seq = [gen.next_event() for _ in range(16)]
    peel_indices = [i for i, e in enumerate(stream_seq) if e.ground_truth_typology == Typology.PEELING_CHAIN]
    assert len(peel_indices) >= 3, "Expected multi-hop peeling chain events"
    assert any((peel_indices[i + 1] - peel_indices[i]) > 1 for i in range(len(peel_indices) - 1)), (
        "Expected genuine licit transactions interlaced between hops of the same illicit attempt!"
    )

    # 2. Verify zero temporal leakage on a realistic training corpus
    corpus_gen = AMLSimStreamGenerator(seed=21, illicit_campaign_prob=0.08)
    nodes, edges, events = corpus_gen.generate_training_corpus(total_events=60, illicit_campaign_prob=0.08)
    ts_sorted = sorted(n.timestamp for n in nodes)
    splitter = TemporalSplitter(
        train_max_ts=ts_sorted[int(len(ts_sorted) * 0.5)],
        val_max_ts=ts_sorted[int(len(ts_sorted) * 0.75)],
        test_max_ts=ts_sorted[-1],
    )
    manifest, _ = splitter.split_and_verify(nodes, edges)
    assert manifest.leakage_verified is True

    # 3. Verify Bit-CHetG + Amatriciana + In-Graph Memory forward pass
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
            logit, _, emb, upd_mem, motif_act, motif_slot = model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx,
                enable_memory=True, enable_in_graph_memory=True
            )
        assert torch.isfinite(logit).all()
        assert 0.0 <= motif_act <= 1.0
        for i, n in enumerate(sub_nodes):
            store.memory[n.node_id] = upd_mem[i]
        if ev.ground_truth_typology == Typology.PEELING_CHAIN:
            target_sub = (ev, sub_nodes, sub_edges, motif_act, motif_slot)

    assert target_sub is not None
    ev, sub_nodes, sub_edges, motif_act, motif_slot = target_sub
    alert = Alert(
        alert_id="ALT-V3TEST",
        correlation_id=ev.correlation_id,
        timestamp=ev.timestamp,
        risk_score=0.89,
        model_version="Temporal-BitCHetG-InGraphMem-v3.0",
        in_graph_memory_activation=round(motif_act, 4),
        matched_motif_slot=motif_slot,
        predicted_typology=Typology.PEELING_CHAIN,
        cadence_signature=ev.cadence_algorithm or "lognormal_peel_jitter_v2",
        window_bounds=(sub_nodes[0].timestamp, ev.timestamp),
        subgraph_nodes=sub_nodes,
        subgraph_edges=sub_edges,
    )

    # 4. Verify GNNExplainer + OSINT Historic Cases Scraper + Compliance Critic + Multi-Chart Visual PDF
    builder = EvidenceArtifactBuilder(model, enable_live_scrape=False)
    evidence = builder.build_sealed_evidence(alert)
    assert len(evidence.historic_case_matches) >= 3
    assert evidence.historic_case_matches[0].similarity_index > 0.50
    assert "fan_out_z_deviation" in evidence.statistical_correlations

    orchestrator = AgenticForensicOrchestrator()
    # Critic must reject hallucinated claims
    ok, _ = orchestrator._compliance_critic_verify(
        [Claim(claim_id="BAD", sentence="Fake claim", evidence_refs=["NON_EXISTENT_ID"])], evidence
    )
    assert ok is False

    report = orchestrator.generate_verified_report(alert, evidence)
    assert report.critic_verdict == "APPROVED"
    assert report.faithfulness_score == 1.0
    assert any("similarity index" in c.sentence.lower() and "escalation rate" in c.sentence.lower() for c in report.claims)

    pdf_path = ForensicDossierPDFExporter.export_pdf(alert, report, output_dir="./test_dossiers")
    assert os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 40000


if __name__ == "__main__":
    test_v3_interlaced_stream_in_graph_memory_and_rich_pdf()
    print("ALL V3 TESTS PASSED (Interlaced AMLSim + In-Graph Memory + OSINT Similarity RAG + Multi-Chart PDF)!")
