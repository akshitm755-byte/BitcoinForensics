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


def test_leakage_clean_split_and_pytorch_end_to_end():
    # 1. Module 1: Generate AMLSim stream with peeling chain & verify zero temporal leakage
    gen = AMLSimStreamGenerator(seed=7)
    nodes, edges, events = gen.generate_training_corpus(num_licit=18, num_illicit_clusters=4)
    ts_sorted = sorted(n.timestamp for n in nodes)
    t_train = ts_sorted[int(len(ts_sorted) * 0.5)]
    t_val = ts_sorted[int(len(ts_sorted) * 0.75)]
    splitter = TemporalSplitter(train_max_ts=t_train, val_max_ts=t_val, test_max_ts=ts_sorted[-1])
    manifest, _ = splitter.split_and_verify(nodes, edges)
    assert manifest.leakage_verified is True

    # 2. Module 2: IncrementalGraphStore + Real PyTorch TemporalBitCHetGNet forward pass
    store = IncrementalGraphStore(memory_dim=32)
    model = TemporalBitCHetGNet()
    model.eval()

    last_sub = None
    for ev in events:
        sub_nodes, sub_edges = store.apply_event(ev)
        x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, _ = subgraph_to_tensors(
            sub_nodes, sub_edges, store.memory, memory_dim=32
        )
        with torch.no_grad():
            logit, typ_logits, emb, upd_mem = model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, enable_memory=True
            )
        assert torch.isfinite(logit).all()
        assert emb.shape[-1] == 32
        for i, n in enumerate(sub_nodes):
            store.memory[n.node_id] = upd_mem[i]
        if ev.ground_truth_typology == Typology.PEELING_CHAIN:
            last_sub = (ev, sub_nodes, sub_edges, float(torch.sigmoid(logit).item()))

    assert last_sub is not None
    ev, sub_nodes, sub_edges, score = last_sub
    alert = Alert(
        alert_id="ALT-TEST01",
        correlation_id=ev.correlation_id,
        timestamp=ev.timestamp,
        risk_score=max(0.75, score),
        model_version="Temporal-BitCHetG-PyTorch-v2.0",
        predicted_typology=Typology.PEELING_CHAIN,
        window_bounds=(sub_nodes[0].timestamp, ev.timestamp),
        subgraph_nodes=sub_nodes,
        subgraph_edges=sub_edges,
    )

    # 3. Module 3: Real PyTorch GNNExplainer + Compliance Critic Rejection Loop + Visual PDF Dossier
    builder = EvidenceArtifactBuilder(model)
    evidence = builder.build_sealed_evidence(alert)
    assert len(evidence.minimal_nodes) >= 1
    assert len(evidence.feature_attributions) == 7

    orchestrator = AgenticForensicOrchestrator()
    # Test that Compliance Critic catches a hallucinated citation
    bad_claims = [Claim(claim_id="BAD", sentence="Hallucinated wallet.", evidence_refs=["fake_ref_999"])]
    ok, checked = orchestrator._compliance_critic_verify(bad_claims, evidence)
    assert ok is False and checked[0].is_grounded is False

    report = orchestrator.generate_verified_report(alert, evidence)
    assert report.critic_verdict == "APPROVED"
    assert report.faithfulness_score == 1.0

    pdf_path = ForensicDossierPDFExporter.export_pdf(alert, report, output_dir="./test_dossiers")
    assert os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 5000


if __name__ == "__main__":
    test_leakage_clean_split_and_pytorch_end_to_end()
    print("ALL PYTORCH, LEAKAGE, GNNEXPLAINER, CRITIC & VISUAL PDF CHECKS PASSED!")
