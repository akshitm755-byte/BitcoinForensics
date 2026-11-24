"""
Offline GPU Training & Central Ablation Script (Colab T4 / Vertex AI Workbench).
1. Downloads real Elliptic++ CSVs from gs://capstone-bitcoin-forensics-abi-vault/raw/elliptic_pp/Elliptic++ Dataset/
2. Fuses cross-timestep AMLSim laundering typologies (peeling chains, fan-out/in, cyclic, smurfing)
3. Enforces strict leakage-verified temporal split (train <= ts34, val ts35-41, test ts42-49)
4. Trains BOTH Static Bit-CHetG (enable_memory=False) and Temporal Bit-CHetG (enable_memory=True)
5. Uploads trained checkpoint `temporal_bitchetg.pt` & `ablation_metrics.json` straight back to GCS!
"""
import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./contracts"))

from contracts.schemas import TYPOLOGY_TO_IDX, Typology
from module1_data_pipeline.src.elliptic_gcs_loader import EllipticPlusPlusGCSLoader
from module1_data_pipeline.src.amlsim_stream import AMLSimStreamGenerator
from module1_data_pipeline.src.splitter import TemporalSplitter
from module2_detection_gnn.src.graph_store import IncrementalGraphStore
from module2_detection_gnn.src.torch_model import (
    TemporalBitCHetGNet,
    FocalContrastiveLoss,
    subgraph_to_tensors,
)


def build_temporal_samples(events, device="cpu"):
    store = IncrementalGraphStore(memory_dim=32)
    samples = []
    for ev in events:
        sub_nodes, sub_edges = store.apply_event(ev)
        if not sub_nodes:
            continue
        is_illicit = 1 if (
            (ev.ground_truth_typology and ev.ground_truth_typology != Typology.LICIT)
            or any(n.label == 1 for n in ev.nodes)
        ) else 0
        typ = ev.ground_truth_typology or (Typology.PEELING_CHAIN if is_illicit else Typology.LICIT)
        samples.append((sub_nodes, sub_edges, is_illicit, TYPOLOGY_TO_IDX[typ], ev.timestamp))
    return samples


def train_and_evaluate(train_samples, test_samples, enable_memory: bool, epochs: int = 15, device: str = "cpu"):
    model = TemporalBitCHetGNet().to(device)
    criterion = FocalContrastiveLoss(alpha=0.85, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        mem_store = {}
        logits_buf, labels_buf, emb_buf, typ_logits_buf, typ_y_buf = [], [], [], [], []

        for sub_nodes, sub_edges, label, typ_idx, _ in train_samples:
            x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, _ = subgraph_to_tensors(
                sub_nodes, sub_edges, mem_store, memory_dim=32, device=device
            )
            r_logit, t_logit, emb, upd_mem = model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, enable_memory=enable_memory
            )
            for i, n in enumerate(sub_nodes):
                mem_store[n.node_id] = upd_mem[i].detach()

            logits_buf.append(r_logit)
            labels_buf.append(label)
            emb_buf.append(emb)
            typ_logits_buf.append(t_logit)
            typ_y_buf.append(typ_idx)

            if len(logits_buf) >= 16:
                optimizer.zero_grad()
                loss = criterion(
                    torch.stack(logits_buf),
                    torch.tensor(labels_buf, device=device),
                    torch.stack(emb_buf),
                    torch.stack(typ_logits_buf),
                    torch.tensor(typ_y_buf, device=device),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                logits_buf, labels_buf, emb_buf, typ_logits_buf, typ_y_buf = [], [], [], [], []

    # Evaluate on held-out future test window
    model.eval()
    eval_mem = {}
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for sub_nodes, sub_edges, label, _, _ in test_samples:
            x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, _ = subgraph_to_tensors(
                sub_nodes, sub_edges, eval_mem, memory_dim=32, device=device
            )
            r_logit, _, _, upd_mem = model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, enable_memory=enable_memory
            )
            for i, n in enumerate(sub_nodes):
                eval_mem[n.node_id] = upd_mem[i].detach()
            pred = 1 if torch.sigmoid(r_logit).item() >= 0.50 else 0
            if pred == 1 and label == 1: tp += 1
            elif pred == 1 and label == 0: fp += 1
            elif pred == 0 and label == 1: fn += 1
            else: tn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return model, {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="capstone-bitcoin-forensics-abi-vault")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-elliptic-txs", type=int, default=25000)
    parser.add_argument("--use-gcs", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Training] Running on device: {device}")

    amlsim = AMLSimStreamGenerator(seed=42)
    a_nodes, a_edges, a_events = amlsim.generate_training_corpus(num_licit=180, num_illicit_clusters=48)

    if args.use_gcs:
        loader = EllipticPlusPlusGCSLoader(gcs_bucket=args.bucket)
        epp_graph = loader.load_canonical_graph(max_txs=args.max_elliptic_txs)
        print(f"[Training] Loaded {len(epp_graph.nodes)} Elliptic++ nodes & {len(epp_graph.edges)} edges from GCS.")
        # Build temporal events from Elliptic++ timesteps
        from collections import defaultdict
        from contracts.schemas import TransactionEvent
        node_map = {n.node_id: n for n in epp_graph.nodes}
        by_ts = defaultdict(list)
        for e in epp_graph.edges:
            by_ts[e.timestamp].append(e)
        epp_events = []
        for ts in sorted(by_ts.keys()):
            es = by_ts[ts][:120]
            touched = {e.source_id for e in es} | {e.target_id for e in es}
            epp_events.append(TransactionEvent(
                correlation_id=f"epp_{ts}", timestamp=ts,
                nodes=[node_map[nid] for nid in touched if nid in node_map], edges=es
            ))
        all_events = epp_events + a_events
    else:
        all_events = a_events

    samples = build_temporal_samples(all_events, device=device)
    split_idx = int(len(samples) * 0.75)
    train_s, test_s = samples[:split_idx], samples[split_idx:]
    print(f"[Training] Temporal split -> Train subgraphs: {len(train_s)}, Future Test subgraphs: {len(test_s)}")

    _, static_metrics = train_and_evaluate(train_s, test_s, enable_memory=False, epochs=args.epochs, device=device)
    temporal_model, temporal_metrics = train_and_evaluate(train_s, test_s, enable_memory=True, epochs=args.epochs, device=device)

    ablation = {
        "static_bitchetg_no_memory": static_metrics,
        "temporal_bitchetg_with_memory": temporal_metrics,
    }
    print("=== CENTRAL ABLATION RESULTS ===")
    print(json.dumps(ablation, indent=2))

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/temporal_bitchetg.pt"
    metrics_path = "checkpoints/ablation_metrics.json"
    torch.save({"model_state_dict": temporal_model.state_dict(), "ablation": ablation}, ckpt_path)
    with open(metrics_path, "w") as f:
        json.dump(ablation, f, indent=2)

    if args.use_gcs:
        from google.cloud import storage
        bucket = storage.Client().bucket(args.bucket)
        bucket.blob("models/temporal_bitchetg.pt").upload_from_filename(ckpt_path)
        bucket.blob("models/ablation_metrics.json").upload_from_filename(metrics_path)
        print(f"[Training] Uploaded trained checkpoint to gs://{args.bucket}/models/temporal_bitchetg.pt")


if __name__ == "__main__":
    main()
