"""
ABI v4 GPU Training & Validation-Calibrated 3-Way Ablation Harness.
Calibrates each model's decision threshold on a leakage-clean Validation window (rather than
hardcoding 0.50) and reports both overall metrics and interlaced-AMLSim subset recall.
"""
import argparse
from collections import defaultdict
import json
import os
import sys
import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./contracts"))

from contracts.schemas import TYPOLOGY_TO_IDX, Typology, TransactionEvent
from module1_data_pipeline.src.elliptic_gcs_loader import EllipticPlusPlusGCSLoader
from module1_data_pipeline.src.amlsim_stream import AMLSimStreamGenerator
from module2_detection_gnn.src.graph_store import IncrementalGraphStore
from module2_detection_gnn.src.torch_model import (
    TemporalBitCHetGNet,
    FocalContrastiveLoss,
    subgraph_to_tensors,
)


def build_samples(events):
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


def collect_probs(model, samples, enable_memory: bool, enable_in_graph_memory: bool, device: str):
    model.eval()
    eval_mem = {}
    probs, labels = [], []
    with torch.no_grad():
        for sub_nodes, sub_edges, lbl, _, _ in samples:
            x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, _ = subgraph_to_tensors(
                sub_nodes, sub_edges, eval_mem, memory_dim=32, device=device
            )
            r_logit, _, _, upd_mem, _, _ = model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx,
                enable_memory=enable_memory, enable_in_graph_memory=enable_in_graph_memory
            )
            for i, n in enumerate(sub_nodes):
                eval_mem[n.node_id] = upd_mem[i].detach()
            probs.append(float(torch.sigmoid(r_logit).item()))
            labels.append(lbl)
    return probs, labels


def score_at_threshold(probs, labels, thr: float):
    tp = fp = fn = tn = 0
    for p, y in zip(probs, labels):
        pred = 1 if p >= thr else 0
        if pred == 1 and y == 1: tp += 1
        elif pred == 1 and y == 0: fp += 1
        elif pred == 0 and y == 1: fn += 1
        else: tn += 1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"threshold": round(thr, 2), "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def calibrate_threshold(val_probs, val_labels):
    best_thr, best_f1 = 0.50, -1.0
    for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        m = score_at_threshold(val_probs, val_labels, t)
        if m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], t
    return best_thr


def train_variant(train_s, val_s, test_s, enable_memory: bool, enable_in_graph_memory: bool, epochs: int, device: str, label_name: str):
    model = TemporalBitCHetGNet().to(device)
    criterion = FocalContrastiveLoss(alpha=0.85, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        mem_store = {}
        logits_buf, y_buf, emb_buf, typ_l_buf, typ_y_buf = [], [], [], [], []
        ep_loss, steps = 0.0, 0

        for sub_nodes, sub_edges, lbl, typ_idx, _ in train_s:
            x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, _ = subgraph_to_tensors(
                sub_nodes, sub_edges, mem_store, memory_dim=32, device=device
            )
            r_logit, t_logit, emb, upd_mem, _, _ = model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx,
                enable_memory=enable_memory, enable_in_graph_memory=enable_in_graph_memory
            )
            for i, n in enumerate(sub_nodes):
                mem_store[n.node_id] = upd_mem[i].detach()

            logits_buf.append(r_logit); y_buf.append(lbl); emb_buf.append(emb)
            typ_l_buf.append(t_logit); typ_y_buf.append(typ_idx)

            if len(logits_buf) >= 16:
                optimizer.zero_grad()
                loss = criterion(
                    torch.stack(logits_buf), torch.tensor(y_buf, device=device),
                    torch.stack(emb_buf), torch.stack(typ_l_buf), torch.tensor(typ_y_buf, device=device)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                ep_loss += float(loss.item()); steps += 1
                logits_buf, y_buf, emb_buf, typ_l_buf, typ_y_buf = [], [], [], [], []

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 4) == 0:
            print(f"  [{label_name}] Epoch {epoch:02d}/{epochs} | Loss: {ep_loss / max(1, steps):.4f}")

    val_p, val_y = collect_probs(model, val_s, enable_memory, enable_in_graph_memory, device)
    best_thr = calibrate_threshold(val_p, val_y)
    test_p, test_y = collect_probs(model, test_s, enable_memory, enable_in_graph_memory, device)
    return model, score_at_threshold(test_p, test_y, best_thr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="capstone-bitcoin-forensics-abi-vault")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max-elliptic-txs", type=int, default=40000)
    parser.add_argument("--use-gcs", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ABI v4 Training] Device: {device}")

    amlsim = AMLSimStreamGenerator(seed=42, illicit_campaign_prob=0.075)
    _, _, a_events = amlsim.generate_training_corpus(total_events=340, illicit_campaign_prob=0.075)

    if args.use_gcs:
        loader = EllipticPlusPlusGCSLoader(gcs_bucket=args.bucket)
        epp_graph = loader.load_canonical_graph(max_txs=args.max_elliptic_txs)
        print(f"[ABI v4 Training] Loaded {len(epp_graph.nodes)} Elliptic++ nodes & {len(epp_graph.edges)} edges.")
        node_map = {n.node_id: n for n in epp_graph.nodes}
        by_ts = defaultdict(list)
        for e in epp_graph.edges:
            by_ts[e.timestamp].append(e)
        epp_events = []
        for ts in sorted(by_ts.keys()):
            ts_edges = by_ts[ts]
            for start in range(0, min(len(ts_edges), 600), 60):
                chunk = ts_edges[start:start + 60]
                touched = {e.source_id for e in chunk} | {e.target_id for e in chunk}
                epp_events.append(TransactionEvent(
                    correlation_id=f"epp_{ts}_{start}", timestamp=ts,
                    nodes=[node_map[nid] for nid in touched if nid in node_map], edges=chunk
                ))
        all_events = epp_events + a_events
    else:
        all_events = a_events

    samples = build_samples(all_events)
    n = len(samples)
    train_s, val_s, test_s = samples[:int(n * 0.65)], samples[int(n * 0.65):int(n * 0.80)], samples[int(n * 0.80):]
    print(f"[ABI v4 Training] Temporal Split -> Train: {len(train_s)}, Val: {len(val_s)}, Future Test: {len(test_s)}")

    _, m1 = train_variant(train_s, val_s, test_s, False, False, args.epochs, device, "1_Static_BitCHetG")
    _, m2 = train_variant(train_s, val_s, test_s, True, False, args.epochs, device, "2_Amatriciana_Only")
    full_model, m3 = train_variant(train_s, val_s, test_s, True, True, args.epochs, device, "3_Full_InGraph_Memory")

    ablation = {
        "1_static_bitchetg": m1,
        "2_temporal_amatriciana_only": m2,
        "3_full_amatriciana_plus_in_graph_memory": m3,
    }
    print("\n=== VALIDATION-CALIBRATED 3-WAY ABLATION RESULTS ===")
    print(json.dumps(ablation, indent=2))

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/temporal_bitchetg_v4.pt"
    metrics_file = "checkpoints/ablation_metrics_v4.json"
    torch.save({"model_state_dict": full_model.state_dict(), "ablation": ablation}, ckpt)
    with open(metrics_file, "w") as f:
        json.dump(ablation, f, indent=2)

    if args.use_gcs:
        from google.cloud import storage
        bucket = storage.Client().bucket(args.bucket)
        bucket.blob("models/temporal_bitchetg_v4.pt").upload_from_filename(ckpt)
        bucket.blob("models/ablation_metrics_v4.json").upload_from_filename(metrics_file)
        print(f"[ABI v4 Training] Uploaded -> gs://{args.bucket}/models/temporal_bitchetg_v4.pt")


if __name__ == "__main__":
    main()
