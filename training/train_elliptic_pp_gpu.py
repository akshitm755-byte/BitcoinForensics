"""
ABI v3 GPU Training & 3-Way Ablation Harness (Colab T4 / Vertex AI Workbench).
Compares:
  1. Static Bit-CHetG (No Temporal Memory, No In-Graph Memory)
  2. Temporal Bit-CHetG + Amatriciana (Node Time-Decay GRU only)
  3. Full Hybrid: Temporal Bit-CHetG + Amatriciana + In-Graph Memory Bank
Trains on real Elliptic++ CSVs from GCS fused with interlaced AMLSim multi-hop campaigns,
and uploads `models/temporal_bitchetg_v3.pt` + `models/ablation_metrics_v3.json` back to GCS!
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


def train_variant(train_samples, test_samples, enable_memory: bool, enable_in_graph_memory: bool, epochs: int, device: str, label_name: str):
    model = TemporalBitCHetGNet().to(device)
    criterion = FocalContrastiveLoss(alpha=0.85, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        mem_store = {}
        logits_buf, y_buf, emb_buf, typ_l_buf, typ_y_buf = [], [], [], [], []
        ep_loss, steps = 0.0, 0

        for sub_nodes, sub_edges, lbl, typ_idx, _ in train_samples:
            x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, _ = subgraph_to_tensors(
                sub_nodes, sub_edges, mem_store, memory_dim=32, device=device
            )
            r_logit, t_logit, emb, upd_mem, _, _ = model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx,
                enable_memory=enable_memory, enable_in_graph_memory=enable_in_graph_memory
            )
            for i, n in enumerate(sub_nodes):
                mem_store[n.node_id] = upd_mem[i].detach()

            logits_buf.append(r_logit)
            y_buf.append(lbl)
            emb_buf.append(emb)
            typ_l_buf.append(t_logit)
            typ_y_buf.append(typ_idx)

            if len(logits_buf) >= 16:
                optimizer.zero_grad()
                loss = criterion(
                    torch.stack(logits_buf),
                    torch.tensor(y_buf, device=device),
                    torch.stack(emb_buf),
                    torch.stack(typ_l_buf),
                    torch.tensor(typ_y_buf, device=device),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                ep_loss += float(loss.item())
                steps += 1
                logits_buf, y_buf, emb_buf, typ_l_buf, typ_y_buf = [], [], [], [], []

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 4) == 0:
            print(f"  [{label_name}] Epoch {epoch:02d}/{epochs} | Loss: {ep_loss / max(1, steps):.4f}")

    model.eval()
    eval_mem = {}
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for sub_nodes, sub_edges, lbl, _, _ in test_samples:
            x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, _ = subgraph_to_tensors(
                sub_nodes, sub_edges, eval_mem, memory_dim=32, device=device
            )
            r_logit, _, _, upd_mem, _, _ = model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx,
                enable_memory=enable_memory, enable_in_graph_memory=enable_in_graph_memory
            )
            for i, n in enumerate(sub_nodes):
                eval_mem[n.node_id] = upd_mem[i].detach()
            pred = 1 if torch.sigmoid(r_logit).item() >= 0.50 else 0
            if pred == 1 and lbl == 1: tp += 1
            elif pred == 1 and lbl == 0: fp += 1
            elif pred == 0 and lbl == 1: fn += 1
            else: tn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return model, {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="capstone-bitcoin-forensics-abi-vault")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-elliptic-txs", type=int, default=35000)
    parser.add_argument("--use-gcs", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ABI v3 Training] Device: {device}")

    amlsim = AMLSimStreamGenerator(seed=42, illicit_campaign_prob=0.075)
    _, _, a_events = amlsim.generate_training_corpus(total_events=340, illicit_campaign_prob=0.075)

    if args.use_gcs:
        loader = EllipticPlusPlusGCSLoader(gcs_bucket=args.bucket)
        epp_graph = loader.load_canonical_graph(max_txs=args.max_elliptic_txs)
        print(f"[ABI v3 Training] Loaded {len(epp_graph.nodes)} Elliptic++ nodes & {len(epp_graph.edges)} edges.")
        node_map = {n.node_id: n for n in epp_graph.nodes}
        by_ts = defaultdict(list)
        for e in epp_graph.edges:
            by_ts[e.timestamp].append(e)
        epp_events = []
        for ts in sorted(by_ts.keys()):
            # Create multiple micro-batches per Elliptic timestep preserving chronological ordering
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
    split_idx = int(len(samples) * 0.75)
    train_s, test_s = samples[:split_idx], samples[split_idx:]
    print(f"[ABI v3 Training] Temporal Split -> Train: {len(train_s)}, Future Test: {len(test_s)}")

    _, m_static = train_variant(train_s, test_s, False, False, args.epochs, device, "1_Static_BitCHetG")
    _, m_amatriciana = train_variant(train_s, test_s, True, False, args.epochs, device, "2_Amatriciana_Only")
    full_model, m_full = train_variant(train_s, test_s, True, True, args.epochs, device, "3_Full_InGraph_Memory")

    ablation = {
        "1_static_bitchetg": m_static,
        "2_temporal_amatriciana_only": m_amatriciana,
        "3_full_amatriciana_plus_in_graph_memory": m_full,
    }
    print("\n=== 3-WAY ABLATION RESULTS (LEAKAGE-CLEAN TEMPORAL SPLIT) ===")
    print(json.dumps(ablation, indent=2))

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/temporal_bitchetg_v3.pt"
    metrics_file = "checkpoints/ablation_metrics_v3.json"
    torch.save({"model_state_dict": full_model.state_dict(), "ablation": ablation}, ckpt)
    with open(metrics_file, "w") as f:
        json.dump(ablation, f, indent=2)

    if args.use_gcs:
        from google.cloud import storage
        bucket = storage.Client().bucket(args.bucket)
        bucket.blob("models/temporal_bitchetg_v3.pt").upload_from_filename(ckpt)
        bucket.blob("models/ablation_metrics_v3.json").upload_from_filename(metrics_file)
        print(f"[ABI v3 Training] Uploaded checkpoint -> gs://{args.bucket}/models/temporal_bitchetg_v3.pt")


if __name__ == "__main__":
    main()
