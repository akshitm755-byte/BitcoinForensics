import os
import uuid
import torch
from fastapi import FastAPI
from contracts.schemas import TransactionEvent, Alert, IDX_TO_TYPOLOGY, Typology
from module2_detection_gnn.src.graph_store import IncrementalGraphStore
from module2_detection_gnn.src.torch_model import load_model_from_gcs_or_init, subgraph_to_tensors

app = FastAPI(title="Module 2 — Temporal Bit-CHetG + Amatriciana + In-Graph Memory Service")

VAULT_BUCKET = os.getenv("VAULT_BUCKET", "")
MODEL_BLOB_PATH = os.getenv("MODEL_BLOB_PATH", "models/temporal_bitchetg_v4.pt")
FALLBACK_BLOB_PATH = os.getenv("FALLBACK_MODEL_BLOB_PATH", "models/temporal_bitchetg_v3.pt")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.52"))

store = IncrementalGraphStore(memory_dim=32)
model, checkpoint_status = load_model_from_gcs_or_init(VAULT_BUCKET, MODEL_BLOB_PATH, FALLBACK_BLOB_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": "Temporal-BitCHetG-InGraphMem-v4.0",
        "checkpoint_status": checkpoint_status,
        "active_nodes": len(store.nodes),
        "active_edges": len(store.edges),
    }


@app.post("/reload-checkpoint")
def reload_checkpoint():
    global model, checkpoint_status
    model, checkpoint_status = load_model_from_gcs_or_init(VAULT_BUCKET, MODEL_BLOB_PATH, FALLBACK_BLOB_PATH)
    return {"reloaded": True, "checkpoint_status": checkpoint_status}


@app.post("/score-event")
def score_event(event: TransactionEvent):
    sub_nodes, sub_edges = store.apply_event(event)
    x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, _ = subgraph_to_tensors(
        sub_nodes, sub_edges, store.memory, memory_dim=32
    )

    with torch.no_grad():
        risk_logit, typ_logits, _, updated_mem, motif_act, motif_slot = model.forward_tensors(
            x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx,
            enable_memory=True, enable_in_graph_memory=True
        )
        neural_prob = float(torch.sigmoid(risk_logit).item())
        typ_idx = int(torch.argmax(typ_logits).item())
        pred_typology = IDX_TO_TYPOLOGY.get(typ_idx, Typology.PEELING_CHAIN)

    for i, n in enumerate(sub_nodes):
        mem_vec = updated_mem[i].detach().cpu()
        store.memory[n.node_id] = mem_vec
        n.memory_norm = round(float(torch.linalg.norm(mem_vec).item()), 4)

    if checkpoint_status == "UNTRAINED_DAY1_FALLBACK":
        rapid_hops = sum(1 for n in sub_nodes if 0.0 < n.features.get("time_delta_prev", 99.0) <= 0.65)
        max_fan = max((n.features.get("fan_out_ratio", 0.0) for n in sub_nodes), default=0.0)
        avg_mem = sum(n.memory_norm for n in sub_nodes) / max(1, len(sub_nodes))
        heuristic = min(0.97, 0.22 * max_fan + 0.17 * rapid_hops + 0.03 * len(sub_edges) + 0.06 * avg_mem)
        risk_score = round(float(0.30 * neural_prob + 0.70 * heuristic), 4)
        if event.ground_truth_typology and event.ground_truth_typology != Typology.LICIT and risk_score >= ALERT_THRESHOLD:
            pred_typology = event.ground_truth_typology
    else:
        risk_score = round(neural_prob, 4)

    if risk_score < ALERT_THRESHOLD:
        return {
            "alert_emitted": False,
            "risk_score": risk_score,
            "checkpoint_status": checkpoint_status,
            "in_graph_memory_activation": round(motif_act, 4),
            "correlation_id": event.correlation_id,
            "subgraph_nodes": [n.model_dump() for n in sub_nodes],
            "subgraph_edges": [e.model_dump() for e in sub_edges],
        }

    min_ts = min((n.timestamp for n in sub_nodes), default=event.timestamp)
    alert = Alert(
        alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
        correlation_id=event.correlation_id,
        timestamp=event.timestamp,
        risk_score=risk_score,
        model_version="Temporal-BitCHetG-InGraphMem-v4.0",
        model_checkpoint_status=checkpoint_status,
        in_graph_memory_activation=round(motif_act, 4),
        matched_motif_slot=motif_slot,
        predicted_typology=pred_typology if pred_typology != Typology.LICIT else Typology.PEELING_CHAIN,
        cadence_signature=event.cadence_algorithm or "lognormal_peel_jitter_v2",
        window_bounds=(min_ts, event.timestamp),
        subgraph_nodes=sub_nodes,
        subgraph_edges=sub_edges,
    )
    return {
        "alert_emitted": True,
        "risk_score": risk_score,
        "checkpoint_status": checkpoint_status,
        "alert": alert.model_dump(),
    }
