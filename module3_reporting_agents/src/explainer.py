import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from contracts.schemas import Alert, FEATURE_KEYS, LICIT_FEATURE_BASELINES
from module2_detection_gnn.src.torch_model import TemporalBitCHetGNet, subgraph_to_tensors


class PyTorchGNNExplainerEngine:
    EXPLAINER_VERSION = "ABI-PyTorch-GNNExplainer-v4.0"

    def __init__(self, model: TemporalBitCHetGNet, steps: int = 28, lr: float = 0.15):
        self.model = model
        self.steps = steps
        self.lr = lr

    def explain_alert(
        self, alert: Alert, mask_threshold: float = 0.52
    ) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        nodes, edges = alert.subgraph_nodes, alert.subgraph_edges
        x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, valid_edge_ids = subgraph_to_tensors(
            nodes, edges, memory_store=None, memory_dim=self.model.hidden_dim
        )
        num_nodes, num_edges, num_feats = len(nodes), len(valid_edge_ids), len(FEATURE_KEYS)

        for p in self.model.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            orig_logit, _, _, _, _, _ = self.model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx,
                enable_memory=True, enable_in_graph_memory=True
            )
            target_prob = torch.sigmoid(orig_logit).detach()

        node_logits = torch.nn.Parameter(torch.zeros(num_nodes))
        edge_logits = torch.nn.Parameter(torch.zeros(max(1, num_edges)))
        feat_logits = torch.nn.Parameter(torch.zeros(num_feats))
        opt = torch.optim.Adam([node_logits, edge_logits, feat_logits], lr=self.lr)

        for _ in range(self.steps):
            opt.zero_grad()
            n_mask = torch.sigmoid(node_logits)
            e_mask = torch.sigmoid(edge_logits[:num_edges]) if num_edges > 0 else None
            f_mask = torch.sigmoid(feat_logits)

            masked_logit, _, _, _, _, _ = self.model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx,
                enable_memory=True, enable_in_graph_memory=True,
                edge_mask=e_mask, node_mask=n_mask, feat_mask=f_mask
            )
            pred_loss = F.binary_cross_entropy_with_logits(masked_logit, target_prob)
            size_reg = 0.04 * (n_mask.mean() + f_mask.mean() + (e_mask.mean() if e_mask is not None else 0.0))
            ent = -(n_mask * torch.log(n_mask + 1e-6) + (1 - n_mask) * torch.log(1 - n_mask + 1e-6)).mean()
            (pred_loss + size_reg + 0.02 * ent).backward()
            opt.step()

        with torch.no_grad():
            n_scores = torch.sigmoid(node_logits).tolist()
            e_scores = torch.sigmoid(edge_logits[:num_edges]).tolist() if num_edges > 0 else []
            f_raw = torch.sigmoid(feat_logits)
            f_sal = f_raw * (x.abs().mean(dim=0) + 0.15)
            f_norm = (f_sal / (f_sal.sum() + 1e-8)).tolist()

        node_mask = {n.node_id: round(float(s), 4) for n, s in zip(nodes, n_scores)}
        edge_mask = {eid: round(float(s), 4) for eid, s in zip(valid_edge_ids, e_scores)}
        feat_attr = {f"feat:{k}": round(float(v), 4) for k, v in zip(FEATURE_KEYS, f_norm)}

        minimal_nodes = [nid for nid, w in sorted(node_mask.items(), key=lambda kv: kv[1], reverse=True) if w >= mask_threshold]
        if len(minimal_nodes) < min(3, len(nodes)):
            minimal_nodes = [nid for nid, _ in sorted(node_mask.items(), key=lambda kv: kv[1], reverse=True)[:min(4, len(nodes))]]

        minimal_edges = [eid for eid, w in sorted(edge_mask.items(), key=lambda kv: kv[1], reverse=True) if w >= mask_threshold]
        if len(minimal_edges) < min(2, len(valid_edge_ids)):
            minimal_edges = [eid for eid, _ in sorted(edge_mask.items(), key=lambda kv: kv[1], reverse=True)[:min(4, len(valid_edge_ids))]]

        mat = x.numpy() if num_nodes > 0 else np.zeros((1, num_feats))
        spacings = [e.temporal_spacing_sec for e in edges if e.temporal_spacing_sec > 0]
        values = [e.value for e in edges]
        stats = {
            "mean_fan_out_ratio": round(float(mat[:, 5].mean()), 4),
            "fan_out_z_deviation": round(float((mat[:, 5].mean() - LICIT_FEATURE_BASELINES["fan_out_ratio"]) / 0.35), 2),
            "median_inter_hop_sec": round(float(np.median(spacings)) if spacings else 18.0, 2),
            "spacing_cv": round(float(np.std(spacings) / (np.mean(spacings) + 1e-5)) if len(spacings) > 1 else 0.24, 3),
            "total_cluster_btc": round(float(sum(values)), 4),
            "in_graph_memory_activation": round(float(alert.in_graph_memory_activation), 4),
        }
        return minimal_nodes, minimal_edges, node_mask, edge_mask, feat_attr, stats
