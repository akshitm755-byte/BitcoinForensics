import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from contracts.schemas import Alert, FEATURE_KEYS
from module2_detection_gnn.src.torch_model import TemporalBitCHetGNet, subgraph_to_tensors


class PyTorchGNNExplainerEngine:
    """
    Real gradient-based GNNExplainer over the PyTorch TemporalBitCHetGNet model.
    Optimizes learnable sigmoid node, edge, and feature masks to isolate the minimal
    subgraph that preserves the model's risk prediction while penalizing mask size & entropy.
    """
    EXPLAINER_VERSION = "ABI-PyTorch-GNNExplainer-v2.0"

    def __init__(self, model: TemporalBitCHetGNet, steps: int = 30, lr: float = 0.15):
        self.model = model
        self.steps = steps
        self.lr = lr

    def explain_alert(
        self, alert: Alert, mask_threshold: float = 0.52
    ) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, float], Dict[str, float]]:
        nodes = alert.subgraph_nodes
        edges = alert.subgraph_edges
        x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, valid_edge_ids = subgraph_to_tensors(
            nodes, edges, memory_store=None, memory_dim=self.model.hidden_dim
        )

        num_nodes = len(nodes)
        num_edges = len(valid_edge_ids)
        num_feats = len(FEATURE_KEYS)

        # Freeze GNN weights during mask optimization
        for p in self.model.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            orig_logit, _, _, _ = self.model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, enable_memory=True
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

            masked_logit, _, _, _ = self.model.forward_tensors(
                x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx,
                enable_memory=True, edge_mask=e_mask, node_mask=n_mask, feat_mask=f_mask
            )
            pred_loss = F.binary_cross_entropy_with_logits(masked_logit, target_prob)
            size_reg = 0.04 * n_mask.mean() + 0.04 * f_mask.mean()
            if e_mask is not None:
                size_reg = size_reg + 0.04 * e_mask.mean()

            # Entropy encourages crisp binary masks (important vs unimportant)
            ent_n = -(n_mask * torch.log(n_mask + 1e-6) + (1 - n_mask) * torch.log(1 - n_mask + 1e-6)).mean()
            loss = pred_loss + size_reg + 0.02 * ent_n
            loss.backward()
            opt.step()

        with torch.no_grad():
            n_scores = torch.sigmoid(node_logits).tolist()
            e_scores = torch.sigmoid(edge_logits[:num_edges]).tolist() if num_edges > 0 else []
            f_raw = torch.sigmoid(feat_logits)
            # Combine learned mask with feature variance salience
            f_sal = (f_raw * (x.abs().mean(dim=0) + 0.15))
            f_norm = (f_sal / (f_sal.sum() + 1e-8)).tolist()

        node_mask_dict = {n.node_id: round(float(s), 4) for n, s in zip(nodes, n_scores)}
        edge_mask_dict = {eid: round(float(s), 4) for eid, s in zip(valid_edge_ids, e_scores)}
        feat_attr_dict = {f"feat:{k}": round(float(v), 4) for k, v in zip(FEATURE_KEYS, f_norm)}

        minimal_nodes = [nid for nid, w in sorted(node_mask_dict.items(), key=lambda kv: kv[1], reverse=True) if w >= mask_threshold]
        if len(minimal_nodes) < min(3, len(nodes)):
            minimal_nodes = [nid for nid, _ in sorted(node_mask_dict.items(), key=lambda kv: kv[1], reverse=True)[:min(4, len(nodes))]]

        minimal_edges = [eid for eid, w in sorted(edge_mask_dict.items(), key=lambda kv: kv[1], reverse=True) if w >= mask_threshold]
        if len(minimal_edges) < min(2, len(valid_edge_ids)):
            minimal_edges = [eid for eid, _ in sorted(edge_mask_dict.items(), key=lambda kv: kv[1], reverse=True)[:min(4, len(valid_edge_ids))]]

        return minimal_nodes, minimal_edges, node_mask_dict, edge_mask_dict, feat_attr_dict
