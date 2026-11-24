import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from contracts.schemas import Node, Edge, NodeType, Typology, IDX_TO_TYPOLOGY, FEATURE_KEYS
from module1_data_pipeline.src.features import ExplicitFeatureBuilder


class HeteroSAGEBlock(nn.Module):
    """
    Pure-PyTorch heterogeneous SAGE message-passing block (Bit-CHetG style).
    Separately aggregates Transaction-Subgraph Embeddings (TSE) and Address-Feature Aggregations (AFA)
    with optional differentiable edge & node masks for PyTorch GNNExplainer.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin_self_tx = nn.Linear(hidden_dim, hidden_dim)
        self.lin_nbr_tx = nn.Linear(hidden_dim, hidden_dim)
        self.lin_self_addr = nn.Linear(hidden_dim, hidden_dim)
        self.lin_nbr_addr = nn.Linear(hidden_dim, hidden_dim)
        self.norm_tx = nn.LayerNorm(hidden_dim)
        self.norm_addr = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        node_is_tx: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_nodes, d = h.shape
        device = h.device

        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            w = edge_weight if edge_weight is not None else torch.ones(src.shape[0], device=device)
            msg = h[src] * w.unsqueeze(-1)
            agg = torch.zeros((num_nodes, d), device=device, dtype=h.dtype)
            deg = torch.zeros((num_nodes, 1), device=device, dtype=h.dtype)
            agg.index_add_(0, dst, msg)
            deg.index_add_(0, dst, w.unsqueeze(-1))
            agg = agg / (deg + 1e-6)
        else:
            agg = torch.zeros_like(h)

        h_tx = self.norm_tx(F.gelu(self.lin_self_tx(h) + self.lin_nbr_tx(agg)))
        h_addr = self.norm_addr(F.gelu(self.lin_self_addr(h) + self.lin_nbr_addr(agg)))

        mask = node_is_tx.unsqueeze(-1).float()
        return mask * h_tx + (1.0 - mask) * h_addr


class AmatricianaMemoryCell(nn.Module):
    """
    Time-decay + sequence-ordered GRU memory encoder (Amatriciana style).
    Applies continuous exponential decay exp(-lambda * dt) before updating node state,
    preserving rapid sequential patterns (peeling chains, cyclic flows) while fading stale history.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.raw_decay = nn.Parameter(torch.tensor(0.25))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.seq_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        h_nodes: torch.Tensor,
        time_deltas: torch.Tensor,
        prev_memories: torch.Tensor,
        order_idx: torch.Tensor,
        enable_memory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not enable_memory or h_nodes.shape[0] == 0:
            return torch.zeros(h_nodes.shape[-1], device=h_nodes.device), prev_memories

        decay_rate = F.softplus(self.raw_decay)
        decay = torch.exp(-decay_rate * torch.clamp(time_deltas, min=0.0, max=60.0)).unsqueeze(-1)
        decayed_mem = prev_memories * decay
        updated_mem = self.gru(h_nodes, decayed_mem)

        # Sequential order-sensitive pooling across chronological node arrivals
        seq_state = torch.zeros((1, h_nodes.shape[-1]), device=h_nodes.device)
        for idx in order_idx:
            seq_state = self.seq_gru(updated_mem[idx].unsqueeze(0), seq_state)
        return seq_state.squeeze(0), updated_mem


class TemporalBitCHetGNet(nn.Module):
    """
    Full Hybrid Deep Learning Architecture:
      - Bit-CHetG heterogeneous encoders (TSE + AFA)
      - Amatriciana continuous time-decay GRU temporal memory
      - Supervised Contrastive projection head
      - Risk logit + 5-class Typology classification head
    """
    def __init__(self, in_dim: int = len(FEATURE_KEYS), hidden_dim: int = 32, num_typologies: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.tx_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.addr_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        self.conv1 = HeteroSAGEBlock(hidden_dim)
        self.conv2 = HeteroSAGEBlock(hidden_dim)
        self.memory_cell = AmatricianaMemoryCell(hidden_dim)

        fused_dim = hidden_dim * 3  # [TSE_pool || AFA_pool || Temporal_Memory]
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.contrastive_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.risk_head = nn.Linear(hidden_dim * 2, 1)
        self.typology_head = nn.Linear(hidden_dim * 2, num_typologies)

    def forward_tensors(
        self,
        x: torch.Tensor,
        node_is_tx: torch.Tensor,
        edge_index: torch.Tensor,
        time_deltas: torch.Tensor,
        prev_memories: torch.Tensor,
        order_idx: torch.Tensor,
        enable_memory: bool = True,
        edge_mask: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        feat_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if feat_mask is not None:
            x = x * feat_mask.unsqueeze(0)
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1)

        m_tx = node_is_tx.unsqueeze(-1).float()
        h = m_tx * self.tx_proj(x) + (1.0 - m_tx) * self.addr_proj(x)

        h = self.conv1(h, node_is_tx, edge_index, edge_weight=edge_mask)
        h = self.conv2(h, node_is_tx, edge_index, edge_weight=edge_mask)

        tx_mask = node_is_tx.bool()
        addr_mask = ~tx_mask
        tse_pool = h[tx_mask].mean(dim=0) if tx_mask.any() else torch.zeros(self.hidden_dim, device=x.device)
        afa_pool = h[addr_mask].mean(dim=0) if addr_mask.any() else torch.zeros(self.hidden_dim, device=x.device)

        mem_seq, updated_mem = self.memory_cell(
            h, time_deltas, prev_memories, order_idx, enable_memory=enable_memory
        )

        fused = self.fusion(torch.cat([tse_pool, afa_pool, mem_seq], dim=-1))
        contrastive_emb = F.normalize(self.contrastive_proj(fused), dim=-1)
        risk_logit = self.risk_head(fused).squeeze(-1)
        typology_logits = self.typology_head(fused)
        return risk_logit, typology_logits, contrastive_emb, updated_mem


def subgraph_to_tensors(
    nodes: List[Node],
    edges: List[Edge],
    memory_store: Optional[Dict[str, torch.Tensor]] = None,
    memory_dim: int = 32,
    device: str = "cpu",
):
    id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}
    x_list = [ExplicitFeatureBuilder.to_vector(n.features) for n in nodes]
    x = torch.tensor(x_list, dtype=torch.float32, device=device) if x_list else torch.zeros((0, len(FEATURE_KEYS)), device=device)
    node_is_tx = torch.tensor([1 if n.node_type == NodeType.TRANSACTION else 0 for n in nodes], dtype=torch.long, device=device)
    time_deltas = torch.tensor([float(n.features.get("time_delta_prev", 0.0)) for n in nodes], dtype=torch.float32, device=device)

    src_idx, dst_idx = [], []
    valid_edge_ids = []
    for e in edges:
        if e.source_id in id_to_idx and e.target_id in id_to_idx:
            src_idx.append(id_to_idx[e.source_id])
            dst_idx.append(id_to_idx[e.target_id])
            valid_edge_ids.append(e.edge_id)
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long, device=device) if src_idx else torch.zeros((2, 0), dtype=torch.long, device=device)

    if memory_store is not None and nodes:
        prev_mem = torch.stack([
            memory_store.get(n.node_id, torch.zeros(memory_dim, dtype=torch.float32)).to(device)
            for n in nodes
        ], dim=0)
    else:
        prev_mem = torch.zeros((len(nodes), memory_dim), dtype=torch.float32, device=device)

    order_idx = torch.tensor(
        sorted(range(len(nodes)), key=lambda i: nodes[i].timestamp),
        dtype=torch.long,
        device=device,
    )
    return x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, valid_edge_ids


class FocalContrastiveLoss(nn.Module):
    """Focal Loss (for ~2% illicit class imbalance) + Supervised Contrastive Loss."""
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0, temperature: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature

    def forward(
        self,
        risk_logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
        typ_logits: Optional[torch.Tensor] = None,
        typ_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(risk_logits, labels.float(), reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * labels.float() + (1.0 - self.alpha) * (1.0 - labels.float())
        focal = (alpha_t * ((1.0 - pt) ** self.gamma) * bce).mean()

        # Supervised contrastive term over batch embeddings
        con_loss = torch.tensor(0.0, device=risk_logits.device)
        if embeddings.shape[0] > 2:
            sim = torch.matmul(embeddings, embeddings.T) / self.temperature
            sim = sim - sim.max(dim=1, keepdim=True).values.detach()
            eye = torch.eye(sim.shape[0], device=sim.device)
            same_cls = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * (1.0 - eye)
            exp_sim = torch.exp(sim) * (1.0 - eye)
            log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
            pos_cnt = same_cls.sum(dim=1)
            valid = pos_cnt > 0
            if valid.any():
                con_loss = -((same_cls * log_prob).sum(dim=1)[valid] / pos_cnt[valid]).mean()

        typ_loss = torch.tensor(0.0, device=risk_logits.device)
        if typ_logits is not None and typ_targets is not None:
            typ_loss = F.cross_entropy(typ_logits, typ_targets)

        return focal + 0.25 * con_loss + 0.35 * typ_loss


def load_model_from_gcs_or_init(
    vault_bucket: Optional[str] = None,
    blob_path: str = "models/temporal_bitchetg.pt",
    local_cache: str = "/tmp/temporal_bitchetg.pt",
) -> Tuple[TemporalBitCHetGNet, str]:
    """
    Automatically loads trained weights from GCS if available (after GPU training).
    Otherwise boots cleanly with calibrated initial weights so Cloud Run works on Day 1.
    """
    model = TemporalBitCHetGNet()
    if vault_bucket:
        try:
            from google.cloud import storage
            client = storage.Client()
            blob = client.bucket(vault_bucket).blob(blob_path)
            if blob.exists():
                blob.download_to_filename(local_cache)
                state = torch.load(local_cache, map_location="cpu")
                model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
                model.eval()
                return model, "TRAINED_GCS_CHECKPOINT"
        except Exception as exc:
            print(f"[ModelLoader] GCS checkpoint not loaded yet ({exc}); using fallback init.")

    if os.path.exists(local_cache):
        state = torch.load(local_cache, map_location="cpu")
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        model.eval()
        return model, "TRAINED_LOCAL_CHECKPOINT"

    model.eval()
    return model, "UNTRAINED_DAY1_FALLBACK"
