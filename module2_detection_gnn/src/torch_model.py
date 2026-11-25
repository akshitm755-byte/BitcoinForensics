import math
import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from contracts.schemas import Node, Edge, NodeType, Typology, IDX_TO_TYPOLOGY, FEATURE_KEYS
from module1_data_pipeline.src.features import ExplicitFeatureBuilder


class HeteroMemorySAGEBlock(nn.Module):
    """
    Bit-CHetG heterogeneous SAGE block upgraded with In-Graph Neighborhood Memory Reading:
    During message passing, each node aggregates both its neighbors' current features AND
    their stored temporal memory vectors (`prev_mem`) via a learned memory gate—allowing brand-new
    transactions in an interlaced laundering campaign to inherit relational history immediately!
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin_self_tx = nn.Linear(hidden_dim, hidden_dim)
        self.lin_nbr_tx = nn.Linear(hidden_dim, hidden_dim)
        self.lin_mem_tx = nn.Linear(hidden_dim, hidden_dim)

        self.lin_self_addr = nn.Linear(hidden_dim, hidden_dim)
        self.lin_nbr_addr = nn.Linear(hidden_dim, hidden_dim)
        self.lin_mem_addr = nn.Linear(hidden_dim, hidden_dim)

        self.mem_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm_tx = nn.LayerNorm(hidden_dim)
        self.norm_addr = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        node_is_tx: torch.Tensor,
        edge_index: torch.Tensor,
        neighbor_mem: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        enable_in_graph_memory: bool = True,
    ) -> torch.Tensor:
        num_nodes, d = h.shape
        device = h.device

        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            w = edge_weight if edge_weight is not None else torch.ones(src.shape[0], device=device)
            deg = torch.zeros((num_nodes, 1), device=device, dtype=h.dtype)
            deg.index_add_(0, dst, w.unsqueeze(-1))

            agg = torch.zeros((num_nodes, d), device=device, dtype=h.dtype)
            agg.index_add_(0, dst, h[src] * w.unsqueeze(-1))
            agg = agg / (deg + 1e-6)

            if enable_in_graph_memory and neighbor_mem is not None:
                mem_agg = torch.zeros((num_nodes, d), device=device, dtype=h.dtype)
                mem_agg.index_add_(0, dst, neighbor_mem[src] * w.unsqueeze(-1))
                mem_agg = mem_agg / (deg + 1e-6)
                gate = torch.sigmoid(self.mem_gate(torch.cat([h, mem_agg], dim=-1)))
                mem_signal = gate * mem_agg
            else:
                mem_signal = torch.zeros_like(h)
        else:
            agg = torch.zeros_like(h)
            mem_signal = torch.zeros_like(h)

        h_tx = self.norm_tx(F.gelu(self.lin_self_tx(h) + self.lin_nbr_tx(agg) + self.lin_mem_tx(mem_signal)))
        h_addr = self.norm_addr(F.gelu(self.lin_self_addr(h) + self.lin_nbr_addr(agg) + self.lin_mem_addr(mem_signal)))

        mask = node_is_tx.unsqueeze(-1).float()
        return mask * h_tx + (1.0 - mask) * h_addr


class InGraphMotifMemoryBank(nn.Module):
    """
    Learnable In-Graph Structural Motif Memory Bank (K prototype slots).
    Stores recurring laundering topology signatures directly inside the GNN so subgraphs
    cross-attend against stored laundering motifs, yielding both an enriched memory vector
    and an interpretable `matched_motif_slot` + `activation_strength`.
    """
    def __init__(self, hidden_dim: int, num_slots: int = 8):
        super().__init__()
        self.num_slots = num_slots
        self.keys = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.2)
        self.values = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.2)
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, tse_pool: torch.Tensor, afa_pool: torch.Tensor, enable_in_graph_memory: bool = True
    ) -> Tuple[torch.Tensor, float, int]:
        if not enable_in_graph_memory:
            return torch.zeros_like(tse_pool), 0.0, 0

        q = self.query_proj(torch.cat([tse_pool, afa_pool], dim=-1))
        scores = torch.matmul(q, self.keys.T) / math.sqrt(q.shape[-1])
        attn = F.softmax(scores, dim=-1)
        readout = self.out_norm(torch.matmul(attn, self.values))
        top_prob, top_slot = torch.max(attn, dim=-1)
        return readout, float(top_prob.item()), int(top_slot.item())


class AmatricianaMemoryCell(nn.Module):
    """Continuous exponential time-decay + chronological GRU sequence memory (Amatriciana style)."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.raw_decay = nn.Parameter(torch.tensor(0.22))
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
        updated_mem = self.gru(h_nodes, prev_memories * decay)

        seq_state = torch.zeros((1, h_nodes.shape[-1]), device=h_nodes.device)
        for idx in order_idx:
            seq_state = self.seq_gru(updated_mem[idx].unsqueeze(0), seq_state)
        return seq_state.squeeze(0), updated_mem


class TemporalBitCHetGNet(nn.Module):
    """
    v3 Full Hybrid Deep Learning Architecture:
      1. Bit-CHetG Heterogeneous Encoders (TSE + AFA)
      2. Amatriciana Time-Decay Sequential GRU Memory
      3. In-Graph Memory (Neighborhood Memory Reading + Motif Memory Bank)
      4. Supervised Contrastive + Risk + 5-Class Typology Head
    Supports 3-way ablation toggles (`enable_memory`, `enable_in_graph_memory`).
    """
    def __init__(self, in_dim: int = len(FEATURE_KEYS), hidden_dim: int = 32, num_typologies: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.tx_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.addr_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        self.conv1 = HeteroMemorySAGEBlock(hidden_dim)
        self.conv2 = HeteroMemorySAGEBlock(hidden_dim)
        self.memory_cell = AmatricianaMemoryCell(hidden_dim)
        self.motif_bank = InGraphMotifMemoryBank(hidden_dim, num_slots=8)

        # [TSE || AFA || Amatriciana_Seq_Memory || InGraph_Motif_Memory]
        fused_dim = hidden_dim * 4
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
        enable_in_graph_memory: bool = True,
        edge_mask: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        feat_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, int]:
        if feat_mask is not None:
            x = x * feat_mask.unsqueeze(0)
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1)

        m_tx = node_is_tx.unsqueeze(-1).float()
        h = m_tx * self.tx_proj(x) + (1.0 - m_tx) * self.addr_proj(x)

        use_ig_mem = enable_memory and enable_in_graph_memory
        h = self.conv1(h, node_is_tx, edge_index, neighbor_mem=prev_memories, edge_weight=edge_mask, enable_in_graph_memory=use_ig_mem)
        h = self.conv2(h, node_is_tx, edge_index, neighbor_mem=prev_memories, edge_weight=edge_mask, enable_in_graph_memory=use_ig_mem)

        tx_mask = node_is_tx.bool()
        addr_mask = ~tx_mask
        tse_pool = h[tx_mask].mean(dim=0) if tx_mask.any() else torch.zeros(self.hidden_dim, device=x.device)
        afa_pool = h[addr_mask].mean(dim=0) if addr_mask.any() else torch.zeros(self.hidden_dim, device=x.device)

        mem_seq, updated_mem = self.memory_cell(
            h, time_deltas, prev_memories, order_idx, enable_memory=enable_memory
        )
        motif_vec, motif_act, motif_slot = self.motif_bank(
            tse_pool, afa_pool, enable_in_graph_memory=use_ig_mem
        )

        fused = self.fusion(torch.cat([tse_pool, afa_pool, mem_seq, motif_vec], dim=-1))
        contrastive_emb = F.normalize(self.contrastive_proj(fused), dim=-1)
        risk_logit = self.risk_head(fused).squeeze(-1)
        typology_logits = self.typology_head(fused)
        return risk_logit, typology_logits, contrastive_emb, updated_mem, motif_act, motif_slot


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

    src_idx, dst_idx, valid_edge_ids = [], [], []
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
        sorted(range(len(nodes)), key=lambda i: nodes[i].timestamp), dtype=torch.long, device=device
    )
    return x, node_is_tx, edge_index, time_deltas, prev_mem, order_idx, valid_edge_ids


class FocalContrastiveLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0, temperature: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature

    def forward(self, risk_logits, labels, embeddings, typ_logits=None, typ_targets=None):
        bce = F.binary_cross_entropy_with_logits(risk_logits, labels.float(), reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * labels.float() + (1.0 - self.alpha) * (1.0 - labels.float())
        focal = (alpha_t * ((1.0 - pt) ** self.gamma) * bce).mean()

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

        typ_loss = F.cross_entropy(typ_logits, typ_targets) if (typ_logits is not None and typ_targets is not None) else torch.tensor(0.0, device=risk_logits.device)
        return focal + 0.25 * con_loss + 0.35 * typ_loss


def load_model_from_gcs_or_init(
    vault_bucket: Optional[str] = None,
    blob_path: str = "models/temporal_bitchetg_v3.pt",
    local_cache: str = "/tmp/temporal_bitchetg_v3.pt",
) -> Tuple[TemporalBitCHetGNet, str]:
    model = TemporalBitCHetGNet()
    if vault_bucket:
        try:
            from google.cloud import storage
            blob = storage.Client().bucket(vault_bucket).blob(blob_path)
            if blob.exists():
                blob.download_to_filename(local_cache)
                state = torch.load(local_cache, map_location="cpu")
                model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
                model.eval()
                return model, "TRAINED_GCS_CHECKPOINT"
        except Exception as exc:
            print(f"[ModelLoader] GCS checkpoint not found yet ({exc}); booting Day-1 fallback.")

    if os.path.exists(local_cache):
        state = torch.load(local_cache, map_location="cpu")
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        model.eval()
        return model, "TRAINED_LOCAL_CHECKPOINT"

    model.eval()
    return model, "UNTRAINED_DAY1_FALLBACK"
