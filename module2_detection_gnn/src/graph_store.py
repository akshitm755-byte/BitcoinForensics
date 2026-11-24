import networkx as nx
import torch
from typing import Dict, List, Tuple
from contracts.schemas import TransactionEvent, Node, Edge
from module1_data_pipeline.src.features import ExplicitFeatureBuilder

class IncrementalGraphStore:
    """
    Maintains the live heterogeneous graph and per-node PyTorch temporal memory vectors.
    Evicts expired edges outside `max_window_seconds` so serverless memory stays bounded.
    """
    def __init__(self, memory_dim: int = 32, max_window_seconds: int = 1800):
        self.g = nx.MultiDiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.memory: Dict[str, torch.Tensor] = {}
        self.last_updated: Dict[str, int] = {}
        self.memory_dim = memory_dim
        self.max_window_seconds = max_window_seconds

    def apply_event(self, event: TransactionEvent) -> Tuple[List[Node], List[Edge]]:
        for n in event.nodes:
            self.nodes[n.node_id] = n
            if n.node_id not in self.memory:
                self.memory[n.node_id] = torch.zeros(self.memory_dim, dtype=torch.float32)

        for e in event.edges:
            self.edges[e.edge_id] = e
            self.g.add_edge(e.source_id, e.target_id, key=e.edge_id, value=e.value, timestamp=e.timestamp)

        touched_ids = {n.node_id for n in event.nodes}
        for nid in touched_ids:
            self.nodes[nid].features = ExplicitFeatureBuilder.compute_node_features(
                nid, event.timestamp, self.g, self.last_updated
            )
            self.last_updated[nid] = event.timestamp

        self._evict_expired(event.timestamp)
        return self.sample_temporal_subgraph(touched_ids, event.timestamp)

    def sample_temporal_subgraph(self, seed_ids: set, current_ts: int, hops: int = 2) -> Tuple[List[Node], List[Edge]]:
        visited = set(seed_ids)
        frontier = set(seed_ids)
        for _ in range(hops):
            next_frontier = set()
            for nid in frontier:
                if nid in self.g:
                    next_frontier |= set(self.g.predecessors(nid)) | set(self.g.successors(nid))
            visited |= next_frontier
            frontier = next_frontier
            if len(visited) > 64:
                break

        sub_nodes = [self.nodes[nid] for nid in visited if nid in self.nodes]
        sub_edges = [
            self.edges[k]
            for _, _, k, d in self.g.edges(visited, keys=True, data=True)
            if k in self.edges and (current_ts - d.get("timestamp", current_ts)) <= self.max_window_seconds
        ]
        return sub_nodes, sub_edges

    def _evict_expired(self, current_ts: int):
        expired = [
            (u, v, k) for u, v, k, d in self.g.edges(keys=True, data=True)
            if (current_ts - d.get("timestamp", current_ts)) > self.max_window_seconds
        ]
        for u, v, k in expired:
            self.g.remove_edge(u, v, key=k)
            self.edges.pop(k, None)
