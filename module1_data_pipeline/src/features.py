import math
import networkx as nx
from typing import Dict
from contracts.schemas import FEATURE_KEYS

class ExplicitFeatureBuilder:
    """
    Computes 7 explicitly defined, recomputable features for any unseen streaming node.
    Replaces Elliptic's undisclosed black-box features so live Bitcoin/AMLSim streams
    share the exact same feature space as training data.
    """
    FEATURE_SPEC = {
        "in_degree": "Incoming edge count in active window",
        "out_degree": "Outgoing edge count in active window",
        "total_in_val": "Log-scaled sum of incoming BTC value",
        "total_out_val": "Log-scaled sum of outgoing BTC value",
        "time_delta_prev": "Seconds elapsed since node's previous event (capped at 3600s)",
        "fan_out_ratio": "out_degree / (in_degree + 1.0) — captures peeling & fan-out",
        "hop2_neighbor_count": "Log-scaled 2-hop neighborhood size",
    }

    @staticmethod
    def compute_node_features(
        node_id: str, timestamp: int, g: nx.MultiDiGraph, last_seen: Dict[str, int]
    ) -> Dict[str, float]:
        in_edges = list(g.in_edges(node_id, data=True)) if node_id in g else []
        out_edges = list(g.out_edges(node_id, data=True)) if node_id in g else []

        in_deg = float(len(in_edges))
        out_deg = float(len(out_edges))
        in_val = sum(float(d.get("value", 0.0)) for _, _, d in in_edges)
        out_val = sum(float(d.get("value", 0.0)) for _, _, d in out_edges)

        prev_ts = last_seen.get(node_id, timestamp)
        raw_dt = max(0, timestamp - prev_ts)
        time_delta = float(min(3600, raw_dt))
        fan_out_ratio = out_deg / (in_deg + 1.0)

        hop1 = (set(g.predecessors(node_id)) | set(g.successors(node_id))) if node_id in g else set()
        hop2 = set()
        for nbr in list(hop1)[:40]:
            hop2 |= set(g.predecessors(nbr)) | set(g.successors(nbr))

        return {
            "in_degree": round(in_deg, 4),
            "out_degree": round(out_deg, 4),
            "total_in_val": round(math.log1p(max(0.0, in_val)), 4),
            "total_out_val": round(math.log1p(max(0.0, out_val)), 4),
            "time_delta_prev": round(time_delta / 60.0, 4),  # minutes
            "fan_out_ratio": round(fan_out_ratio, 4),
            "hop2_neighbor_count": round(math.log1p(len(hop2 - {node_id})), 4),
        }

    @staticmethod
    def to_vector(features: Dict[str, float]) -> list[float]:
        return [float(features.get(k, 0.0)) for k in FEATURE_KEYS]
