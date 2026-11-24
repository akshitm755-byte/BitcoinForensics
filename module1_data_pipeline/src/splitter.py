from typing import List, Tuple
from contracts.schemas import Node, Edge, SplitManifest

class TemporalLeakageError(Exception):
    pass

class TemporalSplitter:
    def __init__(self, train_max_ts: int, val_max_ts: int, test_max_ts: int):
        self.train_max_ts = train_max_ts
        self.val_max_ts = val_max_ts
        self.test_max_ts = test_max_ts

    def split_and_verify(self, nodes: List[Node], edges: List[Edge]) -> Tuple[SplitManifest, List[Edge]]:
        node_ts = {n.node_id: n.timestamp for n in nodes}

        train_ids = [n.node_id for n in nodes if n.timestamp <= self.train_max_ts]
        val_ids = [n.node_id for n in nodes if self.train_max_ts < n.timestamp <= self.val_max_ts]
        test_ids = [n.node_id for n in nodes if n.timestamp > self.val_max_ts]

        train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

        clean_edges = []
        for e in edges:
            # Strip any cross-split boundary edge so past nodes never peek at future neighbors
            if (e.source_id in train_set and e.target_id not in train_set) or \
               (e.target_id in train_set and e.source_id not in train_set) or \
               (e.source_id in val_set and e.target_id in test_set) or \
               (e.target_id in val_set and e.source_id in test_set):
                continue
            clean_edges.append(e)

        self._assert_no_leakage(train_set, val_set, test_set, node_ts, clean_edges)

        manifest = SplitManifest(
            train_max_ts=self.train_max_ts,
            val_max_ts=self.val_max_ts,
            test_max_ts=self.test_max_ts,
            train_node_ids=train_ids,
            val_node_ids=val_ids,
            test_node_ids=test_ids,
            leakage_verified=True,
        )
        return manifest, clean_edges

    def _assert_no_leakage(self, train_set: set, val_set: set, test_set: set, node_ts: dict, edges: List[Edge]):
        if not train_set.isdisjoint(val_set) or not train_set.isdisjoint(test_set) or not val_set.isdisjoint(test_set):
            raise TemporalLeakageError("Partition overlap detected between train/val/test.")
        if train_set and val_set and max(node_ts[n] for n in train_set) >= min(node_ts[n] for n in val_set):
            raise TemporalLeakageError("Train timestamp bleeds into Validation.")
        if val_set and test_set and max(node_ts[n] for n in val_set) >= min(node_ts[n] for n in test_set):
            raise TemporalLeakageError("Validation timestamp bleeds into Test.")
        for e in edges:
            if e.source_id in train_set and e.target_id in (val_set | test_set):
                raise TemporalLeakageError(f"Forward-facing boundary edge leaked: {e.edge_id}")
