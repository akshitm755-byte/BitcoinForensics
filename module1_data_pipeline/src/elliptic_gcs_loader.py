import os
import pandas as pd
import networkx as nx
from typing import Optional, List
from contracts.schemas import Node, Edge, NodeType, EdgeType, HetGraph
from module1_data_pipeline.src.features import ExplicitFeatureBuilder

DEFAULT_ELLIPTIC_PREFIX = "raw/elliptic_pp/Elliptic++ Dataset"

class EllipticPlusPlusGCSLoader:
    """Loads real Elliptic++ CSVs from GCS and rebuilds explicit recomputable features."""
    def __init__(
        self,
        local_dir: str = "/tmp/elliptic_pp_cache",
        gcs_bucket: Optional[str] = None,
        gcs_prefix: str = DEFAULT_ELLIPTIC_PREFIX,
    ):
        self.local_dir = local_dir
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix.rstrip("/")

    def sync_from_gcs(self) -> str:
        if not self.gcs_bucket:
            return self.local_dir
        from google.cloud import storage
        os.makedirs(self.local_dir, exist_ok=True)
        client = storage.Client()
        bucket = client.bucket(self.gcs_bucket)
        required = [
            "txs_classes.csv",
            "txs_edgelist.csv",
            "txs_features.csv",
            "AddrTx_edgelist.csv",
            "TxAddr_edgelist.csv",
            "AddrAddr_edgelist.csv",
            "wallets_classes.csv",
        ]
        for fname in required:
            dst = os.path.join(self.local_dir, fname)
            if not os.path.exists(dst):
                blob = bucket.blob(f"{self.gcs_prefix}/{fname}")
                if blob.exists():
                    print(f"[Elliptic++ GCS] Downloading {fname}...")
                    blob.download_to_filename(dst)
        return self.local_dir

    def load_canonical_graph(self, max_txs: Optional[int] = None) -> HetGraph:
        if self.gcs_bucket:
            self.sync_from_gcs()

        tx_cls = pd.read_csv(os.path.join(self.local_dir, "txs_classes.csv"))
        tx_cls.columns = [c.strip() for c in tx_cls.columns]
        tx_id_col, cls_col = tx_cls.columns[0], tx_cls.columns[1]

        tx_feat = pd.read_csv(os.path.join(self.local_dir, "txs_features.csv"), usecols=[0, 1])
        tx_feat.columns = [tx_id_col, "time_step"]

        tx_df = tx_feat.merge(tx_cls, on=tx_id_col, how="left")
        if max_txs:
            # Keep ALL labeled illicit transactions first so minority class (~2%) isn't lost in subsampling
            illicit_df = tx_df[tx_df[cls_col].astype(str).str.strip() == "1"]
            other_df = tx_df[tx_df[cls_col].astype(str).str.strip() != "1"]
            rem_quota = max(1000, max_txs - len(illicit_df))
            other_sub = other_df.sort_values("time_step").groupby("time_step", group_keys=False).apply(
                lambda df: df.head(max(1, rem_quota // 49))
            )
            tx_df = pd.concat([illicit_df, other_sub], ignore_index=True).drop_duplicates(subset=[tx_id_col])

        def map_label(val):
            s = str(val).strip()
            return 1 if s == "1" else (0 if s == "2" else None)

        tx_nodes_map = {}
        valid_tx_ids = set()
        for row in tx_df.itertuples(index=False):
            tid = str(getattr(row, tx_id_col))
            ts = int(getattr(row, "time_step")) * 1200
            nid = f"tx_{tid}"
            valid_tx_ids.add(tid)
            tx_nodes_map[nid] = Node(
                node_id=nid, node_type=NodeType.TRANSACTION, timestamp=ts, label=map_label(getattr(row, cls_col))
            )

        edges: List[Edge] = []
        addr_nodes_map = {}

        tx_edge_path = os.path.join(self.local_dir, "txs_edgelist.csv")
        if os.path.exists(tx_edge_path):
            df = pd.read_csv(tx_edge_path)
            c0, c1 = df.columns[0], df.columns[1]
            df[c0], df[c1] = df[c0].astype(str), df[c1].astype(str)
            df = df[df[c0].isin(valid_tx_ids) & df[c1].isin(valid_tx_ids)]
            for idx, r in enumerate(df.itertuples(index=False)):
                u, v = f"tx_{getattr(r, c0)}", f"tx_{getattr(r, c1)}"
                ts = max(tx_nodes_map[u].timestamp, tx_nodes_map[v].timestamp)
                edges.append(Edge(
                    edge_id=f"e_txtx_{idx}", source_id=u, target_id=v,
                    edge_type=EdgeType.TX_TO_TX, value=1.0, timestamp=ts
                ))

        for fname, is_in, etype, prefix in [
            ("AddrTx_edgelist.csv", True, EdgeType.ADDR_TO_TX, "e_atx"),
            ("TxAddr_edgelist.csv", False, EdgeType.TX_TO_ADDR, "e_txa"),
        ]:
            path = os.path.join(self.local_dir, fname)
            if os.path.exists(path):
                df = pd.read_csv(path)
                c0, c1 = df.columns[0], df.columns[1]
                df[c0], df[c1] = df[c0].astype(str), df[c1].astype(str)
                tx_col, addr_col = (c1, c0) if is_in else (c0, c1)
                df = df[df[tx_col].isin(valid_tx_ids)]
                for idx, r in enumerate(df.itertuples(index=False)):
                    tid = f"tx_{getattr(r, tx_col)}"
                    aid = f"addr_{getattr(r, addr_col)}"
                    ts = tx_nodes_map[tid].timestamp
                    if aid not in addr_nodes_map or ts < addr_nodes_map[aid].timestamp:
                        addr_nodes_map[aid] = Node(
                            node_id=aid, node_type=NodeType.ADDRESS, timestamp=ts,
                            label=tx_nodes_map[tid].label
                        )
                    src, dst = (aid, tid) if is_in else (tid, aid)
                    edges.append(Edge(
                        edge_id=f"{prefix}_{idx}", source_id=src, target_id=dst,
                        edge_type=etype, value=1.0, timestamp=ts
                    ))

        all_nodes = {**tx_nodes_map, **addr_nodes_map}
        edges.sort(key=lambda e: e.timestamp)
        g = nx.MultiDiGraph()
        last_seen = {}
        for e in edges:
            g.add_edge(e.source_id, e.target_id, key=e.edge_id, value=e.value, timestamp=e.timestamp)
            for nid in (e.source_id, e.target_id):
                if nid in all_nodes:
                    all_nodes[nid].features = ExplicitFeatureBuilder.compute_node_features(
                        nid, e.timestamp, g, last_seen
                    )
                    last_seen[nid] = e.timestamp

        return HetGraph(nodes=sorted(all_nodes.values(), key=lambda n: n.timestamp), edges=edges)
