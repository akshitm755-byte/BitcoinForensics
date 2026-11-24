import os
import pandas as pd
import networkx as nx
from typing import Optional, Tuple, List
from contracts.schemas import Node, Edge, NodeType, EdgeType, HetGraph
from module1_data_pipeline.src.features import ExplicitFeatureBuilder

DEFAULT_ELLIPTIC_PREFIX = "raw/elliptic_pp/Elliptic++ Dataset"

class EllipticPlusPlusGCSLoader:
    """
    Loads the real Elliptic++ dataset directly from GCS (or local cache directory):
      - txs_features.csv, txs_classes.csv, txs_edgelist.csv
      - wallets_features_classes_combined.csv (or wallets_classes.csv)
      - AddrTx_edgelist.csv, TxAddr_edgelist.csv, AddrAddr_edgelist.csv
    Rebuilds explicit recomputable features rather than relying on undisclosed raw features.
    """
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

        tx_cls_path = os.path.join(self.local_dir, "txs_classes.csv")
        tx_feat_path = os.path.join(self.local_dir, "txs_features.csv")
        tx_edge_path = os.path.join(self.local_dir, "txs_edgelist.csv")
        addr_tx_path = os.path.join(self.local_dir, "AddrTx_edgelist.csv")
        tx_addr_path = os.path.join(self.local_dir, "TxAddr_edgelist.csv")

        # 1. Load transaction timestamps & classes (Elliptic++ class: 1=illicit, 2=licit, 3=unknown)
        tx_cls = pd.read_csv(tx_cls_path)
        tx_cls.columns = [c.strip() for c in tx_cls.columns]
        tx_id_col = tx_cls.columns[0]
        cls_col = tx_cls.columns[1]

        tx_feat = pd.read_csv(tx_feat_path, usecols=[0, 1])
        tx_feat.columns = [tx_id_col, "time_step"]

        tx_df = tx_feat.merge(tx_cls, on=tx_id_col, how="left")
        if max_txs:
            # Keep balanced temporal coverage across all 49 timesteps when subsampling
            tx_df = tx_df.sort_values("time_step").groupby("time_step", group_keys=False).apply(
                lambda df: df.head(max(1, max_txs // 49))
            )

        def map_label(val):
            s = str(val).strip()
            if s == "1":
                return 1
            if s == "2":
                return 0
            return None

        tx_nodes_map = {}
        valid_tx_ids = set()
        for row in tx_df.itertuples(index=False):
            tid = str(getattr(row, tx_id_col))
            ts = int(getattr(row, "time_step")) * 1200  # ~20 min per Elliptic timestep
            lbl = map_label(getattr(row, cls_col))
            nid = f"tx_{tid}"
            valid_tx_ids.add(tid)
            tx_nodes_map[nid] = Node(
                node_id=nid, node_type=NodeType.TRANSACTION, timestamp=ts, label=lbl
            )

        edges: List[Edge] = []
        addr_nodes_map = {}

        # 2. Transaction-to-Transaction edges
        if os.path.exists(tx_edge_path):
            tx_edges_df = pd.read_csv(tx_edge_path)
            c0, c1 = tx_edges_df.columns[0], tx_edges_df.columns[1]
            tx_edges_df[c0] = tx_edges_df[c0].astype(str)
            tx_edges_df[c1] = tx_edges_df[c1].astype(str)
            tx_edges_df = tx_edges_df[tx_edges_df[c0].isin(valid_tx_ids) & tx_edges_df[c1].isin(valid_tx_ids)]
            for idx, r in enumerate(tx_edges_df.itertuples(index=False)):
                u, v = f"tx_{getattr(r, c0)}", f"tx_{getattr(r, c1)}"
                ts = max(tx_nodes_map[u].timestamp, tx_nodes_map[v].timestamp)
                edges.append(Edge(
                    edge_id=f"e_txtx_{idx}", source_id=u, target_id=v,
                    edge_type=EdgeType.TX_TO_TX, value=1.0, timestamp=ts
                ))

        # 3. Address <-> Transaction edges (heterogeneous structure for Bit-CHetG AFA)
        for path, is_in, etype, prefix in [
            (addr_tx_path, True, EdgeType.ADDR_TO_TX, "e_atx"),
            (tx_addr_path, False, EdgeType.TX_TO_ADDR, "e_txa"),
        ]:
            if os.path.exists(path):
                df = pd.read_csv(path)
                c0, c1 = df.columns[0], df.columns[1]
                df[c0] = df[c0].astype(str)
                df[c1] = df[c1].astype(str)
                tx_col = c1 if is_in else c0
                addr_col = c0 if is_in else c1
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

        # 4. Recompute explicit features chronologically
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

        sorted_nodes = sorted(all_nodes.values(), key=lambda n: n.timestamp)
        return HetGraph(nodes=sorted_nodes, edges=edges)
