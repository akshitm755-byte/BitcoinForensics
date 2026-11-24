import random
import time
import uuid
from typing import List, Optional, Tuple
from contracts.schemas import Node, Edge, NodeType, EdgeType, TransactionEvent, Typology

class AMLSimStreamGenerator:
    """
    Continuous AMLSim-schema transaction stream generator.
    Generates realistic background Bitcoin/VASP traffic AND injects cross-timestep
    laundering typologies (Peeling Chain, Fan-Out/Fan-In, Cyclic Flow, Smurfing)
    that span multiple consecutive events so the temporal memory encoder can detect them.
    """
    KNOWN_OSINT_ROOTS = [
        "bc1q_lazarus_peel_root",
        "bc1q_darknet_mixer_hub",
        "bc1q_ronin_bridge_hop",
        "bc1q_hydra_smurf_vault",
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.sim_ts = int(time.time())
        self.active_wallets = [f"bc1q_usr_{i:04d}" for i in range(1, 60)]
        self.pending_queue: List[TransactionEvent] = []
        self.tx_counter = 0

    def next_event(self, inject_probability: float = 0.18) -> TransactionEvent:
        if self.pending_queue:
            return self.pending_queue.pop(0)

        if self.rng.random() < inject_probability:
            typ = self.rng.choice([
                Typology.PEELING_CHAIN,
                Typology.FAN_OUT_FAN_IN,
                Typology.CYCLIC_FLOW,
                Typology.SMURFING,
            ])
            self.enqueue_typology(typ)
            return self.pending_queue.pop(0)

        return self._generate_licit_event()

    def enqueue_typology(self, typology: Typology) -> int:
        if typology == Typology.PEELING_CHAIN:
            evs = self._build_peeling_chain()
        elif typology == Typology.FAN_OUT_FAN_IN:
            evs = self._build_fan_out_fan_in()
        elif typology == Typology.CYCLIC_FLOW:
            evs = self._build_cyclic_flow()
        elif typology == Typology.SMURFING:
            evs = self._build_smurfing()
        else:
            evs = [self._generate_licit_event()]
        self.pending_queue.extend(evs)
        return len(evs)

    def generate_training_corpus(self, num_licit: int = 120, num_illicit_clusters: int = 32) -> Tuple[List[Node], List[Edge], List[TransactionEvent]]:
        events: List[TransactionEvent] = []
        all_nodes = {}
        all_edges = []
        typologies = [Typology.PEELING_CHAIN, Typology.FAN_OUT_FAN_IN, Typology.CYCLIC_FLOW, Typology.SMURFING]

        total_steps = num_licit + num_illicit_clusters
        illicit_interval = max(1, total_steps // max(1, num_illicit_clusters))
        illicit_created = 0

        for step in range(total_steps):
            if step % illicit_interval == 0 and illicit_created < num_illicit_clusters:
                typ = typologies[illicit_created % len(typologies)]
                self.enqueue_typology(typ)
                while self.pending_queue:
                    ev = self.pending_queue.pop(0)
                    events.append(ev)
                    for n in ev.nodes:
                        all_nodes[n.node_id] = n
                    all_edges.extend(ev.edges)
                illicit_created += 1
            else:
                ev = self._generate_licit_event()
                events.append(ev)
                for n in ev.nodes:
                    all_nodes[n.node_id] = n
                all_edges.extend(ev.edges)

        return sorted(all_nodes.values(), key=lambda n: n.timestamp), sorted(all_edges, key=lambda e: e.timestamp), events

    def _new_tx_id(self, tag: str) -> str:
        self.tx_counter += 1
        return f"tx_{tag}_{self.tx_counter:05d}"

    def _generate_licit_event(self) -> TransactionEvent:
        self.sim_ts += self.rng.randint(120, 480)
        src = self.rng.choice(self.active_wallets)
        dst = self.rng.choice([w for w in self.active_wallets if w != src])
        tx_id = self._new_tx_id("norm")
        val = round(self.rng.uniform(0.02, 1.25), 4)

        nodes = [
            Node(node_id=src, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=0),
            Node(node_id=tx_id, node_type=NodeType.TRANSACTION, timestamp=self.sim_ts, label=0),
            Node(node_id=dst, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=0),
        ]
        edges = [
            Edge(edge_id=f"e_{tx_id}_in", source_id=src, target_id=tx_id, edge_type=EdgeType.ADDR_TO_TX, value=val, timestamp=self.sim_ts),
            Edge(edge_id=f"e_{tx_id}_out", source_id=tx_id, target_id=dst, edge_type=EdgeType.TX_TO_ADDR, value=round(val * 0.998, 4), timestamp=self.sim_ts),
        ]
        return TransactionEvent(
            correlation_id=f"corr_{uuid.uuid4().hex[:10]}",
            timestamp=self.sim_ts,
            nodes=nodes,
            edges=edges,
            ground_truth_typology=Typology.LICIT,
        )

    def _build_peeling_chain(self, hops: int = 5) -> List[TransactionEvent]:
        events = []
        cluster = uuid.uuid4().hex[:4]
        prev_addr = "bc1q_lazarus_peel_root" if self.rng.random() < 0.7 else f"bc1q_anon_peel_{cluster}"
        amount = round(self.rng.uniform(10.0, 28.0), 4)
        prev_tx = None

        for hop in range(1, hops + 1):
            self.sim_ts += self.rng.randint(8, 22)  # Rapid 8–22s cadence!
            tx_id = self._new_tx_id(f"peel_{cluster}_h{hop}")
            peel_out = f"bc1q_peel_exit_{cluster}_{hop}"
            change_addr = f"bc1q_peel_chg_{cluster}_{hop}"
            peeled = round(amount * self.rng.uniform(0.12, 0.20), 4)
            rem = round(max(0.1, amount - peeled - 0.002), 4)

            nodes = [
                Node(node_id=prev_addr, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=1),
                Node(node_id=tx_id, node_type=NodeType.TRANSACTION, timestamp=self.sim_ts, label=1),
                Node(node_id=peel_out, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=1),
                Node(node_id=change_addr, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=1),
            ]
            edges = [
                Edge(edge_id=f"e_{tx_id}_in", source_id=prev_addr, target_id=tx_id, edge_type=EdgeType.ADDR_TO_TX, value=amount, timestamp=self.sim_ts),
                Edge(edge_id=f"e_{tx_id}_peel", source_id=tx_id, target_id=peel_out, edge_type=EdgeType.TX_TO_ADDR, value=peeled, timestamp=self.sim_ts),
                Edge(edge_id=f"e_{tx_id}_chg", source_id=tx_id, target_id=change_addr, edge_type=EdgeType.TX_TO_ADDR, value=rem, timestamp=self.sim_ts),
            ]
            if prev_tx:
                edges.append(Edge(edge_id=f"e_{tx_id}_chain", source_id=prev_tx, target_id=tx_id, edge_type=EdgeType.TX_TO_TX, value=amount, timestamp=self.sim_ts))

            events.append(TransactionEvent(
                correlation_id=f"corr_peel_{cluster}_{hop}",
                timestamp=self.sim_ts,
                nodes=nodes,
                edges=edges,
                ground_truth_typology=Typology.PEELING_CHAIN,
            ))
            prev_addr = change_addr
            prev_tx = tx_id
            amount = rem
        return events

    def _build_fan_out_fan_in(self, mules: int = 5) -> List[TransactionEvent]:
        events = []
        cluster = uuid.uuid4().hex[:4]
        root = "bc1q_darknet_mixer_hub" if self.rng.random() < 0.7 else f"bc1q_anon_hub_{cluster}"
        sink = f"bc1q_consolidation_{cluster}"
        mule_addrs = [f"bc1q_mule_{cluster}_{i}" for i in range(mules)]
        total = round(self.rng.uniform(8.0, 20.0), 4)
        split_val = round(total / mules, 4)

        # Step 1: Rapid fan-out
        self.sim_ts += self.rng.randint(10, 20)
        tx_out = self._new_tx_id(f"fanout_{cluster}")
        nodes = [Node(node_id=root, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=1),
                 Node(node_id=tx_out, node_type=NodeType.TRANSACTION, timestamp=self.sim_ts, label=1)]
        edges = [Edge(edge_id=f"e_{tx_out}_in", source_id=root, target_id=tx_out, edge_type=EdgeType.ADDR_TO_TX, value=total, timestamp=self.sim_ts)]
        for idx, m in enumerate(mule_addrs):
            nodes.append(Node(node_id=m, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=1))
            edges.append(Edge(edge_id=f"e_{tx_out}_m{idx}", source_id=tx_out, target_id=m, edge_type=EdgeType.TX_TO_ADDR, value=split_val, timestamp=self.sim_ts))
        events.append(TransactionEvent(
            correlation_id=f"corr_fan_{cluster}_1", timestamp=self.sim_ts,
            nodes=nodes, edges=edges, ground_truth_typology=Typology.FAN_OUT_FAN_IN
        ))

        # Step 2: Rapid fan-in consolidation
        self.sim_ts += self.rng.randint(10, 25)
        tx_in = self._new_tx_id(f"fanin_{cluster}")
        nodes2 = [Node(node_id=tx_in, node_type=NodeType.TRANSACTION, timestamp=self.sim_ts, label=1),
                  Node(node_id=sink, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=1)]
        edges2 = [Edge(edge_id=f"e_{tx_in}_sink", source_id=tx_in, target_id=sink, edge_type=EdgeType.TX_TO_ADDR, value=round(total * 0.98, 4), timestamp=self.sim_ts)]
        for idx, m in enumerate(mule_addrs):
            nodes2.append(Node(node_id=m, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=1))
            edges2.append(Edge(edge_id=f"e_{tx_in}_m{idx}", source_id=m, target_id=tx_in, edge_type=EdgeType.ADDR_TO_TX, value=split_val, timestamp=self.sim_ts))
        events.append(TransactionEvent(
            correlation_id=f"corr_fan_{cluster}_2", timestamp=self.sim_ts,
            nodes=nodes2, edges=edges2, ground_truth_typology=Typology.FAN_OUT_FAN_IN
        ))
        return events

    def _build_cyclic_flow(self, ring_len: int = 4) -> List[TransactionEvent]:
        events = []
        cluster = uuid.uuid4().hex[:4]
        ring = ["bc1q_ronin_bridge_hop" if self.rng.random() < 0.65 else f"bc1q_ring_{cluster}_0"] + [
            f"bc1q_ring_{cluster}_{i}" for i in range(1, ring_len)
        ]
        val = round(self.rng.uniform(5.0, 15.0), 4)
        for i in range(ring_len):
            self.sim_ts += self.rng.randint(8, 18)
            src, dst = ring[i], ring[(i + 1) % ring_len]
            tx_id = self._new_tx_id(f"cyc_{cluster}_{i}")
            val = round(val * 0.995, 4)
            events.append(TransactionEvent(
                correlation_id=f"corr_cyc_{cluster}_{i}",
                timestamp=self.sim_ts,
                nodes=[
                    Node(node_id=src, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=1),
                    Node(node_id=tx_id, node_type=NodeType.TRANSACTION, timestamp=self.sim_ts, label=1),
                    Node(node_id=dst, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=1),
                ],
                edges=[
                    Edge(edge_id=f"e_{tx_id}_in", source_id=src, target_id=tx_id, edge_type=EdgeType.ADDR_TO_TX, value=val, timestamp=self.sim_ts),
                    Edge(edge_id=f"e_{tx_id}_out", source_id=tx_id, target_id=dst, edge_type=EdgeType.TX_TO_ADDR, value=val, timestamp=self.sim_ts),
                ],
                ground_truth_typology=Typology.CYCLIC_FLOW,
            ))
        return events

    def _build_smurfing(self, bursts: int = 4) -> List[TransactionEvent]:
        events = []
        cluster = uuid.uuid4().hex[:4]
        vault = "bc1q_hydra_smurf_vault" if self.rng.random() < 0.7 else f"bc1q_smurf_vault_{cluster}"
        for b in range(bursts):
            self.sim_ts += self.rng.randint(6, 15)
            tx_id = self._new_tx_id(f"smurf_{cluster}_{b}")
            src = f"bc1q_smurf_src_{cluster}_{b}"
            # Structuring just below reporting threshold (~0.48 BTC bursts)
            val = round(self.rng.uniform(0.45, 0.495), 4)
            events.append(TransactionEvent(
                correlation_id=f"corr_smurf_{cluster}_{b}",
                timestamp=self.sim_ts,
                nodes=[
                    Node(node_id=src, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=1),
                    Node(node_id=tx_id, node_type=NodeType.TRANSACTION, timestamp=self.sim_ts, label=1),
                    Node(node_id=vault, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=1),
                ],
                edges=[
                    Edge(edge_id=f"e_{tx_id}_in", source_id=src, target_id=tx_id, edge_type=EdgeType.ADDR_TO_TX, value=val, timestamp=self.sim_ts),
                    Edge(edge_id=f"e_{tx_id}_out", source_id=tx_id, target_id=vault, edge_type=EdgeType.TX_TO_ADDR, value=val, timestamp=self.sim_ts),
                ],
                ground_truth_typology=Typology.SMURFING,
            ))
        return events
