import heapq
import math
import random
import time
import uuid
from typing import List, Optional, Tuple
from contracts.schemas import Node, Edge, NodeType, EdgeType, TransactionEvent, Typology


class AMLSimStreamGenerator:
    """
    Realistic Interlaced AMLSim-Schema Stream Scheduler:
      - Low baseline illicit prevalence (~5.5% campaign rate).
      - Multi-hop laundering attempts are scheduled across a min-heap so 2-5 genuine licit
        transactions naturally interlace between hops of the exact same illicit campaign.
    """
    def __init__(self, seed: Optional[int] = None, illicit_campaign_prob: float = 0.055):
        self.rng = random.Random(seed)
        self.sim_ts = int(time.time())
        self.tick = 0
        self.illicit_campaign_prob = illicit_campaign_prob
        self.active_wallets = [f"bc1q_usr_{i:04d}" for i in range(1, 95)]
        self.merchant_wallets = [f"bc1q_merch_{i:03d}" for i in range(1, 16)]
        self._scheduled_heap: List[Tuple[int, int, TransactionEvent]] = []
        self._heap_seq = 0
        self.tx_counter = 0

    @property
    def active_interlaced_hops(self) -> int:
        return len(self._scheduled_heap)

    def next_event(self) -> TransactionEvent:
        self.tick += 1

        if self._scheduled_heap and self._scheduled_heap[0][0] <= self.tick:
            _, _, ev = heapq.heappop(self._scheduled_heap)
            self.sim_ts = max(self.sim_ts + 2, ev.timestamp)
            ev.timestamp = self.sim_ts
            for n in ev.nodes:
                n.timestamp = self.sim_ts
            for e in ev.edges:
                e.timestamp = self.sim_ts
            return ev

        if len(self._scheduled_heap) < 8 and self.rng.random() < self.illicit_campaign_prob:
            typ = self.rng.choice([
                Typology.PEELING_CHAIN,
                Typology.FAN_OUT_FAN_IN,
                Typology.CYCLIC_FLOW,
                Typology.SMURFING,
            ])
            self.enqueue_interlaced_typology(typ, immediate_first_hop=True)
            _, _, ev = heapq.heappop(self._scheduled_heap)
            return ev

        return self._generate_licit_event()

    def enqueue_interlaced_typology(self, typology: Typology, immediate_first_hop: bool = True) -> dict:
        if typology == Typology.PEELING_CHAIN:
            evs, alg, gap = self._build_peeling_chain(), "lognormal_peel_jitter_v2", (2, 4)
        elif typology == Typology.FAN_OUT_FAN_IN:
            evs, alg, gap = self._build_fan_out_fan_in(), "dirichlet_split_synchronized_recombine", (2, 5)
        elif typology == Typology.CYCLIC_FLOW:
            evs, alg, gap = self._build_cyclic_flow(), "periodic_ring_phase_shift", (2, 4)
        elif typology == Typology.SMURFING:
            evs, alg, gap = self._build_smurfing(), "sub_threshold_uniform_dither", (1, 3)
        else:
            evs, alg, gap = [self._generate_licit_event()], "poisson_retail_baseline", (1, 1)

        release_tick = self.tick if immediate_first_hop else self.tick + 1
        for idx, ev in enumerate(evs):
            if idx > 0:
                release_tick += self.rng.randint(gap[0], gap[1])
            self._heap_seq += 1
            heapq.heappush(self._scheduled_heap, (release_tick, self._heap_seq, ev))

        return {
            "typology": typology.value,
            "scheduled_hops": len(evs),
            "cadence_algorithm": alg,
            "interlaced_gap_txs": f"{gap[0]}-{gap[1]} licit txs/hop",
        }

    def generate_training_corpus(
        self, total_events: int = 420, illicit_campaign_prob: float = 0.08
    ) -> Tuple[List[Node], List[Edge], List[TransactionEvent]]:
        old_prob = self.illicit_campaign_prob
        self.illicit_campaign_prob = illicit_campaign_prob
        events: List[TransactionEvent] = []
        all_nodes = {}
        all_edges = []

        for _ in range(total_events):
            ev = self.next_event()
            events.append(ev)
            for n in ev.nodes:
                if n.node_id not in all_nodes or n.timestamp < all_nodes[n.node_id].timestamp:
                    all_nodes[n.node_id] = n
            all_edges.extend(ev.edges)

        while self._scheduled_heap:
            ev = self.next_event()
            events.append(ev)
            for n in ev.nodes:
                if n.node_id not in all_nodes or n.timestamp < all_nodes[n.node_id].timestamp:
                    all_nodes[n.node_id] = n
            all_edges.extend(ev.edges)

        self.illicit_campaign_prob = old_prob
        return sorted(all_nodes.values(), key=lambda n: n.timestamp), sorted(all_edges, key=lambda e: e.timestamp), events

    def _new_tx_id(self, tag: str) -> str:
        self.tx_counter += 1
        return f"tx_{tag}_{self.tx_counter:05d}"

    def _generate_licit_event(self) -> TransactionEvent:
        spacing = self.rng.randint(35, 190)
        self.sim_ts += spacing
        src = self.rng.choice(self.active_wallets)
        dst = self.rng.choice(self.merchant_wallets if self.rng.random() < 0.35 else self.active_wallets)
        while dst == src:
            dst = self.rng.choice(self.active_wallets)
        tx_id = self._new_tx_id("licit")
        val = round(math.exp(self.rng.uniform(-3.2, 0.4)), 4)

        return TransactionEvent(
            correlation_id=f"corr_{uuid.uuid4().hex[:10]}",
            timestamp=self.sim_ts,
            nodes=[
                Node(node_id=src, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=0),
                Node(node_id=tx_id, node_type=NodeType.TRANSACTION, timestamp=self.sim_ts, label=0),
                Node(node_id=dst, node_type=NodeType.ADDRESS, timestamp=self.sim_ts, label=0),
            ],
            edges=[
                Edge(edge_id=f"e_{tx_id}_in", source_id=src, target_id=tx_id, edge_type=EdgeType.ADDR_TO_TX, value=val, timestamp=self.sim_ts, temporal_spacing_sec=float(spacing)),
                Edge(edge_id=f"e_{tx_id}_out", source_id=tx_id, target_id=dst, edge_type=EdgeType.TX_TO_ADDR, value=round(val * 0.998, 4), timestamp=self.sim_ts, temporal_spacing_sec=float(spacing)),
            ],
            cadence_algorithm="poisson_retail_baseline",
            ground_truth_typology=Typology.LICIT,
        )

    def _build_peeling_chain(self, hops: int = 5) -> List[TransactionEvent]:
        events = []
        cid = f"camp_peel_{uuid.uuid4().hex[:5]}"
        prev_addr = "bc1q_lazarus_peel_root" if self.rng.random() < 0.65 else f"bc1q_anon_peel_{cid[-4:]}"
        amount = round(self.rng.uniform(11.0, 32.0), 4)
        prev_tx, ts_cursor = None, self.sim_ts

        for hop in range(1, hops + 1):
            spacing = max(8, int(self.rng.lognormvariate(2.85, 0.28)))
            ts_cursor += spacing
            tx_id = self._new_tx_id(f"peel_{cid[-4:]}_h{hop}")
            peel_out = f"bc1q_peel_exit_{cid[-4:]}_{hop}"
            change_addr = f"bc1q_peel_chg_{cid[-4:]}_{hop}"
            peeled = round(amount * self.rng.uniform(0.11, 0.19), 4)
            rem = round(max(0.15, amount - peeled - 0.0018), 4)

            edges = [
                Edge(edge_id=f"e_{tx_id}_in", source_id=prev_addr, target_id=tx_id, edge_type=EdgeType.ADDR_TO_TX, value=amount, timestamp=ts_cursor, temporal_spacing_sec=float(spacing)),
                Edge(edge_id=f"e_{tx_id}_peel", source_id=tx_id, target_id=peel_out, edge_type=EdgeType.TX_TO_ADDR, value=peeled, timestamp=ts_cursor, temporal_spacing_sec=float(spacing)),
                Edge(edge_id=f"e_{tx_id}_chg", source_id=tx_id, target_id=change_addr, edge_type=EdgeType.TX_TO_ADDR, value=rem, timestamp=ts_cursor, temporal_spacing_sec=float(spacing)),
            ]
            if prev_tx:
                edges.append(Edge(edge_id=f"e_{tx_id}_chain", source_id=prev_tx, target_id=tx_id, edge_type=EdgeType.TX_TO_TX, value=amount, timestamp=ts_cursor, temporal_spacing_sec=float(spacing)))

            events.append(TransactionEvent(
                correlation_id=f"corr_{cid}_h{hop}", timestamp=ts_cursor,
                nodes=[
                    Node(node_id=prev_addr, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1),
                    Node(node_id=tx_id, node_type=NodeType.TRANSACTION, timestamp=ts_cursor, label=1),
                    Node(node_id=peel_out, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1),
                    Node(node_id=change_addr, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1),
                ],
                edges=edges, campaign_id=cid, cadence_algorithm="lognormal_peel_jitter_v2",
                ground_truth_typology=Typology.PEELING_CHAIN,
            ))
            prev_addr, prev_tx, amount = change_addr, tx_id, rem
        return events

    def _build_fan_out_fan_in(self, mules: int = 5) -> List[TransactionEvent]:
        events = []
        cid = f"camp_fan_{uuid.uuid4().hex[:5]}"
        root = "bc1q_darknet_mixer_hub" if self.rng.random() < 0.65 else f"bc1q_anon_hub_{cid[-4:]}"
        sink = f"bc1q_consolidation_{cid[-4:]}"
        mule_addrs = [f"bc1q_mule_{cid[-4:]}_{i}" for i in range(mules)]
        total = round(self.rng.uniform(9.0, 24.0), 4)
        raw_w = [self.rng.gammavariate(2.5, 1.0) for _ in range(mules)]
        s_w = sum(raw_w)
        splits = [round(total * (w / s_w), 4) for w in raw_w]

        ts_cursor = self.sim_ts + self.rng.randint(10, 20)
        tx_out = self._new_tx_id(f"fanout_{cid[-4:]}")
        nodes = [
            Node(node_id=root, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1),
            Node(node_id=tx_out, node_type=NodeType.TRANSACTION, timestamp=ts_cursor, label=1),
        ]
        edges = [Edge(edge_id=f"e_{tx_out}_in", source_id=root, target_id=tx_out, edge_type=EdgeType.ADDR_TO_TX, value=total, timestamp=ts_cursor, temporal_spacing_sec=14.0)]
        for i, (m, val) in enumerate(zip(mule_addrs, splits)):
            nodes.append(Node(node_id=m, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1))
            edges.append(Edge(edge_id=f"e_{tx_out}_m{i}", source_id=tx_out, target_id=m, edge_type=EdgeType.TX_TO_ADDR, value=val, timestamp=ts_cursor, temporal_spacing_sec=14.0))
        events.append(TransactionEvent(
            correlation_id=f"corr_{cid}_fanout", timestamp=ts_cursor, nodes=nodes, edges=edges,
            campaign_id=cid, cadence_algorithm="dirichlet_split_synchronized_recombine", ground_truth_typology=Typology.FAN_OUT_FAN_IN
        ))

        relay_addrs = []
        for i, (m, val) in enumerate(zip(mule_addrs[:2], splits[:2])):
            sp = self.rng.randint(12, 26)
            ts_cursor += sp
            tx_r = self._new_tx_id(f"relay_{cid[-4:]}_{i}")
            r_addr = f"bc1q_relay_{cid[-4:]}_{i}"
            relay_addrs.append((r_addr, round(val * 0.99, 4)))
            events.append(TransactionEvent(
                correlation_id=f"corr_{cid}_r{i}", timestamp=ts_cursor,
                nodes=[
                    Node(node_id=m, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1),
                    Node(node_id=tx_r, node_type=NodeType.TRANSACTION, timestamp=ts_cursor, label=1),
                    Node(node_id=r_addr, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1),
                ],
                edges=[
                    Edge(edge_id=f"e_{tx_r}_in", source_id=m, target_id=tx_r, edge_type=EdgeType.ADDR_TO_TX, value=val, timestamp=ts_cursor, temporal_spacing_sec=float(sp)),
                    Edge(edge_id=f"e_{tx_r}_out", source_id=tx_r, target_id=r_addr, edge_type=EdgeType.TX_TO_ADDR, value=round(val * 0.99, 4), timestamp=ts_cursor, temporal_spacing_sec=float(sp)),
                ],
                campaign_id=cid, cadence_algorithm="dirichlet_split_synchronized_recombine", ground_truth_typology=Typology.FAN_OUT_FAN_IN
            ))

        sp_in = self.rng.randint(14, 28)
        ts_cursor += sp_in
        tx_in = self._new_tx_id(f"fanin_{cid[-4:]}")
        nodes2 = [
            Node(node_id=tx_in, node_type=NodeType.TRANSACTION, timestamp=ts_cursor, label=1),
            Node(node_id=sink, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1),
        ]
        edges2 = [Edge(edge_id=f"e_{tx_in}_sink", source_id=tx_in, target_id=sink, edge_type=EdgeType.TX_TO_ADDR, value=round(total * 0.985, 4), timestamp=ts_cursor, temporal_spacing_sec=float(sp_in))]
        for i, (m, val) in enumerate(relay_addrs + list(zip(mule_addrs[2:], splits[2:]))):
            nodes2.append(Node(node_id=m, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1))
            edges2.append(Edge(edge_id=f"e_{tx_in}_f{i}", source_id=m, target_id=tx_in, edge_type=EdgeType.ADDR_TO_TX, value=val, timestamp=ts_cursor, temporal_spacing_sec=float(sp_in)))
        events.append(TransactionEvent(
            correlation_id=f"corr_{cid}_fanin", timestamp=ts_cursor, nodes=nodes2, edges=edges2,
            campaign_id=cid, cadence_algorithm="dirichlet_split_synchronized_recombine", ground_truth_typology=Typology.FAN_OUT_FAN_IN
        ))
        return events

    def _build_cyclic_flow(self, ring_len: int = 4) -> List[TransactionEvent]:
        events = []
        cid = f"camp_cyc_{uuid.uuid4().hex[:5]}"
        ring = ["bc1q_ronin_bridge_hop" if self.rng.random() < 0.65 else f"bc1q_ring_{cid[-4:]}_0"] + [
            f"bc1q_ring_{cid[-4:]}_{i}" for i in range(1, ring_len)
        ]
        val, ts_cursor = round(self.rng.uniform(6.0, 18.0), 4), self.sim_ts
        for i in range(ring_len):
            sp = self.rng.randint(9, 21)
            ts_cursor += sp
            src, dst = ring[i], ring[(i + 1) % ring_len]
            tx_id = self._new_tx_id(f"cyc_{cid[-4:]}_{i}")
            val = round(val * 0.996, 4)
            events.append(TransactionEvent(
                correlation_id=f"corr_{cid}_s{i}", timestamp=ts_cursor,
                nodes=[
                    Node(node_id=src, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1),
                    Node(node_id=tx_id, node_type=NodeType.TRANSACTION, timestamp=ts_cursor, label=1),
                    Node(node_id=dst, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1),
                ],
                edges=[
                    Edge(edge_id=f"e_{tx_id}_in", source_id=src, target_id=tx_id, edge_type=EdgeType.ADDR_TO_TX, value=val, timestamp=ts_cursor, temporal_spacing_sec=float(sp)),
                    Edge(edge_id=f"e_{tx_id}_out", source_id=tx_id, target_id=dst, edge_type=EdgeType.TX_TO_ADDR, value=val, timestamp=ts_cursor, temporal_spacing_sec=float(sp)),
                ],
                campaign_id=cid, cadence_algorithm="periodic_ring_phase_shift", ground_truth_typology=Typology.CYCLIC_FLOW,
            ))
        return events

    def _build_smurfing(self, bursts: int = 5) -> List[TransactionEvent]:
        events = []
        cid = f"camp_smf_{uuid.uuid4().hex[:5]}"
        vault = "bc1q_hydra_smurf_vault" if self.rng.random() < 0.65 else f"bc1q_smurf_vault_{cid[-4:]}"
        ts_cursor = self.sim_ts
        for b in range(bursts):
            sp = self.rng.randint(7, 17)
            ts_cursor += sp
            tx_id = self._new_tx_id(f"smurf_{cid[-4:]}_{b}")
            src = f"bc1q_smurf_src_{cid[-4:]}_{b}"
            val = round(self.rng.uniform(0.43, 0.495), 4)
            events.append(TransactionEvent(
                correlation_id=f"corr_{cid}_b{b}", timestamp=ts_cursor,
                nodes=[
                    Node(node_id=src, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1),
                    Node(node_id=tx_id, node_type=NodeType.TRANSACTION, timestamp=ts_cursor, label=1),
                    Node(node_id=vault, node_type=NodeType.ADDRESS, timestamp=ts_cursor, label=1),
                ],
                edges=[
                    Edge(edge_id=f"e_{tx_id}_in", source_id=src, target_id=tx_id, edge_type=EdgeType.ADDR_TO_TX, value=val, timestamp=ts_cursor, temporal_spacing_sec=float(sp)),
                    Edge(edge_id=f"e_{tx_id}_out", source_id=tx_id, target_id=vault, edge_type=EdgeType.TX_TO_ADDR, value=val, timestamp=ts_cursor, temporal_spacing_sec=float(sp)),
                ],
                campaign_id=cid, cadence_algorithm="sub_threshold_uniform_dither", ground_truth_typology=Typology.SMURFING,
            ))
        return events
