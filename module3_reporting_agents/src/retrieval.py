from typing import List, Tuple
from contracts.schemas import DocRef

class ThreatIntelRetriever:
    """
    Matches entity/wallet identifiers from the GNNExplainer minimal subgraph against a
    structured OSINT / OFAC / exploit post-mortem corpus.
    Strictly reports `attribution_available=False` on anonymized wallets instead of fabricating context.
    """
    CORPUS = {
        "bc1q_lazarus_peel_root": DocRef(
            doc_id="OSINT-OFAC-2025-412",
            source_title="OFAC SDN Advisory & Elliptic Cross-Chain Exploit Post-Mortem #412",
            entity_id="bc1q_lazarus_peel_root",
            span_start=18,
            span_end=154,
            excerpt="Wallet bc1q_lazarus_peel_root is linked to automated 10–25 second peeling chains draining bridge exploit proceeds across nested VASP endpoints."
        ),
        "bc1q_darknet_mixer_hub": DocRef(
            doc_id="OSINT-FATF-2026-089",
            source_title="FATF High-Risk Non-Custodial Tumbler & Mixer Registry 2026",
            entity_id="bc1q_darknet_mixer_hub",
            span_start=42,
            span_end=169,
            excerpt="Hub bc1q_darknet_mixer_hub executes rapid 5-way fan-out splitting into ephemeral mule addresses followed by synchronized consolidation."
        ),
        "bc1q_ronin_bridge_hop": DocRef(
            doc_id="OSINT-CERT-2025-118",
            source_title="CERT-In & Chainalysis Wash-Trading Ring Analysis #118",
            entity_id="bc1q_ronin_bridge_hop",
            span_start=12,
            span_end=140,
            excerpt="Address bc1q_ronin_bridge_hop participated in circular 4-hop cyclic layering loops designed to obfuscate UTXO provenance."
        ),
        "bc1q_hydra_smurf_vault": DocRef(
            doc_id="OSINT-FIU-2026-204",
            source_title="FIU-IND Structuring & Smurfing Typology Bulletin #204",
            entity_id="bc1q_hydra_smurf_vault",
            span_start=31,
            span_end=162,
            excerpt="Consolidation vault bc1q_hydra_smurf_vault repeatedly aggregates sub-0.5 BTC bursts from disposable deposit wallets within 60-second windows."
        ),
    }

    def retrieve_for_nodes(self, node_ids: List[str]) -> Tuple[bool, List[DocRef]]:
        matches = [self.CORPUS[nid] for nid in node_ids if nid in self.CORPUS]
        if not matches:
            return False, []
        return True, matches
