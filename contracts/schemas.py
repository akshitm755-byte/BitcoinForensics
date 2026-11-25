from enum import Enum
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

FEATURE_KEYS = [
    "in_degree",
    "out_degree",
    "total_in_val",
    "total_out_val",
    "time_delta_prev",
    "fan_out_ratio",
    "hop2_neighbor_count",
]

LICIT_FEATURE_BASELINES = {
    "in_degree": 1.15,
    "out_degree": 1.12,
    "total_in_val": 0.48,
    "total_out_val": 0.46,
    "time_delta_prev": 4.20,  # minutes
    "fan_out_ratio": 0.58,
    "hop2_neighbor_count": 1.35,
}

class NodeType(str, Enum):
    TRANSACTION = "transaction"
    ADDRESS = "address"
    ENTITY = "entity"

class EdgeType(str, Enum):
    TX_TO_TX = "tx_to_tx"
    ADDR_TO_TX = "addr_to_tx"
    TX_TO_ADDR = "tx_to_addr"
    ADDR_TO_ADDR = "addr_to_addr"
    ADDR_TO_ENTITY = "addr_to_entity"

class Typology(str, Enum):
    LICIT = "licit"
    PEELING_CHAIN = "peeling_chain"
    FAN_OUT_FAN_IN = "fan_out_fan_in"
    CYCLIC_FLOW = "cyclic_flow"
    SMURFING = "smurfing"

TYPOLOGY_TO_IDX = {t: i for i, t in enumerate(Typology)}
IDX_TO_TYPOLOGY = {i: t for i, t in enumerate(Typology)}

class Node(BaseModel):
    node_id: str
    node_type: NodeType
    timestamp: int
    features: Dict[str, float] = Field(default_factory=dict)
    memory_norm: float = 0.0
    label: Optional[int] = None

class Edge(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    value: float
    timestamp: int
    temporal_spacing_sec: float = 0.0

class HetGraph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    metapaths: List[List[str]] = Field(
        default_factory=lambda: [
            ["address", "transaction", "address"],
            ["transaction", "transaction"],
            ["address", "address"],
        ]
    )

class TransactionEvent(BaseModel):
    correlation_id: str
    timestamp: int
    nodes: List[Node]
    edges: List[Edge]
    source: str = "amlsim_interlaced_stream"
    campaign_id: Optional[str] = None
    cadence_algorithm: Optional[str] = None
    ground_truth_typology: Optional[Typology] = None

class SplitManifest(BaseModel):
    train_max_ts: int
    val_max_ts: int
    test_max_ts: int
    train_node_ids: List[str]
    val_node_ids: List[str]
    test_node_ids: List[str]
    leakage_verified: bool

class Alert(BaseModel):
    alert_id: str
    correlation_id: str
    timestamp: int
    risk_score: float
    model_version: str
    model_checkpoint_status: str = "UNTRAINED_DAY1_FALLBACK"
    in_graph_memory_activation: float = 0.0
    matched_motif_slot: int = 0
    predicted_typology: Typology
    cadence_signature: str = "standard_jitter"
    window_bounds: Tuple[int, int]
    subgraph_nodes: List[Node]
    subgraph_edges: List[Edge]
    is_false_positive: bool = False

class HistoricCaseMatch(BaseModel):
    case_id: str
    case_name: str
    year: int
    source_domain: str
    source_url: str
    typology: str
    similarity_index: float
    escalation_rate: float
    amount_lost_usd: Optional[str] = None
    temporal_spacing_pattern: str
    pseudorandomization_signature: str
    summary: str
    fetched_live: bool = False

class DocRef(BaseModel):
    doc_id: str
    source_title: str
    source_url: str = ""
    entity_id: str
    span_start: int
    span_end: int
    excerpt: str

class Evidence(BaseModel):
    evidence_id: str
    alert_id: str
    correlation_id: str
    explainer_version: str
    minimal_nodes: List[str]
    minimal_edges: List[str]
    node_importance_mask: Dict[str, float]
    edge_importance_mask: Dict[str, float]
    feature_attributions: Dict[str, float]
    statistical_correlations: Dict[str, float] = Field(default_factory=dict)
    historic_case_matches: List[HistoricCaseMatch] = Field(default_factory=list)
    attribution_available: bool
    doc_refs: List[DocRef] = Field(default_factory=list)

    def resolve(self, ref_id: str) -> bool:
        valid_ids = (
            set(self.minimal_nodes)
            | set(self.minimal_edges)
            | set(self.feature_attributions.keys())
            | {d.doc_id for d in self.doc_refs}
            | {c.case_id for c in self.historic_case_matches}
        )
        return ref_id in valid_ids

class Claim(BaseModel):
    claim_id: str
    sentence: str
    evidence_refs: List[str]
    is_grounded: bool = False

class FigureExplanation(BaseModel):
    figure_id: str          # "Figure 1" .. "Figure 5"
    title: str
    explanation: str        # 2-4 sentence multimodal visual + analytical breakdown
    evidence_refs: List[str] = Field(default_factory=list)
    is_grounded: bool = False

class Report(BaseModel):
    report_id: str
    alert_id: str
    correlation_id: str
    faithfulness_score: float
    completeness_score: float = 1.0
    critic_verdict: str
    narrative_engine: str = "deterministic_grounded"
    trigger_mode: str = "serverless_pubsub"
    executive_summary: str = ""
    claims: List[Claim]
    figure_explanations: List[FigureExplanation] = Field(default_factory=list)
    evidence: Evidence
    pdf_gcs_uri: Optional[str] = None
    report_gcs_uri: Optional[str] = None
    pdf_local_path: Optional[str] = None
