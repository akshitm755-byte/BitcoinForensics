import uuid
from contracts.schemas import Alert, Evidence
from module2_detection_gnn.src.torch_model import TemporalBitCHetGNet
from module3_reporting_agents.src.explainer import PyTorchGNNExplainerEngine
from module3_reporting_agents.src.retrieval import ThreatIntelRetriever

class EvidenceArtifactBuilder:
    def __init__(self, model: TemporalBitCHetGNet):
        self.explainer = PyTorchGNNExplainerEngine(model)
        self.retriever = ThreatIntelRetriever()

    def build_sealed_evidence(self, alert: Alert) -> Evidence:
        min_nodes, min_edges, n_mask, e_mask, feat_attr = self.explainer.explain_alert(alert)
        attr_avail, doc_refs = self.retriever.retrieve_for_nodes(
            list(set(min_nodes) | {n.node_id for n in alert.subgraph_nodes})
        )
        return Evidence(
            evidence_id=f"EVID-{uuid.uuid4().hex[:8].upper()}",
            alert_id=alert.alert_id,
            correlation_id=alert.correlation_id,
            explainer_version=self.explainer.EXPLAINER_VERSION,
            minimal_nodes=min_nodes,
            minimal_edges=min_edges,
            node_importance_mask=n_mask,
            edge_importance_mask=e_mask,
            feature_attributions=feat_attr,
            attribution_available=attr_avail,
            doc_refs=doc_refs,
        )
