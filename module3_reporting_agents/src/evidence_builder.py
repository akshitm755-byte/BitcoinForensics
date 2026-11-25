import uuid
from contracts.schemas import Alert, Evidence
from module2_detection_gnn.src.torch_model import TemporalBitCHetGNet
from module3_reporting_agents.src.explainer import PyTorchGNNExplainerEngine
from module3_reporting_agents.src.osint_scraper import OSINTHistoricCasesScraperAgent


class EvidenceArtifactBuilder:
    def __init__(self, model: TemporalBitCHetGNet, enable_live_scrape: bool = True):
        self.explainer = PyTorchGNNExplainerEngine(model)
        self.scraper = OSINTHistoricCasesScraperAgent()
        self.enable_live_scrape = enable_live_scrape

    def build_sealed_evidence(self, alert: Alert) -> Evidence:
        min_nodes, min_edges, n_mask, e_mask, feat_attr, stats = self.explainer.explain_alert(alert)
        all_ids = list(set(min_nodes) | {n.node_id for n in alert.subgraph_nodes})
        attr_avail, doc_refs, historic_matches = self.scraper.retrieve_and_correlate(
            alert, all_ids, enable_live_scrape=self.enable_live_scrape
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
            statistical_correlations=stats,
            historic_case_matches=historic_matches,
            attribution_available=attr_avail,
            doc_refs=doc_refs,
        )
