import os
from fastapi import FastAPI
from contracts.schemas import Alert
from module2_detection_gnn.src.torch_model import load_model_from_gcs_or_init
from module3_reporting_agents.src.evidence_builder import EvidenceArtifactBuilder
from module3_reporting_agents.src.orchestrator import AgenticForensicOrchestrator
from module3_reporting_agents.src.pdf_exporter import ForensicDossierPDFExporter

app = FastAPI(title="Module 3 — GNNExplainer + OSINT RAG + Compliance Critic + PDF Service")

VAULT_BUCKET = os.getenv("VAULT_BUCKET", "")
MODEL_BLOB_PATH = os.getenv("MODEL_BLOB_PATH", "models/temporal_bitchetg.pt")

model, checkpoint_status = load_model_from_gcs_or_init(VAULT_BUCKET, MODEL_BLOB_PATH)
builder = EvidenceArtifactBuilder(model)
orchestrator = AgenticForensicOrchestrator()


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint_status": checkpoint_status}


@app.post("/explain-and-report")
def explain_and_report(alert: Alert):
    evidence = builder.build_sealed_evidence(alert)
    report = orchestrator.generate_verified_report(alert, evidence)
    ForensicDossierPDFExporter.export_pdf(alert, report, vault_bucket=VAULT_BUCKET or None)
    return report.model_dump()
