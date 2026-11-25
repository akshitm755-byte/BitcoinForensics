import base64
import json
import os
from fastapi import FastAPI, HTTPException, Request
from contracts.schemas import Alert
from module2_detection_gnn.src.torch_model import load_model_from_gcs_or_init
from module3_reporting_agents.src.evidence_builder import EvidenceArtifactBuilder
from module3_reporting_agents.src.orchestrator import AgenticForensicOrchestrator
from module3_reporting_agents.src.pdf_exporter import ForensicDossierPDFExporter

app = FastAPI(title="Module 3 — Serverless Multimodal GNNExplainer + OSINT Scraper + Gemini 3.8 Flash + PDF Service")

VAULT_BUCKET = os.getenv("VAULT_BUCKET", "")
MODEL_BLOB_PATH = os.getenv("MODEL_BLOB_PATH", "models/temporal_bitchetg_v4.pt")
FALLBACK_BLOB_PATH = os.getenv("FALLBACK_MODEL_BLOB_PATH", "models/temporal_bitchetg_v3.pt")

model, checkpoint_status = load_model_from_gcs_or_init(VAULT_BUCKET, MODEL_BLOB_PATH, FALLBACK_BLOB_PATH)
builder = EvidenceArtifactBuilder(model, enable_live_scrape=True)
orchestrator = AgenticForensicOrchestrator()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint_status": checkpoint_status,
        "gemini_enabled": bool(orchestrator.gemini_api_key),
        "gemini_model": orchestrator.primary_model,
        "multimodal_figure_analysis": True,
    }


@app.post("/reload-checkpoint")
def reload_checkpoint():
    global model, checkpoint_status, builder
    model, checkpoint_status = load_model_from_gcs_or_init(VAULT_BUCKET, MODEL_BLOB_PATH, FALLBACK_BLOB_PATH)
    builder = EvidenceArtifactBuilder(model, enable_live_scrape=True)
    return {"reloaded": True, "checkpoint_status": checkpoint_status}


def _run_reporting_pipeline(alert: Alert, trigger_mode: str) -> dict:
    # 1. Run PyTorch GNNExplainer + OSINT Historic Cases Scraper
    evidence = builder.build_sealed_evidence(alert)
    # 2. Render labeled Figures 1-5 + exact chart metrics BEFORE calling the multimodal LLM
    charts = ForensicDossierPDFExporter.render_labeled_charts(alert, evidence)
    # 3. Multimodal Gemini Documentation Agent inspects all 5 figures + verifies via Compliance Critic
    report = orchestrator.generate_verified_report(alert, evidence, trigger_mode=trigger_mode, charts=charts)
    # 4. Assemble multi-page PDF embedding the already-rendered figures + their dedicated explanations
    ForensicDossierPDFExporter.export_pdf(alert, report, vault_bucket=VAULT_BUCKET or None, charts=charts)
    return report.model_dump()


@app.post("/explain-and-report")
def explain_and_report(alert: Alert):
    return _run_reporting_pipeline(alert, trigger_mode="direct_http")


@app.post("/pubsub/explain-and-report")
async def pubsub_explain_and_report(request: Request):
    envelope = await request.json()
    if "message" not in envelope or "data" not in envelope["message"]:
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub push envelope")
    raw_bytes = base64.b64decode(envelope["message"]["data"])
    alert = Alert(**json.loads(raw_bytes.decode("utf-8")))
    result = _run_reporting_pipeline(alert, trigger_mode="serverless_pubsub")
    return {"status": "acked", "report_id": result["report_id"], "alert_id": alert.alert_id}
