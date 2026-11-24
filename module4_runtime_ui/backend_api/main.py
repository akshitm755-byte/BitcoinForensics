import asyncio
import hashlib
import json
import os
import time
from typing import Dict, List, Optional
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from contracts.schemas import Alert, Report, Typology
from module1_data_pipeline.src.amlsim_stream import AMLSimStreamGenerator

app = FastAPI(title="Module 4 — Investigator Forensics Runtime & Live AMLSim Streamer")

VAULT_BUCKET = os.getenv("VAULT_BUCKET", "")
GNN_SERVICE_URL = os.getenv("GNN_SERVICE_URL", "http://localhost:8002").rstrip("/")
REPORTING_SERVICE_URL = os.getenv("REPORTING_SERVICE_URL", "http://localhost:8003").rstrip("/")

streamer = AMLSimStreamGenerator()
STREAM_STATE = {
    "running": False,
    "interval_sec": 1.6,
    "events_processed": 0,
    "checkpoint_status": "CONNECTING",
}

ALERTS_DB: Dict[str, Alert] = {}
REPORTS_BY_ALERT: Dict[str, Report] = {}
REPORTS_DB: Dict[str, Report] = {}
AUDIT_TRACES: Dict[str, List[dict]] = {}
DETECTION_LATENCIES_MS: List[float] = []
RECENT_GRAPH_BATCHES: List[dict] = []


def record_audit_trace(correlation_id: str, stage: str, payload: dict):
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    AUDIT_TRACES.setdefault(correlation_id, []).append({
        "stage": stage,
        "timestamp_ms": int(time.time() * 1000),
        "sha256_hash": digest,
    })


async def process_one_event():
    ev = streamer.next_event(inject_probability=0.20)
    record_audit_trace(ev.correlation_id, "EVENT_INGESTED", {"ts": ev.timestamp, "edges": len(ev.edges)})

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(f"{GNN_SERVICE_URL}/score-event", json=ev.model_dump())
        resp.raise_for_status()
        data = resp.json()

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    DETECTION_LATENCIES_MS.append(elapsed_ms)
    if len(DETECTION_LATENCIES_MS) > 500:
        DETECTION_LATENCIES_MS.pop(0)

    STREAM_STATE["events_processed"] += 1
    STREAM_STATE["checkpoint_status"] = data.get("checkpoint_status", "ACTIVE")

    if data.get("alert_emitted"):
        alert = Alert(**data["alert"])
        ALERTS_DB[alert.alert_id] = alert
        record_audit_trace(alert.correlation_id, "ALERT_EMITTED", {"alert_id": alert.alert_id, "score": alert.risk_score})
        RECENT_GRAPH_BATCHES.append({
            "batch_id": STREAM_STATE["events_processed"],
            "alert_id": alert.alert_id,
            "is_alert": True,
            "risk_score": alert.risk_score,
            "nodes": [n.model_dump() for n in alert.subgraph_nodes],
            "edges": [e.model_dump() for e in alert.subgraph_edges],
        })
        asyncio.create_task(fetch_report_for_alert(alert))
    else:
        RECENT_GRAPH_BATCHES.append({
            "batch_id": STREAM_STATE["events_processed"],
            "alert_id": None,
            "is_alert": False,
            "risk_score": data.get("risk_score", 0.0),
            "nodes": data.get("subgraph_nodes", [n.model_dump() for n in ev.nodes]),
            "edges": data.get("subgraph_edges", [e.model_dump() for e in ev.edges]),
        })

    if len(RECENT_GRAPH_BATCHES) > 25:
        RECENT_GRAPH_BATCHES.pop(0)


async def fetch_report_for_alert(alert: Alert):
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(f"{REPORTING_SERVICE_URL}/explain-and-report", json=alert.model_dump())
            resp.raise_for_status()
            rep = Report(**resp.json())
            # Enforce Evidence Contract on arrival
            for clm in rep.claims:
                if not all(rep.evidence.resolve(r) for r in clm.evidence_refs):
                    record_audit_trace(rep.correlation_id, "CONTRACT_VIOLATION_REJECTED", {"claim": clm.claim_id})
                    return
            REPORTS_DB[rep.report_id] = rep
            REPORTS_BY_ALERT[alert.alert_id] = rep
            record_audit_trace(rep.correlation_id, "REPORT_SEALED", {"report_id": rep.report_id, "faithfulness": rep.faithfulness_score})
    except Exception as exc:
        record_audit_trace(alert.correlation_id, "REPORT_GENERATION_ERROR", {"error": str(exc)})


async def stream_loop():
    while True:
        if STREAM_STATE["running"]:
            try:
                await process_one_event()
            except Exception as exc:
                print(f"[StreamLoop] Step error: {exc}")
        await asyncio.sleep(STREAM_STATE["interval_sec"])


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(stream_loop())


@app.post("/api/stream/control")
async def control_stream(action: str, interval_sec: Optional[float] = None):
    if interval_sec is not None:
        STREAM_STATE["interval_sec"] = max(0.4, min(5.0, interval_sec))
    if action == "start":
        STREAM_STATE["running"] = True
    elif action == "pause":
        STREAM_STATE["running"] = False
    elif action == "step":
        await process_one_event()
    return STREAM_STATE


@app.post("/api/stream/inject/{typology}")
async def inject_typology(typology: str):
    try:
        typ = Typology(typology)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown typology: {typology}")
    count = streamer.enqueue_typology(typ)
    # Immediately fire first hop so the investigator sees instant feedback
    await process_one_event()
    return {"injected": typ.value, "queued_events": count}


@app.get("/api/dashboard/state")
def dashboard_state():
    arr = np.array(DETECTION_LATENCIES_MS) if DETECTION_LATENCIES_MS else np.array([0.0])
    sorted_alerts = sorted(ALERTS_DB.values(), key=lambda a: (a.timestamp, a.risk_score), reverse=True)[:40]
    return {
        "stream": STREAM_STATE,
        "latency": {
            "boundary": "event_arrival_to_alert_emitted",
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        },
        "latest_batch": RECENT_GRAPH_BATCHES[-1] if RECENT_GRAPH_BATCHES else None,
        "alerts": [
            {
                **a.model_dump(exclude={"subgraph_nodes", "subgraph_edges"}),
                "node_count": len(a.subgraph_nodes),
                "has_report": a.alert_id in REPORTS_BY_ALERT,
                "report_id": REPORTS_BY_ALERT[a.alert_id].report_id if a.alert_id in REPORTS_BY_ALERT else None,
            }
            for a in sorted_alerts
        ],
    }


@app.get("/api/alerts/{alert_id}/full")
def get_alert_full(alert_id: str):
    if alert_id not in ALERTS_DB:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert = ALERTS_DB[alert_id]
    rep = REPORTS_BY_ALERT.get(alert_id)
    return {
        "alert": alert.model_dump(),
        "report": rep.model_dump() if rep else None,
        "audit_trace": AUDIT_TRACES.get(alert.correlation_id, []),
    }


@app.post("/api/alerts/{alert_id}/false-positive")
def toggle_false_positive(alert_id: str):
    if alert_id not in ALERTS_DB:
        raise HTTPException(status_code=404, detail="Alert not found")
    ALERTS_DB[alert_id].is_false_positive = not ALERTS_DB[alert_id].is_false_positive
    record_audit_trace(
        ALERTS_DB[alert_id].correlation_id,
        "INVESTIGATOR_FEEDBACK",
        {"is_false_positive": ALERTS_DB[alert_id].is_false_positive},
    )
    return {"alert_id": alert_id, "is_false_positive": ALERTS_DB[alert_id].is_false_positive}


@app.get("/api/reports/{report_id}/export")
def export_report(report_id: str, format: str = "pdf"):
    if report_id not in REPORTS_DB:
        raise HTTPException(status_code=404, detail="Report not found")
    rep = REPORTS_DB[report_id]
    if format == "json":
        return rep.model_dump()

    if rep.pdf_gcs_uri and VAULT_BUCKET:
        from google.cloud import storage
        blob_name = rep.pdf_gcs_uri.replace(f"gs://{VAULT_BUCKET}/", "")
        pdf_bytes = storage.Client().bucket(VAULT_BUCKET).blob(blob_name).download_as_bytes()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{report_id}.pdf"'},
        )
    if rep.pdf_local_path and os.path.exists(rep.pdf_local_path):
        return FileResponse(rep.pdf_local_path, media_type="application/pdf", filename=f"{report_id}.pdf")
    raise HTTPException(status_code=404, detail="PDF file not available in vault")


# Mount built React Forensics Dashboard at root "/"
STATIC_UI_DIR = "/app/static_ui"
if os.path.exists(STATIC_UI_DIR):
    app.mount("/", StaticFiles(directory=STATIC_UI_DIR, html=True), name="static_ui")
