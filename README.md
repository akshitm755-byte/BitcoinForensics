# Autonomous Blockchain Investigator (ABI) v4 — Multi-Agentic & Serverless Forensics Platform

Temporal Heterogeneous GNN (**Bit-CHetG + Amatriciana + In-Graph Memory**) with **Serverless Pub/Sub-Triggered Multi-Agent Reporting** (**PyTorch GNNExplainer + OSINT Historic Cases Scraper + RAG + Gemini 3.8 Flash Documentation Agent + Compliance Critic**), **Interlaced AMLSim Streaming**, and **Bureau-Grade Multi-Chart PDF Dossiers** on **GCP Cloud Run**.

## What's New in v4
1. **True Event-Driven Serverless Module 3 (GCP Pub/Sub Push -> Cloud Run)**:
   - `mod4-investigator-dashboard` publishes flagged `Alert` payloads to `aml-alerts-queue` (Cloud Pub/Sub).
   - A Terraform-managed **Pub/Sub Push Subscription** wakes up `mod3-reporting-agents` (`/pubsub/explain-and-report`) serverlessly with automatic retry backoff, scaling from `0 -> N -> 0`.
   - `mod3` persists sealed `Report` JSON + multi-page `.pdf` dossiers to GCS (`gs://<vault>/reports/` & `dossiers/`), eliminating circular service dependencies.
2. **Gemini 3.8 Flash Multi-Agent Documentation + Abstractive Executive Summary**:
   - Uses `gemini-3.8-flash` (with automatic `gemini-3.5-flash-lite` and grounded deterministic fallback) when `GEMINI_API_KEY` is added to GitHub Secrets.
   - Generates both **Compliance-Critic-verified evidence claims** (`Evidence.resolve()`) AND a unique, plain-English **Concluding Executive Summary** explaining the attack in accessible human language.
3. **Thread-Safe Multi-Chart PDFs & Un-Squeezed Historic Cases Chart**:
   - Eliminates concurrent Matplotlib race conditions via isolated `FigureCanvasAgg` objects.
   - Fixes title clipping and legend overlap on the OSINT Historic Cases Similarity vs. Escalation chart, adds direct bar labels, and renders Section 5 (Executive Summary Brief).
4. **Differentiated OSINT Similarity Index**:
   - Combines cosine alignment + normalized Euclidean distance decay across hack technique, classification, volume, and temporal cadence so live DeFiLlama records and historic AML precedents produce realistic, distinct similarity indices.
5. **Backwards-Compatible Checkpoint Loading & Validation-Calibrated Ablation**:
   - Automatically loads `models/temporal_bitchetg_v4.pt` if present, or falls back seamlessly to your already-trained `models/temporal_bitchetg_v3.pt` in GCS—zero forced retraining required on deploy!
