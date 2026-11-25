# Autonomous Blockchain Investigator (ABI) v3 — Cloud-Native Forensics Platform

## What's New in v3
1. **In-Graph Structural Memory + Amatriciana Temporal Memory (`TemporalBitCHetGNet`)**:
   - Combines Bit-CHetG heterogeneous SAGE blocks (TSE + AFA), Amatriciana continuous time-decay `GRUCell` memory, AND an **In-Graph Memory Bank** (neighborhood temporal memory aggregation + learnable motif prototype slots) so newly arriving transactions read persistent relational state across multi-hop campaigns.
   - Includes a **3-Way Scientific Ablation** in `training/train_elliptic_pp_gpu.py`:
     1. `Static Bit-CHetG` (no memory)
     2. `Temporal Bit-CHetG + Amatriciana` (node decay memory)
     3. `Full Hybrid: Bit-CHetG + Amatriciana + In-Graph Memory`
2. **Realistic Interlaced AMLSim Stream (`module1_data_pipeline/src/amlsim_stream.py`)**:
   - Low realistic baseline illicit rate (~4–6% campaign prevalence) with **multi-transaction illicit attempts interlaced between genuine licit transactions** using pseudorandomized temporal spacing (log-normal jitter, Poisson bursts, uniform peel cadence).
3. **Minimalist Bureau Forensics UI (`module4_runtime_ui/frontend_ui/`)**:
   - Clean dark-mode interface (`Inter` & `JetBrains Mono`) with a **dynamic, pan/zoomable, clickable graph topology**.
   - Clicking **any node or edge** opens a **Timestamped Feature Insight Window** with node/edge importance masks, memory norms, feature deviations vs. licit baseline, and plain-English forensic diagnostics.
4. **GNNExplainer + OSINT & Historic Cases Scraper Agent + Vector RAG + Gemini LLM + Rich Visual PDF Dossiers (`module3_reporting_agents/`)**:
   - Live/cached OSINT & Historic Cases Scraper (`api.llama.fi/hacks` + curated crypto-forensics registry) with **cosine similarity indexing** and **historic bureau escalation rate correlation**.
   - Generates multi-page PDF dossiers containing: Minimal Flagged Subgraph, Feature Attribution Bars, **Feature Correlation Heatmap**, **Temporal Spacing vs. Value Scatter Plot**, **Historic Case Similarity & Escalation Chart**, and Compliance-Critic-verified prose.
