# Autonomous Blockchain Investigator (ABI) — Cloud-Native Forensic Platform

Temporal Heterogeneous GNN (**Bit-CHetG + Amatriciana**) with PyTorch **GNNExplainer**, OSINT **RAG**, **Compliance Critic** rejection loop, **AMLSim** real-time streaming, and bureau-grade **Forensic PDF Dossiers** — built for **GCP Cloud Run**, **Terraform**, and **GitHub Actions (Keyless WIF)**.

## Architecture (Two Clean Loops)
1. **Loop A — Offline GPU Training (`training/`)**: Reads real **Elliptic++** CSVs from `gs://capstone-bitcoin-forensics-abi-vault/raw/elliptic_pp/Elliptic++ Dataset/`, fuses cross-timestep **AMLSim** typologies, runs the leakage-verified **Static vs. Temporal Bit-CHetG** ablation on GPU (Colab / Vertex AI), and uploads trained weights (`temporal_bitchetg.pt`) to GCS.
2. **Loop B — Automated CI/CD Deployment (`.github/workflows/`)**: Every `git push` to `main` runs contract & leakage tests, builds 3 lightweight microservices (`mod2-gnn`, `mod3-agents`, `mod4-runtime` with bundled React Forensics UI), pushes to GCP Artifact Registry via keyless WIF, and deploys to Cloud Run via Terraform.

## Deployment Flow
1. Push this repo to `main` — GitHub Actions automatically deploys all 3 Cloud Run services and prints your live **Investigator Dashboard URL** in the workflow summary.
2. Open `training/Train_ABI_EllipticPP_GPU.ipynb` in Google Colab (T4 GPU) or Vertex AI Workbench to train on the real Elliptic++ dataset in GCS (~10–20 min). Once weights upload to `gs://capstone-bitcoin-forensics-abi-vault/models/temporal_bitchetg.pt`, click **Reload Model** (or restart `mod2-gnn`) and your live Cloud Run app automatically upgrades to the trained checkpoint.
