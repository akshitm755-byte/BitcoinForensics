terraform {
  required_version = ">= 1.5.0"

  backend "gcs" {
    prefix = "terraform/abi-state"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.20"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  registry = "${var.region}-docker.pkg.dev/${var.project_id}/abi-forensics-repo"
}

# Pub/Sub Event Backbone
resource "google_pubsub_topic" "tx_events" {
  name = "tx-events-stream"
}

resource "google_pubsub_topic" "aml_alerts" {
  name = "aml-alerts-queue"
}

# Module 2: Temporal Bit-CHetG + Amatriciana PyTorch Inference Service
resource "google_cloud_run_v2_service" "mod2_gnn" {
  name     = "mod2-gnn-inference"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
    containers {
      image = "${local.registry}/mod2-gnn:${var.image_tag}"
      ports {
        container_port = 8080
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
      env {
        name  = "VAULT_BUCKET"
        value = var.vault_bucket
      }
      env {
        name  = "MODEL_BLOB_PATH"
        value = "models/temporal_bitchetg.pt"
      }
      env {
        name  = "ALERT_THRESHOLD"
        value = "0.58"
      }
    }
  }
}

# Module 3: PyTorch GNNExplainer + OSINT RAG + Compliance Critic + PDF Dossier Service
resource "google_cloud_run_v2_service" "mod3_agents" {
  name     = "mod3-reporting-agents"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
    containers {
      image = "${local.registry}/mod3-agents:${var.image_tag}"
      ports {
        container_port = 8080
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
      env {
        name  = "VAULT_BUCKET"
        value = var.vault_bucket
      }
      env {
        name  = "MODEL_BLOB_PATH"
        value = "models/temporal_bitchetg.pt"
      }
      env {
        name  = "GEMINI_API_KEY"
        value = var.gemini_api_key
      }
    }
  }
}

# Module 4: AMLSim Streamer, Runtime API, Audit Logger & Bundled React Forensics UI
resource "google_cloud_run_v2_service" "mod4_runtime" {
  name     = "mod4-investigator-dashboard"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }
    containers {
      image = "${local.registry}/mod4-runtime:${var.image_tag}"
      ports {
        container_port = 8080
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
      env {
        name  = "VAULT_BUCKET"
        value = var.vault_bucket
      }
      env {
        name  = "GNN_SERVICE_URL"
        value = google_cloud_run_v2_service.mod2_gnn.uri
      }
      env {
        name  = "REPORTING_SERVICE_URL"
        value = google_cloud_run_v2_service.mod3_agents.uri
      }
      env {
        name  = "TX_STREAM_TOPIC"
        value = google_pubsub_topic.tx_events.name
      }
    }
  }
}

# Public invoker bindings for Capstone Demo & Service-to-Service HTTP calls
resource "google_cloud_run_v2_service_iam_member" "mod2_public" {
  name     = google_cloud_run_v2_service.mod2_gnn.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "mod3_public" {
  name     = google_cloud_run_v2_service.mod3_agents.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "mod4_public" {
  name     = google_cloud_run_v2_service.mod4_runtime.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
