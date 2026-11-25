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
  registry   = "${var.region}-docker.pkg.dev/${var.project_id}/abi-forensics-repo"
  compute_sa = "454908242756-compute@developer.gserviceaccount.com"
}

# Grant Cloud Run runtime service account read/write access to GCS vault (checkpoints, reports, PDFs)
resource "google_storage_bucket_iam_member" "cloudrun_vault_access" {
  bucket = var.vault_bucket
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${local.compute_sa}"
}

# Pub/Sub Event Backbone
resource "google_pubsub_topic" "tx_events" {
  name = "tx-events-stream"
}

resource "google_pubsub_topic" "aml_alerts" {
  name = "aml-alerts-queue"
}

# Grant Cloud Run runtime service account permission to publish alerts into Pub/Sub
resource "google_pubsub_topic_iam_member" "cloudrun_alerts_publisher" {
  topic  = google_pubsub_topic.aml_alerts.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${local.compute_sa}"
}

# Module 2: Temporal Bit-CHetG + Amatriciana + In-Graph Memory Service
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
        value = "models/temporal_bitchetg_v4.pt"
      }
      env {
        name  = "FALLBACK_MODEL_BLOB_PATH"
        value = "models/temporal_bitchetg_v3.pt"
      }
      env {
        name  = "ALERT_THRESHOLD"
        value = "0.52"
      }
    }
  }
}

# Module 3: Serverless GNNExplainer + OSINT Scraper + Gemini 3.8 Flash + Visual PDF Service
resource "google_cloud_run_v2_service" "mod3_agents" {
  name     = "mod3-reporting-agents"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = 0
      max_instance_count = 4
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
        value = "models/temporal_bitchetg_v4.pt"
      }
      env {
        name  = "FALLBACK_MODEL_BLOB_PATH"
        value = "models/temporal_bitchetg_v3.pt"
      }
      env {
        name  = "GEMINI_API_KEY"
        value = var.gemini_api_key
      }
      env {
        name  = "GEMINI_MODEL"
        value = var.gemini_model
      }
    }
  }
}

# Serverless Event-Driven Push Subscription: Pub/Sub `aml-alerts-queue` -> Module 3 Cloud Run
resource "google_pubsub_subscription" "alerts_push_to_mod3" {
  name                 = "aml-alerts-to-mod3-push"
  topic                = google_pubsub_topic.aml_alerts.name
  ack_deadline_seconds = 120

  push_config {
    push_endpoint = "${google_cloud_run_v2_service.mod3_agents.uri}/pubsub/explain-and-report"
  }

  retry_policy {
    minimum_backoff = "5s"
    maximum_backoff = "60s"
  }
}

# Module 4: Interlaced AMLSim Streamer, Runtime API, Audit Logger & Minimalist React UI
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
        name  = "GCP_PROJECT_ID"
        value = var.project_id
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
      env {
        name  = "ALERT_PUBSUB_TOPIC"
        value = google_pubsub_topic.aml_alerts.name
      }
      env {
        name  = "ENABLE_PUBSUB_TRIGGER"
        value = "true"
      }
    }
  }
}

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
