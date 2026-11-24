variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "asia-south1"
}

variable "vault_bucket" {
  description = "Pre-created GCS bucket for Elliptic++ data, model checkpoints, and PDF dossiers"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag (GitHub commit SHA) to force clean Cloud Run rollouts"
  type        = string
  default     = "latest"
}

variable "gemini_api_key" {
  description = "Optional Gemini API Key for LLM Narrative Agent (falls back to grounded template if blank)"
  type        = string
  default     = ""
  sensitive   = true
}
