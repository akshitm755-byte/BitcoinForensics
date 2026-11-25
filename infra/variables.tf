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
  description = "GCS vault bucket for Elliptic++ data, model checkpoints, reports, and PDFs"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag (commit SHA)"
  type        = string
  default     = "latest"
}

variable "gemini_api_key" {
  description = "Google AI Studio Gemini API Key (optional; falls back cleanly if blank)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "gemini_model" {
  description = "Primary Gemini API model ID for Multi-Agent Documentation"
  type        = string
  default     = "gemini-3.8-flash"
}
