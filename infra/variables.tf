variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "asia-south1"
}

variable "vault_bucket" {
  type = string
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "gemini_api_key" {
  type      = string
  default   = ""
  sensitive = true
}

variable "gemini_model" {
  type    = string
  default = "gemini-3.5-flash-lite"
}
