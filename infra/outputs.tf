output "investigator_dashboard_url" {
  value = google_cloud_run_v2_service.mod4_runtime.uri
}

output "gnn_service_url" {
  value = google_cloud_run_v2_service.mod2_gnn.uri
}

output "reporting_agents_url" {
  value = google_cloud_run_v2_service.mod3_agents.uri
}
