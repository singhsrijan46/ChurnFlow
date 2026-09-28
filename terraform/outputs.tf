output "s3_bucket_name" {
  description = "Name of the created S3 artifact bucket"
  value       = module.s3_artifacts.bucket_name
}

output "s3_bucket_arn" {
  description = "ARN of the S3 artifact bucket"
  value       = module.s3_artifacts.bucket_arn
}

output "ecr_repository_url" {
  description = "URL of the created ECR container repository"
  value       = module.ecr_repository.repository_url
}

output "ecr_repository_arn" {
  description = "ARN of the ECR container repository"
  value       = module.ecr_repository.repository_arn
}
