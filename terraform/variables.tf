variable "aws_region" {
  description = "AWS region for provisioning MLOps infrastructure"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (e.g. dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Project name identifier used for naming resources"
  type        = string
  default     = "churnflow"
}

variable "ecr_repository_name" {
  description = "Name for Amazon Elastic Container Registry (ECR) repository"
  type        = string
  default     = "churn-api"
}

variable "s3_bucket_name" {
  description = "Unique S3 bucket name for DVC datasets and MLflow model artifacts"
  type        = string
  default     = "churnflow-mlops-artifacts-unique"
}
