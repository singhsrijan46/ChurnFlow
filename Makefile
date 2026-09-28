.PHONY: help install test lint train evaluate drift retrain api docker-build docker-up docker-down dvc-repro clean

help:
	@echo "======================================================================="
	@echo " ChurnFlow: Enterprise Customer Churn Prediction MLOps Platform"
	@echo "======================================================================="
	@echo "Available commands:"
	@echo "  make install       Install python package and dependencies"
	@echo "  make test          Run full unit and integration test suite"
	@echo "  make lint          Run flake8 and code formatting checks"
	@echo "  make train         Run data ingestion and train champion ML model"
	@echo "  make evaluate      Evaluate candidate model pipeline"
	@echo "  make drift         Run data drift detection between reference and current"
	@echo "  make retrain       Execute automated retraining and governance loop"
	@echo "  make api           Run local FastAPI development server"
	@echo "  make docker-build  Build Docker image for the inference API"
	@echo "  make docker-up     Start full local stack (API, MLflow, Postgres, Prometheus, Grafana)"
	@echo "  make docker-down   Stop and clean up local Docker containers"
	@echo "  make dvc-repro     Reproduce full DVC data and ML pipeline"
	@echo "  make clean         Clean temporary caches and artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests/ -v -p no:cov -p no:langsmith

lint:
	flake8 src/ api/ scripts/ tests/ --count --max-line-length=120 --statistics

train:
	python scripts/download_data.py
	python scripts/validate_data.py
	python scripts/train_model.py --model-type all --register

evaluate:
	python scripts/evaluate_model.py

drift:
	python scripts/check_drift.py

retrain:
	python scripts/retrain_pipeline.py --force

api:
	python api/main.py

docker-build:
	docker build -t churn-api:latest -f docker/Dockerfile .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

dvc-repro:
	dvc repro

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
