import pytest
from fastapi.testclient import TestClient
import pandas as pd
from sklearn.linear_model import LogisticRegression

from api.main import app
from src.churn_ml.inference.predictor import get_predictor
from src.churn_ml.features.preprocessing import create_model_pipeline, save_pipeline


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "model_pipeline.joblib"

    # Create & persist dummy pipeline
    df = pd.DataFrame(
        {
            "tenure": [1, 24, 60, 2, 36, 70],
            "MonthlyCharges": [70.0, 50.0, 20.0, 85.0, 45.0, 25.0],
            "TotalCharges": [70.0, 1200.0, 1200.0, 170.0, 1620.0, 1750.0],
            "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 0, 0, 1, 0, 1],
            "Partner": ["No", "Yes", "Yes", "No", "Yes", "Yes"],
            "Dependents": ["No", "No", "Yes", "No", "Yes", "Yes"],
            "PhoneService": ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No", "Yes", "No", "No", "Yes", "No"],
            "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "Yes", "No internet service", "No", "Yes", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service", "No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service", "No", "Yes", "No internet service"],
            "TechSupport": ["No", "Yes", "No internet service", "No", "Yes", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service", "Yes", "No", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service", "Yes", "No", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "No", "Yes", "No", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Electronic check", "Mailed check", "Credit card (automatic)"],
        }
    )
    y = pd.Series([1, 0, 0, 1, 0, 0])
    pipeline = create_model_pipeline(LogisticRegression(max_iter=100))
    pipeline.fit(df, y)
    save_pipeline(pipeline, str(model_path))

    # Configure predictor
    predictor = get_predictor()
    predictor.model_path = str(model_path)
    predictor._load_model()

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "ChurnFlow"


def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert len(data["features_required"]) > 0


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "churn_http_requests_total" in response.text


def test_predict_single_endpoint(client):
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_category"] in ("Low", "Medium", "High")
    assert data["request_id"] is not None


def test_predict_single_invalid_data(client):
    invalid_sample = {
        "gender": "InvalidGender",
        "tenure": 500,
    }
    response = client.post("/predict", json=invalid_sample)
    assert response.status_code == 422  # Unprocessable Entity
