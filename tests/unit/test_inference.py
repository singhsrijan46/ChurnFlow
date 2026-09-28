import pytest
import pandas as pd
from src.churn_ml.inference.predictor import ChurnPredictor, ModelNotReadyError
from src.churn_ml.features.preprocessing import create_model_pipeline
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def trained_predictor(tmp_path):
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

    import joblib
    model_path = tmp_path / "model_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    predictor = ChurnPredictor(model_path=str(model_path), model_version="1.0.0-test")
    return predictor


def test_predictor_is_ready(trained_predictor):
    assert trained_predictor.is_ready() is True


def test_predictor_uninitialized():
    pred = ChurnPredictor(model_path="non_existent_path.joblib")
    assert pred.is_ready() is False
    with pytest.raises(ModelNotReadyError):
        pred.predict_single({"tenure": 10})


def test_predictor_predict_single(trained_predictor):
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
    result = trained_predictor.predict_single(sample)
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_category"] in ("Low", "Medium", "High")
    assert result["model_version"] == "1.0.0-test"


def test_predictor_predict_batch(trained_predictor):
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
    results = trained_predictor.predict_batch([sample, sample])
    assert len(results) == 2
    for r in results:
        assert r["prediction"] in (0, 1)
