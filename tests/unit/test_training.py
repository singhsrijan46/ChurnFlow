import pytest
import pandas as pd
from src.churn_ml.training.train import build_model_instance
from src.churn_ml.training.evaluate import (
    evaluate_model,
    check_promotion_eligibility,
    EvaluationMetrics,
)
from src.churn_ml.features.preprocessing import create_model_pipeline


@pytest.fixture
def mock_train_test_data():
    data = {
        "tenure": [1, 24, 60, 2, 36, 70, 5, 48, 12, 50],
        "MonthlyCharges": [70.0, 50.0, 20.0, 85.0, 45.0, 25.0, 75.0, 30.0, 65.0, 22.0],
        "TotalCharges": [70.0, 1200.0, 1200.0, 170.0, 1620.0, 1750.0, 375.0, 1440.0, 780.0, 1100.0],
        "gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        "Partner": ["No", "Yes", "Yes", "No", "Yes", "Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "No", "Yes"],
        "PhoneService": ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes"],
        "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic", "DSL", "No", "Fiber optic", "No", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service", "No", "Yes", "No internet service", "No", "No internet service", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service", "No", "Yes", "No internet service", "Yes", "No internet service", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service", "No", "Yes", "No internet service", "No", "No internet service", "No", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service", "No", "Yes", "No internet service", "No", "No internet service", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service", "Yes", "No", "No internet service", "No", "No internet service", "No", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service", "Yes", "No", "No internet service", "No", "No internet service", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "Month-to-month", "Two year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Electronic check", "Mailed check", "Credit card (automatic)", "Electronic check", "Mailed check", "Electronic check", "Bank transfer (automatic)"],
        "Churn": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    train_df = df.iloc[:8].copy()
    test_df = df.iloc[8:].copy()
    return train_df, test_df


def test_build_model_instance():
    lr = build_model_instance("logistic_regression")
    assert lr is not None
    assert hasattr(lr, "fit")

    xgb = build_model_instance("xgboost")
    assert xgb is not None
    assert hasattr(xgb, "fit")


def test_evaluate_model_metrics(mock_train_test_data):
    train_df, test_df = mock_train_test_data
    from src.churn_ml.features.preprocessing import prepare_training_data

    X_train, y_train = prepare_training_data(train_df)
    X_test, y_test = prepare_training_data(test_df)

    clf = build_model_instance("logistic_regression")
    pipeline = create_model_pipeline(clf)
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test)
    assert isinstance(metrics, EvaluationMetrics)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0


def test_check_promotion_eligibility():
    good_metrics = EvaluationMetrics(
        accuracy=0.85, precision=0.80, recall=0.75, f1=0.77, roc_auc=0.88, log_loss=0.35, brier_score=0.10, confusion_matrix=[[50, 5], [10, 35]]
    )
    is_eligible, reason = check_promotion_eligibility(good_metrics, min_auc=0.80, min_f1=0.70)
    assert is_eligible is True

    bad_metrics = EvaluationMetrics(
        accuracy=0.60, precision=0.50, recall=0.40, f1=0.44, roc_auc=0.65, log_loss=0.65, brier_score=0.25, confusion_matrix=[[30, 20], [25, 25]]
    )
    is_eligible, reason = check_promotion_eligibility(bad_metrics, min_auc=0.80, min_f1=0.70)
    assert is_eligible is False
    assert "below minimum threshold" in reason
