import pandas as pd
import numpy as np
from src.churn_ml.features.preprocessing import (
    build_preprocessor,
    create_model_pipeline,
    prepare_training_data,
)
from sklearn.linear_model import LogisticRegression


def test_build_preprocessor_transformation():
    df = pd.DataFrame(
        {
            "tenure": [12, np.nan, 24],
            "MonthlyCharges": [70.5, 80.0, np.nan],
            "TotalCharges": [846.0, 160.0, 2000.0],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes"],
            "PhoneService": ["Yes", "No", "Yes"],
            "MultipleLines": ["No", "No phone service", "Yes"],
            "InternetService": ["Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "Yes", "No internet service"],
            "OnlineBackup": ["Yes", "No", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "Yes", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        }
    )

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert transformed is not None
    assert transformed.shape[0] == 3
    # Check no NaN values remaining after imputation
    assert not np.isnan(transformed).any()


def test_create_model_pipeline_fit_predict():
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

    preds = pipeline.predict(df)
    probs = pipeline.predict_proba(df)

    assert len(preds) == len(df)
    assert probs.shape == (len(df), 2)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_prepare_training_data():
    df = pd.DataFrame(
        {
            "customerID": ["C1", "C2"],
            "tenure": [10, 20],
            "Churn": [0, 1],
        }
    )
    X, y = prepare_training_data(df, target_column="Churn", id_column="customerID")
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert list(X.columns) == ["tenure"]
    assert list(y) == [0, 1]
