import pandas as pd
from src.churn_ml.data.validation import validate_raw_data
from src.churn_ml.data.ingestion import clean_raw_data, save_processed_splits
from src.churn_ml.training.train import train_and_compare_models
from src.churn_ml.inference.predictor import ChurnPredictor


def test_end_to_end_mlops_lifecycle(tmp_path):
    # 1. Create representative raw data
    raw_df = pd.DataFrame(
        {
            "customerID": [f"C{i:03d}" for i in range(20)],
            "gender": ["Female", "Male"] * 10,
            "SeniorCitizen": [0, 1] * 10,
            "Partner": ["Yes", "No"] * 10,
            "Dependents": ["No", "Yes"] * 10,
            "tenure": [1, 24, 60, 2, 36, 70, 5, 48, 12, 50, 15, 30, 45, 60, 2, 8, 18, 28, 38, 58],
            "PhoneService": ["Yes"] * 20,
            "MultipleLines": ["No", "Yes"] * 10,
            "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic"] * 5,
            "OnlineSecurity": ["No", "Yes", "No internet service", "No"] * 5,
            "OnlineBackup": ["No", "Yes", "No internet service", "No"] * 5,
            "DeviceProtection": ["No", "Yes", "No internet service", "No"] * 5,
            "TechSupport": ["No", "Yes", "No internet service", "No"] * 5,
            "StreamingTV": ["No", "Yes", "No internet service", "Yes"] * 5,
            "StreamingMovies": ["No", "Yes", "No internet service", "Yes"] * 5,
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"] * 5,
            "PaperlessBilling": ["Yes", "No"] * 10,
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] * 5,
            "MonthlyCharges": [70.0, 50.0, 20.0, 85.0] * 5,
            "TotalCharges": [str(x) for x in [70.0, 1200.0, 1200.0, 170.0] * 5],
            "Churn": ["Yes", "No", "No", "Yes"] * 5,
        }
    )

    # 2. Validation
    val_result = validate_raw_data(raw_df, raise_on_error=True)
    assert val_result.is_valid is True

    # 3. Clean and Split
    cleaned_df = clean_raw_data(raw_df)
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df, test_df = save_processed_splits(
        cleaned_df,
        train_path=str(train_path),
        test_path=str(test_path),
        test_size=0.25,
    )
    assert len(train_df) > 0
    assert len(test_df) > 0

    # 4. Train & Compare
    results = train_and_compare_models(train_df, test_df, register_best=False)
    assert "best_model_type" in results
    assert results["best_metrics"].roc_auc >= 0.0

    # 5. Inference
    predictor = ChurnPredictor()
    predictor.pipeline = results["pipeline"]
    sample_record = train_df.iloc[0].drop(["customerID", "Churn"]).to_dict()
    prediction = predictor.predict_single(sample_record)

    assert prediction["prediction"] in (0, 1)
    assert 0.0 <= prediction["churn_probability"] <= 1.0
