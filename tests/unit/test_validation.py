import pytest
import pandas as pd
from src.churn_ml.data.validation import validate_raw_data, validate_features_df, DataValidationError


@pytest.fixture
def sample_valid_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["001-A", "002-B"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [12, 45],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [70.5, 55.2],
            "TotalCharges": ["846.0", "2484.0"],
            "Churn": ["Yes", "No"],
        }
    )


def test_validate_raw_data_valid(sample_valid_raw_data):
    result = validate_raw_data(sample_valid_raw_data, raise_on_error=True)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validate_raw_data_missing_column(sample_valid_raw_data):
    df_missing = sample_valid_raw_data.drop(columns=["tenure"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_raw_data(df_missing, raise_on_error=True)


def test_validate_raw_data_empty():
    empty_df = pd.DataFrame()
    with pytest.raises(DataValidationError, match="Dataset is empty"):
        validate_raw_data(empty_df, raise_on_error=True)


def test_validate_raw_data_invalid_tenure(sample_valid_raw_data):
    sample_valid_raw_data.loc[0, "tenure"] = -5
    with pytest.raises(DataValidationError, match="tenure values outside"):
        validate_raw_data(sample_valid_raw_data, raise_on_error=True)


def test_validate_raw_data_invalid_target(sample_valid_raw_data):
    sample_valid_raw_data.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataValidationError, match="Invalid target values"):
        validate_raw_data(sample_valid_raw_data, raise_on_error=True)


def test_validate_features_df():
    df = pd.DataFrame({"tenure": [1], "MonthlyCharges": [20.0], "TotalCharges": [20.0]})
    res = validate_features_df(df, expected_features=["tenure", "MonthlyCharges", "TotalCharges"])
    assert res.is_valid is True
