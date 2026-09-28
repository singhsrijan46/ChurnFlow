import pytest
from pydantic import ValidationError
from api.schemas import CustomerFeatures


def test_customer_features_valid():
    payload = {
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
    customer = CustomerFeatures(**payload)
    assert customer.gender == "Female"
    assert customer.tenure == 12
    assert customer.monthly_charges == 70.5


def test_customer_features_invalid_tenure():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 250,  # Invalid: > 120
        "InternetService": "DSL",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 50.0,
    }
    with pytest.raises(ValidationError):
        CustomerFeatures(**payload)


def test_customer_features_invalid_category():
    payload = {
        "gender": "Alien",  # Invalid categorical
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 10,
        "InternetService": "Satellite",  # Invalid
        "Contract": "Ten Year",  # Invalid
        "PaymentMethod": "Crypto",  # Invalid
        "MonthlyCharges": 50.0,
        "TotalCharges": 50.0,
    }
    with pytest.raises(ValidationError):
        CustomerFeatures(**payload)
