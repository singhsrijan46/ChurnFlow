from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """
    Input schema for customer churn prediction attributes.
    """
    gender: Literal["Female", "Male"] = Field(
        ..., description="Customer gender (Female or Male)"
    )
    senior_citizen: Literal[0, 1] = Field(
        ..., alias="SeniorCitizen", description="Whether customer is senior citizen (1) or not (0)"
    )
    partner: Literal["Yes", "No"] = Field(
        ..., alias="Partner", description="Whether customer has a partner"
    )
    dependents: Literal["Yes", "No"] = Field(
        ..., alias="Dependents", description="Whether customer has dependents"
    )
    tenure: int = Field(
        ..., ge=0, le=120, description="Number of months customer has stayed with company"
    )
    phone_service: Literal["Yes", "No"] = Field(
        "Yes", alias="PhoneService", description="Whether customer has phone service"
    )
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        "No", alias="MultipleLines", description="Whether customer has multiple lines"
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., alias="InternetService", description="Customer's internet service provider"
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        "No", alias="OnlineSecurity", description="Whether customer has online security"
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        "No", alias="OnlineBackup", description="Whether customer has online backup"
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        "No", alias="DeviceProtection", description="Whether customer has device protection"
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        "No", alias="TechSupport", description="Whether customer has tech support"
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        "No", alias="StreamingTV", description="Whether customer has streaming TV"
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        "No", alias="StreamingMovies", description="Whether customer has streaming movies"
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., alias="Contract", description="Contract term duration"
    )
    paperless_billing: Literal["Yes", "No"] = Field(
        "Yes", alias="PaperlessBilling", description="Whether customer has paperless billing"
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ..., alias="PaymentMethod", description="Customer payment method"
    )
    monthly_charges: float = Field(
        ..., alias="MonthlyCharges", ge=0.0, le=500.0, description="Monthly amount charged to customer"
    )
    total_charges: float = Field(
        ..., alias="TotalCharges", ge=0.0, le=50000.0, description="Total amount charged to customer"
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        },
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No Churn, 1 = Churn")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated churn probability")
    risk_category: str = Field(..., description="Risk category: Low, Medium, High")
    model_version: str = Field(..., description="Active model version identifier")
    request_id: Optional[str] = Field(None, description="Unique trace request ID")


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    model_version: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    app_name: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str = "ready"
    model_loaded: bool
    model_name: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    is_ready: bool
    model_path: str
    features_required: List[str]
    metadata: Dict[str, Any]


class DriftCheckResponse(BaseModel):
    drift_detected: bool
    drift_share: float
    threshold: float
    number_of_features: int
    number_of_drifted_features: int
    drifted_features: List[str]
    report_html_path: Optional[str] = None
    retraining_recommended: bool


class RetrainingResponse(BaseModel):
    status: str
    champion_model: str
    best_metrics: Dict[str, Any]
    run_id: str
    is_promoted: bool
    promotion_reason: str
