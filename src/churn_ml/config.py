from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "ChurnFlow"
    version: str = "1.0.0"
    environment: str = "development"


class DataConfig(BaseModel):
    raw_data_url: str = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    raw_data_path: str = "data/raw/telco_churn_raw.csv"
    processed_train_path: str = "data/processed/train.csv"
    processed_test_path: str = "data/processed/test.csv"
    reference_data_path: str = "data/processed/reference.csv"
    current_data_path: str = "data/processed/current.csv"
    target_column: str = "Churn"
    id_column: str = "customerID"
    test_size: float = 0.20
    random_state: int = 42


class FeaturesConfig(BaseModel):
    numerical_features: List[str] = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]
    categorical_features: List[str] = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]


class ValidationConfig(BaseModel):
    critical_columns: List[str] = []
    tenure_min: int = 0
    tenure_max: int = 120
    monthly_charges_min: float = 0.0
    monthly_charges_max: float = 500.0
    total_charges_min: float = 0.0
    total_charges_max: float = 50000.0


class LogisticRegressionConfig(BaseModel):
    C: float = 1.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    random_state: int = 42


class XGBoostConfig(BaseModel):
    n_estimators: int = 150
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    eval_metric: str = "logloss"


class ModelsConfig(BaseModel):
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    xgboost: XGBoostConfig = XGBoostConfig()


class RegistryConfig(BaseModel):
    model_name: str = "churn_prediction_model"
    min_auc: float = 0.80
    min_f1: float = 0.55
    promotion_metric: str = "roc_auc"


class MonitoringConfig(BaseModel):
    drift_share_threshold: float = 0.20
    feature_drift_p_value: float = 0.05
    monitored_features: List[str] = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaymentMethod",
    ]


class MLflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "customer-churn-prediction"
    local_artifact_dir: str = "models"


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Loads YAML configuration and applies environment variable overrides.
    """
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        default_yaml = base_dir / "configs" / "config.yaml"
        if default_yaml.exists():
            config_path = str(default_yaml)

    yaml_data: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if loaded:
                yaml_data = loaded

    settings = Settings(**yaml_data)

    # Environment variable overrides
    if os.getenv("MLFLOW_TRACKING_URI"):
        settings.mlflow.tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    if os.getenv("MLFLOW_EXPERIMENT_NAME"):
        settings.mlflow.experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
    if os.getenv("MODEL_NAME"):
        settings.registry.model_name = os.environ["MODEL_NAME"]
    if os.getenv("DRIFT_THRESHOLD"):
        settings.monitoring.drift_share_threshold = float(os.environ["DRIFT_THRESHOLD"])
    if os.getenv("MIN_AUC"):
        settings.registry.min_auc = float(os.environ["MIN_AUC"])
    if os.getenv("MIN_F1"):
        settings.registry.min_f1 = float(os.environ["MIN_F1"])

    return settings


# Global settings instance
settings = load_config()
