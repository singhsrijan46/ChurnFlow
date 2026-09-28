from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import pandas as pd
from sklearn.pipeline import Pipeline
import mlflow.sklearn

from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings
from src.churn_ml.features.preprocessing import load_pipeline


class ModelNotReadyError(Exception):
    """Raised when model is requested for prediction before being loaded."""
    pass


class ChurnPredictor:
    """
    High-performance inference engine for Customer Churn predictions.
    Supports model loading from local disk or MLflow Registry with fallback handling.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        mlflow_model_uri: Optional[str] = None,
        model_version: str = "1.0.0",
    ):
        self.model_path = model_path or str(Path(settings.mlflow.local_artifact_dir) / "model_pipeline.joblib")
        self.mlflow_model_uri = mlflow_model_uri
        self.model_version = model_version
        self.pipeline: Optional[Pipeline] = None
        self.metadata: Dict[str, Any] = {}
        self._load_model()

    def _load_model(self) -> None:
        """Attempts to load model from MLflow URI or local fallback path."""
        # 1. Try MLflow URI if explicitly configured
        if self.mlflow_model_uri:
            try:
                logger.info(f"Attempting to load model from MLflow URI: {self.mlflow_model_uri}")
                self.pipeline = mlflow.sklearn.load_model(self.mlflow_model_uri)
                logger.info("Successfully loaded model from MLflow.")
                return
            except Exception as e:
                logger.warning(f"Could not load model from MLflow ({e}). Falling back to local artifact.")

        # 2. Try local disk path
        if self.model_path and Path(self.model_path).exists():
            try:
                self.pipeline = load_pipeline(self.model_path)
                # Load metadata if available
                meta_path = Path(self.model_path).parent / "model_metadata.json"
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                logger.info(f"Loaded model successfully from local disk path: {self.model_path}")
                return
            except Exception as e:
                logger.error(f"Failed to load local model artifact from {self.model_path}: {e}")

        logger.warning("No pretrained model pipeline found on disk. Predictor is in uninitialized state.")

    def is_ready(self) -> bool:
        """Returns True if model pipeline is loaded and ready to serve inference requests."""
        return self.pipeline is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the active model."""
        return {
            "model_name": settings.registry.model_name,
            "model_version": self.model_version,
            "is_ready": self.is_ready(),
            "model_path": self.model_path,
            "features_required": (
                settings.features.numerical_features + settings.features.categorical_features
            ),
            "metadata": self.metadata,
        }

    def _prepare_dataframe(self, data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
        """Converts input data into a standardized pandas DataFrame."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported input data type: {type(data)}")

        # Ensure column names match expected casing / features
        rename_map = {}
        expected_all = settings.features.numerical_features + settings.features.categorical_features
        lower_to_orig = {col.lower(): col for col in expected_all}

        for col in df.columns:
            if col.lower() in lower_to_orig and col not in expected_all:
                rename_map[col] = lower_to_orig[col.lower()]

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes inference for a single customer payload.
        """
        if not self.is_ready():
            raise ModelNotReadyError("Model is not initialized or trained. Please train the model first.")

        df = self._prepare_dataframe(features)
        pred = int(self.pipeline.predict(df)[0])
        prob = float(self.pipeline.predict_proba(df)[0, 1])

        # Categorize churn risk
        if prob < 0.35:
            risk = "Low"
        elif prob < 0.70:
            risk = "Medium"
        else:
            risk = "High"

        return {
            "prediction": pred,
            "churn_probability": round(prob, 4),
            "risk_category": risk,
            "model_version": self.model_version,
        }

    def predict_batch(self, batch_data: Union[List[Dict[str, Any]], pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Executes inference for a batch of customer records.
        """
        if not self.is_ready():
            raise ModelNotReadyError("Model is not initialized or trained. Please train the model first.")

        df = self._prepare_dataframe(batch_data)
        preds = self.pipeline.predict(df)
        probs = self.pipeline.predict_proba(df)[:, 1]

        results = []
        for pred, prob in zip(preds, probs):
            p_val = float(prob)
            if p_val < 0.35:
                risk = "Low"
            elif p_val < 0.70:
                risk = "Medium"
            else:
                risk = "High"

            results.append(
                {
                    "prediction": int(pred),
                    "churn_probability": round(p_val, 4),
                    "risk_category": risk,
                    "model_version": self.model_version,
                }
            )
        return results


# Global singleton instance holder
_global_predictor: Optional[ChurnPredictor] = None


def get_predictor(force_reload: bool = False) -> ChurnPredictor:
    """Returns or creates the singleton ChurnPredictor instance."""
    global _global_predictor
    if _global_predictor is None or force_reload:
        _global_predictor = ChurnPredictor()
    return _global_predictor
