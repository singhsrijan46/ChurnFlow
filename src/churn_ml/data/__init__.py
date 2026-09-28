"""
Data ingestion and validation module.
"""

from src.churn_ml.data.ingestion import ingest_data, save_processed_splits
from src.churn_ml.data.validation import validate_raw_data, validate_features_df, ValidationResult

__all__ = [
    "ingest_data",
    "save_processed_splits",
    "validate_raw_data",
    "validate_features_df",
    "ValidationResult",
]
