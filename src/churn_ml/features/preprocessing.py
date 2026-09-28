from pathlib import Path
from typing import Any, List, Optional, Tuple
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Selects specified columns from a pandas DataFrame and ensures proper string / float types.
    """
    def __init__(self, attribute_names: List[str]):
        self.attribute_names = attribute_names

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Ensure all required attributes exist (fill missing with None if inference payload lacks an optional field)
        for col in self.attribute_names:
            if col not in X.columns:
                X[col] = None
        return X[self.attribute_names]


def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Builds a Scikit-Learn ColumnTransformer that handles imputation, scaling, and one-hot encoding.
    """
    num_cols = numerical_features or settings.features.numerical_features
    cat_cols = categorical_features or settings.features.categorical_features

    # Numerical Transformer: Median Imputer + Standard Scaler
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical Transformer: Most Frequent Imputer + OneHotEncoder
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def create_model_pipeline(
    model: BaseEstimator,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Pipeline:
    """
    Combines the ColumnTransformer preprocessor with an Estimator into an end-to-end Pipeline.
    Prevents training-serving skew.
    """
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return full_pipeline


def prepare_training_data(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    id_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits DataFrame into features X and target y, dropping non-feature ID column.
    """
    target = target_column or settings.data.target_column
    id_col = id_column or settings.data.id_column

    X = df.drop(columns=[target], errors="ignore")
    if id_col in X.columns:
        X = X.drop(columns=[id_col])

    y = df[target] if target in df.columns else None
    return X, y


def save_pipeline(pipeline: Pipeline, filepath: str) -> None:
    """Saves serialized scikit-learn pipeline to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Model pipeline saved successfully to {path}")


def load_pipeline(filepath: str) -> Pipeline:
    """Loads serialized scikit-learn pipeline from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline artifact not found at {path}")
    pipeline = joblib.load(path)
    logger.info(f"Model pipeline loaded successfully from {path}")
    return pipeline
