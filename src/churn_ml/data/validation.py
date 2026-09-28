from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, int | float] = field(default_factory=dict)


class DataValidationError(Exception):
    """Raised when critical data validation checks fail."""
    pass


ALLOWED_CATEGORICAL_VALUES = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


def validate_raw_data(df: pd.DataFrame, raise_on_error: bool = True) -> ValidationResult:
    """
    Performs comprehensive schema and value validation on raw customer churn dataset.
    """
    errors: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, int | float] = {}

    logger.info(f"Starting raw data validation on dataset with shape {df.shape}")

    # 1. Check if DataFrame is empty
    if df.empty:
        errors.append("Dataset is empty.")
        result = ValidationResult(is_valid=False, errors=errors, warnings=warnings, metrics=metrics)
        if raise_on_error:
            raise DataValidationError("Data validation failed: Dataset is empty.")
        return result

    # 2. Check for critical / required columns
    expected_cols = settings.validation.critical_columns or (
        settings.features.numerical_features
        + settings.features.categorical_features
        + [settings.data.target_column, settings.data.id_column]
    )

    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(list(missing_cols))}")

    # 3. Check for duplicates in ID column if present
    if settings.data.id_column in df.columns:
        dup_count = df[settings.data.id_column].duplicated().sum()
        metrics["duplicate_ids"] = int(dup_count)
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate customer IDs.")

    # 4. Check target column values
    if settings.data.target_column in df.columns:
        unique_targets = set(df[settings.data.target_column].dropna().unique())
        valid_target_sets = [{"Yes", "No"}, {0, 1}, {"0", "1"}, {"True", "False"}, {True, False}]
        if not any(unique_targets.issubset(s) for s in valid_target_sets):
            errors.append(f"Invalid target values found: {unique_targets}. Expected Yes/No or 0/1.")

        target_nulls = df[settings.data.target_column].isnull().sum()
        if target_nulls > 0:
            errors.append(f"Target column '{settings.data.target_column}' contains {target_nulls} null values.")

    # 5. Check categorical values against allowed sets
    for col, allowed_vals in ALLOWED_CATEGORICAL_VALUES.items():
        if col in df.columns:
            col_vals = df[col].dropna().unique()
            col_set = set(col_vals)
            invalid = col_set - allowed_vals
            if col == "SeniorCitizen" and invalid:
                invalid = {v for v in invalid if v not in {"0", "1", 0, 1}}
            if invalid:
                warnings.append(f"Column '{col}' contains unexpected categories: {invalid}")

    # 6. Check numerical bounds
    if "tenure" in df.columns:
        tenure_num = pd.to_numeric(df["tenure"], errors="coerce")
        out_bounds = ((tenure_num < settings.validation.tenure_min) | (tenure_num > settings.validation.tenure_max)).sum()
        if out_bounds > 0:
            errors.append(f"Found {out_bounds} tenure values outside [{settings.validation.tenure_min}, {settings.validation.tenure_max}].")

    if "MonthlyCharges" in df.columns:
        mc_num = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        out_bounds = ((mc_num < settings.validation.monthly_charges_min) | (mc_num > settings.validation.monthly_charges_max)).sum()
        if out_bounds > 0:
            errors.append(f"Found {out_bounds} MonthlyCharges values outside [{settings.validation.monthly_charges_min}, {settings.validation.monthly_charges_max}].")

    if "TotalCharges" in df.columns:
        tc_num = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        out_bounds = ((tc_num < settings.validation.total_charges_min) | (tc_num > settings.validation.total_charges_max)).dropna().sum()
        if out_bounds > 0:
            errors.append(f"Found {out_bounds} TotalCharges values outside [{settings.validation.total_charges_min}, {settings.validation.total_charges_max}].")

    is_valid = len(errors) == 0
    metrics["num_rows"] = len(df)
    metrics["num_cols"] = len(df.columns)
    metrics["num_errors"] = len(errors)
    metrics["num_warnings"] = len(warnings)

    result = ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )

    if not is_valid:
        logger.error(f"Data validation failed with {len(errors)} errors: {errors}")
        if raise_on_error:
            raise DataValidationError(f"Data validation failed: {'; '.join(errors)}")
    else:
        logger.info("Data validation passed successfully.")

    return result


def validate_features_df(df: pd.DataFrame, expected_features: Optional[List[str]] = None) -> ValidationResult:
    """
    Validates features dataframe ready for inference or training.
    """
    if expected_features is None:
        expected_features = settings.features.numerical_features + settings.features.categorical_features

    errors: List[str] = []
    warnings: List[str] = []

    missing = set(expected_features) - set(df.columns)
    if missing:
        errors.append(f"Missing expected features: {sorted(list(missing))}")

    return ValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        metrics={"feature_count": len(df.columns), "row_count": len(df)},
    )
