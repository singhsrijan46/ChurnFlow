from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import os
import urllib.request
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings
from src.churn_ml.features.preprocessing import (
    create_model_pipeline,
    prepare_training_data,
    save_pipeline,
)
from src.churn_ml.training.evaluate import (
    EvaluationMetrics,
    check_promotion_eligibility,
    evaluate_model,
)

# Enable local file store fallback compatibility if needed
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def build_model_instance(model_type: str, params: Optional[Dict[str, Any]] = None) -> BaseEstimator:
    """
    Instantiates a classifier model based on type and hyperparameter dictionary.
    """
    model_type = model_type.lower()
    if model_type in ["logistic_regression", "lr", "logistic"]:
        default_params = settings.models.logistic_regression.model_dump()
        if params:
            default_params.update(params)
        return LogisticRegression(**default_params)
    elif model_type in ["xgboost", "xgb"]:
        default_params = settings.models.xgboost.model_dump()
        if params:
            default_params.update(params)
        return XGBClassifier(**default_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'logistic_regression' or 'xgboost'.")


def is_mlflow_server_running(uri: str, timeout_sec: float = 1.0) -> bool:
    """Checks if remote MLflow server is actively reachable."""
    if not uri.startswith("http"):
        return True
    try:
        urllib.request.urlopen(uri, timeout=timeout_sec)
        return True
    except Exception:
        return False


def setup_mlflow(tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None) -> str:
    """
    Sets up MLflow tracking URI and experiment name.
    Falls back to local SQLite store if remote server is unreachable.
    """
    uri = tracking_uri or settings.mlflow.tracking_uri
    exp_name = experiment_name or settings.mlflow.experiment_name

    target_uri = uri
    if uri.startswith("http") and not is_mlflow_server_running(uri):
        sqlite_db_path = Path("mlruns.db").resolve()
        target_uri = f"sqlite:///{sqlite_db_path.as_posix()}"
        logger.info(f"Remote MLflow server at {uri} not reachable. Using local SQLite store: {target_uri}")

    mlflow.set_tracking_uri(target_uri)
    try:
        mlflow.set_experiment(exp_name)
    except Exception as e:
        logger.warning(f"Error setting MLflow experiment {exp_name}: {e}")

    return mlflow.get_tracking_uri()


def train_model(
    model_type: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hyperparams: Optional[Dict[str, Any]] = None,
    register_model: bool = False,
    model_registry_name: Optional[str] = None,
) -> Tuple[Any, EvaluationMetrics, Optional[str]]:
    """
    Trains a given model pipeline on train_df, evaluates on test_df, and logs everything to MLflow.
    """
    setup_mlflow()
    model_name = model_registry_name or settings.registry.model_name

    X_train, y_train = prepare_training_data(train_df)
    X_test, y_test = prepare_training_data(test_df)

    logger.info(f"Training {model_type} pipeline with {len(X_train)} training records and {len(X_test)} test records.")

    classifier = build_model_instance(model_type, hyperparams)
    pipeline = create_model_pipeline(classifier)

    run_name = f"{model_type}_run"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        if hasattr(classifier, "get_params"):
            for k, v in classifier.get_params().items():
                mlflow.log_param(f"model_param_{k}", v)

        # Train full pipeline
        pipeline.fit(X_train, y_train)

        # Evaluate model
        artifacts_tmp_dir = Path(settings.mlflow.local_artifact_dir) / f"run_{run_id}"
        artifacts_tmp_dir.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_model(pipeline, X_test, y_test, artifact_dir=str(artifacts_tmp_dir))

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", metrics.accuracy)
        mlflow.log_metric("precision", metrics.precision)
        mlflow.log_metric("recall", metrics.recall)
        mlflow.log_metric("f1_score", metrics.f1)
        mlflow.log_metric("roc_auc", metrics.roc_auc)
        mlflow.log_metric("log_loss", metrics.log_loss)
        mlflow.log_metric("brier_score", metrics.brier_score)

        # Log diagnostic artifacts
        for plot_file in artifacts_tmp_dir.glob("*.png"):
            mlflow.log_artifact(str(plot_file), artifact_path="plots")
        for json_file in artifacts_tmp_dir.glob("*.json"):
            mlflow.log_artifact(str(json_file), artifact_path="metrics")

        # Save pipeline locally as primary deployment artifact
        local_model_path = Path(settings.mlflow.local_artifact_dir) / "model_pipeline.joblib"
        save_pipeline(pipeline, str(local_model_path))

        # Infer signature and log model artifact in MLflow with cloudpickle serialization
        sample_input = X_train.head(5)
        signature = infer_signature(sample_input, pipeline.predict(sample_input))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=model_name if register_model else None,
        )

        logger.info(f"Model {model_type} trained successfully. MLflow Run ID: {run_id}")

        return pipeline, metrics, run_id


def train_and_compare_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    register_best: bool = True,
) -> Dict[str, Any]:
    """
    Trains multiple models (Logistic Regression & XGBoost), compares evaluation metrics,
    and saves/registers the best performing model.
    """
    results: Dict[str, Any] = {}
    candidate_types = ["logistic_regression", "xgboost"]

    for model_type in candidate_types:
        logger.info(f"--- Running Training for: {model_type} ---")
        pipeline, metrics, run_id = train_model(
            model_type=model_type,
            train_df=train_df,
            test_df=test_df,
            register_model=False,
        )
        results[model_type] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "run_id": run_id,
        }

    # Select best model based on ROC-AUC (and tie-break with F1)
    best_type = max(
        results.keys(),
        key=lambda k: (results[k]["metrics"].roc_auc, results[k]["metrics"].f1),
    )
    best_info = results[best_type]
    logger.info(
        f"Champion Model: '{best_type}' with ROC-AUC: {best_info['metrics'].roc_auc:.4f} and F1: {best_info['metrics'].f1:.4f}"
    )

    # Check promotion eligibility against thresholds
    is_eligible, reason = check_promotion_eligibility(best_info["metrics"])
    logger.info(f"Promotion Eligibility: {is_eligible} - {reason}")

    # Persist champion pipeline
    champion_path = Path(settings.mlflow.local_artifact_dir) / "model_pipeline.joblib"
    save_pipeline(best_info["pipeline"], str(champion_path))

    # Save champion metadata
    metadata = {
        "best_model_type": best_type,
        "run_id": best_info["run_id"],
        "metrics": best_info["metrics"].to_dict(),
        "is_promoted": is_eligible,
        "promotion_reason": reason,
    }

    import json
    with open(Path(settings.mlflow.local_artifact_dir) / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "best_model_type": best_type,
        "best_metrics": best_info["metrics"],
        "run_id": best_info["run_id"],
        "pipeline": best_info["pipeline"],
        "all_results": {k: v["metrics"].to_dict() for k, v in results.items()},
        "is_eligible_for_promotion": is_eligible,
        "promotion_reason": reason,
    }
