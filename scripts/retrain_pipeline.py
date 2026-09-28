import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings
from src.churn_ml.data.ingestion import ingest_data, save_processed_splits, clean_raw_data
from src.churn_ml.data.validation import validate_raw_data
from src.churn_ml.monitoring.drift import detect_data_drift
from src.churn_ml.training.train import train_and_compare_models


def run_retraining_pipeline(force: bool = False) -> bool:
    """
    Executes the continuous automated retraining and governance loop.
    Returns True if a new model was successfully validated and promoted.
    """
    logger.info("=== Starting Automated Retraining Workflow ===")

    # Step 1: Ingest & Validate
    logger.info("Step 1: Data Ingestion and Validation...")
    df_raw = ingest_data(clean=False, validate=False)
    validate_raw_data(df_raw, raise_on_error=True)
    logger.info(f"Data validation passed. Dataset shape: {df_raw.shape}")

    # Clean data
    df_clean = clean_raw_data(df_raw)
    train_df, test_df = save_processed_splits(df_clean)

    # Step 2: Drift Detection
    logger.info("Step 2: Checking Data Drift against Baseline Reference...")
    drift_result = detect_data_drift(
        reference_df=train_df,
        current_df=test_df,
        drift_threshold=settings.monitoring.drift_share_threshold,
    )

    should_retrain = force or drift_result.drift_detected
    if not should_retrain:
        logger.info("No significant data drift detected. Model retraining skipped.")
        return False

    logger.info("Retraining triggered (Drift detected or force flag enabled).")

    # Step 3: Model Retraining & Multi-Model Comparison
    logger.info("Step 3: Training Candidate Models & MLflow Experiment Tracking...")
    comparison_results = train_and_compare_models(train_df, test_df, register_best=True)

    champion_type = comparison_results["best_model_type"]
    metrics = comparison_results["best_metrics"]
    is_promoted = comparison_results["is_eligible_for_promotion"]
    reason = comparison_results["promotion_reason"]

    logger.info("Retraining Model Results:")
    logger.info(f"- Champion Architecture: {champion_type}")
    logger.info(f"- ROC-AUC: {metrics.roc_auc:.4f}")
    logger.info(f"- F1-Score: {metrics.f1:.4f}")
    logger.info(f"- Precision: {metrics.precision:.4f}")
    logger.info(f"- Recall: {metrics.recall:.4f}")
    logger.info(f"- Promotion Status: {'PROMOTED' if is_promoted else 'REJECTED'}")
    logger.info(f"- Governance Decision Reason: {reason}")

    if not is_promoted:
        logger.warning("Champion model failed promotion criteria. Production model retained.")
        return False

    logger.info("=== Automated Retraining Loop Succeeded & New Champion Model Deployed ===")
    return True


def main():
    parser = argparse.ArgumentParser(description="Trigger automated customer churn retraining loop.")
    parser.add_argument("--force", action="store_true", help="Force retraining even if drift is below threshold")
    args = parser.parse_args()

    success = run_retraining_pipeline(force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
