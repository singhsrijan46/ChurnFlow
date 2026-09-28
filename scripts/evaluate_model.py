import argparse
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings
from src.churn_ml.features.preprocessing import load_pipeline, prepare_training_data
from src.churn_ml.training.evaluate import evaluate_model, check_promotion_eligibility


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model pipeline against test data.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path(settings.mlflow.local_artifact_dir) / "model_pipeline.joblib"),
        help="Path to serialized pipeline artifact",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=settings.data.processed_test_path,
        help="Path to test CSV data",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="reports/evaluation",
        help="Directory to save evaluation plots and metrics JSON",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    test_path = Path(args.test_data)

    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Train model first.")
        sys.exit(1)

    if not test_path.exists():
        logger.error(f"Test data file not found at {test_path}.")
        sys.exit(1)

    pipeline = load_pipeline(str(model_path))
    test_df = pd.read_csv(test_path)
    X_test, y_test = prepare_training_data(test_df)

    metrics = evaluate_model(pipeline, X_test, y_test, artifact_dir=args.artifact_dir)
    is_promotable, reason = check_promotion_eligibility(metrics)

    logger.info("Evaluation Complete:")
    logger.info(f"- ROC-AUC: {metrics.roc_auc:.4f}")
    logger.info(f"- F1-Score: {metrics.f1:.4f}")
    logger.info(f"- Precision: {metrics.precision:.4f}")
    logger.info(f"- Recall: {metrics.recall:.4f}")
    logger.info(f"- Promotable to Production: {is_promotable} ({reason})")

    if not is_promotable:
        logger.warning(f"Model failed promotion criteria: {reason}")
        sys.exit(2)


if __name__ == "__main__":
    main()
