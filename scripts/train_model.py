import argparse
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings
from src.churn_ml.training.train import train_and_compare_models, train_model


def main():
    parser = argparse.ArgumentParser(description="Train customer churn prediction models with MLflow tracking.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="all",
        choices=["all", "logistic_regression", "xgboost"],
        help="Model type to train ('all' trains and compares both)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=settings.data.processed_train_path,
        help="Path to processed training CSV",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=settings.data.processed_test_path,
        help="Path to processed test CSV",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register best model in MLflow Registry",
    )
    args = parser.parse_args()

    train_path = Path(args.train_data)
    test_path = Path(args.test_data)

    if not train_path.exists() or not test_path.exists():
        logger.error("Train or test dataset missing. Running data ingestion first...")
        from src.churn_ml.data.ingestion import ingest_data, save_processed_splits
        df_clean = ingest_data(clean=True, validate=True)
        save_processed_splits(df_clean)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if args.model_type == "all":
        results = train_and_compare_models(train_df, test_df, register_best=args.register)
        logger.info(f"Champion Model: {results['best_model_type']}")
        logger.info(f"Promotion Eligibility: {results['is_eligible_for_promotion']} ({results['promotion_reason']})")
    else:
        pipeline, metrics, run_id = train_model(
            model_type=args.model_type,
            train_df=train_df,
            test_df=test_df,
            register_model=args.register,
        )
        logger.info(f"Model {args.model_type} trained. Run ID: {run_id}, ROC-AUC: {metrics.roc_auc}")


if __name__ == "__main__":
    main()
