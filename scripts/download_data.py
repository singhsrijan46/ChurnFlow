import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.churn_ml.logging_config import logger
from src.churn_ml.data.ingestion import ingest_data, save_processed_splits


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Telco customer churn dataset.")
    parser.add_argument("--url", type=str, default=None, help="Custom dataset download URL")
    parser.add_argument("--raw-path", type=str, default=None, help="Path to save raw CSV file")
    args = parser.parse_args()

    logger.info("Starting Data Ingestion step...")
    df_clean = ingest_data(raw_url=args.url, raw_path=args.raw_path, clean=True, validate=True)
    train_df, test_df = save_processed_splits(df_clean)
    logger.info(f"Data ingestion completed. Train set: {len(train_df)}, Test set: {len(test_df)}")


if __name__ == "__main__":
    main()
