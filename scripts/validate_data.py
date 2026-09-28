import argparse
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings
from src.churn_ml.data.validation import validate_raw_data, DataValidationError


def main():
    parser = argparse.ArgumentParser(description="Validate customer churn dataset schema and values.")
    parser.add_argument(
        "--file-path",
        type=str,
        default=settings.data.raw_data_path,
        help="Path to CSV dataset to validate",
    )
    args = parser.parse_args()

    file_path = Path(args.file_path)
    if not file_path.exists():
        logger.error(f"Dataset file not found at {file_path}")
        sys.exit(1)

    logger.info(f"Validating dataset at: {file_path}")
    df = pd.read_csv(file_path)

    try:
        result = validate_raw_data(df, raise_on_error=True)
        logger.info(f"Validation SUCCESS! Metrics: {result.metrics}")
    except DataValidationError as e:
        logger.error(f"Validation FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
