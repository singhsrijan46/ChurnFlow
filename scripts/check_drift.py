import argparse
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings
from src.churn_ml.monitoring.drift import detect_data_drift


def main():
    parser = argparse.ArgumentParser(
        description="Run statistical data drift detection between reference and current data."
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=settings.data.reference_data_path,
        help="Path to baseline reference CSV",
    )
    parser.add_argument(
        "--current",
        type=str,
        default=settings.data.current_data_path,
        help="Path to production/current CSV",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.monitoring.drift_share_threshold,
        help="Dataset drift share threshold",
    )
    parser.add_argument(
        "--report-html",
        type=str,
        default="reports/data_drift_report.html",
        help="Path to export HTML drift report",
    )
    args = parser.parse_args()

    ref_path = Path(args.reference)
    curr_path = Path(args.current)

    if not ref_path.exists() or not curr_path.exists():
        logger.error(f"Reference ({ref_path}) or Current ({curr_path}) dataset missing.")
        sys.exit(1)

    ref_df = pd.read_csv(ref_path)
    curr_df = pd.read_csv(curr_path)

    report = detect_data_drift(
        reference_df=ref_df,
        current_df=curr_df,
        drift_threshold=args.threshold,
        html_output_path=args.report_html,
    )

    logger.info("Drift Analysis Summary:")
    logger.info(f"- Drift Detected: {report.drift_detected}")
    logger.info(f"- Drift Share: {report.drift_share:.2%} (Threshold: {report.threshold:.2%})")
    logger.info(
        f"- Drifted Features ({report.number_of_drifted_features}/{report.number_of_features}): "
        f"{report.drifted_features}"
    )

    if report.drift_detected:
        logger.warning("Dataset drift threshold exceeded! Retraining recommended.")
        sys.exit(3)


if __name__ == "__main__":
    main()
