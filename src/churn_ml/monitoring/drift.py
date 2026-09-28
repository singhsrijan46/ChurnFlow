from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from scipy import stats

from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings
from src.churn_ml.monitoring.metrics import DATA_DRIFT_SCORE, DATA_DRIFT_STATUS


@dataclass
class FeatureDriftDetail:
    feature_name: str
    drift_detected: bool
    p_value: float
    statistic: float
    test_name: str


@dataclass
class DriftReportResult:
    drift_detected: bool
    drift_share: float
    threshold: float
    number_of_features: int
    number_of_drifted_features: int
    drifted_features: List[str]
    details: Dict[str, Any]
    report_html_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_feature_drift(
    ref_series: pd.Series,
    curr_series: pd.Series,
    feature_name: str,
    p_value_threshold: float = 0.05,
) -> FeatureDriftDetail:
    """
    Computes statistical test for feature distribution shift:
    - Numerical: 2-sample Kolmogorov-Smirnov (KS) test.
    - Categorical: Chi-square test of independence.
    """
    is_numeric = pd.api.types.is_numeric_dtype(ref_series) and not (
        ref_series.nunique() <= 2 and ref_series.isin([0, 1]).all()
    )

    if is_numeric:
        # Clean nulls for KS test
        ref_clean = ref_series.dropna()
        curr_clean = curr_series.dropna()

        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return FeatureDriftDetail(
                feature_name=feature_name,
                drift_detected=False,
                p_value=1.0,
                statistic=0.0,
                test_name="KS-test (skipped empty)",
            )

        stat, p_val = stats.ks_2samp(ref_clean, curr_clean)
        drift = p_val < p_value_threshold
        return FeatureDriftDetail(
            feature_name=feature_name,
            drift_detected=bool(drift),
            p_value=float(p_val),
            statistic=float(stat),
            test_name="Kolmogorov-Smirnov",
        )
    else:
        # Categorical contingency table test
        ref_counts = ref_series.astype(str).value_counts()
        curr_counts = curr_series.astype(str).value_counts()
        all_categories = sorted(list(set(ref_counts.index).union(set(curr_counts.index))))

        ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
        curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]

        # Add Laplace smoothing to prevent zeros
        ref_freq = [x + 1 for x in ref_freq]
        curr_freq = [x + 1 for x in curr_freq]

        try:
            stat, p_val = stats.chisquare(
                f_obs=curr_freq,
                f_exp=[r * (sum(curr_freq) / sum(ref_freq)) for r in ref_freq],
            )
            drift = p_val < p_value_threshold
            return FeatureDriftDetail(
                feature_name=feature_name,
                drift_detected=bool(drift),
                p_value=float(p_val),
                statistic=float(stat),
                test_name="Chi-Square",
            )
        except Exception as e:
            logger.warning(f"Error computing categorical drift for {feature_name}: {e}")
            return FeatureDriftDetail(
                feature_name=feature_name,
                drift_detected=False,
                p_value=1.0,
                statistic=0.0,
                test_name="Chi-Square (failed)",
            )


def export_interactive_html_report(
    details: Dict[str, Any],
    drift_detected: bool,
    drift_share: float,
    threshold: float,
    output_path: str,
) -> str:
    """Generates a standalone, styled HTML Data Drift Report."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for feat, detail in details.items():
        status_color = "#e53e3e" if detail["drift_detected"] else "#38a169"
        status_text = "DRIFT DETECTED" if detail["drift_detected"] else "HEALTHY"
        rows.append(f"""
        <tr>
            <td style="padding: 12px; font-weight: 600;">{feat}</td>
            <td style="padding: 12px;">{detail['test_name']}</td>
            <td style="padding: 12px;">{detail['statistic']:.4f}</td>
            <td style="padding: 12px;">{detail['p_value']:.4e}</td>
            <td style="padding: 12px; font-weight: bold; color: {status_color};">{status_text}</td>
        </tr>
        """)

    table_body = "\n".join(rows)
    overall_badge = (
        '<span style="background: #e53e3e; color: white; padding: 6px 14px; border-radius: 6px; font-weight: 700;">DRIFT ALERT</span>'
        if drift_detected
        else '<span style="background: #38a169; color: white; padding: 6px 14px; border-radius: 6px; font-weight: 700;">SYSTEM HEALTHY</span>'
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ChurnFlow - Data Drift Telemetry Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 30px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        h1 {{ margin-top: 0; font-size: 24px; color: #38bdf8; }}
        .cards {{ display: flex; gap: 20px; margin: 25px 0; }}
        .card {{ flex: 1; background: #334155; padding: 20px; border-radius: 8px; }}
        .card-label {{ font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card-value {{ font-size: 26px; font-weight: 700; margin-top: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #0f172a; border-radius: 8px; overflow: hidden; }}
        th {{ background: #334155; color: #cbd5e1; text-align: left; padding: 12px; font-size: 14px; }}
        td {{ border-bottom: 1px solid #1e293b; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1>ChurnFlow - Data Drift Telemetry Report</h1>
            <div>{overall_badge}</div>
        </div>
        <div class="cards">
            <div class="card">
                <div class="card-label">Overall Drift Share</div>
                <div class="card-value">{drift_share:.1%}</div>
            </div>
            <div class="card">
                <div class="card-label">Drift Threshold</div>
                <div class="card-value">{threshold:.1%}</div>
            </div>
            <div class="card">
                <div class="card-label">Features Monitored</div>
                <div class="card-value">{len(details)}</div>
            </div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Monitored Feature</th>
                    <th>Statistical Test</th>
                    <th>Test Statistic</th>
                    <th>p-value (Threshold: 0.05)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {table_body}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Data Drift HTML report saved to {out_file}")
    return str(out_file)


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features_to_check: Optional[List[str]] = None,
    drift_threshold: Optional[float] = None,
    p_value_threshold: Optional[float] = None,
    html_output_path: Optional[str] = "reports/data_drift_report.html",
) -> DriftReportResult:
    """
    Executes dataset-wide drift detection between reference baseline and current batch.
    """
    features = features_to_check or settings.monitoring.monitored_features
    threshold = drift_threshold if drift_threshold is not None else settings.monitoring.drift_share_threshold
    p_val_thresh = p_value_threshold if p_value_threshold is not None else settings.monitoring.feature_drift_p_value

    logger.info(f"Checking data drift for {len(features)} monitored features...")

    drifted_features: List[str] = []
    details: Dict[str, Any] = {}

    for feat in features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            logger.warning(f"Feature '{feat}' missing from reference or current dataset. Skipping.")
            continue

        detail = calculate_feature_drift(
            ref_series=reference_df[feat],
            curr_series=current_df[feat],
            feature_name=feat,
            p_value_threshold=p_val_thresh,
        )
        details[feat] = asdict(detail)

        if detail.drift_detected:
            drifted_features.append(feat)

    num_features = len(details)
    num_drifted = len(drifted_features)
    drift_share = float(num_drifted / num_features) if num_features > 0 else 0.0
    drift_detected = drift_share >= threshold

    # Update Prometheus metrics
    try:
        DATA_DRIFT_SCORE.set(round(drift_share, 4))
        DATA_DRIFT_STATUS.set(1 if drift_detected else 0)
    except Exception:
        pass

    html_path = None
    if html_output_path:
        html_path = export_interactive_html_report(
            details=details,
            drift_detected=drift_detected,
            drift_share=drift_share,
            threshold=threshold,
            output_path=html_output_path,
        )

    result = DriftReportResult(
        drift_detected=drift_detected,
        drift_share=round(drift_share, 4),
        threshold=threshold,
        number_of_features=num_features,
        number_of_drifted_features=num_drifted,
        drifted_features=drifted_features,
        details=details,
        report_html_path=html_path,
    )

    logger.info(
        f"Data Drift Check -> Drift Detected: {drift_detected} | "
        f"Drift Share: {drift_share:.2%} (Threshold: {threshold:.2%}) | "
        f"Drifted Features: {drifted_features}"
    )

    return result
