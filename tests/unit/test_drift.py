import numpy as np
import pandas as pd
from src.churn_ml.monitoring.drift import detect_data_drift, calculate_feature_drift


def test_calculate_feature_drift_no_drift():
    np.random.seed(42)
    ref = pd.Series(np.random.normal(50, 10, 500))
    curr = pd.Series(np.random.normal(50, 10, 500))

    detail = calculate_feature_drift(ref, curr, feature_name="tenure")
    assert detail.drift_detected is False
    assert detail.p_value > 0.05


def test_calculate_feature_drift_with_drift():
    np.random.seed(42)
    ref = pd.Series(np.random.normal(20, 5, 500))
    curr = pd.Series(np.random.normal(80, 5, 500))  # Significant distribution shift

    detail = calculate_feature_drift(ref, curr, feature_name="tenure")
    assert detail.drift_detected is True
    assert detail.p_value < 0.05


def test_detect_data_drift_overall():
    ref_df = pd.DataFrame(
        {
            "tenure": np.random.normal(30, 5, 200),
            "MonthlyCharges": np.random.normal(70, 10, 200),
            "Contract": ["Month-to-month"] * 100 + ["Two year"] * 100,
        }
    )
    # Severe drift in all features
    curr_df = pd.DataFrame(
        {
            "tenure": np.random.normal(90, 5, 200),
            "MonthlyCharges": np.random.normal(120, 10, 200),
            "Contract": ["One year"] * 200,
        }
    )

    report = detect_data_drift(
        reference_df=ref_df,
        current_df=curr_df,
        features_to_check=["tenure", "MonthlyCharges", "Contract"],
        drift_threshold=0.20,
        html_output_path=None,
    )

    assert report.drift_detected is True
    assert report.drift_share > 0.20
    assert report.number_of_drifted_features >= 2
