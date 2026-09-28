"""
Training and evaluation modules.
"""

from src.churn_ml.training.evaluate import (
    evaluate_model,
    EvaluationMetrics,
    check_promotion_eligibility,
)
from src.churn_ml.training.train import (
    train_model,
    train_and_compare_models,
    build_model_instance,
)

__all__ = [
    "evaluate_model",
    "EvaluationMetrics",
    "check_promotion_eligibility",
    "train_model",
    "train_and_compare_models",
    "build_model_instance",
]
