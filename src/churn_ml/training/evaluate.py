from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.churn_ml.logging_config import logger
from src.churn_ml.config import settings


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    log_loss: float
    brier_score: float
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifact_dir: Optional[str] = None,
) -> EvaluationMetrics:
    """
    Computes comprehensive binary classification metrics and optionally saves diagnostic plots.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    ll = float(log_loss(y_test, y_prob))
    bs = float(brier_score_loss(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = EvaluationMetrics(
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        roc_auc=round(auc, 4),
        log_loss=round(ll, 4),
        brier_score=round(bs, 4),
        confusion_matrix=cm,
    )

    logger.info(
        f"Evaluation Results -> ROC-AUC: {metrics.roc_auc:.4f} | F1: {metrics.f1:.4f} | "
        f"Precision: {metrics.precision:.4f} | Recall: {metrics.recall:.4f} | Accuracy: {metrics.accuracy:.4f}"
    )

    if artifact_dir:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics json
        with open(out_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Save Confusion Matrix Plot
        plt.figure(figsize=(5, 4))
        plt.imshow(np.array(cm), interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["No Churn", "Churn"])
        plt.yticks(tick_marks, ["No Churn", "Churn"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i][j]), horizontalalignment="center", color="black")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png")
        plt.close()

        # Save ROC Curve Plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png")
        plt.close()

    return metrics


def check_promotion_eligibility(
    candidate_metrics: EvaluationMetrics,
    current_production_metrics: Optional[EvaluationMetrics] = None,
    min_auc: Optional[float] = None,
    min_f1: Optional[float] = None,
    promotion_metric: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Evaluates whether a candidate model qualifies for promotion to Production stage.
    """
    threshold_auc = min_auc if min_auc is not None else settings.registry.min_auc
    threshold_f1 = min_f1 if min_f1 is not None else settings.registry.min_f1
    metric_key = promotion_metric or settings.registry.promotion_metric

    # Check minimum thresholds
    if candidate_metrics.roc_auc < threshold_auc:
        reason = f"Candidate ROC-AUC ({candidate_metrics.roc_auc:.4f}) is below minimum threshold ({threshold_auc:.4f})"
        return False, reason

    if candidate_metrics.f1 < threshold_f1:
        reason = f"Candidate F1-score ({candidate_metrics.f1:.4f}) is below minimum threshold ({threshold_f1:.4f})"
        return False, reason

    # Compare with current production model if present
    if current_production_metrics:
        cand_score = getattr(candidate_metrics, metric_key, candidate_metrics.roc_auc)
        prod_score = getattr(current_production_metrics, metric_key, current_production_metrics.roc_auc)
        if cand_score < prod_score:
            reason = (
                f"Candidate {metric_key} ({cand_score:.4f}) is lower than current Production model "
                f"({prod_score:.4f}). Promotion rejected."
            )
            return False, reason

    return True, f"Model passed all promotion criteria (ROC-AUC={candidate_metrics.roc_auc:.4f}, F1={candidate_metrics.f1:.4f})."
