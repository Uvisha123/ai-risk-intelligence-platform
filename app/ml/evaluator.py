"""
Classification metrics for trained models.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate_binary(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels.
        y_proba: Optional predicted probabilities for the positive class.

    Returns:
        Dictionary of metric names to float values.
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def confusion_stats(y_true, y_pred) -> Dict[str, int]:
    """Return TP, TN, FP, FN counts for binary labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn}
