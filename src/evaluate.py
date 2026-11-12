"""
Model evaluation utilities. All metrics here are computed from real
predictions on a held-out split -- nothing is hardcoded.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_pred, y_proba=None, classes=None) -> Dict:
    """
    Compute a full metrics dictionary for a set of predictions.
    y_proba: optional (n_samples, n_classes) probability/score matrix.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classes"] = list(classes) if classes is not None else sorted(set(y_true))

    # Per-class breakdown
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    metrics["per_class"] = {
        str(cls): {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }
        for cls, p, r, f in zip(metrics["classes"], per_class_precision, per_class_recall, per_class_f1)
    }

    # ROC-AUC only meaningful/reliable for binary classification with probabilities
    if y_proba is not None and metrics["classes"] and len(metrics["classes"]) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics
