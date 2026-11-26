"""Evaluation metrics, confusion matrix, ROC-AUC, and reporting."""

from .evaluate import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_multiclass_roc_auc,
    evaluate_model,
    predict_probabilities,
)

__all__ = [
    "predict_probabilities",
    "compute_confusion_matrix",
    "compute_classification_metrics",
    "compute_multiclass_roc_auc",
    "evaluate_model",
]
