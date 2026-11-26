"""Evaluation utilities for multi-class activity recognition models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def predict_probabilities(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Compute softmax class probabilities `(N, num_classes)` for segmented windows."""
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.as_tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(dev))
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(batch_probs)
    return np.concatenate(probs, axis=0) if probs else np.empty((0, 0), dtype=np.float32)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute `(num_classes, num_classes)` confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute accuracy, per-class precision/recall/F1, and macro/weighted averages."""
    num_classes = len(class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    total = int(cm.sum())
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []

    for idx, name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        support = int(cm[idx, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    support_arr = np.asarray(supports, dtype=np.float64)
    weight_sum = float(support_arr.sum())
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_f1 = float(np.average(f1s, weights=support_arr)) if weight_sum > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        "macro_recall": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _binary_roc_curve(y_binary: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute (fpr, tpr, auc) for one binary class using trapezoidal integration."""
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_binary[order]
    pos = int(y_sorted.sum())
    neg = int(len(y_sorted) - pos)
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate(([0.0], tps / pos))
    fpr = np.concatenate(([0.0], fps / neg))
    roc_auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return fpr, tpr, roc_auc


def compute_multiclass_roc_auc(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Compute one-vs-rest ROC curves and macro AUC across classes."""
    per_class_auc: dict[str, float] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    valid_aucs: list[float] = []

    for idx, name in enumerate(class_names):
        y_bin = (y_true == idx).astype(np.int64)
        fpr, tpr, auc_val = _binary_roc_curve(y_bin, y_probs[:, idx])
        curves[name] = (fpr, tpr, auc_val)
        if not np.isnan(auc_val):
            per_class_auc[name] = round(auc_val, 4)
            valid_aucs.append(auc_val)

    macro_auc = round(float(np.mean(valid_aucs)), 4) if valid_aucs else None
    return {
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "curves": curves,
    }


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            color = "white" if cm[r, c] > thresh else "black"
            ax.text(c, r, str(int(cm[r, c])), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, (fpr, tpr, auc_val) in curves.items():
        if np.isnan(auc_val):
            continue
        ax.plot(fpr, tpr, linewidth=1.6, label=f"{name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _format_text_report(metrics: dict[str, Any], class_names: Sequence[str]) -> str:
    lines = [
        "=== Activity Recognition Evaluation Report ===",
        f"Accuracy     : {metrics['accuracy']:.4f}",
        f"Macro F1     : {metrics['macro_f1']:.4f}",
        f"Weighted F1  : {metrics['weighted_f1']:.4f}",
        f"Macro AUC    : {metrics.get('macro_auc')}",
        "",
        f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
        "-" * 60,
    ]
    for name in class_names:
        row = metrics["per_class"][name]
        lines.append(
            f"{name:<15} {row['precision']:>10.4f} {row['recall']:>10.4f} "
            f"{row['f1']:>10.4f} {row['support']:>10d}"
        )
    return "\n".join(lines) + "\n"


def evaluate_model(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    metrics_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run full evaluation and optionally save metrics JSON, report TXT, and plots."""
    y_probs = predict_probabilities(model, X_test, batch_size=batch_size, device=device)
    y_pred = np.argmax(y_probs, axis=1)

    metrics = compute_classification_metrics(y_test, y_pred, class_names)
    roc_info = compute_multiclass_roc_auc(y_test, y_probs, class_names)
    metrics["macro_auc"] = roc_info["macro_auc"]
    metrics["per_class_auc"] = roc_info["per_class_auc"]

    if metrics_dir is not None:
        m_dir = Path(metrics_dir)
        m_dir.mkdir(parents=True, exist_ok=True)
        (m_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if reports_dir is not None:
        r_dir = Path(reports_dir)
        r_dir.mkdir(parents=True, exist_ok=True)
        (r_dir / "classification_report.txt").write_text(
            _format_text_report(metrics, class_names), encoding="utf-8"
        )

    if plots_dir is not None:
        p_dir = Path(plots_dir)
        p_dir.mkdir(parents=True, exist_ok=True)
        cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        _plot_confusion_matrix(cm, class_names, p_dir / "confusion_matrix.png")
        _plot_roc_curves(roc_info["curves"], p_dir / "roc_curves.png")

    return metrics
