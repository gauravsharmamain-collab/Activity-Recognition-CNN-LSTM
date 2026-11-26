"""PyTorch training loop, checkpointing, and learning-curve visualization."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.cnn_lstm import CNNLSTMClassifier, build_model_from_config


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_pref: str = "auto") -> torch.device:
    """Resolve PyTorch compute device (`cpu` or `cuda`)."""
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader from NumPy arrays."""
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_n = targets.size(0)
            total_loss += float(loss.item()) * batch_n
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += batch_n

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def save_training_curves(history: Mapping[str, list[float]], plots_dir: str | Path) -> Path:
    """Plot and save training/validation loss and accuracy curves."""
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    if history.get("val_loss"):
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="o", markersize=3)
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    if history.get("val_acc"):
        axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="o", markersize=3)
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Mapping[str, Any],
    verbose: bool = True,
) -> tuple[CNNLSTMClassifier, dict[str, list[float]]]:
    """Train ``CNNLSTMClassifier`` and persist best checkpoint + training artifacts."""
    train_cfg = cfg.get("training", {})
    results_cfg = cfg.get("results", {})
    seed = int(train_cfg.get("seed", cfg.get("project", {}).get("seed", 42)))
    set_seed(seed)

    device = resolve_device(str(train_cfg.get("device", "auto")))
    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 64))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    checkpoint_path = Path(train_cfg.get("checkpoint_path", "models/cnn_lstm_best.pt"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    has_val = len(X_val) > 0
    val_loader = make_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False) if has_val else None

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        if val_loader is not None:
            val_loss, val_acc = _run_epoch(model, val_loader, criterion, device, optimizer=None)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            monitor_score = val_acc
        else:
            val_loss, val_acc = tr_loss, tr_acc
            monitor_score = tr_acc

        if monitor_score >= best_score:
            best_score = monitor_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": cfg.get("model", {}),
                    "class_names": cfg.get("data", {}).get("class_names", []),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

        if verbose:
            msg = f"Epoch {epoch:02d}/{epochs:02d} - train_loss: {tr_loss:.4f} - train_acc: {tr_acc:.4f}"
            if val_loader is not None:
                msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
            print(msg)

    metrics_dir = Path(results_cfg.get("metrics_dir", "results/metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    plots_dir = Path(results_cfg.get("plots_dir", "results/plots"))
    save_training_curves(history, plots_dir)

    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    return model, history
