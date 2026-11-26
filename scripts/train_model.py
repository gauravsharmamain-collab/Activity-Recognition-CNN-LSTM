"""CLI script to train the PyTorch CNN-LSTM activity recognition model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.data import load_processed_splits
from src.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN-LSTM on processed WISDM splits.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = load_processed_splits(cfg["data"]["processed_dir"])
    _, history = train_model(
        splits["X_train"],
        splits["y_train"],
        splits["X_val"],
        splits["y_val"],
        cfg=cfg,
        verbose=True,
    )
    print(
        f"Training finished. Best val_acc="
        f"{max(history['val_acc']) if history['val_acc'] else max(history['train_acc']):.4f}"
    )


if __name__ == "__main__":
    main()
