"""CLI script to evaluate a trained CNN-LSTM checkpoint on the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.data import load_processed_splits
from src.evaluation import evaluate_model
from src.inference import ActivityPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNN-LSTM checkpoint on test data.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional override checkpoint path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = args.checkpoint or cfg["training"]["checkpoint_path"]
    splits = load_processed_splits(cfg["data"]["processed_dir"])
    predictor = ActivityPredictor.from_checkpoint(
        checkpoint_path=ckpt_path,
        device=str(cfg["training"].get("device", "auto")),
    )

    metrics = evaluate_model(
        model=predictor.model,
        X_test=splits["X_test"],
        y_test=splits["y_test"],
        class_names=cfg["data"]["class_names"],
        metrics_dir=cfg["results"]["metrics_dir"],
        plots_dir=cfg["results"]["plots_dir"],
        reports_dir=cfg["results"]["reports_dir"],
        batch_size=int(cfg["training"].get("batch_size", 64)),
        device=predictor.device,
    )
    print(
        f"Test Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | Macro AUC: {metrics['macro_auc']}"
    )


if __name__ == "__main__":
    main()
