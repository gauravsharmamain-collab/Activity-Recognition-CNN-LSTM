"""Unit tests for the PyTorch CNN-LSTM classifier, training, evaluation, and inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.evaluation.evaluate import compute_classification_metrics, evaluate_model
from src.inference.predict import ActivityPredictor
from src.models.cnn_lstm import CNNLSTMClassifier, build_model_from_config
from src.training.train import train_model


class TestModel(unittest.TestCase):
    def test_forward_output_shape_and_backward_pass(self) -> None:
        model = CNNLSTMClassifier(
            in_channels=3,
            conv_filters=32,
            kernel_size=3,
            lstm_hidden_size=32,
            lstm_num_layers=2,
            fc_hidden_size=32,
            num_classes=6,
            dropout=0.2,
        )
        x = torch.randn(8, 100, 3)
        logits = model(x)
        self.assertEqual(logits.shape, (8, 6))
        loss = logits.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grad)

    def test_build_from_config(self) -> None:
        cfg = {
            "model": {
                "in_channels": 3,
                "conv_filters": 16,
                "kernel_size": 5,
                "lstm_hidden_size": 24,
                "lstm_num_layers": 1,
                "fc_hidden_size": 16,
                "num_classes": 6,
                "dropout": 0.1,
            }
        }
        model = build_model_from_config(cfg)
        out = model(torch.randn(4, 100, 3))
        self.assertEqual(out.shape, (4, 6))

    def test_end_to_end_smoke_train_evaluate_and_predict(self) -> None:
        rng = np.random.default_rng(7)
        X_train = rng.normal(size=(24, 50, 3)).astype(np.float32)
        y_train = np.tile(np.arange(6), 4).astype(np.int64)
        X_val = rng.normal(size=(12, 50, 3)).astype(np.float32)
        y_val = np.tile(np.arange(6), 2).astype(np.int64)
        class_names = ["Walking", "Jogging", "Stairs", "Sitting", "Standing", "Eating"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ckpt = tmp / "best.pt"
            cfg = {
                "data": {"class_names": class_names},
                "model": {
                    "in_channels": 3,
                    "conv_filters": 16,
                    "kernel_size": 3,
                    "lstm_hidden_size": 16,
                    "lstm_num_layers": 2,
                    "fc_hidden_size": 16,
                    "num_classes": 6,
                    "dropout": 0.1,
                },
                "training": {
                    "epochs": 2,
                    "batch_size": 8,
                    "learning_rate": 0.01,
                    "seed": 42,
                    "device": "cpu",
                    "checkpoint_path": str(ckpt),
                },
                "results": {
                    "plots_dir": str(tmp / "plots"),
                    "metrics_dir": str(tmp / "metrics"),
                    "reports_dir": str(tmp / "reports"),
                },
            }

            model, history = train_model(X_train, y_train, X_val, y_val, cfg=cfg, verbose=False)
            self.assertEqual(len(history["train_loss"]), 2)
            self.assertTrue(ckpt.exists())

            metrics = evaluate_model(
                model,
                X_val,
                y_val,
                class_names=class_names,
                metrics_dir=tmp / "metrics",
                plots_dir=tmp / "plots",
                reports_dir=tmp / "reports",
                device="cpu",
            )
            self.assertIn("accuracy", metrics)
            self.assertTrue((tmp / "metrics" / "evaluation_metrics.json").exists())
            self.assertTrue((tmp / "reports" / "classification_report.txt").exists())

            predictor = ActivityPredictor.from_checkpoint(ckpt, device="cpu")
            pred = predictor.predict_window(X_val[0], apply_scaler=False)
            self.assertIn(pred["label"], class_names)
            self.assertGreaterEqual(pred["confidence"], 0.0)
            self.assertLessEqual(pred["confidence"], 1.0)

    def test_classification_metrics_exact(self) -> None:
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 0])
        metrics = compute_classification_metrics(y_true, y_pred, ["A", "B", "C"])
        self.assertAlmostEqual(metrics["accuracy"], 0.75, places=4)


if __name__ == "__main__":
    unittest.main()
