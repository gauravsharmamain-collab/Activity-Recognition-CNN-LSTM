"""Inference pipeline for predicting human activities from sensor windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import DEFAULT_CLASS_NAMES, ChannelStandardizer, clean_sensor_dataframe
from src.data.segmentation import create_segments
from src.models.cnn_lstm import CNNLSTMClassifier
from src.training.train import resolve_device


class ActivityPredictor:
    """Wrapper around a trained ``CNNLSTMClassifier`` and optional feature scaler."""

    def __init__(
        self,
        model: CNNLSTMClassifier,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        scaler: ChannelStandardizer | None = None,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = list(class_names)
        self.scaler = scaler

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        scaler_path: str | Path | None = None,
        device: str = "auto",
    ) -> "ActivityPredictor":
        """Load predictor from a saved `.pt` checkpoint and optional scaler JSON."""
        dev = resolve_device(device)
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        model_cfg = ckpt.get("model_config", {})
        model = CNNLSTMClassifier(
            in_channels=int(model_cfg.get("in_channels", 3)),
            conv_filters=int(model_cfg.get("conv_filters", 64)),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            lstm_hidden_size=int(model_cfg.get("lstm_hidden_size", 100)),
            lstm_num_layers=int(model_cfg.get("lstm_num_layers", 2)),
            fc_hidden_size=int(model_cfg.get("fc_hidden_size", 100)),
            num_classes=int(model_cfg.get("num_classes", 6)),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        class_names = ckpt.get("class_names") or DEFAULT_CLASS_NAMES
        scaler = ChannelStandardizer.load(scaler_path) if scaler_path and Path(scaler_path).exists() else None
        return cls(model=model, class_names=class_names, scaler=scaler, device=str(dev))

    def predict_batch(self, windows: np.ndarray, apply_scaler: bool = True) -> list[dict[str, Any]]:
        """Predict activity labels and class probabilities for a batch `(N, T, C)`."""
        arr = np.asarray(windows, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if apply_scaler and self.scaler is not None:
            arr = np.asarray(self.scaler.transform(arr), dtype=np.float32)

        with torch.no_grad():
            logits = self.model(torch.as_tensor(arr, dtype=torch.float32, device=self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        results: list[dict[str, Any]] = []
        for row_probs in probs:
            cls_id = int(np.argmax(row_probs))
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            results.append(
                {
                    "class_id": cls_id,
                    "label": label,
                    "confidence": float(row_probs[cls_id]),
                    "probabilities": {
                        (self.class_names[i] if i < len(self.class_names) else str(i)): float(p)
                        for i, p in enumerate(row_probs)
                    },
                }
            )
        return results

    def predict_window(self, window: np.ndarray, apply_scaler: bool = True) -> dict[str, Any]:
        """Predict activity for a single sensor window `(T, C)`."""
        return self.predict_batch(window, apply_scaler=apply_scaler)[0]

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        time_steps: int = 100,
        step: int = 50,
        features: Sequence[str] = ("X", "Y", "Z"),
    ) -> list[dict[str, Any]]:
        """Clean, segment, and classify a raw sensor DataFrame."""
        work = clean_sensor_dataframe(df, features=features)
        if self.scaler is not None:
            work = self.scaler.transform(work)
        work = work.copy()
        work["_dummy_label"] = 0
        segments, _ = create_segments(
            work,
            time_steps=time_steps,
            step=step,
            features=features,
            label_col="_dummy_label",
        )
        if len(segments) == 0:
            return []
        return self.predict_batch(segments, apply_scaler=False)
