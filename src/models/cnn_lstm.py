"""Hybrid 1D-CNN + stacked LSTM model for multi-axis wearable activity recognition."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


class CNNLSTMClassifier(nn.Module):
    """1D-CNN feature extractor followed by stacked LSTM sequence classifier.

    Expects inputs of shape ``(batch_size, time_steps, in_channels)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_filters: int = 64,
        kernel_size: int = 3,
        lstm_hidden_size: int = 100,
        lstm_num_layers: int = 2,
        fc_hidden_size: int = 100,
        num_classes: int = 6,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1.")
        if lstm_num_layers < 1:
            raise ValueError("lstm_num_layers must be >= 1.")

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv_filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden_size, fc_hidden_size),
            nn.ReLU(),
            nn.Linear(fc_hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits for a batch of sensor windows ``(B, T, C)``."""
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor (B, T, C), got shape {tuple(x.shape)}.")
        # Conv1d expects (B, C, T)
        features = self.conv(x.permute(0, 2, 1))
        # LSTM expects (B, T', F)
        seq_out, _ = self.lstm(features.permute(0, 2, 1))
        last_hidden = seq_out[:, -1, :]
        return self.classifier(last_hidden)


def build_model_from_config(cfg: Mapping[str, Any]) -> CNNLSTMClassifier:
    """Instantiate ``CNNLSTMClassifier`` from a parsed YAML configuration dict."""
    model_cfg = cfg.get("model", {})
    return CNNLSTMClassifier(
        in_channels=int(model_cfg.get("in_channels", 3)),
        conv_filters=int(model_cfg.get("conv_filters", 64)),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        lstm_hidden_size=int(model_cfg.get("lstm_hidden_size", 100)),
        lstm_num_layers=int(model_cfg.get("lstm_num_layers", 2)),
        fc_hidden_size=int(model_cfg.get("fc_hidden_size", 100)),
        num_classes=int(model_cfg.get("num_classes", 6)),
        dropout=float(model_cfg.get("dropout", 0.5)),
    )
