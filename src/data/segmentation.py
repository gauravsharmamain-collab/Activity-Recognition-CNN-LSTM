"""Sliding-window segmentation for time-series sensor streams."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _window_label(window_labels: np.ndarray, strategy: str) -> int:
    if strategy == "last":
        return int(window_labels[-1])
    if strategy == "majority":
        values, counts = np.unique(window_labels, return_counts=True)
        return int(values[np.argmax(counts)])
    raise ValueError(f"Unsupported label_strategy '{strategy}'. Expected 'majority' or 'last'.")


def create_segments(
    df: pd.DataFrame,
    time_steps: int = 100,
    step: int = 50,
    features: Sequence[str] = ("X", "Y", "Z"),
    label_col: str = "final_activity_int",
    group_cols: Sequence[str] = ("user", "device", "sensor"),
    label_strategy: str = "majority",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert tabular sensor readings into 3D windows `(N, time_steps, n_features)`.

    Windows are sliced strictly within each `(user, device, sensor)` group so
    sequences never span across different subjects or devices.
    """
    if time_steps <= 0 or step <= 0:
        raise ValueError("time_steps and step must be positive integers.")

    active_groups = [c for c in group_cols if c in df.columns]
    grouped = df.groupby(active_groups, sort=False) if active_groups else [(None, df)]

    segments: list[np.ndarray] = []
    labels: list[int] = []
    feat_list = list(features)

    for _, group_df in grouped:
        n_rows = len(group_df)
        if n_rows < time_steps:
            continue
        values = group_df[feat_list].to_numpy(dtype=np.float32)
        target = group_df[label_col].to_numpy(dtype=np.int64)
        for start in range(0, n_rows - time_steps + 1, step):
            end = start + time_steps
            segments.append(values[start:end])
            labels.append(_window_label(target[start:end], label_strategy))

    if not segments:
        return (
            np.empty((0, time_steps, len(feat_list)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return np.asarray(segments, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def save_processed_splits(splits: Mapping[str, np.ndarray], processed_dir: str | Path) -> Path:
    """Save train/val/test NumPy arrays to `processed_dir`."""
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in splits.items():
        np.save(out_dir / f"{name}.npy", arr)
    return out_dir


def load_processed_splits(processed_dir: str | Path) -> dict[str, np.ndarray]:
    """Load train/val/test NumPy arrays from `processed_dir`."""
    in_dir = Path(processed_dir)
    keys = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
    splits: dict[str, np.ndarray] = {}
    for key in keys:
        path = in_dir / f"{key}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed split file: {path}")
        splits[key] = np.load(path)
    return splits
