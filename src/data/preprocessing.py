"""Preprocessing utilities for WISDM activity recognition sensor streams."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_ACTIVITY_MAP: dict[str, int] = {
    "A": 0,  # Walking
    "B": 1,  # Jogging
    "C": 2,  # Stairs
    "D": 3,  # Sitting
    "E": 4,  # Standing
    "H": 5,  # Eating Soup -> Eating
    "I": 5,  # Eating Chips -> Eating
    "J": 5,  # Eating Pasta -> Eating
    "L": 5,  # Eating Sandwich -> Eating
}

DEFAULT_CLASS_NAMES: list[str] = [
    "Walking",
    "Jogging",
    "Stairs",
    "Sitting",
    "Standing",
    "Eating",
]


def clean_sensor_dataframe(
    df: pd.DataFrame,
    features: Sequence[str] = ("X", "Y", "Z"),
) -> pd.DataFrame:
    """Drop invalid/NaN rows and sort sensor readings chronologically per stream."""
    cleaned = df.dropna(subset=["user", "activity_code", "timestamp", *features]).copy()
    for col in features:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna(subset=list(features))

    sort_cols = [c for c in ("user", "device", "sensor", "timestamp") if c in cleaned.columns]
    if sort_cols:
        cleaned = cleaned.sort_values(sort_cols, kind="mergesort")
    return cleaned.reset_index(drop=True)


def filter_and_map_activities(
    df: pd.DataFrame,
    activity_map: Mapping[str, int] | None = None,
    target_col: str = "final_activity_int",
) -> pd.DataFrame:
    """Filter rows to target activity codes and map them to integer class IDs."""
    mapping = dict(activity_map or DEFAULT_ACTIVITY_MAP)
    out = df.copy()
    out[target_col] = out["activity_code"].map(mapping)
    out = out.dropna(subset=[target_col]).copy()
    out[target_col] = out[target_col].astype(int)
    return out.reset_index(drop=True)


class ChannelStandardizer:
    """Zero-mean, unit-variance normalizer for multi-axis sensor features."""

    def __init__(self, features: Sequence[str] = ("X", "Y", "Z")) -> None:
        self.features = list(features)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, data: pd.DataFrame | np.ndarray) -> "ChannelStandardizer":
        values = data[self.features].to_numpy(dtype=np.float64) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float64)
        if values.ndim == 3:
            values = values.reshape(-1, values.shape[-1])
        self.mean_ = values.mean(axis=0)
        std = values.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("ChannelStandardizer must be fitted before calling transform().")
        if isinstance(data, pd.DataFrame):
            out = data.copy()
            vals = out[self.features].to_numpy(dtype=np.float64)
            out[self.features] = ((vals - self.mean_) / self.std_).astype(np.float32)
            return out
        arr = np.asarray(data, dtype=np.float32)
        return ((arr - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        return self.fit(data).transform(data)

    def save(self, path: str | Path) -> Path:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Cannot save an unfitted ChannelStandardizer.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features": self.features,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ChannelStandardizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scaler = cls(features=payload["features"])
        scaler.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        scaler.std_ = np.asarray(payload["std"], dtype=np.float64)
        return scaler


def _stratified_split_indices(
    y: np.ndarray,
    holdout_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train and holdout sets preserving class proportions."""
    train_idx: list[np.ndarray] = []
    holdout_idx: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        n_holdout = max(1, int(round(len(cls_indices) * holdout_ratio))) if len(cls_indices) > 1 and holdout_ratio > 0 else 0
        holdout_idx.append(cls_indices[:n_holdout])
        train_idx.append(cls_indices[n_holdout:])
    train_arr = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    holdout_arr = np.concatenate(holdout_idx) if holdout_idx else np.array([], dtype=int)
    rng.shuffle(train_arr)
    rng.shuffle(holdout_arr)
    return train_arr, holdout_arr


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Split segmented windows into stratified train, validation, and test sets."""
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1.")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be in [0, 1).")

    rng = np.random.default_rng(random_state)
    train_val_idx, test_idx = _stratified_split_indices(y, test_size, rng)
    X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if val_size > 0 and len(y_train_val) > 0:
        rel_val = val_size / (1.0 - test_size)
        rel_val = min(max(rel_val, 0.0), 0.9)
        tr_idx, val_idx = _stratified_split_indices(y_train_val, rel_val, rng)
        X_train, y_train = X_train_val[tr_idx], y_train_val[tr_idx]
        X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]
    else:
        X_train, y_train = X_train_val, y_train_val
        X_val = np.empty((0, *X.shape[1:]), dtype=X.dtype)
        y_val = np.empty((0,), dtype=y.dtype)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
