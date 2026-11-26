"""Time-domain feature extraction and EDA summaries for wearable sensor data."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_class_feature_stats(
    df: pd.DataFrame,
    features: Sequence[str] = ("X", "Y", "Z"),
    label_col: str = "final_activity_int",
) -> pd.DataFrame:
    """Compute per-class mean, std, min, and max across sensor axes."""
    return df.groupby(label_col)[list(features)].agg(["mean", "std", "min", "max"])


def compute_axis_correlations(
    df: pd.DataFrame,
    features: Sequence[str] = ("X", "Y", "Z"),
) -> pd.DataFrame:
    """Compute Pearson correlation matrix across sensor axes."""
    return df[list(features)].corr()


def compute_window_time_domain_features(
    segments: np.ndarray,
    axis_names: Sequence[str] = ("X", "Y", "Z"),
) -> pd.DataFrame:
    """Compute classic HAR time-domain features for each segmented window `(N, T, C)`."""
    if segments.ndim != 3:
        raise ValueError(f"Expected 3D array (N, T, C), got shape {segments.shape}")

    data: dict[str, np.ndarray] = {}
    for idx, axis in enumerate(axis_names):
        sig = segments[:, :, idx]
        data[f"{axis}_mean"] = sig.mean(axis=1)
        data[f"{axis}_std"] = sig.std(axis=1)
        data[f"{axis}_rms"] = np.sqrt(np.mean(sig ** 2, axis=1))
        data[f"{axis}_min"] = sig.min(axis=1)
        data[f"{axis}_max"] = sig.max(axis=1)
        centered = sig - sig.mean(axis=1, keepdims=True)
        zcr = np.mean(np.diff(np.signbit(centered), axis=1), axis=1)
        data[f"{axis}_zcr"] = zcr

    # Signal Magnitude Area (SMA) and mean vector magnitude
    data["sma"] = np.mean(np.sum(np.abs(segments), axis=2), axis=1)
    data["vm_mean"] = np.mean(np.sqrt(np.sum(segments ** 2, axis=2)), axis=1)
    return pd.DataFrame(data)


def plot_eda_summary(
    df: pd.DataFrame,
    plots_dir: str | Path,
    class_names: Sequence[str],
    features: Sequence[str] = ("X", "Y", "Z"),
    label_col: str = "final_activity_int",
) -> list[Path]:
    """Generate and save core EDA plots (class counts, axis means/stds, correlation)."""
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Class distribution
    counts = df[label_col].value_counts().sort_index()
    labels = [class_names[i] if i < len(class_names) else str(i) for i in counts.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, counts.values, color="#2563eb")
    ax.set_title("Activity Class Distribution")
    ax.set_ylabel("Sample Count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p1 = out_dir / "eda_class_distribution.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved.append(p1)

    # 2. Per-class axis std
    stds = df.groupby(label_col)[list(features)].std()
    stds.index = [class_names[i] if i < len(class_names) else str(i) for i in stds.index]
    ax = stds.plot(kind="bar", figsize=(9, 4.5))
    ax.set_title("Axis Standard Deviation per Activity")
    ax.set_ylabel("Standard Deviation")
    ax.tick_params(axis="x", rotation=20)
    fig = ax.get_figure()
    fig.tight_layout()
    p2 = out_dir / "eda_axis_std.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved.append(p2)

    # 3. Correlation heatmap
    corr = compute_axis_correlations(df, features=features)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), features)
    ax.set_yticks(range(len(features)), features)
    for r in range(len(features)):
        for c in range(len(features)):
            ax.text(c, r, f"{corr.values[r, c]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    ax.set_title("Sensor Axis Correlation Heatmap")
    fig.tight_layout()
    p3 = out_dir / "eda_axis_correlation.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved.append(p3)

    return saved
