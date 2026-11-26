"""Utilities for loading raw WISDM smartphone/smartwatch sensor files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

RAW_COLUMNS = ["user", "activity_code", "timestamp", "X", "Y", "Z"]


def resolve_raw_data_dir(
    raw_dir: str | Path,
    fallback_dir: str | Path | None = None,
) -> Path:
    """Resolve the directory containing device/sensor subdirectories (`phone`, `watch`)."""
    candidates = [Path(raw_dir)]
    if fallback_dir is not None:
        candidates.append(Path(fallback_dir))

    for base in candidates:
        if not base.exists():
            continue
        if (base / "phone").exists() or (base / "watch").exists():
            return base
        nested = base / "data" / "raw"
        if (nested / "phone").exists() or (nested / "watch").exists():
            return nested
        double_nested = base / "wisdm-dataset" / "data" / "raw"
        if (double_nested / "phone").exists() or (double_nested / "watch").exists():
            return double_nested

    raise FileNotFoundError(
        f"Could not locate WISDM raw sensor folders ('phone'/'watch') in {raw_dir}"
        + (f" or fallback {fallback_dir}" if fallback_dir else "")
    )


def parse_wisdm_file(
    file_path: str | Path,
    device: str = "phone",
    sensor: str = "accel",
) -> pd.DataFrame:
    """Parse a single WISDM raw `.txt` sensor file into a typed DataFrame."""
    path = Path(file_path)
    df = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        sep=",",
        on_bad_lines="skip",
        engine="python",
    )
    if df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS + ["device", "sensor", "source_file"])

    df["Z"] = df["Z"].astype(str).str.replace(";", "", regex=False).str.strip()
    df["activity_code"] = df["activity_code"].astype(str).str.strip()
    for col in ("user", "timestamp", "X", "Y", "Z"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["user", "activity_code", "timestamp", "X", "Y", "Z"]).copy()
    df["user"] = df["user"].astype(int)
    df["timestamp"] = df["timestamp"].astype("int64")
    df["device"] = device
    df["sensor"] = sensor
    df["source_file"] = path.name
    return df


def load_wisdm_raw(
    raw_dir: str | Path,
    devices: Sequence[str] = ("phone",),
    sensors: Sequence[str] = ("accel",),
    fallback_dir: str | Path | None = None,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Load and concatenate WISDM raw sensor `.txt` files across devices and sensors."""
    base_dir = resolve_raw_data_dir(raw_dir, fallback_dir=fallback_dir)
    frames: list[pd.DataFrame] = []

    for device in devices:
        for sensor in sensors:
            folder = base_dir / device / sensor
            if not folder.exists():
                continue
            files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix == ".txt")
            if max_files is not None:
                files = files[:max_files]
            for fp in files:
                parsed = parse_wisdm_file(fp, device=device, sensor=sensor)
                if not parsed.empty:
                    frames.append(parsed)

    if not frames:
        raise ValueError(
            f"No valid WISDM .txt files loaded from {base_dir} "
            f"for devices={list(devices)}, sensors={list(sensors)}."
        )

    return pd.concat(frames, ignore_index=True)
