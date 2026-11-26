"""CLI script to load raw WISDM files, preprocess, segment, scale, and save splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.data import (
    ChannelStandardizer,
    clean_sensor_dataframe,
    create_segments,
    filter_and_map_activities,
    load_wisdm_raw,
    save_processed_splits,
    stratified_train_val_test_split,
)
from src.features import plot_eda_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WISDM dataset splits for CNN-LSTM.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional file limit for quick runs.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip saving EDA plots.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    seg_cfg = cfg["segmentation"]
    split_cfg = cfg["split"]

    raw_df = load_wisdm_raw(
        raw_dir=data_cfg["raw_dir"],
        devices=data_cfg.get("devices", ["phone"]),
        sensors=data_cfg.get("sensors", ["accel"]),
        fallback_dir=data_cfg.get("fallback_raw_dir"),
        max_files=args.max_files,
    )
    cleaned_df = clean_sensor_dataframe(raw_df, features=data_cfg["features"])
    filtered_df = filter_and_map_activities(cleaned_df, activity_map=data_cfg["activity_map"])

    if not args.skip_eda:
        plot_eda_summary(
            filtered_df,
            plots_dir=cfg["results"]["plots_dir"],
            class_names=data_cfg["class_names"],
            features=data_cfg["features"],
        )

    X, y = create_segments(
        filtered_df,
        time_steps=int(seg_cfg["time_steps"]),
        step=int(seg_cfg["step"]),
        features=data_cfg["features"],
        label_strategy=str(seg_cfg.get("label_strategy", "majority")),
    )
    splits = stratified_train_val_test_split(
        X,
        y,
        test_size=float(split_cfg["test_size"]),
        val_size=float(split_cfg["val_size"]),
        random_state=int(split_cfg["random_state"]),
    )

    # Fit standardizer strictly on training windows to prevent data leakage
    scaler = ChannelStandardizer(features=data_cfg["features"])
    splits["X_train"] = scaler.fit_transform(splits["X_train"])
    if len(splits["X_val"]) > 0:
        splits["X_val"] = scaler.transform(splits["X_val"])
    splits["X_test"] = scaler.transform(splits["X_test"])

    scaler.save(cfg["training"]["scaler_path"])
    out_dir = save_processed_splits(splits, data_cfg["processed_dir"])
    print(
        f"Saved processed splits to {out_dir}: "
        f"train={splits['X_train'].shape}, val={splits['X_val'].shape}, test={splits['X_test'].shape}"
    )


if __name__ == "__main__":
    main()
