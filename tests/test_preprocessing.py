"""Unit tests for data cleaning, activity mapping, standardization, and splitting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import parse_wisdm_file
from src.data.preprocessing import (
    ChannelStandardizer,
    clean_sensor_dataframe,
    filter_and_map_activities,
    stratified_train_val_test_split,
)


class TestPreprocessing(unittest.TestCase):
    def test_parse_wisdm_file_strips_semicolon_and_drops_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sample = Path(tmpdir) / "data_1600_accel_phone.txt"
            sample.write_text(
                "1600,A,100,-0.5,9.8,1.2;\n"
                "1600,B,200,bad_val,9.7,1.1;\n"
                "1600,B,300,0.4,9.6,0.9;\n",
                encoding="utf-8",
            )
            df = parse_wisdm_file(sample, device="phone", sensor="accel")
            self.assertEqual(len(df), 2)
            self.assertEqual(df["user"].dtype.kind, "i")
            self.assertAlmostEqual(float(df.iloc[0]["Z"]), 1.2, places=5)
            self.assertEqual(df.iloc[0]["device"], "phone")

    def test_filter_and_map_activities(self) -> None:
        df = pd.DataFrame(
            {
                "user": [1600, 1600, 1600, 1600],
                "activity_code": ["A", "H", "L", "Z"],
                "timestamp": [1, 2, 3, 4],
                "X": [0.1, 0.2, 0.3, 0.4],
                "Y": [0.1, 0.2, 0.3, 0.4],
                "Z": [0.1, 0.2, 0.3, 0.4],
            }
        )
        cleaned = clean_sensor_dataframe(df)
        mapped = filter_and_map_activities(cleaned)
        self.assertEqual(len(mapped), 3)
        self.assertListEqual(mapped["final_activity_int"].tolist(), [0, 5, 5])

    def test_channel_standardizer_fit_transform_and_persistence(self) -> None:
        arr = np.array(
            [
                [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
                [[5.0, 6.0, 7.0], [7.0, 8.0, 9.0]],
            ],
            dtype=np.float32,
        )
        scaler = ChannelStandardizer()
        norm = scaler.fit_transform(arr)
        self.assertTrue(np.allclose(norm.mean(axis=(0, 1)), 0.0, atol=1e-5))
        self.assertTrue(np.allclose(norm.std(axis=(0, 1)), 1.0, atol=1e-5))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scaler.json"
            scaler.save(path)
            loaded = ChannelStandardizer.load(path)
            norm2 = loaded.transform(arr)
            self.assertTrue(np.allclose(norm, norm2, atol=1e-6))

    def test_stratified_split_preserves_classes_and_sizes(self) -> None:
        X = np.random.randn(60, 10, 3).astype(np.float32)
        y = np.repeat(np.arange(6), 10)
        splits = stratified_train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42)
        total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
        self.assertEqual(total, 60)
        self.assertEqual(set(np.unique(splits["y_test"])), set(range(6)))


if __name__ == "__main__":
    unittest.main()
