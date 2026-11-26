"""Unit tests for sliding-window sensor segmentation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data.segmentation import create_segments


class TestSegmentation(unittest.TestCase):
    def test_windows_do_not_cross_user_boundaries(self) -> None:
        n_per_user = 150
        df = pd.DataFrame(
            {
                "user": np.repeat([1600, 1601], n_per_user),
                "device": ["phone"] * (2 * n_per_user),
                "sensor": ["accel"] * (2 * n_per_user),
                "X": np.concatenate([np.ones(n_per_user), np.full(n_per_user, -1.0)]),
                "Y": np.zeros(2 * n_per_user),
                "Z": np.zeros(2 * n_per_user),
                "final_activity_int": np.repeat([0, 1], n_per_user),
            }
        )
        X, y = create_segments(df, time_steps=100, step=50, label_strategy="majority")
        # Each user of length 150 with window=100, step=50 yields 2 windows -> 4 total
        self.assertEqual(X.shape, (4, 100, 3))
        self.assertListEqual(y.tolist(), [0, 0, 1, 1])
        self.assertTrue(np.all(X[:2, :, 0] == 1.0))
        self.assertTrue(np.all(X[2:, :, 0] == -1.0))

    def test_short_stream_returns_empty_arrays(self) -> None:
        df = pd.DataFrame(
            {
                "user": [1600] * 20,
                "device": ["phone"] * 20,
                "sensor": ["accel"] * 20,
                "X": np.zeros(20),
                "Y": np.zeros(20),
                "Z": np.zeros(20),
                "final_activity_int": [0] * 20,
            }
        )
        X, y = create_segments(df, time_steps=100, step=50)
        self.assertEqual(X.shape, (0, 100, 3))
        self.assertEqual(len(y), 0)

    def test_label_strategy_last_vs_majority(self) -> None:
        labels = [0] * 90 + [2] * 10
        df = pd.DataFrame(
            {
                "user": [1600] * 100,
                "device": ["phone"] * 100,
                "sensor": ["accel"] * 100,
                "X": np.zeros(100),
                "Y": np.zeros(100),
                "Z": np.zeros(100),
                "final_activity_int": labels,
            }
        )
        _, y_maj = create_segments(df, time_steps=100, step=50, label_strategy="majority")
        _, y_last = create_segments(df, time_steps=100, step=50, label_strategy="last")
        self.assertEqual(int(y_maj[0]), 0)
        self.assertEqual(int(y_last[0]), 2)


if __name__ == "__main__":
    unittest.main()
