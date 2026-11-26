"""Check the source contract, split stability, and refusal to hide bad data.

Run: .venv/bin/python -m unittest discover -s tests -v
The original local CSV is required; no models are created or trained.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.lol_preprocessing import (
    FEATURE_NAMES, TARGET, create_train_test_split, feature_decisions,
    load_data, prepare_data, prepare_features, save_split_manifest, validate_data,
)


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_data()
        cls.prepared = prepare_data()

    def test_original_data_contract(self):
        self.assertEqual(self.frame.shape, (9879, 40))
        report = validate_data(self.frame)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["relationships"]["violations"].sum(), 0)
        self.assertEqual(len(feature_decisions()), 40)

    def test_split_partition_and_alignment(self):
        split = self.prepared
        self.assertEqual(split.X_train.shape, (7903, 12))
        self.assertEqual(split.X_test.shape, (1976, 12))
        self.assertEqual(set(split.X_train.index) & set(split.X_test.index), set())
        self.assertEqual(set(split.X_train.index) | set(split.X_test.index), set(self.frame.gameId))
        pd.testing.assert_index_equal(split.X_train.index, split.y_train.index)
        pd.testing.assert_index_equal(split.X_test.index, split.y_test.index)
        self.assertEqual(split.y_train.value_counts().sort_index().tolist(), [3959, 3944])
        self.assertEqual(split.y_test.value_counts().sort_index().tolist(), [990, 986])
        labels = self.frame.set_index("gameId")[TARGET]
        pd.testing.assert_series_equal(split.y_train, labels.loc[split.y_train.index])
        pd.testing.assert_series_equal(split.y_test, labels.loc[split.y_test.index])

    def test_split_reproducibility_and_row_order(self):
        original = create_train_test_split(self.frame)
        shuffled = create_train_test_split(self.frame.sample(frac=1, random_state=42))
        for left, right in zip(original, shuffled):
            if isinstance(left, pd.DataFrame):
                pd.testing.assert_frame_equal(left, right)
            else:
                pd.testing.assert_series_equal(left, right)

    def test_features_do_not_read_target(self):
        without_target = self.frame.drop(columns=TARGET)
        features = prepare_features(without_target)
        pd.testing.assert_frame_equal(features, prepare_features(self.frame))
        self.assertEqual(tuple(features.columns), FEATURE_NAMES)
        self.assertTrue({TARGET, "gameId"}.isdisjoint(features.columns))
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        np.testing.assert_array_equal(features.killsDiff, self.frame.blueKills - self.frame.redKills)
        np.testing.assert_array_equal(features.minionsDiff, self.frame.blueTotalMinionsKilled - self.frame.redTotalMinionsKilled)

    def test_missing_values_rejected(self):
        changed = self.frame.copy()
        changed.loc[0, "blueAvgLevel"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing values"):
            validate_data(changed)

    def test_duplicate_identifiers_rejected(self):
        changed = self.frame.copy()
        changed.loc[1, "gameId"] = changed.loc[0, "gameId"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_data(changed)

    def test_identical_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_data(pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True))

    def test_invalid_target_rejected(self):
        changed = self.frame.copy()
        changed.loc[0, TARGET] = 2
        with self.assertRaisesRegex(ValueError, "Invalid values"):
            validate_data(changed)

    def test_negative_counts_rejected(self):
        changed = self.frame.copy()
        changed.loc[0, "blueKills"] = -1
        with self.assertRaisesRegex(ValueError, "Invalid values"):
            validate_data(changed)

    def test_inconsistent_arithmetic_rejected(self):
        changed = self.frame.copy()
        changed.loc[0, "blueGoldDiff"] += 1
        with self.assertRaisesRegex(ValueError, "arithmetic relationships"):
            validate_data(changed)

    def test_unexpected_schema_and_nonnumeric_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected schema"):
            validate_data(self.frame.drop(columns="redKills"))
        changed = self.frame.copy()
        changed["blueKills"] = changed["blueKills"].astype(str)
        with self.assertRaisesRegex(ValueError, "Non-numeric"):
            validate_data(changed)

    def test_manifest_refuses_changed_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            save_split_manifest(self.prepared, path)
            self.assertEqual(save_split_manifest(self.prepared, path), path)
            manifest = json.loads(path.read_text())
            self.assertNotIn("y_test", manifest)
            manifest["test_game_ids"][0] = -1
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                save_split_manifest(self.prepared, path)

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "high_diamond_ranked_10min.csv"
            path.write_text("changed file\n")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                prepare_data(path)


if __name__ == "__main__":
    unittest.main()
