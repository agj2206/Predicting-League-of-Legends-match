"""Fast Part B checks using a small subset of Diamond TRAINING records only.

The full 84-candidate experiment is never run by this suite. Tiny grids exercise
the same 5-outer/3-inner orchestration, exports and reloads in temporary folders.
"""

import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.lol_model_training import (
    CVConfig, FEATURE_NAMES, METRIC_NAMES, MODEL_NAMES, SCORING,
    build_cv_splits, build_models, classification_metrics, cv_manifest,
    load_final_model, load_run_results, load_training_data, parameter_grids,
    run_training, select_model, validate_training_input,
)


TINY_GRIDS = {
    "decision_tree": {"classifier__criterion": ["gini"],
                      "classifier__max_depth": [3], "classifier__min_samples_leaf": [5]},
    "knn": {"classifier__n_neighbors": [5], "classifier__weights": ["uniform"],
            "classifier__p": [2]},
    "svm": [
        {"classifier__kernel": ["linear"], "classifier__C": [0.1]},
        {"classifier__kernel": ["rbf"], "classifier__C": [0.1], "classifier__gamma": ["scale"]},
    ],
}


class ModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full_X, full_y, cls.provenance = load_training_data()
        cls.X, _, cls.y, _ = train_test_split(
            full_X, full_y, train_size=120, stratify=full_y, random_state=42,
        )
        cls.full_training_ids = set(full_X.index)

    def test_approved_grids_and_unscaled_tree(self):
        models, grids = build_models(), parameter_grids()
        self.assertEqual({k: len(ParameterGrid(v)) for k, v in grids.items()},
                         {"decision_tree": 40, "knn": 24, "svm": 20})
        for name in ("dummy", "decision_tree"):
            self.assertNotIn("scaler", models[name].named_steps)
        for name in ("knn", "svm"):
            self.assertEqual(list(models[name].named_steps), ["scaler", "classifier"])
        self.assertEqual(models["dummy"].get_params()["classifier__strategy"], "most_frequent")
        self.assertFalse(models["svm"].get_params()["classifier__probability"])
        self.assertNotIn("classifier__gamma", grids["svm"][0])
        for name in ("dummy", "decision_tree", "svm"):
            self.assertEqual(models[name].get_params()["classifier__random_state"], 42)
        self.assertEqual(set(SCORING), set(METRIC_NAMES))

    def test_training_api_has_no_test_arguments(self):
        parameters = inspect.signature(run_training).parameters
        self.assertEqual(list(parameters)[:2], ["X_train", "y_train"])
        self.assertNotIn("X_test", parameters)
        self.assertNotIn("y_test", parameters)

    def test_nested_folds_partition_training_only(self):
        outer, inner, final = build_cv_splits(self.X, self.y)
        self.assertEqual((len(outer), len(final)), (5, 3))
        seen_outer_validation = []
        for fold, (tr, va) in enumerate(outer):
            self.assertTrue(set(tr).isdisjoint(va))
            self.assertEqual(set(tr) | set(va), set(range(len(self.X))))
            self.assertEqual(len(inner[fold]), 3)
            seen_outer_validation.extend(va)
            for it, iv in inner[fold]:
                self.assertTrue(set(it).isdisjoint(iv))
                self.assertEqual(set(it) | set(iv), set(range(len(tr))))
                self.assertTrue(set(self.X.index[tr[it]]).isdisjoint(self.X.index[va]))
                self.assertEqual(set(self.y.iloc[tr[it]].unique()), {0, 1})
                self.assertEqual(set(self.y.iloc[tr[iv]].unique()), {0, 1})
            self.assertLess(abs(self.y.iloc[va].mean() - self.y.mean()), 0.05)
        self.assertEqual(sorted(seen_outer_validation), list(range(len(self.X))))
        manifest = cv_manifest(self.X, outer, inner, final, CVConfig())
        self.assertNotIn("y_train", json.dumps(manifest))
        for fold in manifest["outer_folds"]:
            self.assertTrue(set(fold["train_game_ids"]).issubset(self.full_training_ids))
            self.assertTrue(set(fold["validation_game_ids"]).issubset(self.full_training_ids))
        outer_again, inner_again, final_again = build_cv_splits(self.X, self.y)
        self.assertEqual(manifest, cv_manifest(self.X, outer_again, inner_again, final_again, CVConfig()))

    def test_scaler_learns_each_training_fold_only(self):
        # Shift one validation partition to make accidental whole-data scaling visible.
        altered = self.X.astype(float).copy()
        _, _, cv = build_cv_splits(altered, self.y)
        altered.iloc[cv[0][1], 0] += 100000
        for name in ("knn", "svm"):
            results = cross_validate(
                build_models()[name], altered, self.y, cv=cv,
                scoring="accuracy", return_estimator=True, error_score="raise",
            )
            for estimator, (tr, va) in zip(results["estimator"], cv):
                scaler = estimator.named_steps["scaler"]
                np.testing.assert_allclose(scaler.mean_, altered.iloc[tr].mean().to_numpy())
                self.assertEqual(scaler.n_samples_seen_, len(tr))
                old_mean = scaler.mean_.copy()
                estimator.predict(altered.iloc[va])
                np.testing.assert_array_equal(old_mean, scaler.mean_)
            self.assertFalse(np.allclose(
                results["estimator"][0].named_steps["scaler"].mean_, altered.mean(),
            ))

    def test_dummy_ignores_features_and_reports_zero_precision(self):
        # Give the reference model a Red-majority training subset.
        indices = list(self.y[self.y == 0].index[:20]) + list(self.y[self.y == 1].index[:10])
        dummy = build_models()["dummy"].fit(self.X.loc[indices], self.y.loc[indices])
        np.testing.assert_array_equal(dummy.predict(self.X), dummy.predict(self.X * 1000))
        metrics = classification_metrics(dummy, self.X, self.y)
        self.assertEqual(metrics["roc_auc"], 0.5)
        for metric in ("precision", "recall", "f1"):
            self.assertEqual(metrics[metric], 0.0)

    def test_input_contract_rejects_misalignment_and_bad_features(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            validate_training_input(self.X, self.y.iloc[::-1])
        with self.assertRaisesRegex(ValueError, "Feature"):
            validate_training_input(self.X.loc[:, list(FEATURE_NAMES)[::-1]], self.y)
        with self.assertRaisesRegex(ValueError, "Feature"):
            validate_training_input(self.X.assign(blueWins=self.y), self.y)
        invalid = self.X.astype(float).copy()
        invalid.iloc[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_training_input(invalid, self.y)
        with self.assertRaisesRegex(ValueError, "random_state"):
            CVConfig(random_state=7)

    def test_selection_rule_is_fixed_and_dummy_is_reference(self):
        frame = pd.DataFrame({
            "model": MODEL_NAMES,
            "mean_accuracy": [1.0, 0.7, 0.7, 0.7],
            "std_accuracy": [0, 0.02, 0.01, 0.01],
        })
        self.assertEqual(select_model(frame), "knn")
        frame.loc[frame.model == "decision_tree", "std_accuracy"] = 0.01
        self.assertEqual(select_model(frame), "decision_tree")

    def test_complete_small_run_reproducibility_and_artifact_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_fit = Pipeline.fit
            fit_ids = []

            def recording_fit(estimator, X, y=None, **kwargs):
                fit_ids.append(frozenset(X.index))
                return original_fit(estimator, X, y, **kwargs)

            first_X, first_y = self.X.copy(deep=True), self.y.copy(deep=True)
            with patch.object(Pipeline, "fit", recording_fit):
                first = run_training(
                    self.X, self.y, root=root, run_id="unit_first", n_jobs=1,
                    provenance=self.provenance, grids=TINY_GRIDS, progress=None,
                )
            second = run_training(
                self.X, self.y, root=root, run_id="unit_second", n_jobs=1,
                provenance=self.provenance, grids=TINY_GRIDS, progress=None,
            )
            pd.testing.assert_frame_equal(first_X, self.X)
            pd.testing.assert_series_equal(first_y, self.y)
            pd.testing.assert_frame_equal(first.comparison, second.comparison)
            a, b = load_run_results(root, "unit_first"), load_run_results(root, "unit_second")
            self.assertEqual(a["best_params"], b["best_params"])
            self.assertEqual(a["metadata"]["cv"], {"outer_folds": 5, "inner_folds": 3, "random_state": 42})
            self.assertFalse(a["metadata"]["held_out_test_used"])
            self.assertEqual(a["metadata"]["warnings"], [])
            self.assertEqual(len(a["outer_scores"]), 20)
            self.assertEqual(len(fit_ids), a["metadata"]["expected_fit_count"])
            manifest = json.loads((root / a["metadata"]["cv_manifest_path"]).read_text())
            allowed_fit_sets = {frozenset(self.X.index)}
            for fold in manifest["outer_folds"]:
                allowed_fit_sets.add(frozenset(fold["train_game_ids"]))
                allowed_fit_sets.update(frozenset(f["train_game_ids"]) for f in fold["inner_folds"])
            allowed_fit_sets.update(frozenset(f["train_game_ids"]) for f in manifest["final_search_folds"])
            self.assertTrue(all(ids in allowed_fit_sets for ids in fit_ids))
            self.assertTrue(all(ids.issubset(self.full_training_ids) for ids in fit_ids))
            for name in MODEL_NAMES:
                one = load_final_model(name, root=root, run_id="unit_first")
                two = load_final_model(name, root=root, run_id="unit_second")
                np.testing.assert_array_equal(one.predict(self.X), two.predict(self.X))
                if name in ("knn", "svm"):
                    self.assertEqual(one.named_steps["scaler"].n_samples_seen_, len(self.X))
                    np.testing.assert_allclose(one.named_steps["scaler"].mean_, self.X.mean())
            for name in ("decision_tree", "knn", "svm"):
                records = json.loads((first.report_directory / f"cv_results_{name}.json").read_text())
                for metric in METRIC_NAMES:
                    self.assertIn(f"split2_test_{metric}", records[0])
                    self.assertNotIn(f"split3_test_{metric}", records[0])
                for fold in range(1, 6):
                    self.assertTrue((first.report_directory / "inner_cv" / f"outer_{fold}_{name}.json").is_file())
            with self.assertRaises(FileExistsError):
                run_training(self.X, self.y, root=root, run_id="unit_first", grids=TINY_GRIDS, progress=None)
            with self.assertRaisesRegex(ValueError, "run_id"):
                run_training(self.X, self.y, root=root, run_id="../escape", progress=None)
            artifact = first.model_directory / "knn.joblib"
            artifact.write_bytes(artifact.read_bytes() + b"modified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_final_model("knn", root=root, run_id="unit_first")


if __name__ == "__main__":
    unittest.main()
