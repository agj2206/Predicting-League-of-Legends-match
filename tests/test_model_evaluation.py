"""Part C checks: artificial observations only; frozen models are never refitted.

The CSV is never parsed by this suite. Preflight may hash its bytes. Integration
checks use artificial matrices scaled with the stored training parameters.
"""

import ast
import copy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.lol_model_evaluation import (
    compare_with_cv, evaluate_arrays, evaluate_estimator, file_hash,
    load_saved_evaluation, output_paths, positive_scores, reconstruct_linear_svm,
    render_findings, run_final_evaluation, run_preflight, save_preflight,
    validate_features, verify_svm_numerics, render_saved_outputs,
)
from src.lol_model_training import MODEL_NAMES, load_final_model, load_run_results
from src.lol_preprocessing import FEATURE_NAMES, REPOSITORY_ROOT, PreparedData, split_manifest


def artificial_frame(n=10):
    X = pd.DataFrame(np.random.default_rng(42).normal(size=(n, 12)), columns=FEATURE_NAMES,
                     index=pd.Index(np.arange(-n, 0), name="gameId"))
    y = pd.Series(([0, 1] * (n // 2)) + ([0] if n % 2 else []), index=X.index, name="blueWins")
    return X, y


class MetricTests(unittest.TestCase):
    def test_hand_calculated_metrics_and_orientation(self):
        result = evaluate_arrays([0] * 4 + [1] * 6, [0, 0, 0, 1, 0, 0, 1, 1, 1, 1], np.arange(10))
        self.assertEqual(result["confusion"]["matrix"], [[3, 1], [2, 4]])
        self.assertEqual([result["confusion"][k] for k in ("TN", "FP", "FN", "TP")], [3, 1, 2, 4])
        for key, value in {"accuracy": .7, "precision": .8, "recall": 4 / 6, "f1": 8 / 11}.items():
            self.assertAlmostEqual(result["metrics"][key], value)

    def test_known_auc(self):
        result = evaluate_arrays([0, 0, 1, 1], [0, 0, 0, 1], [.1, .4, .35, .8])
        self.assertEqual(result["metrics"]["roc_auc"], .75)
        self.assertEqual(result["roc"]["thresholds"][0], None)
        json.dumps(result, allow_nan=False)

    def test_perfect_predictions(self):
        metrics = evaluate_arrays([0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1])["metrics"]
        self.assertTrue(all(v == 1 for v in metrics.values()))

    def test_no_positive_predictions_and_constant_auc(self):
        metrics = evaluate_arrays([0, 1, 0, 1], [0] * 4, [.2] * 4)["metrics"]
        self.assertEqual(metrics, {"accuracy": .5, "precision": 0., "recall": 0., "f1": 0., "roc_auc": .5})

    def test_no_negative_predictions(self):
        metrics = evaluate_arrays([0, 1, 0, 1], [1] * 4, [.5] * 4)["metrics"]
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["precision"], .5)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_reversed_and_tied_rankings(self):
        y = [0, 0, 1, 1]
        self.assertEqual(evaluate_arrays(y, y, [.9, .8, .2, .1])["metrics"]["roc_auc"], 0.)
        self.assertEqual(evaluate_arrays(y, y, [.2, .2, .2, .2])["metrics"]["roc_auc"], .5)
        self.assertEqual(evaluate_arrays(y, y, [.2, .5, .5, .8])["metrics"]["roc_auc"], .875)

    def test_degenerate_labels_rejected(self):
        for labels in ([], [0, 0], [1, 1], [0, 2], [0, np.nan]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_arrays(labels, labels, np.zeros(len(labels)))

    def test_invalid_dimensions_lengths_and_nonfinite_scores(self):
        for predicted, scores in [([0], [0, 1]), ([0, 2], [0, 1]), ([0, 1], [0, np.nan]),
                                  ([0, 1], [0, np.inf]), ([[0], [1]], [0, 1]), ([0, 1], [[0], [1]])]:
            with self.subTest(predicted=predicted, scores=scores), self.assertRaises(ValueError):
                evaluate_arrays([0, 1], predicted, scores)

    def test_auc_invariant_to_strictly_increasing_score_transform(self):
        y, score = [0, 0, 1, 1], np.array([.1, .4, .35, .8])
        first = evaluate_arrays(y, y, score)["metrics"]["roc_auc"]
        self.assertEqual(first, evaluate_arrays(y, y, 3 * score + 8)["metrics"]["roc_auc"])


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = artificial_frame()

    def test_expected_feature_contract(self):
        validate_features(self.X, self.y)

    def test_feature_order_missing_target_and_id_rejected(self):
        for X in (self.X.iloc[:, ::-1], self.X.iloc[:, :-1], self.X.assign(blueWins=self.y),
                  self.X.reset_index()):
            with self.subTest(columns=list(X)), self.assertRaises(ValueError):
                validate_features(X, self.y)

    def test_alignment_index_and_label_name(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            validate_features(self.X, self.y.iloc[::-1])
        duplicate = self.X.copy()
        duplicate.index = pd.Index([-1] * len(duplicate), name="gameId")
        with self.assertRaises(ValueError):
            validate_features(duplicate)
        with self.assertRaises(TypeError):
            validate_features(self.X, self.y.rename("other"))
        with self.assertRaises(TypeError):
            validate_features(self.X.to_numpy())

    def test_nonfinite_features_and_empty_data(self):
        for value in (np.nan, np.inf):
            X = self.X.copy()
            X.iloc[0, 0] = value
            with self.assertRaises(ValueError):
                validate_features(X)
        with self.assertRaises(ValueError):
            validate_features(self.X.iloc[:0])

    def test_positive_probability_column_is_identified_by_class(self):
        class Reversed:
            classes_ = np.array([1, 0])
            def predict_proba(self, X):
                return np.tile([.8, .2], (len(X), 1))
        np.testing.assert_array_equal(positive_scores(Reversed(), self.X), [.8] * len(self.X))

    def test_reversed_binary_margin_is_oriented_to_blue(self):
        class Reversed:
            classes_ = np.array([1, 0])
            def decision_function(self, X):
                return np.arange(len(X), dtype=float)
        np.testing.assert_array_equal(positive_scores(Reversed(), self.X), -np.arange(len(self.X)))

    def test_invalid_classes_scores_and_probabilities_rejected(self):
        class Invalid:
            classes_ = np.array([0, 1])
            def predict_proba(self, X):
                return np.tile([.8, .8], (len(X), 1))
        with self.assertRaises(ValueError):
            positive_scores(Invalid(), self.X)
        Invalid.classes_ = np.array([0, 2])
        with self.assertRaises(ValueError):
            positive_scores(Invalid(), self.X)
        class BadMargin:
            classes_ = np.array([0, 1])
            def decision_function(self, X):
                return np.full(len(X), np.inf)
        with self.assertRaises(ValueError):
            positive_scores(BadMargin(), self.X)

    def test_existing_predict_rule_is_retained(self):
        class Fixed:
            classes_ = np.array([0, 1])
            def predict(self, X):
                return np.zeros(len(X), dtype=int)
            def decision_function(self, X):
                return np.ones(len(X))
            def fit(self, *args, **kwargs):
                raise AssertionError("Fitting is forbidden")
        result, predictions, _ = evaluate_estimator(Fixed(), self.X, self.y)
        self.assertTrue((predictions == 0).all())
        self.assertEqual(result["metrics"]["recall"], 0)

    def test_unsafe_run_ids_rejected(self):
        for run_id in ("../bad", "a/b", "", "a" * 81):
            with self.assertRaises(ValueError):
                output_paths(run_id=run_id)


class FrozenArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {n: load_final_model(n) for n in MODEL_NAMES}

    def test_saved_models_accept_synthetic_schema_without_fitting_or_mutation(self):
        X, y = artificial_frame(8)
        scaler = self.models["svm"].named_steps["scaler"]
        X.loc[:, :] = X.to_numpy() * scaler.scale_ + scaler.mean_
        classes = (Pipeline, StandardScaler, SVC, KNeighborsClassifier, DecisionTreeClassifier, DummyClassifier)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for klass in classes:
                stack.enter_context(patch.object(klass, "fit", side_effect=AssertionError("No fitting permitted")))
            stack.enter_context(patch.object(StandardScaler, "fit_transform", side_effect=AssertionError("No fitting permitted")))
            for name, model in self.models.items():
                # KNN's tree queries update diagnostic counters. Check learned
                # arrays and tree structure, excluding those runtime counters.
                def learned_state():
                    classifier = model.named_steps["classifier"]
                    attributes = {k: v for k, v in vars(classifier).items() if k != "_tree"}
                    tree_arrays = classifier._tree.get_arrays() if hasattr(classifier, "_tree") else None
                    scaler_state = vars(model.named_steps["scaler"]) if "scaler" in model.named_steps else None
                    return joblib.hash((attributes, tree_arrays, scaler_state))
                before = learned_state()
                X_before, y_before = X.copy(deep=True), y.copy(deep=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    result, _, _ = evaluate_estimator(model, X, y)
                self.assertEqual(learned_state(), before, name)
                self.assertEqual(sum(result["confusion"][k] for k in ("TN", "FP", "FN", "TP")), 8)
                pd.testing.assert_frame_equal(X, X_before)
                pd.testing.assert_series_equal(y, y_before)

    def test_saved_parameters_and_feature_order(self):
        results = load_run_results()
        for name, model in self.models.items():
            self.assertEqual(list(model.feature_names_in_), list(FEATURE_NAMES))
            for key, value in results["best_params"][name].items():
                self.assertEqual(model.get_params()[key], value)
            artifact = REPOSITORY_ROOT / results["metadata"]["model_files"][name]["path"]
            self.assertEqual(file_hash(artifact), results["metadata"]["model_files"][name]["sha256"])

    def test_real_svm_algebra_on_synthetic_points(self):
        report, coefficients = verify_svm_numerics(self.models["svm"])
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["held_out_values_accessed"])
        self.assertEqual(report["synthetic_observations"], 539)
        self.assertLess(report["standardized_score_maximum_absolute_error"], 1e-7)
        self.assertLess(report["original_unit_score_maximum_absolute_error"], 1e-7)
        self.assertEqual(list(coefficients.feature), list(FEATURE_NAMES))
        self.assertIn("public_coef_warnings", report)

    def test_synthetic_parameters_have_hand_calculated_primal_weights(self):
        model = copy.deepcopy(self.models["svm"])
        s, c = model.named_steps["scaler"], model.named_steps["classifier"]
        c.support_vectors_ = np.vstack([np.arange(12), np.ones(12)])
        c.dual_coef_ = np.array([[-2., 3.]])
        c.intercept_ = np.array([.25])
        s.mean_ = np.arange(12, dtype=float)
        s.scale_ = np.arange(1, 13, dtype=float)
        table, intercept, raw_intercept = reconstruct_linear_svm(model)
        expected = -2 * np.arange(12) + 3
        np.testing.assert_array_equal(table.standardized_coefficient, expected)
        np.testing.assert_allclose(table.original_unit_coefficient, expected / s.scale_)
        self.assertEqual(intercept, .25)
        self.assertAlmostEqual(raw_intercept, .25 - sum(expected / s.scale_ * s.mean_))

    def test_interpretation_rejects_wrong_kernel_schema_or_scale(self):
        for change in ("kernel", "schema", "scale"):
            model = copy.deepcopy(self.models["svm"])
            if change == "kernel":
                model.named_steps["classifier"].kernel = "rbf"
            elif change == "schema":
                model.named_steps["scaler"].feature_names_in_ = np.array(list(FEATURE_NAMES)[::-1])
            else:
                model.named_steps["scaler"].scale_[0] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                reconstruct_linear_svm(model)

    def test_preflight_does_not_parse_csv_or_prepare_held_out_objects(self):
        with patch("src.lol_model_evaluation.prepare_data", side_effect=AssertionError("No dataset parsing")), \
                patch("pandas.read_csv", side_effect=AssertionError("No dataset parsing")):
            report, _ = run_preflight()
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["held_out_values_accessed"])


class WorkflowTests(unittest.TestCase):
    def _artificial_workflow(self, root):
        """Create a small, entirely synthetic stand-in for final orchestration."""
        X_train, y_train = artificial_frame(8)
        X_train.index = pd.Index(range(1, 9), name="gameId")
        y_train.index = X_train.index
        X_test, y_test = artificial_frame(10)
        prepared = PreparedData(X_train, X_test, y_train, y_test, root / "artificial.csv", "synthetic-only")
        (root / "data/processed").mkdir(parents=True)
        (root / "data/processed/split_manifest.json").write_text(json.dumps(split_manifest(prepared)))
        protected = root / "frozen.txt"
        protected.write_text("Synthetic frozen artifact")
        numerical, coefficients = verify_svm_numerics(load_final_model("svm"))
        preflight = {"status": "passed", "automatic_tests": {"successful": True},
                     "protected_hashes": {"frozen.txt": file_hash(protected)},
                     "svm_numerical_verification": numerical}
        save_preflight(preflight, coefficients, root=root)
        digest = sha256()
        digest.update(json.dumps(list(X_train.columns)).encode())
        for obj in (X_train, y_train):
            digest.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
        results = {"comparison": load_run_results()["comparison"],
                   "metadata": {"training_data_sha256": digest.hexdigest(), "model_files": {}}}
        return prepared, preflight, coefficients, results

    def test_complete_workflow_on_artificial_inputs_and_saved_output_rendering(self):
        from contextlib import ExitStack
        class ArtificialEstimator:
            classes_ = np.array([0, 1])
            def __init__(self):
                self.predict_calls = 0
                self.score_calls = 0
            def predict(self, X):
                self.predict_calls += 1
                return (X.iloc[:, 0].to_numpy() > 0).astype(int)
            def decision_function(self, X):
                self.score_calls += 1
                return X.iloc[:, 0].to_numpy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, preflight, coefficients, results = self._artificial_workflow(root)
            models = {n: ArtificialEstimator() for n in MODEL_NAMES}
            with ExitStack() as stack:
                stack.enter_context(patch("src.lol_model_evaluation.prepare_data", return_value=prepared))
                stack.enter_context(patch("src.lol_model_evaluation.run_preflight", return_value=(preflight, coefficients)))
                stack.enter_context(patch("src.lol_model_evaluation.load_run_results", return_value=results))
                stack.enter_context(patch("src.lol_model_evaluation.load_final_model", side_effect=lambda n, **kwargs: models[n]))
                saved = run_final_evaluation(authorize=True, root=root)
                self.assertEqual(saved["metadata"]["status"], "complete")
                for model in models.values():
                    self.assertEqual(model.predict_calls, 1)
                    self.assertEqual(model.score_calls, 1)
                cache = output_paths(root)["local"] / "test_predictions.csv"
                cached = pd.read_csv(cache)
                self.assertEqual(len(cached), 10)
                for name in MODEL_NAMES:
                    recomputed = evaluate_arrays(cached.actual_blueWins, cached[f"{name}_prediction"], cached[f"{name}_score"])
                    self.assertEqual(recomputed["confusion"], saved["confusion"][name])
                    for key, value in saved["metrics"][name].items():
                        self.assertAlmostEqual(value, recomputed["metrics"][key])
                before = file_hash(cache)
                with self.assertRaises(FileExistsError):
                    run_final_evaluation(authorize=True, root=root)
                render_saved_outputs(root=root)
                self.assertEqual(file_hash(cache), before)
                self.assertEqual(len(list(output_paths(root)["figures"].glob("*.png"))), 4)
                for model in models.values():
                    self.assertEqual(model.predict_calls, 1)

    def test_failed_workflow_preserves_state_and_prevents_retry(self):
        from contextlib import ExitStack
        class FailingEstimator:
            def predict(self, X):
                raise RuntimeError("Synthetic intentional failure")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, preflight, coefficients, results = self._artificial_workflow(root)
            with ExitStack() as stack:
                stack.enter_context(patch("src.lol_model_evaluation.prepare_data", return_value=prepared))
                stack.enter_context(patch("src.lol_model_evaluation.run_preflight", return_value=(preflight, coefficients)))
                stack.enter_context(patch("src.lol_model_evaluation.load_run_results", return_value=results))
                stack.enter_context(patch("src.lol_model_evaluation.load_final_model", return_value=FailingEstimator()))
                with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                    run_final_evaluation(authorize=True, root=root)
                self.assertEqual(load_saved_evaluation(root=root)["metadata"]["status"], "failed")
                with self.assertRaises(FileExistsError):
                    run_final_evaluation(authorize=True, root=root)

    def test_final_evaluation_requires_explicit_authorization_before_any_access(self):
        with patch("src.lol_model_evaluation.prepare_data", side_effect=AssertionError("Forbidden access")):
            with self.assertRaises(PermissionError):
                run_final_evaluation()

    def test_no_results_are_invented_and_export_round_trip(self):
        model = load_final_model("svm")
        numerical, table = verify_svm_numerics(model)
        preflight = {"status": "passed", "svm_numerical_verification": numerical}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_preflight(preflight, table, root=root)
            saved = load_saved_evaluation(root=root)
            self.assertEqual(saved["metadata"]["status"], "awaiting_authorization")
            self.assertNotIn("metrics", saved)
            self.assertFalse((output_paths(root)["reports"] / "test_metrics.json").exists())
            pd.testing.assert_frame_equal(saved["coefficients"], table)
            render_findings(saved, root=root)
            self.assertIn("Pending final evaluation", (root / "reports/part_c_findings.md").read_text())
            with self.assertRaisesRegex(ValueError, "test suite"):
                run_final_evaluation(authorize=True, root=root)

    def test_existing_evaluation_cannot_be_overwritten(self):
        model = load_final_model("svm")
        numerical, table = verify_svm_numerics(model)
        preflight = {"status": "passed", "svm_numerical_verification": numerical}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_preflight(preflight, table, root=root)
            state = output_paths(root)["reports"] / "evaluation_metadata.json"
            for status in ("running", "failed", "complete"):
                state.write_text(json.dumps({"status": status}))
                with self.subTest(status=status), self.assertRaises(FileExistsError):
                    save_preflight(preflight, table, root=root)
                with self.subTest(status=status), self.assertRaises(FileExistsError):
                    run_final_evaluation(authorize=True, root=root)

    def test_cv_comparison_does_not_reselect_model(self):
        cv = load_run_results()["comparison"]
        metrics = {n: {m: .7 for m in ("accuracy", "precision", "recall", "f1", "roc_auc")}
                   for n in MODEL_NAMES}
        metrics["knn"]["accuracy"] = .99
        comparison = compare_with_cv(metrics, cv)
        self.assertTrue(all(r["selected_before_test"] == (r["model"] == "svm") for r in comparison))
        self.assertEqual(len(comparison), 20)

    def test_notebook_has_no_dataset_loading_or_model_inference_calls(self):
        path = REPOSITORY_ROOT / "notebooks/03_model_evaluation.ipynb"
        notebook = json.loads(path.read_text())
        forbidden = {"predict", "predict_proba", "decision_function", "fit", "fit_transform",
                     "prepare_data", "read_csv", "run_final_evaluation", "evaluate_estimator"}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                tree = ast.parse("".join(cell["source"]))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                        self.assertNotIn(name, forbidden)


if __name__ == "__main__":
    unittest.main()
