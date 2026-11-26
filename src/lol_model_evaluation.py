"""Part C: frozen-model evaluation and interpretation, with a separate preflight.

Preflight reads source bytes and structural manifests, never held-out feature or
label values. Only run_final_evaluation(), with explicit authorization, accesses
the held-out objects. No function fits, tunes, or serializes an estimator.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from src.lol_model_training import (
    DEFAULT_RUN_ID, METRIC_NAMES, MODEL_NAMES, load_final_model, load_run_results,
)
from src.lol_preprocessing import (
    FEATURE_NAMES, ORIGINAL_SHA256, RANDOM_STATE, REPOSITORY_ROOT, TARGET,
    prepare_data, resolve_data_path, split_manifest,
)


COEFFICIENT_ATOL = 1e-8
SCORE_ATOL = 1e-7
NUMERICAL_RTOL = 1e-8
COEFFICIENT_COLUMNS = [
    "feature", "standardized_coefficient", "absolute_standardized_coefficient",
    "training_mean", "training_scale", "original_unit_coefficient",
]
MODEL_DISPLAY_NAMES = {
    "dummy": "Dummy (baseline)", "decision_tree": "Decision Tree",
    "knn": "KNN", "svm": "SVM",
}
RESEARCH_QUESTION = (
    "Can we predict the winning team in a League of Legends match using only "
    "statistics available after the first 10 minutes?"
)


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Write strict JSON; ROC's infinite initial threshold is handled separately."""
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
                          + "\n", encoding="utf-8")


def output_paths(root=REPOSITORY_ROOT, run_id=DEFAULT_RUN_ID):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid run ID.")
    root = Path(root).resolve()
    return {
        "reports": root / "reports/model_evaluation" / run_id,
        "figures": root / "reports/figures/part_c" / run_id,
        "local": root / "data/processed/part_c" / run_id,
    }


def validate_features(X, y=None):
    """Reject schema drift and row misalignment rather than silently fixing them."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Features must be an unscaled pandas DataFrame.")
    if tuple(X.columns) != FEATURE_NAMES:
        raise ValueError("Expected the original 12 features in their recorded order.")
    if X.empty or X.index.name != "gameId" or not X.index.is_unique:
        raise ValueError("Expected nonempty data with unique gameId row indices.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite.")
    if y is not None:
        if not isinstance(y, pd.Series) or y.name != TARGET:
            raise TypeError("Labels must be a pandas Series named blueWins.")
        if not X.index.equals(y.index):
            raise ValueError("Feature and label indices are not aligned.")
        _binary_vector(y, "Labels", both_classes=True)


def _binary_vector(values, name, both_classes=False):
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must be a nonempty one-dimensional binary vector.")
    if both_classes and set(array.tolist()) != {0, 1}:
        raise ValueError("ROC-AUC requires both real classes; no fallback is assigned.")
    return array.astype(int)


def positive_scores(estimator, X):
    """Orient scores towards Blue=1, including a reversed estimator class order."""
    classes = np.asarray(estimator.classes_)
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError("Expected exactly the binary classes 0 and 1.")
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        if scores.ndim != 1:
            raise ValueError("Expected one binary decision score per observation.")
        if classes[1] != 1:
            scores = -scores
    else:
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
        if probabilities.shape != (len(X), 2):
            raise ValueError("Unexpected probability matrix shape.")
        if (not np.isfinite(probabilities).all() or
                (probabilities < 0).any() or (probabilities > 1).any() or
                not np.allclose(probabilities.sum(axis=1), 1)):
            raise ValueError("Invalid class probabilities.")
        scores = probabilities[:, int(np.flatnonzero(classes == 1)[0])]
    if scores.shape != (len(X),) or not np.isfinite(scores).all():
        raise ValueError("Expected one finite score per observation.")
    return scores


def evaluate_arrays(y_true, y_pred, scores):
    """Compute binary metrics and ROC data from cached predictions; never fit."""
    actual = _binary_vector(y_true, "Labels", both_classes=True)
    predicted = _binary_vector(y_pred, "Predictions")
    scores = np.asarray(scores, dtype=float)
    if predicted.shape != actual.shape or scores.shape != actual.shape:
        raise ValueError("Labels, predictions, and scores must have the same length.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(actual, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(actual, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, scores)),
    }
    # Independent arithmetic checks catch confusion-matrix orientation errors.
    manual = {
        "accuracy": (tn + tp) / len(actual),
        "precision": tp / (tp + fp) if tp + fp else 0.,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
    }
    for name, value in manual.items():
        if not math.isclose(metrics[name], value, abs_tol=1e-12):
            raise ArithmeticError(f"Independent {name} verification failed.")
    fpr, tpr, thresholds = roc_curve(actual, scores, pos_label=1)
    return {
        "metrics": metrics,
        "confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp,
                      "matrix": cm.tolist(), "labels": [0, 1],
                      "rows": "Actual class", "columns": "Predicted class"},
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "thresholds": [float(v) if np.isfinite(v) else None for v in thresholds],
                "threshold_note": "null denotes the initial positive-infinity ROC threshold"},
    }


def evaluate_estimator(estimator, X, y):
    """Use the existing predict rule and continuous scores without changing either."""
    validate_features(X, y)
    predictions = estimator.predict(X)
    scores = positive_scores(estimator, X)
    return evaluate_arrays(y, predictions, scores), np.asarray(predictions), scores


def reconstruct_linear_svm(pipeline):
    """Recover primal weights using accurately summed products, avoiding matmul.

    Binary SVC stores signed dual coefficients. They are already oriented towards
    classes_[1]; the frozen project SVM has classes_=[0, 1]. math.fsum limits
    cancellation when thousands of positive and negative terms are combined.
    """
    if list(pipeline.named_steps) != ["scaler", "classifier"]:
        raise ValueError("Expected the saved StandardScaler/SVC pipeline.")
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    if classifier.kernel != "linear" or list(classifier.classes_) != [0, 1]:
        raise ValueError("Interpretation requires a binary linear SVC with classes [0, 1].")
    if list(scaler.feature_names_in_) != list(FEATURE_NAMES):
        raise ValueError("Scaler feature order differs from Part A.")
    if not scaler.with_mean or not scaler.with_std:
        raise ValueError("Expected the original centering and standardization policy.")
    support = np.asarray(classifier.support_vectors_, dtype=float)
    dual = np.asarray(classifier.dual_coef_, dtype=float)
    if support.ndim != 2 or support.shape[1] != 12 or dual.shape != (1, len(support)):
        raise ValueError("Unexpected support-vector or dual-coefficient dimensions.")
    arrays = [support, dual, scaler.mean_, scaler.scale_, classifier.intercept_]
    if not all(np.isfinite(a).all() for a in arrays) or (scaler.scale_ <= 0).any():
        raise ValueError("Non-finite SVM parameters or invalid scaler scale.")
    weights = np.array([
        math.fsum(float(a) * float(v) for a, v in zip(dual[0], support[:, j]))
        for j in range(12)
    ])
    raw = weights / scaler.scale_
    intercept = float(classifier.intercept_[0])
    raw_intercept = intercept - math.fsum(float(b) * float(mu)
                                         for b, mu in zip(raw, scaler.mean_))
    if not np.isfinite(weights).all() or not np.isfinite(raw).all() or not math.isfinite(raw_intercept):
        raise ArithmeticError("Reconstructed coefficients are not finite.")
    table = pd.DataFrame({
        "feature": FEATURE_NAMES, "standardized_coefficient": weights,
        "absolute_standardized_coefficient": np.abs(weights),
        "training_mean": scaler.mean_, "training_scale": scaler.scale_,
        "original_unit_coefficient": raw,
    })
    return table, intercept, raw_intercept


def _accurate_scores(values, weights, intercept):
    return np.array([math.fsum([intercept, *(float(x) * float(w)
                                           for x, w in zip(row, weights))])
                     for row in values])


def verify_svm_numerics(pipeline):
    """Audit learned parameters using artificial points only, never match records."""
    table, intercept, raw_intercept = reconstruct_linear_svm(pipeline)
    classifier = pipeline.named_steps["classifier"]
    scaler = pipeline.named_steps["scaler"]
    weights = table.standardized_coefficient.to_numpy()
    # These points need not describe feasible matches; they verify an algebraic identity.
    rng = np.random.default_rng(RANDOM_STATE)
    z = np.vstack([np.zeros((1, 12)), np.eye(12), -np.eye(12),
                   rng.uniform(-3, 3, size=(512, 12)), np.full((1, 12), 8),
                   np.full((1, 12), -8)])
    X = pd.DataFrame(z * scaler.scale_ + scaler.mean_, columns=FEATURE_NAMES,
                     index=pd.Index(np.arange(-len(z), 0), name="gameId"))
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        public_weights = np.asarray(classifier.coef_)[0]
    if not np.isfinite(public_weights).all():
        raise ArithmeticError("The public SVC coefficients are not finite.")
    np.testing.assert_allclose(public_weights, weights, rtol=NUMERICAL_RTOL,
                               atol=COEFFICIENT_ATOL)
    with warnings.catch_warnings(record=True) as decision_warnings:
        warnings.simplefilter("always")
        direct = pipeline.decision_function(X)
    if decision_warnings or not np.isfinite(direct).all():
        raise ArithmeticError("Synthetic decision_function produced warnings or non-finite scores.")
    standardized = _accurate_scores(z, weights, intercept)
    original = _accurate_scores(X.to_numpy(), table.original_unit_coefficient, raw_intercept)
    np.testing.assert_allclose(standardized, direct, rtol=NUMERICAL_RTOL, atol=SCORE_ATOL)
    np.testing.assert_allclose(original, direct, rtol=NUMERICAL_RTOL, atol=SCORE_ATOL)
    # Compare separate multiplication paths and retain warnings as audit evidence.
    paths = {}
    for label, operation in {
        "numpy_dot": lambda: np.dot(classifier.dual_coef_, classifier.support_vectors_)[0],
        "numpy_matmul": lambda: (classifier.dual_coef_ @ classifier.support_vectors_)[0],
    }.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            candidate = operation()
        np.testing.assert_allclose(candidate, weights, rtol=NUMERICAL_RTOL, atol=COEFFICIENT_ATOL)
        paths[label] = {"warnings": [str(w.message) for w in caught],
                        "maximum_absolute_error": float(np.max(np.abs(candidate - weights)))}
    return {
        "status": "passed", "scope": "Learned parameters and artificial inputs only",
        "held_out_values_accessed": False, "synthetic_observations": len(X),
        "coefficient_method": "math.fsum of signed dual coefficient times support-vector coordinate",
        "coefficient_atol": COEFFICIENT_ATOL, "score_atol": SCORE_ATOL,
        "rtol": NUMERICAL_RTOL, "coefficients_finite": True,
        "public_coef_warnings": [str(w.message) for w in observed],
        "public_coef_maximum_absolute_error": float(np.max(np.abs(public_weights - weights))),
        "standardized_score_maximum_absolute_error": float(np.max(np.abs(standardized - direct))),
        "original_unit_score_maximum_absolute_error": float(np.max(np.abs(original - direct))),
        "decision_function_warnings": [], "multiplication_paths": paths,
        "diagnosis": (
            "Warnings were checked separately for SVC.coef_, NumPy matmul, and NumPy dot. "
            "The local warning is isolated to the matmul path used by SVC.coef_; "
            "a low-level library or hardware cause has not been established. "
            "Finite weights and agreement with libsvm decision_function are independently verified. "
            "Interpretation uses math.fsum, without altering the saved model or suppressing global warnings."
        ),
        "verified_local_call_path": "SVC._get_coef -> safe_sparse_dot -> a @ b (NumPy matmul)",
        "standardized_intercept": intercept, "original_unit_intercept": raw_intercept,
    }, table


def _check_cv_membership(cv, train_ids, test_ids):
    def check(fold, allowed):
        tr, va = set(fold["train_game_ids"]), set(fold["validation_game_ids"])
        if (len(tr) != len(fold["train_game_ids"]) or
                len(va) != len(fold["validation_game_ids"]) or
                not tr.isdisjoint(va) or tr | va != allowed or
                not (tr | va).isdisjoint(test_ids)):
            raise ValueError("CV membership is inconsistent with the held-out split.")
        return tr
    if cv["configuration"] != {"outer_folds": 5, "inner_folds": 3, "random_state": 42}:
        raise ValueError("Unexpected CV configuration.")
    if len(cv["outer_folds"]) != 5 or len(cv["final_search_folds"]) != 3:
        raise ValueError("Unexpected number of CV folds.")
    seen = []
    for fold in cv["outer_folds"]:
        allowed = check(fold, train_ids)
        if len(fold["inner_folds"]) != 3:
            raise ValueError("Expected three inner folds.")
        inner_seen = []
        for inner in fold["inner_folds"]:
            check(inner, allowed)
            inner_seen.extend(inner["validation_game_ids"])
        if len(inner_seen) != len(allowed) or set(inner_seen) != allowed:
            raise ValueError("Inner validation does not partition outer training.")
        seen.extend(fold["validation_game_ids"])
    if len(seen) != len(train_ids) or set(seen) != train_ids:
        raise ValueError("Outer validation does not partition training.")
    final_seen = []
    for fold in cv["final_search_folds"]:
        check(fold, train_ids)
        final_seen.extend(fold["validation_game_ids"])
    if len(final_seen) != len(train_ids) or set(final_seen) != train_ids:
        raise ValueError("Final search validation does not partition training.")


def run_preflight(*, root=REPOSITORY_ROOT, run_id=DEFAULT_RUN_ID, data_path=None):
    """Read-only checks; source CSV bytes are hashed, not parsed into test records."""
    root = Path(root).resolve()
    output_paths(root, run_id)
    results = load_run_results(root, run_id)
    metadata = results["metadata"]
    selection = json.loads((root / "reports/model_training" / run_id / "selection.json").read_text())
    if (metadata["held_out_test_used"] is not False or metadata["threshold_tuning"] is not False
            or metadata["selected_model"] != "svm" or selection["selected_model"] != "svm"
            or selection["held_out_test_used"] is not False or metadata["positive_class"] != 1
            or metadata["primary_metric"] != "accuracy" or metadata["training_rows"] != 7903
            or metadata["feature_names"] != list(FEATURE_NAMES) or metadata["target"] != TARGET):
        raise ValueError("The frozen Part B protocol does not match this evaluation.")
    source = resolve_data_path(data_path)
    if file_hash(source) != ORIGINAL_SHA256 or metadata["source_sha256"] != ORIGINAL_SHA256:
        raise ValueError("Original Diamond source checksum mismatch.")
    provenance = metadata["provenance"]
    protected = {
        "src/lol_preprocessing.py": provenance["preprocessing_sha256"],
        "src/lol_model_training.py": metadata["training_module_sha256"],
        "data/processed/split_manifest.json": provenance["part_a_split_manifest_sha256"],
        metadata["cv_manifest_path"]: metadata["cv_manifest_sha256"],
    }
    integrity = json.loads((root / "reports/model_training" / run_id / "part_a_integrity.json").read_text())
    protected.update({c["path"]: c["sha256"] for c in integrity["checks"]})
    protected.update({c["path"]: c["sha256"] for c in metadata["model_files"].values()})
    for path, expected in protected.items():
        if file_hash(root / path) != expected:
            raise ValueError(f"Protected artifact changed: {path}")
    manifest = json.loads((root / "data/processed/split_manifest.json").read_text())
    if (manifest["feature_names"] != list(FEATURE_NAMES) or manifest["random_state"] != 42
            or manifest["test_size"] != .20 or manifest["target"] != TARGET
            or manifest["source_sha256"] != ORIGINAL_SHA256
            or manifest["ordering"] != "gameId ascending before sklearn train_test_split"):
        raise ValueError("Part A split contract changed.")
    train_ids, test_ids = set(manifest["train_game_ids"]), set(manifest["test_game_ids"])
    if (len(train_ids) != 7903 or len(test_ids) != 1976 or not train_ids.isdisjoint(test_ids)
            or len(manifest["train_game_ids"]) != 7903 or len(manifest["test_game_ids"]) != 1976):
        raise ValueError("Original split membership is invalid.")
    cv = json.loads((root / metadata["cv_manifest_path"]).read_text())
    _check_cv_membership(cv, train_ids, test_ids)
    artifacts = {}
    for name in MODEL_NAMES:
        model = load_final_model(name, root=root, run_id=run_id)
        if list(model.classes_) != [0, 1] or list(model.feature_names_in_) != list(FEATURE_NAMES):
            raise ValueError(f"Saved model schema mismatch: {name}")
        if any(model.get_params()[k] != v for k, v in results["best_params"][name].items()):
            raise ValueError(f"Saved hyperparameters differ: {name}")
        if name in ("svm", "knn"):
            s = model.named_steps["scaler"]
            if s.n_samples_seen_ != 7903 or list(s.feature_names_in_) != list(FEATURE_NAMES):
                raise ValueError(f"Saved scaler schema mismatch: {name}")
            if (not np.isfinite(s.mean_).all() or not np.isfinite(s.var_).all()
                    or not np.isfinite(s.scale_).all() or (s.scale_ <= 0).any()):
                raise ValueError(f"Invalid saved scaler: {name}")
        elif "scaler" in model.named_steps:
            raise ValueError("Unexpected scaler in Dummy or Decision Tree.")
        artifacts[name] = {"checksum_verified": True, "schema_verified": True,
                           "hyperparameters_verified": True}
    svm = load_final_model("svm", root=root, run_id=run_id)
    numerical, coefficients = verify_svm_numerics(svm)
    if any(file_hash(root / p) != h for p, h in protected.items()):
        raise ValueError("A protected artifact changed during preflight.")
    return {
        "status": "passed", "run_id": run_id, "selected_model": "svm",
        "source_sha256": ORIGINAL_SHA256, "feature_names": list(FEATURE_NAMES),
        "training_rows": 7903, "held_out_rows": 1976,
        "held_out_values_accessed": False, "held_out_predictions_computed": False,
        "source_access": "Byte-level checksum only; CSV not parsed",
        "split_and_cv_membership_verified": True, "artifacts": artifacts,
        "protected_hashes": protected, "svm_numerical_verification": numerical,
        "implementation_hashes": {
            p: file_hash(root / p) for p in (
                "src/lol_model_evaluation.py", "scripts/evaluate_models.py",
                "scripts/execute_evaluation_notebook.py", "tests/test_model_evaluation.py",
            )
        },
    }, coefficients


def save_preflight(preflight, coefficients, *, root=REPOSITORY_ROOT, run_id=DEFAULT_RUN_ID):
    paths = output_paths(root, run_id)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    destination = paths["reports"]
    state = destination / "evaluation_metadata.json"
    if (destination / "evaluation_started.json").exists():
        raise FileExistsError("A final evaluation has already been reserved; inspect its saved state.")
    if state.exists() and json.loads(state.read_text())["status"] != "awaiting_authorization":
        raise FileExistsError("An evaluation has already started; use saved outputs.")
    write_json(destination / "preflight.json", preflight)
    write_json(destination / "svm_coefficients.json", coefficients.to_dict(orient="records"))
    coefficients.to_csv(destination / "svm_coefficients.csv", index=False)
    write_json(state, {"status": "awaiting_authorization", "run_id": run_id,
                       "selected_model": "svm", "held_out_values_accessed": False,
                       "held_out_predictions_computed": False,
                       "test_metrics": "Pending final evaluation"})


def compare_with_cv(metrics, cv_comparison):
    """Descriptive comparisons only; this function never selects a model."""
    records = []
    indexed = cv_comparison.set_index("model")
    baseline = metrics["dummy"]
    for name in MODEL_NAMES:
        for metric in METRIC_NAMES:
            records.append({
                "model": name, "metric": metric, "test_value": metrics[name][metric],
                "outer_cv_mean": float(indexed.loc[name, f"mean_{metric}"]),
                "outer_cv_sample_sd": float(indexed.loc[name, f"std_{metric}"]),
                "test_minus_outer_cv_mean": metrics[name][metric] - float(indexed.loc[name, f"mean_{metric}"]),
                "test_minus_dummy": metrics[name][metric] - baseline[metric],
                "accuracy_gain_percentage_points": 100 * (metrics[name][metric] - baseline[metric])
                if metric == "accuracy" else None,
                "selected_before_test": name == "svm",
            })
    return records


def run_final_evaluation(*, authorize=False, root=REPOSITORY_ROOT,
                         run_id=DEFAULT_RUN_ID, data_path=None):
    """Evaluate once, only after explicit user authorization outside this function.

    A running/failed/completed state prevents automatic inference retries. Cached
    predictions support subsequent plotting without repeating model inference.
    """
    if authorize is not True:
        raise PermissionError("Held-out evaluation requires explicit authorization.")
    root = Path(root).resolve()
    paths = output_paths(root, run_id)
    state_path = paths["reports"] / "evaluation_metadata.json"
    preflight_path = paths["reports"] / "preflight.json"
    if not state_path.exists() or not preflight_path.exists():
        raise ValueError("Run and save the preliminary checks first.")
    state = json.loads(state_path.read_text())
    saved = json.loads(preflight_path.read_text())
    if state["status"] != "awaiting_authorization":
        raise FileExistsError("Evaluation already started; inference must not be repeated automatically.")
    if saved.get("automatic_tests", {}).get("successful") is not True:
        raise ValueError("A successful preliminary test suite is required.")
    current, _ = run_preflight(root=root, run_id=run_id, data_path=data_path)
    if current["protected_hashes"] != saved["protected_hashes"]:
        raise ValueError("Protected artifacts changed since preflight.")
    if current.get("implementation_hashes") != saved.get("implementation_hashes"):
        raise ValueError("Part C code changed since preflight; repeat preliminary checks.")
    if any(paths["local"].iterdir()):
        raise FileExistsError("Local prediction output directory is not empty.")
    # Exclusive creation prevents two simultaneous commands from starting inference.
    reservation = paths["reports"] / "evaluation_started.json"
    with reservation.open("x", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "reserved_at_utc": datetime.now(timezone.utc).isoformat()}, handle)
    state.update({"status": "running", "held_out_values_accessed": False,
                  "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "positive_class": 1, "threshold_tuning": False,
                  "model_reselection": False, "source_sha256": ORIGINAL_SHA256})
    write_json(state_path, state)
    cached = pd.DataFrame()
    metrics, matrices, curves = {}, {}, {}
    try:
        # This is the first point where held-out feature and label values are accessed.
        prepared = prepare_data(data_path)
        state["held_out_values_accessed"] = True
        write_json(state_path, state)
        original = json.loads((root / "data/processed/split_manifest.json").read_text())
        if split_manifest(prepared) != original:
            raise ValueError("Reconstructed split differs from the original manifest.")
        digest = sha256()
        digest.update(json.dumps(list(prepared.X_train.columns)).encode())
        for obj in (prepared.X_train, prepared.y_train):
            digest.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
        results = load_run_results(root, run_id)
        if digest.hexdigest() != results["metadata"]["training_data_sha256"]:
            raise ValueError("Reconstructed training-data fingerprint differs.")
        validate_features(prepared.X_test, prepared.y_test)
        state["model_hashes"] = results["metadata"]["model_files"]
        cached = pd.DataFrame({"actual_blueWins": prepared.y_test})
        for name in MODEL_NAMES:
            model = load_final_model(name, root=root, run_id=run_id)
            # Mark inference as started even if a subsequent scoring/export step fails.
            state["held_out_inference_started"] = True
            write_json(state_path, state)
            evaluation, predictions, scores = evaluate_estimator(model, prepared.X_test, prepared.y_test)
            state["held_out_predictions_computed"] = True
            cached[f"{name}_prediction"] = predictions
            cached[f"{name}_score"] = scores
            cached.to_csv(paths["local"] / "test_predictions.csv", index_label="gameId")
            metrics[name], matrices[name], curves[name] = (
                evaluation["metrics"], evaluation["confusion"], evaluation["roc"])
        directory = paths["reports"]
        comparison = compare_with_cv(metrics, results["comparison"])
        write_json(directory / "test_metrics.json", metrics)
        pd.DataFrame.from_dict(metrics, orient="index").rename_axis("model").to_csv(directory / "test_metrics.csv")
        write_json(directory / "confusion_matrices.json", matrices)
        pd.DataFrame([{"model": n, **{k: c[k] for k in ("TN", "FP", "FN", "TP")}}
                      for n, c in matrices.items()]).to_csv(directory / "confusion_matrices.csv", index=False)
        write_json(directory / "roc_curves.json", curves)
        write_json(directory / "cv_test_comparison.json", comparison)
        pd.DataFrame(comparison).to_csv(directory / "cv_test_comparison.csv", index=False)
        if any(file_hash(root / p) != h for p, h in current["protected_hashes"].items()):
            raise ValueError("Protected artifacts changed during evaluation.")
        write_json(directory / "verification.json", {
            "status": "passed", "rows_per_model": len(cached),
            "confusion_counts_match_metrics": True, "protected_hashes_unchanged": True,
            "selected_model": "svm", "selection_reopened": False,
        })
        state.update({"status": "complete", "held_out_predictions_computed": True,
                      "test_metrics": "Available in test_metrics.json",
                      "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        state.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        write_json(state_path, state)
    return load_saved_evaluation(root=root, run_id=run_id)


def load_saved_evaluation(*, root=REPOSITORY_ROOT, run_id=DEFAULT_RUN_ID):
    """Read aggregate outputs only. Safe for notebook execution before authorization."""
    directory = output_paths(root, run_id)["reports"]
    state = json.loads((directory / "evaluation_metadata.json").read_text())
    result = {"metadata": state,
              "preflight": json.loads((directory / "preflight.json").read_text()),
              "coefficients": pd.DataFrame(json.loads((directory / "svm_coefficients.json").read_text()))[COEFFICIENT_COLUMNS]}
    if state["status"] == "complete":
        for key, filename in {"metrics": "test_metrics", "confusion": "confusion_matrices",
                              "roc": "roc_curves", "comparison": "cv_test_comparison"}.items():
            result[key] = json.loads((directory / f"{filename}.json").read_text())
    return result


def create_figures(saved, *, root=REPOSITORY_ROOT, run_id=DEFAULT_RUN_ID):
    """Plot saved coefficients and, when available, saved aggregate test results."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(root) / ".cache/part_c_matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    def canvas(size):
        figure = Figure(figsize=size)
        FigureCanvasAgg(figure)
        return figure

    directory = output_paths(root, run_id)["figures"]
    directory.mkdir(parents=True, exist_ok=True)
    table = saved["coefficients"].sort_values("standardized_coefficient")
    fig = canvas((9, 6))
    ax = fig.subplots()
    ax.barh(table.feature, table.standardized_coefficient,
            color=["#b44b4b" if w < 0 else "#286a9b" for w in table.standardized_coefficient])
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(xlabel="Change in Blue decision score per training standard deviation",
           title="Frozen linear SVM: standardized coefficients")
    fig.tight_layout()
    fig.savefig(directory / "svm_standardized_coefficients.png", dpi=160)
    fig.clear()
    if saved["metadata"]["status"] != "complete":
        return
    fig = canvas((9, 8))
    axes = fig.subplots(2, 2)
    for ax, name in zip(axes.ravel(), MODEL_NAMES):
        cm = np.array(saved["confusion"][name]["matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1976)
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f"{(('TN', 'FP'), ('FN', 'TP'))[i][j]}\n{value}",
                    ha="center", va="center", color="white" if value > 988 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Red (0)", "Blue (1)"],
               yticklabels=["Red (0)", "Blue (1)"], xlabel="Predicted class",
               ylabel="Actual class", title=MODEL_DISPLAY_NAMES[name])
    fig.suptitle("Held-out confusion matrices: Blue is the positive class")
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrices.png", dpi=160)
    fig.clear()
    fig = canvas((7, 6))
    ax = fig.subplots()
    for name in MODEL_NAMES:
        curve = saved["roc"][name]
        ax.plot(curve["fpr"], curve["tpr"], label=f"{MODEL_DISPLAY_NAMES[name]}: AUC={saved['metrics'][name]['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier (AUC = 0.5)")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (Blue recall)",
           title="Held-out ROC curves", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(directory / "roc_curves.png", dpi=160)
    fig.clear()
    comparison = pd.DataFrame(saved["comparison"])
    accuracy = comparison.loc[comparison.metric == "accuracy"].set_index("model").loc[list(MODEL_NAMES)]
    fig = canvas((8, 5))
    ax = fig.subplots()
    positions = np.arange(4)
    ax.errorbar(positions - .08, accuracy.outer_cv_mean, yerr=accuracy.outer_cv_sample_sd,
                fmt="o", capsize=4, label="Outer CV mean ± sample SD (not a CI)")
    ax.scatter(positions + .08, accuracy.test_value, marker="s", label="Held-out test accuracy")
    ax.set(xticks=positions, xticklabels=[MODEL_DISPLAY_NAMES[n] for n in MODEL_NAMES], ylabel="Accuracy", ylim=(0, 1),
           title="Descriptive comparison: nested CV and held-out test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "cv_test_accuracy.png", dpi=160)
    fig.clear()


def render_findings(saved, *, root=REPOSITORY_ROOT, run_id=DEFAULT_RUN_ID):
    """Regenerate Part C findings from verified saved evidence, without inference."""
    numerical = saved["preflight"]["svm_numerical_verification"]
    lines = ["# Part C — Model Evaluation and Interpretation", "",
             f"**Research Question:** {RESEARCH_QUESTION}", "",
             f"Training run: `{run_id}`. Dataset: original Diamond only (9,879 matches).",
             "Positive class: `blueWins=1` (Blue wins). The SVM was selected before test evaluation.", "",
             "## Verification before held-out evaluation", "",
             "Source, preprocessing, split, CV membership, environment, model checksums, feature order, "
             "and saved hyperparameters passed verification. Preflight hashes the original CSV bytes "
             "but does not parse held-out feature or label values.", "",
             f"SVM algebra was checked on {numerical['synthetic_observations']} artificial points. "
             f"Maximum absolute score error against the saved pipeline's decision_function: "
             f"{numerical['standardized_score_maximum_absolute_error']:.3e} in standardized coordinates "
             f"and {numerical['original_unit_score_maximum_absolute_error']:.3e} in original units. "
             f"The predeclared tolerance is atol={SCORE_ATOL:g}, rtol={NUMERICAL_RTOL:g}.", "",
             "Observed public coefficient warnings: " + (
                 "; ".join(numerical["public_coef_warnings"]) or "None in this execution.") + ".", "",
             numerical["diagnosis"], "", "## Held-out evaluation", ""]
    if saved["metadata"]["status"] == "awaiting_authorization":
        lines += ["**Pending final evaluation.** No test accuracy, precision, recall, F1, ROC-AUC, "
                  "confusion matrix, ROC curve, or test-based conclusion has been calculated.", "",
                  "The 1,976 held-out matches have not been used for Part C predictions or scoring. "
                  "Final execution requires separate authorization.", ""]
    elif saved["metadata"]["status"] != "complete":
        lines += ["**Evaluation incomplete.** An evaluation was started but has not completed. "
                  "Inspect evaluation_metadata.json and any local partial predictions before taking "
                  "further action. No final performance claim is available.", ""]
    else:
        lines += ["The frozen pipelines were evaluated on the original 1,976 held-out matches. "
                  "No model, feature, threshold, or hyperparameter was changed.", "",
                  "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name in MODEL_NAMES:
            lines.append("| " + MODEL_DISPLAY_NAMES[name] + " | " + " | ".join(f"{saved['metrics'][name][m]:.4f}" for m in METRIC_NAMES) + " |")
        lines += ["", "| Model | TN | FP | FN | TP |", "|---|---:|---:|---:|---:|"]
        for name in MODEL_NAMES:
            lines.append("| " + MODEL_DISPLAY_NAMES[name] + " | " + " | ".join(str(saved['confusion'][name][k]) for k in ("TN", "FP", "FN", "TP")) + " |")
        svm, dummy = saved["metrics"]["svm"], saved["metrics"]["dummy"]
        gap = 100 * (svm["accuracy"] - dummy["accuracy"])
        lines += ["", f"The preselected SVM achieves {svm['accuracy']:.2%} accuracy versus "
                  f"{dummy['accuracy']:.2%} for Dummy, a difference of {gap:.2f} percentage points. "
                  f"Its ROC-AUC is {svm['roc_auc']:.4f}. "
                  "Precision is the proportion of predicted Blue wins that are correct; recall is the "
                  "proportion of actual Blue wins recovered. ROC-AUC describes ranking, not calibration.", "",
                  "The baseline predicts Red for every match, correctly classifying 990 Red wins and "
                  "missing all 986 Blue wins. Its zero Blue precision, recall, and F1 follow the stated "
                  "zero-division convention. The learned models all improve on this reference.", "",
                  f"For the SVM, {saved['confusion']['svm']['TN'] + saved['confusion']['svm']['TP']:,} "
                  f"of 1,976 predictions are correct. It misses {saved['confusion']['svm']['FN']} Blue wins "
                  f"and incorrectly predicts Blue in {saved['confusion']['svm']['FP']} Red wins. "
                  "These errors illustrate the remaining uncertainty despite above-baseline performance.", "",
                  "![Confusion matrices](figures/part_c/" + run_id + "/confusion_matrices.png)", "",
                  "![ROC curves](figures/part_c/" + run_id + "/roc_curves.png)", "",
                  "## Descriptive comparison with nested CV", "",
                  "| Model | Outer CV accuracy ± sample SD | Test accuracy | Test minus CV mean |",
                  "|---|---:|---:|---:|"]
        for row in saved["comparison"]:
            if row["metric"] == "accuracy":
                lines.append(f"| {MODEL_DISPLAY_NAMES[row['model']]} | {row['outer_cv_mean']:.4f} ± {row['outer_cv_sample_sd']:.4f} | "
                             f"{row['test_value']:.4f} | {row['test_minus_outer_cv_mean']:+.4f} |")
        lines += ["", "All five metrics are compared in cv_test_comparison.json. Outer-fold SD describes "
                  "variation across five folds; it is not a confidence interval. Differences between CV "
                  "and test estimates are descriptive and do not establish statistical significance.", "",
                  "![Nested CV and test accuracy](figures/part_c/" + run_id + "/cv_test_accuracy.png)", "",
                  "## Answer to the Research Question", ""]
        if svm["accuracy"] > dummy["accuracy"] and svm["roc_auc"] > .5:
            lines += [f"Yes, within this Diamond sample, statistics available after the first ten minutes "
                      f"support above-baseline prediction of the winning team. The SVM selected before "
                      f"test evaluation achieves {svm['accuracy']:.2%} accuracy and {svm['roc_auc']:.4f} "
                      f"ROC-AUC on 1,976 held-out matches, exceeding the majority-class baseline "
                      f"accuracy by {gap:.2f} percentage points. Prediction remains imperfect; these "
                      "results do not establish causal effects or reliable transfer to future patches "
                      "or other player populations.", ""]
        else:
            lines += ["The preselected SVM does not show a consistent improvement over the baseline "
                      "across accuracy and ranking on this held-out sample. The evidence is insufficient "
                      "to support an above-baseline predictive conclusion under the frozen protocol.", ""]
    lines += ["## Interpretation of the frozen linear SVM", "",
              "The model is f(x) = b + sum_j w_j (x_j - training_mean_j) / training_scale_j. "
              "Positive coefficients increase the Blue decision score, holding other features fixed. "
              "Absolute standardized coefficients describe sensitivity per training standard deviation. "
              "Original-unit coefficients are w_j / training_scale_j; the intercept is adjusted for "
              "centering. These quantities are neither probabilities nor causal effects.", "",
              "The three largest absolute standardized coefficients in the saved model are " +
              ", ".join(f"`{r.feature}` ({r.standardized_coefficient:+.4f})" for r in
                        saved["coefficients"].sort_values("absolute_standardized_coefficient", ascending=False).head(3).itertuples()) +
              ". This describes the fitted decision function rather than test performance.", "",
              "Negative coefficients for some combat, farming, or tower differences must be read "
              "conditionally on the other retained predictors. They do not show that obtaining "
              "kills, minions, or towers harms a team's chances of winning. Strong correlations "
              "make isolated coefficient interpretations particularly fragile.", "",
              "| Feature | Standardized coefficient |", "|---|---:|",
              *[f"| {r.feature} | {r.standardized_coefficient:+.6f} |" for r in
                saved["coefficients"].sort_values("absolute_standardized_coefficient", ascending=False).itertuples()], "",
              "![Standardized SVM coefficients](figures/part_c/" + run_id + "/svm_standardized_coefficients.png)", "",
              "## Methodological limitations", "",
              "- The random split assesses this Diamond sample; player overlap cannot be excluded because player IDs are unavailable.",
              "- Patch, collection date, and region are unknown. No temporal, regional, or cross-rank validation is established.",
              "- Correlated gold, experience, kills, and assists can redistribute coefficient magnitudes and signs.",
              "- Suspicious ward counts were retained. Relative differences omit absolute game tempo and some tied objective states.",
              "- Model-family selection uses CV and can introduce selection optimism. Five-fold SD is not a confidence interval.",
              "- C=10 and k=81 are at search-grid boundaries; the test cannot justify expanding those grids.",
              "- The SVM is uncalibrated. Its margins are not individual win probabilities.",
              "- No test result may reopen model selection, tuning, feature engineering, or threshold optimization.", ""]
    if saved["metadata"]["status"] == "complete":
        lines += ["## Reproduction and evidence", "",
                  "The final workflow completed successfully and must not be rerun for inference. "
                  "Full-precision metrics, confusion counts, ROC coordinates, and CV comparisons are "
                  f"saved in `model_evaluation/{run_id}/`. Tables round metrics to four decimal places; "
                  "the source JSON and CSV retain full precision. Individual predictions remain in the "
                  "local ignored data directory. Rendering and notebook execution reuse these outputs.", ""]
    destination = Path(root) / "reports/part_c_findings.md"
    destination.write_text("\n".join(lines), encoding="utf-8")


def render_saved_outputs(*, root=REPOSITORY_ROOT, run_id=DEFAULT_RUN_ID):
    saved = load_saved_evaluation(root=root, run_id=run_id)
    create_figures(saved, root=root, run_id=run_id)
    render_findings(saved, root=root, run_id=run_id)
    return saved
