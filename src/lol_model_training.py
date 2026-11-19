"""Part B: training-only nested CV, final refits, and portable local artifacts.

The public experiment function accepts X_train and y_train only. The unchanged
Part A loader creates the held-out split, but no test values enter this module's
fitting, scoring, hyperparameter tuning, or model selection operations.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time
import warnings

import joblib
from joblib import parallel_config
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.lol_preprocessing import (
    FEATURE_NAMES, ORIGINAL_SHA256, RANDOM_STATE, REPOSITORY_ROOT, TARGET,
    prepare_data, save_split_manifest,
)


DEFAULT_RUN_ID = "diamond_5x3_seed42"
MODEL_NAMES = ("dummy", "decision_tree", "knn", "svm")
LEARNED_MODELS = MODEL_NAMES[1:]
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")
PACKAGE_NAMES = (
    "numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl",
)
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "roc_auc": "roc_auc",
}


@dataclass(frozen=True)
class CVConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    random_state: int = RANDOM_STATE

    def __post_init__(self):
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("Cross-validation needs at least two folds.")
        if self.random_state != RANDOM_STATE:
            raise ValueError("This project fixes random_state=42.")


@dataclass
class TrainingRun:
    run_id: str
    report_directory: Path
    model_directory: Path
    comparison: pd.DataFrame
    selected_model: str


def _json_value(value):
    """Convert NumPy values and masked/undefined CSV cells to strict JSON."""
    if value is np.ma.masked:
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n", encoding="utf-8",
    )


def _file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _data_hash(X_train, y_train):
    digest = sha256()
    digest.update(json.dumps(list(X_train.columns)).encode())
    for obj in (X_train, y_train):
        digest.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def environment_versions():
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def load_training_data(path=None):
    """Reuse Part A; only return its training objects and provenance.

    save_split_manifest verifies the existing Part A manifest without rewriting
    it, or creates it on a fresh machine. Its test IDs are structural metadata;
    X_test values and y_test are never used here.
    """
    prepared = prepare_data(path)
    manifest_path = save_split_manifest(prepared)
    provenance = {
        "source_sha256": prepared.source_sha256,
        "part_a_split_manifest_sha256": _file_hash(manifest_path),
        "preprocessing_sha256": _file_hash(
            REPOSITORY_ROOT / "src/lol_preprocessing.py"
        ),
    }
    return prepared.X_train, prepared.y_train, provenance


def validate_training_input(X_train, y_train):
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise TypeError("Use the unscaled pandas objects returned by Part A.")
    if tuple(X_train.columns) != FEATURE_NAMES:
        raise ValueError("Feature names/order differ from the fixed Part A contract.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("Training features and labels are not aligned.")
    if X_train.index.name != "gameId" or not X_train.index.is_unique:
        raise ValueError("Keep unique gameId values as the row index.")
    if y_train.name != TARGET or set(y_train.unique()) != {0, 1}:
        raise ValueError("Expected blueWins with both classes 0 and 1.")
    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Training features must be finite; do not silently impute.")


def build_models():
    """Fresh unfitted Pipelines; no scaling for Dummy or Decision Tree."""
    return {
        "dummy": Pipeline([
            ("classifier", DummyClassifier(
                strategy="most_frequent", random_state=RANDOM_STATE,
            )),
        ]),
        "decision_tree": Pipeline([
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(metric="minkowski", n_jobs=1)),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                probability=False, random_state=RANDOM_STATE, cache_size=256,
            )),
        ]),
    }


def parameter_grids():
    """The approved, fixed 84-candidate search; Dummy has no tuning."""
    return {
        "decision_tree": {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [3, 5, 8, 12, None],
            "classifier__min_samples_leaf": [1, 5, 10, 20],
        },
        "knn": {
            "classifier__n_neighbors": [3, 5, 11, 21, 41, 81],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
        "svm": [
            {
                "classifier__kernel": ["linear"],
                "classifier__C": [0.01, 0.1, 1, 10],
            },
            {
                "classifier__kernel": ["rbf"],
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__gamma": ["scale", 0.01, 0.1, 1],
            },
        ],
    }


def _splitter(n_splits, config):
    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.random_state,
    )


def build_cv_splits(X_train, y_train, config=CVConfig()):
    """Materialize shared folds; inner indices are local to outer training rows."""
    validate_training_input(X_train, y_train)
    if y_train.value_counts().min() < config.outer_folds:
        raise ValueError("Too few records per class for Outer Stratified CV.")
    outer = list(_splitter(config.outer_folds, config).split(X_train, y_train))
    inner = []
    for train_positions, _ in outer:
        X_outer, y_outer = X_train.iloc[train_positions], y_train.iloc[train_positions]
        if y_outer.value_counts().min() < config.inner_folds:
            raise ValueError("Too few records per class for Inner Stratified CV.")
        inner.append(list(_splitter(config.inner_folds, config).split(X_outer, y_outer)))
    final = list(_splitter(config.inner_folds, config).split(X_train, y_train))
    return outer, inner, final


def cv_manifest(X_train, outer, inner, final, config):
    """Record training-only IDs, never feature or target values."""
    def describe(index, splits):
        return [
            {"train_game_ids": index[tr].tolist(),
             "validation_game_ids": index[va].tolist()}
            for tr, va in splits
        ]

    outer_description = describe(X_train.index, outer)
    for fold, (tr, _) in enumerate(outer):
        outer_description[fold]["inner_folds"] = describe(X_train.index[tr], inner[fold])
    return {
        "configuration": asdict(config),
        "scope": "Part A training partition only; no held-out test records",
        "outer_folds": outer_description,
        "final_search_folds": describe(X_train.index, final),
    }


def _search(estimator, grid, cv, n_jobs):
    return GridSearchCV(
        estimator=clone(estimator), param_grid=grid, scoring=SCORING,
        refit="accuracy", cv=cv, n_jobs=n_jobs, pre_dispatch=n_jobs,
        return_train_score=False, error_score="raise",
    )


def classification_metrics(estimator, X_validation, y_validation):
    """Score a validation fold; SVM margins suffice for ROC-AUC."""
    if list(estimator.classes_) != [0, 1]:
        raise ValueError("Unexpected class ordering; scores must refer to Blue wins.")
    predictions = estimator.predict(X_validation)
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_validation)
    else:
        scores = estimator.predict_proba(X_validation)[:, 1]
    return {
        "accuracy": accuracy_score(y_validation, predictions),
        "precision": precision_score(y_validation, predictions, zero_division=0),
        "recall": recall_score(y_validation, predictions, zero_division=0),
        "f1": f1_score(y_validation, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_validation, scores),
    }


def _save_search(search, directory, stem):
    frame = pd.DataFrame(search.cv_results_)
    frame.to_csv(directory / f"{stem}.csv", index=False)
    _write_json(directory / f"{stem}.json", frame.to_dict(orient="records"))


def summarize_outer_scores(scores):
    """Unweighted fold means and sample SD (ddof=1); SD is not a CI."""
    frame = pd.DataFrame(scores)
    records = []
    for name in MODEL_NAMES:
        subset = frame.loc[frame.model == name]
        row = {"model": name, "outer_folds": len(subset),
               "selection_candidate": name in LEARNED_MODELS}
        for metric in METRIC_NAMES:
            row[f"mean_{metric}"] = float(subset[metric].mean())
            row[f"std_{metric}"] = float(subset[metric].std(ddof=1))
        records.append(row)
    result = pd.DataFrame(records)
    baseline = result.loc[result.model == "dummy", "mean_accuracy"].iloc[0]
    result["accuracy_gain_over_dummy"] = result.mean_accuracy - baseline
    return result


def select_model(comparison):
    """Predeclared rule: mean accuracy, then SD, then DT/KNN/SVM on exact ties."""
    candidates = comparison.loc[comparison.model.isin(LEARNED_MODELS)].copy()
    candidates["tie_order"] = candidates.model.map(dict(zip(LEARNED_MODELS, range(3))))
    return candidates.sort_values(
        ["mean_accuracy", "std_accuracy", "tie_order"], ascending=[False, True, True],
    ).iloc[0].model


def _validate_run_id(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run_id must be a simple name (letters, digits, _ or -).")


def run_training(
    X_train, y_train, *, root=REPOSITORY_ROOT, run_id=DEFAULT_RUN_ID,
    config=CVConfig(), n_jobs=2, provenance=None, grids=None, progress=print,
):
    """Run nested CV then final tuning/refits, without accepting a test set.

    `grids` permits tiny integration-test grids; the CLI always uses the full
    approved search. Every run records its exact search space. Existing run
    directories are refused so an earlier experiment cannot be silently replaced.
    """
    validate_training_input(X_train, y_train)
    _validate_run_id(run_id)
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer; only inner searches parallelize.")
    if provenance is not None and provenance.get("source_sha256") != ORIGINAL_SHA256:
        raise ValueError("Only the original Diamond source is supported.")
    root = Path(root).resolve()
    grids = parameter_grids() if grids is None else grids
    if set(grids) != set(LEARNED_MODELS):
        raise ValueError("Supply grids for Decision Tree, KNN and SVM only.")
    outer, inner, final = build_cv_splits(X_train, y_train, config)
    report_directory = root / "reports/model_training" / run_id
    model_directory = root / "models/part_b" / run_id
    manifest_directory = root / "data/processed/part_b" / run_id
    for directory in (report_directory, model_directory, manifest_directory):
        if directory.exists():
            raise FileExistsError(f"Run already exists: {directory}. Choose another --run-id.")
    for directory in (report_directory, model_directory, manifest_directory):
        directory.mkdir(parents=True)
    inner_directory = report_directory / "inner_cv"
    inner_directory.mkdir()
    manifest_path = manifest_directory / "cv_manifest.json"
    _write_json(manifest_path, cv_manifest(X_train, outer, inner, final, config))
    models = build_models()
    counts = {name: len(ParameterGrid(grid)) for name, grid in grids.items()}
    expected_fits = (config.outer_folds + 1) * (
        config.inner_folds * sum(counts.values()) + len(MODEL_NAMES)
    )
    metadata = {
        "run_id": run_id, "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": "Can we predict the winning team in a League of Legends match using only statistics available after the first 10 minutes?",
        "dataset": "Diamond only", "source_sha256": ORIGINAL_SHA256,
        "provenance": provenance or {}, "training_data_sha256": _data_hash(X_train, y_train),
        "training_rows": len(X_train), "feature_names": list(FEATURE_NAMES),
        "training_class_counts": y_train.value_counts().sort_index().to_dict(),
        "target": TARGET, "positive_class": 1, "positive_class_name": "Blue wins",
        "cv": asdict(config), "final_search_folds": config.inner_folds,
        "search_space": grids, "candidate_counts": counts, "expected_fit_count": expected_fits,
        "primary_metric": "accuracy", "metrics": list(METRIC_NAMES),
        "selection_rule": "maximum outer mean accuracy; exact ties: minimum sample SD, then decision_tree/knn/svm",
        "baseline": "DummyClassifier(strategy=most_frequent); diagnostic reference, no tuning",
        "inner_tie_rule": "first candidate in deterministic sklearn ParameterGrid order",
        "standard_deviation": "sample SD across outer folds (ddof=1), not a confidence interval",
        "held_out_test_used": False, "threshold_tuning": False,
        "n_jobs": n_jobs, "inner_thread_limit": 1,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": environment_versions(),
        "training_module_sha256": _file_hash(__file__),
        "cv_manifest_path": str(manifest_path.relative_to(root)),
        "cv_manifest_sha256": _file_hash(manifest_path),
        "model_files": {}, "warnings": [],
    }
    metadata_path = report_directory / "run_metadata.json"
    _write_json(metadata_path, metadata)
    scores, final_parameters = [], {}
    started = time.perf_counter()
    caught_warnings = []

    def notify(message):
        if progress is not None:
            progress(message)

    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            # Outer folds are sequential. Limit workers' BLAS/OpenMP threads too.
            with parallel_config(backend="loky", inner_max_num_threads=1):
                with threadpool_limits(limits=1):
                    for fold, (tr, va) in enumerate(outer, start=1):
                        X_outer, y_outer = X_train.iloc[tr], y_train.iloc[tr]
                        X_validation, y_validation = X_train.iloc[va], y_train.iloc[va]
                        for name in MODEL_NAMES:
                            notify(f"Outer {fold}/{config.outer_folds}: {name} ...")
                            tick = time.perf_counter()
                            if name == "dummy":
                                fitted = clone(models[name]).fit(X_outer, y_outer)
                                best_parameters = {"classifier__strategy": "most_frequent"}
                                inner_accuracy = None
                            else:
                                search = _search(models[name], grids[name], inner[fold - 1], n_jobs)
                                search.fit(X_outer, y_outer)
                                fitted = search.best_estimator_
                                best_parameters = search.best_params_
                                inner_accuracy = search.best_score_
                                _save_search(search, inner_directory, f"outer_{fold}_{name}")
                            fit_seconds = time.perf_counter() - tick
                            tick = time.perf_counter()
                            metrics = classification_metrics(fitted, X_validation, y_validation)
                            scores.append({
                                "outer_fold": fold, "model": name,
                                "train_rows": len(tr), "validation_rows": len(va),
                                **metrics, "inner_best_accuracy": inner_accuracy,
                                "best_params": best_parameters, "fit_seconds": fit_seconds,
                                "score_seconds": time.perf_counter() - tick,
                            })
                            _write_json(report_directory / "nested_cv_scores.json", scores)
                            notify(f"  Accuracy={metrics['accuracy']:.4f}; ROC-AUC={metrics['roc_auc']:.4f}")

                    comparison = summarize_outer_scores(scores)
                    selected_model = select_model(comparison)
                    # Freeze the family choice BEFORE final training-only searches.
                    _write_json(report_directory / "selection.json", {
                        "selected_model": selected_model,
                        "selection_rule": metadata["selection_rule"],
                        "selection_source": "Outer CV scores only; frozen before final searches",
                        "held_out_test_used": False,
                    })
                    comparison["selected"] = comparison.model == selected_model
                    comparison.to_csv(report_directory / "model_comparison.csv", index=False)
                    _write_json(report_directory / "model_comparison.json", comparison.to_dict(orient="records"))
                    csv_scores = pd.DataFrame(scores)
                    csv_scores["best_params"] = csv_scores.best_params.map(lambda p: json.dumps(p, sort_keys=True))
                    csv_scores.to_csv(report_directory / "nested_cv_scores.csv", index=False)
                    baseline = comparison.loc[comparison.model == "dummy", "mean_accuracy"].iloc[0]
                    selected_accuracy = comparison.loc[comparison.selected, "mean_accuracy"].iloc[0]
                    if selected_accuracy <= baseline:
                        metadata["warnings"].append("The selected learned model does not exceed the Dummy baseline in mean Outer CV Accuracy.")

                    for name in MODEL_NAMES:
                        notify(f"Final training-only refit: {name} ...")
                        if name == "dummy":
                            fitted = clone(models[name]).fit(X_train, y_train)
                            final_parameters[name] = {"classifier__strategy": "most_frequent"}
                        else:
                            search = _search(models[name], grids[name], final, n_jobs)
                            search.fit(X_train, y_train)
                            fitted = search.best_estimator_
                            final_parameters[name] = search.best_params_
                            _save_search(search, report_directory, f"cv_results_{name}")
                        artifact = model_directory / f"{name}.joblib"
                        joblib.dump(fitted, artifact, compress=3)
                        metadata["model_files"][name] = {
                            "path": str(artifact.relative_to(root)),
                            "sha256": _file_hash(artifact), "classes": list(fitted.classes_),
                        }
                        notify(f"  saved {name}; parameters={final_parameters[name]}")
                    _write_json(report_directory / "best_params.json", final_parameters)

        metadata.update({
            "status": "complete", "selected_model": selected_model,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
        })
    except Exception as error:
        metadata.update({
            "status": "failed", "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.perf_counter() - started,
        })
        raise
    finally:
        metadata["warnings"].extend(sorted({
            f"{w.category.__name__}: {w.message}" for w in caught_warnings
        }))
        _write_json(metadata_path, metadata)
    notify(f"Complete. Selected using Outer CV: {selected_model}")
    return TrainingRun(run_id, report_directory, model_directory, comparison, selected_model)


def load_run_results(root=REPOSITORY_ROOT, run_id=DEFAULT_RUN_ID):
    """Read aggregate JSON results without loading the source CSV or any model."""
    _validate_run_id(run_id)
    directory = Path(root) / "reports/model_training" / run_id
    metadata = json.loads((directory / "run_metadata.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("This run is incomplete; do not use partial results.")
    return {
        "metadata": metadata,
        "comparison": pd.DataFrame(json.loads((directory / "model_comparison.json").read_text())),
        "outer_scores": pd.DataFrame(json.loads((directory / "nested_cv_scores.json").read_text())),
        "best_params": json.loads((directory / "best_params.json").read_text()),
    }


def load_final_model(name, *, root=REPOSITORY_ROOT, run_id=DEFAULT_RUN_ID):
    """Load a trusted local artifact, checking versions, checksum and schema."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    root = Path(root).resolve()
    metadata = load_run_results(root, run_id)["metadata"]
    if metadata["feature_names"] != list(FEATURE_NAMES) or metadata["source_sha256"] != ORIGINAL_SHA256:
        raise ValueError("Saved run differs from the Part A data contract.")
    if metadata["packages"] != environment_versions():
        raise ValueError("Package versions differ. Use requirements-part-b.txt or regenerate the run.")
    if metadata["python"].split(".")[:2] != platform.python_version().split(".")[:2]:
        raise ValueError("Use the same Python major/minor version as the saved run.")
    artifact = root / "models/part_b" / run_id / f"{name}.joblib"
    if _file_hash(artifact) != metadata["model_files"][name]["sha256"]:
        raise ValueError("Model checksum mismatch; do not evaluate a changed artifact.")
    return joblib.load(artifact)
