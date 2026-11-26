"""Audit completed Part C outputs without source-row loading or model inference."""

import base64
from contextlib import ExitStack
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lol_model_evaluation import (
    MODEL_DISPLAY_NAMES, create_figures, file_hash, load_saved_evaluation,
    output_paths, write_json,
)
from src.lol_preprocessing import resolve_data_path


class TableParser(HTMLParser):
    """Extract notebook table text without optional HTML parsing dependencies."""
    def __init__(self):
        super().__init__()
        self.rows, self.row = [], []
        self.inside = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.inside = True
            self.parts = []

    def handle_data(self, data):
        if self.inside:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            self.row.append("".join(self.parts).strip())
            self.inside = False
        elif tag == "tr":
            self.rows.append(self.row)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_text_tables(text, metrics, matrices):
    metric_models, count_models = set(), set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for name, label in MODEL_DISPLAY_NAMES.items():
            if cells[0] != label:
                continue
            if len(cells) == 6:
                expected = [f"{metrics[name][m]:.4f}" for m in ("accuracy", "precision", "recall", "f1", "roc_auc")]
                check(cells[1:] == expected, f"Metric table mismatch: {label}")
                metric_models.add(name)
            if len(cells) == 5 and all(c.isdigit() for c in cells[1:]):
                check([int(c) for c in cells[1:]] == [matrices[name][k] for k in ("TN", "FP", "FN", "TP")],
                      f"Confusion table mismatch: {label}")
                count_models.add(name)
    check(len(metric_models) == len(count_models) == 4, "Four model rows are required in both tables.")


def main():
    run_id = "diamond_5x3_seed42"
    paths = output_paths(ROOT, run_id)
    directory = paths["reports"]
    saved = load_saved_evaluation(root=ROOT, run_id=run_id)
    check(saved["metadata"]["status"] == "complete", "Final evaluation is not complete.")
    immutable_files = ["test_metrics.json", "test_metrics.csv", "confusion_matrices.json",
                       "confusion_matrices.csv", "roc_curves.json", "cv_test_comparison.json",
                       "cv_test_comparison.csv", "evaluation_metadata.json", "evaluation_started.json"]
    immutable = {name: file_hash(directory / name) for name in immutable_files}
    cache = paths["local"] / "test_predictions.csv"
    cache_hash = file_hash(cache)
    cached = pd.read_csv(cache)
    manifest = json.loads((ROOT / "data/processed/split_manifest.json").read_text())
    check(cached.gameId.tolist() == manifest["test_game_ids"], "Cached IDs differ from the original test order.")
    check(len(cached) == 1976 and cached.gameId.is_unique, "Expected 1,976 unique cached matches.")
    check(set(cached.actual_blueWins) == {0, 1}, "Expected binary actual outcomes.")
    check(int((cached.actual_blueWins == 1).sum()) == 986, "Unexpected Blue count.")
    metrics_csv = pd.read_csv(directory / "test_metrics.csv", index_col="model")
    counts_csv = pd.read_csv(directory / "confusion_matrices.csv", index_col="model")
    comparison_csv = pd.read_csv(directory / "cv_test_comparison.csv")
    training = pd.DataFrame(json.loads((ROOT / "reports/model_training" / run_id / "model_comparison.json").read_text())).set_index("model")
    for name in MODEL_DISPLAY_NAMES:
        y = cached.actual_blueWins.to_numpy()
        predicted = cached[f"{name}_prediction"].to_numpy()
        score = cached[f"{name}_score"].to_numpy()
        check(np.isfinite(score).all() and np.isin(predicted, [0, 1]).all(), "Invalid cached output.")
        tn = int(((y == 0) & (predicted == 0)).sum())
        fp = int(((y == 0) & (predicted == 1)).sum())
        fn = int(((y == 1) & (predicted == 0)).sum())
        tp = int(((y == 1) & (predicted == 1)).sum())
        check(tn + fp + fn + tp == 1976, f"Count mismatch: {name}")
        check([tn, fp, fn, tp] == [saved["confusion"][name][k] for k in ("TN", "FP", "FN", "TP")], "JSON counts differ.")
        check(saved["confusion"][name]["matrix"] == [[tn, fp], [fn, tp]], "Matrix orientation differs.")
        check([tn, fp, fn, tp] == counts_csv.loc[name, ["TN", "FP", "FN", "TP"]].tolist(), "CSV counts differ.")
        n_positive, n_negative = int((y == 1).sum()), int((y == 0).sum())
        # Mann-Whitney rank formula independently verifies AUC, including ties.
        ranks = pd.Series(score).rank(method="average").to_numpy()
        auc = (ranks[y == 1].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
        expected = {"accuracy": (tn + tp) / 1976, "precision": tp / (tp + fp) if tp + fp else 0.,
                    "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn), "roc_auc": auc}
        for metric, value in expected.items():
            check(math.isclose(value, saved["metrics"][name][metric], abs_tol=1e-12, rel_tol=0), "Cached/JSON metric mismatch.")
            check(math.isclose(value, metrics_csv.loc[name, metric], abs_tol=1e-12, rel_tol=0), "Cached/CSV metric mismatch.")
        curve = saved["roc"][name]
        check(np.all(np.diff(curve["fpr"]) >= 0), "ROC false positive rates are not monotonic.")
        check(math.isclose(float(np.trapezoid(curve["tpr"], curve["fpr"])), auc, abs_tol=1e-12), "ROC coordinates do not reproduce AUC.")
    for i, row in enumerate(saved["comparison"]):
        name, metric = row["model"], row["metric"]
        expected = {"test_value": saved["metrics"][name][metric],
                    "outer_cv_mean": float(training.loc[name, f"mean_{metric}"]),
                    "outer_cv_sample_sd": float(training.loc[name, f"std_{metric}"])}
        expected["test_minus_outer_cv_mean"] = expected["test_value"] - expected["outer_cv_mean"]
        expected["test_minus_dummy"] = expected["test_value"] - saved["metrics"]["dummy"][metric]
        for key, value in expected.items():
            check(math.isclose(row[key], value, abs_tol=1e-12), "CV comparison mismatch.")
            check(math.isclose(float(comparison_csv.iloc[i][key]), value, abs_tol=1e-12), "CV comparison CSV mismatch.")
        check(row["selected_before_test"] == (name == "svm"), "Frozen family selection changed.")

    for filename in ("README_PART_C.md", "reports/part_c_findings.md"):
        verify_text_tables((ROOT / filename).read_text(), saved["metrics"], saved["confusion"])
    notebook = json.loads((ROOT / "notebooks/03_model_evaluation.ipynb").read_text())
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    check(all(c["execution_count"] is not None for c in code_cells), "Notebook cells are unexecuted.")
    metric_models, count_models, embedded = set(), set(), set()
    outputs_text = []
    for cell in code_cells:
        for output in cell.get("outputs", []):
            check(output["output_type"] != "error", "Notebook contains an error.")
            data = output.get("data", {})
            html = "".join(data.get("text/html", []))
            parser = TableParser()
            parser.feed(html)
            for row in parser.rows:
                for name, label in MODEL_DISPLAY_NAMES.items():
                    if row and row[0] == label and len(row) == 6:
                        values = [float(v) for v in row[1:]]
                        expected = [float(f"{saved['metrics'][name][m]:.4f}") for m in ("accuracy", "precision", "recall", "f1", "roc_auc")]
                        check(values == expected, "Notebook metric table differs.")
                        metric_models.add(name)
                    if row and row[0] == label and len(row) == 5:
                        check([int(v) for v in row[1:]] == [saved["confusion"][name][k] for k in ("TN", "FP", "FN", "TP")], "Notebook counts differ.")
                        count_models.add(name)
            if "image/png" in data:
                encoded = data["image/png"]
                image_bytes = base64.b64decode("".join(encoded) if isinstance(encoded, list) else encoded)
                matching = [p.name for p in paths["figures"].glob("*.png") if p.read_bytes() == image_bytes]
                check(len(matching) == 1, "Embedded notebook figure does not match a saved figure.")
                embedded.update(matching)
            outputs_text.append("".join(output.get("text", [])))
            outputs_text.append("".join(data.get("text/plain", [])))
            outputs_text.append("".join(data.get("text/markdown", [])))
    check(len(metric_models) == len(count_models) == 4, "Notebook tables omit models.")
    check(len(embedded) == 4, "Expected all four saved figures embedded in the notebook.")
    check("pending final evaluation" not in "\n".join(outputs_text).lower(), "Notebook still displays a pending result.")

    # Check plotted numerical inputs and recreate pixels without loading models.
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/part_c_matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    from matplotlib.axes import Axes
    plotted = {key: [] for key in ("imshow", "barh", "plot", "errorbar", "scatter")}
    def record(method, original):
        def recording(axis, *args, **kwargs):
            plotted[method].append((args, kwargs))
            return original(axis, *args, **kwargs)
        return recording
    with tempfile.TemporaryDirectory() as temporary:
        with ExitStack() as stack:
            stack.enter_context(patch("src.lol_model_evaluation.prepare_data", side_effect=AssertionError("No source parsing")))
            stack.enter_context(patch("src.lol_model_evaluation.load_final_model", side_effect=AssertionError("No model loading")))
            for method in plotted:
                stack.enter_context(patch.object(Axes, method, new=record(method, getattr(Axes, method))))
            create_figures(saved, root=Path(temporary), run_id=run_id)
        recreated = output_paths(temporary, run_id)["figures"]
        for image_path in paths["figures"].glob("*.png"):
            with Image.open(image_path) as original, Image.open(recreated / image_path.name) as reference:
                check(np.array_equal(np.asarray(original), np.asarray(reference)), "Figure pixels do not match saved evidence.")
    for (args, _), name in zip(plotted["imshow"], MODEL_DISPLAY_NAMES):
        check(np.array_equal(args[0], saved["confusion"][name]["matrix"]), "Plotted confusion counts differ.")
    check(len(plotted["imshow"]) == 4, "Expected four confusion matrices.")
    coef = saved["coefficients"].sort_values("standardized_coefficient")
    check(np.array_equal(plotted["barh"][0][0][0], coef.feature), "Coefficient plot labels differ.")
    check(np.array_equal(plotted["barh"][0][0][1], coef.standardized_coefficient), "Coefficient plot values differ.")
    for name, label in MODEL_DISPLAY_NAMES.items():
        lines = [(a, k) for a, k in plotted["plot"] if k.get("label", "").startswith(label + ": AUC=")]
        check(len(lines) == 1, "ROC legend omits a model.")
        args, kwargs = lines[0]
        check(np.array_equal(args[0], saved["roc"][name]["fpr"]) and
              np.array_equal(args[1], saved["roc"][name]["tpr"]), "Plotted ROC coordinates differ.")
        check(kwargs["label"].endswith(f"{saved['metrics'][name]['roc_auc']:.4f}"), "ROC legend AUC differs.")
    random_line = [(a, k) for a, k in plotted["plot"] if k.get("label") == "Random classifier (AUC = 0.5)"]
    check(len(random_line) == 1 and list(random_line[0][0][0]) == [0, 1] and
          list(random_line[0][0][1]) == [0, 1], "Random-classifier diagonal is missing.")
    error_args, error_kwargs = plotted["errorbar"][0]
    check(np.allclose(error_args[1], [training.loc[n, "mean_accuracy"] for n in MODEL_DISPLAY_NAMES]), "CV means plotted incorrectly.")
    check(np.allclose(error_kwargs["yerr"], [training.loc[n, "std_accuracy"] for n in MODEL_DISPLAY_NAMES]), "CV sample SD plotted incorrectly.")
    scatter_args, _ = plotted["scatter"][0]
    check(np.allclose(scatter_args[1], [saved["metrics"][n]["accuracy"] for n in MODEL_DISPLAY_NAMES]), "Test accuracies plotted incorrectly.")

    language_paths = [ROOT / "README_PART_C.md", ROOT / "reports/part_c_findings.md"]
    language_text = "\n".join(p.read_text() for p in language_paths)
    language_text += "\n" + "\n".join("".join(c["source"]) for c in notebook["cells"])
    language_text += "\n" + "\n".join(outputs_text)
    italian_words = r"\b(abbiamo|modello|modelli|partite|risultati|valutazione|preliminare|addestramento|vittoria|squadra|nessuna|nessun|approvazione|autorizzazione|eseguito|confronto|caratteristiche)\b"
    check(not re.search(italian_words, language_text, flags=re.IGNORECASE), "Possible Italian explanatory text remains.")

    baseline = json.loads((directory / "protected_file_hashes.json").read_text())
    for path, expected in baseline["repository_files"].items():
        check(file_hash(ROOT / path) == expected, f"Protected file changed: {path}")
    check(file_hash(resolve_data_path()) == baseline["source_sha256"], "Original dataset bytes changed.")
    check(all(file_hash(directory / name) == value for name, value in immutable.items()), "Evaluation artifacts changed during QA.")
    check(file_hash(cache) == cache_hash, "Prediction cache changed during QA.")
    report = {"status": "passed", "rows_per_model": 1976,
              "independent_auc_check": "Mann-Whitney statistic from average ranks, with half credit for ties",
              "notebook_executed_code_cells": len(code_cells), "notebook_error_outputs": 0,
              "notebook_embedded_figures_match_files": sorted(embedded),
              "json_csv_notebook_readme_report_metrics_match": True,
              "confusion_matrices_match_cached_predictions": True,
              "roc_coordinate_integrals_match_auc": True,
              "plotted_values_and_legends_match_saved_numbers": True,
              "recreated_figure_pixels_match_saved_files": True,
              "english_materials_review": "English prose reviewed; Italian keyword scan passed",
              "protected_files_unchanged": len(baseline["repository_files"]) + 1,
              "prediction_cache_sha256": cache_hash, "evaluation_artifact_hashes": immutable,
              "model_inference_executed_during_qa": False}
    write_json(directory / "quality_assurance.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
