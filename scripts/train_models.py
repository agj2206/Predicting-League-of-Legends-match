"""Run Part B locally. No downloads, test evaluation, or Git operations."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lol_model_training import DEFAULT_RUN_ID, load_training_data, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None, help="Original local Diamond CSV")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="New output name; existing runs are never overwritten")
    parser.add_argument("--n-jobs", type=int, default=2, help="Inner Grid Search workers; outer loop is sequential")
    args = parser.parse_args()
    X_train, y_train, provenance = load_training_data(args.data_path)
    print(f"Diamond training only: {X_train.shape}; 5 Outer / 3 Inner Folds", flush=True)
    run = run_training(
        X_train, y_train, root=ROOT, run_id=args.run_id, n_jobs=args.n_jobs,
        provenance=provenance, progress=lambda message: print(message, flush=True),
    )
    print(run.comparison.to_string(index=False))
    print(f"Results: {run.report_directory}\nModels: {run.model_directory}")


if __name__ == "__main__":
    main()
