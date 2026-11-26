"""Run Part C preflight by default; held-out inference needs an explicit option."""

import argparse
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lol_model_evaluation import (
    output_paths, render_saved_outputs, run_final_evaluation, run_preflight,
    save_preflight, write_json,
)
from src.lol_model_training import DEFAULT_RUN_ID


def run_preliminary_tests():
    """Run Part C alone: artificial inputs and read-only frozen-artifact checks."""
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_model_evaluation.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), end="")
    report = {"successful": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped),
              "scope": "Synthetic inputs and saved parameters only; no Diamond rows parsed"}
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("All preliminary checks must pass without skipped tests.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true", help="Default: no held-out data parsing or inference")
    mode.add_argument("--evaluate-held-out-test", action="store_true", help="Run only after separate user authorization")
    mode.add_argument("--render-saved-results", action="store_true", help="Rebuild plots/report without inference")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--data-path", type=Path, default=None)
    args = parser.parse_args()
    if args.render_saved_results:
        saved = render_saved_outputs(root=ROOT, run_id=args.run_id)
    else:
        tests = run_preliminary_tests()
        preflight, coefficients = run_preflight(root=ROOT, run_id=args.run_id, data_path=args.data_path)
        preflight["automatic_tests"] = tests
        save_preflight(preflight, coefficients, root=ROOT, run_id=args.run_id)
        write_json(output_paths(ROOT, args.run_id)["reports"] / "test_verification.json", tests)
        if args.evaluate_held_out_test:
            run_final_evaluation(authorize=True, root=ROOT, run_id=args.run_id, data_path=args.data_path)
        saved = render_saved_outputs(root=ROOT, run_id=args.run_id)
    print(f"Part C status: {saved['metadata']['status']}")
    if saved["metadata"]["status"] == "awaiting_authorization":
        print("Held-out test predictions and performance metrics: pending final authorization.")


if __name__ == "__main__":
    main()
