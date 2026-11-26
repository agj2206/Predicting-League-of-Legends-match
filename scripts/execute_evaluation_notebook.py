"""Execute Part C using saved evidence only; this script cannot evaluate models."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/part_c_matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache/part_c_ipython"))
os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
for name in ("MPLCONFIGDIR", "IPYTHONDIR"):
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="diamond_5x3_seed42")
    args = parser.parse_args()
    os.environ["LOL_PART_C_RUN_ID"] = args.run_id
    path = ROOT / "notebooks/03_model_evaluation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="lol-part-c-kernel-") as temporary:
        kernel_directory = Path(temporary) / "part-c"
        kernel_directory.mkdir()
        (kernel_directory / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Part C execution", "language": "python",
        }))
        manager = KernelManager(kernel_name="part-c", kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
        client = NotebookClient(notebook, km=manager, timeout=180, allow_errors=False,
                                resources={"metadata": {"path": str(ROOT)}}, record_timing=False)
        client.execute(cleanup_kc=True)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not all(cell.execution_count is not None for cell in code_cells):
        raise RuntimeError("Some Part C code cells did not execute.")
    nbformat.write(notebook, path)
    print(f"Executed {len(code_cells)} Part C code cells using saved evidence only.")


if __name__ == "__main__":
    main()
