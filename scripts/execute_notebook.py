"""Execute Part A with this interpreter, without registering a global kernel.

Run from the repository root: .venv/bin/python scripts/execute_notebook.py
"""

import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache/ipython"))
os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["IPYTHONDIR"]).mkdir(parents=True, exist_ok=True)

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    notebook_path = ROOT / "notebooks/01_data_preprocessing.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    # A temporary kernelspec guarantees that the notebook uses this Python.
    # No kernelspec is installed in the user's account or system directories.
    with tempfile.TemporaryDirectory(prefix="lol-part-a-kernel-") as temporary:
        kernel_directory = Path(temporary) / "part-a"
        kernel_directory.mkdir()
        kernel_spec = {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Part A execution", "language": "python",
        }
        (kernel_directory / "kernel.json").write_text(json.dumps(kernel_spec))
        manager = KernelManager(
            kernel_name="part-a",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]),
        )
        client = NotebookClient(
            notebook, km=manager, timeout=180, allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}}, record_timing=False,
        )
        # We supplied the manager, so explicitly ask nbclient to close it too.
        client.execute(cleanup_kc=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    print(f"Executed {len(code_cells)} code cells successfully: {notebook_path}")


if __name__ == "__main__":
    main()
