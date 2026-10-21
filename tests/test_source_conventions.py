"""Kiểm tra annotation và việc import pipeline không đổi style toàn cục."""

from pathlib import Path
import ast
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_source_functions_have_return_annotations():
    missing = []
    for path in (ROOT / "src/cmapss").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is None:
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    assert not missing, missing


def test_timeseries_import_keeps_plot_style():
    code = (
        "import matplotlib; "
        "matplotlib.rcParams['font.size'] = 17; "
        "matplotlib.rcParams['axes.grid'] = False; "
        "import cmapss.timeseries.pipeline; "
        "assert matplotlib.rcParams['font.size'] == 17; "
        "assert matplotlib.rcParams['axes.grid'] is False"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
