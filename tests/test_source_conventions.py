"""Kiểm tra annotation và kiểu trả về của hàm trong src."""

from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]


def test_source_functions_have_return_annotations():
    missing = []
    for path in (ROOT / "src/cmapss").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is None:
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    assert not missing, missing
