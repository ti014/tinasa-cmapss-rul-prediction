"""Xác minh nguồn thí nghiệm bất biến, độc lập với các lần sửa repository sau này."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


def verify_source_snapshot(run: Path, manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Kiểm tra mọi nguồn và lockfile đã ghi nhận, ưu tiên nội dung snapshot nếu có."""
    expected = dict(manifest["identity"]["source_sha256"])
    expected["uv.lock"] = manifest["identity"]["dependencies_sha256"]
    working_tree_differences: list[str] = []
    archived = 0
    for name, digest in expected.items():
        relative = Path(name.replace("\\", "/"))
        current = (repo_root / relative).resolve()
        if not current.is_relative_to(repo_root.resolve()):
            raise ValueError(f"Đường dẫn nguồn nằm ngoài repository: {name}")
        snapshot = run / "source_snapshot" / relative
        verified = snapshot if snapshot.is_file() else current
        if hashlib.sha256(verified.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mã nguồn thí nghiệm không khớp: {name}")
        archived += int(snapshot.is_file())
        if not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest() != digest:
            working_tree_differences.append(relative.as_posix())
    return {
        "recorded_source_verified": True,
        "verified_files": len(expected),
        "archived_files": archived,
        "working_tree_matches_recorded_source": not working_tree_differences,
        "working_tree_differences": working_tree_differences,
    }


def local_source_files(root: Path, entries: tuple[str, ...]) -> tuple[str, ...]:
    """
    Thu thập các file đầu vào cùng import cmapss phụ thuộc và __init__.py từ repository;
    báo FileNotFoundError nếu thiếu nguồn tường minh.
    """
    pending = list(entries)
    found: set[str] = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu mã nguồn: {name}")
        found.add(name)
        if not name.startswith("src/"):
            continue
        package = Path(name).parent.relative_to("src").as_posix().replace("/", ".")
        parents = Path(name).parent
        while parents != Path("src"):
            initializer = (parents / "__init__.py").as_posix()
            if (root / initializer).is_file():
                pending.append(initializer)
            parents = parents.parent
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1])
                    module = ".".join(filter(None, (base, node.module)))
                else:
                    module = node.module or ""
                modules = [module, *(f"{module}.{alias.name}" for alias in node.names)]
            for module in modules:
                if not module.startswith("cmapss"):
                    continue
                candidate = Path("src") / Path(*module.split("."))
                for local in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                    if (root / local).is_file():
                        pending.append(local.as_posix())
    return tuple(sorted(found))
