"""Verify immutable experiment sources independently of later repository edits."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def verify_source_snapshot(run: Path, manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Verify every recorded source and lockfile, using the archived bytes if present."""
    expected = dict(manifest["identity"]["source_sha256"])
    expected["uv.lock"] = manifest["identity"]["dependencies_sha256"]
    working_tree_differences: list[str] = []
    archived = 0
    for name, digest in expected.items():
        relative = Path(name.replace("\\", "/"))
        current = (repo_root / relative).resolve()
        if not current.is_relative_to(repo_root.resolve()):
            raise ValueError(f"Source path outside repository: {name}")
        snapshot = run / "source_snapshot" / relative
        verified = snapshot if snapshot.is_file() else current
        if hashlib.sha256(verified.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Experiment source checksum mismatch: {name}")
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
