"""Đọc, ghi artifact và metadata của study."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json

from cmapss.modeling import audit_study as audit


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    """Write JSON atomically; NaN/inf are rejected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(audit._json_value(value), ensure_ascii=False,
                                    sort_keys=True, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _freeze_json(path: Path, value: Any) -> None:
    """Write once; on resume, fail if the content differs."""
    normalized = audit._json_value(value)
    if path.exists():
        if _read_json(path) != normalized:
            raise ValueError(f"frozen receipt changed on resume: {path}")
    else:
        _write_json(path, normalized)


def _resolve(root: Path, value: str | Path) -> Path:
    """Resolve a path under the repo root; reject paths outside it."""
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path must remain inside repository: {path}")
    return path


def _progress(output: Path, stage: str, **values: Any) -> None:
    payload = {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **values}
    _write_json(output / "progress.json", payload)
    # Windows pipes can use cp1252; persisted JSON remains UTF-8.
    print(json.dumps(audit._json_value(payload), ensure_ascii=True), flush=True)
