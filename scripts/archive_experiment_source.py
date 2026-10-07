"""Preserve the exact source bytes of an existing experiment before maintenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """Copy only checksum-matching sources; never rewrite the recorded manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = (root / args.run).resolve()
    manifest = json.loads((run / "search_manifest.json").read_text(encoding="utf-8"))
    expected = dict(manifest["identity"]["source_sha256"])
    expected["uv.lock"] = manifest["identity"]["dependencies_sha256"]
    validated: list[tuple[Path, bytes]] = []
    for name, digest in expected.items():
        relative = Path(name.replace("\\", "/"))
        source = (root / relative).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"Source outside repository: {name}")
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Current bytes do not match recorded experiment: {name}")
        destination = run / "source_snapshot" / relative
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError(f"Refusing to replace a different snapshot: {destination}")
        validated.append((destination, payload))
    for destination, payload in validated:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    print(f"Archived {len(validated)} checksum-matching source/dependency files in {run}")


if __name__ == "__main__":
    main()
