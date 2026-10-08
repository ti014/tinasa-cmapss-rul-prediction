"""CLI kiểm tra C-MAPSS và xuất báo cáo JSON, không thay đổi raw."""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from cmapss.data.contracts import DATASET_IDS, RAW_COLUMNS, ContractError
from cmapss.data.readers import read_observations, read_test_labels
from cmapss.data.targets import make_train_targets, validate_train_targets
from cmapss.data.validation import validate_dataset_profile, validate_test_labels


def validate_dataset(
    raw_dir: Path,
    *,
    dataset_id: str = "FD001",
    check_test_labels: bool = False,
) -> dict[str, Any]:
    """Kiểm tra file train/test và trả kết quả kèm checksum nguồn.

    Nhãn test chỉ được đọc khi ``check_test_labels=True``, và chỉ kiểm tra cấu
    trúc, không join vào observations.
    """
    dataset_id = dataset_id.upper()
    if dataset_id not in DATASET_IDS:
        raise ContractError("DATASET_ID", dataset_id, f"dataset_id phải thuộc {DATASET_IDS}")
    checks: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    for split in ("train", "test"):
        path = raw_dir / f"{split}_{dataset_id}.txt"
        frame = read_observations(path, split=split)
        validate_dataset_profile(frame, dataset_id=dataset_id, split=split)
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        checks.append({
            "name": f"{split}_observations",
            "status": "passed",
            "raw_columns": len(RAW_COLUMNS),
            "rows": len(frame),
            "engines": int(frame["engine_id"].nunique()),
            "rules": ["RAW_WIDTH", "SCHEMA", "NUMERIC", "NONEMPTY", "SPLIT",
                      "UNIQUE_KEY", "CYCLE_CONTINUITY", "DATASET_PROFILE"],
        })
        if split == "train":
            targets = make_train_targets(frame)
            validate_train_targets(frame, targets)
            checks.append({
                "name": "train_targets",
                "status": "passed",
                "rows_before": len(frame),
                "rows_after": len(targets),
                "rules": ["TRAIN_ONLY", "TARGET_KEYS", "TRAIN_RUL"],
            })
    if check_test_labels:
        path = raw_dir / f"RUL_{dataset_id}.txt"
        labels = read_test_labels(path)
        validate_test_labels(labels, source=path.name, fd001=False)
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        checks.append({
            "name": "test_label_integrity",
            "status": "passed",
            "rows": len(labels),
            "rules": ["RAW_WIDTH", "NUMERIC", "NONEMPTY", "SPLIT", "UNIQUE_KEY",
                      "LABEL_MAPPING", "DATASET_PROFILE"],
        })
    else:
        checks.append({"name": "test_label_integrity", "status": "not_requested"})
    return {"checks": checks, "input_sha256": inputs}


def main(argv: list[str] | None = None) -> int:
    """Chạy validation; trả mã 0 khi đạt, 1 khi lỗi dữ liệu, 2 khi lỗi output."""
    parser = argparse.ArgumentParser(description="Kiểm tra data contract của NASA C-MAPSS.")
    parser.add_argument("--dataset-id", choices=DATASET_IDS, default="FD001")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/fd001.json"))
    parser.add_argument(
        "--check-test-labels",
        action="store_true",
        help="Kiểm tra riêng tính toàn vẹn nhãn test.",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.is_relative_to(args.raw_dir.resolve()):
        error = ContractError("RAW_IMMUTABLE", str(output), "không ghi output vào raw")
        print(error, file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "numpy": version("numpy"),
            "cmapss-rul": version("cmapss-rul"),
        },
        "test_labels_requested": args.check_test_labels,
        "test_labels_joined": False,
    }
    try:
        report["dataset_id"] = args.dataset_id
        report.update(
            validate_dataset(
                args.raw_dir,
                dataset_id=args.dataset_id,
                check_test_labels=args.check_test_labels,
            )
        )
        report["status"] = "passed"
    except (ContractError, OSError, UnicodeError) as error:
        report.update(status="failed", error=str(error))
        print(error, file=sys.stderr)

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        print(f"Không ghi được báo cáo {output}: {error}", file=sys.stderr)
        return 2
    print(f"Validation {report['status']}: {output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
