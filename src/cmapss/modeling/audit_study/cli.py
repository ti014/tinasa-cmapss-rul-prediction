"""
Nghiên cứu FD001 bằng nested cross-validation: chọn mô hình tại 50/70/90% tuổi thọ và
hiệu chuẩn cảnh báo trên inner OOF; tuổi thọ không là feature.

Report không trình bày nghiên cứu này; package được giữ vì
scripts/build_course_alignment_evidence.py dùng hàm đọc cảnh báo ngưỡng 30/40.
"""

from __future__ import annotations

import argparse
import json

from cmapss.modeling.audit_study.artifact_io import _json_value
from cmapss.modeling.audit_study.config import AuditStudyConfig
from cmapss.modeling.audit_study.runner import run_audit_study


def main(argv: list[str] | None = None) -> None:
    """Điểm vào từ dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", default="outputs/etl/silver/observations.parquet")
    parser.add_argument("--targets", default="outputs/etl/silver/train_targets.parquet")
    parser.add_argument("--test-labels", default="data/raw/RUL_FD001.txt")
    parser.add_argument("--output-dir", default="outputs/experiments/audit_remediation/20261005")
    parser.add_argument("--fit-threads", type=int, default=4)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="chạy thử: 2x2 fold, hai ứng viên, 20 vòng boosting, 20 lần bootstrap, không đọc nhãn test")
    args = parser.parse_args(argv)
    if args.dry_run:
        if args.output_dir == parser.get_default("output_dir"):
            parser.error("--dry-run cần --output-dir riêng, khác thư mục mặc định")
        config = AuditStudyConfig(outer_splits=2, inner_splits=2,
                                  candidate_ids=("hgb_baseline", "hgb_history"),
                                  hgb_max_iter=20, bootstrap_resamples=20,
                                  fit_threads=args.fit_threads)
    else:
        config = AuditStudyConfig(fit_threads=args.fit_threads)
    summary = run_audit_study(observations_path=args.observations, targets_path=args.targets,
                              test_labels_path=args.test_labels, output_dir=args.output_dir,
                              config=config, evaluate_test=not (args.dry_run or args.skip_test_evaluation))
    print(json.dumps(_json_value({key: summary[key] for key in (
        "status", "output_dir", "selected_candidate_id", "warning_policy", "decision",
        "unique_cached_cv_fits", "elapsed_seconds")}), ensure_ascii=False, indent=2), flush=True)
