"""Đọc, ghi artifact và metadata của study."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import pandas as pd

from cmapss.modeling.audit_study.config import AuditStudyConfig
from cmapss.modeling.audit_study.evaluation import (
    _engine_readout,
    _model_readout,
    _paired_comparison,
    _warning_readout,
)
from cmapss.modeling.metrics import add_prediction_errors


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: Any) -> Any:
    """Chuyển sang kiểu JSON thuần; NaN và inf thành None."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(_json_value(payload), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("Đầu vào và output nghiên cứu phải nằm trong repository") from error


def _write_comparisons(predictions: pd.DataFrame, events: pd.DataFrame,
                        config: AuditStudyConfig, output: Path) -> dict[str, Any]:
    """Ghi metric tổng hợp, theo động cơ, vùng RUL, so sánh cặp và decision.json."""
    summaries, engine_tables, region_rows = [], {}, []
    for procedure, frame in predictions.groupby("procedure", sort=True):
        policy_events = events[events.procedure.eq(procedure)]
        summaries.append({"procedure": procedure, **_model_readout(frame, config),
                          **_warning_readout(policy_events)})
        engine_tables[procedure] = _engine_readout(frame, policy_events, config)
        scored = add_prediction_errors(frame)
        scored["rul_region"] = pd.cut(scored.y_true, [-1, 30, 60, 100, 150, np.inf],
                                      labels=["0–30", "31–60", "61–100", "101–150", ">150"])
        for region, group in scored.groupby("rul_region", observed=False):
            if group.empty:
                continue
            region_rows.append({"procedure": procedure, "rul_region": str(region),
                                "n_rows": len(group), "n_engines": group.engine_id.nunique(),
                                "mean_window_mae": float(group.absolute_error.mean()),
                                "mean_engine_mae": float(group.groupby("engine_id").absolute_error.mean().mean()),
                                "signed_bias": float((group.y_pred - group.y_true).mean()),
                                "rmse": float(np.sqrt(group.squared_error.mean()))})
    pd.DataFrame(summaries).to_csv(output / "aggregate_metrics.csv", index=False)
    pd.concat([table.assign(procedure=name).reset_index() for name, table in engine_tables.items()],
              ignore_index=True).to_csv(output / "engine_metrics.csv", index=False)
    comparisons = []
    for reference, candidate in (("baseline_fixed", "baseline_calibrated"),
                                  ("baseline_fixed", "selected_calibrated"),
                                  ("baseline_calibrated", "selected_calibrated")):
        comparison = _paired_comparison(engine_tables[reference], engine_tables[candidate], config)
        comparisons.append(comparison.assign(reference=reference, procedure=candidate))
    table = pd.concat(comparisons, ignore_index=True)
    table.to_csv(output / "paired_comparison.csv", index=False)
    pd.DataFrame(region_rows).to_csv(output / "region_metrics.csv", index=False)
    gate = table[(table.reference == "baseline_fixed") & (table.procedure == "selected_calibrated")
                 & table.metric.eq("endpoint_mean_engine_mae")].iloc[0]
    standard = AuditStudyConfig()
    scientific = (config.outer_splits == 5 and config.inner_splits == 3
                  and config.hgb_max_iter == 200 and config.candidate_ids == standard.candidate_ids
                  and config.endpoint_fractions == standard.endpoint_fractions
                  and config.thresholds == standard.thresholds
                  and config.persistences == standard.persistences
                  and config.lead_requirement == standard.lead_requirement
                  and config.target_on_time_rate == standard.target_on_time_rate
                  and config.seed == standard.seed
                  and config.bootstrap_resamples >= 10000)
    decision = {"full_protocol_configuration": scientific,
                "outer_endpoint_mae_delta": float(gate.delta),
                "outer_endpoint_mae_ci_lower": float(gate.ci_lower),
                "outer_endpoint_mae_ci_upper": float(gate.ci_upper),
                "promotion_gate_met": bool(scientific and gate.ci_upper < 0),
                "historical_baseline_overwritten": False,
                "gate": "paired outer endpoint-MAE delta upper95% CI<0; no test gating",
                "tradeoff_note": "gate is only endpoint accuracy; inspect NASA, full trajectory and warning tradeoffs before describing the result",
                "evaluation_limit": "retrospective train endpoint panel; conditional engine bootstrap; not an untouched real-world test"}
    _write_json(output / "decision.json", decision)
    return decision


def _freeze_selection(path: Path, selection: dict[str, Any]) -> None:
    """Lưu lựa chọn cuối; khi resume, lựa chọn phải giữ nguyên."""
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != _json_value(selection):
            raise ValueError("Lựa chọn cuối đã thay đổi khi resume")
    else:
        _write_json(path, selection)
