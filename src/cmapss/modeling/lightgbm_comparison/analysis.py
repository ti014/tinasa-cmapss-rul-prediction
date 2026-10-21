"""
Chọn ứng viên chỉ trên inner fold và bootstrap cặp động cơ cho LightGBM; CI điều kiện
trên dự đoán đã fit, chưa tính biến động khi fit lại.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.modeling import audit_study as audit
from cmapss.modeling.metrics import add_prediction_errors

if TYPE_CHECKING:
    from cmapss.modeling.lightgbm_comparison.contracts import StudyConfig

ARMS = ("hgb_uncapped", "hgb_cap125", "lgbm_uncapped", "lgbm_cap125")
NUMERIC_METRICS = ("trajectory_rmse", "endpoint_mean_engine_mae", "endpoint_mean_engine_nasa")
PAIRED_ENGINE_METRICS = (
    "endpoint_mean_engine_mae", "endpoint_mean_engine_nasa",
    "trajectory_mean_engine_mae", "trajectory_mean_engine_nasa",
    "on_time_rate", "early_alarm_burden", "mean_transitions", "mean_retractions",
)
INTERVAL_SCOPE = (
    "paired engines conditional on fixed outer predictions; duplicate sampled engines "
    "retain multiplicity; excludes refitting variability"
)


def _require_finite(frame: pd.DataFrame | pd.Series, columns: tuple[str, ...]) -> None:
    """Báo lỗi nếu columns bị thiếu, không phải số hoặc không hữu hạn."""
    missing = set(columns) - set(frame.index if isinstance(frame, pd.Series) else frame.columns)
    if missing:
        raise ValueError(f"Thiếu metric bắt buộc: {sorted(missing)}")
    try:
        values = frame.loc[list(columns)] if isinstance(frame, pd.Series) else frame.loc[:, list(columns)]
        valid = np.isfinite(values.to_numpy(dtype="float64")).all()
    except (TypeError, ValueError) as error:
        raise ValueError("Metric lựa chọn và suy luận phải có kiểu số") from error
    if not valid:
        raise ValueError("Metric lựa chọn và suy luận phải hữu hạn")


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra vector quỹ đạo train và tính lại các cột sai số."""
    required = set(KEY) | {"y_true", "y_pred"}
    if predictions.empty or not required.issubset(predictions.columns):
        raise ValueError("Dự đoán cần khóa không rỗng, y_true gốc và y_pred")
    if predictions[list(KEY)].isna().any().any() or predictions.duplicated(list(KEY)).any():
        raise ValueError("Khóa dự đoán phải có định danh và duy nhất")
    if not predictions.split.eq("train").all():
        raise ValueError("So sánh quỹ đạo chỉ dùng động cơ train")
    _require_finite(predictions, ("engine_id", "cycle", "y_true", "y_pred"))
    if (predictions.y_true.lt(0).any() or predictions.cycle.lt(1).any()
            or not np.equal(predictions.cycle, np.floor(predictions.cycle)).all()
            or not np.equal(predictions.engine_id, np.floor(predictions.engine_id)).all()):
        raise ValueError("Khóa nguyên không hợp lệ hoặc RUL gốc âm")
    result = add_prediction_errors(predictions)
    _require_finite(result, ("absolute_error", "squared_error", "nasa_score_component"))
    return result.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)


def candidate_readout(predictions: pd.DataFrame, cfg: StudyConfig) -> dict[str, Any]:
    """Metric endpoint và quỹ đạo trên RUL gốc của một ứng viên."""
    return audit._model_readout(_validate_predictions(predictions), cfg.audit_config())


def select_lgbm_candidate(
    records: pd.DataFrame, reference: pd.Series, family: str, cfg: StudyConfig,
) -> dict[str, Any]:
    """
    Chọn LightGBM có inner RMSE thấp nhất và endpoint MAE/NASA không kém HGB; nếu không
    ứng viên nào đạt, trả ứng viên không ràng buộc với numeric_controls_met=False.
    """
    if family not in ("lgbm_uncapped", "lgbm_cap125") or "candidate_id" not in records:
        raise ValueError("Họ LightGBM chưa biết hoặc thiếu định danh ứng viên")
    mask = (records.family.eq(family) if "family" in records
            else records.candidate_id.astype(str).str.startswith(family + "_"))
    candidates = records.loc[mask].copy()
    if (candidates.empty or candidates.candidate_id.isna().any()
            or candidates.candidate_id.duplicated().any()):
        raise ValueError("Ứng viên trong họ phải không rỗng và có định danh duy nhất")
    _require_finite(candidates, NUMERIC_METRICS)
    _require_finite(reference, NUMERIC_METRICS[1:])
    controls = (
        candidates.endpoint_mean_engine_mae.le(reference.endpoint_mean_engine_mae + cfg.endpoint_mae_margin)
        & candidates.endpoint_mean_engine_nasa.le(reference.endpoint_mean_engine_nasa + cfg.nasa_score_margin)
    )
    feasible = candidates.loc[controls]
    pool = feasible if not feasible.empty else candidates
    winner = pool.sort_values(
        ["trajectory_rmse", "endpoint_mean_engine_mae", "candidate_id"], kind="mergesort",
    ).iloc[0]
    return {
        "candidate_id": str(winner.candidate_id), "family": family,
        "numeric_controls_met": bool(not feasible.empty),
        "n_evaluated": len(candidates), "n_feasible": len(feasible),
        "rule": "inner pooled trajectory RMSE, then endpoint MAE, then candidate ID; "
                "endpoint engine MAE and NASA <= corresponding HGB; "
                "unconstrained diagnostic winner if none is feasible",
    }


def calibrate_policy(
    predictions: pd.DataFrame, cfg: StudyConfig,
) -> tuple[audit.WarningPolicy, pd.DataFrame]:
    """Hiệu chuẩn lưới chính sách cảnh báo trên dự đoán inner của một nhánh."""
    return audit._policy_grid(_validate_predictions(predictions), cfg.audit_config())


def choose_procedure(arm_records: pd.DataFrame, cfg: StudyConfig) -> dict[str, Any]:
    """
    Chọn nhánh hợp lệ có inner RMSE thấp nhất; endpoint/cảnh báo không kém hgb_uncapped,
    LightGBM còn phải đạt numeric_controls_met.
    """
    if ("procedure" not in arm_records or arm_records.procedure.duplicated().any()
            or set(arm_records.procedure) != set(ARMS)):
        raise ValueError("Lựa chọn quy trình inner cần đúng bốn nhánh duy nhất")
    required = NUMERIC_METRICS + ("on_time_rate", "early_alarm_burden")
    _require_finite(arm_records, required)
    if "numeric_controls_met" not in arm_records:
        raise ValueError("Cần cờ khả thi về số của họ mô hình")
    if not arm_records.numeric_controls_met.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Cờ khả thi về số của họ mô hình phải là boolean")
    records = arm_records.set_index("procedure", drop=False)
    reference = records.loc["hgb_uncapped"]
    checks: dict[str, dict[str, bool]] = {}
    for procedure, row in records.iterrows():
        endpoint = bool(
            row.endpoint_mean_engine_mae <= reference.endpoint_mean_engine_mae + cfg.endpoint_mae_margin
            and row.endpoint_mean_engine_nasa <= reference.endpoint_mean_engine_nasa + cfg.nasa_score_margin
        )
        warnings = bool(
            row.on_time_rate >= reference.on_time_rate - cfg.warning_rate_margin
            and row.early_alarm_burden <= reference.early_alarm_burden
            + getattr(cfg, "early_alarm_burden_margin", 0.0)
        )
        family_feasible = bool(row.numeric_controls_met) if str(procedure).startswith("lgbm_") else True
        checks[str(procedure)] = {
            "numeric_controls_met": endpoint and family_feasible,
            "warning_controls_met": warnings,
            "eligible": endpoint and family_feasible and warnings,
        }
    eligible = [procedure for procedure in ARMS if checks[procedure]["eligible"]]
    winner = records.loc[eligible].reset_index(drop=True).sort_values(
        ["trajectory_rmse", "endpoint_mean_engine_mae", "procedure"], kind="mergesort",
    ).iloc[0]
    procedure_id = str(winner.procedure)
    return {
        "procedure_id": procedure_id, "selected_procedure_id": procedure_id,
        **checks[procedure_id], "arm_controls": checks,
        "rule": "inner pooled trajectory RMSE, then endpoint MAE, then arm ID; "
                "endpoint MAE/NASA and calibrated warning controls versus hgb_uncapped; "
                "LightGBM also requires corresponding-HGB family feasibility",
    }


def _engine_tables(
    predictions: pd.DataFrame, events: pd.DataFrame, cfg: StudyConfig,
) -> dict[str, pd.DataFrame]:
    """Bảng metric theo động cơ cho mỗi quy trình, kèm SSE và số dòng để tính RMSE gộp."""
    if "procedure" not in predictions or "procedure" not in events:
        raise ValueError("Dự đoán và sự kiện cần định danh procedure")
    if predictions.procedure.isna().any() or events.procedure.isna().any():
        raise ValueError("Định danh procedure không được thiếu")
    if set(predictions.procedure) != set(events.procedure):
        raise ValueError("Định danh procedure của dự đoán và cảnh báo khác nhau")
    if "hgb_uncapped" not in set(predictions.procedure):
        raise ValueError("Cần dự đoán tham chiếu hgb_uncapped")
    tables: dict[str, pd.DataFrame] = {}
    reference: pd.DataFrame | None = None
    for procedure, frame in predictions.groupby("procedure", sort=True):
        scored = _validate_predictions(frame)
        keys_truth = scored.loc[:, list(KEY) + ["y_true"]]
        if reference is None:
            reference = keys_truth
        else:
            try:
                pd.testing.assert_frame_equal(keys_truth, reference, check_exact=True)
            except AssertionError as error:
                raise ValueError("Quy trình ghép cặp cần cùng khóa và nhãn gốc") from error
        policy_events = events.loc[events.procedure.eq(procedure)].copy()
        if (policy_events.empty or "engine_id" not in policy_events
                or policy_events.engine_id.isna().any() or policy_events.engine_id.duplicated().any()
                or set(policy_events.engine_id) != set(scored.engine_id)):
            raise ValueError("Sự kiện cảnh báo phải khớp đúng một lần mỗi động cơ dự đoán")
        _require_finite(policy_events, ("on_time", "early_alarm_burden", "n_transitions", "n_retractions"))
        if (not policy_events.on_time.isin((False, True)).all()
                or not policy_events.early_alarm_burden.between(0., 1.).all()
                or policy_events.n_transitions.lt(0).any() or policy_events.n_retractions.lt(0).any()):
            raise ValueError("Tỷ lệ cảnh báo, mức báo sớm hoặc số chuyển trạng thái không hợp lệ")
        table = audit._engine_readout(scored, policy_events, cfg.audit_config()).sort_index()
        sufficient = scored.groupby("engine_id", sort=True).agg(
            trajectory_sse=("squared_error", "sum"), trajectory_n_rows=("squared_error", "size"),
            trajectory_mean_engine_nasa=("nasa_score_component", "mean"),
        )
        table = table.join(sufficient)
        _require_finite(table, PAIRED_ENGINE_METRICS + ("trajectory_sse", "trajectory_n_rows"))
        if table.trajectory_n_rows.le(0).any():
            raise ValueError("Mỗi động cơ phải có dòng quỹ đạo")
        tables[str(procedure)] = table
    return tables


def _paired_table(
    baseline: pd.DataFrame, candidate: pd.DataFrame, cfg: StudyConfig,
) -> pd.DataFrame:
    """Bootstrap ghép cặp theo động cơ cho chênh lệch ứng viên − baseline của mỗi metric."""
    if not baseline.index.equals(candidate.index):
        raise ValueError("Metric cặp động cơ cần cùng ID động cơ theo đúng thứ tự")
    n_engines = len(baseline)
    indices = np.random.default_rng(cfg.seed).integers(
        0, n_engines, size=(cfg.bootstrap_resamples, n_engines),
    )
    rows: list[dict[str, Any]] = []
    for metric in ("trajectory_rmse",) + PAIRED_ENGINE_METRICS:
        if metric == "trajectory_rmse":
            old_sse = baseline.trajectory_sse.to_numpy(dtype="float64")
            new_sse = candidate.trajectory_sse.to_numpy(dtype="float64")
            old_n = baseline.trajectory_n_rows.to_numpy(dtype="float64")
            new_n = candidate.trajectory_n_rows.to_numpy(dtype="float64")
            before = float(np.sqrt(old_sse.sum() / old_n.sum()))
            after = float(np.sqrt(new_sse.sum() / new_n.sum()))
            draws = (np.sqrt(new_sse[indices].sum(axis=1) / new_n[indices].sum(axis=1))
                     - np.sqrt(old_sse[indices].sum(axis=1) / old_n[indices].sum(axis=1)))
        else:
            old = baseline[metric].to_numpy(dtype="float64")
            new = candidate[metric].to_numpy(dtype="float64")
            before, after = float(old.mean()), float(new.mean())
            draws = (new - old)[indices].mean(axis=1)
        rows.append({
            "metric": metric, "baseline": before, "candidate": after, "delta": after - before,
            "ci_lower": float(np.quantile(draws, .025)),
            "ci_upper": float(np.quantile(draws, .975)),
            "n_engines": n_engines, "resamples": cfg.bootstrap_resamples,
            "seed": cfg.seed, "interval_scope": INTERVAL_SCOPE,
        })
    return pd.DataFrame(rows)


def _comparisons_from_tables(
    tables: dict[str, pd.DataFrame], cfg: StudyConfig,
) -> pd.DataFrame:
    """So sánh từng nhánh và selected_pipeline với HGB tham chiếu."""
    pairs = [("hgb_uncapped", arm) for arm in ARMS
             if arm in tables and arm != "hgb_uncapped"]
    if "hgb_cap125" in tables and "lgbm_cap125" in tables:
        pairs.append(("hgb_cap125", "lgbm_cap125"))
    if "selected_pipeline" in tables:
        pairs.extend((reference, "selected_pipeline")
                     for reference in ("hgb_uncapped", "hgb_cap125") if reference in tables)
    return pd.concat([
        _paired_table(tables[reference], tables[procedure], cfg)
        .assign(reference=reference, procedure=procedure)
        for reference, procedure in pairs
    ], ignore_index=True)


def paired_comparisons(
    predictions: pd.DataFrame, events: pd.DataFrame, cfg: StudyConfig,
) -> pd.DataFrame:
    """Bootstrap ghép cặp giữa các quy trình; bắt buộc có hgb_uncapped."""
    return _comparisons_from_tables(_engine_tables(predictions, events, cfg), cfg)


def _gate_controls(comparison: pd.DataFrame) -> dict[str, bool]:
    """Chuyển cận CI 95% thành năm điều kiện chấp nhận."""
    rows = comparison.set_index("metric")
    return {
        "trajectory_rmse_improved": bool(rows.loc["trajectory_rmse", "ci_upper"] < 0.),
        "endpoint_mae_not_worse": bool(rows.loc["endpoint_mean_engine_mae", "ci_upper"] <= 0.),
        "endpoint_nasa_not_worse": bool(rows.loc["endpoint_mean_engine_nasa", "ci_upper"] <= 0.),
        "warning_on_time_not_worse": bool(rows.loc["on_time_rate", "ci_lower"] >= 0.),
        "warning_burden_not_worse": bool(rows.loc["early_alarm_burden", "ci_upper"] <= 0.),
    }


def write_readouts(
    predictions: pd.DataFrame, events: pd.DataFrame, cfg: StudyConfig, output: str | Path,
) -> dict[str, Any]:
    """
    Ghi metric outer và quyết định vào output; đối chiếu selected_pipeline với
    hgb_uncapped, hgb_cap125 chỉ báo cáo, smoke không được chấp nhận.
    """
    expected = set(ARMS) | {"selected_pipeline"}
    if "procedure" not in predictions or set(predictions.procedure) != expected:
        raise ValueError("Metric outer cần bốn nhánh và pipeline được chọn trên inner")
    tables = _engine_tables(predictions, events, cfg)
    summaries: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for procedure, frame in predictions.groupby("procedure", sort=True):
        scored = _validate_predictions(frame)
        policy_events = events.loc[events.procedure.eq(procedure)]
        summaries.append({"procedure": procedure,
                          **audit._model_readout(scored, cfg.audit_config()),
                          **audit._warning_readout(policy_events)})
        scored["rul_region"] = pd.cut(scored.y_true, [-1, 30, 60, 100, 150, np.inf],
                                       labels=["0-30", "31-60", "61-100", "101-150", ">150"])
        for region, group in scored.groupby("rul_region", observed=False):
            if group.empty:
                continue
            region_rows.append({
                "procedure": procedure, "rul_region": str(region), "n_rows": len(group),
                "n_engines": group.engine_id.nunique(),
                "mean_window_mae": float(group.absolute_error.mean()),
                "mean_engine_mae": float(group.groupby("engine_id").absolute_error.mean().mean()),
                "signed_bias": float((group.y_pred - group.y_true).mean()),
                "rmse": float(np.sqrt(group.squared_error.mean())),
            })
    comparisons = _comparisons_from_tables(tables, cfg)
    selected = comparisons.loc[
        comparisons.reference.eq("hgb_uncapped") & comparisons.procedure.eq("selected_pipeline")
    ]
    controls = _gate_controls(selected)
    capped = comparisons.loc[
        comparisons.reference.eq("hgb_cap125") & comparisons.procedure.eq("selected_pipeline")
    ].set_index("metric")
    capped_controls = {
        "endpoint_mae_not_worse": bool(capped.loc["endpoint_mean_engine_mae", "ci_upper"] <= 0.),
        "endpoint_nasa_not_worse": bool(capped.loc["endpoint_mean_engine_nasa", "ci_upper"] <= 0.),
    }
    scientific = bool(cfg.scientific)
    gate_rmse = selected.loc[selected.metric.eq("trajectory_rmse")].iloc[0]
    decision = {
        "full_protocol_configuration": scientific,
        "promotion_gate_met": bool(scientific and all(controls.values())),
        "controls": controls,
        "outer_trajectory_rmse_delta": float(gate_rmse.delta),
        "outer_trajectory_rmse_ci_lower": float(gate_rmse.ci_lower),
        "outer_trajectory_rmse_ci_upper": float(gate_rmse.ci_upper),
        "capped_reference_endpoint_controls": capped_controls,
        "capped_reference_endpoint_controls_met": bool(all(capped_controls.values())),
        "capped_reference_reporting_only": True,
        "gate": "selected_pipeline vs hgb_uncapped: paired trajectory RMSE upper95% CI<0; "
                "endpoint engine MAE/NASA and early burden upper95% CI<=0; "
                "on-time rate lower95% CI>=0; no test gating",
        "selection_source": "inner grouped CV only; no procedure selection from outer readouts",
        "historical_baseline_overwritten": False,
        "evaluation_limit": "exploratory follow-up using historically inspected engine folds; "
                            "retrospective endpoints; fixed-prediction conditional bootstrap",
        "bootstrap_scope": INTERVAL_SCOPE,
    }
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(destination / "aggregate_metrics.csv", index=False)
    pd.concat([table.assign(procedure=name).reset_index() for name, table in tables.items()],
              ignore_index=True).to_csv(destination / "engine_metrics.csv", index=False)
    comparisons.to_csv(destination / "paired_comparison.csv", index=False)
    pd.DataFrame(region_rows).to_csv(destination / "region_metrics.csv", index=False)
    audit._write_json(destination / "decision.json", decision)
    return decision


