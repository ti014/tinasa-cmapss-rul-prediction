"""Independently verify immutable HGB multi-scale study evidence.

Metrics, causal feature values, warning policies, inner choices and bootstrap
math are reconstructed here without calling training or analysis code. The
only new artifact written by the CLI is verification.json; an existing
receipt is never overwritten. Raw test-label files are never opened.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.data.contracts import KEY, OBSERVATION_COLUMNS, SENSORS
from cmapss.modeling.multiscale_comparison.config import StudyConfig
from cmapss.modeling.multiscale_comparison.contracts import FEATURE_COLUMNS_BY_ARM
from cmapss.modeling.multiscale_comparison.estimators import FittedHGB

ARMS = ("hgb_baseline169", "hgb_multiscale10_30", "hgb_initial_profile42")
PROCEDURES = (*ARMS, "selected_pipeline")
ENGINE_METRICS = (
    "endpoint_mean_engine_mae", "endpoint_mean_engine_nasa",
    "trajectory_mean_engine_mae", "trajectory_mean_engine_nasa",
    "on_time_rate", "early_alarm_burden", "mean_transitions", "mean_retractions",
)
INTERVAL_SCOPE = (
    "paired engines conditional on fixed outer predictions; duplicate sampled engines "
    "retain multiplicity; excludes refitting variability"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes repository: {resolved}")
    return resolved


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def _same(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str] | tuple[str, ...],
          *, exact: bool = False, subset: bool = False) -> bool:
    if len(actual) != len(expected) or not set(expected.columns).issubset(actual.columns):
        return False
    if not subset and set(actual.columns) != set(expected.columns):
        return False
    try:
        columns = list(expected.columns)
        left = actual.loc[:, columns].sort_values(list(keys), kind="mergesort").reset_index(drop=True)
        right = expected.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
        left, right = left.where(left.notna(), np.nan), right.where(right.notna(), np.nan)
        pd.testing.assert_frame_equal(
            left, right,
            check_dtype=False, check_exact=exact,
            atol=0.0 if exact else 1e-8, rtol=0.0 if exact else 1e-10,
        )
    except (AssertionError, KeyError, TypeError, ValueError):
        return False
    return True


def _score(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"y_true", "y_pred"}.issubset(frame):
        raise ValueError("nonempty raw truth and predictions required")
    result = frame.copy()
    truth, prediction = (result[column].to_numpy(dtype="float64") for column in ("y_true", "y_pred"))
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or (truth < 0).any():
        raise ValueError("finite predictions and raw nonnegative RUL required")
    error = prediction - truth
    result["absolute_error"] = np.abs(error)
    result["squared_error"] = error ** 2
    result["nasa_score_component"] = np.where(error < 0, np.expm1(-error / 13), np.expm1(error / 10))
    if "signed_error" in result:
        result["signed_error"] = error
    if not np.isfinite(result[["absolute_error", "squared_error", "nasa_score_component"]]).all().all():
        raise ValueError("nonfinite error scores")
    return result


def _endpoints(frame: pd.DataFrame, fractions: list[float] | tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for (_, engine), group in frame.groupby(["split", "engine_id"], sort=True):
        lifetime = group.cycle.to_numpy(dtype="float64") + group.y_true.to_numpy(dtype="float64")
        if not np.isfinite(lifetime).all() or not np.all(lifetime == lifetime[0]):
            raise ValueError(f"inconsistent raw lifetime for engine {engine}")
        for fraction in fractions:
            requested = int(np.floor(float(fraction) * lifetime[0]))
            origin = max(int(group.cycle.min()), requested)
            chosen = group[group.cycle.eq(origin)]
            if len(chosen) != 1:
                raise ValueError("endpoint key missing or duplicated")
            row = chosen.iloc[0].to_dict()
            row.update(endpoint_fraction=float(fraction), requested_cycle=requested,
                       origin_clamped_to_available_prefix=origin != requested)
            rows.append(row)
    return pd.DataFrame(rows)


def _metrics(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    scored = _score(frame)
    engines = scored.groupby("engine_id", sort=True)
    return {
        f"{prefix}_n_rows": len(scored), f"{prefix}_n_engines": scored.engine_id.nunique(),
        f"{prefix}_mean_window_mae": float(scored.absolute_error.mean()),
        f"{prefix}_mean_engine_mae": float(engines.absolute_error.mean().mean()),
        f"{prefix}_rmse": float(np.sqrt(scored.squared_error.mean())),
        f"{prefix}_nasa_score": float(scored.nasa_score_component.sum()),
        f"{prefix}_mean_engine_nasa": float(engines.nasa_score_component.mean().mean()),
    }


def _model_metrics(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    return {**_metrics(_endpoints(frame, cfg["endpoint_fractions"]), "endpoint"),
            **_metrics(frame, "trajectory")}


def _warning_events(frame: pd.DataFrame, policy: dict[str, int], lead_requirement: int) -> pd.DataFrame:
    threshold, persistence = int(policy["threshold"]), int(policy["persistence"])
    rows = []
    for engine, block in frame.groupby("engine_id", sort=True):
        group = block.sort_values("cycle", kind="mergesort")
        cycles, truth = group.cycle.to_numpy(dtype="int64"), group.y_true.to_numpy(dtype="float64")
        if (not group.split.eq("train").all() or not np.all(np.diff(cycles) == 1)
                or not np.array_equal(truth, cycles[-1] - cycles)
                or not np.isfinite(group.y_pred).all()):
            raise ValueError("warnings require contiguous complete train trajectories")
        below = group.y_pred.to_numpy() <= threshold
        alarm = np.zeros(len(group), dtype=bool)
        if persistence <= len(group):
            alarm[persistence - 1:] = np.lib.stride_tricks.sliding_window_view(below, persistence).all(axis=1)
        hits = np.flatnonzero(alarm)
        lead = float(truth[hits[0]]) if len(hits) else np.nan
        eligible = truth > lead_requirement
        transitions = np.diff(np.r_[False, alarm].astype("int64"))
        rows.append({
            "engine_id": int(engine), "threshold": threshold, "persistence": persistence,
            "first_warning_cycle": int(cycles[hits[0]]) if len(hits) else -1,
            "lead_time_cycles": lead, "missed_warning": not bool(len(hits)),
            "on_time": bool(len(hits) and lead >= lead_requirement),
            "early_alarm_burden": float(alarm[eligible].mean()) if eligible.any() else np.nan,
            "n_pre_danger_rows": int(eligible.sum()), "n_alarm_rows": int(alarm.sum()),
            "n_transitions": int(np.count_nonzero(transitions)), "n_retractions": int((transitions == -1).sum()),
            "n_alarm_episodes": int((transitions == 1).sum()), "n_rows": len(group),
            "lead_requirement": lead_requirement,
        })
    return pd.DataFrame(rows)


def _warnings(events: pd.DataFrame) -> dict[str, Any]:
    lead = events.lead_time_cycles.dropna()
    return {
        "on_time_rate": float(events.on_time.mean()), "on_time_count": int(events.on_time.sum()),
        "late_or_missed_rate": float((~events.on_time).mean()), "missed_warning_count": int(events.missed_warning.sum()),
        "early_alarm_burden": float(events.early_alarm_burden.mean()),
        "mean_transitions": float(events.n_transitions.mean()), "mean_retractions": float(events.n_retractions.mean()),
        "engines_with_retractions": int(events.n_retractions.gt(0).sum()),
        "median_lead_time": float(lead.median()) if len(lead) else None,
        "min_lead_time": float(lead.min()) if len(lead) else None,
        "max_lead_time": float(lead.max()) if len(lead) else None, "n_engines": len(events),
    }


def _policy(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, int], pd.DataFrame]:
    grid = pd.DataFrame([
        {"threshold": int(threshold), "persistence": int(persistence),
         **_warnings(_warning_events(frame, {"threshold": threshold, "persistence": persistence}, cfg["lead_requirement"]))}
        for threshold in cfg["thresholds"] for persistence in cfg["persistences"]
    ])
    valid = grid.dropna(subset=["on_time_rate", "early_alarm_burden"])
    if valid.empty:
        raise ValueError("no evaluable warning policies")
    feasible = valid[valid.on_time_rate.ge(cfg["target_on_time_rate"])]
    if feasible.empty:
        winner = valid.sort_values(["on_time_rate", "early_alarm_burden", "threshold", "persistence"],
                                  ascending=[False, True, True, True], kind="mergesort").iloc[0]
    else:
        winner = feasible.sort_values(["early_alarm_burden", "threshold", "persistence"], kind="mergesort").iloc[0]
    policy = {"threshold": int(winner.threshold), "persistence": int(winner.persistence)}
    grid["selected"] = grid.threshold.eq(policy["threshold"]) & grid.persistence.eq(policy["persistence"])
    grid["inner_target_met"] = grid.on_time_rate.ge(cfg["target_on_time_rate"])
    return policy, grid


def _choose(records: pd.DataFrame, cfg: dict[str, Any]) -> tuple[str, dict[str, dict[str, bool]]]:
    if "procedure" not in records or set(records.procedure) != set(ARMS) or records.procedure.duplicated().any():
        raise ValueError("inner selection requires exactly three fixed arms")
    numeric_columns = ["trajectory_rmse", "endpoint_mean_engine_mae", "endpoint_mean_engine_nasa", "on_time_rate", "early_alarm_burden"]
    if not np.isfinite(records.loc[:, numeric_columns].to_numpy(dtype="float64")).all():
        raise ValueError("finite inner selection metrics required")
    table = records.set_index("procedure", drop=False)
    baseline = table.loc[ARMS[0]]
    controls = {}
    for arm in ARMS:
        row = table.loc[arm]
        numeric = bool(row.endpoint_mean_engine_mae <= baseline.endpoint_mean_engine_mae + cfg["endpoint_mae_margin"]
                       and row.endpoint_mean_engine_nasa <= baseline.endpoint_mean_engine_nasa + cfg["nasa_score_margin"])
        warning = bool(row.on_time_rate >= baseline.on_time_rate - cfg["warning_rate_margin"]
                       and row.early_alarm_burden <= baseline.early_alarm_burden + cfg["early_alarm_burden_margin"])
        improves = bool(arm != ARMS[0] and row.trajectory_rmse < baseline.trajectory_rmse)
        controls[arm] = {"endpoint_controls_met": numeric, "warning_controls_met": warning,
                         "trajectory_improves": improves, "eligible": numeric and warning and improves}
    eligible = [arm for arm in ARMS[1:] if controls[arm]["eligible"]]
    if not eligible:
        return ARMS[0], controls
    winner = table.loc[eligible].reset_index(drop=True).sort_values(
        ["trajectory_rmse", "endpoint_mean_engine_mae", "procedure"], kind="mergesort").iloc[0]
    return str(winner.procedure), controls


def _engine_table(frame: pd.DataFrame, events: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    endpoint, trajectory = _score(_endpoints(frame, cfg["endpoint_fractions"])), _score(frame)
    table = pd.DataFrame({
        "endpoint_mean_engine_mae": endpoint.groupby("engine_id").absolute_error.mean(),
        "endpoint_mean_engine_nasa": endpoint.groupby("engine_id").nasa_score_component.mean(),
        "trajectory_mean_engine_mae": trajectory.groupby("engine_id").absolute_error.mean(),
        "trajectory_mean_engine_nasa": trajectory.groupby("engine_id").nasa_score_component.mean(),
        "low_rul_engine_mae": trajectory[trajectory.y_true.le(30)].groupby("engine_id").absolute_error.mean(),
        "trajectory_sse": trajectory.groupby("engine_id").squared_error.sum(),
        "trajectory_n_rows": trajectory.groupby("engine_id").squared_error.size(),
    }).sort_index()
    warning = events.set_index("engine_id").rename(columns={"on_time": "on_time_rate", "n_transitions": "mean_transitions", "n_retractions": "mean_retractions"})
    return table.join(warning.loc[:, ["on_time_rate", "early_alarm_burden", "mean_transitions", "mean_retractions"]])


def _paired(before: pd.DataFrame, after: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if not before.index.equals(after.index) or before.empty:
        raise ValueError("paired bootstrap requires identical ordered engine indices")
    size = len(before)
    indices = np.random.default_rng(cfg["seed"]).integers(0, size, size=(cfg["bootstrap_resamples"], size))
    rows = []
    for metric in ("trajectory_rmse", *ENGINE_METRICS):
        if metric == "trajectory_rmse":
            old_sse, new_sse = (table.trajectory_sse.to_numpy(dtype="float64") for table in (before, after))
            old_n, new_n = (table.trajectory_n_rows.to_numpy(dtype="float64") for table in (before, after))
            old, new = float(np.sqrt(old_sse.sum() / old_n.sum())), float(np.sqrt(new_sse.sum() / new_n.sum()))
            samples = np.sqrt(new_sse[indices].sum(axis=1) / new_n[indices].sum(axis=1)) - np.sqrt(old_sse[indices].sum(axis=1) / old_n[indices].sum(axis=1))
        else:
            old_values, new_values = (table[metric].to_numpy(dtype="float64") for table in (before, after))
            old, new = float(old_values.mean()), float(new_values.mean())
            samples = (new_values - old_values)[indices].mean(axis=1)
        rows.append({"metric": metric, "baseline": old, "candidate": new, "delta": new - old,
                     "ci_lower": float(np.quantile(samples, .025)), "ci_upper": float(np.quantile(samples, .975)),
                     "n_engines": size, "resamples": cfg["bootstrap_resamples"], "seed": cfg["seed"], "interval_scope": INTERVAL_SCOPE})
    return pd.DataFrame(rows)


def _gate(paired: pd.DataFrame) -> dict[str, bool]:
    table = paired.set_index("metric")
    return {
        "trajectory_rmse_improved": bool(table.loc["trajectory_rmse", "ci_upper"] < 0),
        "endpoint_mae_not_worse": bool(table.loc["endpoint_mean_engine_mae", "ci_upper"] <= 0),
        "endpoint_nasa_not_worse": bool(table.loc["endpoint_mean_engine_nasa", "ci_upper"] <= 0),
        "warning_on_time_not_worse": bool(table.loc["on_time_rate", "ci_lower"] >= 0),
        "warning_burden_not_worse": bool(table.loc["early_alarm_burden", "ci_upper"] <= 0),
    }


def _causal_expected(observations: pd.DataFrame) -> pd.DataFrame:
    """Recompute all337 union features directly from raw sensor arrays."""
    if tuple(observations.columns) != OBSERVATION_COLUMNS or observations.empty or observations.duplicated(list(KEY)).any():
        raise ValueError("unique label-free observation schema required")
    frames = []
    for _, group in observations.sort_values(list(KEY), kind="mergesort").groupby(["split", "engine_id"], sort=True):
        cycles = group.cycle.to_numpy(dtype="int64")
        if cycles[0] != 1 or not np.all(np.diff(cycles) == 1) or len(cycles) < 30:
            raise ValueError("complete first30 observation prefix required")
        values = group.loc[:, list(SENSORS)].to_numpy(dtype="float64")
        initial_mean, initial_std = values[:30].mean(axis=0), values[:30].std(axis=0, ddof=1)
        data = {name: group[name].to_numpy()[29:] for name in KEY}
        for window, infix in ((30, "__"), (10, "__w10__")):
            windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
            x = np.arange(1, window + 1, dtype="float64")
            x -= x.mean()
            means = windows.mean(axis=2)
            stats = {"last": windows[:, :, -1], "mean": means, "std": windows.std(axis=2, ddof=1),
                     "min": windows.min(axis=2), "max": windows.max(axis=2),
                     "slope": np.einsum("esw,w->es", windows - means[:, :, None], x) / np.dot(x, x)}
            for i, sensor in enumerate(SENSORS):
                for stat, array in stats.items():
                    data[f"{sensor}{infix}{stat}"] = array[30 - window:, i]
                if window == 30:
                    data[f"{sensor}__from_initial_last"] = stats["last"][:, i] - initial_mean[i]
                    data[f"{sensor}__from_initial_mean"] = stats["mean"][:, i] - initial_mean[i]
                    data[f"{sensor}__initial30_mean"] = np.full(len(cycles) - 29, initial_mean[i])
                    data[f"{sensor}__initial30_std"] = np.full(len(cycles) - 29, initial_std[i])
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def _hashes(root: Path, mapping: dict[str, str]) -> bool:
    return all(_resolve(root, name).is_file() and _sha(_resolve(root, name)) == digest for name, digest in mapping.items())


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(repr(tuple(frame.columns)).encode("utf-8"))
    digest.update(repr(tuple(str(dtype) for dtype in frame.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _params(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"loss": "squared_error", "learning_rate": cfg["hgb_learning_rate"], "max_iter": cfg["hgb_max_iter"],
            "max_leaf_nodes": cfg["hgb_max_leaf_nodes"], "max_depth": None, "min_samples_leaf": cfg["hgb_min_samples_leaf"],
            "l2_regularization": 0.0, "early_stopping": False, "random_state": cfg["seed"]}


def _row_matches(actual: dict[str, Any] | pd.Series, expected: dict[str, Any]) -> bool:
    for name, value in expected.items():
        if name not in actual:
            return False
        observed = actual[name]
        if value is None or isinstance(value, float) and np.isnan(value):
            if observed is not None and not pd.isna(observed):
                return False
        elif isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
            if not np.isclose(float(observed), float(value), atol=1e-8, rtol=1e-10):
                return False
        elif observed != value:
            return False
    return True


def _source_closure(root: Path) -> set[str]:
    pending = [path.relative_to(root).as_posix() for path in (root / "src/cmapss/modeling/multiscale_comparison").glob("*.py")]
    pending.extend(["scripts/run_multiscale_study.py", "experiments/MULTISCALE_FEATURE_PROTOCOL.md"])
    found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        if not name.startswith("src/") or not name.endswith(".py"):
            continue
        path = root / name
        package = Path(name).parent.relative_to("src").as_posix().replace("/", ".")
        parent = Path(name).parent
        while parent != Path("src"):
            initializer = (parent / "__init__.py").as_posix()
            if (root / initializer).is_file():
                pending.append(initializer)
            parent = parent.parent
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = ".".join(package.split(".")[:len(package.split(".")) - node.level + 1]) if node.level else ""
                module = ".".join(filter(None, (base, node.module)))
                modules = [module, *(f"{module}.{item.name}" for item in node.names)]
            for module in modules:
                if not module.startswith("cmapss"):
                    continue
                candidate = Path("src") / Path(*module.split("."))
                for local in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                    if (root / local).is_file():
                        pending.append(local.as_posix())
    return found


def _verify_provenance(root: Path, folder: Path, manifest: dict[str, Any], checks: dict[str, bool]) -> None:
    identity = manifest["identity"]
    checks["identity_digest_matches_frozen_manifest"] = manifest["source_identity"] == hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    sources, inputs = identity.get("source_sha256", {}), identity.get("inputs_sha256", {})
    checks["current_source_inputs_lock_and_project_unchanged"] = (
        bool(sources) and bool(inputs) and _hashes(root, sources) and _hashes(root, inputs)
        and _sha(root / "uv.lock") == identity.get("dependency_lock_sha256")
        and _sha(root / "pyproject.toml") == identity.get("project_sha256"))
    checks["local_import_closure_and_entrypoint_frozen"] = _source_closure(root).issubset(sources)
    checks["exact_source_input_and_dependency_snapshots"] = (
        bool(sources) and bool(inputs) and _hashes(folder / "source_snapshot", sources) and _hashes(folder / "input_snapshot", inputs)
        and (folder / "source_snapshot/uv.lock").is_file()
        and (folder / "source_snapshot/pyproject.toml").is_file()
        and _sha(folder / "source_snapshot/uv.lock") == identity.get("dependency_lock_sha256")
        and _sha(folder / "source_snapshot/pyproject.toml") == identity.get("project_sha256"))
    installed = {name: (platform.python_version() if name == "python" else version("scikit-learn" if name == "sklearn" else name))
                 for name in identity.get("versions", {})}
    checks["numeric_runtime_versions_match_identity"] = bool(installed) and installed == identity.get("versions")
    checks["frozen_matrices_and_outer_assignments_unchanged"] = (
        _sha(folder / "frozen_train_features.parquet") == identity.get("frozen_train_sha256", identity.get("matrix_sha256"))
        and _sha(folder / "frozen_test_features.parquet") == identity.get("frozen_test_sha256")
        and _sha(folder / "outer_assignments.csv") == identity.get("outer_assignments_sha256"))
    specifications = identity.get("feature_specs", [])
    arm_ids = identity.get("arms", [spec["arm_id"] for spec in specifications])
    feature_columns = identity.get("feature_columns", {spec["arm_id"]: spec["feature_columns"] for spec in specifications})
    checks["fixed_three_label_free_feature_specs"] = (
        arm_ids == list(ARMS) and feature_columns == {arm: list(FEATURE_COLUMNS_BY_ARM[arm]) for arm in ARMS}
        and [len(FEATURE_COLUMNS_BY_ARM[arm]) for arm in ARMS] == [169, 295, 211]
        and all(not {"rul", "y_true", "lifetime", "engine_id", "split"}.intersection(columns) for columns in FEATURE_COLUMNS_BY_ARM.values()))


def _verify_features(root: Path, folder: Path, identity: dict[str, Any], train: pd.DataFrame,
                     test: pd.DataFrame, checks: dict[str, bool]) -> None:
    input_names = identity.get("inputs_sha256", {"outputs/etl/silver/observations.parquet": identity.get("observations_sha256"),
                                                "outputs/etl/silver/train_targets.parquet": identity.get("targets_sha256")})
    observation_names = [name for name in input_names if Path(name).name == "observations.parquet"]
    target_names = [name for name in input_names if Path(name).name == "train_targets.parquet"]
    if len(observation_names) != 1 or len(target_names) != 1:
        raise ValueError("one frozen observation and train-target input required")
    input_root = folder / "input_snapshot" if (folder / "input_snapshot").is_dir() else root
    observations = pd.read_parquet(input_root / observation_names[0])
    targets = pd.read_parquet(input_root / target_names[0])
    expected = _causal_expected(observations)
    columns = list(dict.fromkeys((*KEY, *(column for arm in ARMS for column in FEATURE_COLUMNS_BY_ARM[arm]))))
    checks["all_337_union_features_recomputed_from_causal_sensor_windows"] = all(
        _same(frame.loc[:, columns], expected.loc[expected.split.eq(split), columns], list(KEY))
        for split, frame in (("train", train), ("test", test)))
    expected_train = expected[expected.split.eq("train")].loc[:, list(KEY)].merge(targets, on=list(KEY), validate="one_to_one")
    checks["raw_train_truth_and_common_cycle30_panel"] = (
        _same(train.loc[:, [*KEY, "rul"]], expected_train, list(KEY), exact=True)
        and train.cycle.ge(30).all() and test.cycle.ge(30).all()
        and train.split.eq("train").all() and test.split.eq("test").all()
        and not train.duplicated(list(KEY)).any() and not test.duplicated(list(KEY)).any()
        and (train.cycle + train.rul).groupby(train.engine_id).nunique().eq(1).all())
    checks["test_feature_panel_has_no_label_or_lifetime_column"] = not {"rul", "y_true", "lifetime", "n_cycles"}.intersection(test.columns)


def _verify_cache(folder: Path, identity: dict[str, Any], source_identity: str,
                  train: pd.DataFrame, cfg: dict[str, Any], checks: dict[str, bool]) -> dict[tuple, tuple[Path, pd.DataFrame]]:
    columns = list(dict.fromkeys((*KEY, *(column for arm in ARMS for column in FEATURE_COLUMNS_BY_ARM[arm]), "rul")))
    source_hash = _frame_hash(train.loc[:, columns])
    records = {}
    valid = True
    for path in sorted((folder / "fit_cache").glob("*.json")):
        metadata = _json(path)
        task = metadata["identity"]
        arm = task["arm_id"]
        fit_ids, validation_ids = tuple(task["fit_engine_ids"]), tuple(task["validation_engine_ids"])
        key = (arm, fit_ids, validation_ids)
        parquet = path.with_suffix(".parquet")
        valid = valid and arm in ARMS and key not in records
        valid = valid and bool(fit_ids) and bool(validation_ids) and not set(fit_ids).intersection(validation_ids)
        valid = valid and fit_ids == tuple(sorted(set(fit_ids))) and validation_ids == tuple(sorted(set(validation_ids)))
        valid = valid and task["config"] == cfg and task["source_identity"] == source_identity and task["source_hash"] == source_hash
        valid = valid and task["feature_columns"] == list(FEATURE_COLUMNS_BY_ARM.get(arm, ()))
        cache_key = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
        valid = valid and metadata["cache_key"] == path.stem == cache_key and metadata["prediction_sha256"] == _sha(parquet)
        prediction = pd.read_parquet(parquet)
        expected_truth = train[train.engine_id.isin(validation_ids)].loc[:, [*KEY, "rul"]].rename(columns={"rul": "y_true"})
        valid = valid and _same(prediction, expected_truth, list(KEY), exact=True, subset=True)
        valid = valid and prediction.candidate_id.eq(arm).all() and _same(prediction, _score(prediction), list(KEY))
        valid = valid and metadata["effective_parameters"] == _params(cfg)
        counts = train[train.engine_id.isin(fit_ids)].groupby("engine_id").size()
        weight = metadata["weight_metadata"]
        valid = valid and _row_matches(weight, {"scheme": "engine_total_one", "total_weight": len(fit_ids),
                                               "n_rows": int(counts.sum()), "n_engines": len(fit_ids),
                                               "min_row_weight": 1 / counts.max(), "max_row_weight": 1 / counts.min(),
                                               "fit_threads": cfg["fit_threads"], "target_cap": None})
        records[key] = (parquet, prediction)
    checks["checksum_bound_cache_raw_truth_scores_and_fixed_hgb_weights"] = valid
    expected_count = ((cfg["outer_splits"] + 1) * cfg["inner_splits"] + cfg["outer_splits"]) * len(ARMS)
    assignment_path = folder / "outer_assignments.csv"
    if assignment_path.is_file():
        assignments = _csv(assignment_path)
        tasks = set()
        all_ids = set(map(int, train.engine_id.unique()))
        for outer_fold in range(cfg["outer_splits"] + 1):
            validation_ids = tuple(sorted(map(int, assignments.loc[assignments.outer_fold.eq(outer_fold), "engine_id"])))
            cohort_ids = all_ids - set(validation_ids) if outer_fold else all_ids
            cohort = train[train.engine_id.isin(cohort_ids)].reset_index(drop=True)
            if outer_fold:
                tasks.update((arm, tuple(sorted(cohort_ids)), validation_ids) for arm in ARMS)
            for fit_idx, validation_idx in GroupKFold(cfg["inner_splits"]).split(cohort, cohort.rul, cohort.engine_id):
                fit_ids = tuple(sorted(map(int, cohort.iloc[fit_idx].engine_id.unique())))
                held_ids = tuple(sorted(map(int, cohort.iloc[validation_idx].engine_id.unique())))
                tasks.update((arm, fit_ids, held_ids) for arm in ARMS)
        expected_count = len(tasks)
    checks["complete_unique_cv_fit_count"] = len(records) == expected_count
    return records


def _verify_inner(folder: Path, train: pd.DataFrame, assignments: pd.DataFrame,
                  records: dict[tuple, tuple[Path, pd.DataFrame]], cfg: dict[str, Any],
                  source_identity: str, checks: dict[str, bool]) -> tuple[dict[str, dict[str, Any]], set[tuple]]:
    all_ids = set(map(int, train.engine_id.unique()))
    selections, used = {}, set()
    valid_scopes = valid_metrics = valid_policy = valid_choice = True
    for outer_fold in range(cfg["outer_splits"] + 1):
        name = f"outer_{outer_fold}" if outer_fold else "final_inner"
        stage = folder / "stages" / name
        selection = _json(stage / "selection.json")
        selections[name] = selection
        cohort_ids = all_ids - set(assignments.loc[assignments.outer_fold.eq(outer_fold), "engine_id"]) if outer_fold else all_ids
        cohort = train[train.engine_id.isin(cohort_ids)].reset_index(drop=True)
        declared = _csv(stage / "inner_assignments.csv")
        expected_rows, pieces = [], {arm: [] for arm in ARMS}
        for fold, (fit_idx, validation_idx) in enumerate(GroupKFold(cfg["inner_splits"]).split(cohort, cohort.rul, cohort.engine_id), 1):
            fit_ids = tuple(sorted(map(int, cohort.iloc[fit_idx].engine_id.unique())))
            validation_ids = tuple(sorted(map(int, cohort.iloc[validation_idx].engine_id.unique())))
            expected_rows.extend({"inner_fold": fold, "engine_id": engine,
                                  **({"outer_fold": outer_fold} if "outer_fold" in declared else {})} for engine in validation_ids)
            for arm in ARMS:
                key = (arm, fit_ids, validation_ids)
                if key not in records:
                    raise ValueError(f"declared inner task missing: {name}/{arm}/{fold}")
                used.add(key)
                pieces[arm].append(records[key][1])
        valid_scopes = valid_scopes and _same(declared, pd.DataFrame(expected_rows), ["engine_id"], exact=True)
        summaries, grids = [], []
        stored_summary = _csv(stage / "inner_arm_metrics.csv")
        stored_grid = _csv(stage / "inner_policy_metrics.csv")
        for arm in ARMS:
            frame = pd.concat(pieces[arm], ignore_index=True)
            policy, grid = _policy(frame, cfg)
            events = _warning_events(frame, policy, cfg["lead_requirement"])
            row = {"procedure": arm, **_model_metrics(frame, cfg), **_warnings(events)}
            for column, value in (("candidate_id", arm), ("numeric_controls_met", True),
                                  ("policy_threshold", policy["threshold"]), ("policy_persistence", policy["persistence"])):
                if column in stored_summary:
                    row[column] = value
            summaries.append(row)
            grid = grid.assign(procedure=arm)
            for column, value in (("candidate_id", arm), ("stage", name), ("outer_fold", outer_fold)):
                if column in stored_grid:
                    grid[column] = value
            grids.append(grid)
            choice = selection["arms"][arm]
            valid_metrics = valid_metrics and _row_matches(choice["inner_metrics"], row)
            valid_policy = valid_policy and choice["policy"] == policy and choice.get("candidate_id", arm) == arm
        summary = pd.DataFrame(summaries)
        valid_metrics = valid_metrics and _same(stored_summary, summary, ["procedure"])
        valid_policy = valid_policy and _same(stored_grid, pd.concat(grids, ignore_index=True), ["procedure", "threshold", "persistence"])
        winner, controls = _choose(summary, cfg)
        choice = selection["procedure"]
        valid_choice = valid_choice and choice["selected_procedure_id"] == choice["procedure_id"] == winner
        valid_choice = valid_choice and choice["arm_controls"] == controls and choice["fallback_used"] == (winner == ARMS[0])
        valid_choice = valid_choice and selection["source_identity"] == source_identity and selection["test_labels_read"] is False
        valid_choice = valid_choice and (selection.get("stage", name) == name) and (selection.get("outer_fold", outer_fold) == outer_fold)
    checks["all_inner_engine_scopes_match_groupkfold_and_exclude_outer_engines"] = valid_scopes
    checks["inner_raw_metrics_independently_recomputed"] = valid_metrics
    checks["inner_warning_grids_and_policies_independently_recomputed"] = valid_policy
    checks["inner_only_strict_improvement_feasibility_and_fallback_recomputed"] = valid_choice
    return selections, used


def _verify_outer(folder: Path, train: pd.DataFrame, assignments: pd.DataFrame,
                  records: dict[tuple, tuple[Path, pd.DataFrame]], selections: dict[str, dict[str, Any]],
                  cfg: dict[str, Any], scientific: bool, checks: dict[str, bool]) -> set[tuple]:
    outer = pd.read_parquet(folder / "outer_predictions.parquet")
    events = pd.read_parquet(folder / "warning_events.parquet")
    saved_cases = json.loads((folder / "outer_selections.json").read_text(encoding="utf-8"))
    cases = saved_cases["selections"] if isinstance(saved_cases, dict) else saved_cases
    all_ids, used = set(map(int, train.engine_id.unique())), set()
    vectors_valid = scopes_valid = choice_valid = True
    expected_events = []
    for case in cases:
        fold = int(case["outer_fold"])
        validation_ids = tuple(sorted(map(int, assignments.loc[assignments.outer_fold.eq(fold), "engine_id"])))
        fit_ids = tuple(sorted(all_ids - set(validation_ids)))
        inner = selections[f"outer_{fold}"]
        scopes_valid = scopes_valid and case.get("outer_fit_engine_ids", case.get("fit_engine_ids")) == list(fit_ids)
        scopes_valid = scopes_valid and case.get("outer_validation_engine_ids", case.get("validation_engine_ids")) == list(validation_ids)
        choice_valid = choice_valid and case.get("procedure", case.get("choice")) == inner["procedure"]
        if "arms" in case:
            choice_valid = choice_valid and case["arms"] == inner["arms"]
        selected = inner["procedure"]["selected_procedure_id"]
        for arm in ARMS:
            key = (arm, fit_ids, validation_ids)
            used.add(key)
            if key not in records:
                raise ValueError(f"declared outer task missing: {fold}/{arm}")
            cached = records[key][1]
            policy = inner["arms"][arm]["policy"]
            expected = cached.assign(procedure=arm, outer_fold=fold, **policy)
            actual = outer.loc[outer.outer_fold.eq(fold) & outer.procedure.eq(arm)]
            vectors_valid = vectors_valid and _same(actual, expected, list(KEY), exact=True, subset=True)
            derived_events = _warning_events(cached, policy, cfg["lead_requirement"]).assign(procedure=arm, outer_fold=fold)
            expected_events.append(derived_events)
            if arm == selected:
                alias = outer.loc[outer.outer_fold.eq(fold) & outer.procedure.eq("selected_pipeline")]
                vectors_valid = vectors_valid and _same(alias, expected.assign(procedure="selected_pipeline", selected_arm=arm), list(KEY), exact=True, subset=True)
                expected_events.append(derived_events.assign(procedure="selected_pipeline", selected_arm=arm))
    vectors_valid = vectors_valid and set(outer.procedure) == set(PROCEDURES) and not outer.duplicated(["procedure", *KEY]).any()
    scopes_valid = scopes_valid and sorted(int(case["outer_fold"]) for case in cases) == list(range(1, cfg["outer_splits"] + 1))
    checks["complete_outer_engine_scopes_and_all_declared_arms"] = scopes_valid
    checks["outer_choices_equal_preexisting_inner_choices"] = choice_valid
    checks["outer_predictions_cache_exact_and_selected_alias_uses_inner_choice"] = vectors_valid
    checks["outer_warning_events_independently_recomputed"] = _same(events, pd.concat(expected_events, ignore_index=True), ["procedure", "engine_id"])
    summaries, region_rows, engine_tables = [], [], {}
    for procedure in PROCEDURES:
        frame = outer[outer.procedure.eq(procedure)]
        policy_events = events[events.procedure.eq(procedure)]
        summaries.append({"procedure": procedure, **_model_metrics(frame, cfg), **_warnings(policy_events)})
        engine_tables[procedure] = _engine_table(frame, policy_events, cfg)
        scored = _score(frame)
        scored["rul_region"] = pd.cut(scored.y_true, [-1, 30, 60, 100, 150, np.inf], labels=["0-30", "31-60", "61-100", "101-150", ">150"])
        for region, group in scored.groupby("rul_region", observed=False):
            if group.empty:
                continue
            region_rows.append({"procedure": procedure, "rul_region": str(region), "n_rows": len(group), "n_engines": group.engine_id.nunique(),
                                "mean_window_mae": float(group.absolute_error.mean()), "mean_engine_mae": float(group.groupby("engine_id").absolute_error.mean().mean()),
                                "signed_bias": float((group.y_pred - group.y_true).mean()), "rmse": float(np.sqrt(group.squared_error.mean()))})
    checks["outer_aggregate_raw_metrics_and_warning_summary_recomputed"] = _same(_csv(folder / "aggregate_metrics.csv"), pd.DataFrame(summaries), ["procedure"])
    engines = pd.concat([frame.assign(procedure=name).reset_index() for name, frame in engine_tables.items()], ignore_index=True)
    checks["per_engine_50_70_90_endpoints_and_pooled_sufficient_statistics_recomputed"] = _same(_csv(folder / "engine_metrics.csv"), engines, ["procedure", "engine_id"])
    checks["raw_rul_region_errors_and_bias_recomputed"] = _same(_csv(folder / "region_metrics.csv"), pd.DataFrame(region_rows), ["procedure", "rul_region"])
    expected_paired = pd.concat([_paired(engine_tables[ARMS[0]], engine_tables[procedure], cfg).assign(reference=ARMS[0], procedure=procedure)
                                for procedure in PROCEDURES[1:]], ignore_index=True)
    checks["paired_bootstrap_rmse_sse_counts_and_duplicate_multiplicity_recomputed"] = _same(_csv(folder / "paired_comparison.csv"), expected_paired, ["procedure", "reference", "metric"])
    selected_pair = expected_paired[expected_paired.procedure.eq("selected_pipeline")]
    controls = _gate(selected_pair)
    decision = _json(folder / "decision.json")
    checks["fixed_joint_promotion_gate_recomputed_without_test"] = (
        decision["controls"] == controls and decision["promotion_gate_met"] == bool(scientific and all(controls.values()))
        and decision["full_protocol_configuration"] == scientific and decision["selected_reference"] == ARMS[0]
        and decision["historical_baseline_overwritten"] is False)
    checks["outer_decision_frozen_before_test_reporting"] = decision == _json(folder / "outer_decision_before_test.json")
    return used


def _verify_final(folder: Path, train: pd.DataFrame, test: pd.DataFrame, selections: dict[str, dict[str, Any]],
                  cfg: dict[str, Any], source_identity: str, checks: dict[str, bool]) -> dict[str, np.ndarray]:
    selection = _json(folder / "selection.json")
    final_inner = selections["final_inner"]
    checks["final_selection_equals_final_inner_and_has_no_test_selection"] = (
        selection["arms"] == final_inner["arms"] and selection["procedure"] == final_inner["procedure"]
        and selection["test_labels_read_for_selection"] is False and selection["source_identity"] == source_identity)
    selected = final_inner["procedure"]["selected_procedure_id"]
    endpoint = test.groupby("engine_id", sort=True, as_index=False).tail(1).reset_index(drop=True)
    stored = pd.read_parquet(folder / "final_predictions.parquet")
    valid_models = valid_replay = True
    predicted = {}
    for arm in ARMS:
        model_path, meta_path = folder / "models" / arm / "model.joblib", folder / "models" / arm / "model.json"
        meta, model = _json(meta_path), joblib.load(model_path)
        features = tuple(FEATURE_COLUMNS_BY_ARM[arm])
        valid_models = valid_models and isinstance(model, FittedHGB) and model.arm_id == arm and tuple(model.feature_columns) == features
        valid_models = valid_models and meta["arm_id"] == arm and meta["source_identity"] == source_identity and meta["feature_columns"] == list(features)
        valid_models = valid_models and meta["model_sha256"] == _sha(model_path) and meta["fit_engine_ids"] == sorted(map(int, train.engine_id.unique()))
        valid_models = valid_models and meta["effective_parameters"] == model.effective_parameters == _params(cfg)
        valid_models = valid_models and all(model.model.get_params()[name] == value for name, value in _params(cfg).items())
        valid_models = valid_models and model.model.n_features_in_ == len(features) and meta["weight_metadata"] == model.weight_metadata
        with threadpool_limits(limits=cfg["fit_threads"]):
            vector = np.asarray(model.model.predict(endpoint.loc[:, list(features)].to_numpy(dtype="float64")), dtype="float64")
        predicted[arm] = vector
        valid_replay = valid_replay and np.array_equal(vector, model.predict(endpoint))
        expected = endpoint.loc[:, list(KEY)].assign(candidate_id=arm, procedure=arm, y_pred=vector)
        valid_replay = valid_replay and _same(stored[stored.procedure.eq(arm)], expected, list(KEY), exact=True, subset=True)
        if arm == selected:
            valid_replay = valid_replay and _same(stored[stored.procedure.eq("selected_pipeline")], expected.assign(procedure="selected_pipeline", selected_arm=arm), list(KEY), exact=True, subset=True)
    checks["three_saved_models_feature_orders_parameters_and_fit_scopes_valid"] = valid_models
    checks["saved_models_direct_replay_exact_and_selected_pipeline_alias_exact"] = valid_replay and set(stored.procedure) == set(PROCEDURES) and not stored.duplicated(["procedure", *KEY]).any()
    receipt = _json(folder / "final_predictions_receipt.json")
    checks["unlabelled_final_prediction_receipt_hashes_bind_every_model"] = (
        receipt["source_identity"] == source_identity and receipt["predictions_sha256"] == _sha(folder / "final_predictions.parquet")
        and receipt["models_sha256"] == {arm: _sha(folder / "models" / arm / "model.joblib") for arm in ARMS}
        and receipt["saved_model_replay_exact"] is True and receipt["test_labels_read"] is False)
    info = _json(folder / "run_info.json")
    checks["run_completed_with_reporting_only_test_and_final_inner_choice"] = (
        info["status"] == "completed" and info["source_identity"] == source_identity and info["selected_procedure"] == selected
        and info["test_reporting_only"] is True and info["decision"] == _json(folder / "decision.json"))
    if info["test_evaluated"]:
        report = _json(folder / "test_evaluation.json")
        valid_report = (report["reporting_only"] is True and report["selection_used_test_labels"] is False
                        and report.get("outer_gate_changed_by_test", False) is False)
        report_rows = []
        common_truth = None
        for procedure in PROCEDURES:
            evaluation_path = folder / "test" / procedure / "final_evaluation.parquet"
            if not evaluation_path.is_file():
                evaluation_path = folder / "test" / procedure / "evaluation.parquet"
            evaluation = pd.read_parquet(evaluation_path)
            expected = stored[stored.procedure.eq(procedure)].drop(columns="selected_arm", errors="ignore")
            valid_report = valid_report and _same(evaluation, expected, list(KEY), exact=True, subset=True)
            rescored = _score(evaluation)
            valid_report = valid_report and _same(evaluation, rescored, list(KEY))
            truth = evaluation.loc[:, ["split", "engine_id", "y_true"]]
            if common_truth is None:
                common_truth = truth
            else:
                valid_report = valid_report and _same(truth, common_truth, ["split", "engine_id"], exact=True)
            metrics = _metrics(evaluation, "test")
            report_rows.append({"procedure": procedure, **{name.removeprefix("test_"): value for name, value in metrics.items() if name != "test_mean_engine_nasa"}})
        valid_report = valid_report and _same(pd.DataFrame(report["procedures"]), pd.DataFrame(report_rows), ["procedure"])
        checks["stored_test_metrics_recomputed_without_opening_raw_test_labels"] = valid_report
    return predicted


def _verify_baseline(root: Path, folder: Path, train: pd.DataFrame, test: pd.DataFrame,
                     records: dict[tuple, tuple[Path, pd.DataFrame]], predicted: dict[str, np.ndarray],
                     scientific: bool, checks: dict[str, bool]) -> dict[str, Any]:
    source = root / "outputs/experiments/lightgbm_comparison/20261005_study01"
    columns = list(dict.fromkeys((*KEY, *FEATURE_COLUMNS_BY_ARM[ARMS[0]])))
    checks["baseline_169_feature_values_equal_verified_original_exactly"] = all(
        _same(frame.loc[:, columns], pd.read_parquet(source / f"frozen_{split}_features.parquet").loc[:, columns], list(KEY), exact=True)
        for split, frame in (("train", train), ("test", test)))
    original_verification = _json(source / "verification.json")
    checks["baseline_source_is_independently_verified"] = original_verification["status"] == "passed" and original_verification["raw_test_labels_read"] is False and original_verification["model_fit_performed"] is False
    if not scientific:
        return {"source_run": source.relative_to(root).as_posix(), "prediction_parity_scope": "not applicable to reduced smoke iterations/splits", "baseline_feature_parity": True}
    old_records = {}
    for metadata_path in (source / "fit_cache").glob("*.json"):
        metadata = _json(metadata_path)
        if metadata["candidate"]["candidate_id"] == "hgb_uncapped":
            key = (tuple(metadata["fit_engine_ids"]), tuple(metadata["validation_engine_ids"]))
            old_records[key] = metadata_path.with_suffix(".parquet")
    parity = True
    count = 0
    for (arm, fit_ids, validation_ids), (_, frame) in records.items():
        if arm != ARMS[0]:
            continue
        count += 1
        old_path = old_records.get((fit_ids, validation_ids))
        parity = parity and old_path is not None
        if old_path is not None:
            original = pd.read_parquet(old_path).loc[:, [*KEY, "y_true", "y_pred"]]
            parity = parity and _same(frame, original, list(KEY), exact=True, subset=True)
    checks["all_23_baseline_inner_outer_prediction_vectors_equal_original_exactly"] = parity and count == 23
    old_outer = pd.read_parquet(source / "outer_predictions.parquet")
    old_outer = old_outer[old_outer.procedure.eq("hgb_uncapped")].loc[:, [*KEY, "y_true", "y_pred", "outer_fold", "threshold", "persistence"]]
    outer = pd.read_parquet(folder / "outer_predictions.parquet")
    checks["baseline_outer_predictions_and_calibrated_policies_equal_original_exactly"] = _same(outer[outer.procedure.eq(ARMS[0])], old_outer, [*KEY], exact=True, subset=True)
    endpoint = test.groupby("engine_id", sort=True, as_index=False).tail(1).reset_index(drop=True)
    model = joblib.load(source / "models/hgb_uncapped/model.joblib")
    original_prediction = model.predict(endpoint)
    checks["baseline_final_predictions_replay_verified_original_model_exactly"] = np.array_equal(predicted[ARMS[0]], original_prediction)
    return {"source_run": source.relative_to(root).as_posix(), "prediction_parity_scope": "all 23 CV vectors and original final model", "baseline_feature_parity": True}


def verify(run_dir: str | Path, *, repo_root: str | Path = ROOT) -> dict[str, Any]:
    """Verify a completed run without fitting or opening raw test labels."""
    root = Path(repo_root).resolve()
    folder = _resolve(root, run_dir)
    allowed = root / "outputs/experiments/multiscale_comparison"
    if folder == allowed or not folder.is_relative_to(allowed):
        raise ValueError("run directory must be a multiscale_comparison descendant")
    manifest = _json(folder / "manifest.json")
    identity, checks = manifest["identity"], {}
    cfg = identity["config"]
    config_values = dict(cfg)
    for name in ("endpoint_fractions", "thresholds", "persistences"):
        config_values[name] = tuple(config_values[name])
    scientific = StudyConfig(**config_values).scientific
    train = pd.read_parquet(folder / "frozen_train_features.parquet")
    test = pd.read_parquet(folder / "frozen_test_features.parquet")
    assignments = _csv(folder / "outer_assignments.csv")
    _verify_provenance(root, folder, manifest, checks)
    _verify_features(root, folder, identity, train, test, checks)
    checks["outer_assignments_cover_every_engine_once"] = (
        not assignments.engine_id.duplicated().any() and set(assignments.engine_id) == set(train.engine_id)
        and set(assignments.outer_fold) == set(range(1, cfg["outer_splits"] + 1)))
    if cfg["outer_splits"] == 5:
        original_assignments = _csv(root / "outputs/modeling/tables/fold_assignments.csv").loc[:, ["engine_id", "fold"]].rename(columns={"fold": "outer_fold"})
        checks["five_published_outer_engine_folds_retained_exactly"] = _same(assignments, original_assignments, ["engine_id"], exact=True)
    records = _verify_cache(folder, identity, manifest["source_identity"], train, cfg, checks)
    selections, inner_used = _verify_inner(folder, train, assignments, records, cfg, manifest["source_identity"], checks)
    outer_used = _verify_outer(folder, train, assignments, records, selections, cfg, scientific, checks)
    checks["every_unique_cv_task_belongs_to_declared_inner_or_outer_scope"] = set(records) == inner_used | outer_used
    predicted = _verify_final(folder, train, test, selections, cfg, manifest["source_identity"], checks)
    baseline = _verify_baseline(root, folder, train, test, records, predicted, scientific, checks)
    checks = {name: bool(value) for name, value in checks.items()}
    checks["all_checks_passed"] = all(checks.values())
    return {"schema_version": 1, "status": "passed" if checks["all_checks_passed"] else "failed",
            "author": {"full_name": "Phan Nguyễn Mai Phương", "student_id": "25002711"},
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "scientific_configuration": bool(scientific),
            "run_dir": folder.relative_to(root).as_posix(), "source_identity": manifest["source_identity"],
            "source_script_sha256": _sha(Path(__file__).resolve()), "checks": checks,
            "counts": {"cv_cache_records": len(records), "final_models": len(ARMS), "total_fits": len(records) + len(ARMS),
                       "train_engines": int(train.engine_id.nunique()), "test_engines": int(test.engine_id.nunique())},
            "baseline_comparison": baseline, "raw_test_labels_read": False, "model_fit_performed": False,
            "existing_artifacts_modified": False, "bootstrap_scope": INTERVAL_SCOPE,
            "numeric_tolerance": {"prediction_replay_and_baseline_parity": "exact", "reported_metrics": "atol1e-8/rtol1e-10"}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    args = parser.parse_args(argv)
    result = verify(args.run_dir, repo_root=args.repo_root)
    folder = _resolve(Path(args.repo_root).resolve(), args.run_dir)
    receipt = folder / "verification.json"
    if receipt.exists():
        raise FileExistsError(f"verification receipt already exists: {receipt}")
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Keep console output portable on Windows cp1252; the receipt remains UTF-8.
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
