from __future__ import annotations

import csv
import json
import math
import os
from statistics import mean, stdev
from typing import Dict, List, Tuple


# =========================================================
# Small utilities
# =========================================================
def _read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(stdev(values))


def _infer_mode_from_window_horizon(window: int, horizon: int) -> str:
    """
    Your current experimental convention:
    - L96_T96  -> multivariate
    - L192_T96 -> increasing_lookback
    """
    if window == 96 and horizon == 96:
        return "multivariate"
    if window == 192 and horizon == 96:
        return "increasing_lookback"
    return "unknown"


def _flatten_record_for_csv(record: dict) -> dict:
    """
    Keep only simple scalar/string fields for CSV writing.
    """
    return dict(record)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================================================
# Scan latest runs
# =========================================================
def _find_latest_timestamp_dir(base_dir: str) -> str | None:
    if not os.path.isdir(base_dir):
        return None

    candidates = [
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]
    if not candidates:
        return None

    # Timestamp format YYYY-MM-DD_HH-MM sorts correctly lexicographically
    latest = sorted(candidates)[-1]
    return os.path.join(base_dir, latest)


def collect_latest_run_roots(root_dir_runs: str = "runs_debug") -> List[str]:
    """
    Traverse the experiment tree and keep only the most recent timestamp
    for each (dataset, token_style, model, Lx_Ty, seed).
    """
    latest_run_roots = []

    if not os.path.isdir(root_dir_runs):
        raise FileNotFoundError(f"Missing root_dir_runs: {root_dir_runs}")

    for dataset in sorted(os.listdir(root_dir_runs)):
        dataset_dir = os.path.join(root_dir_runs, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        if dataset.startswith("_"):
            continue

        for token_style in sorted(os.listdir(dataset_dir)):
            token_dir = os.path.join(dataset_dir, token_style)
            if not os.path.isdir(token_dir):
                continue

            for model in sorted(os.listdir(token_dir)):
                model_dir = os.path.join(token_dir, model)
                if not os.path.isdir(model_dir):
                    continue

                for wh in sorted(os.listdir(model_dir)):
                    wh_dir = os.path.join(model_dir, wh)
                    if not os.path.isdir(wh_dir):
                        continue

                    for seed_name in sorted(os.listdir(wh_dir)):
                        seed_dir = os.path.join(wh_dir, seed_name)
                        if not os.path.isdir(seed_dir):
                            continue

                        latest_run = _find_latest_timestamp_dir(seed_dir)
                        if latest_run is not None:
                            latest_run_roots.append(latest_run)

    return latest_run_roots


# =========================================================
# Build per-seed records
# =========================================================
def build_all_runs_by_seed_records(root_dir_runs: str = "runs_debug") -> List[dict]:
    """
    Build one flat record per latest run.
    Sources:
    - run_summary.json
    - train/best_metrics.json
    - test/post_training_metrics.json
    """
    run_roots = collect_latest_run_roots(root_dir_runs=root_dir_runs)
    records = []

    for run_root in sorted(run_roots):
        run_summary_path = os.path.join(run_root, "run_summary.json")
        best_metrics_path = os.path.join(run_root, "train", "best_metrics.json")
        post_training_path = os.path.join(run_root, "test", "post_training_metrics.json")

        if not os.path.exists(run_summary_path):
            continue
        if not os.path.exists(best_metrics_path):
            continue
        if not os.path.exists(post_training_path):
            continue

        run_summary = _read_json(run_summary_path)
        best_metrics = _read_json(best_metrics_path)
        post_training = _read_json(post_training_path)

        dataset = run_summary["data"]["dataset"]
        model = run_summary["experiment"]["model_name"]
        token_style = run_summary["experiment"]["token_style"]
        seed = run_summary["experiment"]["seed"]
        timestamp = run_summary["experiment"]["timestamp"]

        window = run_summary["data"]["window"]
        horizon = run_summary["data"]["horizon"]
        mode = _infer_mode_from_window_horizon(window, horizon)

        record = {
            "dataset": dataset,
            "mode": mode,
            "model": model,
            "token_style": token_style,
            "window": int(window),
            "horizon": int(horizon),
            "seed": int(seed),
            "timestamp": timestamp,

            "train_mse": float(best_metrics["best_train_loss"]),
            "train_mae": float(best_metrics["best_train_mae"]),
            "val_mse": float(best_metrics["best_val_loss"]),
            "val_mae": float(best_metrics["best_val_mae"]),
            "test_mse": float(post_training["test"]["mse"]),
            "test_mae": float(post_training["test"]["mae"]),

            "best_epoch": int(best_metrics["best_epoch"]),
            "epochs_ran": int(best_metrics["epochs_ran"]),
            "training_time_sec": float(best_metrics["training_time_sec"]),
            "ms_per_iter": float(best_metrics["ms_per_iter"]),
            "gpu_mem_gb": float(best_metrics["gpu_mem_gb"]),

            "num_trainable_params": int(run_summary["model_stats"]["num_trainable_params"]),
            "size_mb": float(run_summary["model_stats"]["size_mb"]),

            "run_root": run_root,
        }

        records.append(record)

    # Stable sort for readability
    records = sorted(
        records,
        key=lambda r: (
            r["dataset"],
            r["mode"],
            r["model"],
            r["token_style"],
            r["seed"],
        )
    )

    return records


# =========================================================
# Aggregate across seeds
# =========================================================
def build_all_configs_mean_std_records(all_runs_by_seed: List[dict]) -> List[dict]:
    """
    Aggregate per-seed records into one record per config.
    """
    grouped: Dict[Tuple, List[dict]] = {}

    for rec in all_runs_by_seed:
        key = (
            rec["dataset"],
            rec["mode"],
            rec["model"],
            rec["token_style"],
            rec["window"],
            rec["horizon"],
        )
        grouped.setdefault(key, []).append(rec)

    out = []

    metric_fields = [
        "train_mse", "train_mae",
        "val_mse", "val_mae",
        "test_mse", "test_mae",
        "best_epoch", "epochs_ran",
        "training_time_sec", "ms_per_iter", "gpu_mem_gb",
    ]

    for key, rows in sorted(grouped.items()):
        dataset, mode, model, token_style, window, horizon = key
        rows = sorted(rows, key=lambda r: r["seed"])

        agg = {
            "dataset": dataset,
            "mode": mode,
            "model": model,
            "token_style": token_style,
            "window": int(window),
            "horizon": int(horizon),
            "seeds": ",".join(str(r["seed"]) for r in rows),
        }

        for field in metric_fields:
            values = [float(r[field]) for r in rows]
            agg[f"{field}_mean"] = float(mean(values))
            agg[f"{field}_std"] = float(_safe_std(values))

        # model size/params are config-level constants, so take from first row
        agg["num_trainable_params"] = int(rows[0]["num_trainable_params"])
        agg["size_mb"] = float(rows[0]["size_mb"])

        out.append(agg)

    return out


# =========================================================
# Writers
# =========================================================
def write_json_records(records: List[dict], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(records, f, indent=4)


def write_csv_records(records: List[dict], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    if not records:
        with open(path, "w", newline="") as f:
            pass
        return

    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(_flatten_record_for_csv(rec))


# =========================================================
# Main public helper
# =========================================================
def build_aggregate_artifacts(root_dir_runs: str = "runs_debug") -> dict:
    """
    Build all 4 aggregate files:

    runs_debug/_aggregates/
        all_runs_by_seed.json
        all_runs_by_seed.csv
        all_configs_mean_std.json
        all_configs_mean_std.csv
    """
    aggregate_dir = os.path.join(root_dir_runs, "_aggregates")
    _ensure_dir(aggregate_dir)

    all_runs_by_seed = build_all_runs_by_seed_records(root_dir_runs=root_dir_runs)
    all_configs_mean_std = build_all_configs_mean_std_records(all_runs_by_seed)

    paths = {
        "all_runs_by_seed_json": os.path.join(aggregate_dir, "all_runs_by_seed.json"),
        "all_runs_by_seed_csv": os.path.join(aggregate_dir, "all_runs_by_seed.csv"),
        "all_configs_mean_std_json": os.path.join(aggregate_dir, "all_configs_mean_std.json"),
        "all_configs_mean_std_csv": os.path.join(aggregate_dir, "all_configs_mean_std.csv"),
    }

    write_json_records(all_runs_by_seed, paths["all_runs_by_seed_json"])
    write_csv_records(all_runs_by_seed, paths["all_runs_by_seed_csv"])
    write_json_records(all_configs_mean_std, paths["all_configs_mean_std_json"])
    write_csv_records(all_configs_mean_std, paths["all_configs_mean_std_csv"])

    return {
        "paths": paths,
        "num_seed_records": len(all_runs_by_seed),
        "num_config_records": len(all_configs_mean_std),
    }
