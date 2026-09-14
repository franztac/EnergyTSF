from __future__ import annotations

import gc
import traceback
from argparse import Namespace

import torch

from utils.final_run_config import infer_channels, resolve_window_horizon
from utils.final_run_executor_tmp import run_one_experiment, evaluate_one_experiment


def build_base_config(
    *,
    seed: int,
    batch_size: int = 32,
    stride: int = 1,
    ratios=(0.7, 0.15, 0.15),
    min_lr: float = 1e-7,
    early_stop_patience: int = 3,
    num_epochs: int = 10,
    verbose: bool = False,
):
    """
    Build the shared base config used in notebook loops.
    """
    return Namespace(
        seed=seed,
        verbose=verbose,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        stride=stride,
        batch_size=batch_size,
        ratios=tuple(ratios),
        min_lr=min_lr,
        early_stop_patience=early_stop_patience,
        num_epochs=num_epochs,
    )


def make_experiment_config(
    *,
    base_cfg: Namespace,
    experiment_mode: str,
    data_name: str,
    model_name: str,
    token_style: str,
):
    """
    Create one cfg from a base config + one experiment tuple.
    """
    window, horizon = resolve_window_horizon(experiment_mode)

    cfg = Namespace(**vars(base_cfg))
    cfg.data_name = data_name
    cfg.model_name = model_name
    cfg.token_style = token_style
    cfg.window = window
    cfg.horizon = horizon
    cfg.channels = infer_channels(cfg.data_name)
    return cfg


def make_experiment_grid(
    *,
    experiment_modes,
    datasets,
    models,
    token_styles,
):
    """
    Cartesian product helper.

    Returns tuples:
        (experiment_mode, data_name, model_name, token_style)
    """
    grid = []
    for experiment_mode in experiment_modes:
        for data_name in datasets:
            for model_name in models:
                for token_style in token_styles:
                    grid.append((experiment_mode, data_name, model_name, token_style))
    return grid


def run_train_and_eval_loop(
    *,
    experiment_grid,
    base_cfg: Namespace,
    root_dir_runs: str = "runs_debug",
    eval_save_npz_splits=None,
    eval_make_test_plot: bool = False,
    eval_plot_num_random: int = 3,
    eval_plot_seed: int = 40,
    eval_plot_channel_idx: int = -1,
    eval_include_val: bool = False,
    verbose_train: bool = False,
    verbose_eval: bool = False,
):
    """
    Train each experiment, then immediately evaluate it.

    Good for stress-testing or full runs with controlled cleanup.
    """
    train_results = []
    train_failures = []
    eval_results = []
    eval_failures = []

    total = len(experiment_grid)

    for idx, (experiment_mode, data_name, model_name, token_style) in enumerate(experiment_grid, start=1):
        cfg = make_experiment_config(
            base_cfg=base_cfg,
            experiment_mode=experiment_mode,
            data_name=data_name,
            model_name=model_name,
            token_style=token_style,
        )

        tag = (
            f"[{idx}/{total}] "
            f"mode={experiment_mode} | "
            f"{cfg.data_name} | {cfg.model_name} | {cfg.token_style} | "
            f"L{cfg.window}_T{cfg.horizon} | BS={cfg.batch_size} | seed={cfg.seed}"
        )
        print("\n" + "=" * 90)
        print(tag)

        try:
            train_res = run_one_experiment(
                base_cfg=cfg,
                experiment_mode=experiment_mode,
                root_dir_runs=root_dir_runs,
                verbose=verbose_train,
            )
            train_results.append(train_res)

            print(
                f"TRAIN OK  | best_val={train_res['best_val_loss']:.6f} | "
                f"best_epoch={train_res['best_epoch']}"
            )
            print(f"          saved: {train_res['run_root']}")

        except Exception as e:
            train_failures.append(
                {
                    "idx": idx,
                    "mode": experiment_mode,
                    "data_name": data_name,
                    "model_name": model_name,
                    "token_style": token_style,
                    "seed": cfg.seed,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"TRAIN ERROR: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        try:
            eval_res = evaluate_one_experiment(
                cfg=train_res["config"],
                run_root=train_res["run_root"],
                save_npz_splits=eval_save_npz_splits,
                save_summary=True,
                make_test_plot=eval_make_test_plot,
                plot_num_random=eval_plot_num_random,
                plot_seed=eval_plot_seed,
                plot_channel_idx=eval_plot_channel_idx,
                include_val=eval_include_val,
                verbose=verbose_eval,
            )
            eval_results.append(eval_res)

            print(f"EVAL  OK  | test mse={eval_res['metrics']['test']['mse']:.6f}")
            print(f"          test dir: {eval_res['test_dir']}")

        except Exception as e:
            eval_failures.append(
                {
                    "idx": idx,
                    "mode": experiment_mode,
                    "data_name": data_name,
                    "model_name": model_name,
                    "token_style": token_style,
                    "run_root": train_res["run_root"],
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"EVAL ERROR: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print(f"Train completed : {len(train_results)}")
    print(f"Train failed    : {len(train_failures)}")
    print(f"Eval completed  : {len(eval_results)}")
    print(f"Eval failed     : {len(eval_failures)}")

    return {
        "train_results": train_results,
        "train_failures": train_failures,
        "eval_results": eval_results,
        "eval_failures": eval_failures,
    }


def run_eval_only_loop(
    *,
    train_results,
    save_npz_splits=None,
    make_test_plot: bool = False,
    plot_num_random: int = 3,
    plot_seed: int = 40,
    plot_channel_idx: int = -1,
    include_val: bool = False,
    verbose: bool = False,
):
    """
    Evaluate already-trained runs from train_results only.
    No retraining.
    """
    eval_results = []
    eval_failures = []

    total = len(train_results)

    for idx, train_res in enumerate(train_results, start=1):
        cfg = train_res["config"]
        run_root = train_res["run_root"]

        print(
            f"[EVAL {idx}/{total}] "
            f"{cfg.data_name} | {cfg.model_name} | {cfg.token_style} | "
            f"L{cfg.window}_T{cfg.horizon}"
        )

        try:
            eval_res = evaluate_one_experiment(
                cfg=cfg,
                run_root=run_root,
                save_npz_splits=save_npz_splits,
                save_summary=True,
                make_test_plot=make_test_plot,
                plot_num_random=plot_num_random,
                plot_seed=plot_seed,
                plot_channel_idx=plot_channel_idx,
                include_val=include_val,
                verbose=verbose,
            )

            eval_results.append(eval_res)

            print(f"       test mse : {eval_res['metrics']['test']['mse']:.6f}")
            print(f"       npz      : {eval_res['test_npz_path']}")
            print(f"       plot     : {eval_res['test_plot_path']}")

        except Exception as e:
            eval_failures.append(
                {
                    "config": cfg,
                    "run_root": run_root,
                    "error": str(e),
                }
            )
            print(f"       ERROR: {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n========== EVAL SUMMARY ==========")
    print(f"Completed: {len(eval_results)}")
    print(f"Failed   : {len(eval_failures)}")

    return {
        "eval_results": eval_results,
        "eval_failures": eval_failures,
    }
