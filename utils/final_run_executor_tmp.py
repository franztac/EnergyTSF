from __future__ import annotations

import gc
import os
from argparse import Namespace

import torch
import torch.nn as nn
import torch.optim as optim

from engine.seed import set_global_seed
from engine.trainer import Trainer

from data_provider.data_factory_tmp import data_factory
from engine.evaluator_tmp import Evaluator

from utils.final_run_config import load_configs_from_final_yaml
from utils.model_factory import build_model_from_configs
from utils.exp_utils import build_run_paths_from_configs
from utils.plot_utils import make_suptitle
from utils.visualization_tmp import plot_forecasts_from_npz
from utils.runtime_setup import configure_runtime_warnings

import matplotlib
matplotlib.use('Agg')


def cleanup_after_run(device) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_one_experiment(
    *,
    base_cfg: Namespace,
    experiment_mode: str,
    root_dir_runs: str = "runs",
    verbose: bool = False,
):
    """
    Train exactly one experiment.
    This tmp version keeps the training logic unchanged in spirit,
    but uses data_factory_tmp so your original files remain safe.
    """
    cfg = Namespace(**vars(base_cfg))

    configure_runtime_warnings(cfg.model_name)
    set_global_seed(seed=cfg.seed, verbose=False, deterministic=True)

    cfg, yaml_path = load_configs_from_final_yaml(
        cfg=cfg,
        experiment_mode=experiment_mode,
        verbose=verbose,
    )

    if verbose:
        print(
            f"TRAIN -> {cfg.data_name} | {cfg.model_name} | {cfg.token_style} | "
            f"L{cfg.window}_T{cfg.horizon} | seed={cfg.seed}"
        )

    dm = data_factory(
        data_name=cfg.data_name,
        window=cfg.window,
        horizon=cfg.horizon,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        ratios=tuple(cfg.ratios),
        verbose=False,
    )

    train_loader, val_loader, test_loader = dm.setup(
        seed=cfg.seed,
        verbose=False,
    )

    model = build_model_from_configs(cfg, verbose=False).to(cfg.device)

    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    criterion = nn.MSELoss(reduction="mean")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.min_lr,
    )

    paths = build_run_paths_from_configs(
        configs=cfg,
        root_dir=root_dir_runs,
        verbose=False,
    )

    save_run_dir = paths["run_root"]
    save_artifacts_dir = paths["train_dir"]
    save_test_dir = paths["test_dir"]

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=cfg.device,
        patience=cfg.early_stop_patience,
        save_artifacts_dir=save_artifacts_dir,
        save_run_dir=save_run_dir,
        run_config=cfg,
        verbose=False,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.num_epochs,
        trial=None,
        save_all=True,
        verbose=False,
    )

    weights_path = os.path.join(save_artifacts_dir, "best_model.pt")

    result = {
        "config": cfg,
        "yaml_path": yaml_path,
        "run_root": save_run_dir,
        "train_dir": save_artifacts_dir,
        "test_dir": save_test_dir,
        "weights_path": weights_path,
        "best_val_loss": trainer.best_val_loss,
        "best_epoch": trainer.best_epoch,
    }

    del model
    del trainer
    del optimizer
    del train_loader
    del val_loader
    del test_loader
    del dm

    cleanup_after_run(cfg.device)

    return result


def evaluate_one_experiment(
    *,
    cfg: Namespace,
    run_root: str,
    save_npz_splits=("test",),
    save_summary: bool = True,
    make_test_plot: bool = True,
    plot_num_random: int = 3,
    plot_seed: int = 40,
    plot_channel_idx: int = -1,
    verbose: bool = False,
    include_val: bool = False,
):
    """
    Evaluate one already-trained experiment using the shallow tmp path.

    Default behavior:
    - test only
    - post_training_metrics.json contains only test metrics
    - NPZ saves only Y and P
    - optional automated plot under test folder
    """
    configure_runtime_warnings(cfg.model_name)
    set_global_seed(seed=cfg.seed, verbose=False, deterministic=True)

    save_artifacts_dir = os.path.join(run_root, "train")
    save_test_dir = os.path.join(run_root, "test")
    weights_path = os.path.join(save_artifacts_dir, "best_model.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing best_model.pt at: {weights_path}")

    if verbose:
        print(
            f"EVAL  -> {cfg.data_name} | {cfg.model_name} | {cfg.token_style} | "
            f"L{cfg.window}_T{cfg.horizon} | seed={cfg.seed}"
        )

    dm = data_factory(
        data_name=cfg.data_name,
        window=cfg.window,
        horizon=cfg.horizon,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        ratios=tuple(cfg.ratios),
        verbose=False,
    )

    _, val_loader, test_loader = dm.setup_selected(
        include_train=False,
        include_val=include_val,
        include_test=True,
        seed=cfg.seed,
        verbose=False,
    )

    model_eval = build_model_from_configs(cfg, verbose=False).to(cfg.device)
    state_dict = torch.load(
        weights_path,
        map_location=cfg.device,
        weights_only=True,
    )
    model_eval.load_state_dict(state_dict)

    evaluator = Evaluator(device=cfg.device)

    split_loaders = {"test": test_loader}
    if include_val and val_loader is not None:
        split_loaders = {"val": val_loader, "test": test_loader}

    all_results = evaluator.evaluate_splits_shallow(
        model=model_eval,
        split_loaders=split_loaders,
        save_dir=save_test_dir,
        save_npz_splits=save_npz_splits,
        save_summary=save_summary,
        verbose=verbose,
    )

    test_npz_path = os.path.join(save_test_dir, "test_outputs.npz")
    png_path = None

    if make_test_plot and ("test" in (save_npz_splits or [])) and os.path.exists(test_npz_path):
        suptitle = make_suptitle(cfg, split_name="test")
        png_path = plot_forecasts_from_npz(
            npz_path=test_npz_path,
            window_indices="random",
            num_random=plot_num_random,
            seed=plot_seed,
            channel_idx=plot_channel_idx,
            save_dir=save_test_dir,
            show=False,
            verbose=False,
            suptitle=suptitle,
        )

    result = {
        "config": cfg,
        "run_root": run_root,
        "train_dir": save_artifacts_dir,
        "test_dir": save_test_dir,
        "weights_path": weights_path,
        "metrics": all_results,
        "test_npz_path": test_npz_path if os.path.exists(test_npz_path) else None,
        "test_plot_path": png_path,
    }

    del state_dict
    del evaluator
    del model_eval
    del test_loader
    if val_loader is not None:
        del val_loader
    del dm

    cleanup_after_run(cfg.device)

    return result