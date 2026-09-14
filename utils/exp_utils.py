# This ensures your experiments are saved in a clean, 
# categorized way based on the configs Namespace.

import os
from datetime import datetime



def build_run_paths(
    dataset: str,
    token_style: str,
    model_name: str,
    window: int,
    horizon: int,
    seed: int,
    root_dir: str = "runs",
    verbose: bool = True,
):
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    
    # runs/ETTh1/time/gru/L96_T96/seed_42/2026-02-15_18-10/
    run_root = os.path.join(
        root_dir,
        dataset,
        token_style,
        model_name,
        f"L{window}_T{horizon}",
        f"seed_{seed}",
        timestamp,
    )

    train_dir = os.path.join(run_root, "train")
    test_dir = os.path.join(run_root, "test")

    # 3. Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    if verbose:
        print("\n[exp_utils] Run directories created:")
        print(f"  run_root : {run_root}")
        print(f"  train_dir: {train_dir}")
        print(f"  test_dir : {test_dir}\n")

    return {
        "run_root": run_root,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "timestamp": timestamp
    }


def build_run_paths_from_configs(
    configs,
    root_dir: str = "runs",
    verbose: bool = True,
):
    """
    Wrapper that uses your Namespace configs (cleaner notebook usage).
    """
    return build_run_paths(
        dataset=configs.data_name,
        token_style=configs.token_style,
        model_name=configs.model_name,
        window=configs.window,
        horizon=configs.horizon,
        seed=configs.seed,
        root_dir=root_dir,
        verbose=verbose,
    )