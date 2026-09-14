import copy
import torch
import warnings
import optuna
import torch.nn as nn
import torch.optim as optim
from engine.trainer import Trainer
from engine.seed import set_global_seed
from data_provider.data_factory import data_factory
from models.gru import SimpleGRUModel
from models.transformer import TransformerModel
from models.mamba import MambaModel

from utils.optuna_yaml import (
    load_yaml_config,
    validate_yaml_top_keys,
    validate_yaml_identity,
    apply_fixed_params,
    apply_search_space,
    print_yaml_summary,
)




# helpers
def _configure_optuna_logging(optuna_verbosity: bool) -> None:
    optuna.logging.set_verbosity(
        optuna.logging.INFO if optuna_verbosity else optuna.logging.WARNING
    )

def _configure_warning_filters(model_name: str | None) -> None:
    # Only silence this known-noise warning for PyTorch Transformer
    if model_name == "transformer":
        warnings.filterwarnings(
            "ignore",
            message=r"enable_nested_tensor is True, but self.use_nested_tensor is False.*",
            category=UserWarning,
        )



# ---------------------------------------------------------
# Objective Function 
# ---------------------------------------------------------

def objective(trial, base_configs):
    cfg = copy.deepcopy(base_configs)

    # Modular Seed: Reset for every trial
    set_global_seed(cfg.seed, verbose=False)


    # ---------------------------------------------------------
    # SEARCH SPACE
    # ---------------------------------------------------------

    # model common
    cfg.dropout = trial.suggest_float("dropout", 0.05, 0.3)
    cfg.d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    cfg.num_layers = trial.suggest_int("num_layers", 1, 3)

    # Optim
    cfg.lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    cfg.scheduler_factor = trial.suggest_float("scheduler_factor", 0.4, 0.6)
    cfg.scheduler_patience = trial.suggest_int("scheduler_patience", 2, 5)

    # Time-Token pooling (Both GRU & Transformer)
    if cfg.token_style == "time":
        cfg.pooling = trial.suggest_categorical("pooling", ["last", "attention"])
        if cfg.pooling == "attention":
            cfg.temperature = trial.suggest_float("temperature", 0.5, 2.0)
        else:
            cfg.temperature = 1.0
    
    # GRU specific
    if cfg.model_name == "gru":
        cfg.bidirectional = trial.suggest_categorical("bidirectional", [True, False])

    # Transformer specific
    elif cfg.model_name == "transformer":
        cfg.nhead = trial.suggest_categorical("nhead", [4, 8])
        cfg.dim_feedforward = trial.suggest_categorical("dim_feedforward", [256, 512, 1024])
        cfg.activation = trial.suggest_categorical("activation", ["gelu", "relu"])
        cfg.norm_first = trial.suggest_categorical("norm_first", [True, False])
        cfg.layer_norm_eps = getattr(cfg, "layer_norm_eps", 1e-5)
        cfg.batch_first    = getattr(cfg, "batch_first", True)

        # Positional Encoding logic (Only relevant for Time-wise)
        if cfg.token_style == "time":
            cfg.pos_emb = trial.suggest_categorical("pos_emb", ["sin", "learnable"])
        else:
            cfg.pos_emb = None

    # Mamba specific
    elif cfg.model_name == "mamba":
        cfg.d_state = trial.suggest_categorical("d_state", [8, 16, 32])
        cfg.expand = trial.suggest_categorical("expand", [1, 2, 4])
        cfg.d_conv = trial.suggest_categorical("d_conv", [2, 4, 8])
        cfg.d_ff = trial.suggest_categorical("d_ff", [128, 256, 512])
        cfg.dt_min = getattr(cfg, "dt_min", .001)
        cfg.dt_max = getattr(cfg, "dt_max", .1)
        cfg.dt_rank = getattr(cfg, "dt_rank", "auto")
        cfg.use_fast_path = getattr(cfg, "use_fast_path", True)

    else:
        raise ValueError(f"Unsupported model_name: {cfg.model_name}")


    
    # ---------------------------------------------------------
    # EXECUTION PIPELINE
    # ---------------------------------------------------------
    #  -- Data --
    dm = data_factory(
        data_name=cfg.data_name,
        window=cfg.window,
        horizon=cfg.horizon,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        ratios=cfg.ratios,
        verbose=False,

    )

    train_loader, val_loader, _ = dm.setup(
        seed=cfg.seed,
        verbose=False,
    )

    # -- Model --
    if cfg.model_name == "gru":
        model = SimpleGRUModel(configs=cfg, verbose=False).to(cfg.device)
    elif cfg.model_name == "transformer":
        model = TransformerModel(configs=cfg, verbose=False).to(cfg.device)
    elif cfg.model_name == "mamba":
        model = MambaModel(configs=cfg, verbose=False).to(cfg.device)
    else:
        raise ValueError(f"Unsupported model_name: {cfg.model_name}")

    # -- Standard Optimization Setup --
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.min_lr,
    )

    criterion = nn.MSELoss(reduction="mean")


    # -- Trainer --
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=cfg.device,
        patience=cfg.early_stop_patience,
        save_artifacts_dir=None,
        save_run_dir=None,
        run_config=None,
        verbose=False,
    )
    
    # We do NOT pass the trial here to avoid internal trainer warnings
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.num_epochs,
        trial=None,
        save_all=False,
        verbose=False,
    )


    # ---------------------------------------------------------
    # POST-HOC REPORTING & PRUNING
    # ---------------------------------------------------------
    val_loss = trainer.best_val_loss

    # Report the best result found. 
    # Use best_epoch if your Trainer tracks it, otherwise use total epochs.
    trial.report(
        value=val_loss, 
        step=getattr(trainer, 'best_epoch', cfg.num_epochs),
    )
    
    # The Pruner checks if this trial is significantly worse than previous successful trials
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_loss




def objective_yaml(trial, base_configs, print_yaml: bool = False,):
    cfg = copy.deepcopy(base_configs)

    # reset seed every trial
    set_global_seed(cfg.seed, verbose=False)

    # 1) load yaml
    yaml_path = getattr(cfg, "search_space_yaml", None)
    if yaml_path is None:
        raise ValueError("Missing cfg.search_space_yaml for objective_mode='yaml'")

    yaml_cfg = load_yaml_config(yaml_path)
    validate_yaml_top_keys(yaml_cfg, yaml_path=yaml_path)

    if print_yaml and trial.number == 0:
        print_yaml_summary(yaml_cfg, yaml_path)

    meta = yaml_cfg["meta"]
    fixed_params = yaml_cfg["fixed_params"]
    search_space = yaml_cfg["search_space"]

    # 2) safety check: yaml identity must match notebook config
    validate_yaml_identity(cfg, meta, yaml_path=yaml_path)

    # 3) apply fixed params from YAML
    apply_fixed_params(cfg, fixed_params)

    # 4) apply optuna suggestions from YAML
    apply_search_space(trial, cfg, search_space)


    # 5) Data
    dm = data_factory(
        data_name=cfg.data_name,
        window=cfg.window,
        horizon=cfg.horizon,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        ratios=tuple(cfg.ratios),
        verbose=False,
    )

    train_loader, val_loader, _ = dm.setup(
        seed=cfg.seed,
        verbose=False,
    )

    # 6) model
    if cfg.model_name == "gru":
        model = SimpleGRUModel(configs=cfg, verbose=False).to(cfg.device)

    elif cfg.model_name == "transformer":
        model = TransformerModel(configs=cfg, verbose=False).to(cfg.device)

    elif cfg.model_name == "mamba":
        model = MambaModel(configs=cfg, verbose=False).to(cfg.device)

    else:
        raise ValueError(f"Unsupported model_name: {cfg.model_name}")

    # 7) optimization
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.min_lr,
    )

    criterion = nn.MSELoss(reduction="mean")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=cfg.device,
        patience=cfg.early_stop_patience,
        save_artifacts_dir=None,
        save_run_dir=None,
        run_config=None,
        verbose=False,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.num_epochs,
        trial=None,
        save_all=False,
        verbose=False,
    )

    val_loss = trainer.best_val_loss

    trial.report(
        value=val_loss,
        step=getattr(trainer, "best_epoch", cfg.num_epochs),
    )

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_loss

    


# ---------------------------------------------------------
# Study Runner
# ---------------------------------------------------------

def run_optuna_study(
    base_configs,
    storage,
    study_name,
    n_trials: int = 20,
    timeout: int = None, 
    show_progress_bar: bool = True,
    optuna_verbosity: bool = True,
    objective_mode: str = "classic",   # "classic" | "yaml"
    print_yaml: bool = False,
):
    
    # Optuna logging control
    _configure_optuna_logging(optuna_verbosity)
    # Silence PyTorch Transformer nested-tensor warning (only for transformer runs)
    _configure_warning_filters(getattr(base_configs, "model_name", None))
    
    # sampler = optuna.samplers.TPESampler(seed=base_configs.seed)
    sampler = optuna.samplers.TPESampler(seed=None)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)

    # Create/Load Study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,   
        load_if_exists=True,
    )
    
    # select objective
    if objective_mode == "classic":
        obj_fn = lambda trial: objective(trial, base_configs)
    elif objective_mode == "yaml":
        obj_fn = lambda trial: objective_yaml(trial, base_configs, print_yaml)
    else:
        raise ValueError(
            f"Unsupported objective_mode='{objective_mode}'. "
            f"Use 'classic' or 'yaml'."
        )
    

    study.optimize(
        func=obj_fn,
        n_trials=n_trials, 
        timeout=timeout,  
        show_progress_bar=show_progress_bar,
        gc_after_trial=True,   # The "Cache Cleaner"
        n_jobs=1,
        catch=(RuntimeError,),
    )

    return study

