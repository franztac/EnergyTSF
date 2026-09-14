from __future__ import annotations

import os
from argparse import Namespace

import yaml


MODE_TO_WINDOW_HORIZON = {
    "multivariate": (96, 96),
    "increasing_lookback": (192, 96),
}

DATASET_TO_CHANNELS = {
    "ETTh1": 7,
    "ECL": 321,
}


def resolve_window_horizon(experiment_mode: str) -> tuple[int, int]:
    if experiment_mode not in MODE_TO_WINDOW_HORIZON:
        raise ValueError(
            f"Unsupported EXPERIMENT_MODE='{experiment_mode}'. "
            f"Expected one of: {list(MODE_TO_WINDOW_HORIZON.keys())}"
        )
    return MODE_TO_WINDOW_HORIZON[experiment_mode]


def infer_channels(data_name: str) -> int:
    if data_name not in DATASET_TO_CHANNELS:
        raise ValueError(
            f"Unknown data_name='{data_name}'. "
            f"Expected one of: {list(DATASET_TO_CHANNELS.keys())}"
        )
    return DATASET_TO_CHANNELS[data_name]


def build_final_yaml_path(
    *,
    experiment_mode: str,
    data_name: str,
    model_name: str,
    token_style: str,
    root_dir: str = "configs/final_runs",
) -> str:
    window, horizon = resolve_window_horizon(experiment_mode)

    filename = (
        f"{data_name.lower()}_"
        f"{model_name}_"
        f"{token_style}_"
        f"L{window}_T{horizon}.yaml"
    )

    yaml_path = os.path.join(root_dir, experiment_mode, filename)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Missing final-run YAML: {yaml_path}"
        )

    return yaml_path


def load_final_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty YAML file: {yaml_path}")

    return data


def validate_final_yaml_top_keys(yaml_cfg: dict, yaml_path: str = "") -> None:
    required_keys = ["meta", "params"]
    for key in required_keys:
        if key not in yaml_cfg:
            raise ValueError(
                f"Missing top-level key '{key}' in YAML file: {yaml_path}"
            )


def validate_yaml_identity(cfg, meta: dict, yaml_path: str = "") -> None:
    identity_keys = [
        "data_name",
        "model_name",
        "token_style",
        "window",
        "horizon",
        "channels",
    ]

    for key in identity_keys:
        yaml_value = meta.get(key, None)
        cfg_value = getattr(cfg, key, None)

        if yaml_value is None:
            raise ValueError(
                f"Missing identity key '{key}' in YAML meta: {yaml_path}"
            )

        if cfg_value is None:
            raise ValueError(
                f"Missing identity key '{key}' in cfg before validation: {key}"
            )

        if yaml_value != cfg_value:
            raise ValueError(
                f"Identity mismatch in YAML '{yaml_path}' for key '{key}': "
                f"yaml={yaml_value}, cfg={cfg_value}"
            )


def _required_params_for(cfg) -> set[str]:
    model_name = cfg.model_name
    token_style = cfg.token_style

    common = {
        "d_model",
        "num_layers",
        "dropout",
        "lr",
        "weight_decay",
        "scheduler_factor",
        "scheduler_patience",
    }

    if model_name == "gru":
        required = common | {"bidirectional"}

    elif model_name == "transformer":
        required = common | {
            "nhead",
            "dim_feedforward",
            "activation",
            "norm_first",
            "layer_norm_eps",
            "batch_first",
        }

    elif model_name == "mamba":
        required = common | {
            "d_state",
            "expand",
            "d_conv",
            "d_ff",
            "dt_rank",
            "dt_min",
            "dt_max",
            "use_fast_path",
        }

    else:
        raise ValueError(f"Unsupported model_name='{model_name}'")

    if token_style == "time":
        required = required | {"pooling", "temperature"}

        if model_name == "transformer":
            required = required | {"pos_emb"}

        if model_name == "mamba":
            required = required | {"bidirectional"}

    elif token_style == "variate":
        pass

    else:
        raise ValueError(f"Unsupported token_style='{token_style}'")

    return required


def validate_required_params(cfg, params: dict, yaml_path: str = "") -> None:
    required = _required_params_for(cfg)
    missing = sorted(k for k in required if k not in params)

    if missing:
        raise ValueError(
            f"Missing required params in YAML '{yaml_path}': {missing}"
        )


def warn_irrelevant_params(cfg, params: dict) -> list[str]:
    model_name = cfg.model_name
    token_style = cfg.token_style

    irrelevant = []

    transformer_only = {
        "nhead", "dim_feedforward", "activation", "norm_first",
        "layer_norm_eps", "batch_first", "pos_emb",
    }
    mamba_only = {
        "d_state", "expand", "d_conv", "d_ff",
        "dt_rank", "dt_min", "dt_max", "use_fast_path",
    }

    time_only = {"pooling", "temperature", "pos_emb"}

    if model_name != "transformer":
        irrelevant.extend([k for k in params if k in transformer_only])

    if model_name != "mamba":
        irrelevant.extend([k for k in params if k in mamba_only])

    if token_style != "time":
        irrelevant.extend([k for k in params if k in time_only])

    # unique + sorted
    return sorted(set(irrelevant))


def apply_yaml_params(cfg, params: dict) -> None:
    for key, value in params.items():
        setattr(cfg, key, value)


def load_configs_from_final_yaml(
    *,
    cfg: Namespace,
    experiment_mode: str,
    root_dir: str = "configs/final_runs",
    verbose: bool = True,
) -> tuple[Namespace, str]:
    yaml_path = build_final_yaml_path(
        experiment_mode=experiment_mode,
        data_name=cfg.data_name,
        model_name=cfg.model_name,
        token_style=cfg.token_style,
        root_dir=root_dir,
    )

    yaml_cfg = load_final_yaml(yaml_path)
    validate_final_yaml_top_keys(yaml_cfg, yaml_path=yaml_path)

    meta = yaml_cfg["meta"]
    params = yaml_cfg["params"]

    validate_yaml_identity(cfg, meta, yaml_path=yaml_path)
    validate_required_params(cfg, params, yaml_path=yaml_path)

    warnings = warn_irrelevant_params(cfg, params)
    if verbose and warnings:
        print(f"[final_run_config][warn] Irrelevant extra params in YAML: {warnings}")

    apply_yaml_params(cfg, params)
    cfg.final_yaml_path = yaml_path

    return cfg, yaml_path