from __future__ import annotations
import yaml



def load_yaml_config(yaml_path: str) -> dict:
    """
    Load one YAML file and return it as a Python dict.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty YAML file: {yaml_path}")
    return data



def validate_yaml_top_keys(yaml_cfg: dict, yaml_path: str = "") -> None:
    """
    Check that the YAML has the 3 required top-level sections.
    """
    required_keys = ["meta", "fixed_params", "search_space"]
    for key in required_keys:
        if key not in yaml_cfg:
            raise ValueError(
                f"Missing top-level key '{key}' in YAML file: {yaml_path}"
            )
        


def validate_yaml_identity(cfg, meta: dict, yaml_path: str = "") -> None:
    """
    Check that notebook/base config identity matches YAML identity.
    """
    identity_keys = ["data_name", "model_name", "token_style", "window", "horizon", "channels"]

    for key in identity_keys:
        yaml_value = meta.get(key, None)
        cfg_value = getattr(cfg, key, None)

        if yaml_value is not None and cfg_value is not None and yaml_value != cfg_value:
            raise ValueError(
                f"Identity mismatch in YAML '{yaml_path}' for key '{key}': "
                f"yaml={yaml_value}, cfg={cfg_value}"
            )



def apply_fixed_params(cfg, fixed_params: dict) -> None:
    """
    Set fixed params from YAML directly on cfg.
    """
    for param_name, value in fixed_params.items():
        setattr(cfg, param_name, value)




def suggest_param_from_spec(trial, param_name: str, spec: dict):
    """
    Convert one YAML param spec into one Optuna suggestion.
    """
    param_type = spec["type"]

    if param_type == "float":
        return trial.suggest_float(
            param_name,
            spec["low"],
            spec["high"],
            log=spec.get("log", False),
        )

    if param_type == "int":
        return trial.suggest_int(
            param_name,
            spec["low"],
            spec["high"],
            step=spec.get("step", 1),
            log=spec.get("log", False),
        )

    if param_type == "categorical":
        return trial.suggest_categorical(
            param_name,
            spec["choices"],
        )

    raise ValueError(
        f"Unsupported param type '{param_type}' for param '{param_name}'"
    )




def apply_search_space(trial, cfg, search_space: dict) -> None:
    """
    Apply searchable params from YAML to cfg.

    Special rule:
    - if temperature exists, we suggest it only when pooling == 'attention'
    - otherwise we force temperature = 1.0
    """
    # 1) apply all normal params except conditional ones
    conditional_params = {"temperature"}

    for param_name, spec in search_space.items():
        if param_name in conditional_params:
            continue

        value = suggest_param_from_spec(trial, param_name, spec)
        setattr(cfg, param_name, value)
    
    # 2) conditional rule for temperature
    if "temperature" in search_space:
        pooling = getattr(cfg, "pooling", None)

        if pooling is None:
            raise ValueError(
                "'temperature' exists in search_space but cfg.pooling is missing. "
                "Temperature is only valid when pooling logic exists."
            )

        if pooling == "attention":
            cfg.temperature = suggest_param_from_spec(
                trial,
                "temperature",
                search_space["temperature"],
            )
        else:
            cfg.temperature = 1.0




def print_yaml_summary(yaml_cfg: dict, yaml_path: str) -> None:
    meta = yaml_cfg.get("meta", {})
    search_space = yaml_cfg.get("search_space", {})

    print(
        f"[YAML] {yaml_path} | "
        f"{meta.get('data_name')} | "
        f"{meta.get('model_name')} | "
        f"{meta.get('token_style')} | "
        f"L{meta.get('window')}_T{meta.get('horizon')} | "
        f"search={len(search_space)}"
    )