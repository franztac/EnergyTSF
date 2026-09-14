# utils/config_sanitize.py
"""
Config sanitation utilities.

Goals:
- Avoid 'null' spam in JSON (no irrelevant keys).
- Keep structure consistent across encoders and token styles.
- Split model hparams into:
    - hparams_common: shared keys
    - hparams_specific: only keys relevant to that encoder + token_style
"""

from __future__ import annotations
from typing import Any, Dict



def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy with all None values removed."""
    return {k: v for k, v in d.items() if v is not None}





def _get(cfg, key: str, default=None):
    """Safe getattr for Namespace-like objects."""
    return getattr(cfg, key, default)





def build_model_hparams(cfg) -> Dict[str, Dict[str, Any]]:
    """
    Build encoder-agnostic (common) + encoder/token-specific hparams.

    Returns:
        {"hparams_common": {...}, "hparams_specific": {...}}
    """
    model_name = _get(cfg, "model_name")
    token_style = _get(cfg, "token_style")

    # ---- Common keys (always useful) ----
    common = {
        "channels": _get(cfg, "channels"),
        "d_model": _get(cfg, "d_model"),
        "num_layers": _get(cfg, "num_layers"),
        "dropout": _get(cfg, "dropout"),
    }

    specific: Dict[str, Any] = {}

    # token-style: time pooling
    if token_style == "time":
        pooling = _get(cfg, "pooling")
        specific["pooling"] = pooling

        if pooling == "attention":
            specific["temperature"] = _get(cfg, "temperature")

    if model_name == "gru":
        # GRU-specific
        specific["bidirectional"] = _get(cfg, "bidirectional")

    elif model_name == "transformer":
        specific.update({
            "nhead": _get(cfg, "nhead"),
            "dim_feedforward": _get(cfg, "d_ff"),  # usually 4*d_model
            "activation": _get(cfg, "activation"),
            "norm_first": _get(cfg, "norm_first"),
        })

        # pos_emb only for time token transformer
        if token_style == "time":
            specific["pos_emb"] = _get(cfg, "pos_emb")  # "sin" | "learnable"
        # variate: omit pos_emb entirely (your preference)

    elif model_name == "mamba":
        # --- Mamba-specific updates ---
        specific.update({
            "d_state": _get(cfg, "d_state"),
            "expand": _get(cfg, "expand"),
            "d_conv": _get(cfg, "d_conv"),
            "dt_rank": _get(cfg, "dt_rank"),
            "dt_min": _get(cfg, "dt_min"),
            "dt_max": _get(cfg, "dt_max"),
            "use_fast_path": _get(cfg, "use_fast_path"),
            "d_ff": _get(cfg, "d_ff"),          # Capture the FFN capacity
            "activation": _get(cfg, "activation") # Capture FFN activation choice
        })

        # Time token can toggle bidirectional; variate is hardcoded True
        if token_style == "time":
            specific["bidirectional"] = _get(cfg, "bidirectional")
        else:
            specific["bidirectional"] = True

    # Remove None values to avoid JSON null spam
    return {
        "hparams_common": _drop_none(common),
        "hparams_specific": _drop_none(specific),
    }




def sanitize_for_summary(cfg) -> Dict[str, Any]:
    """
    Produce a clean dict suitable for run_summary.json.

    This does NOT attempt to include *every* cfg field, only the ones you track.
    """
    model_name = _get(cfg, "model_name")
    token_style = _get(cfg, "token_style")

    model_hparams = build_model_hparams(cfg)

    out = {
        "experiment": {
            "model_name": model_name,
            "token_style": token_style,
            "seed": _get(cfg, "seed"),
            "device": str(_get(cfg, "device")),
            # timestamp handled by run_io from folder name
        },
        "data": {
            "dataset": _get(cfg, "data_name"),
            "window": _get(cfg, "window"),
            "horizon": _get(cfg, "horizon"),
            "stride": _get(cfg, "stride"),
            "batch_size": _get(cfg, "batch_size"),
            "ratios": list(_get(cfg, "ratios", [])),
            "channels": _get(cfg, "channels"),
        },
        "model": {
            "family": model_name,
            "token_style": token_style,
            **model_hparams,
        },
        "training": {
            "num_epochs": _get(cfg, "num_epochs"),
            "early_stop_patience": _get(cfg, "early_stop_patience"),
        },
    }

    # Final drop-none pass on top-level subdicts
    out["experiment"] = _drop_none(out["experiment"])
    out["data"] = _drop_none(out["data"])
    out["training"] = _drop_none(out["training"])

    return out