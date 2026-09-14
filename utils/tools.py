# utils/tools.py
import torch


def count_parameters(model):
    """
    Returns the number of trainable parameters and estimated size in MB (float32).
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (num_params * 4) / (1024**2)
    return num_params, size_mb


def print_diagnostics(model, config, params, size_mb):
    """
    Ordered diagnostics:

    1) Positional embedding (if applicable)
    2) Common hparams
    3) Encoder-specific hparams
    4) Pooling (time token only)
    5) Params / size
    """

    model_name = getattr(config, "model_name", None)
    token_style = getattr(config, "token_style", None)

    print(f"\n--- 🤖 Model: {model.__class__.__name__} Ready ---")
    print(f"Family: {model_name} | Token Style: {token_style}")

    # 1) Positional Embedding (Transformer Time only)
    if model_name == "transformer" and token_style == "time":
        pos_emb = getattr(config, "pos_emb", None)
        if pos_emb is not None:
            print(f"Positional Embedding: {pos_emb}")

    # 2) Common (Always Printed)
    channels = getattr(config, "channels", None)
    d_model = getattr(config, "d_model", None)
    num_layers = getattr(config, "num_layers", None)
    dropout = getattr(config, "dropout", None)
    print(f"Common: channels={channels} | d_model={d_model} | layers={num_layers} | dropout={dropout}")

    # 3) Encoder-specific
    if model_name == "gru":
        bidir = getattr(config, "bidirectional", None)
        print(f"GRU: bidirectional={bidir}")

    elif model_name == "transformer":
        nhead = getattr(config, "nhead", None)
        dff = getattr(config, "dim_feedforward", None)
        act = getattr(config, "activation", None)
        norm_first = getattr(config, "norm_first", None)
        print(f"Transformer: nhead={nhead} | d_ff={dff} | act={act} | norm_first={norm_first}")

    elif model_name == "mamba":
        d_state = getattr(config, "d_state", None)
        expand = getattr(config, "expand", None)
        d_conv = getattr(config, "d_conv", None)
        bidir = getattr(config, "bidirectional", None) if token_style == "time" else True
        dff = getattr(config, "d_ff", 512)
        act = getattr(config, "activation", "gelu")
        print(f"Mamba: d_state={d_state} | expand={expand} | d_conv={d_conv} | bidirectional={bidir}")
        # FFN Line for Mamba
        print(f"Mamba FFN: d_ff={dff} | act={act}")

    # 4) Pooling (time token only)
    if token_style == "time":
        pooling = getattr(config, "pooling", None)
        if pooling is not None:
            msg = f"Pooling: {pooling}"
            if pooling == "attention":
                temp = getattr(config, "temperature", None)
                msg += f" | temperature={temp}"
            print(msg)
    
    # 5) Params
    print(f"Params: {params:,} | Size: {size_mb:.2f} MB\n")