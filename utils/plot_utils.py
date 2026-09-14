def make_suptitle(cfg, split_name=None):
    """Generates a standardized title based on experiment configuration."""
    prefix = f"{split_name.upper()} " if split_name else ""

    base = f"{prefix}FORECASTS | {cfg.data_name} | {cfg.model_name} | {cfg.token_style} | seed={cfg.seed} | L{cfg.window} T{cfg.horizon}"
    
    if cfg.token_style == "time":
        base += f" | pool={cfg.pooling}"
        if cfg.model_name == "transformer":
            base += f" | pe={cfg.pos_emb}"

    return base