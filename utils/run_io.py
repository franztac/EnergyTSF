#  - uses config_sanitize.sanitize_for_summary() to avoid irrelevant fields
#  - writes JSON with pretty indent BUT keeps "ratios" on one line
#  - keeps optimizer/scheduler/model_stats as before


import os
import json
import re
from utils.tools import count_parameters
from utils.config_sanitize import sanitize_for_summary



def _json_dump_compact_ratios(payload: dict, indent: int = 4) -> str:
    """
    Pretty JSON but force "ratios": [..] to be on a single line.
    """
    s = json.dumps(payload, indent=indent, default=str)
    # Replace the ratios block: "ratios": [\n  ... \n]
    pattern = r'("ratios"\s*:\s*)\[\s*([^\]]*?)\s*\]'
    def repl(m):
        inner = " ".join(m.group(2).split())  # collapse whitespace
        inner = inner.replace(", ]", "]")
        return f'{m.group(1)}[{inner}]'
    return re.sub(pattern, repl, s, flags=re.DOTALL)



def save_run_summary(
    *,
    save_run_dir: str,
    run_config,
    model,
    optimizer=None,
    scheduler=None,
    filename: str = "run_summary.json",
):
    """
    Save run-level metadata JSON into:
        save_run_dir/run_summary.json

    v1 (clean + model/token conditional):
    - experiment/data/model/training from sanitized config
    - optimizer + scheduler (simple)
    - model_stats (params + size)
    """
    os.makedirs(save_run_dir, exist_ok=True)

    # timestamp from folder name (your current convention)
    timestamp = os.path.basename(os.path.normpath(save_run_dir))

    # ---- Base summary (cleaned; no null spam) ----
    summary = sanitize_for_summary(run_config)
    summary["experiment"]["timestamp"] = timestamp  # add timestamp from path

    # ---- Model stats ----
    num_params, size_mb = count_parameters(model)
    summary["model_stats"] = {
        "num_trainable_params": int(num_params),
        "size_mb": float(size_mb),
    }

    # ---- Optimizer ----
    opt_dict = {}
    if optimizer is not None:
        opt_dict["name"] = optimizer.__class__.__name__.lower()
        if hasattr(optimizer, "param_groups") and optimizer.param_groups:
            g0 = optimizer.param_groups[0]
            opt_dict["lr"] = float(g0.get("lr", 0.0))
            wd = g0.get("weight_decay", None)
            if wd is not None:
                opt_dict["weight_decay"] = float(wd)
    summary["optimizer"] = opt_dict

    # ---- Scheduler (plateau-friendly) ----
    sch_dict = {}
    if scheduler is not None:
        sch_dict["name"] = scheduler.__class__.__name__.lower()
        # ReduceLROnPlateau: factor/patience/min_lr commonly exist
        if hasattr(scheduler, "factor"):
            sch_dict["factor"] = float(scheduler.factor)
        if hasattr(scheduler, "patience"):
            sch_dict["patience"] = int(scheduler.patience)
        if hasattr(scheduler, "min_lrs") and isinstance(scheduler.min_lrs, (list, tuple)) and scheduler.min_lrs:
            sch_dict["min_lr"] = float(scheduler.min_lrs[0])
        elif hasattr(scheduler, "min_lr"):
            sch_dict["min_lr"] = float(scheduler.min_lr)
    summary["scheduler"] = sch_dict

    # ---- Write JSON (ratios one line) ----
    out_path = os.path.join(save_run_dir, filename)

    # CHANGED: write text produced by helper (instead of json.dump directly)
    text = _json_dump_compact_ratios(summary, indent=4)
    with open(out_path, "w") as f:
        f.write(text)

    return out_path