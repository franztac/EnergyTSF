from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np


PathLike = Union[str, Path]

TOP_ROW = [
    ("gru", "variate"),
    ("transformer", "variate"),
    ("mamba", "variate"),
]

BOTTOM_ROW = [
    ("gru", "time"),
    ("transformer", "time"),
    ("mamba", "time"),
]

ALL_CONFIGS = TOP_ROW + BOTTOM_ROW

FIXED_SEED = 42
FIXED_WINDOW = 96
FIXED_HORIZON = 96
FIXED_SPLIT = "test"
FIXED_NPZ_NAME = "test_outputs.npz"

# === CHANGED: centralized forecast-grid font sizes ===
FORECAST_TITLE_FS = 18
FORECAST_SUBPLOT_TITLE_FS = 14
FORECAST_AXIS_LABEL_FS = 14
FORECAST_TICK_FS = 12
FORECAST_LEGEND_FS = 12
# === END CHANGED ===


def _as_path(path: PathLike) -> Path:
    return Path(path)


def _find_latest_test_npz(
    runs_debug_root: PathLike,
    dataset: str,
    token_style: str,
    model: str,
) -> Path:
    base = (
        _as_path(runs_debug_root)
        / dataset
        / token_style
        / model
        / f"L{FIXED_WINDOW}_T{FIXED_HORIZON}"
        / f"seed_{FIXED_SEED}"
    )
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")

    timestamp_dirs = [p for p in base.iterdir() if p.is_dir()]
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamp folders found under: {base}")

    latest_dir = sorted(timestamp_dirs, key=lambda p: p.name)[-1]
    npz_path = latest_dir / FIXED_SPLIT / FIXED_NPZ_NAME
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    return npz_path


def _plot_dataset_forecast_grid(
    runs_debug_root: PathLike,
    dataset: str,
    window_idx: int,
    channel_idx: int = -1,
    save_path: Optional[PathLike] = None,
    show: bool = True,
    verbose: bool = True,
) -> None:
    arrays = {}
    ymin = np.inf
    ymax = -np.inf
    time_points = None

    for model, token_style in ALL_CONFIGS:
        npz_path = _find_latest_test_npz(
            runs_debug_root=runs_debug_root,
            dataset=dataset,
            token_style=token_style,
            model=model,
        )

        data = np.load(npz_path)
        y = data["Y"]
        p = data["P"]

        n, t, v = y.shape

        if verbose and (model, token_style) == ALL_CONFIGS[0]:
            print(
                f"[forecast_grid] available windows: N={n} | horizon T={t} | channels V={v}"
            )

        if not (-n <= window_idx < n):
            raise IndexError(
                f"window_idx={window_idx} out of range for {model}-{token_style}; valid range is 0..{n-1}"
            )
        if not (-v <= channel_idx < v):
            raise IndexError(
                f"channel_idx={channel_idx} out of range for {model}-{token_style}; valid range is {-v}..{v-1}"
            )

        true_series = y[window_idx, :, channel_idx]
        pred_series = p[window_idx, :, channel_idx]
        arrays[(model, token_style)] = (true_series, pred_series)

        ymin = min(ymin, float(np.min(true_series)), float(np.min(pred_series)))
        ymax = max(ymax, float(np.max(true_series)), float(np.max(pred_series)))
        time_points = np.arange(t)

    yrange = ymax - ymin
    pad = 0.05 * yrange if yrange > 0 else 0.05
    ymin -= pad
    ymax += pad

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8.2),
        sharex=True,
        sharey=True,
    )

    for col, (model, token_style) in enumerate(TOP_ROW):
        ax = axes[0, col]
        truth, pred = arrays[(model, token_style)]
        ax.plot(time_points, truth, color="tab:blue", linewidth=2.0, label="Truth")
        ax.plot(time_points, pred, color="tab:orange", linewidth=2.0, label="Prediction")

        # === CHANGED: subplot titles italicized ===
        ax.set_title(
            f"{model.upper()} - {token_style}",
            fontsize=FORECAST_SUBPLOT_TITLE_FS,
            fontstyle="italic",
        )
        # === END CHANGED ===

        ax.set_ylim(ymin, ymax)

        # === CHANGED: axis labels back to normal font ===
        ax.set_xlabel("Time points", fontsize=FORECAST_AXIS_LABEL_FS)
        # === END CHANGED ===

        ax.tick_params(axis="both", labelsize=FORECAST_TICK_FS)
        ax.grid(True, alpha=0.2)

    for col, (model, token_style) in enumerate(BOTTOM_ROW):
        ax = axes[1, col]
        truth, pred = arrays[(model, token_style)]
        ax.plot(time_points, truth, color="tab:blue", linewidth=2.0, label="Truth")
        ax.plot(time_points, pred, color="tab:orange", linewidth=2.0, label="Prediction")

        # === CHANGED: subplot titles italicized ===
        ax.set_title(
            f"{model.upper()} - {token_style}",
            fontsize=FORECAST_SUBPLOT_TITLE_FS,
            fontstyle="italic",
        )
        # === END CHANGED ===

        ax.set_ylim(ymin, ymax)

        # === CHANGED: axis labels back to normal font ===
        ax.set_xlabel("Time points", fontsize=FORECAST_AXIS_LABEL_FS)
        # === END CHANGED ===

        ax.tick_params(axis="both", labelsize=FORECAST_TICK_FS)
        ax.grid(True, alpha=0.2)

    for ax in axes[:, 0]:
        # === CHANGED: y-label back to normal font ===
        ax.set_ylabel("Value", fontsize=FORECAST_AXIS_LABEL_FS)
        # === END CHANGED ===

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    # === CHANGED: legend lower and boxed ===
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.942),
        fontsize=FORECAST_LEGEND_FS,
    )
    # === END CHANGED ===

    fig.suptitle(
        f"{dataset} forecasts | L={FIXED_WINDOW}, T={FIXED_HORIZON} | channel={channel_idx} | window={window_idx} | seed={FIXED_SEED}",
        fontsize=FORECAST_TITLE_FS,
        fontweight="bold",
        y=0.975,
    )

    fig.subplots_adjust(
        # === CHANGED: less space between title and first row ===
        top=0.81,
        # === CHANGED: a bit more space between first and second row ===
        hspace=0.52,
        # === END CHANGED ===
        wspace=0.18,
    )

    if save_path is not None:
        save_path = _as_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_ecl_forecast_grid(
    runs_debug_root: PathLike,
    window_idx: int,
    channel_idx: int = -1,
    save_path: Optional[PathLike] = None,
    show: bool = True,
    verbose: bool = True,
) -> None:
    _plot_dataset_forecast_grid(
        runs_debug_root=runs_debug_root,
        dataset="ECL",
        window_idx=window_idx,
        channel_idx=channel_idx,
        save_path=save_path,
        show=show,
        verbose=verbose,
    )


def plot_etth1_forecast_grid(
    runs_debug_root: PathLike,
    window_idx: int,
    channel_idx: int = -1,
    save_path: Optional[PathLike] = None,
    show: bool = True,
    verbose: bool = True,
) -> None:
    _plot_dataset_forecast_grid(
        runs_debug_root=runs_debug_root,
        dataset="ETTh1",
        window_idx=window_idx,
        channel_idx=channel_idx,
        save_path=save_path,
        show=show,
        verbose=verbose,
    )


# === CHANGED: save-all helper for the 2 NPZ forecast grids ===
def save_all_forecast_grids(
    runs_debug_root: PathLike,
    out_dir: PathLike = "windows11/figures",
    ecl_window_idx: int = 3754,
    ecl_channel_idx: int = -1,
    etth1_window_idx: int = 1885,
    etth1_channel_idx: int = -1,
    show: bool = False,
    verbose: bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_ecl_forecast_grid(
        runs_debug_root=runs_debug_root,
        window_idx=ecl_window_idx,
        channel_idx=ecl_channel_idx,
        save_path=out_dir / "ECL_forecast_grid_L96_T96.png",
        show=show,
        verbose=verbose,
    )

    plot_etth1_forecast_grid(
        runs_debug_root=runs_debug_root,
        window_idx=etth1_window_idx,
        channel_idx=etth1_channel_idx,
        save_path=out_dir / "ETTh1_forecast_grid_L96_T96.png",
        show=show,
        verbose=verbose,
    )
# === END CHANGED ===