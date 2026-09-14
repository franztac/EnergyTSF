from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import pandas as pd


PathLike = Union[str, Path]

ROW_ORDER = [
    ("gru", "variate"),
    ("transformer", "variate"),
    ("mamba", "variate"),
    ("gru", "time"),
    ("transformer", "time"),
    ("mamba", "time"),
]

DATASETS = ["ECL", "ETTh1"]

GRID_COL_SPECS = [
    ("L96 MSE", 96, "test_mse_mean", "test_mse_std"),
    ("L96 MAE", 96, "test_mae_mean", "test_mae_std"),
    ("L192 MSE", 192, "test_mse_mean", "test_mse_std"),
    ("L192 MAE", 192, "test_mae_mean", "test_mae_std"),
]

HIST_COLOR_L96 = "#9ecae1"
HIST_COLOR_L192 = "#f7b267"

BUBBLE_COLORS = {
    ("gru", "variate"): "#1f77b4",
    ("transformer", "variate"): "#6baed6",
    ("mamba", "variate"): "#004b9a",
    ("gru", "time"): "#ff7f0e",
    ("transformer", "time"): "#f4a259",
    ("mamba", "time"): "#d94801",
}

# === CHANGED: centralized font sizes for slide-friendly plots ===
TITLE_FS = 18
AXIS_LABEL_FS = 16
TICK_FS = 13
LEGEND_FS = 12
ANNOTATION_FS = 17
GRID_TITLE_FS = 17
GRID_CELL_FS = 10.5
# === END CHANGED ===

# === NEW: Custom constants for the Efficiency Grid ===
EFF_TITLE_FS = 30
EFF_AXIS_LABEL_FS = 26
EFF_ANNOTATION_FS = 26
EFF_TICK_FS = 24       # Way bigger axis numbers
EFF_SUPTITLE_FS = 42
EFF_MARKER_SIZE = 400
# === END NEW ===


def load_aggregate_csv(csv_path: PathLike) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "dataset",
        "mode",
        "model",
        "token_style",
        "window",
        "horizon",
        "test_mse_mean",
        "test_mse_std",
        "test_mae_mean",
        "test_mae_std",
        "ms_per_iter_mean",
        "ms_per_iter_std",
        "gpu_mem_gb_mean",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")
    return df


def _ensure_parent(save_path: Optional[PathLike]) -> Optional[Path]:
    if save_path is None:
        return None
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finalize(fig, save_path: Optional[PathLike], show: bool) -> None:
    path = _ensure_parent(save_path)
    if path is not None:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _dataset_df(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    sub = df[df["dataset"] == dataset].copy()
    if sub.empty:
        raise ValueError(f"No rows found for dataset={dataset!r}")
    return sub


def _fmt_metric(mean_val: float, std_val: float) -> str:
    return f"{mean_val:.4f} ± {std_val:.4f}"


# === CHANGED: helper to draw horizontal gray marks on bar pairs ===
def _add_bar_group_hlines(
    ax,
    x_centers,
    vals_left,
    vals_right,
    split_idx: int = 3,
    color: str = "#5f5f5f",
    linewidth: float = 2.2,
) -> None:
    # left block
    for i in range(split_idx):
        y_low = min(vals_left[i], vals_right[i])
        y_high = max(vals_left[i], vals_right[i])
        ys = [
            y_low + 0.15 * (y_high - y_low + 1e-12),
            y_low + 0.40 * (y_high - y_low + 1e-12),
            y_low + 0.65 * (y_high - y_low + 1e-12),
            y_low + 0.88 * (y_high - y_low + 1e-12),
        ]
        for y in ys:
            ax.hlines(
                y=y,
                xmin=x_centers[i] - 0.22,
                xmax=x_centers[i] + 0.22,
                colors=color,
                linewidth=linewidth,
                alpha=0.95,
                zorder=5,
            )

    # right block
    for i in range(split_idx, len(x_centers)):
        y_low = min(vals_left[i], vals_right[i])
        y_high = max(vals_left[i], vals_right[i])
        ys = [
            y_low + 0.18 * (y_high - y_low + 1e-12),
            y_low + 0.45 * (y_high - y_low + 1e-12),
            y_low + 0.72 * (y_high - y_low + 1e-12),
        ]
        for y in ys:
            ax.hlines(
                y=y,
                xmin=x_centers[i] - 0.22,
                xmax=x_centers[i] + 0.22,
                colors=color,
                linewidth=linewidth,
                alpha=0.95,
                zorder=5,
            )
# === END CHANGED ===


def plot_test_metrics_grid(
    df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    rows = []
    labels = []
    kinds = []

    for ds in DATASETS:
        rows.append([ds, "", "", ""])
        labels.append("")
        kinds.append(("header", ds))
        sub = _dataset_df(df, ds)

        for model, token in ROW_ORDER:
            vals = []
            for _, win, mean_col, std_col in GRID_COL_SPECS:
                r = sub[
                    (sub["model"] == model)
                    & (sub["token_style"] == token)
                    & (sub["window"] == win)
                ]
                if r.empty:
                    raise ValueError(f"Missing row for {ds}, {model}, {token}, L={win}")
                rr = r.iloc[0]
                vals.append(_fmt_metric(rr[mean_col], rr[std_col]))
            rows.append(vals)
            labels.append(f"{model}-{token}")
            kinds.append(("data", ds, model, token))

    styles = {}
    for ds in DATASETS:
        sub = _dataset_df(df, ds)
        for cidx, (_, win, mean_col, _) in enumerate(GRID_COL_SPECS):
            vals = []
            for i, kind in enumerate(kinds):
                if kind[0] == "data" and kind[1] == ds:
                    _, _, model, token = kind
                    rr = sub[
                        (sub["model"] == model)
                        & (sub["token_style"] == token)
                        & (sub["window"] == win)
                    ].iloc[0]
                    vals.append((i, float(rr[mean_col])))
            vals = sorted(vals, key=lambda x: x[1])
            styles[(vals[0][0], cidx)] = "bold"
            styles[(vals[1][0], cidx)] = "purple"

    # === CHANGED: use a slightly shorter figure so the table fills more vertical space ===
    fig, ax = plt.subplots(figsize=(12.2, 7.2))
    # === END CHANGED ===
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        rowLabels=labels,
        colLabels=[c[0] for c in GRID_COL_SPECS],
        loc="center",
        cellLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(GRID_CELL_FS)
    # === CHANGED: slightly taller table so it sits closer to the title ===
    tbl.scale(1.02, 1.70)
    # === END CHANGED ===

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(12.5)

    for i, kind in enumerate(kinds):
        tr = i + 1
        if kind[0] == "header":
            for c in range(-1, len(GRID_COL_SPECS)):
                if (tr, c) in tbl.get_celld():
                    cell = tbl[(tr, c)]
                    cell.set_facecolor("#cfcfcf")
                    cell.set_linewidth(0)
                    if c == 0:
                        cell.get_text().set_text(kind[1])
                        cell.get_text().set_weight("bold")
                        cell.get_text().set_ha("left")
                        cell.get_text().set_fontsize(12.5)
                    else:
                        cell.get_text().set_text("")
        else:
            for cidx in range(len(GRID_COL_SPECS)):
                txt = tbl[(tr, cidx)].get_text()
                style = styles.get((i, cidx))
                if style == "bold":
                    txt.set_weight("bold")
                elif style == "purple":
                    txt.set_color("purple")

    # === CHANGED: title anchored in axes coordinates, directly above the table ===
    ax.text(
        0.5,
        0.91,   # was 0.89
        "TEST METRICS (T=96)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=20,
        fontweight="bold",
    )
    # === END CHANGED ===

    _finalize(fig, save_path, show)


def plot_mse_histogram(
    df: pd.DataFrame,
    dataset: str,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    sub = _dataset_df(df, dataset)
    labels, vals96, vals192, err96, err192 = [], [], [], [], []

    for model, token in ROW_ORDER:
        r96 = sub[
            (sub["model"] == model)
            & (sub["token_style"] == token)
            & (sub["window"] == 96)
        ].iloc[0]
        r192 = sub[
            (sub["model"] == model)
            & (sub["token_style"] == token)
            & (sub["window"] == 192)
        ].iloc[0]
        labels.append(f"{model}\n{token}")
        vals96.append(r96["test_mse_mean"])
        vals192.append(r192["test_mse_mean"])
        err96.append(r96["test_mse_std"])
        err192.append(r192["test_mse_std"])

    x = list(range(len(labels)))
    w = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(
        [i - w / 2 for i in x],
        vals96,
        w,
        yerr=err96,
        capsize=3,
        label="L96",
        color=HIST_COLOR_L96,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.bar(
        [i + w / 2 for i in x],
        vals192,
        w,
        yerr=err192,
        capsize=3,
        label="L192",
        color=HIST_COLOR_L192,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FS)
    ax.axvline(2.5, linewidth=1)
    # === CHANGED: bigger labels/title for slides ===
    ax.set_ylabel("Test MSE", fontsize=18)
    ax.set_title(
        r"$\bf{" + dataset + "}$" + r" - grouped $\bf{test\ MSE}$ comparison (T = 96)",
        fontsize=TITLE_FS,
    )
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS)
    # === END CHANGED ===
    _finalize(fig, save_path, show)


def plot_increasing_lookback(
    df: pd.DataFrame,
    dataset: str,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    sub = _dataset_df(df, dataset)

    fig, ax = plt.subplots(figsize=(8.9, 5.6))
    var_handles = []
    time_handles = []

    for model, token in ROW_ORDER:
        s = sub[
            (sub["model"] == model) & (sub["token_style"] == token)
        ].sort_values("window")

        handle, = ax.plot(
            s["window"],
            s["test_mse_mean"],
            marker=("D" if token == "variate" else "o"),
            markersize=8,
            linewidth=2.2,
            label=f"{model}-{token}",
        )

        if token == "variate":
            var_handles.append(handle)
        else:
            time_handles.append(handle)

    ax.set_xlabel("L (window)", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Test MSE", fontsize=AXIS_LABEL_FS)
    ax.set_xticks([96, 192])
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_title(
        r"$\bf{" + dataset + "}$" + " — increasing lookback comparison (T = 96)",
        fontsize=TITLE_FS,
    )

    # === CHANGED: final legend placements ===
    if dataset == "ECL":
        # Time → LEFT (swapped)
        leg1 = ax.legend(
            handles=time_handles,
            title="Time",
            title_fontsize=LEGEND_FS + 4,
            fontsize=LEGEND_FS + 1,
            loc="center",
            bbox_to_anchor=(0.20, 0.64),
            framealpha=0.9,
        )
        ax.add_artist(leg1)

        # Variate → RIGHT (swapped)
        ax.legend(
            handles=var_handles,
            title="Variate",
            title_fontsize=LEGEND_FS + 4,
            fontsize=LEGEND_FS + 1,
            loc="center",
            bbox_to_anchor=(0.83, 0.28),
            framealpha=0.9,
        )

    elif dataset == "ETTh1":
        # Variate → LOWER (as requested)
        leg1 = ax.legend(
            handles=var_handles,
            title="Variate",
            title_fontsize=LEGEND_FS + 4,
            fontsize=LEGEND_FS + 1,
            loc="center",
            bbox_to_anchor=(0.26, 0.40),
            framealpha=0.9,
        )
        ax.add_artist(leg1)

        # Time → SLIGHTLY ABOVE (as requested)
        ax.legend(
            handles=time_handles,
            title="Time",
            title_fontsize=LEGEND_FS + 4,
            fontsize=LEGEND_FS + 1,
            loc="center",
            bbox_to_anchor=(0.82, 0.52),
            framealpha=0.9,
        )

    else:
        leg1 = ax.legend(
            handles=var_handles,
            title="Variate",
            loc="center",
            bbox_to_anchor=(0.20, 0.70),
        )
        ax.add_artist(leg1)

        ax.legend(
            handles=time_handles,
            title="Time",
            loc="center",
            bbox_to_anchor=(0.82, 0.30),
        )
    # === END CHANGED ===

    _finalize(fig, save_path, show)


def plot_bubble_ecl(
    df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    sub = _dataset_df(df, "ECL")
    sub = sub[sub["window"] == 96].copy()

    # === CHANGED: reposition labels to match desired slide layout ===
    offsets = {
        ("gru", "variate"): (-54, -2, "right"),
        ("transformer", "variate"): (46, 18, "left"),
        ("mamba", "variate"): (-48, 18, "right"),
        ("gru", "time"): (-52, 4, "right"),
        ("transformer", "time"): (0, 74, "center"),
        ("mamba", "time"): (48, 28, "left"),
    }
    # === END CHANGED ===

    fig, ax = plt.subplots(figsize=(12.6, 7.6))
    # === CHANGED: cut extra right blank space more aggressively ===
    ax.set_xlim(-10, 46)
    # === END CHANGED ===
    ax.set_ylim(0.126, 0.312)

    for _, r in sub.iterrows():
        key = (r["model"], r["token_style"])
        size = 100 + 9000 * r["gpu_mem_gb_mean"]
        ax.scatter(
            r["ms_per_iter_mean"],
            r["test_mse_mean"],
            s=size,
            alpha=0.62,
            color=BUBBLE_COLORS[key],
            edgecolors="none",
        )

    for _, r in sub.iterrows():
        key = (r["model"], r["token_style"])
        dx, dy, ha = offsets[key]
        ax.annotate(
            f'{r["model"]}-{r["token_style"]}\n{r["gpu_mem_gb_mean"]:.2f}Gb',
            (r["ms_per_iter_mean"], r["test_mse_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=ANNOTATION_FS,
            color=BUBBLE_COLORS[key],
        )

    ax.set_xlabel("Training Time (ms/iter)", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("MSE", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_title(r"$\bf{ECL}$" + " {L = 96, T = 96}", pad=16, fontsize=TITLE_FS)

    _finalize(fig, save_path, show)


def plot_bubble_etth1(
    df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    sub = _dataset_df(df, "ETTh1")
    sub = sub[sub["window"] == 96].copy()

    # === CHANGED: ETTh1 label placements fixed to match marked target boxes ===
    offsets = {
        # moved clearly to the right of the left bubble
        ("gru", "variate"): (26, 0, "left"),

        # moved to the right and slightly up of the middle light-blue bubble
        ("transformer", "variate"): (34, 20, "left"),

        # kept readable near the right bubble
        ("mamba", "variate"): (-16, 24, "right"),

        ("gru", "time"): (42, 24, "left"),
        ("transformer", "time"): (48, 2, "left"),
        ("mamba", "time"): (-46, 12, "right"),
    }
    # === END CHANGED ===

    fig, ax = plt.subplots(figsize=(12.6, 7.6))

    # === CHANGED: tight x-limits with room for left/right labels ===
    xmin = sub["ms_per_iter_mean"].min()
    xmax = sub["ms_per_iter_mean"].max()
    left_pad = 0.7
    right_pad = 0.6
    ax.set_xlim(xmin - left_pad, xmax + right_pad)
    # === END CHANGED ===

    ax.set_ylim(0.39, 0.73)

    for _, r in sub.iterrows():
        key = (r["model"], r["token_style"])
        size = 100 + 9000 * r["gpu_mem_gb_mean"]
        ax.scatter(
            r["ms_per_iter_mean"],
            r["test_mse_mean"],
            s=size,
            alpha=0.62,
            color=BUBBLE_COLORS[key],
            edgecolors="none",
        )

    for _, r in sub.iterrows():
        key = (r["model"], r["token_style"])
        dx, dy, ha = offsets[key]
        ax.annotate(
            f'{r["model"]}-{r["token_style"]}\n{r["gpu_mem_gb_mean"]:.2f}Gb',
            (r["ms_per_iter_mean"], r["test_mse_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=ANNOTATION_FS,
            color=BUBBLE_COLORS[key],
        )

    ax.set_xlabel("Training Time (ms/iter)", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("MSE", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_title(r"$\bf{ETTh1}$" + " {L = 96, T = 96}", pad=16, fontsize=TITLE_FS)

    _finalize(fig, save_path, show)


def plot_ms_iter_histogram(
    df: pd.DataFrame,
    dataset: str,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    sub = _dataset_df(df, dataset)
    labels, vals96, vals192, err96, err192 = [], [], [], [], []

    for model, token in ROW_ORDER:
        r96 = sub[
            (sub["model"] == model)
            & (sub["token_style"] == token)
            & (sub["window"] == 96)
        ].iloc[0]
        r192 = sub[
            (sub["model"] == model)
            & (sub["token_style"] == token)
            & (sub["window"] == 192)
        ].iloc[0]
        labels.append(f"{model}\n{token}")
        vals96.append(r96["ms_per_iter_mean"])
        vals192.append(r192["ms_per_iter_mean"])
        err96.append(r96["ms_per_iter_std"])
        err192.append(r192["ms_per_iter_std"])

    x = list(range(len(labels)))
    w = 0.36

    fig, ax = plt.subplots(figsize=(10.7, 4.9))

    # === CHANGED: custom hatch styling ===
    ax.bar(
        [i - w / 2 for i in x],
        vals96,
        w,
        yerr=err96,
        capsize=3,
        label="L96",
        color=HIST_COLOR_L96,
        edgecolor="gray",        # hatch color
        linewidth=0.6,
        hatch="////",            # denser oblique
    )

    ax.bar(
        [i + w / 2 for i in x],
        vals192,
        w,
        yerr=err192,
        capsize=3,
        label="L192",
        color=HIST_COLOR_L192,
        edgecolor="gray",
        linewidth=0.6,
        hatch="oo",
    )
    # === END CHANGED ===

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FS)
    ax.axvline(2.5, linewidth=1)

    ax.set_ylabel("ms / iter", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_title(
        r"$\bf{" + dataset + "}$" + r" - $\bf{Training\ Time}$ (T = 96)",
        fontsize=TITLE_FS,
    )
    ax.legend(fontsize=LEGEND_FS)

    _finalize(fig, save_path, show)


def plot_ms_iter_histogram_cross_dataset(
    df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    """
    Compare training time (ms/iter) between ETTh1 and ECL for L=96 only.
    Bars now use a dense gray cross-hatch pattern.
    """
    sub_etth1 = _dataset_df(df, "ETTh1")
    sub_ecl = _dataset_df(df, "ECL")

    labels = []
    vals_etth1, vals_ecl = [], []
    err_etth1, err_ecl = [], []

    for model, token in ROW_ORDER:
        r_etth1 = sub_etth1[
            (sub_etth1["model"] == model)
            & (sub_etth1["token_style"] == token)
            & (sub_etth1["window"] == 96)
        ]
        r_ecl = sub_ecl[
            (sub_ecl["model"] == model)
            & (sub_ecl["token_style"] == token)
            & (sub_ecl["window"] == 96)
        ]

        if r_etth1.empty:
            raise ValueError(f"Missing row for ETTh1, {model}, {token}, L=96")
        if r_ecl.empty:
            raise ValueError(f"Missing row for ECL, {model}, {token}, L=96")

        rr_etth1 = r_etth1.iloc[0]
        rr_ecl = r_ecl.iloc[0]

        labels.append(f"{model}\n{token}")
        vals_etth1.append(rr_etth1["ms_per_iter_mean"])
        vals_ecl.append(rr_ecl["ms_per_iter_mean"])
        err_etth1.append(rr_etth1["ms_per_iter_std"])
        err_ecl.append(rr_ecl["ms_per_iter_std"])

    x = list(range(len(labels)))
    w = 0.36

    fig, ax = plt.subplots(figsize=(10.9, 4.9))

    x_left = [i - w / 2 for i in x]
    x_right = [i + w / 2 for i in x]

    # Pattern: '////' + '\\\\' = dense crosshatch
    # The number of slashes controls the density.
    hatch_pattern = "////\\\\\\\\" 

    ax.bar(
        x_left,
        vals_etth1,
        w,
        yerr=err_etth1,
        capsize=3,
        label="ETTh1",
        color="#d8b4f8",   
        edgecolor="gray",  # This also colors the hatch lines
        linewidth=0.6,
        hatch=hatch_pattern,
    )

    ax.bar(
        x_right,
        vals_ecl,
        w,
        yerr=err_ecl,
        capsize=3,
        label="ECL",
        color="#b7e4c7",   
        edgecolor="gray",  # This also colors the hatch lines
        linewidth=0.6,
        hatch=hatch_pattern,
    )

    # Note: Removed the ax.scatter star loops to clean up the bar interior

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FS)
    ax.axvline(2.5, linewidth=1, color="black", alpha=0.3)

    ax.set_ylabel("ms / iter", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_title(
        r"$\bf{Training\ Time}$" + r" comparison across datasets (L = 96, T = 96)",
        fontsize=TITLE_FS,
    )
    ax.legend(fontsize=LEGEND_FS)

    _finalize(fig, save_path, show)



def plot_ecl_efficiency_grid(
    df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    """3-column grid for ECL L=96 with manual offsets and oversized axis ticks."""
    sub = _dataset_df(df, "ECL")
    sub = sub[sub["window"] == 96].copy()

    fig, axes = plt.subplots(1, 3, figsize=(28, 12))
    fig.suptitle("ECL param size mb", fontsize=EFF_SUPTITLE_FS, fontweight="bold", y=1.05)

    metrics = [
        ("test_mse_mean", r"$\bf{Error}$ vs Size", "Test MSE"),
        ("ms_per_iter_mean", r"$\bf{Speed}$ vs Size", "ms / iter"),
        ("gpu_mem_gb_mean", r"$\bf{Memory}$ vs Size", "GPU Memory (GB)")
    ]

    for i, (col, title, ylabel) in enumerate(metrics):
        ax = axes[i]
        for _, r in sub.iterrows():
            m, t = r["model"], r["token_style"]
            key = (m, t)
            color = BUBBLE_COLORS[key]
            ax.scatter(r["size_mb"], r[col], marker="+", s=EFF_MARKER_SIZE, color=color, linewidth=4)
            
            # --- FINAL COORDINATE LOGIC ---
            ha, va = 'left', 'bottom'
            dx, dy = 16, 16

            if t == "variate":
                # Zig-zag cluster logic
                if m == "mamba": va, dy, ha, dx = 'top', -25, 'right', -15
                elif m == "transformer": va, dy, ha, dx = 'bottom', 25, 'center', 0
                elif m == "gru": 
                    va, dy, ha, dx = 'top', -25, 'left', 15
                    if col == "gpu_mem_gb_mean": # Plot 3 request: Gru-variate far left
                        ha, dx = 'right', -20
                        va, dy = 'center', 0
            else:
                # Time cluster logic
                if col == "test_mse_mean":
                    if m == "mamba": va, dy, ha, dx = 'top', -25, 'right', -15 # Plot 1 request: Mamba-time below
                    if m == "gru": va, dy = 'top', -25 
                
                if col == "ms_per_iter_mean" and m == "gru": va, dy = 'top', -25
                if m == "transformer": ha, dx = 'right', -18

            ax.annotate(
                f"{m}-{t}", (r["size_mb"], r[col]), xytext=(dx, dy), 
                textcoords="offset points", fontsize=EFF_ANNOTATION_FS, 
                color=color, fontweight="bold", ha=ha, va=va
            )
        
        ax.set_title(f"ECL: {title}\n(L=96, T=96)", fontweight="bold", fontsize=EFF_TITLE_FS, pad=35)
        ax.set_xlabel("Model Size (MB)", fontsize=EFF_AXIS_LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=EFF_AXIS_LABEL_FS)
        
        # Way bigger axis numbers
        ax.tick_params(axis='both', which='major', labelsize=EFF_TICK_FS)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        ax.set_xlim(-65, 285)
        if col == "test_mse_mean": ax.set_ylim(0.07, 0.35)
        elif col == "ms_per_iter_mean": ax.set_ylim(-5, 50)
        elif col == "gpu_mem_gb_mean": ax.set_ylim(-0.1, 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Custom finalize with high DPI for Word documents
    path = _ensure_parent(save_path)
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_all_current_plots(
    csv_path: PathLike,
    out_dir: PathLike = "windows11/figures",
    show: bool = False,
) -> None:
    df = load_aggregate_csv(csv_path=csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_test_metrics_grid(
        df=df,
        save_path=out_dir / "grid_test_metrics_T96.png",
        show=show,
    )

    plot_mse_histogram(
        df=df,
        dataset="ECL",
        save_path=out_dir / "ECL_test_mse_hist_L96_L192_T96.png",
        show=show,
    )
    plot_increasing_lookback(
        df=df,
        dataset="ECL",
        save_path=out_dir / "ECL_increasing_lookback_L96_L192_T96.png",
        show=show,
    )
    plot_bubble_ecl(
        df=df,
        save_path=out_dir / "ECL_bubble_L96_T96.png",
        show=show,
    )
    plot_ms_iter_histogram(
        df=df,
        dataset="ECL",
        save_path=out_dir / "ECL_training_time_hist_L96_L192_T96.png",
        show=show,
    )

    plot_mse_histogram(
        df=df,
        dataset="ETTh1",
        save_path=out_dir / "ETTh1_test_mse_hist_L96_L192_T96.png",
        show=show,
    )
    plot_increasing_lookback(
        df=df,
        dataset="ETTh1",
        save_path=out_dir / "ETTh1_increasing_lookback_L96_L192_T96.png",
        show=show,
    )
    plot_bubble_etth1(
        df=df,
        save_path=out_dir / "ETTh1_bubble_L96_T96.png",
        show=show,
    )
    plot_ms_iter_histogram(
        df=df,
        dataset="ETTh1",
        save_path=out_dir / "ETTh1_training_time_hist_L96_L192_T96.png",
        show=show,
    )
    plot_ms_iter_histogram_cross_dataset(
        df=df,
        save_path=out_dir / "cross_dataset_training_time_hist_L96_T96.png",
        show=show,
    )

    # Final Efficiency Grid call
    plot_ecl_efficiency_grid(
        df=df, 
        save_path=out_dir / "ECL_efficiency_param_grid_L96.png", 
        show=show
    )