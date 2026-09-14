import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Optional, Union, List

# Importing centralized style constants
import windows11.utils.visual_utils as vu

PathLike = Union[str, Path]

def _format_stat_val(val: float) -> str:
    """Helper to format value: bold minus sign with 1 space to its LEFT."""
    s = f"{val:.2f}"
    if val < 0:
        return s.replace("-", r" $\mathbf{-}$")
    return s

def _plot_logic(ax, data_train, data_test, channel_name, use_density):
    """Internal core logic: Manual markers, stats below, enlarged fonts, 15 bins."""
    mu_train, std_train = np.mean(data_train), np.std(data_train)
    mu_test, std_test = np.mean(data_test), np.std(data_test)
    
    bins = 15 
    train_col = '#1f77b4'
    test_col = '#ff7f0e'

    # 1. Draw Histograms
    ax.hist(data_train, bins=bins, density=use_density, alpha=0.5, 
            color=train_col, edgecolor='white', linewidth=0.5)
    
    ax.hist(data_test, bins=bins, density=use_density, alpha=0.4, 
            color=test_col, edgecolor='white', linewidth=0.5)
    
    # 2. Styling & X-Axis Clamp
    ax.set_xlim(-3.5, 3.5)
    ax.set_title(f"Channel {channel_name}", fontsize=vu.GRID_TITLE_FS + 6, pad=10)
    ax.tick_params(axis='both', labelsize=vu.TICK_FS)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 3. Stats Text & Colored Dots placed BELOW axis
    label_fs = vu.LEGEND_FS * 1.30
    y_base = -0.18 
    line_spacing = 0.07

    train_str = f"Train ($\mathbf{{\mu}}$={_format_stat_val(mu_train)}, $\mathbf{{\sigma}}$={_format_stat_val(std_train)})"
    test_str = f"Test  ($\mathbf{{\mu}}$={_format_stat_val(mu_test)}, $\mathbf{{\sigma}}$={_format_stat_val(std_test)})"

    # Draw Colored Dots (Scatter) and Labels
    ax.scatter(0.20, y_base, color=train_col, s=80, transform=ax.transAxes, clip_on=False)
    ax.text(0.25, y_base, train_str, transform=ax.transAxes, ha='left', va='center', fontsize=label_fs)

    ax.scatter(0.20, y_base - line_spacing, color=test_col, s=80, transform=ax.transAxes, clip_on=False)
    ax.text(0.25, y_base - line_spacing, test_str, transform=ax.transAxes, ha='left', va='center', fontsize=label_fs)

def plot_distribution_shift(
    dataset_name: str,
    channel_idx: int,
    ratios: tuple = (0.7, 0.15, 0.15),
    use_density: bool = True,
    save_path: Optional[PathLike] = None,
    show: bool = True,
    verbose: bool = True
) -> None:
    """Plots a single channel distribution comparison."""
    data_map = {'ETTh1': 'data/ETT/ETTh1.csv', 'ECL': 'data/ECL/electricity.csv'}
    path = data_map.get(dataset_name)
    if not path or not Path(path).exists(): return

    df = pd.read_csv(path)
    data = df[[c for c in df.columns if c != "date"]].values.astype(np.float32)
    
    n = len(data)
    train_end = int(n * ratios[0])
    test_start = int(n * (ratios[0] + ratios[1]))
    train_raw, test_raw = data[:train_end], data[test_start:]
    
    scaler = StandardScaler().fit(train_raw)
    train_scaled = scaler.transform(train_raw)[:, channel_idx]
    test_scaled = scaler.transform(test_raw)[:, channel_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6.5))
    _plot_logic(ax, train_scaled, test_scaled, channel_idx, use_density)
    
    ax.set_title(r"$\bf{" + dataset_name + "}$" + f" — distribution shift (Channel {channel_idx})", 
                 fontsize=vu.TITLE_FS + 6, pad=35)
    ax.set_xlabel("Standardized Values (relative to Train)", fontsize=vu.AXIS_LABEL_FS + 2)
    ax.set_ylabel("Density" if use_density else "Frequency", fontsize=vu.AXIS_LABEL_FS + 6)

    if save_path:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)

def plot_distribution_shift_grid(
    dataset_name: str,
    channel_indices: List[int],
    ratios: tuple = (0.7, 0.15, 0.15),
    use_density: bool = True,
    save_path: Optional[PathLike] = None,
    show: bool = True,
    verbose: bool = True
) -> None:
    """Plots a row of 3 channels with stats below and balanced spacing."""
    data_map = {'ETTh1': 'data/ETT/ETTh1.csv', 'ECL': 'data/ECL/electricity.csv'}
    path = data_map.get(dataset_name)
    if not path or not Path(path).exists(): return

    df = pd.read_csv(path)
    data = df[[c for c in df.columns if c != "date"]].values.astype(np.float32)

    if verbose: print(f"📊 Dataset: {dataset_name} | Total Channels: {data.shape[1]}")

    n = len(data)
    train_end = int(n * ratios[0])
    test_start = int(n * (ratios[0] + ratios[1]))
    train_raw, test_raw = data[:train_end], data[test_start:]
    
    scaler = StandardScaler().fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    
    fig.suptitle(r"$\bf{" + dataset_name + "}$" + " — Distribution Shift Multi-Channel Glimpse", 
                 fontsize=vu.TITLE_FS + 10, y=0.98)

    for i, ch_idx in enumerate(channel_indices[:3]):
        _plot_logic(axes[i], train_scaled[:, ch_idx], test_scaled[:, ch_idx], ch_idx, use_density)
        if i == 0:
            axes[i].set_ylabel("Density" if use_density else "Frequency", fontsize=vu.AXIS_LABEL_FS + 6)

    plt.subplots_adjust(top=0.80, bottom=0.25, wspace=0.35)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        if verbose: print(f"💾 Saved Grid Plot: {save_path}")
    if show: plt.show()
    else: plt.close(fig)