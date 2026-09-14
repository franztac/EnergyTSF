import matplotlib.pyplot as plt
import torch
import numpy as np
import os


# ----------------------------------------------------------------------------
def plot_learning_curves(
    history, 
    show=True, 
    save_path=None,
    suptitle=None,
):
    """Plots MSE for training and validation across epochs."""
    plt.figure(figsize=(7, 4.8))
    
    epochs = range(len(history["train_mse"]))
    plt.plot(epochs, history["train_mse"], label="Train MSE", color="royalblue", lw=2)
    plt.plot(epochs, history["val_mse"], label="Val MSE", color="orange", lw=2)
    
    # Identify the best epoch 
    best_val = min(history["val_mse"])
    best_epoch = history["val_mse"].index(best_val)
    plt.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.6, label="Best Model")
    
    plt.title("Training and Validation MSE")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if suptitle:
        plt.suptitle(suptitle, fontsize=11, fontweight='bold', y=.98)
        plt.subplots_adjust(top=0.85)

    if save_path:
        plt.savefig(
            save_path, 
            dpi=150, 
            bbox_inches='tight', 
            facecolor='white', 
            transparent=False,
        )


    if show:
        plt.show()
    else:
        plt.close()






def plot_forecasts_from_npz(
    npz_path: str,
    window_indices=(0, 50, 100, 200),
    channel_idx: int = -1,
    save_dir: str | None = None,
    show: bool = True,
    verbose: bool = True,
    num_random: int = 6,          
    seed: int | None = None,      
    suptitle: str | None = None,
):
    """
    Plot forecasts using pre-saved arrays from evaluator outputs (.npz).
    Expects keys: Y (N,T,V), P (N,T,V).
    """

    filename = os.path.basename(npz_path)
    split_name = filename.split("_")[0] if "_" in filename else "unknown"

    data = np.load(npz_path)
    Y, P = data["Y"], data["P"]

    N, L, V = X.shape
    _, T, _ = Y.shape

    if verbose:
        print(f"[plot] Loaded NPZ : X{X.shape}, Y{Y.shape}, P{P.shape}")
        print(f"[plot] Available windows: N={N} | lookback L={L} | horizon T={T} | channels V={V}")

    # ---- window selection ----
    if window_indices is None or window_indices == "random":
        rng = np.random.default_rng(seed)
        k = min(int(num_random), N)
        window_indices = rng.choice(N, size=k, replace=False).tolist()

        if verbose:
            print(f"[plot] Using random window_indices (k={k}): {window_indices}")
    
    else:
        window_indices = list(window_indices)

    num_plots = len(window_indices)

    fig, axes = plt.subplots(
        num_plots, 1,
        figsize=(15, 4 * num_plots),
    )
    if num_plots == 1:
        axes = [axes]

    lookback_range = np.arange(L)
    forecast_range = np.arange(L, L + T)

    for i, win_idx in enumerate(window_indices):
        ax = axes[i]

        if win_idx >= N:
            if verbose:
                print(f"[plot][warn] window_idx {win_idx} out of range (valid: 0..{N-1}). Skipping subplot.")
            ax.set_title(f"⚠️ Window {win_idx} out of range (N={N})")
            ax.axis("off")
            continue
        
        # Lookback
        ax.plot(
            lookback_range,
            X[win_idx, :, channel_idx],
            label="Input (Lookback)",
            color="tab:blue",
            alpha=0.8,
        )

        # Truth
        ax.plot(
            forecast_range,
            Y[win_idx, :, channel_idx],
            label="Actual (Truth)",
            color="tab:blue",
            marker="o",
            markersize=3,
        )

        # Pred
        ax.plot(
            forecast_range,
            P[win_idx, :, channel_idx],
            label="Predicted",
            color="tab:orange",
            marker="x",
            markersize=3,
            linestyle="--",
        )

        ax.axvline(x=L - 1, color="black", linestyle=":", alpha=0.5)
        ax.set_title(
            f"{split_name.upper()} | Window={win_idx} | Channel={channel_idx}", fontsize=13, loc='left'
        )
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left')
    

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=.99)
        plt.subplots_adjust(top=0.94, hspace=0.3)


    # Save one consolidated  (avoid overwrite)
    out_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base_name = f"{split_name}_{num_plots}_forecasts_ch{channel_idx}.png"
        out_path = os.path.join(save_dir, base_name)

        # if exists, append _v2, _v3, ...
        if os.path.exists(out_path):
            v = 2
            while True:
                candidate = os.path.join(
                    save_dir,
                    f"{split_name}_{num_plots}_forecasts_ch{channel_idx}_v{v}.png"
                )
                if not os.path.exists(candidate):
                    out_path = candidate
                    break
                v += 1

        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        if verbose:
            print(f"[plot] Saved -> {out_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return out_path