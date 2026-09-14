"""
Shallow evaluator for post-training evaluation.

Main goals:
- stream metrics batch-by-batch
- avoid keeping X
- optionally save only Y and P (for plotting)
- support test-only evaluation by default from the caller

This file is intentionally separate from engine/evaluator.py
so your original evaluator stays untouched.
"""

from __future__ import annotations

import gc
import json
import os
from typing import Dict, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn


class Evaluator:
    def __init__(self, device: torch.device):
        self.device = device
        self.mse_criterion = nn.MSELoss(reduction="mean")
        self.mae_criterion = nn.L1Loss(reduction="mean")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _save_json(
        self,
        *,
        path: str,
        payload: Dict,
        verbose: bool,
        tag: str = "JSON",
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
        if verbose:
            print(f"[evaluator_tmp] Saved {tag} -> {path}")

    def _save_npz_yp(
        self,
        *,
        path: str,
        Y: np.ndarray,
        P: np.ndarray,
        verbose: bool,
    ) -> None:
        """
        Save only Y and P.
        We intentionally do NOT save X to reduce RAM and disk usage.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, Y=Y, P=P)
        if verbose:
            print(f"[evaluator_tmp] Saved NPZ (Y,P only) -> {path}")

    def _stream_metrics_only(
        self,
        *,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        verbose: bool,
    ) -> Dict[str, float]:
        """
        Memory-light evaluation:
        - no X kept
        - no Y/P arrays kept
        - metrics accumulated batch-by-batch
        """
        model.eval()

        total_mse = 0.0
        total_mae = 0.0
        total_windows = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                preds = model(xb)

                batch_size = xb.size(0)
                mse = self.mse_criterion(preds, yb)
                mae = self.mae_criterion(preds, yb)

                total_mse += float(mse.item()) * batch_size
                total_mae += float(mae.item()) * batch_size
                total_windows += batch_size

                del xb, yb, preds, mse, mae

        if total_windows == 0:
            raise ValueError("Loader produced zero windows during evaluation.")

        metrics = {
            "mse": total_mse / total_windows,
            "mae": total_mae / total_windows,
            "num_windows": int(total_windows),
        }

        if verbose:
            print(
                f"[evaluator_tmp][metrics-only] "
                f"windows={metrics['num_windows']} | "
                f"mse={metrics['mse']:.6f} | mae={metrics['mae']:.6f}"
            )

        return metrics

    def _stream_metrics_and_collect_yp(
        self,
        *,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        verbose: bool,
    ):
        """
        Still memory-aware, but collects only Y and P.
        X is intentionally ignored.
        """
        model.eval()

        total_mse = 0.0
        total_mae = 0.0
        total_windows = 0

        all_y = []
        all_p = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                preds = model(xb)

                batch_size = xb.size(0)
                mse = self.mse_criterion(preds, yb)
                mae = self.mae_criterion(preds, yb)

                total_mse += float(mse.item()) * batch_size
                total_mae += float(mae.item()) * batch_size
                total_windows += batch_size

                all_y.append(yb.detach().cpu())
                all_p.append(preds.detach().cpu())

                del xb, yb, preds, mse, mae

        if total_windows == 0:
            raise ValueError("Loader produced zero windows during evaluation.")

        Y = torch.cat(all_y, dim=0).numpy()
        P = torch.cat(all_p, dim=0).numpy()

        del all_y, all_p
        gc.collect()

        metrics = {
            "mse": total_mse / total_windows,
            "mae": total_mae / total_windows,
            "num_windows": int(total_windows),
        }

        if verbose:
            print(
                f"[evaluator_tmp][metrics+npz] "
                f"Y{Y.shape}, P{P.shape} | "
                f"mse={metrics['mse']:.6f} | mae={metrics['mae']:.6f}"
            )

        return metrics, Y, P

    # ------------------------------------------------------------------
    # Public shallow API
    # ------------------------------------------------------------------
    def evaluate_split_shallow(
        self,
        *,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        save_dir: str,
        split: Optional[str] = None,
        save_json: bool = True,
        save_npz: bool = False,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate one split in shallow mode.

        - metrics are always streaming
        - if save_npz=True, only Y and P are cached and saved
        - X is never kept
        """
        if split is None:
            split = getattr(loader, "split_name", "unknown")

        if save_npz:
            metrics, Y, P = self._stream_metrics_and_collect_yp(
                model=model,
                loader=loader,
                verbose=verbose,
            )
        else:
            metrics = self._stream_metrics_only(
                model=model,
                loader=loader,
                verbose=verbose,
            )
            Y, P = None, None

        payload = {
            "split": split,
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "num_windows": metrics["num_windows"],
            "batch_size": int(getattr(loader, "batch_size", 0)),
        }

        if save_json:
            json_path = os.path.join(save_dir, f"{split}_metrics.json")
            self._save_json(
                path=json_path,
                payload=payload,
                verbose=verbose,
            )

        if save_npz:
            npz_path = os.path.join(save_dir, f"{split}_outputs.npz")
            self._save_npz_yp(
                path=npz_path,
                Y=Y,
                P=P,
                verbose=verbose,
            )
            del Y, P
            gc.collect()

        return payload

    def evaluate_splits_shallow(
        self,
        *,
        model: torch.nn.Module,
        split_loaders: Dict[str, torch.utils.data.DataLoader],
        save_dir: str,
        save_npz_splits: Optional[Iterable[str]] = ("test",),
        save_summary: bool = True,
        verbose: bool = True,
        post_training_filename: str = "post_training_metrics.json",
    ) -> Dict[str, Dict]:
        """
        Evaluate one or more splits in shallow mode.

        Typical use:
            {"test": test_loader}

        Optional:
            {"val": val_loader, "test": test_loader}
        """
        results: Dict[str, Dict] = {}
        npz_targets = set(save_npz_splits or [])

        if save_summary or len(npz_targets) > 0:
            os.makedirs(save_dir, exist_ok=True)

        for split_name, loader in split_loaders.items():
            if verbose:
                print(f"[evaluator_tmp] Processing split: {split_name}")

            payload = self.evaluate_split_shallow(
                model=model,
                loader=loader,
                save_dir=save_dir,
                split=split_name,
                save_json=False,
                save_npz=(split_name in npz_targets),
                verbose=verbose,
            )

            results[split_name] = {
                "mse": payload["mse"],
                "mae": payload["mae"],
                "num_windows": payload["num_windows"],
            }

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if save_summary:
            summary_path = os.path.join(save_dir, post_training_filename)
            self._save_json(
                path=summary_path,
                payload=results,
                verbose=verbose,
                tag="Summary JSON",
            )

        return results