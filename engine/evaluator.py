"""
Responsibilities:
- Run ONE forward pass over a DataLoader
- Compute scalar metrics (MSE, MAE) on the normalized scale

Two evaluation styles are supported:

1) "full" path (existing idea):
   - can collect arrays for later saving
   - useful when you really want cached outputs

2) "shallow" path (NEW):
   - streams metrics batch-by-batch
   - does NOT keep X
   - optionally caches only Y and P
   - intended for large datasets like ECL

Plotting is intentionally kept outside this file.
"""

from __future__ import annotations
import gc
import os
import json
from typing import Dict, Tuple, Optional, Iterable
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
            print(f"[evaluator] Saved {tag} -> {path}")

    
    def _save_npz_yp(
        self,
        *,
        path: str,
        Y: np.ndarray,
        P: np.ndarray,
        verbose: bool,
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, Y=Y, P=P)
        if verbose:
            print(f"[evaluator] Saved NPZ (Y,P only) -> {path}")


    def _compute_metrics_from_numpy(
        self,
        *,
        P: np.ndarray,
        Y: np.ndarray,
    ) -> Dict[str, float]:
        """Scalar metrics on numpy arrays."""
        p = torch.from_numpy(P)
        y = torch.from_numpy(Y)
        mse = float(self.mse_criterion(p, y).item())
        mae = float(self.mae_criterion(p, y).item())
        return {"mse": mse, "mae": mae}


    
     # ------------------------------------------------------------------
    # Existing full collector (kept for backward compatibility)
    # ------------------------------------------------------------------
    def _collect_outputs(
        self,
        *,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        verbose: bool,
        desc: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect ALL X, Y, P from a loader in a single inference pass.

        NOTE:
        This is the heavier path. It is kept only for compatibility.
        For large datasets, prefer the new shallow methods below.

        Returns:
          X: [N, L, V], Y: [N, T, V], P: [N, T, V]
        """
        model.eval()
        all_x, all_y, all_p = [], [], []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                preds = model(xb)  #  [B, T, V]

                all_x.append(xb.detach().cpu())
                all_y.append(yb.detach().cpu())
                all_p.append(preds.detach().cpu())

                del xb, yb, preds

        # Concatenate on batch dimension -> [N, ...]
        X = torch.cat(all_x, dim=0).numpy()
        Y = torch.cat(all_y, dim=0).numpy()
        P = torch.cat(all_p, dim=0).numpy()

        del all_x, all_y, all_p
        gc.collect()

        if verbose:
            print(f"[evaluator] Inference complete: X{X.shape}, Y{Y.shape}, P{P.shape}")

        return X, Y, P

    # ------------------------------------------------------------------
    # NEW: shallow helpers
    # ------------------------------------------------------------------
    def _stream_metrics_only(
        self,
        *,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        verbose: bool,
    ) -> Dict[str, float]:
        """
        NEW:
        Memory-light evaluation.
        - no X kept
        - no Y/P arrays kept
        - metrics accumulated batch by batch
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
                f"[evaluator][shallow] done -> "
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
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        NEW:
        Still memory-aware, but caches ONLY Y and P.
        X is intentionally ignored.

        This is much lighter than the old full collector.
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
                f"[evaluator][shallow+npz] done -> "
                f"Y{Y.shape}, P{P.shape} | "
                f"mse={metrics['mse']:.6f} | mae={metrics['mae']:.6f}"
            )

        return metrics, Y, P



    
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
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
        Evaluate ONE loader. 
        Files are saved based on save_json/save_npz flags.
        Logs are printed based on verbose flag.
        """

        if split is None:
            split = getattr(loader, "split_name", "unknown")

        # 1) Collect outputs once
        X, Y, P = self._collect_outputs(
            model=model,
            loader=loader, 
            verbose=verbose,
        )

        # 2) Compute scalar metrics
        metrics = self._compute_metrics(P=P, Y=Y)

        # 3) Save metrics JSON
        payload = {
            "split": split,
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "num_windows": int(X.shape[0]),
            "batch_size": int(getattr(loader, "batch_size", 0)),
        }

        if save_json:
            json_path = os.path.join(save_dir, f"{split}_metrics.json")
            self._save_json(
                path=json_path, 
                payload=payload, 
                verbose=verbose,
            )

        # 4) Optionally save cached outputs NPZ for later plotting
        if save_npz:
            # Backward-compatible save for old path only.
            npz_path = os.path.join(save_dir, f"{split}_outputs.npz")
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)
            np.savez_compressed(npz_path, X=X, Y=Y, P=P)
            if verbose:
                print(f"[evaluator] Saved NPZ  -> {npz_path}")
        
        del X, Y, P
        gc.collect()

        return payload
    


    def evaluate_splits(
        self,
        *,
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: str,
        save_npz_splits: Optional[Iterable[str]] = ("test",), 
        save_summary: bool = True,
        verbose: bool = True,
        post_training_filename: str = "post_training_metrics.json",
    ) -> Dict[str, Dict]:
        """
        Runs evaluation on all 3 splits. 
        Saves ONLY one consolidated JSON and NPZs for selected splits.
        Runs evaluation. 100% in-memory if save_summary=False and save_npz_splits=None.
        """
        results: Dict[str, Dict] = {}
        npz_targets = save_npz_splits if save_npz_splits is not None else []
        
        if verbose:
            print(f"\n--- 🧪 Starting Split Evaluation (NPZ targets: {list(npz_targets)}) ---")

        # Only create directory if we are actually writing something
        if save_summary or len(list(npz_targets)) > 0:
            os.makedirs(save_dir, exist_ok=True)

        # Define the mapping to iterate through
        split_map = [
            ("val", val_loader), 
            ("test", test_loader)
        ]

        for name, loader in split_map:
            if verbose: print(f"\n[evaluator] Processing split: {name}...")
            
            payload = self.evaluate(
                model=model,
                loader=loader,
                save_dir=save_dir,
                split=name,
                save_json=False,  # We don't want the individual files
                save_npz=(name in npz_targets), 
                verbose=verbose,
            )
            
            # Store in summary dictionary
            results[name] = {
                "mse": payload["mse"],
                "mae": payload["mae"],
                "num_windows": payload["num_windows"],
            }

        # Only save final summary JSON if flag is True
        if save_summary:
            summary_path = os.path.join(save_dir, post_training_filename)
            self._save_json(
                path=summary_path, 
                payload=results, 
                verbose=verbose,
                tag="Summary JSON",
            )
        elif verbose:
            print("[evaluator] save_summary is False: skipping JSON write.")

        return results
    

    # ------------------------------------------------------------------
    # NEW public shallow API
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
        NEW:
        Shallow single-split evaluation.

        - metrics are always computed in streaming mode
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
        NEW:
        Memory-light multi-split evaluation.

        Example split_loaders:
            {"test": test_loader}
        or:
            {"val": val_loader, "test": test_loader}

        This is the path that should be used for large datasets.
        """
        results: Dict[str, Dict] = {}
        npz_targets = set(save_npz_splits or [])

        if save_summary or len(npz_targets) > 0:
            os.makedirs(save_dir, exist_ok=True)

        for split_name, loader in split_loaders.items():
            if verbose:
                print(f"[evaluator][shallow] Processing split: {split_name}")

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
    


    
    


