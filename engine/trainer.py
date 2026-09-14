import torch
import torch.nn as nn
import time
import os
import json
import numpy as np
from utils.visualization import plot_learning_curves
from utils.run_io import save_run_summary    



class Trainer:
    """
    Saves:
    - Train artifacts inside save_artifacts_dir (.../timestamp/train)
    - Run metadata JSON inside save_run_dir (.../timestamp)
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        device,
        patience: int = 10,
        save_artifacts_dir: str | None = None,
        save_run_dir: str | None = None,
        run_config=None,
        verbose: bool = True,
    ):  
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device=device
        self.patience=patience
        self.verbose = verbose

        self.save_artifacts_dir = save_artifacts_dir
        self.save_run_dir = save_run_dir
        self.run_config = run_config


        # Directory creation is now optional
        if self.save_artifacts_dir:
            os.makedirs(self.save_artifacts_dir, exist_ok=True)
        if self.save_run_dir is not None:
            os.makedirs(self.save_run_dir, exist_ok=True)

        
        # Tracking best
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.best_metrics = {}
        
        # History (epoch-level)
        self.history = {
            "train_mse": [], 
            "train_mae": [],
            "val_mse": [], 
            "val_mae": [],
            "ms_per_iter": [], 
            "gpu_mem_gb": []
        }
        self.mae_criterion = nn.L1Loss()  

        # timing 
        self._fit_start_time = None
        self._fit_end_time = None

        # Guard: avoid writing run_summary.json multiple times
        self._run_summary_saved = False


    def train_one_epoch(self, loader):
        """One training epoch. Returns: mse, mae, ms_per_iter, gpu_mem_gb."""
        self.model.train()
        total_mse, total_mae, count = 0.0, 0.0, 0
        batch_times = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            start_time = time.time() 

            self.optimizer.zero_grad()
            pred = self.model(xb)
            loss = self.criterion(pred, yb)
            mae = self.mae_criterion(pred, yb)
            loss.backward()
            self.optimizer.step()

            # batch speed
            batch_times.append((time.time() - start_time) * 1000)

            batch_size = xb.size(0)
            total_mse += loss.item() * batch_size
            total_mae += mae.item() * batch_size
            count += batch_size
        
        ms_iter = float(np.mean(batch_times)) if len(batch_times) > 0 else 0.0
        mem = (
            torch.cuda.memory_allocated(self.device) / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        
        return total_mse / count, total_mae / count, ms_iter, mem




    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluation loop. Returns dataset-average mse and mae."""
        self.model.eval()
        total_mse, total_mae, count = 0.0, 0.0, 0

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = self.model(xb)
            mse = self.criterion(pred, yb)
            mae = self.mae_criterion(pred, yb)

            batch_size = xb.size(0)
            total_mse += mse.item() * batch_size
            total_mae += mae.item() * batch_size
            count += batch_size

        return total_mse / count, total_mae / count
    


    def _save_run_summary(self, save_all: bool, verbose: bool):
        """
        Writes run_summary.json into save_run_dir once per Trainer instance.
        """
        if not save_all:
            return
        if self._run_summary_saved:
            return
        if self.save_run_dir is None or self.run_config is None:
            return
        
        json_path = save_run_summary(
            save_run_dir=self.save_run_dir,
            run_config=self.run_config,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            filename="run_summary.json",  # default
        )
        self._run_summary_saved = True

        if verbose:
            print(f"🧾 Run summary saved to {json_path}")

    

    def fit(
        self, 
        train_loader,
        val_loader, 
        num_epochs: int, 
        trial=None,
        save_all: bool = True,
        verbose: bool = True,
    ):
        
        # Save run metadata at start
        self._save_run_summary(save_all=save_all, verbose=verbose)

        if verbose:
            print(f"Starting training... Checkpoints: {self.save_artifacts_dir}")

        # start wall-clock timer
        self._fit_start_time = time.time()
        epochs_ran = 0

        for epoch in range(num_epochs):
            epochs_ran += 1

            tr_mse, tr_mae, ms_it, mem = self.train_one_epoch(train_loader)
            val_mse, val_mae = self.evaluate(val_loader)


            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_mse)
            else:
                self.scheduler.step()

            # Store history
            for k, v in zip(
                self.history.keys(),
                [tr_mse, tr_mae, val_mse, val_mae, ms_it, mem]
            ):
                self.history[k].append(v)

            if verbose:
                print(f"Epoch {epoch:02d} | Tr_MSE: {tr_mse:.4f} | Val_MSE: {val_mse:.4f}")


            # Best checkpoint
            if val_mse < self.best_val_loss:
                self.best_val_loss = val_mse
                self.best_epoch = epoch
                self.epochs_no_improve = 0

                self.best_metrics = {
                    "best_epoch": epoch,
                    "best_train_loss": tr_mse, 
                    "best_val_loss": val_mse,
                    "best_train_mae": tr_mae,
                    "best_val_mae": val_mae,
                    "ms_per_iter": ms_it,
                    "gpu_mem_gb": mem,
                }

                if save_all and self.save_artifacts_dir:
                    torch.save(
                        self.model.state_dict(), 
                        os.path.join(self.save_artifacts_dir, "best_model.pt")
                    )
            else:
                self.epochs_no_improve += 1


            # Early stopping
            if self.epochs_no_improve >= self.patience:
                if verbose:
                    print(f"\n🚨 Early stopping triggered at epoch {epoch} (patience={self.patience})")
                    print(
                        f"🏆 Best Epoch: {self.best_epoch} | "
                        f"Best Train Loss: {self.best_metrics['best_train_loss']:.6f} | "
                        f"Best Val Loss: {self.best_val_loss:.6f}"
                    )
                break


        # Final timing
        self._fit_end_time = time.time()
        training_time_sec = float(self._fit_end_time - self._fit_start_time)  
        
        # Add run-level stats into best_metrics
        self.best_metrics["epochs_ran"] = int(epochs_ran)
        self.best_metrics["training_time_sec"] = float(training_time_sec)


        self._save_final_artifacts(save_all=save_all, verbose=verbose)
        
        return self.history
    



    def _save_final_artifacts(self, save_all: bool, verbose:bool = True):
        """Save JSON metrics, full history, and learning curve plot.
        No disk writes for metrics.npz, best_metrics.json, or plots if save_all=False.
        """
        if not save_all or self.save_artifacts_dir is None:
            if verbose and not save_all:
                print("[trainer] save_all=False: skipping artifacts.")
            return
        
        # ... rest of saving logic
        # 1) best metrics
        with open(os.path.join(self.save_artifacts_dir, "best_metrics.json"), "w") as f:
            json.dump(self.best_metrics, f, indent=4)

        # 2) history
        np.savez_compressed(
            os.path.join(self.save_artifacts_dir, "metrics.npz"), 
            **self.history,
        )

        # 3) plot
        plot_learning_curves(
            history=self.history, 
            show=False,
            save_path=os.path.join(self.save_artifacts_dir, "epoch_scores.png"),
        ) 
        
        if verbose:
            print(f"✅ Training artifacts saved to {self.save_artifacts_dir}")
        