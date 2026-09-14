import gc

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from engine.seed import seed_worker


class EnergyDataModule:
    def __init__(
        self,
        df: pd.DataFrame,
        name="Dataset",
        window=96,
        horizon=96,
        stride=1,
        batch_size=128,
        ratios=(0.7, 0.15, 0.15),
        verbose=True,
    ):
        self.df = df.copy()
        self.name = name
        self.window = window
        self.horizon = horizon
        self.stride = stride
        self.batch_size = batch_size
        self.ratios = ratios
        self.verbose = verbose

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    # ------------------------------------------------------------------
    def build_features(self):
        """Return raw inputs and raw outputs."""
        raw = self.df[[c for c in self.df.columns if c != "date"]].values.astype(np.float32)
        return raw, raw.copy()

    # ------------------------------------------------------------------
    def temporal_split_raw(self, X: np.ndarray, y: np.ndarray):
        """Split data into train, val, and test sets based on ratios."""
        ratios = self.ratios
        N = len(X)
        train_cut = int(N * ratios[0])
        val_cut = int(N * (ratios[0] + ratios[1]))

        X_train, y_train = X[:train_cut], y[:train_cut]
        X_val, y_val = X[train_cut:val_cut], y[train_cut:val_cut]
        X_test, y_test = X[val_cut:], y[val_cut:]

        if self.verbose:
            print("🧩 Created raw splits:")
            print(f"   Train: X={X_train.shape}, y={y_train.shape}")
            print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
            print(f"   Test:  X={X_test.shape}, y={y_test.shape}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # ------------------------------------------------------------------
    def create_windows(self, X: np.ndarray, y: np.ndarray, stride=1):
        """Create sliding windows for time series forecasting."""
        N = len(X)
        X_list, y_list = [], []

        for start in range(0, N - self.window - self.horizon + 1, stride):
            end = start + self.window
            target_end = end + self.horizon
            X_list.append(X[start:end])
            y_list.append(y[end:target_end])

        X_seq = np.stack(X_list)
        y_seq = np.stack(y_list)

        del X_list, y_list
        gc.collect()

        if self.verbose:
            print(f"🔁 Created windowed data: X_seq {X_seq.shape}, y_seq {y_seq.shape}")

        return X_seq, y_seq

    # ------------------------------------------------------------------
    def _prepare_split_tensors(
        self,
        *,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
    ):
        """
        Prepare tensors for exactly one split.
        This lets evaluation build only test instead of train+val+test.
        """
        X = self.scaler_x.transform(X_raw)
        y = self.scaler_y.transform(y_raw)

        X_seq, y_seq = self.create_windows(X, y, stride=self.stride)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)

        del X, y, X_seq, y_seq
        gc.collect()

        return X_tensor, y_tensor

    def _make_loader(
        self,
        *,
        X_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        shuffle: bool,
        seed: int | None,
        split_name: str,
    ):
        ds = TensorDataset(X_tensor, y_tensor)

        generator = None
        if shuffle and seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
            worker_init_fn=seed_worker if shuffle else None,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        loader.split_name = split_name
        return loader

    # ------------------------------------------------------------------
    def setup_selected(
        self,
        *,
        include_train: bool = True,
        include_val: bool = True,
        include_test: bool = True,
        seed: int | None = None,
        verbose: bool | None = None,
    ):
        """
        Build only the requested splits.
        """
        if verbose is not None:
            self.verbose = verbose

        if self.verbose:
            print(f"\n--- 📊 DataModule: {self.name} ---")
            print(f"Window: {self.window} | Horizon: {self.horizon} | Stride: {self.stride}")
            print(f"Batch Size: {self.batch_size} | Ratios: {self.ratios}")
            print(
                f"Requested splits -> "
                f"train={include_train}, val={include_val}, test={include_test}"
            )

        X_full, Y_full = self.build_features()
        splits = self.temporal_split_raw(X_full, Y_full)
        (X_train_raw, y_train_raw), (X_val_raw, y_val_raw), (X_test_raw, y_test_raw) = splits

        # Fit scalers on train raw data only
        self.scaler_x.fit(X_train_raw)
        self.scaler_y.fit(y_train_raw)

        train_loader = None
        val_loader = None
        test_loader = None

        if include_train:
            X_train_tensor, y_train_tensor = self._prepare_split_tensors(
                X_raw=X_train_raw,
                y_raw=y_train_raw,
            )
            train_loader = self._make_loader(
                X_tensor=X_train_tensor,
                y_tensor=y_train_tensor,
                shuffle=True,
                seed=seed,
                split_name="train",
            )

        if include_val:
            X_val_tensor, y_val_tensor = self._prepare_split_tensors(
                X_raw=X_val_raw,
                y_raw=y_val_raw,
            )
            val_loader = self._make_loader(
                X_tensor=X_val_tensor,
                y_tensor=y_val_tensor,
                shuffle=False,
                seed=None,
                split_name="val",
            )

        if include_test:
            X_test_tensor, y_test_tensor = self._prepare_split_tensors(
                X_raw=X_test_raw,
                y_raw=y_test_raw,
            )
            test_loader = self._make_loader(
                X_tensor=X_test_tensor,
                y_tensor=y_test_tensor,
                shuffle=False,
                seed=None,
                split_name="test",
            )

        del X_full, Y_full
        del X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw
        gc.collect()

        if self.verbose and train_loader is not None:
            bx, by = next(iter(train_loader))
            print(f"📦 Train batch → X: {bx.shape}, Y: {by.shape}")
            del bx, by

        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------
    def setup(
        self,
        seed: int | None = None,
        verbose: bool | None = None,
    ):
        """
        Backward-compatible path used by training.
        """
        return self.setup_selected(
            include_train=True,
            include_val=True,
            include_test=True,
            seed=seed,
            verbose=verbose,
        )