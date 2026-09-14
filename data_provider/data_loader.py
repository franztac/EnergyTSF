import gc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
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
    def temporal_split_raw(self, X: np.ndarray, y:np.ndarray):
        """Split data into train, val, and test sets based on ratios."""
        ratios = self.ratios
        N = len(X)
        train_cut = int(N * ratios[0])
        val_cut = int(N * (ratios[0] + ratios[1]))

        X_train, y_train = X[:train_cut], y[:train_cut]
        X_val, y_val = X[train_cut:val_cut], y[train_cut:val_cut]
        X_test, y_test = X[val_cut:], y[val_cut:]

        if self.verbose:
            print(f"🧩 Created raw splits:")
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
        
        # Stack into giant arrays
        X_seq = np.stack(X_list)
        y_seq = np.stack(y_list)

        # IMMEDIATELY clear the lists to free up RAM
        del X_list, y_list
        gc.collect()

        if self.verbose:
            print(f"🔁 Created windowed data: X_seq {X_seq.shape}, y_seq {y_seq.shape}")

        return X_seq, y_seq
    
    


    # ------------------------------------------------------------------
    def setup(
        self,
        seed: int | None = None, 
        verbose: bool | None = None,
    ):
        """Full pipeline: Build features, split, normalize, window, and return loaders."""
        
        if verbose is not None:
            self.verbose = verbose

        # DataModule Summary ---
        if self.verbose:
            print(f"\n--- 📊 DataModule: {self.name} ---")
            print(f"Window: {self.window} | Horizon: {self.horizon} | Stride: {self.stride}")
            print(f"Batch Size: {self.batch_size} | Ratios: {self.ratios}")
            
        X_full, Y_full = self.build_features()

        # temporal splits
        splits = self.temporal_split_raw(X_full, Y_full)
        (X_train_raw, y_train_raw), (X_val_raw, y_val_raw), (X_test_raw, y_test_raw) = splits

        # normalization
        self.scaler_x.fit(X_train_raw)   # compute mean/std per column (d_tot)
        self.scaler_y.fit(y_train_raw)   # compute mean/std for targets (raw features)

        # 1. Process Train
        X_train = self.scaler_x.transform(X_train_raw)
        y_train = self.scaler_y.transform(y_train_raw)
        X_train_seq, y_train_seq = self.create_windows(X_train, y_train, stride=self.stride)
        self.X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
        del X_train_seq, y_train_seq, X_train, y_train # CLEANUP

        # 2. Process Val
        X_val = self.scaler_x.transform(X_val_raw)
        y_val = self.scaler_y.transform(y_val_raw)
        X_val_seq, y_val_seq = self.create_windows(X_val, y_val, stride=self.stride)
        self.X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
        self.y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
        del X_val_seq, y_val_seq, X_val, y_val # CLEANUP

        # 3. Process Test
        X_test = self.scaler_x.transform(X_test_raw)
        y_test = self.scaler_y.transform(y_test_raw)
        X_test_seq, y_test_seq = self.create_windows(X_test, y_test, stride=self.stride)
        self.X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)
        del X_test_seq, y_test_seq, X_test, y_test # CLEANUP

        import gc
        gc.collect()

        # Datasets & loaders
        self.train_ds = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.val_ds = TensorDataset(self.X_val_tensor, self.y_val_tensor)
        self.test_ds = TensorDataset(self.X_test_tensor, self.y_test_tensor)

        # Deterministic shuffle (optional)
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        self.train_loader = DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            generator=generator,
            worker_init_fn=seed_worker,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        self.val_loader = DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        
        self.test_loader = DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        
        # Explicitly tag the loaders so the Evaluator knows what they are
        self.train_loader.split_name = "train"
        self.val_loader.split_name = "val"
        self.test_loader.split_name = "test"

        if self.verbose:
            bx, by = next(iter(self.train_loader))
            print(f"📦 Train batch → X: {bx.shape}, Y: {by.shape}")

        return self.train_loader, self.val_loader, self.test_loader



