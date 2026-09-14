import torch
import torch.nn as nn


@torch.no_grad()
def test_model(model, loader, device, verbose=False):
    """Final unbiased evaluation on Test Set."""
    model.eval()
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    
    total_mse, total_mae, count = 0.0, 0.0, 0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        
        total_mse += mse_func(pred, yb).item() * xb.size(0)
        total_mae += mae_func(pred, yb).item() * xb.size(0)
        count += xb.size(0)
        
    mse = total_mse / count
    mae = total_mae / count
    
    if verbose:
        print(f"\n--- scores ---")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
    
    return mse, mae


def count_parameters(model):
    """
    Returns the number of trainable parameters 
    and the estimated size in MB.
    """
    # Sum of all individual parameter tensors
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate size in MB (assuming float32 = 4 bytes)
    size_mb = (num_params * 4) / (1024**2)
    
    return num_params, size_mb