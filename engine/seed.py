import os
import random
import numpy as np
import torch


def set_global_seed(
    seed: int, 
    verbose: bool = True,
    deterministic: bool = True,
):

    # Environment seed (important for dataloaders)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN determinism
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    if verbose:
        print(f"[seed] Global seed set to {seed} | deterministic={deterministic}")




def seed_worker(worker_id):
    """
    For PyTorch DataLoader workers reproducibility.

    Use in DataLoader:
        worker_init_fn=seed_worker
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)