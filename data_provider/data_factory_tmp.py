from data_provider.data_loader_tmp import EnergyDataModule
import pandas as pd
import os


def data_factory(
    data_name,
    window,
    horizon,
    stride,
    batch_size,
    ratios,
    verbose,
):
    """
    Temporary factory that points to data_loader_tmp.py
    so your original factory and loader stay untouched.
    """
    data_map = {
        'ETTh1': 'data/ETT/ETTh1.csv',
        'ETTh2': 'data/ETT/ETTh2.csv',
        'ETTm1': 'data/ETT/ETTm1.csv',
        'ETTm2': 'data/ETT/ETTm2.csv',
        'ECL':   'data/ECL/electricity.csv',
    }

    path = data_map.get(data_name)

    if not path or not os.path.exists(path):
        raise ValueError(f"Dataset {data_name} not found at {path}")

    df = pd.read_csv(path, parse_dates=["date"])

    dm = EnergyDataModule(
        df=df,
        name=data_name,
        window=window,
        horizon=horizon,
        stride=stride,
        batch_size=batch_size,
        ratios=ratios,
        verbose=verbose,
    )

    return dm