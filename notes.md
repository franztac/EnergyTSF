*** Repo Structure

EnergyTSF/
├── configs/
│   └── search_spaces/         # YAML search spaces for Optuna tuning
│
├── data/                      # Raw CSV datasets (download separately)
│   ├── ETT/
│   └── ECL/
|
├── docs/                      # Thesis and defense presentation
│   ├── thesis.pdf
│   └── thesis_presentation.pdf
│
├── data_provider/             # Data loading pipeline
│   ├── data_factory.py
│   └── data_loader.py
│
├── engine/                    # Training / tuning logic
│   ├── trainer.py
│   ├── tuning.py
│   └── seed.py
│
├── layers/                    # Core model building blocks
│   ├── attention.py
│   ├── embed.py
│   └── blocks.py
│
├── models/                    # Forecasting architectures
│   ├── gru.py
│   ├── transformer.py
│   └── mamba.py
│
├── utils/                     # Utilities
│   ├── optuna_utils.py
│   ├── optuna_yaml.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── plot_utils.py
│   └── tools.py
│
├── optuna_runs_yaml/          # Optuna studies (YAML-driven tuning)
│   └── {dataset}/{model_token}/L{window}_T{horizon}/study.db
│
├── runs/                      # Full experiment artifacts
│   ├── checkpoints
│   ├── metrics
│   └── plots
│
├── optuna_tuning.ipynb        # Hyperparameter tuning notebook
├── gru.ipynb                  # GRU experiments
├── transformer.ipynb          # Transformer experiments
├── mamba.ipynb                # Mamba experiments
│
├── requirements.txt
└── notes.md


*** Datasets Source
https://github.com/wzhwzhwzh0921/S-D-Mamba/releases/tag/datasets

# Extract S-Mamba_datasets.zip and place the files inside data/:
data/ECL/
├── behind_electricity.csv
├── electricity.csv
└── middle_electricity.csv

data/ETT/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
└── ETTm2.csv