# EnergyTSF: Deep Learning Architectures for Energy Time Series Forecasting

A modular comparative framework evaluating deep sequence architectures (GRU, Transformer, and Mamba state-space models) on multivariate energy load and transformer temperature forecasting benchmarks.

---

## Highlights

* Architectures: GRU, Transformer, and Mamba (State Space Model) implementations for long-horizon multivariate forecasting.
* Datasets: Electricity Load Diagrams (ECL) and Electricity Transformer Temperature (ETTh1).
* Automated Optimization: Hyperparameter tuning pipelines powered by Optuna with YAML-driven search spaces.
* Evaluation Scenarios: Standard multivariate horizon mapping (L=96, T=96) and extended lookback evaluation (L=192, T=96).
* Artifacts & Thesis: Includes complete analytical plots, presentation slides, and the full thesis manuscript.

---

## Project Structure

EnergyTSF/
├── configs/
│   ├── final_runs/          # Tuned execution configs (L96 & L192)
│   ├── recap/               # Search space summary reports
│   └── search_spaces/       # Optuna parameter space definitions
├── data_provider/           # Dataset factories and custom PyTorch DataLoaders
├── docs/                    # Full thesis PDF and defense presentation
├── engine/                  # Trainer, evaluator, and hyperparameter tuning loop
├── layers/                  # Attention, embedding, and block primitives
├── models/                  # Core model implementations (gru.py, mamba.py, transformer.py)
├── notebooks/               # Interactive exploration and tuning notebooks
├── optuna_runs_yaml/        # Aggregated optimization summaries and Top-K metrics
├── utils/                   # Metrics calculation, sanitization, and I/O handlers
├── windows11/               # Windows evaluation scripts and final publication figures
├── requirements.txt         # Environment dependencies
└── notes.md                 # Original setup and data notes

---

## Dataset Setup

*** Datasets Source [https://github.com/wzhwzhwzh0921/S-D-Mamba/releases/tag/datasets](https://github.com/wzhwzhwzh0921/S-D-Mamba/releases/tag/datasets)

Extract S-Mamba_datasets.zip and place the files inside data/:

data/
├── ECL/
│   ├── behind_electricity.csv
│   ├── electricity.csv
│   └── middle_electricity.csv
└── ETT/
    ├── ETTh1.csv
    ├── ETTh2.csv
    ├── ETTm1.csv
    └── ETTm2.csv

---

## Installation and Execution

Clone the repository:
git clone [https://github.com/franztac/EnergyTSF.git](https://github.com/franztac/EnergyTSF.git)
cd EnergyTSF

Install dependencies:
pip install -r requirements.txt

Hyperparameter Search:
Optuna tuning runs are configured via YAML specs located in configs/search_spaces/.
Open and run notebooks/optuna_tuning.ipynb to initiate automated study trials.

Final Model Training & Evaluation:
To execute runs using the optimal configuration profiles, execute final_run.ipynb or call the run executor directly using targets in configs/final_runs/.

Visualizations:
Comparative figures, error distributions, and lookback impact charts are generated through windows11/utils/ and stored in windows11/figures/.