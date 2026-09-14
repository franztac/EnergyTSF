"""
Optuna utilities (single DB per setting, multiple studies per DB).

DB-only convention:
  .optuna_runs/{dataset}/{model}_{token}/L{L}_T{T}/study.db

Each "round" of tuning is a separate STUDY in the same DB:
  tuning_{timestamp}

Timestamp format defaults to minute resolution: YYYY-MM-DD_HH-MM
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


# Timestamp helpers
def make_timestamp(with_seconds: bool = False) -> str:
    """
    Timestamp safe for study names and paths.
    Default: minute resolution (no seconds), per user preference.
    """
    fmt = "%Y-%m-%d_%H-%M-%S" if with_seconds else "%Y-%m-%d_%H-%M"
    return datetime.now().strftime(fmt)



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def _abs_sqlite_storage(db_path: str) -> str:
    """
    Build absolute sqlite storage URI (recommended to avoid relative path issues).
    """
    abs_db = os.path.abspath(db_path)
    return f"sqlite:///{abs_db}"



# Paths container
@dataclass(frozen=True)
class OptunaSettingPaths:
    setting_dir: str   # .../.optuna_runs/ETTh1/gru_time/L96_T96
    db_path: str       # .../study.db
    storage: str       # sqlite:////abs/.../study.db



# Core builders
def build_optuna_setting_paths(
    *,
    dataset: str,
    token_style: str,
    model_name: str,
    window: int,
    horizon: int,
    root_dir: str = ".optuna_runs",
    create: bool = True,
) -> OptunaSettingPaths:
    """
    Build the single-DB location for a given (dataset, model, token, L, T).
    Creates directories if create=True.

    Example:
        setting_dir = .optuna_runs/ETTh1/gru_time/L96_T96
        db_path     = .../study.db
        storage     = sqlite:////abs/.../study.db
    """
    # New structure: ./optuna_runs/{task}/{dataset}/{model}_{token}/L{L}_T{T}/study.db
    setting_dir = os.path.join(
        root_dir,
        dataset,
        f"{model_name}_{token_style}",
        f"L{window}_T{horizon}",
    )

    if create:
        ensure_dir(setting_dir)

    db_path = os.path.join(setting_dir, "study.db")
    storage = _abs_sqlite_storage(db_path)

    return OptunaSettingPaths(
        setting_dir=setting_dir,
        db_path=db_path,
        storage=storage,
    )



def build_optuna_setting_paths_from_configs(
    *,
    configs,
    root_dir: str = "./optuna_runs",
    create: bool = True,
) -> OptunaSettingPaths:
    """
    Wrapper using your Namespace configs.
    """
    return build_optuna_setting_paths(
        dataset=getattr(configs, "data_name"),
        token_style=getattr(configs, "token_style"),
        model_name=getattr(configs, "model_name"),
        window=getattr(configs, "window"),
        horizon=getattr(configs, "horizon"),
        root_dir=root_dir,
        create=create,
    )


def make_study_name(
    *,
    prefix: str = "tuning",
    timestamp: Optional[str] = None,
    with_seconds: bool = False,
) -> str:
    """
    Create a simple study name like:
      tuning_YYYY-MM-DD_HH-MM
    """
    if timestamp is None:
        timestamp = make_timestamp(with_seconds=with_seconds)

    return f"{prefix}_{timestamp}"



def make_study_name_from_configs(
    *,
    configs,
    prefix: str = "tuning",
    timestamp: Optional[str] = None,
    with_seconds: bool = False,
) -> str:
    # configs not used anymore, but kept for API symmetry / easy notebook use
    return make_study_name(
        prefix=prefix,
        timestamp=timestamp,
        with_seconds=with_seconds,
    )



def list_existing_setting_dirs(root_dir: str = ".optuna_runs") -> List[str]:
    """
    List all setting directories that contain a study.db.
    Useful when you want to inspect what's available on disk.
    """
    setting_dirs = []

    if not os.path.exists(root_dir):
        return setting_dirs

    for dirpath, _, filenames in os.walk(root_dir):
        if "study.db" in filenames:
            setting_dirs.append(dirpath)

    setting_dirs.sort()
    return setting_dirs



# Discovery helpers (Optuna DB)
# helper 2
def list_studies_in_storage(storage: str, verbose: bool = True):
    """
    Return study summaries inside a DB.
    
    If verbose=True, prints a compact overview.
    """
    import optuna

    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage)
    except Exception as e:
        if verbose:
            print(f"⚠️ Could not read storage: {e}")
        return []
    
    # sorted chronologically by name (timestamp-based)
    summaries = sorted(summaries, key=lambda s: s.study_name)

    if verbose:
        print(f"📚 Found {len(summaries)} studies in {storage}")

        if len(summaries) == 0:
            print("No studies yet.")
        else:
            for s in summaries:
                if s.best_trial is None or s.best_trial.value is None:
                    best_str = "no valid trials"
                else:
                    best_str = f"{s.best_trial.value:.6f}"

                print(
                    f"🔹 {s.study_name:30} "
                    f"| trials={s.n_trials:3d} "
                    f"| best={best_str:15} "
                    f"| direction={s.direction.name}"
                )

    return summaries




def infer_channels(data_name: str) -> int:
    """
    Infer the number of channels from dataset name.
    """
    if data_name == "ETTh1":
        return 7
    if data_name == "ECL":
        return 321

    raise ValueError(f"Unknown data_name: {data_name}")




def build_search_space_yaml_path(
    configs, 
    root_dir="configs/search_spaces",
    mode="multivariate",
) -> str:
    """
    Build the YAML path automatically from config identity and folder mode.
    
    Modes:
        'multivariate'        -> configs/search_spaces/multivariate/...
        'increasing_lookback' -> configs/search_spaces/increasing_lookback/...

    Example:
        configs/search_spaces/multivariate/etth1_transformer_time_L96_T96.yaml
    """
    filename = (
        f"{configs.data_name.lower()}_"
        f"{configs.model_name}_"
        f"{configs.token_style}_"
        f"L{configs.window}_T{configs.horizon}.yaml"
    )
    
    # Now joins with the 'mode' subfolder
    yaml_path = os.path.join(root_dir, mode, filename)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Missing YAML search space file: {yaml_path}\n"
            f"Check if EXPERIMENT_MODE matches the folder structure."
        )
    
    return yaml_path




def extract_top_k_trials(study, yaml_path: str, db_path: str, k: int = 5) -> dict:
    """
    Return a compact summary of the top-k COMPLETE trials in one study.

    The output is designed to be JSON-friendly and compact:
    - db_path identifies the config/storage
    - study_name identifies the specific study inside that DB
    - yaml_path tells which search-space file was used
    - top_trials contains rank/value/params only

    """
    import optuna

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]

    complete_trials = sorted(complete_trials, key=lambda t: t.value)

    top_trials = []
    for rank, trial in enumerate(complete_trials[:k], start=1):
        top_trials.append({
            "rank": rank,
            "trial_number": trial.number,
            "value": trial.value,
            "params": trial.params,
        })

    return {
        "db_path": db_path,
        "study_name": study.study_name,
        "yaml_path": yaml_path,
        "top_trials": top_trials,
    }


# helper 1 
def save_topk_summaries_json(
    summaries: list,
    save: bool = False,
    root_dir: str = "./optuna_runs_yaml/summaries",
    filename_prefix: str = "topk_summary",
    verbose: bool = True,
):
    """
    Optionally save loop_topk_summaries to a JSON file.
    """

    if not save:
        return None
    
    if not summaries:
        if verbose:
            print("⚠️ No summaries to save.")
        return None


    ensure_dir(root_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{filename_prefix}_{timestamp}.json"
    path = os.path.join(root_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    
    if verbose:
        print(f"\n💾 Top-k summaries saved to:\n{path}")

    return path



def load_json_file(json_path: str, verbose: bool = True):
    """
    Load one JSON file from disk and return the parsed Python object.

    Generic on purpose:
    - helper1 JSON -> usually a list
    - helper3 JSON -> usually a dict
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if verbose:
        obj_type = type(data).__name__
        size_hint = len(data) if hasattr(data, "__len__") else "n/a"
        print(f"📄 Loaded JSON: {json_path}")
        print(f"   type={obj_type} | size={size_hint}")

    return data



def _storage_from_db_path(db_path: str) -> str:
    """
    Build a sqlite storage URI from a DB path.
    For scanning, keep it readable in prints.
    """
    return f"sqlite:///{db_path}"



def _setting_key_from_db_path(db_path: str, root_dir: str) -> str:
    """
    Convert:
      ./optuna_runs_yaml/ETTh1/gru_variate/L96_T96/study.db
    into:
      ETTh1/gru_variate/L96_T96
    """
    rel_path = os.path.relpath(db_path, root_dir)
    parts = rel_path.split(os.sep)

    if len(parts) < 4:
        raise ValueError(f"Unexpected DB path structure: {db_path}")

    return "/".join(parts[:-1])



def _best_entry_from_summaries(summaries):
    """
    Return the best study info from list_studies_in_storage(...) output.
    """
    best_study = None
    best_value = None

    for s in summaries:
        if s.best_trial is None or s.best_trial.value is None:
            continue

        if best_value is None or s.best_trial.value < best_value:
            best_value = s.best_trial.value
            best_study = s.study_name

    if best_study is None:
        return {
            "best_study": None,
            "best_value": None,
        }

    return {
        "best_study": best_study,
        "best_value": float(best_value),
    }




def save_scan_summary_json(
    summary: dict,
    save: bool = False,
    root_dir: str = "./optuna_runs_yaml/summaries",
    filename_prefix: str = "scan_summary",
    verbose: bool = True,
):
    """
    Optionally save helper3 scan summary to JSON.
    """
    if not save:
        return None

    if not summary:
        if verbose:
            print("⚠️ No scan summary to save.")
        return None

    ensure_dir(root_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{filename_prefix}_{timestamp}.json"
    path = os.path.join(root_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"💾 Scan summary saved to:\n{path}")

    return path



def scan_all_optuna_storages(
    root_dir: str = "./optuna_runs_yaml",
    save: bool = False,
    save_root_dir: str = "./optuna_runs_yaml/summaries",
    filename_prefix: str = "scan_summary",
    verbose: bool = True,
):
    """
    Scan all study.db files under root_dir and build a compact best-study summary.

    Output example:
    {
      "ETTh1/gru_variate/L96_T96": {
        "best_study": "tuning_yaml_2026-03-08_20-36",
        "best_value": 0.375778
      }
    }
    """
    setting_dirs = list_existing_setting_dirs(root_dir=root_dir)
    db_paths = [os.path.join(d, "study.db") for d in setting_dirs]

    if verbose:
        print(f"🔎 Found {len(db_paths)} study.db files under {root_dir}")

    scan_summary = {}

    for db_path in db_paths:
        storage = _storage_from_db_path(db_path)

        summaries = list_studies_in_storage(
            storage=storage,
            verbose=verbose,
        )

        setting_key = _setting_key_from_db_path(db_path, root_dir=root_dir)
        scan_summary[setting_key] = _best_entry_from_summaries(summaries)

    save_path = save_scan_summary_json(
        summary=scan_summary,
        save=save,
        root_dir=save_root_dir,
        filename_prefix=filename_prefix,
        verbose=verbose,
    )

    return scan_summary, save_path



# ---------------------------------------------------------
# Helper4 (LATEST STUDY PER DB)
# ---------------------------------------------------------
# Scan all Optuna DBs under root_dir, select the most recent
# study in each DB (based on study_name ordering), and build
# helper1-like top-k summaries for those latest studies.
#
# Useful when loop_topk_summaries was not saved in memory.
# ---------------------------------------------------------

def _latest_study_summary_from_summaries(summaries):
    """
    Return the most recent study summary from one DB.

    Assumes study names are timestamp-based, e.g.
    tuning_yaml_2026-03-09_18-40
    so lexicographic sorting is enough.
    """
    if not summaries:
        return None

    summaries = sorted(summaries, key=lambda s: s.study_name)
    return summaries[-1]


def _yaml_path_from_setting_key(
    setting_key: str,
    root_dir: str = "configs/search_spaces",
    mode: str = "multivariate",
) -> str:
    """
    Convert a DB setting key like: ETTh1/gru_variate/L96_T96
    into: configs/search_spaces/{mode}/etth1_gru_variate_L96_T96.yaml
    """
    dataset, model_token, lt = setting_key.split("/")
    filename = f"{dataset.lower()}_{model_token}_{lt}.yaml"

    # We now respect the mode subfolder here too
    return os.path.join(root_dir, mode, filename)


def save_latest_topk_summaries_json(
    summaries: list,
    save: bool = False,
    root_dir: str = "./optuna_runs_yaml/summaries",
    filename_prefix: str = "latest_topk_summary",
    verbose: bool = True,
):
    """
    Optionally save latest-study top-k summaries to JSON.
    """
    if not save:
        return None

    if not summaries:
        if verbose:
            print("⚠️ No latest-study summaries to save.")
        return None

    ensure_dir(root_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{filename_prefix}_{timestamp}.json"
    path = os.path.join(root_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    if verbose:
        print(f"\n💾 Latest-study top-k summaries saved to:\n{path}")

    return path


def scan_latest_studies_topk(
    root_dir: str = "./optuna_runs_yaml",
    yaml_root_dir: str = "configs/search_spaces",
    mode: str = "multivariate",
    k: int = 5,
    save: bool = False,
    save_root_dir: str = "./optuna_runs_yaml/summaries",
    filename_prefix: str = "latest_topk_summary",
    verbose: bool = True,
):
    """
    Scan all study.db files under root_dir, take the most recent
    study in each DB, and return helper1-like top-k summaries.

    Returns
    -------
    summaries : list[dict]
        Same structure as extract_top_k_trials(...), one entry per DB.

    save_path : str | None
        JSON path if save=True, else None.
    """
    import optuna

    setting_dirs = list_existing_setting_dirs(root_dir=root_dir)

    if verbose:
        print(f"🔎 Found {len(setting_dirs)} study.db files under {root_dir}")

    latest_topk_summaries = []

    for setting_dir in setting_dirs:
        db_path = os.path.join(setting_dir, "study.db")
        storage = _storage_from_db_path(db_path)

        summaries = list_studies_in_storage(
            storage=storage,
            verbose=verbose,
        )

        latest_summary = _latest_study_summary_from_summaries(summaries)
        if latest_summary is None:
            continue

        study = optuna.load_study(
            storage=storage,
            study_name=latest_summary.study_name,
        )

        setting_key = _setting_key_from_db_path(db_path, root_dir=root_dir)

        # Pass the mode to reconstruct the correct YAML path
        yaml_path = _yaml_path_from_setting_key(
            setting_key, 
            root_dir=yaml_root_dir,
            mode=mode,
        )

        topk_summary = extract_top_k_trials(
            study=study,
            yaml_path=yaml_path,
            db_path=db_path,
            k=k,
        )
        latest_topk_summaries.append(topk_summary)

        if verbose:
            top_trials = topk_summary.get("top_trials", [])
            if top_trials:
                top = top_trials[0]
                print(
                    f"✅ Latest study for {setting_key} -> "
                    f"{latest_summary.study_name} | "
                    f"best={top['value']:.6f}"
                )
            else:
                print(
                    f"✅ Latest study for {setting_key} -> "
                    f"{latest_summary.study_name} | "
                    f"no complete trials"
                )
    
    # Ensure saving is handled
    save_path = save_latest_topk_summaries_json(
        summaries=latest_topk_summaries,
        save=save,
        root_dir=save_root_dir,
        filename_prefix=filename_prefix,
        verbose=verbose,
    )

    return latest_topk_summaries, save_path




def _study_best_value_or_inf(summary) -> float:
    """
    Return best value for sorting.
    Studies without a valid best trial are treated as +inf.
    """
    if summary.best_trial is None or summary.best_trial.value is None:
        return float("inf")
    return float(summary.best_trial.value)


def prune_optuna_studies(
    root_dir: str = "./optuna_runs_yaml",
    keep_top_n: int = 3,
    prune: bool = False,
    verbose: bool = True,
):
    """
    Scan all Optuna DBs under root_dir and keep only the best keep_top_n
    studies in each DB (lowest best value, because direction='minimize').

    Parameters
    ----------
    root_dir : str
        Root folder containing study.db files.

    keep_top_n : int
        Number of best studies to keep in each DB.

    prune : bool
        False -> dry run only (show what would be deleted)
        True  -> actually delete extra studies

    verbose : bool
        If True, print only DBs that have more than keep_top_n studies.
        If False, stay quiet.

    Returns
    -------
    deleted_total : int
        Total number of deleted studies (or studies that would be deleted if prune=False).
    """
    import optuna

    setting_dirs = list_existing_setting_dirs(root_dir=root_dir)
    deleted_total = 0

    if verbose:
        mode = "PRUNE" if prune else "DRY RUN"
        print(f"🧹 {mode} | Found {len(setting_dirs)} study.db files under {root_dir}")

    for setting_dir in setting_dirs:
        db_path = os.path.join(setting_dir, "study.db")
        storage = _storage_from_db_path(db_path)

        summaries = list_studies_in_storage(
            storage=storage,
            verbose=False,
        )

        # Skip quiet if DB already small enough
        if len(summaries) <= keep_top_n:
            continue

        ranked = sorted(
            summaries,
            key=lambda s: (_study_best_value_or_inf(s), s.study_name),
        )

        keep = ranked[:keep_top_n]
        to_delete = ranked[keep_top_n:]

        deleted_total += len(to_delete)

        if verbose:
            setting_key = _setting_key_from_db_path(db_path, root_dir=root_dir)

            print("\n" + "=" * 90)
            print(f"{setting_key}")
            print(f"studies={len(summaries)} | keep={len(keep)} | delete={len(to_delete)}")

            print("KEEP:")
            for s in keep:
                best_val = _study_best_value_or_inf(s)
                best_str = "None" if best_val == float("inf") else f"{best_val:.6f}"
                print(f"  ✅ {s.study_name:28} | best={best_str:10} | trials={s.n_trials}")

            print("DELETE:")
            for s in to_delete:
                best_val = _study_best_value_or_inf(s)
                best_str = "None" if best_val == float("inf") else f"{best_val:.6f}"
                print(f"  ❌ {s.study_name:28} | best={best_str:10} | trials={s.n_trials}")

        if prune:
            for s in to_delete:
                optuna.delete_study(
                    study_name=s.study_name,
                    storage=storage,
                )

    if verbose:
        if prune:
            print(f"\nDone. Deleted {deleted_total} studies.")
        else:
            print(f"\nDry run finished. {deleted_total} studies would be deleted.")

    return deleted_total




