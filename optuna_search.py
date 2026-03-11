"""
optuna_search.py — Bayesian hyperparameter search for Autoresearch 2.0

Runs Optuna TPE search over architecture and optimizer hyperparameters.
Each trial spawns a subprocess training run and reads val_bpb from the log.

Usage:
  uv run python optuna_search.py                              # 20 trials, 300s each
  uv run python optuna_search.py --trials 50 --time-budget 120 --language hi
  uv run python optuna_search.py --resume                     # continue existing study
  uv run python optuna_search.py --best                       # print best params and exit
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

STORAGE  = "sqlite:///optuna_study.db"
LOG_PATH = Path("run.log")
TRIALS_LOG = Path("optuna_trials.jsonl")

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def suggest(trial: optuna.Trial) -> dict:
    return {
        "depth":            trial.suggest_int("depth", 2, 10, step=2),
        "aspect_ratio":     trial.suggest_categorical("aspect_ratio", [32, 48, 64, 96]),
        "head_dim":         trial.suggest_categorical("head_dim", [64, 128]),
        "optimizer_type":   trial.suggest_categorical("optimizer_type", ["muon_adamw", "lion", "adafactor"]),
        "matrix_lr":        trial.suggest_float("matrix_lr", 0.005, 0.1, log=True),
        "weight_decay":     trial.suggest_float("weight_decay", 0.05, 0.5, log=True),
        "use_swiglu":       trial.suggest_categorical("use_swiglu", [True, False]),
        "use_prenorm":      trial.suggest_categorical("use_prenorm", [True, False]),
        "use_weight_tying": trial.suggest_categorical("use_weight_tying", [True, False]),
    }


def build_cmd(params: dict, time_budget: int, language: str) -> list[str]:
    cmd = [
        sys.executable, "train.py",
        "--language", language,
        "--time-budget", str(time_budget),
        "--depth", str(params["depth"]),
        "--aspect-ratio", str(params["aspect_ratio"]),
        "--head-dim", str(params["head_dim"]),
        "--optimizer", params["optimizer_type"],
    ]
    if params.get("use_swiglu"):  cmd.append("--use-swiglu")
    if params.get("use_prenorm"): cmd.append("--use-prenorm")
    return cmd


def parse_bpb(log_path: Path) -> float | None:
    try:
        for line in log_path.read_text().splitlines():
            if line.startswith("val_bpb:"):
                return float(line.split(":")[1].strip())
    except Exception:
        pass
    return None


def print_best(study: optuna.Study, time_budget: int, language: str) -> None:
    try:
        best = study.best_trial
    except ValueError:
        print("No completed trials yet.")
        return
    print(f"\n{'='*60}")
    print(f"BEST TRIAL  #{best.number}  val_bpb={best.value:.6f}")
    print(f"{'='*60}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"\n  Reproduce:")
    print("  " + " ".join(build_cmd(best.params, time_budget, language)))
    print(f"{'='*60}\n")

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(time_budget: int, language: str):
    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        print(f"\nTrial #{trial.number}: {params}")
        LOG_PATH.write_text("")
        cmd = build_cmd(params, time_budget, language)
        t0 = time.time()
        with open(LOG_PATH, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        bpb = parse_bpb(LOG_PATH)
        if bpb is None:
            raise optuna.exceptions.TrialPruned("Training crashed or produced no val_bpb")
        print(f"  → val_bpb={bpb:.6f}  ({elapsed:.0f}s)")
        record = {
            "trial": trial.number, "val_bpb": bpb,
            "elapsed_s": round(elapsed), "timestamp": datetime.now().isoformat(),
            **params,
        }
        with open(TRIALS_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        return bpb
    return objective

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for Autoresearch 2.0")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials (default: 20)")
    parser.add_argument("--time-budget", type=int, default=300, help="Seconds per trial (default: 300)")
    parser.add_argument("--language", type=str, default="en", help="Language code (default: en)")
    parser.add_argument("--study-name", type=str, default="autoresearch_hpo", help="Optuna study name")
    parser.add_argument("--resume", action="store_true", help="Resume existing study from DB")
    parser.add_argument("--best", action="store_true", help="Print best params and exit")
    args = parser.parse_args()

    sampler = TPESampler(seed=42, n_startup_trials=5)
    pruner  = MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=STORAGE,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    if args.best:
        print_best(study, args.time_budget, args.language)
        return

    done_before = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nAutoresearch 2.0 — Optuna HPO")
    print(f"Study:      {args.study_name}  ({done_before} trials already done)")
    print(f"Language:   {args.language}")
    print(f"Time/trial: {args.time_budget}s")
    print(f"New trials: {args.trials}\n")

    try:
        study.optimize(
            make_objective(args.time_budget, args.language),
            n_trials=args.trials,
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nSearch interrupted.")

    print_best(study, args.time_budget, args.language)
    done   = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Completed: {len(done)}  Pruned: {len(pruned)}")
    print(f"Results log: {TRIALS_LOG}  |  DB: {STORAGE}")


# ---------------------------------------------------------------------------
# Agent integration helpers
# ---------------------------------------------------------------------------

def create_or_load_study(study_name: str = "autoresearch_hpo") -> "optuna.Study":
    """Create or load an Optuna study for use inside the agent loop."""
    sampler = TPESampler(seed=42, n_startup_trials=5)
    pruner  = MedianPruner(n_startup_trials=5)
    return optuna.create_study(
        study_name=study_name,
        storage=STORAGE,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def ask_trial(study: "optuna.Study") -> tuple:
    """Ask the study for a new trial and return (trial, params_dict)."""
    trial  = study.ask()
    params = suggest(trial)
    return trial, params


def tell_trial(study: "optuna.Study", trial, val_bpb: float | None) -> None:
    """Report the training result back to the study."""
    import optuna as _opt
    if val_bpb is not None:
        study.tell(trial, val_bpb)
    else:
        study.tell(trial, state=_opt.trial.TrialState.FAIL)


if __name__ == "__main__":
    main()
