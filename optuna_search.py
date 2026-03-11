"""
optuna_search.py — Bayesian hyperparameter search for Autoresearch 2.0

Uses Optuna to efficiently explore the hyperparameter space and find the
best configuration for your hardware. Each trial runs train.py as a
subprocess and reports val_bpb back to Optuna.

Usage:
  uv run python optuna_search.py                        # 20 trials, 300s each
  uv run python optuna_search.py --trials 50            # 50 trials
  uv run python optuna_search.py --time-budget 120      # shorter runs per trial
  uv run python optuna_search.py --language fr          # French
  uv run python optuna_search.py --resume               # resume an existing study
  uv run python optuna_search.py --best                 # print best params and exit

The study is saved to optuna_study.db (SQLite) so it survives interruptions.
"""

from __future__ import annotations
import sys
import os
import subprocess
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STUDY_NAME = "autoresearch_hpo"
STORAGE = "sqlite:///optuna_study.db"
LOG_PATH = "run.log"


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial) -> dict:
    """Define the hyperparameter search space."""

    # Architecture
    depth = trial.suggest_int("depth", 2, 10, step=2)
    aspect_ratio = trial.suggest_categorical("aspect_ratio", [32, 48, 64, 96])
    head_dim = trial.suggest_categorical("head_dim", [64, 128])

    # Optimizer
    optimizer_type = trial.suggest_categorical("optimizer_type", [
        "muon_adamw", "lion", "adafactor"
    ])
    matrix_lr = trial.suggest_float("matrix_lr", 0.005, 0.1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.05, 0.5, log=True)

    # Architecture variants
    use_swiglu = trial.suggest_categorical("use_swiglu", [True, False])
    use_prenorm = trial.suggest_categorical("use_prenorm", [True, False])
    use_weight_tying = trial.suggest_categorical("use_weight_tying", [True, False])

    return {
        "depth": depth,
        "aspect_ratio": aspect_ratio,
        "head_dim": head_dim,
        "optimizer_type": optimizer_type,
        "matrix_lr": matrix_lr,
        "weight_decay": weight_decay,
        "use_swiglu": use_swiglu,
        "use_prenorm": use_prenorm,
        "use_weight_tying": use_weight_tying,
    }


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def build_train_cmd(params: dict, time_budget: int, language: str) -> list[str]:
    """Build the train.py command from suggested params."""
    cmd = [
        sys.executable, "train.py",
        "--language", language,
        "--time-budget", str(time_budget),
        "--depth", str(params["depth"]),
        "--aspect-ratio", str(params["aspect_ratio"]),
        "--head-dim", str(params["head_dim"]),
        "--optimizer", params["optimizer_type"],
    ]
    if params.get("use_swiglu"):
        cmd.append("--use-swiglu")
    if params.get("use_prenorm"):
        cmd.append("--use-prenorm")
    return cmd


def parse_val_bpb(log_path: str) -> float | None:
    """Extract val_bpb from run.log."""
    try:
        with open(log_path) as f:
            for line in f:
                if line.startswith("val_bpb:"):
                    return float(line.split(":")[1].strip())
    except Exception:
        pass
    return None


def run_trial(trial: optuna.Trial, time_budget: int, language: str) -> float:
    """Execute one Optuna trial."""
    params = suggest_params(trial)

    print(f"\n{'─'*60}")
    print(f"Trial #{trial.number}")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"{'─'*60}")

    cmd = build_train_cmd(params, time_budget, language)

    t0 = time.time()
    Path(LOG_PATH).write_text("")  # clear log

    try:
        with open(LOG_PATH, "w") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise optuna.exceptions.TrialPruned("Interrupted by user")

    elapsed = time.time() - t0
    val_bpb = parse_val_bpb(LOG_PATH)

    if val_bpb is None:
        print(f"  [FAILED] No val_bpb in log after {elapsed:.0f}s — pruning trial")
        raise optuna.exceptions.TrialPruned("Training crashed or no val_bpb produced")

    print(f"  val_bpb: {val_bpb:.6f}  ({elapsed:.0f}s)")

    # Save params + result to a JSONL log for easy inspection
    record = {
        "trial": trial.number,
        "val_bpb": val_bpb,
        "elapsed_s": round(elapsed),
        "timestamp": datetime.now().isoformat(),
        **params,
    }
    with open("optuna_trials.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    return val_bpb


# ---------------------------------------------------------------------------
# Best params display
# ---------------------------------------------------------------------------

def print_best(study: optuna.Study):
    """Print the best trial found so far."""
    try:
        best = study.best_trial
    except ValueError:
        print("No completed trials yet.")
        return

    print(f"\n{'='*60}")
    print("BEST TRIAL")
    print(f"{'='*60}")
    print(f"  Trial #:   {best.number}")
    print(f"  val_bpb:   {best.value:.6f}")
    print(f"\n  Parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    print(f"\n  To reproduce:")
    cmd = build_train_cmd(best.params, 300, "en")
    print("  " + " ".join(cmd))
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for Autoresearch 2.0")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--time-budget", type=int, default=300, help="Seconds per trial")
    parser.add_argument("--language", type=str, default="en", help="Language code (en, fr, hi, ...)")
    parser.add_argument("--resume", action="store_true", help="Resume an existing study from DB")
    parser.add_argument("--best", action="store_true", help="Print best params from existing study and exit")
    parser.add_argument("--study-name", type=str, default=STUDY_NAME, help="Optuna study name")
    args = parser.parse_args()

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(seed=42, n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    if args.resume:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=STORAGE,
            sampler=sampler,
            pruner=pruner,
        )
        print(f"Resumed study '{args.study_name}' with {len(study.trials)} existing trials.")
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=STORAGE,
            direction="minimize",   # lower val_bpb = better
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,    # safe to re-run without --resume
        )

    if args.best:
        print_best(study)
        return

    completed_before = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nAutoresearch 2.0 — Optuna Hyperparameter Search")
    print(f"Study:       {args.study_name}")
    print(f"Storage:     {STORAGE}")
    print(f"Language:    {args.language}")
    print(f"Time/trial:  {args.time_budget}s")
    print(f"Trials:      {args.trials} new  ({completed_before} already done)")
    print(f"Objective:   minimize val_bpb\n")

    try:
        study.optimize(
            lambda trial: run_trial(trial, args.time_budget, args.language),
            n_trials=args.trials,
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nSearch interrupted.")

    print_best(study)

    # Summary table
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Completed trials: {len(completed)}")
    print(f"Pruned trials:    {len(pruned)}")
    print(f"Results log:      optuna_trials.jsonl")
    print(f"Study DB:         optuna_study.db")


if __name__ == "__main__":
    main()
