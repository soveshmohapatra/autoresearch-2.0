"""
run.py — Unified entry point for Autoresearch 2.0

Usage:
  uv run python run.py train                              # single training run
  uv run python run.py train --language hi --resume       # warm-start Hindi

  uv run python run.py agent                              # autonomous agent loop (forever)
  uv run python run.py agent --max-runs 20 --dry-run

  uv run python run.py optuna                             # Bayesian HPO, 20 trials x 300s
  uv run python run.py optuna --trials 50 --time-budget 120 --language hi
  uv run python run.py optuna --best                      # print best params found so far

  uv run python run.py all                                # agent + optuna in parallel
  uv run python run.py all --agent-runs 20 --optuna-trials 30 --language hi
"""

from __future__ import annotations
import sys
import signal
import subprocess
import argparse
from pathlib import Path

HERE = Path(__file__).parent


def _run(cmd: list[str]) -> int:
    """Run a command in the same directory, inheriting stdio."""
    return subprocess.run(cmd, cwd=HERE).returncode


def _popen(cmd: list[str], log_path: Path) -> subprocess.Popen:
    """Start a background process, writing stdout+stderr to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Autoresearch 2.0 — unified launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", metavar="MODE")
    sub.required = True

    # ── train ────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Single training run")
    p_train.add_argument("--language", default="en")
    p_train.add_argument("--depth", type=int)
    p_train.add_argument("--aspect-ratio", type=int)
    p_train.add_argument("--head-dim", type=int)
    p_train.add_argument("--batch-size", type=int)
    p_train.add_argument("--seq-len", type=int)
    p_train.add_argument("--optimizer")
    p_train.add_argument("--time-budget", type=int)
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--use-swiglu", action="store_true")
    p_train.add_argument("--use-prenorm", action="store_true")
    p_train.add_argument("--use-gqa", action="store_true")
    p_train.add_argument("--use-moe", action="store_true")
    p_train.add_argument("--experiment-name")

    # ── agent ────────────────────────────────────────────────────────────────
    p_agent = sub.add_parser("agent", help="Autonomous Claude agent loop")
    p_agent.add_argument("--max-runs", type=int, default=0)
    p_agent.add_argument("--dry-run", action="store_true")
    p_agent.add_argument("--tag")

    # ── optuna ───────────────────────────────────────────────────────────────
    p_optuna = sub.add_parser("optuna", help="Bayesian hyperparameter search")
    p_optuna.add_argument("--trials", type=int, default=20)
    p_optuna.add_argument("--time-budget", type=int, default=300)
    p_optuna.add_argument("--language", default="en")
    p_optuna.add_argument("--study-name", default="autoresearch_hpo")
    p_optuna.add_argument("--resume", action="store_true")
    p_optuna.add_argument("--best", action="store_true")

    # ── all ──────────────────────────────────────────────────────────────────
    p_all = sub.add_parser(
        "all",
        help="Run agent + optuna in parallel (both call train internally)",
    )
    p_all.add_argument("--language", default="en")
    p_all.add_argument("--agent-runs", type=int, default=0,
                       help="Max agent experiments (0 = infinite)")
    p_all.add_argument("--agent-tag", default=None,
                       help="Git branch tag for agent experiments")
    p_all.add_argument("--optuna-trials", type=int, default=20)
    p_all.add_argument("--time-budget", type=int, default=300,
                       help="Seconds per training run (used by both agent and optuna)")
    p_all.add_argument("--study-name", default="autoresearch_hpo")
    p_all.add_argument("--logs-dir", default="logs",
                       help="Directory for background process logs (default: logs/)")

    args = parser.parse_args()

    # ── dispatch ─────────────────────────────────────────────────────────────
    if args.mode == "train":
        cmd = [sys.executable, "train.py", "--language", args.language]
        if args.depth:           cmd += ["--depth", str(args.depth)]
        if args.aspect_ratio:    cmd += ["--aspect-ratio", str(args.aspect_ratio)]
        if args.head_dim:        cmd += ["--head-dim", str(args.head_dim)]
        if args.batch_size:      cmd += ["--batch-size", str(args.batch_size)]
        if args.seq_len:         cmd += ["--seq-len", str(args.seq_len)]
        if args.optimizer:       cmd += ["--optimizer", args.optimizer]
        if args.time_budget:     cmd += ["--time-budget", str(args.time_budget)]
        if args.experiment_name: cmd += ["--experiment-name", args.experiment_name]
        if args.resume:          cmd.append("--resume")
        if args.use_swiglu:      cmd.append("--use-swiglu")
        if args.use_prenorm:     cmd.append("--use-prenorm")
        if args.use_gqa:         cmd.append("--use-gqa")
        if args.use_moe:         cmd.append("--use-moe")
        sys.exit(_run(cmd))

    elif args.mode == "agent":
        cmd = [sys.executable, "agent.py"]
        if args.max_runs: cmd += ["--max-runs", str(args.max_runs)]
        if args.dry_run:  cmd.append("--dry-run")
        if args.tag:      cmd += ["--tag", args.tag]
        sys.exit(_run(cmd))

    elif args.mode == "optuna":
        cmd = [sys.executable, "optuna_search.py",
               "--trials", str(args.trials),
               "--time-budget", str(args.time_budget),
               "--language", args.language,
               "--study-name", args.study_name]
        if args.resume: cmd.append("--resume")
        if args.best:   cmd.append("--best")
        sys.exit(_run(cmd))

    elif args.mode == "all":
        logs = HERE / args.logs_dir

        agent_cmd = [sys.executable, "agent.py"]
        if args.agent_runs: agent_cmd += ["--max-runs", str(args.agent_runs)]
        if args.agent_tag:  agent_cmd += ["--tag", args.agent_tag]

        optuna_cmd = [sys.executable, "optuna_search.py",
                      "--trials", str(args.optuna_trials),
                      "--time-budget", str(args.time_budget),
                      "--language", args.language,
                      "--study-name", args.study_name]

        agent_log  = logs / "agent.log"
        optuna_log = logs / "optuna.log"

        print(f"Starting agent  → log: {agent_log}")
        print(f"Starting optuna → log: {optuna_log}")
        print("Press Ctrl+C to stop both.\n")

        agent_proc  = _popen(agent_cmd,  agent_log)
        optuna_proc = _popen(optuna_cmd, optuna_log)
        procs = [agent_proc, optuna_proc]

        def _stop(*_):
            print("\nStopping all processes...")
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass

        signal.signal(signal.SIGINT,  _stop)
        signal.signal(signal.SIGTERM, _stop)

        # Stream both logs to the terminal in real time
        tail_proc = subprocess.Popen(
            ["tail", "-f", str(agent_log), str(optuna_log)],
            cwd=HERE,
        )

        # Wait until both finish (or Ctrl+C)
        for p in procs:
            p.wait()
        tail_proc.terminate()

        codes = [p.returncode for p in procs]
        print(f"\nDone. agent={codes[0]}  optuna={codes[1]}")
        sys.exit(max(c or 0 for c in codes))


if __name__ == "__main__":
    main()
