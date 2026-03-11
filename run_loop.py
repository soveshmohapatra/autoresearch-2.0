"""
run_loop.py — Autoresearch 2.0 experiment runner

Handles the full experiment cycle so AI agents only need to edit train.py:
  1. Shows memory context (past experiments, hypothesis)
  2. Runs training, captures output
  3. Parses results
  4. Records to memory + results.tsv
  5. Handles git keep/discard

Usage:
  uv run python run_loop.py                         # run one experiment (interactive keep/discard)
  uv run python run_loop.py --desc "SwiGLU test"    # provide description directly
  uv run python run_loop.py --auto                  # auto-decide keep/discard (agent mode)
  uv run python run_loop.py --detect                # detect hardware and exit

The agent loop in program.md uses this instead of running train.py directly.
"""

import os
import sys
import json
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], check=True) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def get_commit() -> str:
    try:
        return run_cmd(["git", "rev-parse", "--short", "HEAD"])
    except Exception:
        return "unknown"


def parse_run_log(log_path: str) -> dict:
    """Extract key metrics from run.log."""
    metrics = {}
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("val_bpb:"):
                    metrics["val_bpb"] = float(line.split(":")[1].strip())
                elif line.startswith("peak_vram_mb:"):
                    metrics["peak_vram_mb"] = float(line.split(":")[1].strip())
                elif line.startswith("mfu_percent:"):
                    metrics["mfu_percent"] = float(line.split(":")[1].strip())
                elif line.startswith("training_seconds:"):
                    metrics["training_seconds"] = float(line.split(":")[1].strip())
                elif line.startswith("num_params_M:"):
                    metrics["num_params_M"] = float(line.split(":")[1].strip())
                elif line.startswith("depth:"):
                    metrics["depth"] = int(line.split(":")[1].strip())
                elif line.startswith("device:"):
                    metrics["device"] = line.split(":")[1].strip()
                elif line.startswith("optimizer:"):
                    metrics["optimizer"] = line.split(":")[1].strip()
                elif line.startswith("arch_variants:"):
                    metrics["arch_variants"] = line.split(":", 1)[1].strip()
    except Exception:
        pass
    return metrics


def show_memory_context():
    """Print experiment history context for the agent."""
    try:
        from agents.memory import ExperimentMemory
        memory = ExperimentMemory()
        stats = memory.get_statistics()

        if stats["total"] == 0:
            print("No experiments recorded yet. This will be the baseline.")
            return

        print(f"\n{'='*60}")
        print("EXPERIMENT MEMORY CONTEXT")
        print(f"{'='*60}")
        print(f"Total experiments: {stats['total']}")
        keep_pct = stats.get('keep_rate', 0) * 100
        best = stats.get('best_bpb', float('inf'))
        print(f"Keep rate:         {keep_pct:.0f}%")
        if best != float('inf'):
            print(f"Best val_bpb:      {best:.6f}  (commit {stats.get('best_commit', '?')})")

        recent = memory.get_recent_experiments(5)
        if recent:
            print("\nRecent experiments:")
            for exp in reversed(recent):
                arrow = "✓" if exp.status == "keep" else "✗"
                print(f"  {arrow} [{exp.status:7s}] bpb={exp.val_bpb:.6f}  {exp.description}")

        hypothesis = memory.generate_hypothesis()
        print(f"\nSuggestion: {hypothesis}")
        print(f"{'='*60}\n")
    except ImportError:
        pass  # memory module not available, skip


def record_experiment(
    commit: str,
    val_bpb: float,
    peak_vram_mb: float,
    status: str,
    description: str,
    metrics: dict,
):
    """Record to memory + results.tsv."""
    # Write to results.tsv
    results_file = Path("results.tsv")
    if not results_file.exists():
        results_file.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\ttimestamp\n")

    memory_gb = round(peak_vram_mb / 1024, 1)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(results_file, "a") as f:
        desc_clean = description.replace("\t", " ")
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb}\t{status}\t{desc_clean}\t{timestamp}\n")

    print(f"Recorded to results.tsv: {commit} | {val_bpb:.6f} | {memory_gb}GB | {status} | {timestamp}")

    # Write to experiment memory
    try:
        from agents.memory import ExperimentMemory, ExperimentRecord
        memory = ExperimentMemory()
        record = ExperimentRecord(
            commit=commit,
            val_bpb=val_bpb,
            memory_mb=peak_vram_mb,
            status=status,
            description=description,
            timestamp=datetime.now().isoformat(),
            config_snapshot=metrics,
            metrics=metrics,
        )
        memory.add_experiment(record)
        print("Recorded to experiment_memory.json")
    except Exception as e:
        print(f"Warning: could not write to experiment_memory.json: {e}")


# ---------------------------------------------------------------------------
# Loss curve display
# ---------------------------------------------------------------------------

LOSS_CURVE_FILE = Path("loss_curve.jsonl")
_SPARK = "▁▂▃▄▅▆▇█"


def _read_loss_curve():
    """Return list of (step, loss, progress, remaining) from loss_curve.jsonl."""
    points = []
    try:
        with open(LOSS_CURVE_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    points.append((d["step"], d["loss"], d.get("progress", 0.0), d.get("remaining", 0.0)))
    except Exception:
        pass
    return points


def _sparkline(values, width=50):
    """Render an ASCII sparkline. Low loss = tall bar (inverted so improvement is visually up)."""
    if len(values) < 2:
        return _SPARK[0] * min(len(values), width)
    mn, mx = min(values), max(values)
    if mx == mn:
        return _SPARK[-1] * min(len(values), width)
    if len(values) > width:
        idxs = [int(i * (len(values) - 1) / (width - 1)) for i in range(width)]
        values = [values[i] for i in idxs]
    chars = []
    for v in values:
        idx = int((mx - v) / (mx - mn) * (len(_SPARK) - 1))  # invert: lower loss = taller
        chars.append(_SPARK[idx])
    return "".join(chars)


def _render_loss_line(points, elapsed):
    """Return a single-line string showing the live loss curve."""
    if not points:
        return f"  Waiting for training to start...  ({elapsed:.0f}s elapsed)"
    losses = [p[1] for p in points]
    step, last_loss, progress, remaining = points[-1]
    first_loss = losses[0]
    spark = _sparkline(losses)
    pct = progress * 100
    return (
        f"  loss [{spark}]  "
        f"{first_loss:.4f}→{last_loss:.4f}  "
        f"step {step}  {pct:.0f}%  ~{remaining:.0f}s left"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autoresearch 2.0 experiment runner")
    parser.add_argument("--desc", type=str, default=None, help="Experiment description")
    parser.add_argument("--auto", action="store_true", help="Auto keep/discard (agent mode)")
    parser.add_argument("--detect", action="store_true", help="Detect hardware and exit")
    parser.add_argument("--no-memory", action="store_true", help="Skip memory context display")
    parser.add_argument("--time-budget", type=int, default=None, help="Override TIME_BUDGET for this run (seconds)")
    args = parser.parse_args()

    # Hardware detect only
    if args.detect:
        from hardware import detect_hardware, print_hardware_report
        hw = detect_hardware()
        print_hardware_report(hw)
        return

    # Show memory context
    if not args.no_memory:
        show_memory_context()

    # Experiment description
    description = args.desc
    if not description:
        try:
            description = input("Experiment description (what are you testing?): ").strip()
        except (EOFError, KeyboardInterrupt):
            description = f"auto_{datetime.now().strftime('%m%d_%H%M')}"
    if not description:
        description = f"auto_{datetime.now().strftime('%m%d_%H%M')}"

    # Run training
    log_path = "run.log"
    train_cmd = [sys.executable, "train.py"]
    if args.time_budget is not None:
        train_cmd += ["--time-budget", str(args.time_budget)]
    print(f"\nRunning: {' '.join(train_cmd)}  (output → {log_path})")
    print("Press Ctrl+C to abort.\n")

    LOSS_CURVE_FILE.write_text("")  # clear before run
    t0 = time.time()
    _display_lines = 0
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                train_cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            while proc.poll() is None:
                time.sleep(2.0)
                elapsed = time.time() - t0
                points = _read_loss_curve()
                line = _render_loss_line(points, elapsed)
                # Overwrite previous display line
                if _display_lines:
                    sys.stdout.write("\033[1A\033[2K")
                print(line, flush=True)
                _display_lines = 1
    except KeyboardInterrupt:
        proc.terminate()
        print("\nAborted.")
        return

    # Clear the live display line
    if _display_lines:
        sys.stdout.write("\033[1A\033[2K")

    elapsed = time.time() - t0
    print(f"Finished in {elapsed:.0f}s")

    # Parse results
    metrics = parse_run_log(log_path)
    commit = get_commit()

    val_bpb = metrics.get("val_bpb")
    peak_vram_mb = metrics.get("peak_vram_mb", 0.0)

    if val_bpb is None:
        print("\n[CRASH] val_bpb not found in run.log — training crashed or didn't finish.")
        print("Run: tail -50 run.log")

        if not args.auto:
            try:
                ans = input("Record as crash? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "y"
            if ans in ("", "y", "yes"):
                record_experiment(commit, 0.0, 0.0, "crash", description, metrics)
                print("Reverting: git reset --hard HEAD~1")
                subprocess.run(["git", "reset", "--hard", "HEAD~1"])
        else:
            print("[auto] Recording as crash, reverting.")
            record_experiment(commit, 0.0, 0.0, "crash", description, metrics)
            subprocess.run(["git", "reset", "--hard", "HEAD~1"])
        return

    # Print summary
    print(f"\n{'─'*50}")
    print(f"val_bpb:     {val_bpb:.6f}")
    print(f"vram:        {peak_vram_mb/1024:.1f} GB")
    mfu = metrics.get("mfu_percent")
    if mfu:
        print(f"mfu:         {mfu:.1f}%")
    arch = metrics.get("arch_variants", "")
    if arch:
        print(f"variants:    {arch}")
    print(f"{'─'*50}")

    # Compare to best
    keep = True
    try:
        from agents.memory import ExperimentMemory
        mem = ExperimentMemory()
        stats = mem.get_statistics()
        best = stats.get("best_bpb", float("inf"))
        if best != float("inf"):
            delta = best - val_bpb
            if delta > 0:
                print(f"Improvement: {delta:.6f} better than best ({best:.6f})")
            elif delta < -0.0001:
                print(f"Degradation: {abs(delta):.6f} worse than best ({best:.6f})")
                keep = False
            else:
                print(f"No change vs best ({best:.6f})")
                keep = False
    except Exception:
        pass

    # Keep/discard decision
    if args.auto:
        status = "keep" if keep else "discard"
        print(f"[auto] Decision: {status.upper()}")
    else:
        default = "y" if keep else "n"
        try:
            ans = input(f"Keep this result? [{'Y' if keep else 'N'}/{'n' if keep else 'y'}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = default
        if not ans:
            ans = default
        status = "keep" if ans in ("y", "yes") else "discard"

    # Record
    record_experiment(commit, val_bpb, peak_vram_mb, status, description, metrics)

    # Git action
    if status == "discard":
        print("Reverting: git reset --hard HEAD~1")
        subprocess.run(["git", "reset", "--hard", "HEAD~1"])
        print("Reverted. Ready for next experiment.")
    else:
        print("Keeping. Branch advanced.")


if __name__ == "__main__":
    main()
