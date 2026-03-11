"""
agent.py — Autonomous Claude-powered research agent for Autoresearch 2.0

Runs the experiment loop autonomously using the Anthropic API:
  1. Reads experiment history + train.py AGENT EDIT ZONE
  2. Asks Claude to propose the next experiment change
  3. Applies the suggested edit to train.py
  4. Runs run_loop.py --auto (git commit + train + record + keep/discard)
  5. Repeats forever (until stopped)

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  uv run python agent.py                    # run forever
  uv run python agent.py --max-runs 20      # run 20 experiments
  uv run python agent.py --dry-run          # propose but don't run
  uv run python agent.py --tag mar10        # use existing branch tag
"""

from __future__ import annotations
import os
import re
import sys
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_PY = Path("train.py")
RESULTS_TSV = Path("results.tsv")
MODEL = "claude-opus-4-6"

AGENT_EDIT_ZONE_START = "# ==========================================================================="
AGENT_EDIT_ZONE_HEADER = "# AGENT EDIT ZONE"
AGENT_EDIT_ZONE_END = "# END AGENT EDIT ZONE"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd: list[str], check=True) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def get_agent_edit_zone(train_py_path: Path) -> str:
    """Extract the AGENT EDIT ZONE section from train.py."""
    content = train_py_path.read_text()
    lines = content.splitlines()
    in_zone = False
    zone_lines = []
    for line in lines:
        if AGENT_EDIT_ZONE_HEADER in line:
            in_zone = True
        if in_zone:
            zone_lines.append(line)
        if in_zone and AGENT_EDIT_ZONE_END in line:
            break
    return "\n".join(zone_lines)


def apply_agent_edit_zone(train_py_path: Path, new_zone: str) -> None:
    """Replace the AGENT EDIT ZONE section in train.py with new_zone."""
    content = train_py_path.read_text()
    lines = content.splitlines()

    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if AGENT_EDIT_ZONE_HEADER in line and start_idx is None:
            # Find the === line just before it
            for j in range(i, max(i - 3, -1), -1):
                if AGENT_EDIT_ZONE_START in lines[j]:
                    start_idx = j
                    break
            if start_idx is None:
                start_idx = i
        if start_idx is not None and AGENT_EDIT_ZONE_END in line:
            # Find the === line on or after the END comment
            for j in range(i, min(i + 3, len(lines))):
                if AGENT_EDIT_ZONE_START in lines[j]:
                    end_idx = j + 1
                    break
            if end_idx is None:
                end_idx = i + 1
            break

    if start_idx is None or end_idx is None:
        raise ValueError("Could not locate AGENT EDIT ZONE in train.py")

    new_lines = lines[:start_idx] + new_zone.splitlines() + lines[end_idx:]
    train_py_path.write_text("\n".join(new_lines) + "\n")


def get_results_history() -> str:
    """Return last 20 rows of results.tsv as a string."""
    if not RESULTS_TSV.exists():
        return "(no results yet)"
    lines = RESULTS_TSV.read_text().splitlines()
    header = lines[0] if lines else ""
    rows = lines[1:][-20:]  # last 20 experiments
    return "\n".join([header] + rows) if rows else "(no experiments yet)"


def get_current_branch() -> str:
    try:
        return run_cmd(["git", "branch", "--show-current"])
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    return """You are an autonomous ML research agent running Autoresearch 2.0 experiments.

Your job: propose and implement ONE experiment per turn to minimize val_bpb (bits per byte) on a language model trained for 5 minutes.

You will receive:
- The current AGENT EDIT ZONE from train.py
- The experiment history (results.tsv)
- Hardware info (device, constraints)

You must respond with EXACTLY two sections:

DESCRIPTION: <one-line description of the change, ≤80 chars>

AGENT EDIT ZONE:
```python
# ===========================================================================
# AGENT EDIT ZONE
# This is the only section you need to modify to run experiments.
# Edit these constants directly — changes take effect immediately on next run.
# ===========================================================================

<full AGENT EDIT ZONE contents with your change applied>

# ===========================================================================
# END AGENT EDIT ZONE
# ===========================================================================
```

Rules:
- Change exactly ONE thing per experiment (one variable or one flag).
- Keep all other constants at their current values.
- Never add imports, functions, or code outside the AGENT EDIT ZONE.
- Never change TOTAL_BATCH_SIZE beyond 2**21 (memory risk).
- On MPS (Apple Silicon), DEPTH≤4, ASPECT_RATIO≤32, DEVICE_BATCH_SIZE≤4, MAX_SEQ_LEN≤512 are hard limits.
- Respond with ONLY the two sections above. No extra explanation."""


def build_user_prompt(edit_zone: str, results: str, device: str) -> str:
    return f"""## Current AGENT EDIT ZONE (train.py)

```python
{edit_zone}
```

## Experiment history (results.tsv, newest last)

```
{results}
```

## Hardware

Device: {device}

## Task

Propose the single most promising change to minimize val_bpb.
Think step-by-step about what the history shows, then propose ONE change."""


def parse_claude_response(response: str) -> tuple[str, str]:
    """Parse DESCRIPTION and AGENT EDIT ZONE from Claude's response."""
    desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    description = desc_match.group(1).strip() if desc_match else f"auto_{datetime.now().strftime('%m%d_%H%M')}"

    zone_match = re.search(
        r"```python\s*(# =+\s*# AGENT EDIT ZONE.+?# END AGENT EDIT ZONE\s*# =+)\s*```",
        response, re.DOTALL | re.IGNORECASE
    )
    if not zone_match:
        # Try without outer === lines
        zone_match = re.search(
            r"```python\s*(# AGENT EDIT ZONE.+?# END AGENT EDIT ZONE)\s*```",
            response, re.DOTALL | re.IGNORECASE
        )

    if not zone_match:
        raise ValueError(f"Could not parse AGENT EDIT ZONE from Claude response:\n{response[:500]}")

    new_zone = zone_match.group(1).strip()
    return description, new_zone


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Detect the hardware device string."""
    try:
        result = run_cmd([sys.executable, "run_loop.py", "--detect"], check=False)
        for line in result.splitlines():
            if line.startswith("device:"):
                return line.split(":", 1)[1].strip()
        if "mps" in result.lower():
            return "mps"
        if "cuda" in result.lower():
            return "cuda"
    except Exception:
        pass
    return "unknown"


def run_experiment(description: str, dry_run: bool = False) -> bool:
    """
    Commit current train.py and run the experiment via run_loop.py --auto.
    Returns True if the experiment was kept.
    """
    # Git commit
    run_cmd(["git", "add", "train.py"])
    run_cmd(["git", "commit", "-m", f"experiment: {description}"])
    commit = run_cmd(["git", "rev-parse", "--short", "HEAD"])
    print(f"  Committed: {commit}")

    if dry_run:
        print("  [dry-run] Skipping actual training run.")
        return False

    # Run the experiment
    print(f"  Running experiment: {description}")
    result = subprocess.run(
        [sys.executable, "run_loop.py", "--auto", "--desc", description, "--no-memory"],
        capture_output=False,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Autonomous Claude-powered research agent")
    parser.add_argument("--max-runs", type=int, default=0, help="Max experiments (0 = infinite)")
    parser.add_argument("--dry-run", action="store_true", help="Propose edits but don't train")
    parser.add_argument("--tag", type=str, default=None, help="Branch tag (e.g. mar10)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Setup branch if requested
    if args.tag:
        branch = f"autoresearch/{args.tag}"
        try:
            run_cmd(["git", "checkout", "-b", branch])
            print(f"Created branch: {branch}")
        except RuntimeError:
            run_cmd(["git", "checkout", branch])
            print(f"Switched to branch: {branch}")

    branch = get_current_branch()
    print(f"Branch: {branch}")

    device = detect_device()
    print(f"Device: {device}")

    system_prompt = build_system_prompt()
    run_count = 0

    print("\n" + "="*60)
    print("AUTORESEARCH 2.0 AGENT LOOP — Press Ctrl+C to stop")
    print("="*60 + "\n")

    while True:
        if args.max_runs > 0 and run_count >= args.max_runs:
            print(f"Reached max runs ({args.max_runs}). Stopping.")
            break

        run_count += 1
        print(f"\n{'─'*60}")
        print(f"Run #{run_count}  |  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*60}")

        # Read current state
        edit_zone = get_agent_edit_zone(TRAIN_PY)
        results = get_results_history()

        print("Asking Claude for next experiment...")

        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": build_user_prompt(edit_zone, results, device)
                }]
            )
            response = message.content[0].text
        except Exception as e:
            print(f"  Claude API error: {e}")
            print("  Waiting 30s before retry...")
            time.sleep(30)
            continue

        # Parse response
        try:
            description, new_zone = parse_claude_response(response)
        except ValueError as e:
            print(f"  Parse error: {e}")
            print("  Skipping this run.")
            continue

        print(f"  Proposal: {description}")

        # Show the diff
        old_zone = get_agent_edit_zone(TRAIN_PY)
        if old_zone.strip() == new_zone.strip():
            print("  Warning: Claude proposed no change. Skipping.")
            continue

        # Apply edit
        try:
            apply_agent_edit_zone(TRAIN_PY, new_zone)
        except Exception as e:
            print(f"  Failed to apply edit: {e}")
            continue

        # Run experiment
        try:
            kept = run_experiment(description, dry_run=args.dry_run)
            status = "kept" if kept else "discarded/crashed"
            print(f"  Result: {status}")
        except Exception as e:
            print(f"  Experiment failed: {e}")
            # Try to revert
            try:
                subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)
                print("  Reverted failed commit.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
