"""
Utility functions for the agent framework.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any


def get_current_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return "unknown"


def get_current_branch() -> str:
    """Get current git branch."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return "unknown"


def parse_train_log(log_path: str = "run.log") -> Optional[Dict[str, Any]]:
    """Parse training log and extract metrics."""
    path = Path(log_path)
    if not path.exists():
        return None
    
    metrics = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("val_bpb:"):
                metrics["val_bpb"] = float(line.split(":")[1].strip())
            elif line.startswith("peak_vram_mb:"):
                metrics["memory_mb"] = float(line.split(":")[1].strip())
            elif line.startswith("training_seconds:"):
                metrics["training_seconds"] = float(line.split(":")[1].strip())
            elif line.startswith("mfu_percent:"):
                metrics["mfu"] = float(line.split(":")[1].strip())
            elif line.startswith("total_tokens_M:"):
                metrics["total_tokens_M"] = float(line.split(":")[1].strip())
    
    return metrics if metrics else None


def create_experiment_record(
    description: str,
    status: str = "pending",
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create an experiment record dictionary."""
    from datetime import datetime
    
    return {
        "commit": get_current_commit(),
        "val_bpb": metrics.get("val_bpb", 0.0) if metrics else 0.0,
        "memory_mb": metrics.get("memory_mb", 0.0) if metrics else 0.0,
        "status": status,
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "config_snapshot": config or {},
        "metrics": metrics or {},
    }


def print_agent_suggestions(suggestions: list, agent_name: str) -> None:
    """Pretty print agent suggestions."""
    print(f"\n{'='*60}")
    print(f"{agent_name} Suggestions")
    print('='*60)
    
    for i, sug in enumerate(suggestions[:5], 1):  # Top 5
        print(f"\n{i}. [{sug['risk'].upper()}] {sug['change']}")
        print(f"   Rationale: {sug['rationale']}")
        print(f"   Expected: {sug.get('expected_impact', 'N/A')}")
    
    print()


def print_memory_summary(memory) -> None:
    """Print experiment memory summary."""
    stats = memory.get_statistics()
    
    print(f"\n{'='*60}")
    print("Experiment Memory Summary")
    print('='*60)
    print(f"Total experiments: {stats['total']}")
    print(f"Keep rate: {stats.get('keep_rate', 0)*100:.1f}%")
    print(f"Best val_bpb: {stats.get('best_bpb', 'N/A'):.6f}" if stats.get('best_bpb') != float('inf') else "Best val_bpb: N/A")
    print(f"Total improvement: {stats.get('total_improvement', 0):.6f} bpb")
    print()
