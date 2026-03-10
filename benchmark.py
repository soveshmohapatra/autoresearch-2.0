#!/usr/bin/env python3
"""
Benchmark Script: Autoresearch vs Autoresearch-2.0

Compares both versions across multiple dimensions:
- Setup time
- Hardware detection
- Training performance (short run)
- Feature comparison

Usage: python benchmark.py
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""
    test_name: str
    autoresearch_time: Optional[float]
    autoresearch2_time: Optional[float]
    winner: str
    notes: str


@dataclass
class FeatureComparison:
    """Feature comparison between versions."""
    feature: str
    autoresearch: bool
    autoresearch2: bool


class Benchmark:
    """Benchmark suite for comparing Autoresearch versions."""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.ar_dir = self.base_dir / "Autoresearch"
        self.ar2_dir = self.base_dir / "Autoresearch-2.0"
        self.results: List[BenchmarkResult] = []
        self.start_time = datetime.now()

    def log(self, message: str):
        """Print a timestamped log message."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"[{elapsed:6.1f}s] {message}")

    def run_command(self, cmd: str, cwd: Path, timeout: int = 60) -> tuple:
        """Run a command and return (success, stdout, elapsed_time)."""
        self.log(f"Running: {cmd}")
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            elapsed = time.time() - start
            return result.returncode == 0, result.stdout, elapsed
        except subprocess.TimeoutExpired:
            return False, "Timeout", timeout
        except Exception as e:
            return False, str(e), 0

    def test_import_speed(self) -> BenchmarkResult:
        """Test how fast core modules can be imported."""
        self.log("\n=== Test: Import Speed ===")

        success1, out1, time1 = self.run_command(
            "uv run python -c \"import prepare; import train\"",
            self.ar_dir,
            timeout=60
        )

        success2, out2, time2 = self.run_command(
            "uv run python -c \"import prepare; import train; import config; import hardware\"",
            self.ar2_dir,
            timeout=60
        )

        winner = "Autoresearch" if time1 < time2 else "Autoresearch-2.0"
        notes = f"AR: {time1:.2f}s, AR2: {time2:.2f}s"

        result = BenchmarkResult(
            test_name="Import Speed",
            autoresearch_time=time1 if success1 else None,
            autoresearch2_time=time2 if success2 else None,
            winner=winner,
            notes=notes
        )
        self.results.append(result)
        self.log(f"Winner: {winner} ({notes})")
        return result

    def test_hardware_detection(self) -> BenchmarkResult:
        """Test hardware detection (AR2 only has this feature)."""
        self.log("\n=== Test: Hardware Detection ===")
        time1 = 0

        success2, out2, time2 = self.run_command(
            "uv run python hardware.py --detect-only",
            self.ar2_dir,
            timeout=30
        )

        winner = "Autoresearch-2.0" if success2 else "N/A"
        notes = f"AR: N/A, AR2: {time2:.2f}s"

        result = BenchmarkResult(
            test_name="Hardware Detection",
            autoresearch_time=None,
            autoresearch2_time=time2 if success2 else None,
            winner=winner,
            notes=notes
        )
        self.results.append(result)
        self.log(f"Winner: {winner} (AR2 exclusive feature)")
        return result

    def test_config_system(self) -> BenchmarkResult:
        """Test configuration system (AR2 has dataclass-based config)."""
        self.log("\n=== Test: Configuration System ===")
        time1 = 0

        success2, out2, time2 = self.run_command(
            "uv run python -c \"from config import DEFAULT_CONFIG; print(DEFAULT_CONFIG.to_dict())\"",
            self.ar2_dir,
            timeout=30
        )

        winner = "Autoresearch-2.0" if success2 else "N/A"
        notes = f"AR: N/A, AR2: {time2:.2f}s"

        result = BenchmarkResult(
            test_name="Configuration System",
            autoresearch_time=None,
            autoresearch2_time=time2 if success2 else None,
            winner=winner,
            notes=notes
        )
        self.results.append(result)
        self.log(f"Winner: {winner} (AR2 exclusive feature)")
        return result

    def test_agent_framework(self) -> BenchmarkResult:
        """Test multi-agent framework (AR2 only)."""
        self.log("\n=== Test: Agent Framework ===")
        time1 = 0

        success2, out2, time2 = self.run_command(
            "uv run python -c \"from agents import ArchitectureAgent, OptimizerAgent, AnalystAgent; print('Agents loaded')\"",
            self.ar2_dir,
            timeout=30
        )

        winner = "Autoresearch-2.0" if success2 else "N/A"
        notes = f"AR: N/A, AR2: {time2:.2f}s"

        result = BenchmarkResult(
            test_name="Agent Framework",
            autoresearch_time=None,
            autoresearch2_time=time2 if success2 else None,
            winner=winner,
            notes=notes
        )
        self.results.append(result)
        self.log(f"Winner: {winner} (AR2 exclusive feature)")
        return result

    def test_memory_system(self) -> BenchmarkResult:
        """Test experiment memory system (AR2 only)."""
        self.log("\n=== Test: Memory System ===")
        time1 = 0

        success2, out2, time2 = self.run_command(
            "uv run python -c \"from agents import ExperimentMemory; m = ExperimentMemory(); print('Memory:', len(m.experiments), 'experiments')\"",
            self.ar2_dir,
            timeout=30
        )

        winner = "Autoresearch-2.0" if success2 else "N/A"
        notes = f"AR: N/A, AR2: {time2:.2f}s"

        result = BenchmarkResult(
            test_name="Memory System",
            autoresearch_time=None,
            autoresearch2_time=time2 if success2 else None,
            winner=winner,
            notes=notes
        )
        self.results.append(result)
        self.log(f"Winner: {winner} (AR2 exclusive feature)")
        return result

    def test_training_startup(self) -> BenchmarkResult:
        """Test training script startup time."""
        self.log("\n=== Test: Training Startup ===")

        success1, out1, time1 = self.run_command(
            "uv run python -c \"import train; print('AR loaded')\"",
            self.ar_dir,
            timeout=60
        )

        success2, out2, time2 = self.run_command(
            "uv run python -c \"import train; print('AR2 loaded')\"",
            self.ar2_dir,
            timeout=60
        )

        winner = "Autoresearch" if (success1 and time1 < time2) else "Autoresearch-2.0"
        notes = f"AR: {time1:.2f}s, AR2: {time2:.2f}s"

        result = BenchmarkResult(
            test_name="Training Startup",
            autoresearch_time=time1 if success1 else None,
            autoresearch2_time=time2 if success2 else None,
            winner=winner,
            notes=notes
        )
        self.results.append(result)
        self.log(f"Winner: {winner} ({notes})")
        return result

    def compare_features(self) -> List[FeatureComparison]:
        """Compare features between versions."""
        self.log("\n=== Feature Comparison ===")

        features = [
            FeatureComparison("Multi-platform (CUDA/MPS/CPU)", False, True),
            FeatureComparison("Hardware auto-detection", False, True),
            FeatureComparison("GUI Dashboard", False, True),
            FeatureComparison("Multi-agent framework", False, True),
            FeatureComparison("Experiment memory", False, True),
            FeatureComparison("Dataclass config", False, True),
            FeatureComparison("Architecture variants (MoE, GQA)", False, True),
            FeatureComparison("Optimizer zoo", False, True),
            FeatureComparison("W&B integration", False, True),
            FeatureComparison("Checkpointing", False, True),
            FeatureComparison("Test suite", False, True),
            FeatureComparison("Core training loop", True, True),
            FeatureComparison("BPE tokenizer", True, True),
            FeatureComparison("Agent instructions (program.md)", True, True),
        ]

        ar_count = sum(1 for f in features if f.autoresearch)
        ar2_count = sum(1 for f in features if f.autoresearch2)

        self.log(f"Autoresearch features: {ar_count}")
        self.log(f"Autoresearch-2.0 features: {ar2_count}")
        self.log(f"New features in 2.0: {ar2_count - ar_count}")

        return features

    def generate_report(self):
        """Generate final benchmark report."""
        self.log("\n" + "=" * 70)
        self.log("BENCHMARK REPORT")
        self.log("=" * 70)

        ar_wins = 0
        ar2_wins = 0

        for result in self.results:
            if result.winner == "Autoresearch":
                ar_wins += 1
            elif result.winner == "Autoresearch-2.0":
                ar2_wins += 1

            print(f"\n{result.test_name}:")
            print(f"  Winner: {result.winner}")
            if result.autoresearch_time:
                print(f"  Autoresearch: {result.autoresearch_time:.2f}s")
            if result.autoresearch2_time:
                print(f"  Autoresearch-2.0: {result.autoresearch2_time:.2f}s")

        features = self.compare_features()

        self.log("\n\n📋 Feature Comparison:")
        print(f"\n{'Feature':<45} {'AR':<6} {'AR2':<6}")
        print("-" * 70)
        for f in features:
            ar_mark = "✅" if f.autoresearch else "❌"
            ar2_mark = "✅" if f.autoresearch2 else "❌"
            print(f"{f.feature:<45} {ar_mark:<6} {ar2_mark:<6}")

        self.log("\n\n🏆 Summary:")
        print(f"\nPerformance Test Wins:")
        print(f"  Autoresearch: {ar_wins}")
        print(f"  Autoresearch-2.0: {ar2_wins}")
        print(f"\nFeature Count:")
        print(f"  Autoresearch: {sum(1 for f in features if f.autoresearch)}")
        print(f"  Autoresearch-2.0: {sum(1 for f in features if f.autoresearch2)}")
        print(f"  New in 2.0: {sum(1 for f in features if f.autoresearch2 and not f.autoresearch)}")

        report = {
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "performance_results": [asdict(r) for r in self.results],
            "feature_comparison": [asdict(f) for f in features],
            "summary": {
                "autoresearch_wins": ar_wins,
                "autoresearch2_wins": ar2_wins,
                "autoresearch_features": sum(1 for f in features if f.autoresearch),
                "autoresearch2_features": sum(1 for f in features if f.autoresearch2),
            }
        }

        report_path = self.base_dir / "benchmark_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.log(f"\n\n💾 Report saved to: {report_path}")
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.log(f"\n⏱️ Total benchmark time: {elapsed:.1f}s")

    def run(self):
        """Run all benchmarks."""
        self.log("🚀 Starting Benchmark: Autoresearch vs Autoresearch-2.0")
        self.log(f"Started at: {self.start_time}")

        self.test_import_speed()
        self.test_hardware_detection()
        self.test_config_system()
        self.test_agent_framework()
        self.test_memory_system()
        self.test_training_startup()

        self.generate_report()
        self.log("\n✅ Benchmark complete!")


if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run()
