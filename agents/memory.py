"""
Experiment memory and knowledge base for autoresearch.
Stores experiment history, learns from patterns, and provides insights.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path


@dataclass
class ExperimentRecord:
    """Record of a single experiment."""
    commit: str
    val_bpb: float
    memory_mb: float
    status: str  # "keep", "discard", "crash"
    description: str
    timestamp: str
    config_snapshot: Dict[str, Any]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> ExperimentRecord:
        return cls(**data)


class ExperimentMemory:
    """
    Persistent memory for experiment history.
    Learns patterns and provides insights for future experiments.
    """
    
    def __init__(self, memory_path: str = "./experiment_memory.json"):
        self.memory_path = Path(memory_path)
        self.experiments: List[ExperimentRecord] = []
        self.best_bpb: float = float('inf')
        self.best_commit: str = ""
        self.load()
    
    def load(self) -> None:
        """Load memory from disk."""
        if self.memory_path.exists():
            with open(self.memory_path, "r") as f:
                data = json.load(f)
            self.experiments = [ExperimentRecord.from_dict(e) for e in data.get("experiments", [])]
            self.best_bpb = data.get("best_bpb", float('inf'))
            self.best_commit = data.get("best_commit", "")
            print(f"Loaded experiment memory: {len(self.experiments)} experiments")
    
    def save(self) -> None:
        """Save memory to disk."""
        data = {
            "experiments": [e.to_dict() for e in self.experiments],
            "best_bpb": self.best_bpb,
            "best_commit": self.best_commit,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.memory_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_experiment(self, record: ExperimentRecord) -> None:
        """Add a new experiment record."""
        self.experiments.append(record)
        
        if record.status == "keep" and record.val_bpb < self.best_bpb:
            self.best_bpb = record.val_bpb
            self.best_commit = record.commit
        
        self.save()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of all experiments."""
        if not self.experiments:
            return {"total": 0}
        
        kept = [e for e in self.experiments if e.status == "keep"]
        discarded = [e for e in self.experiments if e.status == "discard"]
        crashed = [e for e in self.experiments if e.status == "crash"]
        
        return {
            "total": len(self.experiments),
            "kept": len(kept),
            "discarded": len(discarded),
            "crashed": len(crashed),
            "keep_rate": len(kept) / len(self.experiments) if self.experiments else 0,
            "best_bpb": self.best_bpb,
            "best_commit": self.best_commit,
            "avg_bpb_kept": sum(e.val_bpb for e in kept) / len(kept) if kept else 0,
            "total_improvement": self.experiments[0].val_bpb - self.best_bpb if kept else 0,
        }
    
    def get_recent_experiments(self, n: int = 10) -> List[ExperimentRecord]:
        """Get the N most recent experiments."""
        return self.experiments[-n:]
    
    def get_best_experiments(self, n: int = 5) -> List[ExperimentRecord]:
        """Get the N best experiments by val_bpb."""
        kept = [e for e in self.experiments if e.status == "keep"]
        return sorted(kept, key=lambda e: e.val_bpb)[:n]
    
    def get_failed_experiments(self, n: int = 10) -> List[ExperimentRecord]:
        """Get recent failed experiments (discard or crash)."""
        failed = [e for e in self.experiments if e.status in ["discard", "crash"]]
        return failed[-n:]
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in experiment history."""
        patterns = {
            "successful_changes": [],
            "failed_changes": [],
            "recommendations": [],
        }
        
        if len(self.experiments) < 2:
            return patterns
        
        # Analyze what types of changes work
        kept = [e for e in self.experiments if e.status == "keep"]
        discarded = [e for e in self.experiments if e.status == "discard"]
        
        # Simple pattern: count keywords in descriptions
        change_types = {
            "learning_rate": {"keep": 0, "discard": 0},
            "batch_size": {"keep": 0, "discard": 0},
            "model_size": {"keep": 0, "discard": 0},
            "architecture": {"keep": 0, "discard": 0},
            "optimizer": {"keep": 0, "discard": 0},
        }
        
        for exp in kept + discarded:
            desc_lower = exp.description.lower()
            status = "keep" if exp in kept else "discard"
            
            if "lr" in desc_lower or "learning rate" in desc_lower:
                change_types["learning_rate"][status] += 1
            if "batch" in desc_lower:
                change_types["batch_size"][status] += 1
            if "depth" in desc_lower or "width" in desc_lower or "layer" in desc_lower:
                change_types["model_size"][status] += 1
            if "attention" in desc_lower or "activation" in desc_lower or "norm" in desc_lower:
                change_types["architecture"][status] += 1
            if "muon" in desc_lower or "adam" in desc_lower or "optimizer" in desc_lower:
                change_types["optimizer"][status] += 1
        
        patterns["change_type_success"] = {}
        for change_type, counts in change_types.items():
            total = counts["keep"] + counts["discard"]
            if total > 0:
                patterns["change_type_success"][change_type] = {
                    "keep": counts["keep"],
                    "discard": counts["discard"],
                    "rate": counts["keep"] / total,
                }
        
        # Generate recommendations
        for change_type, stats in patterns["change_type_success"].items():
            if stats["rate"] > 0.6 and stats["keep"] >= 2:
                patterns["recommendations"].append(
                    f"{change_type} changes have {stats['rate']*100:.0f}% success rate - continue exploring"
                )
            elif stats["rate"] < 0.3 and stats["discard"] >= 3:
                patterns["recommendations"].append(
                    f"{change_type} changes have low success rate - try different approach"
                )
        
        return patterns
    
    def generate_hypothesis(self) -> str:
        """Generate a hypothesis for the next experiment based on memory."""
        stats = self.get_statistics()
        patterns = self.analyze_patterns()
        
        if stats["total"] == 0:
            return "Establish baseline with default configuration."
        
        # Get recent trend
        recent = self.get_recent_experiments(5)
        if len(recent) >= 3:
            recent_kept = [e for e in recent if e.status == "keep"]
            if len(recent_kept) >= 2:
                trend = recent_kept[0].val_bpb - recent_kept[-1].val_bpb
                if trend > 0.01:
                    return f"Recent experiments show improvement trend ({trend:.4f} bpb gain). Continue current strategy with refined hyperparameters."
                elif trend < -0.01:
                    return f"Recent experiments show degradation ({abs(trend):.4f} bpb loss). Consider resetting to best configuration (commit {self.best_commit}) and trying orthogonal direction."
        
        # Pattern-based suggestions
        if patterns["recommendations"]:
            return f"Based on {stats['total']} experiments: {patterns['recommendations'][0]}"
        
        # Default suggestions based on state
        if stats["keep_rate"] < 0.3:
            return "Low keep rate suggests exploring fundamentally different approaches. Consider architecture changes or hyperparameter resets."
        elif stats["keep_rate"] > 0.7:
            return "High keep rate indicates good search direction. Continue with incremental improvements."
        else:
            return "Moderate success rate. Balance exploration (new directions) with exploitation (refining successful changes)."
    
    def get_context_for_agent(self, agent_type: str) -> str:
        """Get relevant context for a specific agent type."""
        stats = self.get_statistics()
        recent = self.get_recent_experiments(5)
        
        best_bpb = stats.get('best_bpb', float('inf'))
        best_bpb_str = f"{best_bpb:.6f}" if best_bpb != float('inf') else "N/A"
        
        context = f"""
=== Experiment Memory Context for {agent_type} ===
Total experiments: {stats.get('total', 0)}
Keep rate: {stats.get('keep_rate', 0)*100:.1f}%
Best val_bpb: {best_bpb_str} (commit: {stats.get('best_commit', 'N/A')})

Recent experiments:
"""
        for i, exp in enumerate(recent[-3:], 1):
            context += f"  {i}. [{exp.status}] {exp.description} -> bpb={exp.val_bpb:.6f}\n"
        
        patterns = self.analyze_patterns()
        if patterns.get("recommendations"):
            context += "\nRecommendations:\n"
            for rec in patterns["recommendations"][:2]:
                context += f"  - {rec}\n"
        
        return context.strip()
