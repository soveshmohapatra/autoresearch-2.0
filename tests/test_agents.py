"""Tests for agent framework."""

import pytest
import sys
import json
sys.path.insert(0, '..')

from agents import (
    ExperimentMemory, ExperimentRecord,
    ArchitectureAgent, OptimizerAgent, HyperparameterAgent, AnalystAgent
)


class TestExperimentMemory:
    """Test ExperimentMemory."""
    
    def test_init(self, tmp_path):
        """Test memory initialization."""
        memory_path = tmp_path / "memory.json"
        memory = ExperimentMemory(str(memory_path))
        assert memory.experiments == []
        assert memory.best_bpb == float('inf')
    
    def test_add_experiment(self, tmp_path):
        """Test adding experiments."""
        memory_path = tmp_path / "memory.json"
        memory = ExperimentMemory(str(memory_path))
        
        record = ExperimentRecord(
            commit="abc123",
            val_bpb=1.5,
            memory_mb=1024,
            status="keep",
            description="test experiment",
            timestamp="2024-01-01T00:00:00",
            config_snapshot={},
            metrics={},
        )
        memory.add_experiment(record)
        
        assert len(memory.experiments) == 1
        assert memory.best_bpb == 1.5
        assert memory.best_commit == "abc123"
    
    def test_statistics(self, tmp_path):
        """Test statistics calculation."""
        memory_path = tmp_path / "memory.json"
        memory = ExperimentMemory(str(memory_path))
        
        # Add some experiments
        for i, (bpb, status) in enumerate([
            (1.5, "keep"),
            (1.6, "discard"),
            (1.4, "keep"),
        ]):
            record = ExperimentRecord(
                commit=f"abc{i}",
                val_bpb=bpb,
                memory_mb=1024,
                status=status,
                description=f"experiment {i}",
                timestamp="2024-01-01T00:00:00",
                config_snapshot={},
                metrics={},
            )
            memory.add_experiment(record)
        
        stats = memory.get_statistics()
        assert stats["total"] == 3
        assert stats["kept"] == 2
        assert stats["discarded"] == 1
        assert stats["best_bpb"] == 1.4
    
    def test_persistence(self, tmp_path):
        """Test memory persistence."""
        memory_path = tmp_path / "memory.json"
        
        # Create and save
        memory1 = ExperimentMemory(str(memory_path))
        record = ExperimentRecord(
            commit="abc123",
            val_bpb=1.5,
            memory_mb=1024,
            status="keep",
            description="test",
            timestamp="2024-01-01T00:00:00",
            config_snapshot={},
            metrics={},
        )
        memory1.add_experiment(record)
        
        # Load and verify
        memory2 = ExperimentMemory(str(memory_path))
        assert len(memory2.experiments) == 1
        assert memory2.best_bpb == 1.5


class TestArchitectureAgent:
    """Test ArchitectureAgent."""
    
    def test_init(self):
        """Test agent initialization."""
        agent = ArchitectureAgent()
        assert agent.name == "ArchitectureAgent"
        assert len(agent.specialties) > 0
    
    def test_get_suggestions(self):
        """Test suggestion generation."""
        agent = ArchitectureAgent()
        config = {"depth": 4, "aspect_ratio": 32, "window_pattern": "SSSL"}
        suggestions = agent.get_suggestions(config)
        
        assert len(suggestions) > 0
        for sug in suggestions:
            assert "type" in sug
            assert "change" in sug
            assert "rationale" in sug
            assert "risk" in sug


class TestOptimizerAgent:
    """Test OptimizerAgent."""
    
    def test_init(self):
        """Test agent initialization."""
        agent = OptimizerAgent()
        assert agent.name == "OptimizerAgent"
    
    def test_get_suggestions(self):
        """Test suggestion generation."""
        agent = OptimizerAgent()
        config = {"matrix_lr": 0.04, "weight_decay": 0.2}
        suggestions = agent.get_suggestions(config)
        
        assert len(suggestions) > 0


class TestAnalystAgent:
    """Test AnalystAgent."""
    
    def test_analyze_result_improvement(self):
        """Test result analysis - improvement case."""
        analyst = AnalystAgent()
        
        result = {"val_bpb": 1.4, "memory_mb": 1024, "status": "keep"}
        baseline = {"val_bpb": 1.5, "memory_mb": 1000}
        
        analysis = analyst.analyze_result(result, baseline)
        assert analysis["decision"] == "keep"
        assert analysis["improvement"] > 0
    
    def test_analyze_result_degradation(self):
        """Test result analysis - degradation case."""
        analyst = AnalystAgent()
        
        result = {"val_bpb": 1.6, "memory_mb": 1024, "status": "keep"}
        baseline = {"val_bpb": 1.5, "memory_mb": 1000}
        
        analysis = analyst.analyze_result(result, baseline)
        assert analysis["decision"] == "discard"
        assert analysis["improvement"] < 0
    
    def test_generate_insights(self):
        """Test insight generation."""
        analyst = AnalystAgent()
        
        experiments = [
            {"status": "keep", "description": "increase lr", "val_bpb": 1.5},
            {"status": "keep", "description": "increase lr more", "val_bpb": 1.4},
            {"status": "discard", "description": "change activation", "val_bpb": 1.6},
        ]
        
        insights = analyst.generate_insights(experiments)
        assert isinstance(insights, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
