"""
Multi-agent framework for autoresearch.
Specialized agents for different aspects of neural network experimentation.
"""

from .memory import ExperimentMemory, ExperimentRecord
from .architecture import ArchitectureAgent
from .optimizer import OptimizerAgent
from .hyperparameter import HyperparameterAgent
from .analyst import AnalystAgent
from .utils import (
    get_current_commit,
    get_current_branch,
    parse_train_log,
    create_experiment_record,
    print_agent_suggestions,
    print_memory_summary,
)

__all__ = [
    "ExperimentMemory",
    "ExperimentRecord", 
    "ArchitectureAgent",
    "OptimizerAgent",
    "HyperparameterAgent",
    "AnalystAgent",
    "get_current_commit",
    "get_current_branch",
    "parse_train_log",
    "create_experiment_record",
    "print_agent_suggestions",
    "print_memory_summary",
]
