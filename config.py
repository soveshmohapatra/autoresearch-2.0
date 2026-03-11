"""
Configuration system for autoresearch.
Dataclass-based configuration for easy experimentation and serialization.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import json
import os


@dataclass
class DeviceConfig:
    """Device configuration with auto-detection."""
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    
    def get_device(self) -> str:
        """Get the actual device string, auto-detecting if needed."""
        import torch
        
        if self.device != "auto":
            return self.device
        
        # Auto-detect: prefer CUDA > MPS > CPU
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def get_torch_dtype(self) -> torch.dtype:
        """Get the PyTorch dtype."""
        import torch
        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        # MPS doesn't support bfloat16 on older macOS
        if self.get_device() == "mps" and self.dtype == "bfloat16":
            return torch.float32
        return dtypes[self.dtype]
    
    def get_peak_flops(self) -> float:
        """Get peak FLOPS for MFU calculation."""
        device = self.get_device()
        if device == "cuda":
            return 989.5e12
        elif device == "mps":
            return 3e12  # M-series base; hardware.py uses accurate per-chip values
        else:
            return 50e9


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    depth: int = 8
    aspect_ratio: int = 64
    head_dim: int = 128
    window_pattern: str = "SSSL"

    # Architecture variants
    use_moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 2
    use_gqa: bool = False
    gqa_kv_groups: int = 4
    use_swiglu: bool = False
    use_geglu: bool = False
    use_prenorm: bool = False

    @property
    def model_dim(self) -> int:
        base_dim = self.depth * self.aspect_ratio
        return ((base_dim + self.head_dim - 1) // self.head_dim) * self.head_dim
    
    @property
    def num_heads(self) -> int:
        return self.model_dim // self.head_dim
    
    def to_gpt_config(self, vocab_size: int, seq_len: int) -> dict:
        return {
            "sequence_len": seq_len,
            "vocab_size": vocab_size,
            "n_layer": self.depth,
            "n_head": self.num_heads,
            "n_kv_head": self.num_heads,
            "n_embd": self.model_dim,
            "window_pattern": self.window_pattern,
        }


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    optimizer_type: str = "muon_adamw"  # "muon_adamw", "lion", "adafactor"
    embedding_lr: float = 0.6
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.04
    scalar_lr: float = 0.5
    weight_decay: float = 0.2
    adam_betas: tuple = (0.8, 0.95)
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_beta2: float = 0.95


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    total_batch_size: int = 2**19
    device_batch_size: int = 128
    time_budget: int = 300
    max_seq_len: int = 2048
    eval_tokens: int = 40 * 524288
    gc_interval: int = 5000


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = False
    project: str = "autoresearch"
    entity: Optional[str] = None
    tags: list = field(default_factory=list)
    notes: Optional[str] = None
    
    def should_log(self) -> bool:
        return self.enabled and os.environ.get("WANDB_DISABLED", "false").lower() != "true"


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    enabled: bool = True
    save_dir: str = "./checkpoints"
    save_interval: int = 60
    keep_last: int = 3


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    tag: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> ExperimentConfig:
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        # Reconstruct nested dataclasses
        data["device"] = DeviceConfig(**data["device"])
        data["model"] = ModelConfig(**data["model"])
        data["optimizer"] = OptimizerConfig(**data["optimizer"])
        data["training"] = TrainingConfig(**data["training"])
        data["wandb"] = WandbConfig(**data["wandb"])
        data["checkpoint"] = CheckpointConfig(**data["checkpoint"])
        
        return cls(**data)
    
    def get_commit_hash(self) -> str:
        import subprocess
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            return "unknown"


DEFAULT_CONFIG = ExperimentConfig()
