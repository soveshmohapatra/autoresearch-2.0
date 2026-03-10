"""
LLM Model Catalog for Autoresearch.
Defines different model architectures and their hardware requirements.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelSize(Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


@dataclass
class ModelConfig:
    """Configuration for a specific LLM architecture."""
    name: str
    size_category: ModelSize
    description: str
    depth: int
    aspect_ratio: int
    head_dim: int = 128
    window_pattern: str = "SSSL"
    use_moe: bool = False
    moe_num_experts: int = 4
    use_gqa: bool = False
    use_swiglu: bool = True
    use_prenorm: bool = True
    recommended_batch_size: int = 64
    recommended_seq_len: int = 2048
    optimizer: str = "muon_adamw"
    min_vram_gb: float = 8.0
    recommended_vram_gb: float = 16.0

    @property
    def model_dim(self) -> int:
        base_dim = self.depth * self.aspect_ratio
        return ((base_dim + self.head_dim - 1) // self.head_dim) * self.head_dim

    @property
    def num_heads(self) -> int:
        return self.model_dim // self.head_dim

    @property
    def param_count_millions(self) -> float:
        vocab = 8192
        layer_params = 4 * (self.model_dim ** 2)
        if self.use_moe:
            layer_params *= (self.moe_num_experts // 2)
        total_layer_params = layer_params * self.depth
        embed_params = vocab * self.model_dim
        total_params = embed_params + total_layer_params + embed_params
        return total_params / 1e6

    def is_compatible(self, hardware_info: Dict[str, Any]) -> bool:
        device_vram = hardware_info.get('total_memory_gb', 0)
        if hardware_info.get('device_type') == 'mps':
            return device_vram >= self.min_vram_gb * 0.6
        if hardware_info.get('device_type') == 'cuda':
            return device_vram >= self.min_vram_gb
        if hardware_info.get('device_type') == 'cpu':
            return self.size_category == ModelSize.TINY
        return False

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'size_category': self.size_category.value,
            'description': self.description,
            'depth': self.depth,
            'aspect_ratio': self.aspect_ratio,
            'head_dim': self.head_dim,
            'use_moe': self.use_moe,
            'use_gqa': self.use_gqa,
            'use_swiglu': self.use_swiglu,
            'recommended_batch_size': self.recommended_batch_size,
            'recommended_seq_len': self.recommended_seq_len,
            'optimizer': self.optimizer,
            'param_count_millions': round(self.param_count_millions, 1),
        }


MODEL_CATALOG = [
    ModelConfig(name="Nano-160K", size_category=ModelSize.TINY, description="Tiny model for testing",
                depth=2, aspect_ratio=32, min_vram_gb=2.0, use_swiglu=False, optimizer="lion"),
    ModelConfig(name="Micro-440K", size_category=ModelSize.TINY, description="Rapid prototyping",
                depth=4, aspect_ratio=32, min_vram_gb=3.0, optimizer="lion"),
    ModelConfig(name="Small-2.5M", size_category=ModelSize.SMALL, description="Consumer hardware",
                depth=4, aspect_ratio=64, min_vram_gb=6.0, use_gqa=True),
    ModelConfig(name="Base-8M", size_category=ModelSize.SMALL, description="Baseline for experiments",
                depth=6, aspect_ratio=64, min_vram_gb=8.0),
    ModelConfig(name="Medium-25M", size_category=ModelSize.MEDIUM, description="Serious experiments",
                depth=8, aspect_ratio=64, min_vram_gb=12.0),
    ModelConfig(name="Medium-MoE-50M", size_category=ModelSize.MEDIUM, description="Mixture of Experts",
                depth=8, aspect_ratio=64, use_moe=True, moe_num_experts=4, min_vram_gb=14.0, use_gqa=True),
    ModelConfig(name="Large-125M", size_category=ModelSize.LARGE, description="Production-quality",
                depth=12, aspect_ratio=96, min_vram_gb=24.0),
    ModelConfig(name="Large-MoE-300M", size_category=ModelSize.LARGE, description="Sparse MoE high-end",
                depth=12, aspect_ratio=96, use_moe=True, moe_num_experts=8, min_vram_gb=24.0, use_gqa=True),
    ModelConfig(name="XL-500M", size_category=ModelSize.XLARGE, description="Research clusters",
                depth=16, aspect_ratio=128, min_vram_gb=40.0, use_gqa=True),
    ModelConfig(name="XL-MoE-1B", size_category=ModelSize.XLARGE, description="Billion-param MoE",
                depth=16, aspect_ratio=128, use_moe=True, moe_num_experts=16, min_vram_gb=80.0, use_gqa=True),
]


def get_compatible_models(hardware_info: Dict[str, Any]) -> List[ModelConfig]:
    return [m for m in MODEL_CATALOG if m.is_compatible(hardware_info)]


def get_model_by_name(name: str) -> Optional[ModelConfig]:
    for model in MODEL_CATALOG:
        if model.name == name:
            return model
    return None
