"""Tests for configuration system."""

import pytest
import torch
import sys
sys.path.insert(0, '..')

from config import ExperimentConfig, DeviceConfig, ModelConfig


class TestDeviceConfig:
    """Test DeviceConfig."""
    
    def test_auto_detect(self):
        """Test auto device detection."""
        device_config = DeviceConfig()
        device = device_config.get_device()
        assert device in ["auto", "cuda", "mps", "cpu"]
    
    def test_explicit_device(self):
        """Test explicit device setting."""
        device_config = DeviceConfig(device="cpu")
        assert device_config.get_device() == "cpu"
    
    def test_dtype(self):
        """Test dtype retrieval."""
        device_config = DeviceConfig()
        dtype = device_config.get_torch_dtype()
        assert dtype in [torch.float32, torch.float16, torch.bfloat16]


class TestModelConfig:
    """Test ModelConfig."""
    
    def test_model_dim_calculation(self):
        """Test model dimension calculation."""
        model_config = ModelConfig(depth=4, aspect_ratio=32, head_dim=128)
        assert model_config.model_dim == 4 * 32  # 128, aligned to head_dim
    
    def test_num_heads_calculation(self):
        """Test number of heads calculation."""
        model_config = ModelConfig(depth=4, aspect_ratio=32, head_dim=128)
        assert model_config.num_heads == model_config.model_dim // 128
    
    def test_gpt_config_conversion(self):
        """Test conversion to GPTConfig dict."""
        model_config = ModelConfig()
        gpt_config = model_config.to_gpt_config(vocab_size=8192, seq_len=2048)
        assert gpt_config["vocab_size"] == 8192
        assert gpt_config["sequence_len"] == 2048
        assert gpt_config["n_layer"] == model_config.depth


class TestExperimentConfig:
    """Test ExperimentConfig."""
    
    def test_default_creation(self):
        """Test default config creation."""
        config = ExperimentConfig()
        assert config.model.depth == 8
        assert config.training.time_budget == 300
    
    def test_serialization(self, tmp_path):
        """Test config serialization."""
        config = ExperimentConfig()
        config_path = tmp_path / "config.json"
        config.save(str(config_path))
        
        loaded = ExperimentConfig.load(str(config_path))
        # loaded is ExperimentConfig object
        assert loaded.model.depth == config.model.depth
        assert loaded.training.time_budget == config.training.time_budget
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ExperimentConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "model" in config_dict
        assert "training" in config_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
