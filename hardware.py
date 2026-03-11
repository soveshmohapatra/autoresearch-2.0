"""
Hardware detection and auto-configuration for autoresearch.
Automatically detects GPU type and optimizes configuration.
"""

import platform
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class DeviceType(Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass
class HardwareInfo:
    """Detected hardware information."""
    device_type: DeviceType
    device_name: str
    total_memory_gb: float
    recommended_batch_size: int
    recommended_depth: int
    recommended_seq_len: int
    recommended_dtype: str
    peak_flops: float
    is_high_end: bool
    
    def to_dict(self) -> dict:
        return {
            "device_type": self.device_type.value,
            "device_name": self.device_name,
            "total_memory_gb": self.total_memory_gb,
            "recommended_batch_size": self.recommended_batch_size,
            "recommended_depth": self.recommended_depth,
            "recommended_seq_len": self.recommended_seq_len,
            "recommended_dtype": self.recommended_dtype,
            "is_high_end": self.is_high_end,
        }


def get_nvidia_gpu_info() -> Optional[Dict[str, Any]]:
    """Get NVIDIA GPU information via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) == 2:
                    gpu_info.append({
                        "name": parts[0].strip(),
                        "memory_gb": float(parts[1].strip()) / 1024
                    })
            return gpu_info[0] if gpu_info else None
    except:
        pass
    return None


def get_apple_gpu_info() -> Optional[Dict[str, Any]]:
    """Get Apple Silicon GPU information."""
    if platform.system() != "Darwin":
        return None
    
    try:
        # Get unified memory info
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            total_memory_gb = int(result.stdout.strip()) / (1024**3)
            
            # Detect chip type
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10
            )
            chip_name = "Apple Silicon"
            if result.returncode == 0:
                chip_name = result.stdout.strip()
            
            # Estimate GPU memory (typically 50-75% of unified memory available)
            gpu_memory_gb = total_memory_gb * 0.6
            
            return {
                "name": chip_name,
                "memory_gb": gpu_memory_gb,
                "total_system_memory_gb": total_memory_gb
            }
    except:
        pass
    return None


def detect_hardware() -> HardwareInfo:
    """
    Detect available hardware and return optimized configuration.
    """
    import torch
    
    # Check for CUDA
    if torch.cuda.is_available():
        nvidia_info = get_nvidia_gpu_info()
        if nvidia_info:
            gpu_name = nvidia_info["name"]
            memory_gb = nvidia_info["memory_gb"]
            
            # Classify GPU tier
            is_high_end = memory_gb >= 40  # A100, H100, RTX 4090
            
            # Recommend configuration based on VRAM
            if memory_gb >= 80:  # A100 80GB, H100
                return HardwareInfo(
                    device_type=DeviceType.CUDA,
                    device_name=gpu_name,
                    total_memory_gb=memory_gb,
                    recommended_batch_size=256,
                    recommended_depth=12,
                    recommended_seq_len=4096,
                    recommended_dtype="bfloat16",
                    peak_flops=989.5e12,
                    is_high_end=True
                )
            elif memory_gb >= 40:  # A100 40GB, RTX 4090
                return HardwareInfo(
                    device_type=DeviceType.CUDA,
                    device_name=gpu_name,
                    total_memory_gb=memory_gb,
                    recommended_batch_size=128,
                    recommended_depth=10,
                    recommended_seq_len=2048,
                    recommended_dtype="bfloat16",
                    peak_flops=312e12,
                    is_high_end=True
                )
            elif memory_gb >= 24:  # RTX 3090/4080
                return HardwareInfo(
                    device_type=DeviceType.CUDA,
                    device_name=gpu_name,
                    total_memory_gb=memory_gb,
                    recommended_batch_size=64,
                    recommended_depth=8,
                    recommended_seq_len=2048,
                    recommended_dtype="float16",
                    peak_flops=156e12,
                    is_high_end=False
                )
            elif memory_gb >= 16:  # RTX 3080/4070
                return HardwareInfo(
                    device_type=DeviceType.CUDA,
                    device_name=gpu_name,
                    total_memory_gb=memory_gb,
                    recommended_batch_size=32,
                    recommended_depth=6,
                    recommended_seq_len=1024,
                    recommended_dtype="float16",
                    peak_flops=98e12,
                    is_high_end=False
                )
            else:  # < 16GB
                return HardwareInfo(
                    device_type=DeviceType.CUDA,
                    device_name=gpu_name,
                    total_memory_gb=memory_gb,
                    recommended_batch_size=16,
                    recommended_depth=4,
                    recommended_seq_len=512,
                    recommended_dtype="float32",
                    peak_flops=50e12,
                    is_high_end=False
                )
    
    # Check for Apple Silicon (MPS)
    elif torch.backends.mps.is_available():
        apple_info = get_apple_gpu_info()
        if apple_info:
            gpu_name = apple_info["name"]
            memory_gb = apple_info["memory_gb"]
            total_memory = apple_info.get("total_system_memory_gb", memory_gb)
            
            # M-series classification
            is_high_end = "Max" in gpu_name or "Ultra" in gpu_name
            
            if total_memory >= 64:  # M1/M2/M3 Max or Ultra (~10-27 TFLOPS)
                return HardwareInfo(
                    device_type=DeviceType.MPS,
                    device_name=gpu_name,
                    total_memory_gb=memory_gb,
                    recommended_batch_size=32,
                    recommended_depth=8,
                    recommended_seq_len=2048,
                    recommended_dtype="float32",
                    peak_flops=10e12,
                    is_high_end=is_high_end
                )
            elif total_memory >= 24:  # M1/M2/M3 Pro (~5-7 TFLOPS)
                return HardwareInfo(
                    device_type=DeviceType.MPS,
                    device_name=gpu_name,
                    total_memory_gb=memory_gb,
                    recommended_batch_size=16,
                    recommended_depth=6,
                    recommended_seq_len=1024,
                    recommended_dtype="float32",
                    peak_flops=6e12,
                    is_high_end=is_high_end
                )
            else:  # M1/M2/M3 base (8-16GB, ~2.6-3.6 TFLOPS)
                return HardwareInfo(
                    device_type=DeviceType.MPS,
                    device_name=gpu_name,
                    total_memory_gb=memory_gb,
                    recommended_batch_size=4,
                    recommended_depth=4,
                    recommended_seq_len=512,
                    recommended_dtype="float32",
                    peak_flops=3e12,
                    is_high_end=is_high_end
                )
    
    # Fallback to CPU
    cpu_count = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.thread_count"] if platform.system() == "Darwin"
        else ["nproc"],
        capture_output=True, text=True, timeout=10
    )
    cpu_cores = int(cpu_count.stdout.strip()) if cpu_count.returncode == 0 else 4
    
    return HardwareInfo(
        device_type=DeviceType.CPU,
        device_name=f"CPU ({cpu_cores} threads)",
        total_memory_gb=0,
        recommended_batch_size=4,
        recommended_depth=4,
        recommended_seq_len=256,
        recommended_dtype="float32",
        peak_flops=50e9,
        is_high_end=False
    )


def generate_config_for_hardware(hardware: HardwareInfo) -> Dict[str, Any]:
    """Generate optimal configuration based on detected hardware."""
    
    # Calculate batch sizes
    total_batch_map = {
        4: 2**12,      # 4K tokens
        16: 2**14,     # 16K tokens
        32: 2**15,     # 32K tokens
        64: 2**17,     # 131K tokens
        128: 2**18,    # 262K tokens
        256: 2**19,    # 524K tokens
    }
    
    total_batch = total_batch_map.get(hardware.recommended_batch_size, 2**14)
    
    return {
        # Device settings
        "device": hardware.device_type.value,
        "dtype": hardware.recommended_dtype,
        
        # Model architecture
        "depth": hardware.recommended_depth,
        "aspect_ratio": 64 if hardware.is_high_end else 32,
        "head_dim": 128,
        
        # Training
        "device_batch_size": hardware.recommended_batch_size,
        "total_batch_size": total_batch,
        "max_seq_len": hardware.recommended_seq_len,
        
        # Optimization
        "matrix_lr": 0.04 if hardware.is_high_end else 0.02,
        "weight_decay": 0.2,
        
        # Architecture features (enable on high-end)
        "use_moe": hardware.is_high_end and hardware.device_type == DeviceType.CUDA,
        "use_gqa": hardware.is_high_end,
        "use_swiglu": True,  # Generally beneficial
        "use_prenorm": hardware.recommended_depth >= 8,
        
        # Optimizer
        "optimizer_type": "muon_adamw" if hardware.is_high_end else "lion",
        
        # Features
        "enable_wandb": False,  # User can enable
        "enable_checkpointing": True,
        
        # Hardware info for display
        "hardware_info": hardware.to_dict(),
    }


def print_hardware_report(hardware: HardwareInfo) -> None:
    """Print a nice hardware detection report."""
    print("\n" + "="*60)
    print("🔍 HARDWARE DETECTION REPORT")
    print("="*60)
    print(f"Device Type:     {hardware.device_type.value.upper()}")
    print(f"Device Name:     {hardware.device_name}")
    print(f"Memory:          {hardware.total_memory_gb:.1f} GB")
    print(f"Performance Tier: {'HIGH-END' if hardware.is_high_end else 'STANDARD'}")
    print()
    print("📋 RECOMMENDED CONFIGURATION")
    print("-"*60)
    print(f"Model Depth:     {hardware.recommended_depth} layers")
    print(f"Batch Size:      {hardware.recommended_batch_size}")
    print(f"Sequence Length: {hardware.recommended_seq_len}")
    print(f"Precision:       {hardware.recommended_dtype}")
    print(f"Peak FLOPS:      {hardware.peak_flops/1e12:.1f} TFLOPS")
    print("="*60 + "\n")


if __name__ == "__main__":
    hardware = detect_hardware()
    print_hardware_report(hardware)
    config = generate_config_for_hardware(hardware)
    print("Generated config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
