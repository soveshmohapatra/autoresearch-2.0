"""
Autoresearch pretraining script. Multi-platform support (CUDA/MPS/CPU).
Enhanced with: Architecture variants, Optimizer zoo, W&B tracking, Checkpointing.
Usage: uv run train.py [--depth 8] [--aspect-ratio 64] [--batch-size 64] ...
"""

from __future__ import annotations
import os
import gc
import time
import math
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Command Line Arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Autoresearch training script")
    
    # Model architecture
    parser.add_argument("--depth", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--aspect-ratio", type=int, default=None, help="Aspect ratio for model dim")
    parser.add_argument("--head-dim", type=int, default=None, help="Attention head dimension")
    parser.add_argument("--batch-size", type=int, default=None, help="Device batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer type")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name for logging")
    
    # Architecture variants
    parser.add_argument("--use-moe", action="store_true", help="Enable Mixture of Experts")
    parser.add_argument("--moe-experts", type=int, default=4, help="Number of MoE experts")
    parser.add_argument("--use-gqa", action="store_true", help="Enable Grouped Query Attention")
    parser.add_argument("--use-swiglu", action="store_true", help="Enable SwiGLU activation")
    parser.add_argument("--use-prenorm", action="store_true", help="Enable pre-norm architecture")
    
    # Training
    parser.add_argument("--time-budget", type=int, default=None, help="Training time budget in seconds")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--language", type=str, default="en", help="Language code (en, fr, es, de, hi, zh, ja, gu, nl, or)")

    return parser.parse_args()

# Parse arguments
args = parse_args()

# Import configuration
from config import ExperimentConfig, DEFAULT_CONFIG

# Load base config and override with CLI args
config = ExperimentConfig()

# Apply CLI overrides
if args.depth is not None:
    config.model.depth = args.depth
if args.aspect_ratio is not None:
    config.model.aspect_ratio = args.aspect_ratio
if args.head_dim is not None:
    config.model.head_dim = args.head_dim
if args.batch_size is not None:
    config.training.device_batch_size = args.batch_size
if args.seq_len is not None:
    config.training.max_seq_len = args.seq_len
if args.optimizer is not None:
    config.optimizer.optimizer_type = args.optimizer
if args.time_budget is not None:
    config.training.time_budget = args.time_budget

# Architecture variants from CLI
if args.use_moe:
    config.model.use_moe = True
    config.model.moe_num_experts = args.moe_experts
if args.use_gqa:
    config.model.use_gqa = True
if args.use_swiglu:
    config.model.use_swiglu = True
if args.use_prenorm:
    config.model.use_prenorm = True

# Language-specific directory setup
import json as _json
_LANGUAGE = getattr(args, 'language', 'en')
_lang_cache = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", _LANGUAGE)
LANG_DATA_DIR = os.path.join(_lang_cache, "data")
LANG_TOKENIZER_DIR = os.path.join(_lang_cache, "tokenizer")
LANG_CHECKPOINT_DIR = os.path.join(_lang_cache, "checkpoints")
_meta_path = os.path.join(_lang_cache, "meta.json")
if os.path.exists(_meta_path):
    with open(_meta_path) as _f:
        LANG_VAL_FILENAME = _json.load(_f).get("val_filename")
else:
    LANG_VAL_FILENAME = None  # prepare.py will use its own default

# Override checkpoint dir with language-specific path
config.checkpoint.save_dir = LANG_CHECKPOINT_DIR

# Platform-specific setup
DEVICE = config.device.get_device()
DTYPE = config.device.get_torch_dtype()
PEAK_FLOPS = config.device.get_peak_flops()

print(f"Using device: {DEVICE}")
print(f"Using dtype: {DTYPE}")
print(f"Model config: depth={config.model.depth}, aspect_ratio={config.model.aspect_ratio}")
print(f"Batch size: {config.training.device_batch_size}, seq_len: {config.training.max_seq_len}")
print(f"Architecture: MoE={config.model.use_moe}, GQA={config.model.use_gqa}, SwiGLU={config.model.use_swiglu}")
print(f"Optimizer: {config.optimizer.optimizer_type}")
if args.experiment_name:
    print(f"Experiment: {args.experiment_name}")

# Memory-efficient settings for MPS (target: 8GB VRAM)
if DEVICE == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    config.training.device_batch_size = 4
    config.training.total_batch_size = 2**14
    config.model.depth = 4
    config.model.aspect_ratio = 32
    DTYPE = torch.float32
    print("Configured for 8GB MPS target (small batches, small model)")
elif DEVICE == "cuda":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Try to import Flash Attention for CUDA
FA3_AVAILABLE = False
if DEVICE == "cuda":
    try:
        from kernels import get_kernel
        cap = torch.cuda.get_device_capability()
        repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
        fa3 = get_kernel(repo).flash_attn_interface
        FA3_AVAILABLE = True
        print("Flash Attention 3 available")
    except Exception as e:
        print(f"Flash Attention 3 not available: {e}")

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb, get_token_bytes

# MAX_SEQ_LEN and TIME_BUDGET are set in the AGENT EDIT ZONE below.

# ---------------------------------------------------------------------------
# Architecture Configuration
# ---------------------------------------------------------------------------

@dataclass
class ArchConfig:
    """Architecture variant configuration."""
    # Base architecture
    depth: int = 4
    aspect_ratio: int = 32
    head_dim: int = 128
    window_pattern: str = "SSSL"
    
    # Architecture variants
    use_moe: bool = False  # Mixture of Experts
    moe_num_experts: int = 4
    moe_top_k: int = 2
    
    use_gqa: bool = False  # Grouped Query Attention
    gqa_kv_groups: int = 4
    
    use_swiglu: bool = False  # SwiGLU activation
    use_geglu: bool = False  # GeGLU activation
    use_prenorm: bool = False  # Pre-norm architecture
    use_weight_tying: bool = False  # Tie lm_head weights to wte
    
    # Computed
    @property
    def model_dim(self) -> int:
        base_dim = self.depth * self.aspect_ratio
        return ((base_dim + self.head_dim - 1) // self.head_dim) * self.head_dim
    
    @property
    def num_heads(self) -> int:
        return self.model_dim // self.head_dim
    
    @property
    def num_kv_heads(self) -> int:
        if self.use_gqa:
            return max(1, self.num_heads // self.gqa_kv_groups)
        return self.num_heads


# ARCH_CONFIG built after AGENT EDIT ZONE — see below.

# ---------------------------------------------------------------------------
# GPT Model with Architecture Variants
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x, norm_type: str = "rms"):
    """Normalization with variant support."""
    if norm_type == "layer":
        return F.layer_norm(x, x.shape[-1:])
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class MoELayer(nn.Module):
    """Mixture of Experts layer."""
    
    def __init__(self, n_embd: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd, bias=False),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd, bias=False)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(n_embd, num_experts, bias=False)
    
    def forward(self, x):
        B, T, C = x.shape
        N = B * T
        x_flat = x.reshape(N, C)

        # Gating
        gate_weights = F.softmax(self.gate(x_flat), dim=-1)  # (N, E)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Flatten: each of N tokens has top_k (token, expert, weight) assignments
        token_ids = torch.arange(N, device=x_flat.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        flat_expert_ids = top_k_indices.reshape(-1)   # (N*top_k,)
        flat_weights = top_k_weights.reshape(-1)      # (N*top_k,)

        # Dispatch: run each expert only on its assigned tokens  — O(N*top_k/E per expert)
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            sel = (flat_expert_ids == expert_idx).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            tok = token_ids[sel]                          # token indices → this expert
            x_out = self.experts[expert_idx](x_flat[tok])  # (n_i, C)
            output.index_add_(0, tok, x_out * flat_weights[sel].unsqueeze(-1))

        return output.reshape(B, T, C)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Flash Attention or fallback
        if FA3_AVAILABLE:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
            k = k.transpose(1, 2)  # (B, n_kv_head, T, head_dim)
            v = v.transpose(1, 2)  # (B, n_kv_head, T, head_dim)
            # GQA: expand k/v to match q's head count
            if self.n_kv_head != self.n_head:
                groups = self.n_head // self.n_kv_head
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)
            # Sliding window mask for non-full-context layers
            ws = window_size[0]
            if ws < T:
                # Causal + sliding window: each token only attends to last ws tokens
                positions = torch.arange(T, device=q.device)
                mask = (positions.unsqueeze(0) >= positions.unsqueeze(1) - ws + 1) & \
                       (positions.unsqueeze(0) <= positions.unsqueeze(1))
                attn_bias = torch.zeros(T, T, dtype=q.dtype, device=q.device)
                attn_bias.masked_fill_(~mask, float('-inf'))
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            y = y.transpose(1, 2)
        
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """MLP with activation variant support."""
    def __init__(self, config):
        super().__init__()
        self.use_swiglu = ARCH_CONFIG.use_swiglu
        self.use_geglu = ARCH_CONFIG.use_geglu
        # For gated activations (SwiGLU/GeGLU), c_fc outputs 4*n_embd which is
        # split into gate (2*n_embd) + value (2*n_embd), so c_proj takes 2*n_embd.
        # For standard activations, c_proj takes the full 4*n_embd.
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        proj_in = 2 * config.n_embd if (self.use_swiglu or self.use_geglu) else 4 * config.n_embd
        self.c_proj = nn.Linear(proj_in, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        if self.use_swiglu:
            gate, value = x.chunk(2, dim=-1)
            x = F.silu(gate) * value
        elif self.use_geglu:
            gate, value = x.chunk(2, dim=-1)
            x = F.gelu(gate) * value
        else:
            x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config) if not ARCH_CONFIG.use_moe else None
        self.moe = MoELayer(config.n_embd, ARCH_CONFIG.moe_num_experts, ARCH_CONFIG.moe_top_k) if ARCH_CONFIG.use_moe else None
        self.use_prenorm = ARCH_CONFIG.use_prenorm

    def forward(self, x, ve, cos_sin, window_size):
        if self.use_prenorm:
            # Pre-norm: norm before attention/MLP
            x_norm = norm(x)
            x = x + self.attn(x_norm, ve, cos_sin, window_size)
            if self.mlp:
                x = x + self.mlp(norm(x))
            elif self.moe:
                x = x + self.moe(norm(x))
        else:
            # Post-norm (default)
            x = x + self.attn(norm(x), ve, cos_sin, window_size)
            if self.mlp:
                x = x + self.mlp(norm(x))
            elif self.moe:
                x = x + self.moe(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s) if block.mlp else None
            torch.nn.init.zeros_(block.mlp.c_proj.weight) if block.mlp else None
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=DTYPE)
        for ve in self.value_embeds.values():
            ve.to(dtype=DTYPE)
        # Weight tying: share lm_head and wte weights
        if ARCH_CONFIG.use_weight_tying:
            self.lm_head.weight = self.transformer.wte.weight

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(DTYPE), sin.to(DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                        optimizer_type: str = "muon_adamw"):
        """Setup optimizer with variant support."""
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        if optimizer_type == "lion":
            # Lion optimizer for all params
            param_groups = [dict(
                kind='lion', params=list(self.parameters()),
                lr=matrix_lr * dmodel_lr_scale, betas=adam_betas, weight_decay=weight_decay
            )]
            optimizer = LionAdamW(param_groups)
        elif optimizer_type == "adafactor":
            # Adafactor
            optimizer = Adafactor(self.parameters(), lr=matrix_lr * dmodel_lr_scale)
        else:
            # Default: Muon + AdamW
            param_groups = [
                dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            ]
            for shape in sorted({p.shape for p in matrix_params}):
                group_params = [p for p in matrix_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))
            optimizer = MuonAdamW(param_groups)
        
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer Zoo (MuonAdamW, Lion, Adafactor)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16() if g.device.type == 'cuda' else g.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)


class LionAdamW(torch.optim.Optimizer):
    """Lion optimizer - memory efficient, sign-based optimizer."""
    
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={'betas': (0.9, 0.99)})
        self._step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group.get('betas', (0.9, 0.99))[0]
            beta2 = group.get('betas', (0.9, 0.99))[1]
            wd = group.get('weight_decay', 0.0)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                exp_avg.lerp_(grad, 1 - beta2)
                
                # Update = sign(beta1 * exp_avg + (1 - beta1) * grad)
                update = (beta1 * exp_avg + (1 - beta1) * grad).sign()
                
                # Weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)
                
                p.add_(update, alpha=-lr)


class Adafactor(torch.optim.Optimizer):
    """Adafactor optimizer - adaptive LR without momentum."""
    
    def __init__(self, params, lr=0.001, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, beta1=None, weight_decay=0.0):
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                       decay_rate=decay_rate, beta1=beta1, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                
                if not state:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if group['beta1'] is not None:
                        state['exp_avg'] = torch.zeros_like(p)
                
                state['step'] += 1
                beta2 = 1.0 - (state['step'] ** group['decay_rate'])
                
                # RMS normalization
                rms = grad.pow(2).mean().sqrt() + group['eps'][1]
                
                # Adaptive LR
                alpha = group['lr'] * max(1e-3, rms) / group['clip_threshold']
                
                # Update exp_avg_sq
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute update
                update = grad / (state['exp_avg_sq'].sqrt() + group['eps'][0])
                
                if group['beta1'] is not None:
                    state['exp_avg'].lerp_(grad, 1 - group['beta1'])
                    update = state['exp_avg']
                
                # Weight decay
                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                p.add_(update, alpha=-alpha)

# ===========================================================================
# AGENT EDIT ZONE
# This is the only section you need to modify to run experiments.
# Edit these constants directly — changes take effect immediately on next run.
# ===========================================================================

# --- Model architecture ---
DEPTH         = 8           # number of transformer layers
ASPECT_RATIO  = 64          # model_dim = depth * aspect_ratio (must give model_dim % head_dim == 0)
HEAD_DIM      = 128         # attention head dimension (keep at 128)
WINDOW_PATTERN = "SSSL"     # attention pattern: L=full context, S=half context

# --- Architecture variants (True/False to toggle) ---
USE_MOE       = False       # Mixture of Experts (sparse FFN)
MOE_NUM_EXPERTS = 4         # total experts (if USE_MOE)
MOE_TOP_K     = 2           # experts activated per token (if USE_MOE)

USE_GQA       = False       # Grouped Query Attention (fewer KV heads)
GQA_KV_GROUPS = 4           # divide num_heads by this for KV heads (if USE_GQA)

USE_SWIGLU    = False       # SwiGLU activation (replaces ReLU²)
USE_GEGLU     = False       # GeGLU activation (replaces ReLU²; pick at most one gated)
USE_PRENORM   = False       # Pre-norm residual stream (default: post-norm)
USE_WEIGHT_TYING = False    # Tie lm_head weights to wte (reduces params, often helps)

# --- Optimizer ---
OPTIMIZER_TYPE    = "muon_adamw"   # "muon_adamw" | "lion" | "adafactor"
EMBEDDING_LR      = 0.6
UNEMBEDDING_LR    = 0.004
MATRIX_LR         = 0.04
SCALAR_LR         = 0.5
WEIGHT_DECAY      = 0.2
ADAM_BETAS        = (0.8, 0.95)
WARMUP_RATIO      = 0.0            # fraction of budget for LR warmup
WARMDOWN_RATIO    = 0.5            # fraction of budget for LR cooldown
FINAL_LR_FRAC     = 0.0            # final LR as fraction of peak

# --- Training ---
TIME_BUDGET       = 300            # seconds of training (wall clock, excl. startup)
TOTAL_BATCH_SIZE  = 2**19          # ~524K tokens per optimizer step
DEVICE_BATCH_SIZE = 128            # per-device batch size (reduce if OOM)
MAX_SEQ_LEN       = MAX_SEQ_LEN    # inherits from prepare.py (2048); override here if needed
GRAD_CLIP         = 1.0            # gradient clipping norm (0.0 = disabled)

# ===========================================================================
# END AGENT EDIT ZONE
# ===========================================================================

# CLI overrides — only apply when the flag is explicitly passed (used by gui.py / run_loop.py)
if args.depth is not None:        DEPTH = args.depth
if args.aspect_ratio is not None: ASPECT_RATIO = args.aspect_ratio
if args.head_dim is not None:     HEAD_DIM = args.head_dim
if args.batch_size is not None:   DEVICE_BATCH_SIZE = args.batch_size
if args.seq_len is not None:      MAX_SEQ_LEN = args.seq_len
if args.optimizer is not None:    OPTIMIZER_TYPE = args.optimizer
if args.time_budget is not None:  TIME_BUDGET = args.time_budget
if args.use_moe:                  USE_MOE = True; MOE_NUM_EXPERTS = args.moe_experts
if args.use_gqa:                  USE_GQA = True
if args.use_swiglu:               USE_SWIGLU = True
if args.use_prenorm:              USE_PRENORM = True

# MPS caps — hardware memory limits on Apple Silicon
if DEVICE == "mps":
    DEPTH             = min(DEPTH, 4)
    ASPECT_RATIO      = min(ASPECT_RATIO, 32)
    DEVICE_BATCH_SIZE = min(DEVICE_BATCH_SIZE, 4)
    TOTAL_BATCH_SIZE  = min(TOTAL_BATCH_SIZE, 2**14)
    MAX_SEQ_LEN       = min(MAX_SEQ_LEN, 512)
    print(f"MPS: capped to depth={DEPTH}, aspect={ASPECT_RATIO}, batch={DEVICE_BATCH_SIZE}, seqlen={MAX_SEQ_LEN}")

# Build ARCH_CONFIG from resolved constants
ARCH_CONFIG = ArchConfig(
    depth=DEPTH, aspect_ratio=ASPECT_RATIO, head_dim=HEAD_DIM, window_pattern=WINDOW_PATTERN,
    use_moe=USE_MOE, moe_num_experts=MOE_NUM_EXPERTS, moe_top_k=MOE_TOP_K,
    use_gqa=USE_GQA, gqa_kv_groups=GQA_KV_GROUPS,
    use_swiglu=USE_SWIGLU, use_geglu=USE_GEGLU, use_prenorm=USE_PRENORM,
    use_weight_tying=USE_WEIGHT_TYING,
)

# ---------------------------------------------------------------------------
# W&B Tracking & Checkpointing
# ---------------------------------------------------------------------------

class WandbTracker:
    """Weights & Biases experiment tracker."""
    
    def __init__(self, enabled=False, project="autoresearch", entity=None, tags=None):
        self.enabled = enabled and not os.environ.get("WANDB_DISABLED", "false").lower() == "true"
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.run = None
        
        if self.enabled:
            try:
                import wandb
                self.run = wandb.init(
                    project=self.project,
                    entity=self.entity,
                    tags=self.tags,
                    config=config.to_dict(),
                    reinit=True
                )
                print(f"W&B tracking enabled: {self.run.get_url()}")
            except Exception as e:
                print(f"W&B init failed: {e}")
                self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics to W&B."""
        if self.enabled and self.run:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except:
                pass
    
    def finish(self):
        """Finish W&B run."""
        if self.enabled and self.run:
            try:
                import wandb
                wandb.finish()
            except:
                pass


class CheckpointManager:
    """Manages model checkpointing."""
    
    def __init__(self, enabled=True, save_dir="./checkpoints", save_interval=60, keep_last=3):
        self.enabled = enabled
        self.save_dir = Path(save_dir)
        self.save_interval = save_interval
        self.keep_last = keep_last
        self.checkpoints = []
        self.last_save_time = time.time()
        
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, model, optimizer, step, val_bpb, config_snapshot, force=False):
        """Save checkpoint."""
        if not self.enabled:
            return

        current_time = time.time()
        if not force and current_time - self.last_save_time < self.save_interval:
            return
        
        self.last_save_time = current_time
        
        checkpoint = {
            'step': step,
            'val_bpb': val_bpb,
            'config': config_snapshot,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': time.time(),
        }
        
        filename = f"checkpoint_step{step:06d}_bpb{val_bpb:.6f}.pt"
        filepath = self.save_dir / filename
        
        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)
        
        # Remove old checkpoints
        while len(self.checkpoints) > self.keep_last:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        print(f"Saved checkpoint: {filename}")
    
    def load_latest(self, model, optimizer):
        """Load latest checkpoint. Returns checkpoint dict or None on failure."""
        if not self.enabled or not self.save_dir.exists():
            return None

        checkpoints = list(self.save_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            print("Resume: no checkpoint found, starting from scratch.")
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        try:
            checkpoint = torch.load(latest, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Resumed from checkpoint: {latest.name} (step {checkpoint.get('step', '?')}, bpb {checkpoint.get('val_bpb', '?')})")
            return checkpoint
        except Exception as e:
            print(f"Resume: failed to load {latest.name} ({e}), starting from scratch.")
            return None


# Initialize tracking and checkpointing
wandb = WandbTracker(
    enabled=config.wandb.enabled,
    project=config.wandb.project,
    entity=config.wandb.entity,
    tags=config.wandb.tags,
)

checkpoint_mgr = CheckpointManager(
    enabled=config.checkpoint.enabled,
    save_dir=config.checkpoint.save_dir,
    save_interval=config.checkpoint.save_interval,
    keep_last=config.checkpoint.keep_last,
)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if DEVICE == "cuda":
    torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

if DEVICE == "cuda":
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=DTYPE)
elif DEVICE == "mps":
    autocast_ctx = torch.autocast(device_type="mps", dtype=torch.bfloat16)
else:
    autocast_ctx = torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)

tokenizer = Tokenizer.from_directory(LANG_TOKENIZER_DIR)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    num_kv_heads = max(1, num_heads // ARCH_CONFIG.gqa_kv_groups) if ARCH_CONFIG.use_gqa else num_heads
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

gpt_config = build_model_config(DEPTH)
print(f"Model config: {asdict(gpt_config)}")
print(f"Architecture variants: MoE={ARCH_CONFIG.use_moe}, GQA={ARCH_CONFIG.use_gqa}, SwiGLU={ARCH_CONFIG.use_swiglu}, PreNorm={ARCH_CONFIG.use_prenorm}")
print(f"Optimizer: {OPTIMIZER_TYPE}")

with torch.device("meta"):
    model = GPT(gpt_config)
model.to_empty(device=DEVICE)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
    optimizer_type=OPTIMIZER_TYPE,
)

if args.resume:
    checkpoint_mgr.load_latest(model, optimizer)

if DEVICE == "cuda":
    model = torch.compile(model, dynamic=False)
    print("Model compiled with torch.compile (CUDA)")
elif DEVICE == "mps":
    # torch.compile works on MPS with PyTorch 2.3+ / macOS 14.4+
    _mac_ver = tuple(int(x) for x in __import__("platform").mac_ver()[0].split(".")[:2])
    if torch.__version__ >= "2.3" and _mac_ver >= (14, 4):
        try:
            model = torch.compile(model, dynamic=False)
            print("Model compiled with torch.compile (MPS)")
        except Exception as e:
            print(f"torch.compile on MPS failed ({e}), running eager")
    else:
        print("Skipping torch.compile on MPS (needs PyTorch>=2.3 + macOS>=14.4)")
elif DEVICE == "cpu":
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"CPU threads: {os.cpu_count()}")

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", device=DEVICE, data_dir=LANG_DATA_DIR, val_filename=LANG_VAL_FILENAME)
x, y, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
best_val_bpb = float('inf')
last_periodic_eval = 0.0  # wall time of last mid-training eval
# Periodic eval fires every 20% of budget, but only when budget is long enough
# (eval takes ~100s on MPS, so skip for budgets under 5min)
PERIODIC_EVAL_INTERVAL = TIME_BUDGET / 5
PERIODIC_EVAL_ENABLED = TIME_BUDGET >= 300

# Loss curve side-channel for live terminal display
LOSS_CURVE_FILE = Path("loss_curve.jsonl")
LOSS_CURVE_FILE.write_text("")  # clear from previous run

print("\nStarting training...")

while True:
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] in ['muon', 'lion']:
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    # Gradient clipping (before optimizer step)
    grad_norm = 0.0
    if GRAD_CLIP > 0.0:
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
    else:
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail
    if train_loss_f > 100:
        print("\nFAIL: Loss exploding!")
        exit(1)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | gnorm: {grad_norm:.2f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # Write to loss curve side-channel every 10 steps
    if step % 10 == 0:
        with open(LOSS_CURVE_FILE, "a") as _lf:
            _lf.write(json.dumps({"step": step, "loss": round(debiased_smooth_loss, 6), "progress": round(progress, 4), "remaining": round(remaining, 1)}) + "\n")

    # W&B logging
    if step % 10 == 0:
        wandb.log({
            "train_loss": debiased_smooth_loss,
            "grad_norm": grad_norm,
            "lr_multiplier": lrm,
            "tokens_per_sec": tok_per_sec,
            "mfu_percent": mfu,
            "step": step,
        }, step=step)

    # Periodic mid-training eval (every ~20% of time budget, only for long runs)
    if (PERIODIC_EVAL_ENABLED and step > 10
            and total_training_time - last_periodic_eval >= PERIODIC_EVAL_INTERVAL
            and total_training_time < TIME_BUDGET * 0.9):
        print()
        print(f"[periodic eval @ {total_training_time:.0f}s / {TIME_BUDGET}s]")
        model.eval()
        with torch.no_grad():
            mid_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, device=DEVICE, seq_len=MAX_SEQ_LEN, data_dir=LANG_DATA_DIR, val_filename=LANG_VAL_FILENAME)
        model.train()
        print(f"  mid val_bpb: {mid_bpb:.6f}  (best so far: {min(best_val_bpb, mid_bpb):.6f})")
        if mid_bpb < best_val_bpb:
            best_val_bpb = mid_bpb
        wandb.log({"mid_val_bpb": mid_bpb}, step=step)
        last_periodic_eval = total_training_time

    # Checkpointing
    checkpoint_mgr.save(model, optimizer, step, best_val_bpb, config.to_dict())

    # GC management (freeze on first step to avoid ~500ms GC stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % config.training.gc_interval == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval — cap steps for short time budgets to avoid hanging
_eval_max_steps = None if TIME_BUDGET >= 300 else 100
print(f"Running final evaluation{f' (capped at {_eval_max_steps} steps)' if _eval_max_steps else ''}...")
model.eval()
with torch.no_grad():
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, device=DEVICE, seq_len=MAX_SEQ_LEN, max_steps=_eval_max_steps, data_dir=LANG_DATA_DIR, val_filename=LANG_VAL_FILENAME)

# Log final results
wandb.log({
    "val_bpb": val_bpb,
    "best_val_bpb": min(best_val_bpb, val_bpb),
}, step=step)

# Update best and always force-save so the next cycle can warm-start
if val_bpb < best_val_bpb:
    best_val_bpb = val_bpb
checkpoint_mgr.save(model, optimizer, step, best_val_bpb, config.to_dict(), force=True)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / PEAK_FLOPS if total_training_time > 0 else 0

if DEVICE == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
elif DEVICE == "mps":
    peak_vram_mb = torch.mps.current_allocated_memory() / 1024 / 1024 if hasattr(torch.mps, 'current_allocated_memory') else 0
else:
    peak_vram_mb = 0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"best_val_bpb:     {best_val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"device:           {DEVICE}")
print(f"dtype:            {DTYPE}")
print(f"arch_variants:    MoE={ARCH_CONFIG.use_moe}, GQA={ARCH_CONFIG.use_gqa}, SwiGLU={ARCH_CONFIG.use_swiglu}")
print(f"optimizer:        {OPTIMIZER_TYPE}")

# Finish W&B
wandb.finish()
