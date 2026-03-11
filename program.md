# autoresearch 2.0

Autonomous LLM research. You run experiments, record results, and iterate forever.

---

## Setup

1. **Agree on a run tag** (e.g. `mar10`). Create branch: `git checkout -b autoresearch/<tag>`.
2. **Read these files** — all small, read them fully:
   - `prepare.py` — fixed constants, data, tokenizer, evaluation. **Do not modify.**
   - `train.py` — the only file you edit.
3. **Verify data**: `~/.cache/autoresearch/` must exist. If not: `uv run prepare.py`.
4. **Check hardware**: `uv run python gui.py --detect` — shows device tier and recommended config.
5. **Initialize results**: `echo -e "commit\tval_bpb\tmemory_gb\tstatus\tdescription" > results.tsv`
6. **Go**: first run is always the baseline (no changes).

---

## The only file you edit: `train.py`

Find the **`AGENT EDIT ZONE`** section. It is the only part you need to change.

```python
# ===========================================================================
# AGENT EDIT ZONE
# ===========================================================================

# --- Model architecture ---
DEPTH         = 8
ASPECT_RATIO  = 64          # model_dim = depth * aspect_ratio
HEAD_DIM      = 128
WINDOW_PATTERN = "SSSL"     # L=full context, S=half context

# --- Architecture variants ---
USE_MOE       = False       # Mixture of Experts
MOE_NUM_EXPERTS = 4
MOE_TOP_K     = 2

USE_GQA       = False       # Grouped Query Attention
GQA_KV_GROUPS = 4

USE_SWIGLU    = False       # SwiGLU activation (vs ReLU²)
USE_GEGLU     = False       # GeGLU activation
USE_PRENORM   = False       # Pre-norm residual stream

# --- Optimizer ---
OPTIMIZER_TYPE    = "muon_adamw"    # "muon_adamw" | "lion" | "adafactor"
EMBEDDING_LR      = 0.6
UNEMBEDDING_LR    = 0.004
MATRIX_LR         = 0.04
SCALAR_LR         = 0.5
WEIGHT_DECAY      = 0.2
ADAM_BETAS        = (0.8, 0.95)
WARMUP_RATIO      = 0.0
WARMDOWN_RATIO    = 0.5
FINAL_LR_FRAC     = 0.0

# --- Training ---
TIME_BUDGET       = 300            # seconds (wall clock, excl. startup)
TOTAL_BATCH_SIZE  = 2**19          # ~524K tokens per step
DEVICE_BATCH_SIZE = 128            # per-device batch (reduce if OOM)
MAX_SEQ_LEN       = MAX_SEQ_LEN    # override with e.g. 1024
```

**On Apple Silicon (MPS)**: script auto-caps `DEPTH ≤ 4`, `ASPECT_RATIO ≤ 32`, `DEVICE_BATCH_SIZE ≤ 4`, `MAX_SEQ_LEN ≤ 512`. Set lower values here if still OOM.

---

## Goal

**Lowest `val_bpb`** (bits per byte) wins. The time budget is fixed at 5 minutes of training (excluding startup/compilation). Change anything in the AGENT EDIT ZONE.

**Simplicity rule**: 0.001 improvement from 20 lines of complex code? Not worth it. 0.001 improvement by deleting code? Keep. Equal val_bpb, simpler code? Keep.

---

## Experiment loop

**LOOP FOREVER:**

1. **Recall history**: `tail -20 results.tsv`
2. **Form hypothesis**: what is the single most promising change?
3. **Edit `train.py`**: change ONE thing in the AGENT EDIT ZONE
4. **Commit**: `git add train.py && git commit -m "experiment: <description>"`
5. **Run**: `uv run train.py > run.log 2>&1`
6. **Check**: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
   - Empty → crash. `tail -50 run.log` to debug.
7. **Record** in `results.tsv` (tabs, NOT commas):
   ```
   <7-char-commit>	<val_bpb>	<vram_gb>	<status>	<description>
   ```
   Status: `keep`, `discard`, or `crash`. Use `0.000000` / `0.0` for crashes.
8. **Branch**:
   - **keep** (val_bpb improved) → stay. Advance branch.
   - **discard** (equal or worse) → `git reset --hard HEAD~1`. Try something else.

---

## Output format

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
device:           cuda
optimizer:        muon_adamw
arch_variants:    MoE=False, GQA=False, SwiGLU=False
```

---

## Architecture variants — what they do

| Constant | Effect | When to try |
|----------|--------|-------------|
| `USE_SWIGLU = True` | SwiGLU replaces ReLU² in MLP | **Early — high value, low risk** |
| `USE_PRENORM = True` | Norm before attn/MLP | Try at DEPTH ≥ 8 |
| `USE_GQA = True` | Fewer KV heads → fits larger model | When OOM trying bigger model |
| `USE_MOE = True` | Sparse MoE FFN, more params, same compute | H100/A100 only |
| `WINDOW_PATTERN = "LLLL"` | Full attention every layer | If "SSSL" plateaus |
| `WINDOW_PATTERN = "SSSS"` | All sliding-window | More efficient |

**GQA**: `GQA_KV_GROUPS = 4` means KV heads = num_heads / 4. Lets you fit a bigger DEPTH or ASPECT_RATIO for the same VRAM.

**MoE**: More parameters, same per-token compute. Helps on large GPUs. `MOE_TOP_K = 2` means each token activates 2 of `MOE_NUM_EXPERTS` experts.

**SwiGLU note**: The MLP expansion factor stays at 4× but splits into gate + value (2× each), so the effective hidden dim is halved. This is intentional — SwiGLU benefits outweigh the reduced hidden size at typical scales.

---

## Strategy by stage

**Runs 1–10** (explore):
- Baseline first (required)
- `USE_SWIGLU = True` — easy win, try second
- LR sweep: `MATRIX_LR = 0.02` then `MATRIX_LR = 0.06`
- Depth: ±2 layers
- `WARMDOWN_RATIO = 0.3` vs `0.7`

**Runs 10–40** (exploit):
- Refine the winning direction (smaller LR steps)
- `USE_PRENORM = True` if DEPTH ≥ 8
- `USE_GQA = True` to fit a bigger model in same VRAM
- Attention patterns: `"LLLL"`, `"SLLL"`, `"SSLL"`
- Batch size: `TOTAL_BATCH_SIZE = 2**18` or `2**20`

**Runs 40+** (fine-tune + radical):
- Combine successful single changes
- `OPTIMIZER_TYPE = "lion"` (memory efficient)
- `USE_MOE = True` on capable GPU
- Reset to best commit + try orthogonal direction

---

## Multi-platform

Auto-detected at startup (first lines of `run.log`):

| Platform | `torch.compile` | Flash Attn 3 | Recommended DEVICE_BATCH_SIZE |
|----------|----------------|--------------|-------------------------------|
| CUDA H100 | Yes | Yes (Hopper) | 128–256 |
| CUDA A100 | Yes | Yes | 64–128 |
| CUDA RTX 4090 | Yes | Yes | 32–64 |
| Apple M-Max/Ultra | No | No | 16–32 |
| Apple M-Base | No | No | 4 (auto-capped) |
| CPU | No | No | 4 |

---

## Never stop

Once the loop begins, do **NOT** pause to ask if you should continue. The human may be asleep.

If stuck: look at `results.tsv` for patterns, combine two near-misses, reset to best commit and try a completely different axis. The loop runs until the human stops it.
