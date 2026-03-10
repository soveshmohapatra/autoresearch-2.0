# autoresearch - Multi-Agent System

This is an experiment to have LLM agents do autonomous AI research.

## System Overview

The autoresearch system uses **specialized AI agents** that collaborate to improve a language model through iterative experimentation. Each agent has a specific role and expertise.

### Agent Roles

1. **Architecture Agent** - Model architecture (depth, width, attention, activations)
2. **Optimizer Agent** - Optimization (learning rates, schedules, optimizer hyperparams)
3. **Hyperparameter Agent** - Training hyperparameters (batch size, sequence length)
4. **Analyst Agent** - Results analysis, keep/discard decisions, insights

### Experiment Memory

All experiments are recorded in `experiment_memory.json`. The system learns from history:
- Tracks what types of changes succeed/fail
- Generates hypotheses based on patterns
- Provides context-aware suggestions

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `config.py` — configuration system.
   - `agents/` — multi-agent framework.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Initialize experiment memory**: The system will auto-create `experiment_memory.json` on first use.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

---

## Experimentation Loop

### Each experiment runs on a single GPU/MPS for a **fixed time budget of 5 minutes**.

**Launch command**: `uv run train.py`

### What you CAN do:
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Use the agent framework — import agents and get suggestions.
- Consult experiment memory — learn from past experiments.

### What you CANNOT do:
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness.

**The goal is simple: get the lowest val_bpb** (bits per byte).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great win.

---

## Using the Agent Framework

### Import and Initialize

```python
from agents import ExperimentMemory, ArchitectureAgent, OptimizerAgent, HyperparameterAgent, AnalystAgent

# Load experiment memory
memory = ExperimentMemory()

# Create agents
arch_agent = ArchitectureAgent(memory)
opt_agent = OptimizerAgent(memory)
hyper_agent = HyperparameterAgent(memory)
analyst = AnalystAgent(memory)
```

### Get Suggestions

```python
# Current configuration
current_config = {
    "depth": 4,
    "aspect_ratio": 32,
    "window_pattern": "SSSL",
    "matrix_lr": 0.04,
    "embedding_lr": 0.6,
    "weight_decay": 0.2,
    "total_batch_size": 16384,
    "device_batch_size": 4,
    "max_seq_len": 512,
}

# Get suggestions from each agent
arch_suggestions = arch_agent.get_suggestions(current_config)
opt_suggestions = opt_agent.get_suggestions(current_config)
hyper_suggestions = hyper_agent.get_suggestions(current_config)

# Get memory context
context = memory.get_context_for_agent("ArchitectureAgent")
hypothesis = memory.generate_hypothesis()
```

### Record Results

```python
from agents import ExperimentRecord
from datetime import datetime

# After experiment completes
record = ExperimentRecord(
    commit="abc1234",
    val_bpb=1.234567,
    memory_mb=2048.5,
    status="keep",  # or "discard" or "crash"
    description="Increased depth from 4 to 6",
    timestamp=datetime.now().isoformat(),
    config_snapshot=current_config,
    metrics={"mfu": 45.2, "tokens_per_sec": 70000},
)

memory.add_experiment(record)
```

### Analyze Results

```python
# Analyze a result
analysis = analyst.analyze_result(
    result={"val_bpb": 1.23, "memory_mb": 2048, "status": "keep"},
    baseline={"val_bpb": 1.25, "memory_mb": 2000}
)

# Get summary
summary = analyst.get_experiment_summary(memory.experiments)

# Get insights
insights = analyst.generate_insights(memory.experiments)

# Get recommendation
recommendation = analyst.recommend_next_direction(memory.experiments)
```

---

## Agent Decision Framework

### When to Keep an Experiment

| Condition | Decision | Confidence |
|-----------|----------|------------|
| val_bpb improved > 0.001 | KEEP | High |
| val_bpb similar, less memory | KEEP | Medium |
| val_bpb similar, simpler code | KEEP | Medium |
| val_bpb degraded > 0.001 | DISCARD | High |
| val_bpb neutral, more complex | DISCARD | Medium |
| Crash (OOM, bug) | DISCARD | Certain |

### Experiment Selection Strategy

1. **First run**: Always establish baseline
2. **Early runs (1-10)**: Explore high-impact changes (depth, LR, batch size)
3. **Mid runs (10-50)**: Refine based on patterns, try architecture changes
4. **Late runs (50+)**: Fine-tune, try combinations, explore radical ideas

### Priority Order for Changes

**High Priority (try first):**
- Learning rate adjustments
- Batch size changes
- Model depth/width

**Medium Priority:**
- Attention pattern changes
- Activation functions
- Weight decay

**Lower Priority (try when stuck):**
- Normalization variants
- Residual connection modifications
- Optimizer replacements

---

## Output Format

After training completes, extract results:

```bash
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

### Logging to results.tsv

Tab-separated format:
```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	2.0	keep	baseline
b2c3d4e	0.993200	2.1	keep	increase LR to 0.04
c3d4e5f	1.005000	2.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

---

## The Experiment Loop

LOOP FOREVER:

1. **Consult memory**: `memory.get_context_for_agent()` and `memory.generate_hypothesis()`
2. **Get suggestions**: From relevant agents based on current state
3. **Select experiment**: Based on suggestions, memory patterns, and risk tolerance
4. **Modify train.py**: Implement the change
5. **Git commit**: `git commit -am "experiment: <description>

Co-authored-by: Qwen-Coder <qwen-coder@alibabacloud.com>"`
6. **Run experiment**: `uv run train.py > run.log 2>&1`
7. **Parse results**: `grep "^val_bpb:" run.log`
8. **Analyze**: Use AnalystAgent to decide keep/discard
9. **Record**: Add to memory and results.tsv
10. **Branch decision**:
    - If KEEP: stay on current commit (advance branch)
    - If DISCARD: `git reset --hard HEAD~1` (revert)
11. **Repeat**

### Timeout Handling

- Expected runtime: ~5 minutes + evaluation
- Timeout threshold: 10 minutes
- If exceeded: kill, mark as crash, revert

### Crash Recovery

1. Check `run.log` for error
2. If simple fix (typo, import): fix and re-run
3. If fundamental (OOM, architecture): mark crash, revert, try different idea

---

## Strategy Guidelines

### Early Stage (0-10 experiments)

**Goal**: Establish baseline, explore high-impact dimensions

**Recommended experiments:**
1. Baseline (required first)
2. Learning rate sweep (0.5x, 2x)
3. Batch size sweep (0.5x, 2x)
4. Depth change (±2 layers)
5. Width change (aspect ratio ±25%)

### Mid Stage (10-50 experiments)

**Goal**: Exploit promising directions, try architecture changes

**Recommended experiments:**
- Attention pattern variations
- Activation function swaps
- Weight decay tuning
- LR schedule modifications
- Combinations of successful single changes

### Late Stage (50+ experiments)

**Goal**: Fine-tune, escape local minima, explore radical ideas

**Recommended experiments:**
- Hyperparameter fine-tuning
- Architecture modifications (GQA, MoE)
- Alternative optimizers
- Reset to best + orthogonal direction

---

## Memory-Driven Decision Making

### When Memory Says "Continue Current Strategy"

If recent experiments show consistent improvement:
- Continue refining the same dimension
- Try smaller step sizes
- Explore nearby hyperparameters

### When Memory Says "Reset and Try Orthogonal"

If recent experiments show degradation:
- Reset to best commit: `git reset --hard <best_commit>`
- Try a fundamentally different approach
- Consult agents for orthogonal suggestions

### When Memory Shows Pattern

If `change_type_success` shows >60% success rate:
- Continue exploring that dimension
- Increase step size if improvements are small

If <30% success rate:
- Avoid that dimension temporarily
- Try fundamentally different approaches

---

## Code Modification Reference

### Key Constants in train.py

```python
# Model architecture
DEPTH = 4                    # Number of transformer layers
ASPECT_RATIO = 32            # model_dim = depth * aspect_ratio
HEAD_DIM = 128               # Attention head dimension
WINDOW_PATTERN = "SSSL"      # L=full, S=half attention

# Optimization
TOTAL_BATCH_SIZE = 2**14     # Total tokens per update
DEVICE_BATCH_SIZE = 4        # Per-device batch
MATRIX_LR = 0.04             # Learning rate for matrices
EMBEDDING_LR = 0.6           # Learning rate for embeddings
WEIGHT_DECAY = 0.2           # Weight decay

# Training
TIME_BUDGET = 300            # 5 minutes
MAX_SEQ_LEN = 512            # Sequence length (MPS)
```

### Common Modifications

**Change depth:**
```python
DEPTH = 6  # was 4
```

**Change learning rate:**
```python
MATRIX_LR = 0.06  # was 0.04 (1.5x increase)
```

**Change attention pattern:**
```python
WINDOW_PATTERN = "LLLL"  # was "SSSL" (full attention)
```

**Change activation (in MLP class):**
```python
# Replace:
x = F.relu(x).square()
# With (SwiGLU):
x = F.silu(x)
```

---

## Never Stop

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. The human expects you to work indefinitely until manually stopped.

If you run out of ideas:
1. Re-read this document for inspiration
2. Consult experiment memory for patterns
3. Try combining previous near-misses
4. Explore more radical architectural changes
5. Reset to best and try orthogonal direction

The loop runs until the human interrupts you, period.

---

## Quick Reference

```
# Setup
git checkout -b autoresearch/<tag>
uv run prepare.py  # if needed

# Run experiment
uv run train.py > run.log 2>&1

# Extract results
grep "^val_bpb:\|^peak_vram_mb:" run.log

# Record (TSV format)
echo -e "<commit>\t<bpb>\t<mem_gb>\t<status>\t<desc>" >> results.tsv

# Git operations
git add train.py && git commit -m "experiment: <description>"
git reset --hard HEAD~1  # if discard
```
