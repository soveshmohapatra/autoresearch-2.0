"""
GUI Dashboard for Autoresearch 2.0.
Multi-model, multi-experiment interface with REAL training.
"""

import os
import json
import time
import math
import subprocess
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import gradio as gr

from hardware import detect_hardware, generate_config_for_hardware, print_hardware_report
from config import ExperimentConfig
from agents import ExperimentMemory, ExperimentRecord, get_current_commit
from models import MODEL_CATALOG, get_compatible_models, get_model_by_name, ModelConfig


class ExperimentRunner:
    """Manages a single real training experiment."""
    def __init__(self, exp_id: str, model: ModelConfig, exp_name: str):
        self.exp_id = exp_id
        self.model = model
        self.exp_name = exp_name
        self.process = None
        self.start_time = None
        self.cycles_completed = 0
        self.current_bpb = None
        self.status = "pending"
        self.log_lines = []
        self.pid = None
    
    def start(self):
        """Start real training process."""
        self.start_time = datetime.now()
        self.status = "starting"
        
        # Build train.py command with model config
        cmd = [
            "uv", "run", "python", "train.py",
            "--depth", str(self.model.depth),
            "--aspect-ratio", str(self.model.aspect_ratio),
            "--batch-size", str(self.model.recommended_batch_size),
            "--seq-len", str(self.model.recommended_seq_len),
            "--optimizer", self.model.optimizer,
            "--experiment-name", self.exp_name,
        ]
        
        if self.model.use_moe:
            cmd.extend(["--use-moe", "--moe-experts", str(self.model.moe_num_experts)])
        if self.model.use_gqa:
            cmd.append("--use-gqa")
        if self.model.use_swiglu:
            cmd.append("--use-swiglu")
        if self.model.use_prenorm:
            cmd.append("--use-prenorm")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid
            )
            self.pid = self.process.pid
            self.status = "running"
            return True, f"Started training (PID: {self.pid})"
        except Exception as e:
            self.status = "failed"
            return False, str(e)
    
    def stop(self):
        """Stop training process."""
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except:
                if self.process.poll() is None:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.status = "stopped"
            return True
        self.status = "stopped"
        return False
    
    def update(self) -> Dict[str, Any]:
        """Update status from running process."""
        if self.process:
            # Read any new output
            if self.process.stdout:
                line = self.process.stdout.readline()
                if line:
                    self.log_lines.append(line.strip())
                    # Parse val_bpb from output
                    if "val_bpb" in line.lower():
                        try:
                            parts = line.split()
                            for i, p in enumerate(parts):
                                if "val_bpb" in p.lower() and i+1 < len(parts):
                                    self.current_bpb = float(parts[i+1])
                                    break
                        except:
                            pass
            
            # Check if process ended
            if self.process.poll() is not None and self.status == "running":
                self.status = "completed"
                self.cycles_completed += 1
        
        elapsed = 0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "status": self.status,
            "elapsed": elapsed,
            "bpb": self.current_bpb,
            "cycles": self.cycles_completed,
            "logs": "\n".join(self.log_lines[-10:]),  # Last 10 lines
        }
    
    def get_final_bpb(self) -> float:
        """Get final val_bpb from logs."""
        # Parse logs for final val_bpb
        for line in reversed(self.log_lines):
            if "val_bpb" in line.lower():
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "val_bpb" in p.lower() and i+1 < len(parts):
                            return float(parts[i+1])
                except:
                    pass
        return self.current_bpb or 1.5


class AutoresearchGUI:
    """Gradio-based GUI for autoresearch."""

    def __init__(self):
        self.hardware = detect_hardware()
        self.hw_info = self.hardware.to_dict()
        self.config = generate_config_for_hardware(self.hardware)
        self.memory = ExperimentMemory()
        self.experiments: Dict[str, ExperimentRunner] = {}
        self.compatible_models = get_compatible_models(self.hw_info)

    def get_hardware_info(self) -> str:
        hw = self.hardware
        tier = "HIGH-END" if hw.is_high_end else "STANDARD"
        return f"""
### 🔍 Your Hardware

| Property | Value |
|----------|-------|
| Device | {hw.device_type.value.upper()} |
| Name | {hw.device_name} |
| Memory | {hw.total_memory_gb:.1f} GB |
| Tier | {tier} |
| Compatible Models | {len(self.compatible_models)} / {len(MODEL_CATALOG)} |
"""

    def get_model_options(self) -> List[Dict]:
        """Get dropdown options for models."""
        options = []
        for model in MODEL_CATALOG:
            compatible = model.is_compatible(self.hw_info)
            label = f"{'✅' if compatible else '❌'} {model.name} ({model.param_count_millions:.1f}M) - {model.description}"
            options.append({"label": label, "value": model.name, "disabled": not compatible})
        return options

    def get_model_details(self, model_name: str) -> str:
        """Get detailed info about selected model."""
        model = get_model_by_name(model_name)
        if not model:
            return "Select a model to see details"
        
        compatible = model.is_compatible(self.hw_info)
        status = "✅ Compatible" if compatible else "❌ Not compatible with your hardware"
        
        features = []
        if model.use_moe:
            features.append(f"MoE ({model.moe_num_experts} experts)")
        if model.use_gqa:
            features.append("GQA")
        if model.use_swiglu:
            features.append("SwiGLU")
        if model.use_prenorm:
            features.append("Pre-norm")
        
        return f"""
### {model.name}

{status}

| Property | Value |
|----------|-------|
| Parameters | {model.param_count_millions:.1f}M |
| Depth | {model.depth} layers |
| Dimension | {model.model_dim} |
| Heads | {model.num_heads} |
| Batch Size | {model.recommended_batch_size} |
| Sequence Length | {model.recommended_seq_len} |
| Min VRAM | {model.min_vram_gb:.1f} GB |
| Optimizer | {model.optimizer} |
| Features | {', '.join(features) if features else 'Standard'} |
"""

    def start_experiment(self, model_name: str, exp_name: str) -> str:
        """Start a new experiment."""
        model = get_model_by_name(model_name)
        if not model:
            return "❌ Invalid model selected"
        
        if not model.is_compatible(self.hw_info):
            return f"❌ {model_name} is not compatible with your hardware!"
        
        exp_id = f"exp_{datetime.now().strftime('%m%d_%H%M%S')}"
        runner = ExperimentRunner(exp_id, model, exp_name)
        success, msg = runner.start()
        
        if success:
            self.experiments[exp_id] = runner
            return f"✅ Started `{exp_name}` with {model_name}\n{msg}"
        else:
            return f"❌ Failed to start: {msg}"

    def stop_experiment(self, exp_id: str) -> str:
        """Stop an experiment."""
        if exp_id not in self.experiments:
            return "❌ Experiment not found"
        
        runner = self.experiments[exp_id]
        runner.stop()
        
        # Record experiment
        try:
            final_bpb = runner.get_final_bpb()
            record = ExperimentRecord(
                commit=get_current_commit(),
                val_bpb=final_bpb,
                memory_mb=1024,
                status="keep",
                description=f"GUI: {runner.exp_name} ({runner.model.name})",
                timestamp=datetime.now().isoformat(),
                config_snapshot=runner.model.to_dict(),
                metrics={"val_bpb": final_bpb, "cycles": runner.cycles_completed, "pid": runner.pid}
            )
            self.memory.add_experiment(record)
            return f"⏹️ Stopped {runner.exp_name}\nFinal val_bpb: {final_bpb:.6f}"
        except Exception as e:
            return f"⏹️ Stopped {runner.exp_name}\nError recording: {e}"

    def get_experiment_status(self) -> str:
        """Get status of all experiments."""
        if not self.experiments:
            return "🟢 No experiments running"
        
        lines = ["### 🧪 Active Experiments\n"]
        
        for exp_id, runner in self.experiments.items():
            status = runner.update()
            
            if runner.status == "running":
                elapsed = status.get('elapsed', 0)
                bpb = status.get('bpb', None)
                bpb_str = f"{bpb:.6f}" if bpb else "N/A"
                
                lines.append(f"""
#### 🔴 {runner.exp_name} ({runner.model.name})

| Metric | Value |
|--------|-------|
| Status | Running (PID: {runner.pid}) |
| Elapsed | {elapsed:.0f}s |
| val_bpb | {bpb_str} |

**Recent logs:**
```
{status.get('logs', 'Starting...')}
```
""")
            elif runner.status == "starting":
                lines.append(f"\n#### 🟡 {runner.exp_name} ({runner.model.name}) - Starting...\n")
            elif runner.status == "completed":
                lines.append(f"\n#### ✅ {runner.exp_name} ({runner.model.name}) - Completed\n")
            elif runner.status == "stopped":
                lines.append(f"\n#### ⏹️ {runner.exp_name} ({runner.model.name}) - Stopped\n")
            elif runner.status == "failed":
                lines.append(f"\n#### ❌ {runner.exp_name} ({runner.model.name}) - Failed\n")
        
        return "\n".join(lines)

    def get_experiment_stats(self) -> str:
        stats = self.memory.get_statistics()
        if stats["total"] == 0:
            return "📊 No experiments recorded yet"
        
        keep_rate = stats.get('keep_rate', 0) * 100
        best_bpb = stats.get('best_bpb', float('inf'))
        best_bpb_str = f"{best_bpb:.6f}" if best_bpb != float('inf') else "N/A"
        
        return f"""
### 📈 Statistics

| Metric | Value |
|--------|-------|
| Total | {stats['total']} |
| Kept | {stats['kept']} ({keep_rate:.1f}%) |
| Best val_bpb | {best_bpb_str} |
"""

    def get_recent_experiments(self) -> str:
        recent = self.memory.get_recent_experiments(10)
        if not recent:
            return "No experiments recorded"
        
        rows = ["| Status | Model | val_bpb | Description |", "|--------|-------|---------|-------------|"]
        for exp in recent:
            emoji = {"keep": "✅", "discard": "❌", "crash": "💥"}.get(exp.status, "❓")
            model = exp.config_snapshot.get('name', 'Unknown') if isinstance(exp.config_snapshot, dict) else 'Unknown'
            desc = exp.description[:25] + "..." if len(exp.description) > 25 else exp.description
            rows.append(f"| {emoji} | {model} | {exp.val_bpb:.6f} | {desc} |")
        
        return "\n".join(rows)

    def refresh_all(self) -> tuple:
        """Refresh all dynamic content."""
        # Update stop dropdown choices
        stop_choices = [
            (f"{r.exp_name} ({r.model.name}) - {r.status}", eid) 
            for eid, r in self.experiments.items()
        ]
        
        return (
            self.get_experiment_status(),
            self.get_experiment_stats(),
            self.get_recent_experiments(),
            gr.Dropdown(choices=stop_choices, value=None)
        )

    def launch(self, share: bool = False):
        """Launch the Gradio interface."""
        
        with gr.Blocks(title="Autoresearch 2.0", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🧠 Autoresearch 2.0")
            gr.Markdown("Multi-model LLM experimentation with hardware-aware selection")
            gr.Markdown("---")
            
            # Row 1: Hardware & Model Selection
            with gr.Row():
                with gr.Column(scale=1):
                    hw_info = gr.Markdown(self.get_hardware_info())
                
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        label="Select Model",
                        choices=[(f"{'✅' if m.is_compatible(self.hw_info) else '❌'} {m.name} ({m.param_count_millions:.1f}M)", m.name) 
                                for m in self.compatible_models],
                        value=self.compatible_models[0].name if self.compatible_models else None
                    )
                    model_details = gr.Markdown(self.get_model_details(self.compatible_models[0].name if self.compatible_models else ""))
            
            model_dropdown.change(
                fn=self.get_model_details,
                inputs=[model_dropdown],
                outputs=[model_details]
            )
            
            # Row 2: Start Experiment
            gr.Markdown("---")
            gr.Markdown("## 🚀 Start New Experiment")
            
            with gr.Row():
                with gr.Column(scale=1):
                    exp_name_input = gr.Textbox(
                        label="Experiment Name",
                        placeholder="e.g., mar10-nano-test",
                        value=f"exp_{datetime.now().strftime('%m%d_%H%M')}"
                    )
                    start_btn = gr.Button("▶️ Start Experiment", variant="primary")
                
                with gr.Column(scale=1):
                    start_status = gr.Textbox(label="Status", interactive=False, max_lines=2)
            
            start_btn.click(
                fn=self.start_experiment,
                inputs=[model_dropdown, exp_name_input],
                outputs=[start_status]
            )
            
            # Row 3: Active Experiments
            gr.Markdown("---")
            gr.Markdown("## 🧪 Active Experiments")
            
            exp_status = gr.Markdown(self.get_experiment_status())
            
            stop_exp_id = gr.Dropdown(
                label="Select Experiment to Stop",
                choices=[],
                interactive=True
            )
            stop_btn = gr.Button("⏹️ Stop Selected", variant="stop")
            
            stop_status = gr.Textbox(label="Stop Status", interactive=False, max_lines=3)
            
            stop_btn.click(
                fn=self.stop_experiment,
                inputs=[stop_exp_id],
                outputs=[stop_status]
            )
            
            # Row 4: History
            gr.Markdown("---")
            gr.Markdown("## 📊 Experiment History")
            
            with gr.Row():
                stats_md = gr.Markdown(self.get_experiment_stats())
                recent_md = gr.Markdown(self.get_recent_experiments())
            
            refresh_btn = gr.Button("🔄 Refresh All", variant="secondary")
            refresh_btn.click(
                fn=self.refresh_all,
                outputs=[exp_status, stats_md, recent_md, stop_exp_id]
            )
            
            # Auto-refresh every 2 seconds
            timer = gr.Timer(value=2)
            timer.tick(
                fn=self.refresh_all,
                outputs=[exp_status, stats_md, recent_md, stop_exp_id]
            )
            
            # Footer
            gr.Markdown("---")
            gr.Markdown("*Autoresearch 2.0 - Based on Andrej Karpathy's autoresearch*")
        
        demo.launch(share=share, server_name="0.0.0.0")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autoresearch GUI")
    parser.add_argument("--share", action="store_true", help="Public link")
    parser.add_argument("--detect-only", action="store_true", help="Only detect hardware")
    args = parser.parse_args()

    if args.detect_only:
        hw = detect_hardware()
        print_hardware_report(hw)
        compatible = get_compatible_models(hw.to_dict())
        print(f"\nCompatible models ({len(compatible)}):")
        for m in compatible:
            print(f"  {m.name:<20} {m.param_count_millions:>6.1f}M - {m.description}")
        return

    gui = AutoresearchGUI()
    gui.launch(share=args.share)


if __name__ == "__main__":
    main()
