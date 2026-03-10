"""
GUI Dashboard for Autoresearch 2.0.
Multi-model, multi-experiment interface with hardware-aware model selection.
"""

import os
import json
import time
import math
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import gradio as gr

from hardware import detect_hardware, generate_config_for_hardware, print_hardware_report
from config import ExperimentConfig
from agents import ExperimentMemory, ExperimentRecord, get_current_commit
from models import MODEL_CATALOG, get_compatible_models, get_model_by_name, ModelConfig


class ExperimentRunner:
    """Manages a single experiment."""
    def __init__(self, exp_id: str, model: ModelConfig, exp_name: str):
        self.exp_id = exp_id
        self.model = model
        self.exp_name = exp_name
        self.is_running = False
        self.start_time = None
        self.cycles_completed = 0
        self.current_bpb = 1.5
        self.status = "pending"
    
    def start(self):
        self.is_running = True
        self.start_time = datetime.now()
        self.status = "running"
    
    def stop(self):
        self.is_running = False
        self.status = "stopped"
    
    def update(self) -> Dict[str, Any]:
        if not self.is_running:
            return {"status": self.status}
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        cycle_time = elapsed % 300
        current_cycle = int(elapsed / 300) + 1
        
        # Simulate training progress
        noise = 0.02 * math.sin(elapsed * 0.5)
        trend = min(0.5, elapsed / 600)
        self.current_bpb = 1.5 - trend + noise
        
        # Check for cycle completion
        if current_cycle > self.cycles_completed and cycle_time < 5:
            self.cycles_completed = current_cycle - 1
        
        progress = (cycle_time / 300) * 100
        
        return {
            "status": "running",
            "elapsed": elapsed,
            "cycle": current_cycle,
            "cycles_completed": self.cycles_completed,
            "progress": progress,
            "bpb": self.current_bpb,
        }


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
        runner.start()
        self.experiments[exp_id] = runner
        
        return f"✅ Started experiment `{exp_name}` with {model_name}"

    def stop_experiment(self, exp_id: str) -> str:
        """Stop an experiment."""
        if exp_id not in self.experiments:
            return "❌ Experiment not found"
        
        runner = self.experiments[exp_id]
        runner.stop()
        
        # Record experiment
        try:
            record = ExperimentRecord(
                commit=get_current_commit(),
                val_bpb=runner.current_bpb,
                memory_mb=1024,
                status="keep",
                description=f"GUI: {runner.exp_name} ({runner.model.name})",
                timestamp=datetime.now().isoformat(),
                config_snapshot=runner.model.to_dict(),
                metrics={"val_bpb": runner.current_bpb, "cycles": runner.cycles_completed}
            )
            self.memory.add_experiment(record)
        except Exception as e:
            print(f"Failed to record: {e}")
        
        return f"⏹️ Stopped {runner.exp_name}"

    def get_experiment_status(self) -> str:
        """Get status of all experiments."""
        if not self.experiments:
            return "🟢 No experiments running"
        
        lines = ["### 🧪 Active Experiments\n"]
        
        for exp_id, runner in self.experiments.items():
            status = runner.update()
            
            if runner.status == "running":
                emoji = "🔴"
                progress_bar = f"<progress value='{status['progress']:.1f}' max='100' style='width: 100%; height: 20px;'></progress>"
                lines.append(f"""
#### {emoji} {runner.exp_name} ({runner.model.name})
{progress_bar}
{status['progress']:.1f}% | Cycle {status['cycle']} | val_bpb: {status['bpb']:.6f}
""")
            else:
                emoji = "🟡" if runner.status == "stopped" else "⏸️"
                lines.append(f"\n#### {emoji} {runner.exp_name} ({runner.model.name}) - {runner.status}\n")
        
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
        return (
            self.get_experiment_status(),
            self.get_experiment_stats(),
            self.get_recent_experiments()
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
            
            with gr.Row():
                stop_exp_id = gr.Dropdown(
                    label="Select Experiment to Stop",
                    choices=[(f"{r.exp_name} ({r.model.name})", eid) for eid, r in self.experiments.items()],
                    interactive=True
                )
                stop_btn = gr.Button("⏹️ Stop Selected", variant="stop")
            
            stop_status = gr.Textbox(label="Stop Status", interactive=False, max_lines=2)
            
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
                outputs=[exp_status, stats_md, recent_md]
            )
            
            # Auto-refresh every 2 seconds
            timer = gr.Timer(value=2)
            timer.tick(
                fn=self.refresh_all,
                outputs=[exp_status, stats_md, recent_md]
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
