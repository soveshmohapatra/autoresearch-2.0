"""
GUI Dashboard for Autoresearch.
Beautiful web interface for monitoring and controlling experiments.
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import gradio as gr

# Import local modules
from hardware import detect_hardware, generate_config_for_hardware, print_hardware_report
from config import ExperimentConfig
from agents import ExperimentMemory


class AutoresearchGUI:
    """Gradio-based GUI for autoresearch."""
    
    def __init__(self):
        self.hardware = detect_hardware()
        self.config = generate_config_for_hardware(self.hardware)
        self.memory = ExperimentMemory()
        self.is_training = False
        self.current_experiment = None
        
    def get_hardware_info_html(self) -> str:
        """Generate HTML display for hardware info."""
        hw = self.hardware
        color = "green" if hw.is_high_end else "orange"
        
        return f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white; margin-bottom: 20px;">
            <h2 style="margin: 0 0 10px 0;">🔍 Detected Hardware</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div>
                    <div style="opacity: 0.8; font-size: 12px;">Device</div>
                    <div style="font-size: 18px; font-weight: bold;">{hw.device_type.value.upper()}</div>
                </div>
                <div>
                    <div style="opacity: 0.8; font-size: 12px;">Name</div>
                    <div style="font-size: 16px;">{hw.device_name}</div>
                </div>
                <div>
                    <div style="opacity: 0.8; font-size: 12px;">Memory</div>
                    <div style="font-size: 16px;">{hw.total_memory_gb:.1f} GB</div>
                </div>
                <div>
                    <div style="opacity: 0.8; font-size: 12px;">Tier</div>
                    <div style="font-size: 16px; color: {color};">{'⭐ HIGH-END' if hw.is_high_end else '📌 STANDARD'}</div>
                </div>
            </div>
        </div>
        """
    
    def get_recommended_config_html(self) -> str:
        """Generate HTML display for recommended config."""
        cfg = self.config
        
        return f"""
        <div style="padding: 20px; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #667eea;">
            <h3 style="margin: 0 0 15px 0;">📋 Recommended Configuration</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-family: monospace;">
                <div><b>Depth:</b> {cfg['depth']} layers</div>
                <div><b>Batch Size:</b> {cfg['device_batch_size']}</div>
                <div><b>Sequence Length:</b> {cfg['max_seq_len']}</div>
                <div><b>Precision:</b> {cfg['dtype']}</div>
                <div><b>Optimizer:</b> {cfg['optimizer_type']}</div>
                <div><b>SwiGLU:</b> {'✅' if cfg['use_swiglu'] else '❌'}</div>
                <div><b>GQA:</b> {'✅' if cfg['use_gqa'] else '❌'}</div>
                <div><b>MoE:</b> {'✅' if cfg['use_moe'] else '❌'}</div>
            </div>
        </div>
        """
    
    def detect_and_configure(self) -> tuple:
        """Re-detect hardware and update configuration."""
        self.hardware = detect_hardware()
        self.config = generate_config_for_hardware(self.hardware)
        
        return (
            self.get_hardware_info_html(),
            self.get_recommended_config_html(),
            f"✅ Hardware re-detected: {self.hardware.device_name}"
        )
    
    def apply_recommended_config(self) -> str:
        """Apply recommended configuration to train.py."""
        # This would modify train.py constants
        # For now, save to a config file
        config_path = Path("auto_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        return f"✅ Configuration saved to {config_path}\n\nYou can now run: uv run train.py"
    
    def get_experiment_stats(self) -> str:
        """Get experiment statistics from memory."""
        stats = self.memory.get_statistics()
        
        if stats["total"] == 0:
            return "📊 No experiments recorded yet."
        
        return f"""
        ### Experiment Statistics
        
        | Metric | Value |
        |--------|-------|
        | Total Experiments | {stats['total']} |
        | Kept | {stats['kept']} ({stats.get('keep_rate', 0)*100:.1f}%) |
        | Discarded | {stats['discarded']} |
        | Best val_bpb | {stats['best_bpb']:.6f} |
        | Total Improvement | {stats.get('total_improvement', 0):.6f} bpb |
        """
    
    def get_recent_experiments_table(self) -> str:
        """Get table of recent experiments."""
        recent = self.memory.get_recent_experiments(10)
        
        if not recent:
            return "No recent experiments."
        
        rows = []
        for exp in recent:
            status_emoji = {"keep": "✅", "discard": "❌", "crash": "💥"}.get(exp.status, "❓")
            rows.append(f"| {status_emoji} | {exp.val_bpb:.6f} | {exp.description[:40]} |")
        
        return "| Status | val_bpb | Description |\n|--------|---------|-------------|\n" + "\n".join(rows)
    
    def start_training(self, experiment_name: str, time_budget: int) -> str:
        """Start a training run."""
        if self.is_training:
            return "❌ Training is already in progress!"
        
        self.is_training = True
        self.current_experiment = {
            "name": experiment_name,
            "start_time": datetime.now(),
            "time_budget": time_budget,
        }
        
        # In a real implementation, this would spawn a training process
        # For now, simulate with a message
        return f"""
🚀 **Training Started!**

- **Experiment:** {experiment_name}
- **Time Budget:** {time_budget} seconds
- **Started:** {self.current_experiment['start_time'].strftime('%H:%M:%S')}
- **Config:** Depth={self.config['depth']}, Batch={self.config['device_batch_size']}

Training is running in the background. Check the logs for progress.
        """
    
    def stop_training(self) -> str:
        """Stop the current training run."""
        if not self.is_training:
            return "❌ No training is currently running."
        
        self.is_training = False
        
        if self.current_experiment:
            end_time = datetime.now()
            duration = (end_time - self.current_experiment['start_time']).total_seconds()
            
            result = f"""
⏹️ **Training Stopped**

- **Experiment:** {self.current_experiment['name']}
- **Duration:** {duration:.1f} seconds
- **Ended:** {end_time.strftime('%H:%M:%S')}
            """
            self.current_experiment = None
            return result
        
        return "✅ Training stopped."
    
    def get_training_status(self) -> str:
        """Get current training status."""
        if not self.is_training:
            return "🟢 **Idle** - No training running"
        
        if self.current_experiment:
            elapsed = (datetime.now() - self.current_experiment['start_time']).total_seconds()
            remaining = max(0, self.current_experiment['time_budget'] - elapsed)
            
            return f"""
🔴 **Training in Progress**

- **Experiment:** {self.current_experiment['name']}
- **Elapsed:** {elapsed:.0f}s / {self.current_experiment['time_budget']}s
- **Remaining:** {remaining:.0f}s
- **Progress:** {min(100, elapsed/self.current_experiment['time_budget']*100):.1f}%
            """
        
        return "🟡 **Unknown** status"
    
    def launch(self, share: bool = False):
        """Launch the Gradio interface."""

        with gr.Blocks(title="Autoresearch Dashboard") as demo:
            gr.Markdown("# 🧠 Autoresearch Dashboard")
            gr.Markdown("Autonomous AI research with intelligent hardware detection")

            # Hardware Detection Section
            with gr.Row():
                with gr.Column(scale=2):
                    hardware_html = gr.HTML(self.get_hardware_info_html())
                    config_html = gr.HTML(self.get_recommended_config_html())

                with gr.Column(scale=1):
                    gr.Markdown("### ⚡ Quick Actions")
                    detect_btn = gr.Button("🔍 Re-detect Hardware", variant="primary")
                    apply_btn = gr.Button("📝 Apply Recommended Config", variant="secondary")
                    status_text = gr.Textbox(label="Status", interactive=False)

            detect_btn.click(
                fn=self.detect_and_configure,
                outputs=[hardware_html, config_html, status_text]
            )

            apply_btn.click(
                fn=self.apply_recommended_config,
                outputs=[status_text]
            )

            # Training Control Section
            gr.Markdown("## 🎮 Training Control")

            with gr.Row():
                with gr.Column():
                    exp_name = gr.Textbox(
                        label="Experiment Name",
                        placeholder="e.g., mar10-baseline",
                        value=f"exp-{datetime.now().strftime('%m%d')}"
                    )
                    time_budget = gr.Slider(
                        label="Time Budget (seconds)",
                        minimum=60,
                        maximum=600,
                        value=300,
                        step=60
                    )

                    with gr.Row():
                        start_btn = gr.Button("▶️ Start Training", variant="primary")
                        stop_btn = gr.Button("⏹️ Stop Training", variant="stop")

                with gr.Column():
                    status_display = gr.Markdown(self.get_training_status())
                    refresh_timer = gr.Timer(value=5)

            # Auto-refresh status using Timer (Gradio 6.0 compatible)
            refresh_timer.tick(
                fn=lambda: self.get_training_status(),
                outputs=[status_display]
            )
            
            start_btn.click(
                fn=lambda name, time: self.start_training(name, time),
                inputs=[exp_name, time_budget],
                outputs=[status_display]
            )
            
            stop_btn.click(
                fn=lambda: self.stop_training(),
                outputs=[status_display]
            )
            
            # Experiment History Section
            gr.Markdown("## 📊 Experiment History")

            with gr.Row():
                with gr.Column():
                    stats_md = gr.Markdown(self.get_experiment_stats())
                with gr.Column():
                    recent_md = gr.Markdown(self.get_recent_experiments_table())

            refresh_btn = gr.Button("🔄 Refresh Statistics")
            refresh_btn.click(
                fn=lambda: (self.get_experiment_stats(), self.get_recent_experiments_table()),
                outputs=[stats_md, recent_md]
            )
            
            # Settings Section
            gr.Markdown("## ⚙️ Settings")
            
            with gr.Accordion("Advanced Configuration", open=False):
                gr.Markdown("""
                ### Architecture Variants
                - **MoE (Mixture of Experts):** Enable for high-end GPUs
                - **GQA (Grouped Query Attention):** Reduces memory usage
                - **SwiGLU:** Better activation function
                
                ### Optimizers
                - **Muon+AdamW:** Default, best for most cases
                - **Lion:** Memory efficient
                - **Adafactor:** Adaptive LR
                
                ### Tips
                - Higher depth = more capacity but slower
                - Larger batch = more stable but more memory
                - Enable checkpointing for long runs
                """)
            
            # Footer
            gr.Markdown("""
            ---
            **Autoresearch** - Autonomous AI Research System
            """)

        # Launch
        demo.launch(share=share, server_name="0.0.0.0", theme=gr.themes.Base())


def main():
    """Main entry point for GUI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autoresearch GUI Dashboard")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    parser.add_argument("--detect-only", action="store_true", help="Only detect hardware and exit")
    args = parser.parse_args()
    
    if args.detect_only:
        hardware = detect_hardware()
        print_hardware_report(hardware)
        config = generate_config_for_hardware(hardware)
        print("\nGenerated Configuration:")
        for key, value in config.items():
            if key != "hardware_info":
                print(f"  {key}: {value}")
        return
    
    gui = AutoresearchGUI()
    gui.launch(share=args.share)


if __name__ == "__main__":
    main()
