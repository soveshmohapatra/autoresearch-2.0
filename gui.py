"""
Autoresearch 2.0 — Terminal Dashboard
Beautiful Rich-based TUI that works on Mac, SSH, tmux, GPU servers — anywhere.
Usage: uv run python gui.py [--run MODEL_NAME] [--detect]
"""

import os
import sys
import time
import signal
import subprocess
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.prompt import Prompt
from rich import box

from hardware import detect_hardware, print_hardware_report
from models import MODEL_CATALOG, get_compatible_models, get_model_by_name
from agents.memory import ExperimentMemory, ExperimentRecord
from agents.utils import get_current_commit

console = Console()

LANGUAGE_OPTIONS = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "hi": "Hindi",
    "zh": "Chinese",
    "ja": "Japanese",
    "gu": "Gujarati",
    "nl": "Dutch",
    "or": "Odia",
}


# ---------------------------------------------------------------------------
# Hardware detection banner
# ---------------------------------------------------------------------------

def render_hardware_panel(hardware) -> Panel:
    tier_color = "green" if hardware.is_high_end else "yellow"
    tier = "HIGH-END" if hardware.is_high_end else "STANDARD"
    device_icon = {"cuda": "[bold cyan]NVIDIA CUDA[/]", "mps": "[bold magenta]Apple Silicon[/]", "cpu": "[bold white]CPU[/]"}
    icon = device_icon.get(hardware.device_type.value, hardware.device_type.value.upper())

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Device", icon)
    table.add_row("Name", hardware.device_name)
    table.add_row("Memory", f"{hardware.total_memory_gb:.1f} GB")
    table.add_row("Tier", f"[{tier_color}]{tier}[/{tier_color}]")
    table.add_row("Peak FLOPS", f"{hardware.peak_flops / 1e12:.1f} TFLOPS")
    table.add_row("Rec. Batch", str(hardware.recommended_batch_size))
    table.add_row("Rec. Depth", str(hardware.recommended_depth))
    table.add_row("Rec. SeqLen", str(hardware.recommended_seq_len))

    return Panel(table, title="[bold]Hardware[/bold]", border_style="bright_blue")


# ---------------------------------------------------------------------------
# Model catalog table
# ---------------------------------------------------------------------------

def render_model_table(hardware) -> Table:
    hw_dict = hardware.to_dict()
    table = Table(
        title="Model Catalog",
        box=box.ROUNDED,
        show_lines=False,
        border_style="bright_blue",
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="bold", min_width=18)
    table.add_column("Params", justify="right", width=8)
    table.add_column("Depth", justify="right", width=6)
    table.add_column("Dim", justify="right", width=5)
    table.add_column("Features", min_width=20)
    table.add_column("Optimizer", width=12)
    table.add_column("Min VRAM", justify="right", width=9)
    table.add_column("Status", width=10)

    for i, m in enumerate(MODEL_CATALOG, 1):
        compatible = m.is_compatible(hw_dict)
        status = "[green]Compatible[/]" if compatible else "[red]Too Large[/]"
        row_style = "" if compatible else "dim"

        feats = []
        if m.use_moe:
            feats.append(f"MoE×{m.moe_num_experts}")
        if m.use_gqa:
            feats.append("GQA")
        if m.use_swiglu:
            feats.append("SwiGLU")
        if m.use_prenorm:
            feats.append("PreNorm")
        feat_str = ", ".join(feats) if feats else "Standard"

        table.add_row(
            str(i),
            m.name,
            f"{m.param_count_millions:.1f}M",
            str(m.depth),
            str(m.model_dim),
            feat_str,
            m.optimizer,
            f"{m.min_vram_gb:.0f} GB",
            status,
            style=row_style,
        )

    return table


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Manages a single live training process, parsing its stdout."""

    def __init__(self, exp_id: str, model_name: str, exp_name: str, time_budget: int = 300, resume: bool = False, language: str = "en"):
        self.exp_id = exp_id
        self.model_name = model_name
        self.exp_name = exp_name
        self.time_budget = time_budget
        self.resume = resume
        self.language = language
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.status = "pending"
        self.log_lines: list[str] = []
        self.loss_history: list[tuple[int, float]] = []  # (step, loss)
        self.current_step = 0
        self.current_loss: Optional[float] = None
        self.current_bpb: Optional[float] = None
        self.current_mfu: Optional[float] = None
        self.current_pct: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> tuple[bool, str]:
        model = get_model_by_name(self.model_name)
        if not model:
            return False, f"Unknown model: {self.model_name}"

        cmd = [
            sys.executable, "train.py",
            "--depth", str(model.depth),
            "--aspect-ratio", str(model.aspect_ratio),
            "--batch-size", str(model.recommended_batch_size),
            "--seq-len", str(model.recommended_seq_len),
            "--optimizer", model.optimizer,
            "--experiment-name", self.exp_name,
            "--time-budget", str(self.time_budget),
        ]
        if model.use_moe:
            cmd += ["--use-moe", "--moe-experts", str(model.moe_num_experts)]
        if model.use_gqa:
            cmd.append("--use-gqa")
        if model.use_swiglu:
            cmd.append("--use-swiglu")
        if model.use_prenorm:
            cmd.append("--use-prenorm")
        if self.resume:
            cmd.append("--resume")
        cmd += ["--language", self.language]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
            )
            self.start_time = time.time()
            self.status = "running"
            self._thread = threading.Thread(target=self._read_output, daemon=True)
            self._thread.start()
            return True, f"PID {self.process.pid}"
        except Exception as e:
            self.status = "failed"
            return False, str(e)

    def _read_output(self):
        """Background thread: read stdout and parse metrics."""
        for line in self.process.stdout:
            line = line.rstrip()
            with self._lock:
                self.log_lines.append(line)
                if len(self.log_lines) > 200:
                    self.log_lines.pop(0)
            self._parse_line(line)
        rc = self.process.wait()
        with self._lock:
            self.status = "completed" if rc == 0 else "failed"

    def _parse_line(self, line: str):
        """Parse step/loss/mfu/pct from training log lines."""
        # Format: step 00042 (17.3%) | loss: 3.456789 | lrm: 1.00 | ... | mfu: 42.1%
        try:
            if "step " in line and "loss:" in line:
                parts = line.split("|")
                step_part = parts[0].strip()
                # step NNNNN (PP.P%)
                sp = step_part.split()
                if len(sp) >= 3:
                    self.current_step = int(sp[1])
                    pct_str = sp[2].strip("()")
                    self.current_pct = float(pct_str.rstrip("%"))
                for p in parts:
                    p = p.strip()
                    if p.startswith("loss:"):
                        self.current_loss = float(p.split(":")[1].strip())
                        self.loss_history.append((self.current_step, self.current_loss))
                    elif p.startswith("mfu:"):
                        self.current_mfu = float(p.split(":")[1].strip().rstrip("%"))
            elif "val_bpb:" in line:
                val = line.split("val_bpb:")[1].strip().split()[0]
                self.current_bpb = float(val)
        except Exception:
            pass

    def stop(self):
        if self.process and self.process.poll() is None:
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
        self.status = "stopped"

    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0

    def get_recent_logs(self, n: int = 8) -> list[str]:
        with self._lock:
            return self.log_lines[-n:]

    def get_final_bpb(self) -> Optional[float]:
        return self.current_bpb


# ---------------------------------------------------------------------------
# Live monitor panel
# ---------------------------------------------------------------------------

def render_experiment_panel(runner: ExperimentRunner) -> Panel:
    status_color = {
        "running": "green", "completed": "cyan", "failed": "red",
        "stopped": "yellow", "pending": "dim", "starting": "blue",
    }.get(runner.status, "white")

    elapsed = runner.elapsed()
    m, s = divmod(int(elapsed), 60)

    # Progress bar
    bar_width = 30
    filled = int(bar_width * runner.current_pct / 100)
    bar = "[green]" + "█" * filled + "[/][dim]" + "░" * (bar_width - filled) + "[/dim]"

    header = Table(box=None, show_header=False, padding=(0, 2))
    header.add_column("k")
    header.add_column("v", style="bold")
    header.add_row("Experiment", runner.exp_name)
    header.add_row("Model", runner.model_name)
    header.add_row("Status", f"[{status_color}]{runner.status.upper()}[/{status_color}]")
    header.add_row("Elapsed", f"{m:02d}:{s:02d}")
    is_starting_up = runner.current_loss is None and runner.status == "running"
    if is_starting_up:
        header.add_row("Progress", "[dim]Loading model / compiling...[/dim]")
    else:
        header.add_row("Progress", f"{bar} {runner.current_pct:.1f}%")
    if runner.current_loss is not None:
        header.add_row("Train Loss", f"{runner.current_loss:.6f}")
    if runner.current_bpb is not None:
        header.add_row("Val BPB", f"[bold yellow]{runner.current_bpb:.6f}[/]")
    if runner.current_mfu is not None:
        header.add_row("MFU", f"{runner.current_mfu:.1f}%")

    # Loss curve
    from rich.console import Group
    loss_panel = _render_loss_curve(runner.loss_history)
    content = Group(header, loss_panel)

    return Panel(content, title=f"[bold]{runner.exp_name}[/bold]", border_style=status_color)


_SPARK = "▁▂▃▄▅▆▇█"


def _render_loss_curve(history: list[tuple[int, float]]) -> Panel:
    """Render a sparkline loss curve panel."""
    if not history:
        return Panel("[dim]Waiting for training data...[/dim]", title="[dim]Loss Curve[/dim]", border_style="dim", padding=(0, 1))

    losses = [l for _, l in history]
    lo, hi = min(losses), max(losses)
    width = min(len(losses), 60)
    sampled = losses[-width:]  # show most recent points

    # Build sparkline
    if hi > lo:
        spark = "".join(_SPARK[int((v - lo) / (hi - lo) * 7)] for v in sampled)
    else:
        spark = _SPARK[3] * len(sampled)

    first_loss = losses[0]
    last_loss = losses[-1]
    drop = first_loss - last_loss
    drop_str = f"[green]▼ {drop:.4f}[/green]" if drop > 0 else f"[red]▲ {abs(drop):.4f}[/red]"

    t = Text()
    t.append(f" {spark}\n", style="cyan")
    t.append(f" Start: {first_loss:.4f}  →  Now: {last_loss:.4f}  {drop_str}  Min: {min(losses):.4f}", style="dim")

    return Panel(t, title="[dim]Loss Curve[/dim]", border_style="dim", padding=(0, 1))


# ---------------------------------------------------------------------------
# Stats panel
# ---------------------------------------------------------------------------

def render_stats_panel(memory: ExperimentMemory) -> Panel:
    stats = memory.get_statistics()

    if stats["total"] == 0:
        return Panel("[dim]No experiments recorded yet.[/dim]", title="[bold]Statistics[/bold]", border_style="bright_blue")

    keep_pct = stats.get("keep_rate", 0) * 100
    best = stats.get("best_bpb", float("inf"))
    best_str = f"{best:.6f}" if best != float("inf") else "N/A"

    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    summary.add_column("k", style="dim")
    summary.add_column("v", style="bold")
    summary.add_row("Total", str(stats["total"]))
    summary.add_row("Kept", f"{stats['kept']} ({keep_pct:.0f}%)")
    summary.add_row("Best Val BPB", f"[bold green]{best_str}[/]")

    # Recent experiments table
    recent = memory.get_recent_experiments(8)
    if recent:
        t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold dim", padding=(0, 1))
        t.add_column("Status", width=8)
        t.add_column("Model", width=14)
        t.add_column("Val BPB", justify="right", width=10)
        t.add_column("Description", min_width=20)
        for exp in reversed(recent):
            emoji = {"keep": "[green]KEEP[/]", "discard": "[red]DROP[/]", "crash": "[yellow]CRASH[/]"}.get(exp.status, exp.status)
            model = exp.config_snapshot.get("name", "?") if isinstance(exp.config_snapshot, dict) else "?"
            desc = exp.description[:28] + "…" if len(exp.description) > 28 else exp.description
            t.add_row(emoji, model, f"{exp.val_bpb:.6f}", desc)

        from rich.console import Group
        content = Group(summary, t)
    else:
        content = summary

    return Panel(content, title="[bold]Experiment History[/bold]", border_style="bright_blue")


# ---------------------------------------------------------------------------
# Main interactive dashboard
# ---------------------------------------------------------------------------

class Dashboard:
    def __init__(self):
        self.hardware = detect_hardware()
        self.memory = ExperimentMemory()
        self.runners: Dict[str, ExperimentRunner] = {}
        self.hw_compatible = get_compatible_models(self.hardware.to_dict())

    def show_welcome(self):
        console.print()
        console.print(Panel(
            "[bold cyan]Autoresearch 2.0[/bold cyan] — Hardware-aware LLM research platform\n"
            "[dim]Multi-platform: NVIDIA CUDA · Apple Silicon · CPU[/dim]",
            border_style="bright_blue",
            padding=(1, 4),
        ))
        console.print(render_hardware_panel(self.hardware))
        console.print()

    def show_models(self):
        console.print(render_model_table(self.hardware))
        console.print(f"\n  [dim]{len(self.hw_compatible)}/{len(MODEL_CATALOG)} models compatible with your hardware[/dim]\n")

    def pick_language(self) -> str:
        """Interactive language picker."""
        console.print("\n[bold]Select training language:[/bold]")
        langs = list(LANGUAGE_OPTIONS.items())
        for i, (code, name) in enumerate(langs, 1):
            lang_data = Path.home() / ".cache" / "autoresearch" / code / "data"
            ready = lang_data.exists() and any(lang_data.iterdir()) if lang_data.exists() else False
            status = "[green]ready[/green]" if ready else "[dim]not downloaded[/dim]"
            console.print(f"  [cyan]{i}[/cyan]. {name} [{code}] · {status}")

        choice = Prompt.ask("\nPick language", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(langs):
                return langs[idx][0]
        except ValueError:
            pass
        return "en"

    def pick_model(self) -> Optional[str]:
        """Interactive model picker."""
        hw_dict = self.hardware.to_dict()
        compatible = [m for m in MODEL_CATALOG if m.is_compatible(hw_dict)]
        if not compatible:
            console.print("[red]No compatible models found for your hardware.[/red]")
            return None

        console.print("\n[bold]Compatible models:[/bold]")
        for i, m in enumerate(compatible, 1):
            feats = []
            if m.use_moe: feats.append(f"MoE×{m.moe_num_experts}")
            if m.use_gqa: feats.append("GQA")
            if m.use_swiglu: feats.append("SwiGLU")
            feat_str = f"  [dim]({', '.join(feats)})[/dim]" if feats else ""
            console.print(f"  [cyan]{i}[/cyan]. [bold]{m.name}[/bold]  {m.param_count_millions:.1f}M params{feat_str}")

        choice = Prompt.ask("\nPick model number", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(compatible):
                return compatible[idx].name
        except ValueError:
            pass
        console.print("[red]Invalid choice[/red]")
        return None

    def run_experiment(self, model_name: Optional[str] = None, language: Optional[str] = None):
        """Start and monitor training experiments in a loop until Ctrl+C."""
        if language is None:
            language = self.pick_language()
        lang_name = LANGUAGE_OPTIONS.get(language, language)

        # Ensure language data is prepared
        lang_data_dir = Path.home() / ".cache" / "autoresearch" / language / "data"
        lang_tok_dir = Path.home() / ".cache" / "autoresearch" / language / "tokenizer"
        data_ready = (lang_data_dir.exists() and any(lang_data_dir.iterdir())
                      and (lang_tok_dir / "tokenizer.pkl").exists())
        if not data_ready:
            console.print(f"\n[yellow]Preparing {lang_name} data (first time only)...[/yellow]")
            num_shards = "3" if language != "en" else "10"
            ret = subprocess.run(
                [sys.executable, "prepare.py", "--language", language, "--num-shards", num_shards],
                check=False,
            )
            if ret.returncode != 0:
                console.print(f"[red]Failed to prepare {lang_name} data.[/red]")
                return

        if model_name is None:
            model_name = self.pick_model()
        if model_name is None:
            return

        model = get_model_by_name(model_name)
        if model is None:
            console.print(f"[red]Model '{model_name}' not found.[/red]")
            return

        hw_dict = self.hardware.to_dict()
        if not model.is_compatible(hw_dict):
            console.print(f"[red]{model_name} requires {model.min_vram_gb:.0f} GB VRAM — not compatible with your hardware.[/red]")
            return

        default_name = f"exp_{datetime.now().strftime('%m%d_%H%M')}"
        base_name = Prompt.ask("Experiment base name", default=default_name)
        time_budget_str = Prompt.ask("Time budget per cycle (seconds)", default="300")
        try:
            time_budget = max(30, int(time_budget_str))
        except ValueError:
            time_budget = 300

        console.print(f"\n[bold]Starting continuous run[/bold] — [cyan]{base_name}[/cyan] · [magenta]{lang_name}[/magenta] · [bold]{model_name}[/bold]")
        console.print(f"[dim]{time_budget}s per cycle · Press Ctrl+C anytime to stop[/dim]\n")

        cycle = 1
        while True:
            exp_name = f"{base_name}_{cycle}"
            exp_id = f"exp_{int(time.time())}"
            runner = ExperimentRunner(exp_id, model_name, exp_name, time_budget=time_budget, resume=(cycle > 1), language=language)
            self.runners[exp_id] = runner

            console.print(f"[dim]── Cycle {cycle} ──[/dim] [cyan]{exp_name}[/cyan]")
            ok, msg = runner.start()
            if not ok:
                console.print(f"[red]Failed to start: {msg}[/red]")
                break
            console.print(f"[green]Running[/green] (PID {msg}) · startup may take 30-60s on MPS\n")

            stopped_by_user = False
            try:
                with Live(console=console, refresh_per_second=2, screen=True) as live:
                    while runner.status == "running":
                        live.update(render_experiment_panel(runner))
                        time.sleep(0.5)
                    live.update(render_experiment_panel(runner))
            except KeyboardInterrupt:
                runner.stop()
                stopped_by_user = True

            # Record result (auto-keep all cycles)
            final_bpb = runner.get_final_bpb()
            if final_bpb is not None:
                record = ExperimentRecord(
                    commit=get_current_commit(),
                    val_bpb=final_bpb,
                    memory_mb=0,
                    status="keep",
                    description=f"{exp_name} ({model_name})",
                    timestamp=datetime.now().isoformat(),
                    config_snapshot=model.to_dict(),
                    metrics={"val_bpb": final_bpb, "mfu": runner.current_mfu or 0},
                )
                self.memory.add_experiment(record)
                console.print(f"\n[green]Cycle {cycle} recorded[/green] · val_bpb: [yellow]{final_bpb:.6f}[/yellow]")
            else:
                console.print(f"[dim]Cycle {cycle}: no val_bpb captured — not recorded.[/dim]")

            if stopped_by_user:
                console.print("\n[yellow]Stopped by user.[/yellow]")
                break

            cycle += 1
            console.print()

    def show_history(self):
        console.print(render_stats_panel(self.memory))
        hypothesis = self.memory.generate_hypothesis()
        console.print(Panel(
            f"[bold]Next experiment suggestion:[/bold]\n{hypothesis}",
            border_style="dim",
            padding=(0, 2),
        ))

    def run_menu(self):
        """Main interactive menu loop."""
        self.show_welcome()

        while True:
            console.print("\n[bold]What would you like to do?[/bold]")
            console.print("  [cyan]1[/cyan]. View model catalog")
            console.print("  [cyan]2[/cyan]. Run an experiment")
            console.print("  [cyan]3[/cyan]. View experiment history")
            console.print("  [cyan]4[/cyan]. Detect hardware")
            console.print("  [cyan]q[/cyan]. Quit")

            choice = Prompt.ask("\nChoice", default="2")

            if choice == "1":
                self.show_models()
            elif choice == "2":
                self.run_experiment()
            elif choice == "3":
                self.show_history()
            elif choice == "4":
                print_hardware_report(self.hardware)
            elif choice.lower() in ("q", "quit", "exit"):
                console.print("[dim]Bye.[/dim]")
                break
            else:
                console.print("[red]Unknown choice[/red]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autoresearch 2.0 Terminal Dashboard")
    parser.add_argument("--run", metavar="MODEL", help="Start an experiment directly with this model name")
    parser.add_argument("--detect", action="store_true", help="Only detect and print hardware info")
    parser.add_argument("--history", action="store_true", help="Show experiment history and exit")
    parser.add_argument("--models", action="store_true", help="List model catalog and exit")
    args = parser.parse_args()

    dash = Dashboard()

    if args.detect:
        print_hardware_report(dash.hardware)
        return

    if args.history:
        dash.show_history()
        return

    if args.models:
        dash.show_models()
        return

    if args.run:
        dash.show_welcome()
        dash.run_experiment(model_name=args.run)
        return

    # Full interactive menu
    dash.run_menu()


if __name__ == "__main__":
    main()
