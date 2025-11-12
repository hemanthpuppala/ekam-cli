"""Live foreground TUI for benchmarking (three-modal dashboard).

Renders:
- Header: suite, elapsed, CPU/GPU usage, selected params, prompts/runs
- Modal 1: Loading/Loaded <model @ endpoint>
- Modal 2: Runs: warmup x/y, run x/y, latest latency
- Modal 3: Queue: next model and completed models with status

This module is O(1) per refresh: we only read current state and one
SystemMonitor snapshot.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Dict, List, Set

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from .progress_display import ProgressDisplay, ProgressPhase
from ...system_specs.monitor import SystemMonitor
from ...cli.tui_manager import tui


class BenchmarkLiveUI:
    """Foreground live UI manager for benchmarks."""

    def __init__(
        self,
        progress: ProgressDisplay,
        suite_name: str,
        config_params: Dict[str, any],
        models: List[str],
        endpoints: Dict[str, List[str]],
        num_warmup: int,
        num_runs: int,
        failed_models_ref: Optional[Set[str]] = None,
    ) -> None:
        self.progress = progress
        self.suite_name = suite_name
        self.params = config_params or {}
        self.models = models
        self.endpoints = endpoints
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        self.failed_models_ref = failed_models_ref or set()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._monitor = SystemMonitor()
        self._start_time = time.time()

    def _display_name(self, model_id: str) -> str:
        """Extract a clean display name from model identifier.

        - quantized:gguf:/path/to/model.gguf -> model.gguf
        - /abs/path/to/model.gguf -> model.gguf
        - plain-name or hf id -> last path component or as-is
        """
        try:
            if model_id.startswith("quantized:"):
                parts = model_id.split(":", 2)
                if len(parts) == 3:
                    from pathlib import Path as _P
                    return _P(parts[2]).name
            if "/" in model_id or "\\" in model_id:
                from pathlib import Path as _P
                return _P(model_id).name
        except Exception:
            pass
        return model_id

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="benchmark-live-ui", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # Render helpers
    def _render_header(self):
        elapsed = int(time.time() - self._start_time)
        mins, secs = divmod(elapsed, 60)

        # Prefer suite-reported metrics for realtime header; fallback to instant snapshot
        state = self.progress.state
        lm = getattr(state, 'latest_metrics', {}) or {}
        cpu_pct = None
        mem_mb = None
        gpu_pct = None
        vram_mb = None
        try:
            if 'cpu_percent_avg' in lm:
                cpu_pct = float(lm['cpu_percent_avg'])
            elif 'cpu_percent_peak' in lm:
                cpu_pct = float(lm['cpu_percent_peak'])
        except Exception:
            cpu_pct = None
        try:
            if 'memory_mb_avg' in lm:
                mem_mb = float(lm['memory_mb_avg'])
            elif 'memory_mb_peak' in lm:
                mem_mb = float(lm['memory_mb_peak'])
        except Exception:
            mem_mb = None
        try:
            if 'gpu_percent_avg' in lm:
                gpu_pct = float(lm['gpu_percent_avg'])
            elif 'gpu_percent_peak' in lm:
                gpu_pct = float(lm['gpu_percent_peak'])
        except Exception:
            gpu_pct = None
        try:
            if 'vram_mb_avg' in lm:
                vram_mb = float(lm['vram_mb_avg'])
            elif 'vram_mb_peak' in lm:
                vram_mb = float(lm['vram_mb_peak'])
        except Exception:
            vram_mb = None

        # Fallback to instantaneous snapshot if suite metrics not present yet
        if cpu_pct is None or (gpu_pct is None and vram_mb is None) or mem_mb is None:
            snap = self._monitor.get_snapshot()
            try:
                if cpu_pct is None and snap and getattr(snap, 'cpu', None):
                    cpu_pct = float(getattr(snap.cpu, 'percent', 0.0) or 0.0)
            except Exception:
                pass
            try:
                if (gpu_pct is None or vram_mb is None) and snap and snap.gpus:
                    g = snap.gpus[0]
                    if gpu_pct is None:
                        gpu_pct = float(getattr(g, 'utilization', 0.0) or 0.0)
                    if vram_mb is None:
                        vram_mb = float(getattr(g, 'memory_used_mb', 0.0) or 0.0)
            except Exception:
                pass
            try:
                if mem_mb is None and snap and getattr(snap, 'memory', None):
                    mem_mb = float(getattr(snap.memory, 'used_mb', 0.0) or 0.0)
            except Exception:
                pass
        # Normalize defaults
        cpu_pct = cpu_pct if cpu_pct is not None else 0.0
        gpu_pct = gpu_pct if gpu_pct is not None else 0.0
        vram_mb = vram_mb if vram_mb is not None else 0.0
        mem_mb = mem_mb if mem_mb is not None else 0.0

        kv = []
        for k in ["temperature", "top_p", "top_k", "max_tokens"]:
            if k in self.params:
                kv.append(f"{k}={self.params[k]}")
        params_str = ", ".join(kv) if kv else ""

        runs_info = f"warmup {self.num_warmup}, runs {self.num_runs}"

        # Progress and ETA
        progress_pct = 0.0
        eta_str = "—"
        phase = ""
        try:
            ps = self.progress.state
            progress_pct = ps.overall_progress
            rem = ps.estimated_remaining
            if rem:
                rmins = int(rem.total_seconds() // 60)
                rsecs = int(rem.total_seconds() % 60)
                eta_str = f"~{rmins}m {rsecs}s"
            phase = ps.phase.value.replace("_", " ").title()
        except Exception:
            pass

        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(style="cyan", justify="right")
        tbl.add_column(style="white")
        tbl.add_row("Suite:", f"{self.suite_name}")
        tbl.add_row("Elapsed:", f"{mins}m {secs}s")
        tbl.add_row("Phase:", phase or "—")
        tbl.add_row("CPU:", f"{cpu_pct:.1f}% • Mem {mem_mb:.0f} MB")
        tbl.add_row("GPU:", f"{gpu_pct:.1f}% • {vram_mb:.0f} MB")
        if params_str:
            tbl.add_row("Params:", params_str)
        tbl.add_row("Runs:", runs_info)
        tbl.add_row("Progress:", f"{progress_pct:.1f}% • ETA {eta_str}")
        
        return Panel(tbl, title="[bold cyan]Benchmark[/bold cyan]", border_style="cyan")

    def _render_modal_loading(self):
        state = self.progress.state
        model_raw = state.current_model or (self.models[state.current_model_index-1] if 0 < state.current_model_index <= len(self.models) else "—")
        model = self._display_name(model_raw)
        phase = state.phase
        endpoint = state.current_endpoint or (self.endpoints.get(model, ["N/A"]) or ["N/A"])[0]
        status = "Loading" if phase == ProgressPhase.LOADING_MODEL else "Loaded"

        # Header block
        header_tbl = Table.grid(padding=(0, 1))
        header_tbl.add_column(style="cyan", justify="right")
        header_tbl.add_column(style="white")
        header_tbl.add_row("Status:", f"{status}")
        header_tbl.add_row("Model:", f"{model}")
        header_tbl.add_row("Endpoint:", f"{endpoint}")

        # Parameter boxes
        boxes_tbl = Table.grid(expand=True)
        param_defs = [
            ("Warmup runs", str(self.num_warmup), "0-5"),
            ("Temperature", str(self.params.get("temperature", "-")), "0.0-1.0"),
            ("Top-p", str(self.params.get("top_p", "-")), "0.0-1.0"),
            ("Top-k", str(self.params.get("top_k", "-")), "0-200"),
            ("Max tokens", str(self.params.get("max_tokens", "-")), "1-4096"),
            ("n_ctx", str(self.params.get("n_ctx", "-")), "256-32768"),
        ]
        row_panels = []
        for label, cur, rng in param_defs:
            inner = Table.grid(padding=(0, 1))
            inner.add_column(style="white")
            inner.add_row(f"Currently: {cur}")
            inner.add_row(f"Range: {rng}")
            row_panels.append(Panel(inner, title=f"{label}", border_style="cyan"))
        for i in range(0, len(row_panels), 3):
            boxes_tbl.add_row(*row_panels[i:i+3])

        content = Table.grid(expand=True)
        content.add_row(header_tbl)
        content.add_row("")
        content.add_row(boxes_tbl)

        return Panel(content, title="Model", border_style="cyan")

    def _render_modal_runs(self):
        state = self.progress.state
        run = max(0, state.current_run)
        latest_latency = None
        try:
            if "latency_ms" in state.latest_metrics:
                latest_latency = float(state.latest_metrics["latency_ms"])
        except Exception:
            pass

        warmup_done = min(run, self.num_warmup)
        counted_done = max(0, run - self.num_warmup)
        # Derive total counted runs from progress state's total_runs
        total_counted = 0
        try:
            total_counted = max(0, self.progress.state.total_runs - self.num_warmup)
        except Exception:
            total_counted = max(0, self.num_runs)

        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(style="cyan", justify="right")
        tbl.add_column(style="white")
        # Per-model counters when available, otherwise show dashes
        try:
            warm_row = f"{warmup_done}/{self.num_warmup}" if self.num_warmup else "—"
            run_row = f"{counted_done}/{total_counted}" if total_counted else "—"
        except Exception:
            warm_row = "—"; run_row = "—"
        tbl.add_row("Warmup:", warm_row)
        tbl.add_row("Run:", run_row)
        if latest_latency is not None:
            tbl.add_row("Latency:", f"{latest_latency:.2f} ms")
        # Enhanced live metrics if available
        state = self.progress.state
        try:
            if "ttft_ms" in state.latest_metrics:
                tbl.add_row("TTFT:", f"{float(state.latest_metrics['ttft_ms']):.2f} ms")
            if "tokens_per_sec" in state.latest_metrics:
                tbl.add_row("Tokens/s:", f"{float(state.latest_metrics['tokens_per_sec']):.2f}")
            if "itl_ms" in state.latest_metrics:
                tbl.add_row("ITL:", f"{float(state.latest_metrics['itl_ms']):.2f} ms")
            if "decode_latency_ms" in state.latest_metrics:
                tbl.add_row("Decode:", f"{float(state.latest_metrics['decode_latency_ms']):.2f} ms")
            # Suite-specific common metrics (Resources/Stress/Quality)
            for key, label in [
                ("cpu_percent_peak", "CPU Peak %"),
                ("cpu_percent_avg", "CPU Avg %"),
                ("memory_mb_peak", "Mem Peak MB"),
                ("memory_mb_avg", "Mem Avg MB"),
                ("vram_mb_peak", "VRAM Peak MB"),
                ("vram_mb_avg", "VRAM Avg MB"),
                ("gpu_percent_peak", "GPU Peak %"),
                ("gpu_percent_avg", "GPU Avg %"),
                ("cpu_temp_celsius_peak", "CPU Temp °C"),
                ("gpu_temp_celsius_peak", "GPU Temp °C"),
            ]:
                if key in state.latest_metrics:
                    try:
                        val = float(state.latest_metrics[key])
                        if key.endswith("percent"):
                            tbl.add_row(label + ":", f"{val:.1f}%")
                        elif key.endswith("MB") or "mb" in key:
                            tbl.add_row(label + ":", f"{val:.0f} MB")
                        elif "Temp" in label:
                            tbl.add_row(label + ":", f"{val:.1f}")
                        else:
                            tbl.add_row(label + ":", f"{val:.2f}")
                    except Exception:
                        tbl.add_row(label + ":", str(state.latest_metrics[key]))
        except Exception:
            pass
        # Always show overall runs as fallback for suites not reporting per-run
        try:
            overall = f"{state.overall_completed_runs}/{state.total_runs_all}"
            tbl.add_row("Overall:", overall)
        except Exception:
            pass

        # Run history: display all recorded runs for current model
        if not hasattr(self, "_run_history"):
            self._run_history = {}
            self._recorded_keys = set()
        try:
            model_raw = state.current_model or (self.models[state.current_model_index-1] if 0 < state.current_model_index <= len(self.models) else None)
            if model_raw:
                if latest_latency is not None and run > 0:
                    phase = "W" if run <= self.num_warmup else "R"
                    key = (model_raw, run, phase)
                    if key not in self._recorded_keys:
                        self._recorded_keys.add(key)
                        label_num = run if phase == 'W' else (run - self.num_warmup)
                        line = f"{phase}{label_num}: {latest_latency:.2f} ms"
                        self._run_history.setdefault(model_raw, []).append(line)
                lines = self._run_history.get(model_raw, [])
                if lines:
                    hist_tbl = Table.grid(padding=(0, 0))
                    hist_tbl.add_column(style="white")
                    for ln in lines:
                        hist_tbl.add_row(ln)
                    tbl.add_row("", "")
                    tbl.add_row("[bold]Runs:[/bold]", "")
                    tbl.add_row("", hist_tbl)
        except Exception:
            pass

        return Panel(tbl, title="Runs", border_style="green")

    def _render_modal_queue(self):
        state = self.progress.state
        idx = max(1, state.current_model_index)
        next_model_raw = self.models[idx] if idx < len(self.models) else None
        next_model = self._display_name(next_model_raw) if next_model_raw else None
        completed_raw = self.models[:idx-1] if idx > 1 else []
        completed = [self._display_name(m) for m in completed_raw]

        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(style="cyan", justify="right")
        tbl.add_column(style="white")
        tbl.add_row("Next:", next_model or "None")

        if completed:
            status_lines = []
            for m in completed:
                st_failed = m in self.failed_models_ref
                badge = "[red]failed[/red]" if st_failed else "[green]done[/green]"
                status_lines.append(f"• {m} [{badge}]")
            tbl.add_row("Completed:", "\n".join(status_lines))
        else:
            tbl.add_row("Completed:", "—")

        # Overall counters and ETA
        try:
            overall = f"{state.overall_completed_runs}/{state.total_runs_all}"
        except Exception:
            overall = "—"
        tbl.add_row("Overall:", overall)
        tbl.add_row("Success:", str(getattr(state, 'total_success', 0)))
        tbl.add_row("Failed:", str(getattr(state, 'total_failed', 0)))
        # Simple ETA reused from header estimated_remaining
        try:
            rem = state.estimated_remaining
            if rem:
                rmins = int(rem.total_seconds() // 60)
                rsecs = int(rem.total_seconds() % 60)
                tbl.add_row("ETA:", f"~{rmins}m {rsecs}s")
        except Exception:
            pass

        return Panel(tbl, title="Queue", border_style="blue")

    def _compose_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=7),
            Layout(name="body")
        )
        body = Layout()
        body.split_row(
            Layout(name="modal1"),
            Layout(name="modal2"),
            Layout(name="modal3"),
        )
        layout["body"].update(body)

        layout["header"].update(self._render_header())
        body["modal1"].update(self._render_modal_loading())
        body["modal2"].update(self._render_modal_runs())
        body["modal3"].update(self._render_modal_queue())
        return layout

    def _run_loop(self):
        with Live(self._compose_layout(), console=tui.console, refresh_per_second=4) as live:
            while not self._stop_event.is_set():
                try:
                    live.update(self._compose_layout())
                except Exception:
                    pass
                time.sleep(0.25)
