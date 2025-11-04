"""
Real-time progress display for benchmark execution.

Provides:
- Live progress tracking with Rich progress bars
- Estimated time remaining
- Current status and metrics
- Model/run/suite information
- Resource usage updates
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskProgressColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from loguru import logger


class ProgressPhase(str, Enum):
    """Benchmark execution phases."""

    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    WARMUP = "warmup"
    RUNNING = "running"
    AGGREGATING = "aggregating"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ProgressState:
    """
    State of benchmark progress.

    Attributes:
        phase: Current execution phase
        current_model: Model being benchmarked
        current_model_index: Index of current model (1-based)
        total_models: Total number of models
        current_run: Current run number (1-based)
        total_runs: Total runs per model
        current_endpoint: Current endpoint being tested
        start_time: Benchmark start time
        estimated_end_time: Estimated completion time
        latest_metrics: Latest run metrics
        error_message: Error message if failed
    """

    phase: ProgressPhase = ProgressPhase.INITIALIZING
    current_model: Optional[str] = None
    current_model_index: int = 0
    total_models: int = 0
    current_run: int = 0
    total_runs: int = 0
    current_endpoint: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    estimated_end_time: Optional[datetime] = None
    latest_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    @property
    def overall_progress(self) -> float:
        """Calculate overall progress percentage (0-100)."""
        if self.total_models == 0 or self.total_runs == 0:
            return 0.0

        # Calculate completed work
        completed_models = self.current_model_index - 1
        completed_runs = completed_models * self.total_runs + (self.current_run - 1)
        total_work = self.total_models * self.total_runs

        if total_work == 0:
            return 0.0

        return (completed_runs / total_work) * 100.0

    @property
    def elapsed_time(self) -> timedelta:
        """Time elapsed since start."""
        return datetime.now() - self.start_time

    @property
    def estimated_remaining(self) -> Optional[timedelta]:
        """Estimate time remaining."""
        if self.estimated_end_time:
            remaining = self.estimated_end_time - datetime.now()
            return remaining if remaining.total_seconds() > 0 else timedelta(0)

        # Simple estimation based on progress
        progress = self.overall_progress
        if progress <= 0:
            return None

        elapsed = self.elapsed_time.total_seconds()
        total_estimated = elapsed / (progress / 100.0)
        remaining_seconds = total_estimated - elapsed

        return timedelta(seconds=max(0, remaining_seconds))


class ProgressDisplay:
    """
    Real-time progress display for benchmarks.

    Uses Rich library to show:
    - Progress bars for overall and current model
    - Current status and phase
    - Live metrics
    - Estimated time remaining
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize progress display.

        Args:
            console: Rich Console instance (creates new one if None)
        """
        self.console = console or Console()
        self.state = ProgressState()

        # Rich Progress instance
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console
        )

        # Task IDs
        self._overall_task_id: Optional[int] = None
        self._current_model_task_id: Optional[int] = None
        self._current_run_task_id: Optional[int] = None

        # Live display
        self._live: Optional[Live] = None
        self._is_running = False

        logger.debug("ProgressDisplay initialized")

    def start(
        self,
        total_models: int,
        total_runs_per_model: int,
        suite_name: str = "Benchmark"
    ) -> None:
        """
        Start progress display.

        Args:
            total_models: Total number of models to benchmark
            total_runs_per_model: Number of runs per model
            suite_name: Name of benchmark suite
        """
        self.state.total_models = total_models
        self.state.total_runs = total_runs_per_model
        self.state.start_time = datetime.now()
        self.state.phase = ProgressPhase.INITIALIZING

        # Create progress tasks
        total_runs = total_models * total_runs_per_model

        self._overall_task_id = self.progress.add_task(
            f"[cyan]{suite_name} - Overall Progress",
            total=total_runs
        )

        self._current_model_task_id = self.progress.add_task(
            "[green]Current Model",
            total=total_runs_per_model,
            visible=False
        )

        self._is_running = True

        logger.info(
            f"Progress tracking started: {total_models} models × {total_runs_per_model} runs = {total_runs} total"
        )

    def update_phase(self, phase: ProgressPhase, message: Optional[str] = None) -> None:
        """
        Update current execution phase.

        Args:
            phase: New phase
            message: Optional status message
        """
        self.state.phase = phase

        if message:
            logger.info(f"[{phase.value}] {message}")

    def update_model(self, model_id: str, model_index: int, endpoint: str = "N/A") -> None:
        """
        Update current model being benchmarked.

        Args:
            model_id: Model identifier
            model_index: Index of model (1-based)
            endpoint: Endpoint being tested (default: "N/A" for suites without endpoints)
        """
        self.state.current_model = model_id
        self.state.current_model_index = model_index
        self.state.current_endpoint = endpoint
        self.state.current_run = 0
        self.state.phase = ProgressPhase.LOADING_MODEL

        # Update progress bar
        if self._current_model_task_id is not None:
            self.progress.update(
                self._current_model_task_id,
                description=f"[green]Model {model_index}/{self.state.total_models}: {model_id}",
                completed=0,
                visible=True
            )

        logger.info(f"Benchmarking model {model_index}/{self.state.total_models}: {model_id} @ {endpoint}")

    def update_run(self, run_number: int, is_warmup: bool = False) -> None:
        """
        Update current run number.

        Args:
            run_number: Run number (1-based)
            is_warmup: Whether this is a warmup run
        """
        self.state.current_run = run_number
        self.state.phase = ProgressPhase.WARMUP if is_warmup else ProgressPhase.RUNNING

        # Update current model progress
        if self._current_model_task_id is not None:
            self.progress.update(
                self._current_model_task_id,
                completed=run_number
            )

        # Update overall progress
        if self._overall_task_id is not None:
            completed_models = self.state.current_model_index - 1
            total_completed = completed_models * self.state.total_runs + run_number

            self.progress.update(
                self._overall_task_id,
                completed=total_completed
            )

        run_type = "warmup" if is_warmup else "benchmark"
        logger.debug(f"Starting {run_type} run {run_number}/{self.state.total_runs}")

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update latest run metrics.

        Args:
            metrics: Dict of metric_name -> value
        """
        self.state.latest_metrics.update(metrics)

        # Log key metrics
        if "latency_ms" in metrics:
            logger.debug(f"Run completed: {metrics['latency_ms']:.2f}ms latency")

    def stop(self) -> None:
        """Stop progress display and mark as complete."""
        self.mark_complete()

    def mark_complete(self) -> None:
        """Mark benchmark as complete."""
        self.state.phase = ProgressPhase.COMPLETE

        # Complete all progress bars
        if self._overall_task_id is not None:
            self.progress.update(self._overall_task_id, completed=self.state.total_models * self.state.total_runs)

        if self._current_model_task_id is not None:
            self.progress.update(self._current_model_task_id, completed=self.state.total_runs)

        self._is_running = False

        logger.info(f"Benchmark complete! Total time: {self.state.elapsed_time}")

    def mark_failed(self, error_message: str) -> None:
        """
        Mark benchmark as failed.

        Args:
            error_message: Error description
        """
        self.state.phase = ProgressPhase.FAILED
        self.state.error_message = error_message
        self._is_running = False

        logger.error(f"Benchmark failed: {error_message}")

    def render_status_panel(self) -> Panel:
        """
        Render current status as a Rich Panel.

        Returns:
            Rich Panel with status information
        """
        # Build status table
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white")

        # Phase
        phase_emoji = {
            ProgressPhase.INITIALIZING: "⚙️",
            ProgressPhase.LOADING_MODEL: "📥",
            ProgressPhase.WARMUP: "🔥",
            ProgressPhase.RUNNING: "🏃",
            ProgressPhase.AGGREGATING: "📊",
            ProgressPhase.COMPLETE: "✅",
            ProgressPhase.FAILED: "❌"
        }
        emoji = phase_emoji.get(self.state.phase, "")
        table.add_row("Phase:", f"{emoji} {self.state.phase.value.replace('_', ' ').title()}")

        # Current model
        if self.state.current_model:
            table.add_row(
                "Model:",
                f"{self.state.current_model} ({self.state.current_model_index}/{self.state.total_models})"
            )

        # Current endpoint
        if self.state.current_endpoint:
            table.add_row("Endpoint:", self.state.current_endpoint)

        # Current run
        if self.state.current_run > 0:
            table.add_row("Run:", f"{self.state.current_run}/{self.state.total_runs}")

        # Progress percentage
        table.add_row("Progress:", f"{self.state.overall_progress:.1f}%")

        # Time information
        elapsed = self.state.elapsed_time
        table.add_row("Elapsed:", self._format_timedelta(elapsed))

        remaining = self.state.estimated_remaining
        if remaining:
            table.add_row("Remaining:", f"~{self._format_timedelta(remaining)}")

        # Latest metrics
        if self.state.latest_metrics:
            table.add_row("", "")  # Separator
            table.add_row("[bold]Latest Metrics:[/bold]", "")

            if "latency_ms" in self.state.latest_metrics:
                table.add_row("  Latency:", f"{self.state.latest_metrics['latency_ms']:.2f} ms")

            if "tokens_per_second" in self.state.latest_metrics:
                table.add_row("  Throughput:", f"{self.state.latest_metrics['tokens_per_second']:.2f} tok/s")

            if "memory_peak_mb" in self.state.latest_metrics:
                table.add_row("  Memory:", f"{self.state.latest_metrics['memory_peak_mb']:.1f} MB")

        # Error message
        if self.state.error_message:
            table.add_row("", "")
            table.add_row("[bold red]Error:[/bold red]", f"[red]{self.state.error_message}[/red]")

        return Panel(table, title="[bold cyan]Benchmark Status[/bold cyan]", border_style="cyan")

    def print_progress(self) -> None:
        """Print current progress to console."""
        self.console.print(self.render_status_panel())

    def get_progress_context(self):
        """
        Get Rich Progress context manager for use in with statement.

        Returns:
            Rich Progress context manager

        Example:
            ```python
            with display.get_progress_context():
                # Run benchmark
                for model in models:
                    display.update_model(model, i)
                    # ... run benchmark ...
            ```
        """
        return self.progress

    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta as human-readable string."""
        total_seconds = int(td.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProgressDisplay("
            f"phase={self.state.phase.value}, "
            f"progress={self.state.overall_progress:.1f}%, "
            f"running={self._is_running}"
            f")"
        )
