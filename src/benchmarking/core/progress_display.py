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
    RETRYING = "retrying"


class RunStatus(str, Enum):
    """Status of individual run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    RETRY_SUCCESS = "retry_success"
    RETRY_FAILED = "retry_failed"


class ModelStatus(str, Enum):
    """Status of model in queue."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class RunInfo:
    """Information about a single run.

    O(1) time and space complexity for all operations.
    """
    run_number: int
    status: RunStatus = RunStatus.PENDING
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    retry_count: int = 0
    timestamp: Optional[datetime] = None

    def get_badge(self) -> str:
        """Get text badge for status."""
        badges = {
            RunStatus.PENDING: "[dim]⏸ PENDING[/dim]",
            RunStatus.RUNNING: "[yellow]▶ RUNNING[/yellow]",
            RunStatus.SUCCESS: "[green]✓ OK[/green]",
            RunStatus.FAILED: "[red]✗ FAIL[/red]",
            RunStatus.RETRYING: "[cyan]↻ RETRY[/cyan]",
            RunStatus.RETRY_SUCCESS: "[green]✓ OK[/green][dim](retry)[/dim]",
            RunStatus.RETRY_FAILED: "[red]✗ FAIL[/red][dim](retry)[/dim]",
        }
        return badges.get(self.status, "[dim]?[/dim]")

    def get_symbol(self) -> str:
        """Get colored symbol for status."""
        symbols = {
            RunStatus.PENDING: "[dim]○[/dim]",
            RunStatus.RUNNING: "[yellow]●[/yellow]",
            RunStatus.SUCCESS: "[green]●[/green]",
            RunStatus.FAILED: "[red]●[/red]",
            RunStatus.RETRYING: "[cyan]◐[/cyan]",
            RunStatus.RETRY_SUCCESS: "[green]◉[/green]",
            RunStatus.RETRY_FAILED: "[red]◉[/red]",
        }
        return symbols.get(self.status, "[dim]○[/dim]")


@dataclass
class ModelInfo:
    """Information about a model in queue.

    O(1) lookups via dictionary-based run tracking.
    """
    model_id: str
    model_index: int
    status: ModelStatus = ModelStatus.QUEUED
    runs: Dict[int, RunInfo] = field(default_factory=dict)  # run_number -> RunInfo
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    retry_runs: int = 0

    def get_progress_bar(self, width: int = 20) -> str:
        """Get text-based progress bar."""
        if self.total_runs == 0:
            return "[dim]" + "─" * width + "[/dim]"

        completed = self.completed_runs
        total = self.total_runs
        filled = int((completed / total) * width)
        bar = "█" * filled + "░" * (width - filled)

        # Color based on status
        if self.status == ModelStatus.FAILED:
            return f"[red]{bar}[/red]"
        elif self.status == ModelStatus.COMPLETED:
            return f"[green]{bar}[/green]"
        elif self.status == ModelStatus.RUNNING:
            return f"[yellow]{bar}[/yellow]"
        else:
            return f"[dim]{bar}[/dim]"

    def get_summary(self) -> str:
        """Get summary string."""
        success = self.completed_runs - self.failed_runs
        return f"{self.completed_runs}/{self.total_runs} runs | {success} ✓ | {self.failed_runs} ✗"


@dataclass
class ProgressState:
    """
    State of benchmark progress with O(1) time/space tracking.

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
        models: Dictionary of model info (model_id -> ModelInfo) for O(1) lookup
        failed_runs_queue: List of (model_id, run_number) for batch retry
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
    # Global run/counter tracking
    overall_completed_runs: int = 0
    total_runs_all: int = 0
    total_success: int = 0
    total_failed: int = 0
    # Enhanced tracking structures (O(1) lookups)
    models: Dict[str, ModelInfo] = field(default_factory=dict)  # model_id -> ModelInfo
    failed_runs_queue: list = field(default_factory=list)  # List of (model_id, run_number) tuples
    model_order: list = field(default_factory=list)  # Ordered list of model_ids

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
        # Initialize overall counters
        self.state.total_runs_all = total_runs
        self.state.overall_completed_runs = 0
        self.state.total_success = 0
        self.state.total_failed = 0

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
            self.state.overall_completed_runs = total_completed

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

    def report_failure(self) -> None:
        """Increment failed run counter."""
        try:
            self.state.total_failed += 1
        except Exception:
            pass

    def report_success(self) -> None:
        """Increment successful run counter."""
        try:
            self.state.total_success += 1
        except Exception:
            pass

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

    # ========================================================================
    # Enhanced Progress Tracking Methods (O(1) time/space complexity)
    # ========================================================================

    def initialize_models(self, model_ids: list[str], total_runs_per_model: int) -> None:
        """
        Initialize model tracking structures.

        O(n) time where n = number of models, but called only once.
        All subsequent operations are O(1).

        Args:
            model_ids: List of model identifiers
            total_runs_per_model: Number of runs per model
        """
        self.state.model_order = model_ids
        for idx, model_id in enumerate(model_ids, start=1):
            model_info = ModelInfo(
                model_id=model_id,
                model_index=idx,
                status=ModelStatus.QUEUED,
                total_runs=total_runs_per_model
            )
            # Pre-initialize all runs as PENDING (O(1) per run)
            for run_num in range(1, total_runs_per_model + 1):
                model_info.runs[run_num] = RunInfo(run_number=run_num)

            self.state.models[model_id] = model_info

        logger.debug(f"Initialized tracking for {len(model_ids)} models with {total_runs_per_model} runs each")

    def start_model(self, model_id: str) -> None:
        """
        Mark model as currently running.

        O(1) time complexity.

        Args:
            model_id: Model identifier
        """
        if model_id in self.state.models:
            self.state.models[model_id].status = ModelStatus.RUNNING
            logger.debug(f"Model {model_id} started")

    def start_run(self, model_id: str, run_number: int) -> None:
        """
        Mark a specific run as started.

        O(1) time complexity via dictionary lookup.

        Args:
            model_id: Model identifier
            run_number: Run number (1-based)
        """
        if model_id in self.state.models and run_number in self.state.models[model_id].runs:
            self.state.models[model_id].runs[run_number].status = RunStatus.RUNNING
            self.state.models[model_id].runs[run_number].timestamp = datetime.now()

    def record_run_success(
        self,
        model_id: str,
        run_number: int,
        latency_ms: Optional[float] = None,
        is_retry: bool = False
    ) -> None:
        """
        Record successful run completion.

        O(1) time complexity.

        Args:
            model_id: Model identifier
            run_number: Run number (1-based)
            latency_ms: Run latency in milliseconds
            is_retry: Whether this was a retry attempt
        """
        if model_id not in self.state.models or run_number not in self.state.models[model_id].runs:
            return

        run_info = self.state.models[model_id].runs[run_number]
        run_info.status = RunStatus.RETRY_SUCCESS if is_retry else RunStatus.SUCCESS
        run_info.latency_ms = latency_ms
        run_info.timestamp = datetime.now()

        # Update model counters
        model_info = self.state.models[model_id]
        model_info.completed_runs += 1
        if is_retry:
            model_info.retry_runs += 1

        # Update global counters
        self.state.total_success += 1
        self.state.overall_completed_runs += 1

        logger.debug(f"Run {run_number} for {model_id} completed: {latency_ms:.2f}ms (retry={is_retry})")

    def record_run_failure(
        self,
        model_id: str,
        run_number: int,
        error: str,
        is_retry: bool = False
    ) -> None:
        """
        Record failed run.

        O(1) time complexity.

        Args:
            model_id: Model identifier
            run_number: Run number (1-based)
            error: Error message
            is_retry: Whether this was a retry attempt
        """
        if model_id not in self.state.models or run_number not in self.state.models[model_id].runs:
            return

        run_info = self.state.models[model_id].runs[run_number]
        run_info.status = RunStatus.RETRY_FAILED if is_retry else RunStatus.FAILED
        run_info.error = error
        run_info.timestamp = datetime.now()

        # Update model counters
        model_info = self.state.models[model_id]
        model_info.completed_runs += 1
        model_info.failed_runs += 1

        # Update global counters
        self.state.total_failed += 1
        self.state.overall_completed_runs += 1

        # Add to retry queue if this is first failure (not a retry)
        if not is_retry:
            self.state.failed_runs_queue.append((model_id, run_number))
            logger.debug(f"Run {run_number} for {model_id} failed (queued for retry): {error}")
        else:
            logger.warning(f"Run {run_number} for {model_id} failed after retry: {error}")

    def complete_model(self, model_id: str) -> None:
        """
        Mark model as completed.

        O(1) time complexity.

        Args:
            model_id: Model identifier
        """
        if model_id in self.state.models:
            model_info = self.state.models[model_id]
            # Determine final status based on failures
            if model_info.failed_runs > 0:
                model_info.status = ModelStatus.FAILED
            else:
                model_info.status = ModelStatus.COMPLETED

            logger.debug(f"Model {model_id} completed with status: {model_info.status.value}")

    def get_failed_runs(self) -> list:
        """
        Get list of failed runs for batch retry.

        O(1) time complexity (returns reference to existing list).

        Returns:
            List of (model_id, run_number) tuples
        """
        return self.state.failed_runs_queue

    def clear_retry_queue(self) -> None:
        """
        Clear the retry queue after processing.

        O(1) time complexity (list clear).
        """
        self.state.failed_runs_queue.clear()
        logger.debug("Retry queue cleared")

    def render_run_status_panel(self, model_id: str) -> Panel:
        """
        Render detailed run-by-run status for a model.

        O(n) where n = number of runs for the model.

        Args:
            model_id: Model identifier

        Returns:
            Rich Panel with run status table
        """
        if model_id not in self.state.models:
            return Panel("[red]Model not found[/red]", title="Run Status")

        model_info = self.state.models[model_id]

        # Create table with columns
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Run", style="dim", width=4)
        table.add_column("Status", width=18)
        table.add_column("Symbol", width=3)
        table.add_column("Latency", width=12)
        table.add_column("Progress", width=22)

        # Add rows for each run
        for run_num in sorted(model_info.runs.keys()):
            run_info = model_info.runs[run_num]

            # Format latency
            latency_str = f"{run_info.latency_ms:.2f}ms" if run_info.latency_ms else "-"

            # Create mini progress bar for this run
            if run_info.status == RunStatus.SUCCESS or run_info.status == RunStatus.RETRY_SUCCESS:
                progress_bar = "[green]" + "█" * 10 + "[/green]"
            elif run_info.status == RunStatus.FAILED or run_info.status == RunStatus.RETRY_FAILED:
                progress_bar = "[red]" + "█" * 10 + "[/red]"
            elif run_info.status == RunStatus.RUNNING:
                progress_bar = "[yellow]" + "█" * 5 + "░" * 5 + "[/yellow]"
            else:
                progress_bar = "[dim]" + "░" * 10 + "[/dim]"

            table.add_row(
                f"{run_num:2d}",
                run_info.get_badge(),
                run_info.get_symbol(),
                latency_str,
                progress_bar
            )

        # Add summary row
        success_count = sum(1 for r in model_info.runs.values()
                           if r.status in [RunStatus.SUCCESS, RunStatus.RETRY_SUCCESS])
        fail_count = sum(1 for r in model_info.runs.values()
                        if r.status in [RunStatus.FAILED, RunStatus.RETRY_FAILED])

        table.add_row("", "", "", "", "")  # Separator
        table.add_row(
            "[bold]Total[/bold]",
            f"[green]{success_count} ✓[/green] [red]{fail_count} ✗[/red]",
            "",
            "",
            model_info.get_progress_bar(width=10)
        )

        title = f"[bold cyan]Run Status: {model_info.model_id.split('/')[-1][:40]}[/bold cyan]"
        return Panel(table, title=title, border_style="cyan")

    def render_queue_modal(self) -> Panel:
        """
        Render queue status modal showing all models.

        O(n) where n = number of models.

        Returns:
            Rich Panel with queue status
        """
        table = Table(show_header=True, header_style="bold yellow", box=None)
        table.add_column("#", style="dim", width=3)
        table.add_column("Model", width=40)
        table.add_column("Status", width=12)
        table.add_column("Progress", width=22)
        table.add_column("Summary", width=28)

        # Find current, next, completed, failed models
        current_model_id = None
        next_model_id = None
        completed_models = []
        failed_models = []

        for model_id in self.state.model_order:
            model_info = self.state.models[model_id]

            if model_info.status == ModelStatus.RUNNING:
                current_model_id = model_id
            elif model_info.status == ModelStatus.QUEUED and next_model_id is None:
                next_model_id = model_id
            elif model_info.status == ModelStatus.COMPLETED:
                completed_models.append(model_id)
            elif model_info.status == ModelStatus.FAILED:
                failed_models.append(model_id)

        # Add header info
        if current_model_id:
            table.add_row(
                "→",
                "[bold yellow]CURRENT:[/bold yellow] " + current_model_id.split('/')[-1][:35],
                "",
                "",
                ""
            )

        if next_model_id:
            table.add_row(
                "↓",
                "[bold cyan]NEXT:[/bold cyan] " + next_model_id.split('/')[-1][:35],
                "",
                "",
                ""
            )

        table.add_row("", "", "", "", "")  # Separator

        # Add all models
        for model_id in self.state.model_order:
            model_info = self.state.models[model_id]

            # Truncate model name
            model_name = model_info.model_id.split('/')[-1][:38]

            # Status badge
            status_badge = {
                ModelStatus.QUEUED: "[dim]⏸ QUEUED[/dim]",
                ModelStatus.RUNNING: "[yellow]▶ RUNNING[/yellow]",
                ModelStatus.COMPLETED: "[green]✓ DONE[/green]",
                ModelStatus.FAILED: "[red]✗ FAILED[/red]",
                ModelStatus.RETRYING: "[cyan]↻ RETRY[/cyan]",
            }.get(model_info.status, "[dim]?[/dim]")

            table.add_row(
                str(model_info.model_index),
                model_name,
                status_badge,
                model_info.get_progress_bar(width=10),
                model_info.get_summary()
            )

        # Add summary footer
        table.add_row("", "", "", "", "")  # Separator
        summary_text = (
            f"[bold]Total:[/bold] {len(self.state.model_order)} | "
            f"[green]Done: {len(completed_models)}[/green] | "
            f"[red]Failed: {len(failed_models)}[/red] | "
            f"[yellow]Pending: {len([m for m in self.state.models.values() if m.status == ModelStatus.QUEUED])}[/yellow]"
        )
        table.add_row("", summary_text, "", "", "")

        if len(self.state.failed_runs_queue) > 0:
            table.add_row(
                "",
                f"[bold red]⚠ {len(self.state.failed_runs_queue)} runs queued for retry[/bold red]",
                "",
                "",
                ""
            )

        return Panel(table, title="[bold yellow]Benchmark Queue[/bold yellow]", border_style="yellow")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProgressDisplay("
            f"phase={self.state.phase.value}, "
            f"progress={self.state.overall_progress:.1f}%, "
            f"running={self._is_running}"
            f")"
        )
