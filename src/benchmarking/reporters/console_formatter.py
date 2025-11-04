"""
Console formatter for benchmark results.

Formats benchmark results for display in terminal using Rich library.
"""

from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from src.benchmarking.models.suite_result import SuiteResult, AggregateStats
from src.benchmarking.models.metric_types import ResultStatus


class ConsoleFormatter:
    """
    Formats benchmark results for terminal display.

    Uses Rich library for beautiful terminal output with:
    - Color-coded status indicators
    - Formatted tables for metrics
    - Summary panels
    - Progress indicators
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize console formatter.

        Args:
            console: Optional Rich Console instance
        """
        self.console = console or Console()

    def print_result(self, result: SuiteResult, detailed: bool = True) -> None:
        """
        Print benchmark result to console.

        Args:
            result: SuiteResult to display
            detailed: Whether to show detailed metrics
        """
        # Print header
        self._print_header(result)

        # Print summary
        self._print_summary(result)

        # Print aggregate statistics
        self._print_aggregates(result)

        # Print per-model breakdown if detailed
        if detailed and result.aggregates_per_model:
            self._print_per_model_breakdown(result)

    def print_comparison(
        self,
        results: List[SuiteResult],
        metric_name: str = "latency_ms"
    ) -> None:
        """
        Print comparison table across multiple results.

        Args:
            results: List of SuiteResult objects
            metric_name: Metric to compare
        """
        if not results:
            self.console.print("[yellow]No results to compare[/yellow]")
            return

        table = Table(title=f"Model Comparison: {metric_name}")
        table.add_column("Model", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Std Dev", justify="right")

        for result in results:
            for model_id in result.models_tested:
                if model_id in result.aggregates_per_model:
                    model_aggregates = result.aggregates_per_model[model_id]

                    if metric_name in model_aggregates:
                        agg = model_aggregates[metric_name]
                        table.add_row(
                            model_id,
                            f"{agg.mean:.2f} {agg.unit.value}",
                            f"{agg.median:.2f} {agg.unit.value}",
                            f"{agg.min:.2f} {agg.unit.value}",
                            f"{agg.max:.2f} {agg.unit.value}",
                            f"{agg.std_dev:.2f} {agg.unit.value}" if agg.std_dev else "N/A"
                        )

        self.console.print(table)

    def print_summary(self, results: List[SuiteResult]) -> None:
        """
        Print summary of multiple benchmark results.

        Args:
            results: List of SuiteResult objects
        """
        if not results:
            self.console.print("[yellow]No results to display[/yellow]")
            return

        table = Table(title="Benchmark Summary")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Suite", style="blue")
        table.add_column("Models", style="magenta")
        table.add_column("Status", justify="center")
        table.add_column("Runs", justify="right")
        table.add_column("Success Rate", justify="right")
        table.add_column("Duration", justify="right")

        for result in results:
            status_style = self._get_status_style(result.status)

            table.add_row(
                result.benchmark_id[:8],
                result.suite_type.value,
                ", ".join(result.models_tested)[:30],
                f"[{status_style}]{result.status.value}[/{status_style}]",
                f"{result.successful_runs}/{result.total_runs}",
                f"{result.success_rate():.1f}%",
                f"{result.duration_seconds:.1f}s"
            )

        self.console.print(table)

    def _print_header(self, result: SuiteResult) -> None:
        """Print result header."""
        status_style = self._get_status_style(result.status)

        header_text = Text()
        header_text.append("Benchmark Result: ", style="bold")
        header_text.append(result.suite_type.display_name(), style="cyan bold")
        header_text.append(f" [{result.status.value}]", style=f"bold {status_style}")

        panel = Panel(
            header_text,
            border_style=status_style,
            padding=(0, 2)
        )
        self.console.print(panel)

    def _print_summary(self, result: SuiteResult) -> None:
        """Print result summary."""
        table = Table(title="Summary", show_header=False, box=None)
        table.add_column("Property", style="bold")
        table.add_column("Value")

        table.add_row("Benchmark ID", result.benchmark_id)
        table.add_row("Suite Type", result.suite_type.display_name())
        table.add_row("Model Type", result.model_type.display_name())
        table.add_row("Models Tested", ", ".join(result.models_tested))
        table.add_row("Total Runs", str(result.total_runs))
        table.add_row("Successful", str(result.successful_runs))
        table.add_row("Failed", str(result.failed_runs))
        table.add_row("Success Rate", f"{result.success_rate():.1f}%")
        table.add_row("Duration", f"{result.duration_seconds:.2f} seconds")

        if result.start_time:
            table.add_row("Start Time", result.start_time.strftime("%Y-%m-%d %H:%M:%S"))

        self.console.print(table)
        self.console.print()

    def _print_aggregates(self, result: SuiteResult) -> None:
        """Print aggregate statistics."""
        if not result.aggregates_counted:
            return

        table = Table(title="Aggregate Statistics (Counted Runs)")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Std Dev", justify="right")
        table.add_column("Count", justify="right")

        for metric_name, agg in result.aggregates_counted.items():
            table.add_row(
                metric_name,
                f"{agg.mean:.2f}",
                f"{agg.median:.2f}",
                f"{agg.min:.2f}",
                f"{agg.max:.2f}",
                f"{agg.std_dev:.2f}" if agg.std_dev else "N/A",
                str(agg.count)
            )

        self.console.print(table)
        self.console.print()

    def _print_per_model_breakdown(self, result: SuiteResult) -> None:
        """Print per-model metric breakdown."""
        for model_id, model_aggregates in result.aggregates_per_model.items():
            if not model_aggregates:
                continue

            table = Table(title=f"Model: {model_id}")
            table.add_column("Metric", style="cyan")
            table.add_column("Mean", justify="right")
            table.add_column("Median", justify="right")
            table.add_column("Std Dev", justify="right")

            for metric_name, agg in model_aggregates.items():
                table.add_row(
                    metric_name,
                    f"{agg.mean:.2f} {agg.unit.value}",
                    f"{agg.median:.2f} {agg.unit.value}",
                    f"{agg.std_dev:.2f}" if agg.std_dev else "N/A"
                )

            self.console.print(table)
            self.console.print()

    def _get_status_style(self, status: ResultStatus) -> str:
        """
        Get Rich style for status.

        Args:
            status: ResultStatus

        Returns:
            Style string
        """
        if status == ResultStatus.SUCCESS:
            return "green"
        elif status == ResultStatus.FAILED:
            return "red"
        elif status == ResultStatus.PARTIAL:
            return "yellow"
        else:
            return "white"

    def format_aggregate(self, agg: AggregateStats) -> str:
        """
        Format aggregate statistics as string.

        Args:
            agg: AggregateStats object

        Returns:
            Formatted string
        """
        return (
            f"{agg.metric_name}: "
            f"mean={agg.mean:.2f}, "
            f"median={agg.median:.2f}, "
            f"min={agg.min:.2f}, "
            f"max={agg.max:.2f}, "
            f"std_dev={agg.std_dev:.2f if agg.std_dev else 'N/A'} "
            f"({agg.unit.value})"
        )

    def __repr__(self) -> str:
        """String representation."""
        return "ConsoleFormatter()"
