"""Rich display formatters for terminal UI."""

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..models.model import ModelInfo
from ..models.session import SessionStatistics
from ..models.system import SystemSpecs

console = Console()


def display_system_specs(specs: SystemSpecs) -> None:
    """Display system specifications in Rich Panel.

    Args:
        specs: SystemSpecs to display
    """
    content = f"""[bold cyan]Platform:[/bold cyan] {specs.platform} ({specs.architecture})
[bold cyan]Category:[/bold cyan] {specs.device_category}

[bold green]CPU:[/bold green]
  • Physical Cores: {specs.cpu_cores_physical}
  • Logical Cores: {specs.cpu_cores_logical}

[bold green]Memory:[/bold green]
  • Total RAM: {specs.total_ram_gb:.1f} GB
  • Available RAM: {specs.available_ram_gb:.1f} GB
  • Recommended Model Size: {specs.recommended_model_size_gb:.1f} GB

[bold green]GPU:[/bold green]
  • Available: {"Yes" if specs.gpu.available else "No"}
  • Type: {specs.gpu.gpu_type.upper()}"""

    if specs.gpu.device_name:
        content += f"\n  • Device: {specs.gpu.device_name}"
    if specs.gpu.memory_gb:
        content += f"\n  • Memory: {specs.gpu.memory_gb:.1f} GB"

    panel = Panel(content, title="[bold]System Specifications[/bold]", border_style="blue")
    console.print(panel)
    console.print()


def display_model_table(models: list[ModelInfo]) -> Table:
    """Create Rich Table for model list with compatibility icons.

    Args:
        models: List of ModelInfo to display

    Returns:
        Rich Table instance
    """
    table = Table(title="Available Models", show_header=True, header_style="bold magenta")

    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Status", justify="center", width=6)
    table.add_column("Model Name", style="cyan", width=30)
    table.add_column("Type", justify="center", width=8)
    table.add_column("Size", justify="right", width=8)
    table.add_column("Provider", justify="center", width=12)

    for idx, model in enumerate(models, 1):
        # Format model type (already string due to use_enum_values=True)
        model_type = str(model.model_type).upper()

        # Format size
        size_str = f"{model.size_gb:.1f} GB"

        # Style row based on compatibility (already string due to use_enum_values=True)
        if model.compatibility == "perfect_fit":
            status_icon = "✅"
            name_style = "green"
        elif model.compatibility == "tight_fit":
            status_icon = "⚠️ "
            name_style = "yellow"
        else:  # too_large
            status_icon = "❌"
            name_style = "red"

        table.add_row(
            str(idx),
            status_icon,
            f"[{name_style}]{model.name}[/{name_style}]",
            model_type,
            size_str,
            str(model.provider),
        )

    return table


def display_model_table_print(models: list[ModelInfo]) -> None:
    """Display model table directly to console.

    Args:
        models: List of ModelInfo to display
    """
    table = display_model_table(models)
    console.print(table)
    console.print()


def show_loading_spinner(message: str = "Loading model...") -> Progress:
    """Create indeterminate spinner for model loading.

    Args:
        message: Message to display with spinner

    Returns:
        Progress instance (call .stop() when done)
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )
    progress.add_task(description=message, total=None)
    return progress


def print_success(message: str) -> None:
    """Print success message with icon.

    Args:
        message: Success message
    """
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print error message with icon.

    Args:
        message: Error message
    """
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print warning message with icon.

    Args:
        message: Warning message
    """
    console.print(f"[bold yellow]⚠[/bold yellow]  {message}")


def print_info(message: str) -> None:
    """Print info message.

    Args:
        message: Info message
    """
    console.print(f"[cyan]ℹ[/cyan] {message}")


def show_download_progress() -> tuple:
    """Create Rich Progress instance for model download tracking.

    Returns:
        Tuple of (Progress instance, task_id)
    """
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
    )

    task_id = progress.add_task("Downloading model...", total=100)
    return progress, task_id


def display_statistics(stats: SessionStatistics) -> None:
    """Display session statistics in Rich Table (per FR-031).

    Args:
        stats: SessionStatistics to display
    """
    # Create main statistics table
    table = Table(
        title="Session Statistics",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        expand=True,
    )

    table.add_column("Metric", style="white", no_wrap=True)
    table.add_column("Value", style="green", justify="right")

    # Session info
    session_duration_min = stats.session_duration_seconds / 60
    table.add_row("Session Duration", f"{session_duration_min:.1f} minutes")
    table.add_row("Session ID", stats.session_id)

    # Current model
    if stats.current_model_id:
        table.add_row("Current Model", stats.current_model_id)
        table.add_row("Provider", str(stats.current_provider) if stats.current_provider else "N/A")
    else:
        table.add_row("Current Model", "[dim]No model loaded[/dim]")

    # Separator
    table.add_row("", "")

    # Inference metrics
    table.add_row("[bold]Total Inferences[/bold]", f"[bold]{stats.total_inferences}[/bold]")

    if stats.total_inferences > 0:
        total_time_sec = stats.total_time_ms / 1000
        table.add_row("Total Inference Time", f"{total_time_sec:.2f}s")
        table.add_row("Average Time", f"{stats.average_time_ms:.0f}ms")
    else:
        table.add_row("Total Inference Time", "[dim]0.00s[/dim]")
        table.add_row("Average Time", "[dim]N/A[/dim]")

    # Create panel
    panel = Panel(table, border_style="cyan", expand=True)
    console.print(panel)
    console.print()

    # Show per-endpoint breakdown if there are inferences
    if stats.inferences_by_endpoint:
        breakdown_table = Table(
            title="Breakdown by Endpoint",
            show_header=True,
            header_style="bold magenta",
            border_style="magenta",
            expand=True,
        )

        breakdown_table.add_column("Endpoint", style="cyan")
        breakdown_table.add_column("Count", justify="right", style="white")
        breakdown_table.add_column("Percentage", justify="right", style="green")

        # Sort by count (descending)
        sorted_endpoints = sorted(
            stats.inferences_by_endpoint.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for endpoint, count in sorted_endpoints:
            percentage = (count / stats.total_inferences) * 100
            # Format endpoint name nicely
            endpoint_name = str(endpoint).upper() if isinstance(endpoint, str) else endpoint.value.upper()
            breakdown_table.add_row(
                endpoint_name,
                str(count),
                f"{percentage:.1f}%"
            )

        breakdown_panel = Panel(breakdown_table, border_style="magenta", expand=True)
        console.print(breakdown_panel)
        console.print()


def display_conversation_history(session: "ConversationSession") -> None:
    """Display conversation history in scrolling format with thinking hidden.

    Args:
        session: ConversationSession with exchanges to display
    """
    from ..models.session import ConversationSession
    from ..utils.response_formatter import extract_thinking_blocks

    if not session.exchanges:
        console.print("[dim]No conversation history yet[/dim]\n")
        return

    # Show session info
    console.print(f"[bold cyan]Session {session.session_id}[/bold cyan] • {session.exchange_count} exchanges\n")

    # Display each exchange
    for exchange in session.exchanges:
        # User message
        console.print(f"[bold green]You:[/bold green] {exchange.user_message}")

        # AI response - hide thinking blocks in history
        _, clean_response = extract_thinking_blocks(exchange.ai_response)
        console.print(f"[bold cyan]AI:[/bold cyan] {clean_response}")

        # Show timing if available
        if exchange.inference_time_ms > 0:
            console.print(f"[dim]({exchange.inference_time_ms / 1000:.2f}s)[/dim]")

        console.print()  # Blank line between exchanges
