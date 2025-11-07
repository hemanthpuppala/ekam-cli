"""Quantization integration for main app."""

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from ..cli.tui_manager import tui
from ..cli.diagnostics import SystemDiagnostics
from ..quantization.manager import QuantizationManager
from ..quantization.ui import run_quantization_workflow, show_background_jobs_monitor
from ..services.session import SessionManager

if TYPE_CHECKING:
    from ..benchmarking.core.benchmark_runner import BenchmarkRunner
    from ..benchmarking.models.benchmark_config import BenchmarkConfig


def show_quantization_or_inference_menu() -> str:
    """Show menu to choose between Quantization, Finetuning, Inference, or Benchmarking.

    Returns:
        "quantization", "finetuning", "inference", "benchmarking", "diagnostics", or "quit"
    """
    from ..cli.text_input import professional_prompt

    tui.clear_screen()

    # Define menu options as (value, label, description) tuples
    options = [
        (
            "inference",
            "[green]Inference[/green]",
            "Run model inference (Chat, QA, Caption, Detect, Point) - Use existing models for text/vision tasks"
        ),
        (
            "quantization",
            "[yellow]Quantization[/yellow]",
            "Reduce model size through quantization - Create optimized models for faster inference"
        ),
        (
            "finetuning",
            "[magenta]Finetuning[/magenta]",
            "Adapt models to your domain using LoRA, QLoRA, or Full finetuning - Improve performance on custom tasks"
        ),
        (
            "benchmarking",
            "[cyan]Benchmarking[/cyan]",
            "Measure model performance and efficiency - Speed, resources, quality, and stress testing"
        ),
        (
            "diagnostics",
            "[blue]System Diagnostics[/blue]",
            "View hardware capabilities and quantization support - Check installed dependencies and recommendations"
        ),
        (
            "quit",
            "Quit Application",
            "Exit the application"
        ),
    ]

    # Show step heading with underline (Ekam-CLI header already shown by clear_screen)
    tui.show_step_heading("Main Menu")
    tui.console.print("[dim]Production-ready for Windows, Linux, and macOS[/dim]")
    tui.console.print("[dim]Full transparency and hardware capability detection[/dim]")
    tui.console.print()

    # Use arrow-key selection
    choice = professional_prompt.get_arrow_selection(
        options=options,
        title="Select Operation Mode",
        instructions="Use ↑/↓ arrows to navigate, Enter to select, q to quit"
    )

    # Handle None (cancelled) as quit
    if choice is None:
        return "quit"

    return choice


def initialize_quantization_manager(session_manager: SessionManager) -> QuantizationManager:
    """Initialize quantization manager.

    Args:
        session_manager: Session manager with system specs

    Returns:
        Initialized QuantizationManager
    """
    output_dir = Path("results/quantizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_manager = QuantizationManager(
        system_specs=session_manager.system_specs,
        output_dir=output_dir,
    )

    logger.info("Quantization manager initialized")
    return quant_manager


def run_quantization_mode(session_manager: SessionManager) -> str:
    """Run quantization mode.

    Args:
        session_manager: Session manager

    Returns:
        "quit" to quit app
        "main_menu" to return to main menu
        "switch_inference" to switch to inference mode
    """
    # Initialize quantization manager
    quant_manager = initialize_quantization_manager(session_manager)

    # Register globally for /background command and notifications
    from . import app
    app._global_quantization_manager = quant_manager

    # Track completed jobs for notifications
    completed_tasks = set()

    while True:
        # Check for notifications using global notification system
        from . import app
        from ..cli.text_input import professional_prompt

        app.check_and_show_completion_notifications()

        # Get status for status bar
        status_text = ""
        if quant_manager.has_active_jobs():
            status_text = quant_manager.get_status_summary()

        # Show header
        tui.clear_screen()
        terminal_width = tui.console.width
        tui.console.print()
        tui.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
        tui.console.print("[bold cyan]Quantization Mode[/bold cyan]".center(terminal_width))
        tui.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
        tui.console.print()

        if status_text:
            tui.console.print(f"[yellow]Status:[/yellow] {status_text}")
            tui.console.print()

        # Build arrow-key options
        arrow_options = [
            ("1", "[green]Start New Quantization[/green]", "Configure and start a new model quantization"),
            ("2", "[yellow]Monitor Background Jobs[/yellow]", "View and manage running quantization jobs"),
            ("3", "[blue]Switch to Inference Mode[/blue]", "Return to inference mode"),
            ("main_menu", "[cyan]Main Menu (Home)[/cyan]", "Return to operation mode selection"),
            ("quit", "[red]Quit Application[/red]", "Exit the application"),
        ]

        # Use arrow-key selection
        choice = professional_prompt.get_arrow_selection(
            options=arrow_options,
            title="Quantization Mode",
            instructions="Use ↑/↓ arrows to navigate, Enter to select, q to quit"
        )

        if choice == "quit" or choice is None:
            return "quit"
        elif choice == "main_menu":
            return "main_menu"
        elif choice == "1":
            # Start new quantization
            run_quantization_workflow(session_manager.model_discovery, quant_manager)
        elif choice == "2":
            # Monitor jobs
            show_background_jobs_monitor(quant_manager)
        elif choice == "3":
            # Switch to inference mode
            return "switch_inference"


def run_finetuning_mode(session_manager: SessionManager) -> str:
    """Run finetuning mode.

    Args:
        session_manager: Session manager

    Returns:
        "quit" to quit app
        "main_menu" to return to main menu
        "switch_inference" to switch to inference mode
        "switch_quantization" to switch to quantization mode
    """
    from .app_finetuning import (
        show_finetuning_home_menu,
        start_finetuning_workflow,
        execute_finetuning,
    )

    while True:
        # Show finetuning home menu
        try:
            action = show_finetuning_home_menu()

            if action == "quit":
                return "quit"
            elif action == "main_menu":
                return "main_menu"
            elif action == "inference":
                return "switch_inference"
            elif action == "quantization":
                return "switch_quantization"
            elif action == "start_finetuning":
                # Start new finetuning workflow
                pipeline = start_finetuning_workflow(session_manager)
                if pipeline:
                    execute_finetuning(pipeline)
            elif action == "load_finetuning_config":
                tui.show_info("Loading previous configuration not yet implemented.\nUse 'Start New' instead.")

        except KeyboardInterrupt:
            logger.info("User interrupted finetuning mode")
            return "main_menu"
        except Exception as e:
            logger.error(f"Error in finetuning mode: {e}", exc_info=True)
            tui.show_error(f"Error: {e}")
            tui.prompt("Press Enter to continue...", style="dim")


def run_benchmarking_mode(session_manager: SessionManager) -> str:
    """Run benchmarking mode with comprehensive configuration flow.

    Args:
        session_manager: Session manager

    Returns:
        "quit" to quit app
        "main_menu" to return to main menu
        "switch_inference" to switch to inference mode
    """
    from ..cli.benchmark_menu import BenchmarkMenu
    from ..benchmarking.cli.benchmark_menu import BenchmarkMenuFlow
    from ..benchmarking.core.benchmark_runner import BenchmarkRunner
    from ..benchmarking.models.metric_types import ExecutionMode

    # Initialize benchmark runner
    runner = BenchmarkRunner(session_manager=session_manager)

    while True:
        try:
            from ..cli.text_input import professional_prompt

            # Show header
            tui.clear_screen()
            terminal_width = tui.console.width
            tui.console.print()
            tui.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
            tui.console.print("[bold cyan]Benchmark Mode[/bold cyan]".center(terminal_width))
            tui.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
            tui.console.print()
            tui.console.print("[dim]Comprehensive performance testing and analysis[/dim]")
            tui.console.print()

            # Build arrow-key options
            arrow_options = [
                (
                    "1",
                    "[green]Configure New Benchmark[/green]",
                    "Interactive 7-step configuration: Model type, Suite, Models, Endpoints, Test data, Parameters"
                ),
                (
                    "2",
                    "[blue]View Previous Results[/blue]",
                    "Browse benchmark history, compare models, view detailed metrics"
                ),
                ("main_menu", "[cyan]Back to Main Menu[/cyan]", "Return to operation mode selection"),
                ("quit", "[red]Quit Application[/red]", "Exit the application"),
            ]

            # Use arrow-key selection
            choice = professional_prompt.get_arrow_selection(
                options=arrow_options,
                title="Benchmark Mode",
                instructions="Use ↑/↓ arrows to navigate, Enter to select, q to quit"
            )

            # Handle navigation
            if choice == "quit" or choice is None:
                logger.info("User quit from benchmarking mode")
                return "quit"
            elif choice == "main_menu":
                logger.info("Returning to main menu from benchmarking")
                return "main_menu"

            # Handle benchmark configuration
            if choice == "1":
                logger.info("Starting benchmark configuration flow")

                # Run comprehensive configuration flow
                flow = BenchmarkMenuFlow(session_manager)
                config = flow.run()

                if config is None:
                    # User cancelled configuration
                    logger.info("User cancelled benchmark configuration")
                    continue

                # Execute benchmark based on execution mode
                if config.execution_mode == ExecutionMode.FOREGROUND:
                    logger.info("Running benchmark in foreground mode")
                    _run_foreground_benchmark(runner, config)
                else:
                    logger.info("Running benchmark in background mode")
                    _run_background_benchmark(runner, config)

            elif choice == "2":
                logger.info("Viewing previous benchmark results")
                _view_benchmark_results(runner)

        except KeyboardInterrupt:
            logger.info("User interrupted benchmarking mode")
            return "main_menu"
        except Exception as e:
            logger.error(f"Error in benchmarking mode: {e}", exc_info=True)
            tui.show_error(f"Error: {e}")
            tui.prompt("Press Enter to continue...", style="dim")


def _run_foreground_benchmark(runner: "BenchmarkRunner", config: "BenchmarkConfig") -> None:
    """Execute benchmark in foreground with real-time progress and per-run feedback.

    Args:
        runner: BenchmarkRunner instance
        config: BenchmarkConfig to execute
    """
    from ..cli.tui_manager import tui
    from rich.table import Table
    from rich.panel import Panel
    import re

    # Clear screen completely - no previous steps visible
    tui.clear_screen()
    terminal_width = tui.console.width

    # Show concise configuration summary at the top
    tui.console.print()
    tui.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
    tui.console.print("[bold cyan]Running Benchmark[/bold cyan]".center(terminal_width))
    tui.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
    tui.console.print()

    # Configuration summary table
    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column(style="cyan", justify="right")
    summary_table.add_column(style="white")

    summary_table.add_row("Suite:", config.suite_type.display_name())
    summary_table.add_row("Model Type:", config.model_type.value.upper())
    summary_table.add_row("Models:", f"{len(config.models)} model(s)")
    summary_table.add_row("Models List:", ", ".join([m[:30] + "..." if len(m) > 30 else m for m in config.models]))
    summary_table.add_row("Runs per model:", f"{config.num_runs} ({config.num_warmup} warmup)")
    summary_table.add_row("Export formats:", ", ".join(config.export_formats))

    tui.console.print(Panel(
        summary_table,
        title="[bold]Configuration Summary[/bold]",
        border_style="cyan",
        padding=(1, 2)
    ))
    tui.console.print()

    # Section separator
    tui.console.print("[bold cyan]" + "─" * terminal_width + "[/bold cyan]")
    tui.console.print("[bold cyan]Execution Progress[/bold cyan]")
    tui.console.print("[bold cyan]" + "─" * terminal_width + "[/bold cyan]")
    tui.console.print()

    # Track run completion for real-time display
    run_tracker = {
        "current_model": None,
        "runs_completed": 0,
        "total_runs": config.num_runs,
        "warmup_runs": 0,
        "total_warmup": config.num_warmup
    }

    def progress_callback(message: str):
        """Display progress messages with enhanced per-run feedback."""
        # Parse latency information from progress messages
        # Expected format: "Run X/Y completed in Z.ZZms" or similar patterns

        # Check for run completion with latency
        run_pattern = r"(?:Run|Warmup run)\s+(\d+)/(\d+).*?(\d+\.?\d*)\s*ms"
        match = re.search(run_pattern, message, re.IGNORECASE)

        if match:
            run_num = match.group(1)
            total = match.group(2)
            latency = float(match.group(3))

            # Determine if warmup or regular run
            is_warmup = "warmup" in message.lower()

            if is_warmup:
                run_tracker["warmup_runs"] = int(run_num)
                tui.console.print(
                    f"  [dim cyan]Warmup {run_num}/{total}:[/dim cyan] "
                    f"[yellow]{latency:.2f}ms[/yellow]"
                )
            else:
                run_tracker["runs_completed"] = int(run_num)
                # Show with green checkmark for completed runs
                tui.console.print(
                    f"  [green]✓ Run {run_num}/{total}:[/green] "
                    f"[bold yellow]{latency:.2f}ms[/bold yellow] latency"
                )

        # Check for model transitions
        elif "Benchmarking model:" in message:
            model_name = message.split("Benchmarking model:")[-1].strip()
            run_tracker["current_model"] = model_name
            run_tracker["runs_completed"] = 0
            run_tracker["warmup_runs"] = 0

            tui.console.print()
            tui.console.print(f"[bold magenta]📊 Model:[/bold magenta] [cyan]{model_name}[/cyan]")

        # Check for warmup start
        elif "warmup runs" in message.lower() and "Running" in message:
            tui.console.print(f"[dim]{message}[/dim]")

        # Check for suite completion messages
        elif "complete" in message.lower() or "finished" in message.lower():
            tui.console.print(f"[green]{message}[/green]")

        # Default: show dimmed message
        else:
            tui.console.print(f"[dim]{message}[/dim]")

    try:
        tui.console.print("[bold blue]Starting benchmark execution...[/bold blue]")
        tui.console.print()

        result = runner.run(config, progress_callback=progress_callback)

        # Clear and show completion summary
        tui.console.print()
        tui.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
        tui.console.print()
        tui.console.print(f"[bold green]✓ Benchmark Complete![/bold green]")
        tui.console.print()
        tui.console.print(f"  [cyan]Status:[/cyan] {result.status.value}")
        tui.console.print(f"  [cyan]Duration:[/cyan] {result.duration_seconds:.2f}s")
        tui.console.print(f"  [cyan]Successful runs:[/cyan] {result.successful_runs}/{result.total_runs}")

        if result.successful_runs < result.total_runs:
            failed = result.total_runs - result.successful_runs
            tui.console.print(f"  [yellow]Failed runs:[/yellow] {failed}")

        tui.console.print()
        tui.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
        tui.console.print()

        # Ask if user wants to see detailed results
        show_details = tui.prompt("Show detailed results? [Y/n]:", style="green").strip().lower() != "n"

        if show_details:
            runner.print_result(result, detailed=True)

    except Exception as e:
        tui.console.print()
        tui.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
        tui.console.print()
        tui.console.print(f"[bold red]✗ Benchmark Failed[/bold red]")
        tui.console.print(f"[red]Error: {str(e)}[/red]")
        logger.error(f"Benchmark execution failed: {e}", exc_info=True)
        tui.console.print()
        tui.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")

    tui.console.print()
    tui.prompt("Press Enter to continue...", style="dim")


def _run_background_benchmark(runner: "BenchmarkRunner", config: "BenchmarkConfig") -> None:
    """Execute benchmark in background thread.

    Args:
        runner: BenchmarkRunner instance
        config: BenchmarkConfig to execute
    """
    from ..cli.tui_manager import tui
    import threading

    tui.clear_screen()
    tui.console.print(f"\n[bold cyan]Starting {config.suite_type.display_name()} in background...[/bold cyan]\n")

    # Create background thread
    benchmark_thread = threading.Thread(
        target=_background_worker,
        args=(runner, config),
        daemon=True,
        name=f"benchmark-{config.suite_type.value}"
    )

    benchmark_thread.start()

    tui.console.print("[green]✓ Benchmark started in background[/green]")
    tui.console.print("[dim]You can continue using the application while benchmark runs[/dim]")
    tui.console.print(f"[dim]Results will be saved to: {config.output_dir}[/dim]\n")

    tui.prompt("Press Enter to continue...", style="dim")


def _background_worker(runner: "BenchmarkRunner", config: "BenchmarkConfig") -> None:
    """Background worker thread for running benchmarks.

    Args:
        runner: BenchmarkRunner instance
        config: BenchmarkConfig to execute
    """
    try:
        logger.info(f"Background benchmark worker started for {config.suite_type.value}")

        result = runner.run(config, progress_callback=None)

        logger.info(
            f"Background benchmark complete: {result.status.value}, "
            f"{result.successful_runs}/{result.total_runs} runs succeeded"
        )

        # Notification (logged, could be extended to system notifications)
        from ..cli.tui_manager import tui
        tui.console.print(f"\n[bold green]🎉 Benchmark Complete![/bold green]")
        tui.console.print(f"[green]{config.suite_type.display_name()} finished with status: {result.status.value}[/green]\n")

    except Exception as e:
        logger.error(f"Background benchmark failed: {e}", exc_info=True)


def _view_benchmark_results(runner: "BenchmarkRunner") -> None:
    """View previous benchmark results.

    Args:
        runner: BenchmarkRunner instance with results_manager
    """
    from ..cli.tui_manager import tui
    from rich.table import Table

    tui.clear_screen()

    try:
        results = runner.list_results(limit=20)

        if not results:
            tui.show_message(
                "No benchmark results found yet.\n\n"
                "Run a benchmark to see results here.",
                title="No Results",
                style="yellow"
            )
            tui.prompt("Press Enter to continue...", style="dim")
            return

        tui.console.print(f"\n[bold cyan]Benchmark Results ({len(results)} found)[/bold cyan]\n")

        table = Table(title="Recent Benchmarks", border_style="cyan")
        table.add_column("#", style="cyan")
        table.add_column("ID", style="yellow")
        table.add_column("Suite", style="green")
        table.add_column("Models", style="magenta")
        table.add_column("Status")
        table.add_column("Date")

        for idx, result in enumerate(results, 1):
            table.add_row(
                str(idx),
                result["benchmark_id"][:8],
                result["suite_type"],
                ", ".join(result["models_tested"])[:20] if result.get("models_tested") else "N/A",
                result["status"],
                result.get("date", "N/A")
            )

        tui.show_panel(table)
        tui.prompt("\nPress Enter to continue...", style="dim")

    except Exception as e:
        logger.error(f"Failed to view results: {e}", exc_info=True)
        tui.show_error(f"Failed to load results: {e}")
        tui.prompt("Press Enter to continue...", style="dim")
