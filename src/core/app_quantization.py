"""Quantization integration for main app."""

from pathlib import Path

from loguru import logger

from ..cli.tui_manager import tui
from ..cli.diagnostics import SystemDiagnostics
from ..quantization.manager import QuantizationManager
from ..quantization.ui import run_quantization_workflow, show_background_jobs_monitor
from ..services.session import SessionManager


def show_quantization_or_inference_menu() -> str:
    """Show menu to choose between Quantization, Finetuning, or Inference.

    Returns:
        "quantization", "finetuning", "inference", "diagnostics", or "quit"
    """
    tui.clear_screen()

    tui.show_panel(
        """[bold cyan]Select Operation Mode[/bold cyan]

[bold]What would you like to do?[/bold]

[1] [green]Inference[/green]
    Run model inference (Chat, QA, Caption, Detect, Point)
    Use existing models for text/vision tasks

[2] [yellow]Quantization[/yellow]
    Reduce model size through quantization
    Create optimized models for faster inference

[3] [magenta]Finetuning[/magenta]
    Adapt models to your domain using LoRA, QLoRA, or Full finetuning
    Improve performance on custom tasks with your data

[4] [blue]System Diagnostics[/blue]
    View hardware capabilities and quantization support
    Check installed dependencies and recommendations

[bold]Navigation:[/bold]
  [q] Quit application

[dim]Production-ready for Windows, Linux, and macOS
Full transparency and hardware capability detection[/dim]""",
        title="Operation Mode",
        border_style="cyan",
    )

    while True:
        choice = tui.prompt("Choose [1/2/3/4/q]:", style="cyan").strip().lower()

        if choice in ["q", "quit", "exit"]:
            return "quit"
        elif choice == "1":
            return "inference"
        elif choice == "2":
            return "quantization"
        elif choice == "3":
            return "finetuning"
        elif choice == "4":
            return "diagnostics"
        else:
            tui.show_error("Invalid choice. Enter 1, 2, 3, 4, or 'q'")


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
        app.check_and_show_completion_notifications()

        # Build menu content
        menu_content = """[bold cyan]Quantization Mode[/bold cyan]

[1] [green]Start New Quantization[/green]
[2] [yellow]Monitor Background Jobs[/yellow]
[3] [blue]Switch to Inference Mode[/blue]

[bold]Navigation:[/bold]
  [h] Main menu (home)
  [b] Back to main menu
  [q] Quit application

[bold]Commands:[/bold]
  [yellow]/background[/yellow] - Monitor background jobs (if running)
"""

        # Get status for status bar
        status_text = ""
        if quant_manager.has_active_jobs():
            status_text = quant_manager.get_status_summary()

        # Show menu with status bar
        tui.clear_and_show_with_status(menu_content, status_text)

        choice = tui.prompt("\nChoose [1-3/h/b/q or /background]:", style="cyan").strip().lower()

        # Handle /background command
        if choice.startswith("/background") or choice == "/bg":
            from . import app
            app.handle_background_command()
            continue

        if choice in ["q", "quit", "exit"]:
            return "quit"
        elif choice in ["b", "back", "h", "home", "main_menu"]:
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
        else:
            tui.show_error("Invalid choice. Enter 1, 2, 3, 'b', 'm', or 'q'")
            tui.prompt("Press Enter to continue...", style="dim")


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
