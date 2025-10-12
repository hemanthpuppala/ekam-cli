"""Quantization integration for main app."""

from pathlib import Path

from loguru import logger

from ..cli.tui_manager import tui
from ..quantization.manager import QuantizationManager
from ..quantization.workflow import run_quantization_workflow, show_background_jobs_monitor
from ..services.session import SessionManager


def show_quantization_or_inference_menu() -> str:
    """Show menu to choose between Quantization or Inference.

    Returns:
        "quantization", "inference", or "quit"
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

[dim]Quantization Phase 1: GGUF models only
Future: HuggingFace and Ollama model conversion[/dim]""",
        title="Operation Mode",
        border_style="cyan",
    )

    while True:
        choice = tui.prompt("Choose [1/2] or 'q' to quit:", style="cyan").strip().lower()

        if choice in ["q", "quit", "exit"]:
            return "quit"
        elif choice == "1":
            return "inference"
        elif choice == "2":
            return "quantization"
        else:
            tui.show_error("Invalid choice. Enter 1, 2, or 'q'")


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


def run_quantization_mode(session_manager: SessionManager) -> bool:
    """Run quantization mode.

    Args:
        session_manager: Session manager

    Returns:
        True to continue app, False to quit
    """
    # Initialize quantization manager
    quant_manager = initialize_quantization_manager(session_manager)

    while True:
        # Show quantization menu
        tui.clear_screen()
        tui.console.print("[bold cyan]Quantization Mode[/bold cyan]\n")
        tui.console.print("[1] [green]Start New Quantization[/green]")
        tui.console.print("[2] [yellow]Monitor Background Jobs[/yellow]")
        tui.console.print("[b] [dim]Back to main menu[/dim]")
        tui.console.print("[q] [dim]Quit application[/dim]\n")

        # Show status bar if there are active jobs
        if quant_manager.has_active_jobs():
            status = quant_manager.get_status_summary()
            tui.console.print(f"[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]")
            tui.console.print(f"[bold yellow]{status}[/bold yellow]")
            tui.console.print(f"[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]\n")

        choice = tui.prompt("Choose:", style="cyan").strip().lower()

        if choice in ["q", "quit", "exit"]:
            return False  # Quit app
        elif choice in ["b", "back"]:
            return True  # Back to main menu
        elif choice == "1":
            # Start new quantization
            run_quantization_workflow(session_manager.model_discovery, quant_manager)
        elif choice == "2":
            # Monitor jobs
            show_background_jobs_monitor(quant_manager)
        else:
            tui.show_error("Invalid choice. Enter 1, 2, 'b', or 'q'")
            tui.prompt("Press Enter to continue...", style="dim")
