"""Main quantization workflow."""

from loguru import logger

from ...cli.prompts import UserExitException
from ...cli.tui_manager import tui
from ...services.model_discovery import ModelDiscovery
from ..manager import QuantizationManager
from .display import (
    ask_background_mode,
    ask_gpu_preference,
    confirm_quantization,
    select_model_to_quantize,
    select_quantization_type,
    show_live_progress,
    show_quantization_intro,
)


def run_quantization_workflow(
    model_discovery: ModelDiscovery,
    quantization_manager: QuantizationManager,
) -> None:
    """Run the complete quantization workflow.

    Args:
        model_discovery: Model discovery service
        quantization_manager: Quantization manager
    """
    try:
        # Step 1: Show introduction
        show_quantization_intro()

        # Step 2: Get all available models
        all_models = model_discovery.discover_all_models()

        # Step 3: Filter to quantizable models (GGUF only for Phase 1)
        quantizable_models = quantization_manager.get_quantizable_models(all_models)

        # Step 4: Select model
        selected_model = select_model_to_quantize(quantizable_models)
        if not selected_model:
            logger.info("User cancelled model selection")
            return

        logger.info(f"User selected model: {selected_model.name}")

        # Step 5: Ask GPU preference
        try:
            use_gpu = ask_gpu_preference()
            logger.info(f"GPU preference: {use_gpu}")
        except UserExitException:
            logger.info("User cancelled at GPU selection")
            return

        # Step 6: Get recommendations
        recommendations = quantization_manager.get_recommendations(
            selected_model,
            use_gpu=use_gpu,
        )

        if not recommendations:
            tui.show_error("No quantization options available for this model.")
            tui.prompt("Press Enter to continue...", style="dim")
            return

        # Step 7: Select quantization type
        try:
            selected_rec = select_quantization_type(recommendations)
            if not selected_rec:
                logger.info("User cancelled quantization type selection")
                return

            logger.info(f"User selected quantization: {selected_rec.quant_type.display_name}")
        except UserExitException:
            logger.info("User cancelled at quantization type selection")
            return

        # Step 8: Confirm quantization
        try:
            if not confirm_quantization(selected_model, selected_rec):
                logger.info("User cancelled at confirmation")
                return
        except UserExitException:
            logger.info("User cancelled at confirmation")
            return

        # Step 9: Ask processing mode (live vs background)
        try:
            run_background = ask_background_mode()
            logger.info(f"Processing mode: {'background' if run_background else 'live'}")
        except UserExitException:
            logger.info("User cancelled at processing mode selection")
            return

        # Step 10: Create task
        task = quantization_manager.create_task(
            model_info=selected_model,
            quant_type=selected_rec.quant_type,
            module=selected_rec.module,
            use_gpu=use_gpu,
            background=run_background,
        )

        # Step 11: Submit task
        task_id = quantization_manager.submit_task(task)
        logger.info(f"Submitted quantization task: {task_id}")

        # Step 12: Monitor based on mode
        if run_background:
            # Background mode - show confirmation and return to menu
            tui.clear_screen()
            tui.show_message(
                f"""[bold green]✓ Quantization Started in Background[/bold green]

[bold]Task ID:[/bold] {task_id}
[bold]Model:[/bold] {selected_model.name}
[bold]Type:[/bold] {selected_rec.quant_type.display_name}

A status bar will appear at the bottom of all screens showing progress.

[bold]Commands:[/bold]
  • Type [yellow]/background[/yellow] to monitor progress
  • Type [yellow]/cancel {task_id}[/yellow] to cancel

[dim]Estimated completion: {selected_rec.estimated_time_minutes:.0f} minutes[/dim]""",
                title="Background Job Started",
                style="green",
            )
            tui.prompt("Press Enter to return to menu...", style="dim")
        else:
            # Live mode - show real-time progress
            show_live_progress(task)

    except UserExitException:
        logger.info("User exited quantization workflow")
    except Exception as e:
        logger.error(f"Quantization workflow error: {e}", exc_info=True)
        tui.show_error(f"Unexpected error: {str(e)}")
        tui.prompt("Press Enter to continue...", style="dim")


def show_background_jobs_monitor(quantization_manager: QuantizationManager) -> None:
    """Show monitor for background quantization jobs.

    Args:
        quantization_manager: Quantization manager
    """
    tui.clear_screen()

    active_tasks = quantization_manager.get_active_tasks()
    all_tasks = quantization_manager.get_all_tasks()

    if not all_tasks:
        tui.show_panel(
            """[bold yellow]No Quantization Jobs[/bold yellow]

No quantization jobs have been run yet.

[dim]Press Enter to go back...[/dim]""",
            title="Background Jobs",
            border_style="yellow",
        )
        tui.prompt("", style="dim")
        return

    # Show active jobs
    tui.console.print("[bold cyan]Background Quantization Jobs[/bold cyan]\n")

    if active_tasks:
        tui.console.print("[bold green]Active Jobs:[/bold green]\n")
        for task in active_tasks:
            progress_bar = "█" * int(task.progress / 5) + "▒" * (20 - int(task.progress / 5))
            eta_str = ""
            if task.eta_seconds:
                mins = int(task.eta_seconds / 60)
                secs = int(task.eta_seconds % 60)
                eta_str = f"ETA: {mins}m {secs}s"

            tui.console.print(f"  {task.status.emoji} [cyan]{task.model_info.name}[/cyan]")
            tui.console.print(f"     Type: {task.quant_type.display_name}")
            tui.console.print(f"     [{progress_bar}] {task.progress:.1f}% {eta_str}")
            tui.console.print(f"     [dim]ID: {task.task_id}[/dim]\n")
    else:
        tui.console.print("[dim]No active jobs[/dim]\n")

    # Show completed jobs
    completed_tasks = [t for t in all_tasks if t.is_finished]
    if completed_tasks:
        tui.console.print("[bold]Recent Completed Jobs:[/bold]\n")
        for task in completed_tasks[-5:]:  # Show last 5
            status_color = "green" if task.status.value == "completed" else "red"
            tui.console.print(f"  {task.status.emoji} [{status_color}]{task.model_info.name}[/{status_color}]")
            tui.console.print(f"     Type: {task.quant_type.display_name}")
            if task.status.value == "completed":
                tui.console.print(f"     Output: {task.output_path.name}")
            else:
                tui.console.print(f"     [red]Error: {task.error}[/red]")
            tui.console.print(f"     [dim]ID: {task.task_id}[/dim]\n")

    tui.console.print("\n[dim]Press Enter to go back...[/dim]")
    tui.prompt("", style="dim")
