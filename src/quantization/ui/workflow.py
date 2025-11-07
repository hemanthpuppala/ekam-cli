"""Main quantization workflow."""

from pathlib import Path
from loguru import logger

from ...cli.prompts import UserExitException
from ...cli.tui_manager import tui
from ...services.model_discovery import ModelDiscoveryService
from ..manager import QuantizationManager
from ..models import QuantizationTask
from .display import (
    ask_background_mode,
    ask_gpu_preference,
    ask_quantization_method,
    ask_vlm_components,
    ask_vlm_quantization_scope,
    confirm_quantization,
    select_model_to_quantize,
    select_quantization_type,
    show_live_progress,
    show_quantization_intro,
)
from .enhanced_display import show_enhanced_live_progress


def run_quantization_workflow(
    model_discovery: ModelDiscoveryService,
    quantization_manager: QuantizationManager,
) -> None:
    """Run the complete quantization workflow with step-by-step navigation.

    This function now uses a step-based implementation that supports:
    - Going back one step (b hotkey)
    - Jumping to main menu (m hotkey)
    - Jumping to home menu (h hotkey)

    Args:
        model_discovery: Model discovery service
        quantization_manager: Quantization manager
    """
    from .workflow_steps import run_step_based_workflow
    from .navigation import NavigationAction

    try:
        result = run_step_based_workflow(model_discovery, quantization_manager)

        # Handle navigation result
        if result == NavigationAction.HOME:
            logger.info("User requested to return to home menu")
            # The caller (app_quantization.py) will handle this
            # For now, just return
            return
        else:
            # Normal completion or return to main menu
            logger.info("Quantization workflow completed or returned to main menu")
            return

    except UserExitException:
        logger.info("User exited quantization workflow")
    except Exception as e:
        logger.error(f"Quantization workflow error: {e}", exc_info=True)
        tui.show_error(f"Unexpected error: {str(e)}")
        tui.prompt("Press Enter to continue...", style="dim")


def show_background_jobs_monitor(quantization_manager: QuantizationManager) -> None:
    """Show interactive monitor for background quantization jobs.

    Args:
        quantization_manager: Quantization manager
    """
    while True:
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

        # Build job list for selection
        job_list = []
        tui.console.print("[bold cyan]📊 Quantization Jobs Monitor[/bold cyan]\n")

        # Show active jobs
        if active_tasks:
            tui.console.print("[bold green]⚙️  Active Jobs:[/bold green]\n")
            for idx, task in enumerate(active_tasks, 1):
                job_list.append(task)
                progress_bar = "█" * int(task.progress / 5) + "▒" * (20 - int(task.progress / 5))
                eta_str = ""
                if task.eta_seconds:
                    mins = int(task.eta_seconds / 60)
                    secs = int(task.eta_seconds % 60)
                    eta_str = f"| ETA: {mins}m {secs}s"

                tui.console.print(f"  [{idx}] {task.status.emoji} [cyan]{task.model_info.name}[/cyan]")
                tui.console.print(f"      {task.quant_type.display_name}")
                tui.console.print(f"      [{progress_bar}] {task.progress:.1f}% {eta_str}\n")

        # Show completed jobs
        completed_tasks = [t for t in all_tasks if t.is_finished]
        if completed_tasks:
            tui.console.print("[bold]✅ Completed Jobs:[/bold]\n")
            for task in completed_tasks[-10:]:  # Show last 10
                job_list.append(task)
                idx = len(job_list)
                status_color = "green" if task.status.value == "completed" else "red"
                elapsed_str = ""
                if task.elapsed_seconds:
                    mins = int(task.elapsed_seconds / 60)
                    secs = int(task.elapsed_seconds % 60)
                    elapsed_str = f"| Time: {mins}m {secs}s"

                tui.console.print(f"  [{idx}] {task.status.emoji} [{status_color}]{task.model_info.name}[/{status_color}]")
                tui.console.print(f"      {task.quant_type.display_name} {elapsed_str}")
                if task.status.value == "completed":
                    tui.console.print(f"      [dim]→ {task.output_path.name}[/dim]")
                    # Show intermediate file info for GGUF conversions
                    if task.intermediate_file and task.intermediate_file.exists():
                        file_size_mb = task.intermediate_file.stat().st_size / (1024 ** 2)
                        tui.console.print(f"      [yellow]ℹ[/yellow]  [dim]Intermediate FP16: {task.intermediate_file.name} ({file_size_mb:.0f} MB)[/dim]")
                        tui.console.print(f"          [dim]You can use or delete it from: {task.intermediate_file}[/dim]\n")
                    else:
                        tui.console.print()
                else:
                    tui.console.print(f"      [red]✗ {task.error}[/red]\n")

        # Show navigation
        tui.console.print("\n[bold]Options:[/bold]")
        tui.console.print(f"  [cyan]1-{len(job_list)}[/cyan] = View detailed job info")
        tui.console.print("  [cyan bold]d[/cyan bold] = Delete completed/failed jobs")
        tui.console.print("  [cyan bold]r[/cyan bold] = Refresh")
        tui.console.print("  [cyan bold]b[/cyan bold] = Go back")
        tui.console.print()

        choice = tui.prompt(f"Choose [1-{len(job_list)}/d/r/b]:", style="cyan").strip().lower()

        if choice in ["b", "back"]:
            return
        elif choice in ["r", "refresh"]:
            continue
        elif choice in ["d", "delete"]:
            # Show deletable jobs and let user select which to delete
            deletable_tasks = [t for t in all_tasks if t.is_finished]
            if not deletable_tasks:
                tui.console.print(f"\n[yellow]No completed/failed jobs to delete[/yellow]")
                tui.prompt("Press Enter to continue...", style="dim")
                continue

            # Show deletable jobs
            tui.console.print("\n[bold yellow]📋 Select Jobs to Delete:[/bold yellow]\n")
            deletable_map = {}
            for idx, task in enumerate(deletable_tasks, 1):
                deletable_map[idx] = task
                status_color = "green" if task.status.value == "completed" else "red"
                tui.console.print(f"  [{idx}] {task.status.emoji} [{status_color}]{task.model_info.name}[/{status_color}] - {task.quant_type.display_name}")

            tui.console.print("\n[dim]Examples: '1 3 5' or '1-3' or 'all' or 'b' to cancel[/dim]")
            delete_choice = tui.prompt("Enter job numbers to delete:", style="cyan").strip().lower()

            if delete_choice in ["b", "back", "cancel"]:
                continue

            # Parse selection
            selected_indices = set()
            if delete_choice == "all":
                selected_indices = set(deletable_map.keys())
            else:
                try:
                    # Parse comma-separated and range notation
                    parts = delete_choice.replace(",", " ").split()
                    for part in parts:
                        if "-" in part:
                            # Range notation: 1-3
                            start, end = map(int, part.split("-"))
                            selected_indices.update(range(start, end + 1))
                        else:
                            # Single number
                            selected_indices.add(int(part))
                except ValueError:
                    tui.show_error("Invalid format. Use numbers like '1 3 5' or '1-3'")
                    tui.prompt("Press Enter to continue...", style="dim")
                    continue

            # Validate and delete
            deleted_count = 0
            for idx in selected_indices:
                if idx in deletable_map:
                    task = deletable_map[idx]
                    if quantization_manager.remove_task(task.task_id):
                        deleted_count += 1

            if deleted_count > 0:
                tui.console.print(f"\n[green]✓ Deleted {deleted_count} job(s)[/green]")
            else:
                tui.console.print(f"\n[yellow]No jobs were deleted[/yellow]")
            tui.prompt("Press Enter to continue...", style="dim")
            continue
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(job_list):
                    show_job_details(job_list[idx - 1], quantization_manager)
                else:
                    tui.show_error(f"Invalid selection. Enter 1-{len(job_list)}")
                    tui.prompt("Press Enter to continue...", style="dim")
            except ValueError:
                tui.show_error("Invalid input. Enter a number or 'b' to go back")
                tui.prompt("Press Enter to continue...", style="dim")


def show_job_details(task: QuantizationTask, quantization_manager: QuantizationManager) -> None:
    """Show comprehensive details for a quantization job.

    Args:
        task: Quantization task to display
        quantization_manager: Quantization manager for additional context
    """
    try:
        tui.clear_screen()

        # Header
        status_color = "green" if task.status.value == "completed" else "red" if task.status.value == "failed" else "yellow"
        tui.console.print(f"[bold {status_color}]{task.status.emoji} Job Details: {task.model_info.name}[/bold {status_color}]\n")

        # Basic Information Section
        tui.console.print("[bold cyan]📋 Basic Information[/bold cyan]")
        tui.console.print(f"  Model Name:     {task.model_info.name}")
        tui.console.print(f"  Model ID:       {task.model_info.model_id}")

        # Handle model_type gracefully (could be enum or string)
        model_type_str = "N/A"
        if hasattr(task.model_info, 'model_type'):
            if hasattr(task.model_info.model_type, 'value'):
                # It's an enum
                model_type_str = task.model_info.model_type.value.upper()
            elif isinstance(task.model_info.model_type, str):
                # It's already a string
                model_type_str = task.model_info.model_type.upper()
            else:
                # Convert to string
                model_type_str = str(task.model_info.model_type).upper()

        tui.console.print(f"  Model Type:     {model_type_str}")
        tui.console.print(f"  Task ID:        {task.task_id}")
        tui.console.print(f"  Status:         {task.status.emoji} {task.status.value.upper()}")
        tui.console.print()

        # Quantization Details Section
        tui.console.print("[bold cyan]🔧 Quantization Details[/bold cyan]")
        tui.console.print(f"  Method:         {task.quant_type.display_name}")
        tui.console.print(f"  Method Family:  {task.quant_type.method_family}")
        tui.console.print(f"  Module Used:    {task.module.display_name}")

        # Show precision conversion
        original_precision = "FP32 (32-bit floating point)"  # Default assumption
        if "fp16" in str(task.model_info.model_id).lower():
            original_precision = "FP16 (16-bit floating point)"

        target_precision = task.quant_type.display_name
        tui.console.print(f"  Conversion:     [yellow]{original_precision}[/yellow] → [green]{target_precision}[/green]")
        tui.console.print(f"  Device:         {'GPU' if task.use_gpu else 'CPU'}")
        tui.console.print(f"  Mode:           {'Background' if task.background else 'Live'}")
        tui.console.print()

        # Timing Information Section
        tui.console.print("[bold cyan]⏱️  Timing Information[/bold cyan]")
        if task.started_at:
            tui.console.print(f"  Started:        {task.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if task.completed_at:
            tui.console.print(f"  Completed:      {task.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if task.elapsed_seconds:
            hours = int(task.elapsed_seconds // 3600)
            mins = int((task.elapsed_seconds % 3600) // 60)
            secs = int(task.elapsed_seconds % 60)
            if hours > 0:
                elapsed_str = f"{hours}h {mins}m {secs}s"
            elif mins > 0:
                elapsed_str = f"{mins}m {secs}s"
            else:
                elapsed_str = f"{secs}s"
            tui.console.print(f"  Elapsed Time:   [bold green]{elapsed_str}[/bold green]")
        if task.eta_seconds and task.is_active:
            tui.console.print(f"  ETA:            {int(task.eta_seconds / 60)}m {int(task.eta_seconds % 60)}s")
        tui.console.print()

        # Progress Section (for active jobs)
        if task.is_active:
            tui.console.print("[bold cyan]📊 Progress[/bold cyan]")
            progress_bar = "█" * int(task.progress / 5) + "▒" * (20 - int(task.progress / 5))
            tui.console.print(f"  [{progress_bar}] {task.progress:.1f}%")
            tui.console.print()

        # File Paths Section
        tui.console.print("[bold cyan]📁 File Paths[/bold cyan]")
        tui.console.print(f"  Output Path:    {task.output_path}")

        # Show intermediate file info for GGUF conversions
        if task.intermediate_file:
            tui.console.print(f"  [yellow]Intermediate:[/yellow]  {task.intermediate_file}")
            if task.intermediate_file.exists():
                int_size_mb = task.intermediate_file.stat().st_size / (1024**2)
                int_size_gb = int_size_mb / 1024
                if int_size_gb >= 0.1:
                    tui.console.print(f"  [yellow]Int. Size:[/yellow]    [green]{int_size_gb:.2f} GB[/green] [dim](FP16 format, can be used or deleted)[/dim]")
                else:
                    tui.console.print(f"  [yellow]Int. Size:[/yellow]    [green]{int_size_mb:.2f} MB[/green] [dim](FP16 format, can be used or deleted)[/dim]")
            else:
                tui.console.print(f"  [yellow]Int. Status:[/yellow]  [dim](already deleted)[/dim]")

        # Check if file exists and get size
        if task.output_path.exists():
            if task.output_path.is_file():
                size_bytes = task.output_path.stat().st_size
                size_gb = size_bytes / (1024**3)
                size_mb = size_bytes / (1024**2)
                if size_gb >= 0.1:
                    tui.console.print(f"  File Size:      [green]{size_gb:.2f} GB[/green]")
                else:
                    tui.console.print(f"  File Size:      [green]{size_mb:.2f} MB[/green]")
                tui.console.print(f"  File Exists:    [green]✓ Yes[/green]")
            elif task.output_path.is_dir():
                # Directory - calculate total size
                total_bytes = sum(f.stat().st_size for f in task.output_path.rglob("*") if f.is_file())
                size_gb = total_bytes / (1024**3)
                tui.console.print(f"  Directory Size: [green]{size_gb:.2f} GB[/green]")
                tui.console.print(f"  Directory Exists: [green]✓ Yes[/green]")
        else:
            tui.console.print(f"  File Exists:    [red]✗ Not found[/red]")

        # Show original model size and compression if available
        try:
            if hasattr(task.model_info, 'size_gb') and task.model_info.size_gb and task.model_info.size_gb > 0:
                tui.console.print(f"  Original Size:  {task.model_info.size_gb:.2f} GB")
                if task.output_path.exists():
                    if task.output_path.is_file():
                        size_gb = task.output_path.stat().st_size / (1024**3)
                    else:
                        total_bytes = sum(f.stat().st_size for f in task.output_path.rglob("*") if f.is_file())
                        size_gb = total_bytes / (1024**3)

                    if size_gb > 0 and task.model_info.size_gb > 0:
                        compression_ratio = ((task.model_info.size_gb - size_gb) / task.model_info.size_gb) * 100
                        tui.console.print(f"  Size Reduction: [bold green]{compression_ratio:.1f}%[/bold green]")
        except Exception as e:
            # Silently skip size comparison if calculation fails
            logger.debug(f"Could not calculate size reduction: {e}")

        tui.console.print()

        # Provider Availability Section
        tui.console.print("[bold cyan]🔌 Provider Availability[/bold cyan]")
        providers = detect_quantized_model_providers(task)
        if providers:
            tui.console.print("  Available in:")
            for provider in providers:
                tui.console.print(f"    [green]✓[/green] {provider}")
        else:
            tui.console.print("  [yellow]⚠ Not yet registered in any provider[/yellow]")
            tui.console.print("  [dim]The model will be available after restarting the app[/dim]")
        tui.console.print()

        # Error Section (if failed)
        if task.status.value == "failed" and task.error:
            tui.console.print("[bold red]❌ Error Details[/bold red]")
            tui.console.print(f"  {task.error}")
            tui.console.print()

        # Actions
        tui.console.print("[bold]Actions:[/bold]")
        if task.status.value == "completed" and task.output_path.exists():
            tui.console.print("  [cyan bold]o[/cyan bold] = Open containing folder")
        tui.console.print("  [cyan bold]b[/cyan bold] = Go back")
        tui.console.print()

        choice = tui.prompt("Choose [o/b]:" if task.status.value == "completed" else "Choose [b]:", style="cyan").strip().lower()

        if choice == "o" and task.status.value == "completed":
            # Open folder in file manager
            import subprocess
            import platform
            folder_path = task.output_path.parent if task.output_path.is_file() else task.output_path
            try:
                if platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", str(folder_path)])
                elif platform.system() == "Windows":
                    subprocess.run(["explorer", str(folder_path)])
                else:  # Linux
                    subprocess.run(["xdg-open", str(folder_path)])
                tui.console.print("[green]✓ Opened folder in file manager[/green]")
                tui.prompt("\nPress Enter to continue...", style="dim")
            except Exception as e:
                tui.show_error(f"Failed to open folder: {e}")
                tui.prompt("Press Enter to continue...", style="dim")

    except Exception as e:
        logger.error(f"Error displaying job details: {e}", exc_info=True)
        tui.show_error(
            f"Failed to display job details:\n{str(e)}\n\n"
            "The job data may be corrupted or incomplete."
        )
        tui.prompt("Press Enter to continue...", style="dim")


def detect_quantized_model_providers(task: QuantizationTask) -> list[str]:
    """Detect which providers can use this quantized model.

    Args:
        task: Quantization task with output information

    Returns:
        List of provider names where model is available
    """
    providers = []

    if not task.output_path.exists():
        return providers

    # Determine provider based on output format
    if task.output_path.is_file():
        # Single file - likely GGUF
        if task.output_path.suffix == ".gguf":
            providers.append("GGUF Provider")
            # GGUF files can also be used by Ollama if imported
            providers.append("Ollama (after import with 'ollama create')")
        elif task.output_path.suffix in [".safetensors", ".bin"]:
            # Could be HuggingFace or Quantized provider
            providers.append("Quantized Provider")
    elif task.output_path.is_dir():
        # Directory with model files - HuggingFace format
        # Check for config.json
        if (task.output_path / "config.json").exists():
            providers.append("HuggingFace Provider")
            providers.append("Quantized Provider")

    # Add method-specific provider info
    method_family = task.quant_type.method_family
    if method_family == "GGUF":
        if "GGUF Provider" not in providers:
            providers.append("GGUF Provider (restart required)")
    elif method_family in ["Generic", "GPTQ", "AWQ", "BitsAndBytes"]:
        if "HuggingFace Provider" not in providers:
            providers.append("HuggingFace Provider (restart required)")

    return providers
