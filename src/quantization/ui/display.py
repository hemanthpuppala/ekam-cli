"""UI components for quantization workflow."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger

from ...cli.prompts import UserExitException, prompt_yes_no
from ...cli.tui_manager import tui
from ...models.model import ModelInfo
from ..models import QuantizationModule, QuantizationRecommendation, QuantizationType

if TYPE_CHECKING:
    from ..manager import QuantizationManager


def show_quantization_intro():
    """Show introduction to quantization feature."""
    tui.clear_screen()
    tui.show_panel(
        """[bold cyan]Model Quantization[/bold cyan]

[bold]What is Quantization?[/bold]
Quantization reduces model size and improves inference speed by using
lower-precision numbers (4-bit, 5-bit, 6-bit, 8-bit instead of 16/32-bit).

[bold green]Benefits:[/bold green]
  • [green]Smaller file size[/green] (50-90% reduction)
  • [green]Faster inference[/green] (2-4x speedup)
  • [green]Lower memory usage[/green] (runs on edge devices)
  • [green]Minimal quality loss[/green] (with proper quantization)

[bold yellow]Current Support:[/bold yellow]
  • Phase 1: GGUF models only (using llama.cpp)
  • Future: HuggingFace and Ollama model conversion

[dim]Press Enter to continue...[/dim]""",
        title="Quantization Overview",
        border_style="cyan",
    )
    tui.prompt("", style="dim")


def select_model_to_quantize(
    quantizable_models: list[ModelInfo],
    quantization_manager: Optional["QuantizationManager"] = None,
) -> Optional[ModelInfo]:
    """Show model selection menu.

    Args:
        quantizable_models: List of models that can be quantized
        quantization_manager: Manager for compatibility checking

    Returns:
        Selected model, or None if user wants to go back

    Raises:
        UserExitException: If user wants to exit
    """
    tui.clear_screen()

    # Show available models (even if empty, show manual option)
    tui.console.print("[bold cyan]Select Model to Quantize[/bold cyan]\n")

    if quantizable_models:
        from ..models.endpoints import ProviderType

        # Group by provider
        gguf_models = [m for m in quantizable_models if m.provider == ProviderType.GGUF]
        hf_models = [m for m in quantizable_models if m.provider == ProviderType.HUGGINGFACE]

        if gguf_models:
            tui.console.print("[dim]GGUF models (direct quantization):[/dim]\n")
            for idx, model in enumerate(gguf_models, 1):
                size_str = f"{model.size_gb:.1f}GB" if model.size_gb else "Unknown size"
                tui.console.print(f"  [{idx}] [cyan]{model.name}[/cyan] ({size_str})")
                tui.console.print(f"      [dim]Provider: {model.provider}[/dim]")

        if hf_models:
            tui.console.print("\n[dim]HuggingFace models (generic quantization or GGUF conversion):[/dim]\n")
            start_idx = len(gguf_models) + 1
            for idx, model in enumerate(hf_models, start_idx):
                size_str = f"{model.size_gb:.1f}GB" if model.size_gb else "Unknown size"
                tui.console.print(f"  [{idx}] [cyan]{model.name}[/cyan] ({size_str})")
                tui.console.print(f"      [dim]Provider: {model.provider}[/dim]")

        tui.console.print(f"\n  [m] [yellow]Manually specify model path[/yellow]")
        tui.console.print(f"  [b] [dim]Go back[/dim]")
        tui.console.print()

        while True:
            choice = tui.prompt(f"Choose [1-{len(quantizable_models)}/m/b]:", style="cyan").strip().lower()

            if choice in ["b", "back", "exit", "quit"]:
                return None

            if choice == "m":
                return handle_manual_model_path(quantization_manager)

            try:
                idx = int(choice)
                if 1 <= idx <= len(quantizable_models):
                    return quantizable_models[idx - 1]
                else:
                    tui.show_error(f"Invalid choice. Enter 1-{len(quantizable_models)}, 'm', or 'b'")
            except ValueError:
                tui.show_error("Invalid input. Enter a number, 'm', or 'b'")
    else:
        # No models found - show manual option only
        tui.show_panel(
            """[bold yellow]No Quantizable Models in Registry[/bold yellow]

No GGUF or HuggingFace models found in your model registry.

[bold]Options:[/bold]
  • Use the manual path option below to specify a model
  • Or install models first using the main menu

[dim]Supported formats:[/dim]
  • GGUF files (.gguf)
  • HuggingFace model directories (with config.json + model files)""",
            title="No Models Available",
            border_style="yellow",
        )

        tui.console.print(f"\n  [m] [yellow]Manually specify model path[/yellow]")
        tui.console.print(f"  [b] [dim]Go back[/dim]")
        tui.console.print()

        while True:
            choice = tui.prompt("Choose [m/b]:", style="cyan").strip().lower()

            if choice in ["b", "back", "exit", "quit"]:
                return None

            if choice == "m":
                return handle_manual_model_path(quantization_manager)

            tui.show_error("Invalid choice. Enter 'm' or 'b'")


def handle_manual_model_path(quantization_manager: Optional["QuantizationManager"]) -> Optional[ModelInfo]:
    """Handle manual model path input with compatibility checking.

    Args:
        quantization_manager: Manager for compatibility checking

    Returns:
        ModelInfo if compatible, None otherwise
    """
    from pathlib import Path as PathLib
    from ..models.endpoints import ProviderType

    tui.clear_screen()
    tui.show_panel(
        """[bold cyan]Manual Model Path[/bold cyan]

Provide the path to a model directory or file.

[bold]Supported formats:[/bold]
  • GGUF files: /path/to/model.gguf
  • HuggingFace directories: /path/to/model/ (with config.json)

[bold]Examples:[/bold]
  • ~/models/llama-7b-q4.gguf
  • ~/models/minicpm-2b-fp16/
  • /Users/name/.cache/huggingface/hub/models--model-name/

[dim]Press Enter with empty input to cancel[/dim]""",
        title="Manual Path Input",
        border_style="cyan",
    )

    while True:
        model_path_str = tui.prompt("Model path:", style="cyan").strip()

        if not model_path_str:
            return None

        # Expand ~ and resolve path
        model_path = PathLib(model_path_str).expanduser().resolve()

        # Check compatibility
        if not quantization_manager:
            tui.show_error("Quantization manager not available")
            return None

        tui.console.print("\n[dim]Checking compatibility...[/dim]\n")

        compat = quantization_manager.check_model_path_compatibility(model_path)

        # Show compatibility results
        tui.clear_screen()
        if compat["compatible"]:
            # Compatible - show options
            methods_str = "\n  • ".join(compat["available_methods"])

            tui.show_panel(
                f"""[bold green]✓ Compatible Model Found[/bold green]

[bold]Path:[/bold] {model_path}
[bold]Type:[/bold] {compat["model_type"].upper()}

[bold]Reason:[/bold]
{compat["reason"]}

[bold]Available Quantization Methods:[/bold]
  • {methods_str}

[dim]Proceeding to quantization options...[/dim]""",
                title="Compatibility Check",
                border_style="green",
            )
            tui.prompt("Press Enter to continue...", style="dim")

            # Create ModelInfo for this manual path
            file_size = 0
            if model_path.is_file():
                file_size = model_path.stat().st_size
            else:
                # Sum all files in directory
                file_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

            size_gb = file_size / (1024 ** 3)

            # Determine provider type
            provider = ProviderType.GGUF if compat["model_type"] == "gguf" else ProviderType.HUGGINGFACE

            # Create ModelInfo
            manual_model = ModelInfo(
                model_id=f"manual/{model_path.name}",
                name=f"{model_path.name} (manual)",
                provider=provider,
                model_type="llm",  # Assume LLM for now
                size_gb=size_gb,
                file_path=str(model_path),
                compatibility="perfect_fit",  # Will be rechecked by system
                is_installed=True,
                metadata={"manual_path": True}
            )

            return manual_model
        else:
            # Not compatible - show reason
            tui.show_panel(
                f"""[bold red]✗ Incompatible Model[/bold red]

[bold]Path:[/bold] {model_path}

[bold]Reason:[/bold]
{compat["reason"]}

[dim]Press Enter to try again or 'b' to go back[/dim]""",
                title="Compatibility Check Failed",
                border_style="red",
            )

            retry = tui.prompt("Try again? [y/n]:", style="yellow").strip().lower()
            if retry not in ["y", "yes"]:
                return None


def ask_gpu_preference() -> bool:
    """Ask user if they want to use GPU for quantization.

    Returns:
        True if GPU should be used

    Raises:
        UserExitException: If user wants to exit
    """
    tui.clear_screen()
    tui.show_panel(
        """[bold cyan]Hardware Selection[/bold cyan]

[bold]Choose processing device:[/bold]

[1] [green]CPU[/green]  - Works on all systems, slower
[2] [cyan]GPU[/cyan]  - Faster (if supported), requires CUDA/Metal

[dim]Note: GGUF quantization with llama.cpp primarily uses CPU.
GPU acceleration is limited for quantization (vs inference).[/dim]""",
        title="GPU Preference",
        border_style="cyan",
    )

    while True:
        choice = tui.prompt("Choose [1/2]:", style="cyan").strip()

        if choice in ["quit", "exit", "back", "b"]:
            raise UserExitException("User cancelled")

        if choice == "1":
            return False
        elif choice == "2":
            return True
        else:
            tui.show_error("Invalid choice. Enter 1 or 2")


def select_quantization_type(
    recommendations: list[QuantizationRecommendation],
) -> Optional[QuantizationRecommendation]:
    """Show quantization type selection with recommendations.

    Args:
        recommendations: List of recommendations

    Returns:
        Selected recommendation, or None if user goes back

    Raises:
        UserExitException: If user wants to exit
    """
    tui.clear_screen()

    tui.console.print("[bold cyan]Select Quantization Type[/bold cyan]\n")
    tui.console.print("[dim]Recommendations based on your system:[/dim]\n")

    for idx, rec in enumerate(recommendations, 1):
        marker = "⭐" if rec.is_recommended else "  "
        quality_bar = "█" * rec.quality_score + "▒" * (10 - rec.quality_score)
        speed_bar = "█" * rec.speed_score + "▒" * (10 - rec.speed_score)

        tui.console.print(f"{marker} [{idx}] [cyan]{rec.quant_type.display_name}[/cyan]")
        tui.console.print(f"      [dim]Size: ~{rec.estimated_size_gb:.1f}GB | Time: ~{rec.estimated_time_minutes:.0f}min[/dim]")
        tui.console.print(f"      Quality: {quality_bar} {rec.quality_score}/10")
        tui.console.print(f"      Speed:   {speed_bar} {rec.speed_score}/10")
        tui.console.print(f"      [dim]Best for: {rec.best_for}[/dim]")

        if rec.is_recommended:
            tui.console.print(f"      [bold green]⭐ RECOMMENDED[/bold green]")

        tui.console.print()

    tui.console.print("  [b] [dim]Go back[/dim]\n")

    while True:
        choice = tui.prompt(f"Choose [1-{len(recommendations)}/b]:", style="cyan").strip().lower()

        if choice in ["b", "back", "exit", "quit"]:
            return None

        try:
            idx = int(choice)
            if 1 <= idx <= len(recommendations):
                return recommendations[idx - 1]
            else:
                tui.show_error(f"Invalid choice. Enter 1-{len(recommendations)} or 'b'")
        except ValueError:
            tui.show_error("Invalid input. Enter a number or 'b'")


def confirm_quantization(
    model_info: ModelInfo,
    recommendation: QuantizationRecommendation,
) -> bool:
    """Show confirmation dialog before starting quantization.

    Args:
        model_info: Model to quantize
        recommendation: Selected quantization

    Returns:
        True if user confirms

    Raises:
        UserExitException: If user wants to exit
    """
    tui.clear_screen()

    message = f"""[bold cyan]Confirm Quantization[/bold cyan]

[bold]Model:[/bold] {model_info.name}
[bold]Original Size:[/bold] {model_info.size_gb:.1f}GB
[bold]Quantization:[/bold] {recommendation.quant_type.display_name}
[bold]Module:[/bold] {recommendation.module.display_name}

[bold]Expected Output:[/bold]
  • Size: ~{recommendation.estimated_size_gb:.1f}GB ({(recommendation.estimated_size_gb / model_info.size_gb * 100):.0f}% of original)
  • Time: ~{recommendation.estimated_time_minutes:.0f} minutes
  • Quality: {recommendation.quality_score}/10
  • Speed: {recommendation.speed_score}/10

[bold]Best for:[/bold] {recommendation.best_for}

{recommendation.reason}"""

    tui.show_panel(message, title="Confirm Quantization", border_style="cyan")

    return prompt_yes_no("Proceed with quantization?", default=True)


def ask_background_mode() -> bool:
    """Ask if user wants to run quantization in background.

    Returns:
        True for background, False for live monitoring

    Raises:
        UserExitException: If user wants to exit
    """
    tui.clear_screen()
    tui.show_panel(
        """[bold cyan]Processing Mode[/bold cyan]

[bold]Choose how to run quantization:[/bold]

[1] [green]Live monitoring[/green]
    • Watch progress in real-time
    • See detailed output
    • Blocks other operations

[2] [cyan]Background[/cyan]
    • Returns to menu immediately
    • Persistent status bar shows progress
    • Can perform other tasks
    • Use [yellow]/background[/yellow] command to monitor

[dim]Recommendation: Use background for long-running quantizations (>10min)[/dim]""",
        title="Processing Mode",
        border_style="cyan",
    )

    while True:
        choice = tui.prompt("Choose [1/2]:", style="cyan").strip()

        if choice in ["quit", "exit", "back", "b"]:
            raise UserExitException("User cancelled")

        if choice == "1":
            return False  # Live
        elif choice == "2":
            return True  # Background
        else:
            tui.show_error("Invalid choice. Enter 1 or 2")


def show_live_progress(task):
    """Show live progress for a quantization task.

    Args:
        task: QuantizationTask being monitored
    """
    import time

    tui.clear_screen()

    tui.console.print(f"[bold cyan]Quantizing: {task.model_info.name}[/bold cyan]")
    tui.console.print(f"[dim]Type: {task.quant_type.display_name} | Output: {task.output_path.name}[/dim]\n")

    last_progress = -1

    while task.is_active:
        if task.progress != last_progress:
            progress_bar = "█" * int(task.progress / 5) + "▒" * (20 - int(task.progress / 5))
            eta_str = ""
            if task.eta_seconds:
                mins = int(task.eta_seconds / 60)
                secs = int(task.eta_seconds % 60)
                eta_str = f" | ETA: {mins}m {secs}s"

            tui.console.print(
                f"\r{task.status.emoji} [{progress_bar}] {task.progress:.1f}%{eta_str}    ",
                end="",
            )
            last_progress = task.progress

        time.sleep(0.5)

    # Final status
    tui.console.print("\n")

    if task.status.value == "completed":
        tui.show_message(
            f"""[bold green]✅ Quantization Completed![/bold green]

[bold]Output:[/bold] {task.output_path}
[bold]Size:[/bold] {task.output_path.stat().st_size / (1024**3):.2f}GB
[bold]Time:[/bold] {task.elapsed_seconds:.0f}s

The quantized model has been saved and is ready to use.""",
            title="Success",
            style="green",
        )
    else:
        tui.show_error(f"Quantization failed: {task.error}")

    tui.prompt("Press Enter to continue...", style="dim")
