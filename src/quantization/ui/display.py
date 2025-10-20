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
  • [green]50-90% smaller file size[/green]
  • [green]2-4x faster inference[/green]
  • [green]Runs on edge devices[/green] (lower memory requirements)
  • [green]Minimal quality loss[/green] (with proper quantization method)

[bold cyan]Supported Methods:[/bold cyan]

[bold]1. Generic Quantization (FP16/INT8/INT4)[/bold]
  • Direct PyTorch/HuggingFace quantization
  • Best for: Quick size reduction, HuggingFace ecosystem
  • Output: .safetensors format

[bold]2. GGUF Conversion + Quantization (Q4/Q5/Q6/Q8)[/bold]
  • Convert HuggingFace models to GGUF format
  • Best for: Maximum compatibility (llama.cpp, Ollama)
  • Output: .gguf format (portable, CPU-optimized)

[bold yellow]Supported Models:[/bold yellow]
  • HuggingFace LLMs (Llama, Mistral, Phi, Qwen, etc.)
  • HuggingFace VLMs (LLaVA, Qwen-VL, etc.) - Generic + GGUF (experimental)
  • GGUF models (direct quantization only)

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
        from rich.table import Table
        from rich.panel import Panel
        from ...models.endpoints import ModelType

        # Categorize by model type (LLM/VLM/Embedding/etc)
        categorized = {}
        for model in quantizable_models:
            model_type = str(model.model_type).upper()
            if model_type not in categorized:
                categorized[model_type] = []
            categorized[model_type].append(model)

        # Define category display order and metadata
        category_info = {
            "LLM": {"name": "Large Language Models", "color": "cyan"},
            "VLM": {"name": "Vision-Language Models", "color": "magenta"},
            "EMBEDDING": {"name": "Embedding Models", "color": "yellow"},
        }
        category_order = ["LLM", "VLM", "EMBEDDING"]

        # Build numbered model list and display by category
        display_ordered_models = []
        idx = 1

        for category in category_order:
            if category not in categorized:
                continue  # Skip categories with no models

            category_meta = category_info.get(category, {"name": category, "color": "white"})

            # Create table for this category
            table = Table(
                show_header=True,
                header_style="bold white",
                expand=True,
                border_style=category_meta["color"],
            )

            table.add_column("#", style="dim", width=3, justify="right")
            table.add_column("Model Name", style="white", no_wrap=False, min_width=20)
            table.add_column("Provider", justify="center", width=10)
            table.add_column("Params", justify="right", width=7)
            table.add_column("Quant", justify="center", width=8)
            table.add_column("RAM", justify="right", width=7)
            table.add_column("Size", justify="right", width=7)
            table.add_column("Fit", justify="center", width=10)

            # Add models in this category
            for model in categorized[category]:
                # Style based on compatibility
                if model.compatibility == "perfect_fit":
                    status_text = "[green]OPTIMAL[/green]"
                    name_style = "white"
                elif model.compatibility == "tight_fit":
                    status_text = "[yellow]TIGHT[/yellow]"
                    name_style = "white"
                else:  # too_large
                    status_text = "[red]LARGE[/red]"
                    name_style = "dim"

                # Format parameters
                if model.params_billions and model.params_billions > 0:
                    params_str = f"{model.params_billions:.1f}B"
                else:
                    params_str = "[dim]?[/dim]"

                # Format quantization
                if model.quantization and model.quantization != "unknown":
                    quant_display = model.quantization
                    if quant_display.startswith("q") and "_" in quant_display:
                        quant_display = quant_display.replace("_", "").upper()
                    elif quant_display in ["fp32", "fp16", "bf16"]:
                        quant_display = quant_display.upper()
                    quant_str = f"[yellow]{quant_display}[/yellow]"
                else:
                    quant_str = "[dim]?[/dim]"

                # Format RAM estimate
                if model.ram_gb and model.ram_gb > 0:
                    ram_str = f"{model.ram_gb:.1f}GB"
                else:
                    ram_str = "[dim]?[/dim]"

                # Format file size
                size_str = f"{model.size_gb:.1f}GB"

                # Provider name
                provider_str = str(model.provider).upper() if hasattr(model.provider, 'value') else str(model.provider).upper()

                table.add_row(
                    str(idx),
                    f"[{name_style}]{model.name}[/{name_style}]",
                    provider_str,
                    params_str,
                    quant_str,
                    ram_str,
                    size_str,
                    status_text,
                )

                display_ordered_models.append(model)
                idx += 1

            # Wrap table in panel with category title
            panel = Panel(
                table,
                title=f"[bold {category_meta['color']}]{category_meta['name']}[/bold {category_meta['color']}]",
                border_style=category_meta["color"],
                expand=True,
            )

            tui.console.print(panel)
            tui.console.print()  # Add spacing between categories

        tui.console.print(f"  [m] [yellow]Manually specify model path[/yellow]")
        tui.console.print(f"  [b] [dim]Go back[/dim] | [h] [dim]Home[/dim]")
        tui.console.print()

        while True:
            choice = tui.prompt(f"Choose [1-{len(display_ordered_models)}/m/b/h]:", style="cyan").strip().lower()

            # Check for navigation hotkeys (h only - b is back, m is manual path)
            from .navigation import NavigationException, NavigationAction

            if choice in ["h", "home"]:
                raise NavigationException(NavigationAction.HOME, "User requested home menu")

            if choice in ["b", "back", "exit", "quit"]:
                return None

            if choice == "m":
                return handle_manual_model_path(quantization_manager)

            try:
                idx = int(choice)
                if 1 <= idx <= len(display_ordered_models):
                    # Return from display-ordered list to match what user sees
                    selected_model = display_ordered_models[idx - 1]
                    logger.debug(f"User selected index {idx}: {selected_model.name} ({selected_model.provider})")
                    return selected_model
                else:
                    tui.show_error(f"Invalid choice. Enter 1-{len(display_ordered_models)}, 'm' (manual), 'b', or 'h'")
            except ValueError:
                tui.show_error("Invalid input. Enter a number, 'm' (manual), 'b', or 'h'")
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
        tui.console.print(f"  [b] [dim]Go back[/dim] | [h] [dim]Home[/dim]")
        tui.console.print()

        while True:
            choice = tui.prompt("Choose [m/b/h]:", style="cyan").strip().lower()

            # Check for navigation hotkeys (h only - b is back, m is manual path)
            from .navigation import NavigationException, NavigationAction

            if choice in ["h", "home"]:
                raise NavigationException(NavigationAction.HOME, "User requested home menu")

            if choice in ["b", "back", "exit", "quit"]:
                return None

            if choice == "m":
                return handle_manual_model_path(quantization_manager)

            tui.show_error("Invalid choice. Enter 'm' (manual), 'b', or 'h'")


def handle_manual_model_path(quantization_manager: Optional["QuantizationManager"]) -> Optional[ModelInfo]:
    """Handle manual model path input with compatibility checking.

    Args:
        quantization_manager: Manager for compatibility checking

    Returns:
        ModelInfo if compatible, None otherwise
    """
    from pathlib import Path as PathLib
    from ...models.endpoints import ProviderType

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

            # Import required types
            from ...models.endpoints import ModelType, EndpointType, CompatibilityStatus

            # Create ModelInfo with all required fields
            # Use the actual path as model_id so providers can load it
            manual_model = ModelInfo(
                model_id=str(model_path),  # Use full path as ID for loading
                name=f"{model_path.name} (manual)",
                provider=provider,
                model_type=ModelType.LLM,  # Assume LLM for quantization
                size_gb=size_gb,
                capabilities=[EndpointType.TEXT],  # Default to text-only for LLMs
                compatibility=CompatibilityStatus.PERFECT_FIT,  # Will be assessed later
                compatibility_message=f"Manual model path: {model_path}",
                is_installed=True,
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


def ask_quantization_method(model_info: ModelInfo) -> str:
    """Ask user to choose quantization method for HuggingFace models.

    Args:
        model_info: Model to be quantized

    Returns:
        "generic", "advanced", or "gguf_conversion"

    Raises:
        UserExitException: If user wants to exit
    """
    from ...models.endpoints import ProviderType, ModelType

    # Only show this menu for HuggingFace models
    if model_info.provider != ProviderType.HUGGINGFACE:
        return "gguf"  # GGUF models use direct GGUF quantization

    # Check if CUDA GPU is available (required for Advanced methods)
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    # Check if it's a VLM
    is_vlm = model_info.model_type == ModelType.VLM

    tui.clear_screen()

    # Add VLM-specific notes
    model_type_str = "Vision-Language Model (VLM)" if is_vlm else "Language Model (LLM)"
    vlm_note = ""
    if is_vlm:
        if has_cuda:
            vlm_note = (
                "\n[bold yellow]🎨 VLM Detected[/bold yellow]\n"
                "  • Has vision encoder + language decoder\n"
                "  • GGUF conversion NOT supported (llama.cpp limitation)\n"
                "  • [green]✓ Advanced 4-bit quantization available (CUDA detected)[/green]\n"
            )
        else:
            vlm_note = (
                "\n[bold yellow]🎨 VLM Detected[/bold yellow]\n"
                "  • Has vision encoder + language decoder\n"
                "  • GGUF conversion NOT supported (llama.cpp limitation)\n"
                "  • [yellow]ℹ Advanced methods require CUDA GPU (not available)[/yellow]\n"
            )

    # Detect platform for system info
    import platform
    system_name = platform.system()
    machine = platform.machine()

    if system_name == "Darwin":
        platform_desc = "macOS"
        if machine in ["arm64", "aarch64"]:
            platform_desc += " (Apple Silicon)"
            is_apple_silicon = True
        else:
            platform_desc += " (Intel)"
            is_apple_silicon = False
    elif "arm" in machine.lower() or "aarch" in machine.lower():
        platform_desc = "ARM CPU"
        is_apple_silicon = False
    else:
        platform_desc = "CPU"
        is_apple_silicon = False

    # Check for MLX and OpenVINO
    has_mlx = False
    has_openvino = False
    try:
        import mlx.core  # noqa
        has_mlx = True
    except ImportError:
        pass
    try:
        import openvino  # noqa
        has_openvino = True
    except ImportError:
        pass

    # Build header with system info
    if has_cuda:
        system_header = f"""[bold green]✓ System: CUDA GPU Detected[/bold green]
All quantization methods available on your system."""
    else:
        system_header = f"""[bold yellow]⚠ System: {platform_desc} - No CUDA GPU[/bold yellow]
Advanced methods disabled (require NVIDIA CUDA GPU)."""

    # Always show all 3 options, but gray out incompatible ones
    if has_cuda:
        # CUDA available - all methods enabled
        menu_text = f"""[bold cyan]Choose Quantization Method[/bold cyan]

{system_header}

[bold]Model:[/bold] {model_info.name} ({model_info.size_gb:.1f} GB)
[bold]Type:[/bold] {model_type_str}{vlm_note}

[bold]1. Generic Quantization[/bold] [dim](FP16/INT8/INT4)[/dim] [green]✓ Available[/green]
  • Simple PyTorch quantization
  • Works on CPU and GPU
  • Output: .safetensors format
  • Best for: Quick size reduction, all devices
  • {"Quantizes entire VLM (vision + language)" if is_vlm else "Stable for all models"}

[bold]2. Advanced 4-bit Quantization[/bold] [dim](GPTQ/AWQ/BnB)[/dim] [green]✓ Available[/green]
  • High-quality 4-bit quantization
  • Requires CUDA GPU (detected ✓)
  • Output: .safetensors format
  • Best for: Maximum quality at 4-bit, production use
  • {"Works with VLMs (quantizes entire model)" if is_vlm else "Works with all LLMs"}
  • Slower processing time

{"[bold]3. GGUF Conversion[/bold] [dim](Q4_K_M/Q5_K_M/Q6_K/Q8_0)[/dim] [green]✓ Available[/green]" if not is_vlm else "[dim][bold]3. GGUF Conversion[/bold] [dim](Q4_K_M/Q5_K_M/Q6_K/Q8_0)[/dim] [red]✗ Disabled[/red][/dim]"}
  {"• Maximum compatibility (llama.cpp, Ollama)" if not is_vlm else "[dim]• Maximum compatibility (llama.cpp, Ollama)[/dim]"}
  {"• CPU-optimized for inference" if not is_vlm else "[dim]• CPU-optimized for inference[/dim]"}
  {"• Portable across platforms" if not is_vlm else "[dim]• Portable across platforms[/dim]"}
  {"• Output: .gguf format" if not is_vlm else "[dim]• Output: .gguf format[/dim]"}
  {"• Takes longer (includes conversion step)" if not is_vlm else "[dim]• Takes longer (includes conversion step)[/dim]"}
  {"• Stable for LLMs" if not is_vlm else "[dim]• [red]NOT SUPPORTED for VLMs[/red] (llama.cpp limitation)[/dim]"}

{"[bold]4. MLX Quantization[/bold] [dim](4-bit for Apple Silicon)[/dim] [green]✓ Available[/green]" if is_apple_silicon and has_mlx else "[dim][bold]4. MLX Quantization[/bold] [dim](4-bit for Apple Silicon)[/dim] [red]✗ Disabled[/red][/dim]"}
  {"• Native Metal GPU acceleration" if is_apple_silicon and has_mlx else "[dim]• Native Metal GPU acceleration[/dim]"}
  {"• 85x faster inference on Mac" if is_apple_silicon and has_mlx else "[dim]• 85x faster inference on Mac[/dim]"}
  {"• VLM: Quantizes entire model (both vision + language)" if is_vlm and is_apple_silicon and has_mlx else "[dim]• VLM: Quantizes entire model (both vision + language)[/dim]"}
  {"• Output: MLX format" if is_apple_silicon and has_mlx else "[dim]• Output: MLX format[/dim]"}
  {"• " if not (is_apple_silicon and has_mlx) else ""}{"[red]Requires: macOS + Apple Silicon + pip install mlx mlx-lm mlx-vlm[/red]" if not (is_apple_silicon and has_mlx) else ""}

{"[bold]5. OpenVINO Quantization[/bold] [dim](INT4/INT8 for Intel)[/dim] [green]✓ Available[/green]" if has_openvino else "[dim][bold]5. OpenVINO Quantization[/bold] [dim](INT4/INT8 for Intel)[/dim] [red]✗ Disabled[/red][/dim]"}
  {"• Optimized for Intel CPUs (AVX-512, VNNI, AMX)" if has_openvino else "[dim]• Optimized for Intel CPUs (AVX-512, VNNI, AMX)[/dim]"}
  {"• Supports Intel iGPU acceleration" if has_openvino else "[dim]• Supports Intel iGPU acceleration[/dim]"}
  {"• VLM: Vision encoder + language decoder supported" if has_openvino else "[dim]• VLM: Vision encoder + language decoder supported[/dim]"}
  {"• Output: OpenVINO IR format" if has_openvino else "[dim]• Output: OpenVINO IR format[/dim]"}
  {"• " if has_openvino else ""}{"[red]Requires: pip install openvino optimum[openvino][/red]" if not has_openvino else ""}

[bold]Navigation:[/bold]
  [b] Go back | [m] Main menu | [h] Home

[dim]Recommendation: {"Use Generic (1), Advanced (2), or MLX (4)" if is_vlm else "Use Advanced (2), MLX (4), or GGUF (3)"}[/dim]"""
        # Determine valid choices
        base_choices = ["1", "2"] if is_vlm else ["1", "2", "3"]
        if is_apple_silicon and has_mlx:
            base_choices.append("4")
        if has_openvino:
            base_choices.append("5")
        valid_choices = base_choices
    else:
        # No CUDA - show all 3 options but gray out Advanced
        menu_text = f"""[bold cyan]Choose Quantization Method[/bold cyan]

{system_header}

[bold]Model:[/bold] {model_info.name} ({model_info.size_gb:.1f} GB)
[bold]Type:[/bold] {model_type_str}{vlm_note}

[bold]1. Generic Quantization[/bold] [dim](FP16 only)[/dim] [green]✓ Available[/green]
  • Simple PyTorch quantization
  • Works on Mac CPU/Metal
  • Output: .safetensors format (HuggingFace compatible)
  • 50% size reduction (e.g., 4GB → 2GB)
  • Fast process (~1-2 minutes per GB)
  • {"Quantizes entire VLM (vision + language)" if is_vlm else "Stable for all models"}
  • [cyan]💡 Recommended for Mac users[/cyan]

[dim][bold]2. Advanced 4-bit Quantization[/bold] [dim](GPTQ/AWQ/BnB)[/dim] [red]✗ Disabled[/red]
  • High-quality 4-bit quantization
  • [red]Requires NVIDIA CUDA GPU (not available on Mac)[/red]
  • Your system: {platform_desc} with Metal GPU
  • Disabled methods: Generic INT8/INT4, BitsAndBytes, GPTQ, AWQ
  • 💡 Use Generic FP16 or GGUF instead[/dim]

{"[bold]3. GGUF Conversion[/bold] [dim](Q4_K_M/Q5_K_M/Q6_K/Q8_0)[/dim] [green]✓ Available[/green]" if not is_vlm else "[dim][bold]3. GGUF Conversion[/bold] [dim](Q4_K_M/Q5_K_M/Q6_K/Q8_0)[/dim] [red]✗ Disabled[/red][/dim]"}
  {"• Maximum compatibility (llama.cpp, Ollama)" if not is_vlm else "[dim]• Maximum compatibility (llama.cpp, Ollama)[/dim]"}
  {"• CPU-optimized for inference" if not is_vlm else "[dim]• CPU-optimized for inference[/dim]"}
  {"• Multiple bit depths available:" if not is_vlm else "[dim]• Multiple bit depths available:[/dim]"}
    {"- Q4_K_M: ~50% of original (good quality)" if not is_vlm else "[dim]- Q4_K_M: ~50% of original (good quality)[/dim]"}
    {"- Q5_K_M: ~60% of original (better quality)" if not is_vlm else "[dim]- Q5_K_M: ~60% of original (better quality)[/dim]"}
    {"- Q8_0: ~90% of original (minimal quality loss)" if not is_vlm else "[dim]- Q8_0: ~90% of original (minimal quality loss)[/dim]"}
  {"• Output: .gguf format" if not is_vlm else "[dim]• Output: .gguf format[/dim]"}
  {"• Slower process (includes conversion step)" if not is_vlm else "[dim]• Slower process (includes conversion step)[/dim]"}
  {"• Works for pure LLMs" if not is_vlm else "[dim]• [red]NOT SUPPORTED for VLMs[/red] (llama.cpp doesn't support VLM architectures)[/dim]"}

{"[bold]4. MLX Quantization[/bold] [dim](4-bit for Apple Silicon)[/dim] [green]✓ Available[/green]" if is_apple_silicon and has_mlx else "[dim][bold]4. MLX Quantization[/bold] [dim](4-bit for Apple Silicon)[/dim] [red]✗ Disabled[/red][/dim]"}
  {"• Native Metal GPU acceleration" if is_apple_silicon and has_mlx else "[dim]• Native Metal GPU acceleration[/dim]"}
  {"• 85x faster inference on Mac" if is_apple_silicon and has_mlx else "[dim]• 85x faster inference on Mac[/dim]"}
  {"• VLM: Quantizes entire model (both vision + language)" if is_vlm and is_apple_silicon and has_mlx else "[dim]• VLM: Quantizes entire model (both vision + language)[/dim]"}
  {"• Output: MLX format" if is_apple_silicon and has_mlx else "[dim]• Output: MLX format[/dim]"}
  {"• " if not (is_apple_silicon and has_mlx) else ""}{"[red]Requires: macOS + Apple Silicon + pip install mlx mlx-lm mlx-vlm[/red]" if not (is_apple_silicon and has_mlx) else ""}

{"[bold]5. OpenVINO Quantization[/bold] [dim](INT4/INT8 for Intel)[/dim] [green]✓ Available[/green]" if has_openvino else "[dim][bold]5. OpenVINO Quantization[/bold] [dim](INT4/INT8 for Intel)[/dim] [red]✗ Disabled[/red][/dim]"}
  {"• Optimized for Intel CPUs (AVX-512, VNNI, AMX)" if has_openvino else "[dim]• Optimized for Intel CPUs (AVX-512, VNNI, AMX)[/dim]"}
  {"• Supports Intel iGPU acceleration" if has_openvino else "[dim]• Supports Intel iGPU acceleration[/dim]"}
  {"• VLM: Vision encoder + language decoder supported" if has_openvino else "[dim]• VLM: Vision encoder + language decoder supported[/dim]"}
  {"• Output: OpenVINO IR format" if has_openvino else "[dim]• Output: OpenVINO IR format[/dim]"}
  {"• " if has_openvino else ""}{"[red]Requires: pip install openvino optimum[openvino][/red]" if not has_openvino else ""}

[bold]Navigation:[/bold]
  [b] Go back | [m] Main menu | [h] Home

{'─'*60}
[dim]💡 [bold]Recommendation for your {platform_desc} system:[/bold]
{"• MLX (4) recommended for Mac, or Generic FP16 (1)" if is_vlm and is_apple_silicon and has_mlx else "• Use Generic FP16 (option 1)"}
• Advanced methods require NVIDIA CUDA GPU (use on Windows/Linux with NVIDIA GPU)[/dim]"""

        # Determine valid choices based on system
        base_choices = ["1"]  # Generic always available
        if not is_vlm:
            base_choices.append("3")  # GGUF for LLMs
        if is_apple_silicon and has_mlx:
            base_choices.append("4")  # MLX
        if has_openvino:
            base_choices.append("5")  # OpenVINO
        valid_choices = base_choices

    tui.show_panel(menu_text, title="Quantization Method Selection", border_style="cyan")

    while True:
        # Dynamic prompt based on available options
        choice = tui.prompt("Choose [1/2/3/4/5/b/m/h]:", style="cyan").strip().lower()

        # Check for navigation hotkeys
        from .navigation import check_navigation_input, NavigationException, NavigationAction

        nav_action = check_navigation_input(choice)
        if nav_action:
            if nav_action == NavigationAction.BACK:
                raise UserExitException("User cancelled")
            elif nav_action == NavigationAction.MAIN_MENU:
                raise NavigationException(NavigationAction.MAIN_MENU, "User requested main menu")
            elif nav_action == NavigationAction.HOME:
                raise NavigationException(NavigationAction.HOME, "User requested home menu")

        # Legacy support for "quit" and "exit"
        if choice in ["quit", "exit"]:
            raise UserExitException("User cancelled")

        # Check if choice is valid for this system/model
        if choice not in valid_choices and choice not in ["quit", "exit", "back", "b"]:
            # User selected a disabled option - provide helpful feedback
            if choice == "2" and not has_cuda:
                tui.show_error(
                    f"[red]✗ Option 2 (Advanced 4-bit) is disabled on {platform_desc}[/red]\n"
                    f"Requires NVIDIA CUDA GPU. Use option 1 (Generic FP16) instead."
                )
                continue
            elif choice == "3" and is_vlm:
                tui.show_error(
                    "[red]✗ Option 3 (GGUF) is disabled for VLMs[/red]\n"
                    "llama.cpp doesn't support VLM architectures. Use option 1 (Generic FP16) instead."
                )
                continue
            else:
                tui.show_error(f"Invalid choice. Valid options: {', '.join(valid_choices)}, or 'b' to go back")
                continue

        # Valid choice - process it
        if choice == "1":
            logger.info(f"User selected Generic quantization for {model_info.name}")
            return "generic"
        elif choice == "2":
            logger.info(f"User selected Advanced 4-bit quantization for {model_info.name}")
            return "advanced"
        elif choice == "3":
            logger.info(f"User selected GGUF conversion for {model_info.name}")
            return "gguf_conversion"
        elif choice == "4":
            logger.info(f"User selected MLX quantization for {model_info.name}")
            return "mlx"
        elif choice == "5":
            logger.info(f"User selected OpenVINO quantization for {model_info.name}")
            return "openvino"


def ask_vlm_quantization_scope(model_info: ModelInfo, quant_method: str = None) -> str:
    """Ask user if they want standard or component-level quantization for VLMs.

    Only shown for Vision-Language Models (VLMs).

    Args:
        model_info: VLM model to quantize
        quant_method: Selected quantization method (for MLX warning)

    Returns:
        "standard" or "component_level"

    Raises:
        UserExitException: If user wants to exit
    """
    from ...models.model import ModelType

    # Only ask for VLMs
    if model_info.model_type != ModelType.VLM:
        return "standard"

    # Build warning message for MLX
    mlx_warning = ""
    if quant_method == "mlx":
        mlx_warning = "\n[bold yellow]⚠ MLX Note:[/bold yellow] MLX requires full model quantization (option 1 only).\n"

    tui.clear_screen()
    tui.show_panel(
        f"""[bold cyan]VLM Quantization Scope[/bold cyan]

[bold]Model:[/bold] {model_info.name}
[bold]Type:[/bold] Vision-Language Model (VLM)
{mlx_warning}
[bold]Choose quantization scope:[/bold]

[bold]1. Standard Quantization[/bold] [dim](Recommended)[/dim]
  • Quantize entire model (vision + language)
  • Simpler, faster setup
  • Most common use case
  {"• [green]✓ Works with MLX[/green]" if quant_method == "mlx" else ""}

[bold]2. Component-Level Control[/bold] [dim](Advanced)[/dim]
  • Choose specific components to quantize
  • Vision encoder only, language decoder only, or both
  • Maximum flexibility for optimization
  {"• [red]✗ NOT supported by MLX[/red]" if quant_method == "mlx" else ""}

[bold]Navigation:[/bold]
  [b] Go back | [m] Main menu | [h] Home

[dim]Tip: Most users should choose Standard (option 1).
{"For MLX, only Standard is supported." if quant_method == "mlx" else "Component-level is for advanced optimization scenarios."}[/dim]""",
        title="VLM Quantization Scope",
        border_style="cyan",
    )

    while True:
        choice = tui.prompt("Choose [1/2/b/m/h]:", style="cyan").strip().lower()

        # Check for navigation hotkeys
        from .navigation import check_navigation_input, NavigationException, NavigationAction

        nav_action = check_navigation_input(choice)
        if nav_action:
            if nav_action == NavigationAction.BACK:
                raise UserExitException("User cancelled")
            elif nav_action == NavigationAction.MAIN_MENU:
                raise NavigationException(NavigationAction.MAIN_MENU, "User requested main menu")
            elif nav_action == NavigationAction.HOME:
                raise NavigationException(NavigationAction.HOME, "User requested home menu")

        if choice in ["quit", "exit"]:
            raise UserExitException("User cancelled")

        if choice == "1":
            logger.info("User selected standard VLM quantization (entire model)")
            return "standard"
        elif choice == "2":
            # Check if MLX - block component-level for MLX
            if quant_method == "mlx":
                tui.show_error(
                    "[red]✗ Component-level quantization is NOT supported by MLX[/red]\n\n"
                    "MLX can only quantize the entire VLM model (both vision + language).\n\n"
                    "Options:\n"
                    "  • Select option 1 (Standard) to continue with MLX\n"
                    "  • Press 'b' to go back and choose a different quantization method\n"
                    "    (Generic or OpenVINO support component-level quantization)"
                )
                continue

            logger.info("User selected component-level VLM quantization")
            return "component_level"
        else:
            tui.show_error("Invalid choice. Enter 1, 2, 'b', 'm', or 'h'")
            continue


def ask_vlm_components(model_info: ModelInfo, quant_method: str) -> str:
    """Ask user which VLM components to quantize.

    Only shown when component-level control is selected.

    Args:
        model_info: VLM model to quantize
        quant_method: Selected quantization method ("mlx", "openvino", etc.)

    Returns:
        "vision", "language", or "both"

    Raises:
        UserExitException: If user wants to exit
    """
    tui.clear_screen()

    # MLX: Force full model quantization (both components)
    if quant_method == "mlx":
        tui.show_panel(
            f"""[bold yellow]MLX Quantization - Component Selection[/bold yellow]

[bold]Model:[/bold] {model_info.name}

[bold red]⚠ MLX VLM Limitation[/bold red]

MLX quantization for VLMs requires quantizing the ENTIRE model:
  • [yellow]⚠ Vision Encoder + Language Decoder[/yellow] (both required)
  • [red]✗ Component-level extraction NOT supported[/red]

[bold yellow]Auto-Selected: Both Components (Full Model)[/bold yellow]
  • Vision encoder will be quantized to 4-bit
  • Language decoder will be quantized to 4-bit
  • MLX converts the entire VLM model to 4-bit format

[bold]Architecture:[/bold]
  Vision Encoder (4-bit) → Projection → Language Decoder (4-bit)

[bold]Note:[/bold] For language-only quantization, use Generic FP16 method instead.

[bold]Navigation:[/bold]
  [Enter] Continue | [b] Go back | [m] Main menu | [h] Home

[dim]Press Enter to proceed with full model quantization...[/dim]""",
            title="MLX Component Selection (Auto)",
            border_style="yellow",
        )

        choice = tui.prompt("[Enter/b/m/h]:", style="cyan").strip().lower()

        from .navigation import check_navigation_input, NavigationException, NavigationAction
        nav_action = check_navigation_input(choice)
        if nav_action:
            if nav_action == NavigationAction.BACK:
                raise UserExitException("User cancelled")
            elif nav_action == NavigationAction.MAIN_MENU:
                raise NavigationException(NavigationAction.MAIN_MENU, "User requested main menu")
            elif nav_action == NavigationAction.HOME:
                raise NavigationException(NavigationAction.HOME, "User requested home menu")

        if choice in ["quit", "exit", "b"]:
            raise UserExitException("User cancelled")

        logger.info("MLX: Auto-selected full model quantization (both components - forced)")
        return "both"

    # OpenVINO and others: Show all options (gray out MLX-incompatible choices)
    # Note: This shouldn't normally be reached for MLX since we block at scope selection,
    # but if it is, we gray out options 1 and 2
    is_mlx_blocked = quant_method == "mlx"

    tui.show_panel(
        f"""[bold cyan]Select VLM Components to Quantize[/bold cyan]

[bold]Model:[/bold] {model_info.name}
{f"[bold yellow]⚠ Method:[/bold yellow] {quant_method.upper()} - Component extraction not supported" if is_mlx_blocked else ""}

[bold]VLM Architecture:[/bold]
  Vision Encoder → Projection → Language Decoder

[bold]Choose components to quantize:[/bold]

{"[dim]" if is_mlx_blocked else ""}[bold]1. Vision Encoder Only[/bold]{"[/dim]" if is_mlx_blocked else ""} {"[red]✗ NOT supported by MLX[/red]" if is_mlx_blocked else ""}
  {"[dim]" if is_mlx_blocked else ""}• Quantize image/video processing component
  • Language decoder stays in original precision
  • Use case: Reduce vision model size, keep language quality{"[/dim]" if is_mlx_blocked else ""}

{"[dim]" if is_mlx_blocked else ""}[bold]2. Language Decoder Only[/bold]{"[/dim]" if is_mlx_blocked else ""} {"[red]✗ NOT supported by MLX[/red]" if is_mlx_blocked else "[dim][green]⭐ RECOMMENDED for VLMs![/green][/dim]"}
  {"[dim]" if is_mlx_blocked else ""}• Quantize text generation component (largest part)
  • Vision encoder stays in FULL PRECISION (FP32/FP16)
  • {"[red]MLX doesn't support component extraction[/red]" if is_mlx_blocked else "[green]✓ Smaller model size (~70% reduction)[/green]"}
  • {"[/dim]" if is_mlx_blocked else ""}{"[green]✓ Preserved vision quality (no vision quantization)[/green]" if not is_mlx_blocked else ""}
  {"[green]✓ Best quality/size trade-off for VLMs![/green]" if not is_mlx_blocked else ""}{"[/dim]" if is_mlx_blocked else ""}

[bold]3. Both Components[/bold] {"[green]✓ Works with MLX[/green]" if is_mlx_blocked else ""}
  • Quantize entire VLM (vision + language)
  • Maximum size reduction
  • Most common use case
  {"• [green]ONLY option for MLX quantization[/green]" if is_mlx_blocked else ""}

[bold]Navigation:[/bold]
  [b] Go back | [m] Main menu | [h] Home

[dim]Tip: {"For MLX, only 'Both Components' (option 3) is supported." if is_mlx_blocked else "Most users should choose Both (option 3). Component-specific quantization is for advanced scenarios."}[/dim]""",
        title="VLM Component Selection",
        border_style="cyan",
    )

    while True:
        choice = tui.prompt("Choose [1/2/3/b/m/h]:", style="cyan").strip().lower()

        # Check for navigation hotkeys
        from .navigation import check_navigation_input, NavigationException, NavigationAction

        nav_action = check_navigation_input(choice)
        if nav_action:
            if nav_action == NavigationAction.BACK:
                raise UserExitException("User cancelled")
            elif nav_action == NavigationAction.MAIN_MENU:
                raise NavigationException(NavigationAction.MAIN_MENU, "User requested main menu")
            elif nav_action == NavigationAction.HOME:
                raise NavigationException(NavigationAction.HOME, "User requested home menu")

        if choice in ["quit", "exit"]:
            raise UserExitException("User cancelled")

        if choice == "1":
            # Check if blocked by MLX
            if is_mlx_blocked:
                tui.show_error(
                    "[red]✗ Vision encoder only is NOT supported by MLX[/red]\n\n"
                    "MLX can only quantize the entire VLM model (both vision + language).\n\n"
                    "Options:\n"
                    "  • Select option 3 (Both Components) to continue with MLX\n"
                    "  • Press 'b' to go back and choose a different quantization method"
                )
                continue

            logger.info("User selected vision encoder only")
            return "vision"
        elif choice == "2":
            # Check if blocked by MLX
            if is_mlx_blocked:
                tui.show_error(
                    "[red]✗ Language decoder only is NOT supported by MLX[/red]\n\n"
                    "MLX can only quantize the entire VLM model (both vision + language).\n\n"
                    "Options:\n"
                    "  • Select option 3 (Both Components) to continue with MLX\n"
                    "  • Press 'b' to go back and choose a different quantization method\n"
                    "    (Generic FP16 or OpenVINO support component-level quantization)"
                )
                continue

            logger.info("User selected language decoder only")
            return "language"
        elif choice == "3":
            logger.info("User selected both components")
            return "both"
        else:
            tui.show_error("Invalid choice. Enter 1, 2, 3, 'b', 'm', or 'h'")
            continue


def ask_gpu_preference() -> bool:
    """Ask user if they want to use GPU for quantization.

    If no CUDA GPU is available, automatically returns False (CPU).
    Only asks if CUDA GPU is detected.

    Returns:
        True if GPU should be used (only if CUDA available)

    Raises:
        UserExitException: If user wants to exit
    """
    # Check if CUDA GPU is available
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    # If no CUDA GPU, automatically use CPU (don't ask)
    if not has_cuda:
        from loguru import logger
        import platform

        # Detect specific platform
        system_name = platform.system()
        machine = platform.machine()

        if system_name == "Darwin":
            platform_desc = "macOS"
            if machine in ["arm64", "aarch64"]:
                platform_desc += " (Apple Silicon - Metal GPU)"
                gpu_type = "Apple Metal GPU"
            else:
                platform_desc += " (Intel)"
                gpu_type = "CPU only"
        else:
            platform_desc = f"{system_name} - CPU"
            gpu_type = "CPU only"

        logger.info(f"No CUDA GPU detected - using CPU for quantization (Platform: {platform_desc})")
        tui.clear_screen()
        tui.show_panel(
            f"""[bold yellow]⚠ Hardware Detection: Mac/CPU System[/bold yellow]

[bold]Your System:[/bold] {platform_desc}
[bold]GPU Type:[/bold] {gpu_type}
[bold]CUDA Support:[/bold] Not available (NVIDIA GPU required)

[bold]What this means:[/bold]
Your Mac uses Apple's Metal GPU, which is excellent for running models
but doesn't support CUDA-based quantization libraries (BitsAndBytes,
GPTQ, AWQ). These advanced methods have been automatically disabled.

[bold green]✓ What works on your Mac:[/bold green]
  • Generic FP16 quantization (50% size reduction)
  • GGUF quantization (Q4_K_M, Q8_0, etc. - for compatible models)
  • All models can be quantized with FP16
  • Fast, CPU/Metal-optimized workflows

[bold red]✗ What's disabled on Mac:[/bold red]
  • Generic INT8/INT4 (requires CUDA + bitsandbytes)
  • BitsAndBytes 4-bit (requires CUDA)
  • GPTQ 4-bit (requires CUDA)
  • AWQ 4-bit (requires CUDA)

[dim]💡 Note: Mac Metal GPU is used for inference performance, but
   quantization uses CPU-based PyTorch operations.[/dim]

[green]Press Enter to continue...[/green]""",
            title="Mac/CPU Quantization Mode",
            border_style="yellow",
        )
        tui.prompt("")
        return False

    # CUDA GPU available - ask user preference
    tui.clear_screen()
    tui.show_panel(
        """[bold cyan]Hardware Selection[/bold cyan]

[bold]CUDA GPU detected![/bold]

Choose processing device:

[1] [green]CPU[/green]  - Works on all systems, slower
[2] [cyan]GPU (CUDA)[/cyan]  - Faster, enables all quantization methods

[bold]Navigation:[/bold]
  [b] Go back | [m] Main menu | [h] Home

[dim]Note: GGUF quantization with llama.cpp primarily uses CPU.
BitsAndBytes/GPTQ/AWQ require GPU (option 2).[/dim]""",
        title="GPU Preference",
        border_style="cyan",
    )

    while True:
        choice = tui.prompt("Choose [1/2/b/m/h]:", style="cyan").strip().lower()

        # Check for navigation hotkeys
        from .navigation import check_navigation_input, NavigationException, NavigationAction

        nav_action = check_navigation_input(choice)
        if nav_action:
            if nav_action == NavigationAction.BACK:
                raise UserExitException("User cancelled")
            elif nav_action == NavigationAction.MAIN_MENU:
                raise NavigationException(NavigationAction.MAIN_MENU, "User requested main menu")
            elif nav_action == NavigationAction.HOME:
                raise NavigationException(NavigationAction.HOME, "User requested home menu")

        if choice in ["quit", "exit"]:
            raise UserExitException("User cancelled")

        if choice == "1":
            return False
        elif choice == "2":
            return True
        else:
            tui.show_error("Invalid choice. Enter 1, 2, 'b', 'm', or 'h'")


def select_quantization_type(
    recommendations: list[QuantizationRecommendation],
) -> Optional[QuantizationRecommendation]:
    """Show quantization type selection with recommendations.

    Shows ALL quantization types, with unavailable ones grayed out.

    Args:
        recommendations: List of recommendations (including unavailable ones)

    Returns:
        Selected recommendation, or None if user goes back

    Raises:
        UserExitException: If user wants to exit
    """
    tui.clear_screen()

    tui.console.print("[bold cyan]Select Quantization Type[/bold cyan]\n")
    tui.console.print("[dim]All available options for this method (grayed = not available on your system):[/dim]\n")

    # Track which indices are selectable
    valid_indices = []

    for idx, rec in enumerate(recommendations, 1):
        # Determine styling based on availability
        if not rec.is_available:
            # Grayed out - not available
            marker = "[dim]✗"
            name_style = f"[dim]{rec.quant_type.display_name} - [red]NOT AVAILABLE[/red][/dim]"
            quality_bar = "[dim]" + ("█" * rec.quality_score + "▒" * (10 - rec.quality_score)) + "[/dim]"
            speed_bar = "[dim]" + ("█" * rec.speed_score + "▒" * (10 - rec.speed_score)) + "[/dim]"
        elif rec.is_recommended:
            # Recommended
            marker = "⭐"
            name_style = f"[cyan]{rec.quant_type.display_name}[/cyan]"
            quality_bar = "█" * rec.quality_score + "▒" * (10 - rec.quality_score)
            speed_bar = "█" * rec.speed_score + "▒" * (10 - rec.speed_score)
            valid_indices.append(idx)
        else:
            # Available but not recommended
            marker = "  "
            name_style = f"[cyan]{rec.quant_type.display_name}[/cyan]"
            quality_bar = "█" * rec.quality_score + "▒" * (10 - rec.quality_score)
            speed_bar = "█" * rec.speed_score + "▒" * (10 - rec.speed_score)
            valid_indices.append(idx)

        tui.console.print(f"{marker} [{idx}] {name_style}")

        if rec.is_available:
            tui.console.print(f"      [dim]Size: ~{rec.estimated_size_gb:.1f}GB | Time: ~{rec.estimated_time_minutes:.0f}min[/dim]")
            tui.console.print(f"      Quality: {quality_bar} {rec.quality_score}/10")
            tui.console.print(f"      Speed:   {speed_bar} {rec.speed_score}/10")
            tui.console.print(f"      [dim]Best for: {rec.best_for}[/dim]")

            if rec.is_recommended:
                tui.console.print(f"      [bold green]⭐ RECOMMENDED[/bold green]")
        else:
            # Show why it's unavailable
            tui.console.print(f"      [dim red]Reason: {rec.unavailable_reason}[/dim red]")
            tui.console.print(f"      [dim]Would be: ~{rec.estimated_size_gb:.1f}GB, Quality {rec.quality_score}/10[/dim]")

        tui.console.print()

    tui.console.print("  [b] [dim]Go back[/dim] | [m] [dim]Main menu[/dim] | [h] [dim]Home[/dim]\n")

    while True:
        choice = tui.prompt(f"Choose [1-{len(recommendations)}/b/m/h]:", style="cyan").strip().lower()

        # Check for navigation hotkeys
        from .navigation import check_navigation_input, NavigationException, NavigationAction

        nav_action = check_navigation_input(choice)
        if nav_action:
            if nav_action == NavigationAction.BACK:
                return None
            elif nav_action == NavigationAction.MAIN_MENU:
                raise NavigationException(NavigationAction.MAIN_MENU, "User requested main menu")
            elif nav_action == NavigationAction.HOME:
                raise NavigationException(NavigationAction.HOME, "User requested home menu")

        # Legacy support
        if choice in ["exit", "quit"]:
            return None

        try:
            idx = int(choice)
            if 1 <= idx <= len(recommendations):
                selected_rec = recommendations[idx - 1]

                # Check if this option is available
                if not selected_rec.is_available:
                    tui.show_error(
                        f"[red]✗ {selected_rec.quant_type.display_name} is not available on your system[/red]\n"
                        f"{selected_rec.unavailable_reason}\n"
                        f"Please select an available option ({', '.join(map(str, valid_indices))})."
                    )
                    continue

                return selected_rec
            else:
                tui.show_error(f"Invalid choice. Enter 1-{len(recommendations)}, 'b', 'm', or 'h'")
        except ValueError:
            tui.show_error("Invalid input. Enter a number, 'b', 'm', or 'h'")


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

[bold]Navigation:[/bold]
  [b] Go back | [m] Main menu | [h] Home

[dim]Recommendation: Use background for long-running quantizations (>10min)[/dim]""",
        title="Processing Mode",
        border_style="cyan",
    )

    while True:
        choice = tui.prompt("Choose [1/2/b/m/h]:", style="cyan").strip().lower()

        # Check for navigation hotkeys
        from .navigation import check_navigation_input, NavigationException, NavigationAction

        nav_action = check_navigation_input(choice)
        if nav_action:
            if nav_action == NavigationAction.BACK:
                raise UserExitException("User cancelled")
            elif nav_action == NavigationAction.MAIN_MENU:
                raise NavigationException(NavigationAction.MAIN_MENU, "User requested main menu")
            elif nav_action == NavigationAction.HOME:
                raise NavigationException(NavigationAction.HOME, "User requested home menu")

        if choice in ["quit", "exit"]:
            raise UserExitException("User cancelled")

        if choice == "1":
            return False  # Live
        elif choice == "2":
            return True  # Background
        else:
            tui.show_error("Invalid choice. Enter 1 or 2")


def show_live_progress(task):
    """Show live progress for a quantization task with real-time updates.

    Args:
        task: QuantizationTask being monitored
    """
    import time
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

    # Create rich progress bar
    progress_display = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:.1f}%"),
        TimeRemainingColumn(),
        expand=True
    )

    task_progress = progress_display.add_task(
        f"Quantizing {task.model_info.name}",
        total=100
    )

    # Create panel with info
    info_text = f"""[bold cyan]Model:[/bold cyan] {task.model_info.name}
[bold cyan]Type:[/bold cyan] {task.quant_type.display_name}
[bold cyan]Output:[/bold cyan] {task.output_path.name}

[dim]Press Ctrl+C to cancel (not recommended - may corrupt output)[/dim]"""

    with Live(
        Panel(progress_display, title="🔧 Quantization Progress", border_style="cyan"),
        refresh_per_second=4,  # 4 updates per second = 250ms refresh
        console=tui.console
    ) as live:
        last_progress = -1

        while task.is_active:
            if task.progress != last_progress:
                progress_display.update(task_progress, completed=task.progress)
                last_progress = task.progress

            # Faster polling for smoother updates
            time.sleep(0.25)

    # Final status
    tui.console.print("\n")

    if task.status.value == "completed":
        # Calculate actual size (handle both files and directories)
        quantized_size_gb = 0.0
        if task.output_path.exists():
            if task.output_path.is_file():
                # Single file (e.g., GGUF)
                quantized_size_gb = task.output_path.stat().st_size / (1024 ** 3)
            elif task.output_path.is_dir():
                # Directory (e.g., HuggingFace format) - sum all files
                total_bytes = sum(f.stat().st_size for f in task.output_path.rglob("*") if f.is_file())
                quantized_size_gb = total_bytes / (1024 ** 3)

        tui.show_message(
            f"""[bold green]✅ Quantization Completed![/bold green]

[bold]Output:[/bold] {task.output_path}
[bold]Size:[/bold] {quantized_size_gb:.2f}GB
[bold]Time:[/bold] {task.elapsed_seconds:.0f}s

The quantized model has been saved and is ready to use.""",
            title="Success",
            style="green",
        )
    else:
        tui.show_error(f"Quantization failed: {task.error}")

    tui.prompt("Press Enter to continue...", style="dim")
