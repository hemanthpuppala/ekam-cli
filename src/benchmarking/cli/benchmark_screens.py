"""
Benchmark configuration screens for TUI.

Provides interactive screens for users to configure benchmarks:
- Model type selection
- Suite selection
- Model selection (multi-select)
- Endpoint selection
- Test data configuration
- Parameters configuration
- Execution mode selection
- Configuration review
"""

from typing import List, Dict, Optional, Any
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.benchmarking.models.metric_types import SuiteType, ModelType, ExecutionMode
from src.benchmarking.cli.prompt_helpers import get_selection, get_confirmation, get_text_input
from loguru import logger


console = Console()


def _extract_model_display_name(model_id: str) -> str:
    """
    Extract a clean display name from a model ID.

    Args:
        model_id: Full model ID (could be path, quantized prefix, or simple name)

    Returns:
        Clean display name for the table
    """
    from pathlib import Path

    # Handle quantized models with prefixes
    if model_id.startswith("quantized:"):
        # Format: quantized:gguf:/path/to/model.gguf or quantized:hf:/path/to/model_int4
        parts = model_id.split(":", 2)
        if len(parts) == 3:
            quant_type = parts[1]  # gguf or hf
            path_part = parts[2]

            if quant_type == "hf":
                # For HF quantized models, just get the last directory name
                return Path(path_part).name
            else:
                # For GGUF quantized models, get the filename
                return Path(path_part).name

    # Handle full file paths (GGUF files)
    if "/" in model_id or "\\" in model_id:
        return Path(model_id).name

    # Otherwise return as-is (Ollama models, HF models without paths)
    return model_id


def _show_step_header(step_number: int, total_steps: int, title: str, subtitle: str = ""):
    """
    Show step information header with consistent formatting.

    Note: Ekam-CLI header is now shown automatically by tui.clear_screen()

    Args:
        step_number: Current step number
        total_steps: Total number of steps
        title: Step title
        subtitle: Optional subtitle text
    """
    from src.cli.tui_manager import tui

    # Show step heading with consistent format
    step_heading = f"Step {step_number}/{total_steps}: {title}"
    tui.show_step_heading(step_heading)

    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")
        console.print()


def show_model_type_selection(current_value: Optional[ModelType] = None) -> Optional[tuple]:
    """
    Display model type selection screen with arrow-key navigation.

    Args:
        current_value: Previously selected ModelType for pre-filling

    Returns:
        Tuple of (ModelType, action) where action is "next", "back", or "cancel"
        Returns (None, "cancel") if cancelled
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui

    tui.clear_screen()
    _show_step_header(1, 7, "Model Type Selection", "Select the type of models you want to benchmark")

    # Build arrow-key options
    options = [
        ("llm", "[green]LLM[/green]", "Large Language Models (text-only)"),
        ("vlm", "[magenta]VLM[/magenta]", "Vision Language Models (image + text)")
    ]

    # Add navigation options (no Previous on first step)
    options.append(("__cancel", "[red]Cancel[/red]", "Cancel benchmark configuration"))

    # Pre-select current value if provided
    # (arrow selector always starts at index 0, but we show the previous choice in the description)
    if current_value:
        current_str = "llm" if current_value == ModelType.LLM else "vlm"
        console.print(f"[dim]Current selection: {current_str.upper()}[/dim]")
        console.print()

    # Get user selection
    choice = professional_prompt.get_arrow_selection(
        options=options,
        title="Select Model Type",
        instructions="Use ↑/↓ arrows to navigate, Enter to select, q to cancel"
    )

    # Handle cancellation
    if choice is None or choice == "__cancel":
        return (None, "cancel")

    # Convert choice to ModelType
    model_type = ModelType.LLM if choice == "llm" else ModelType.VLM

    return (model_type, "next")


def show_suite_selection(current_value: Optional[SuiteType] = None) -> Optional[tuple]:
    """
    Display suite selection screen with arrow-key navigation.

    Args:
        current_value: Previously selected SuiteType for pre-filling

    Returns:
        Tuple of (SuiteType, action) where action is "next", "back", or "cancel"
        Returns (None, action) if cancelled/back
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui

    tui.clear_screen()
    _show_step_header(2, 7, "Benchmark Suite Selection", "Choose which benchmark suite to run")

    # Show informational table
    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Suite", style="cyan", width=20)
    table.add_column("Description", width=40)
    table.add_column("Duration", style="yellow", width=12)

    table.add_row(
        "Complete System",
        "Run all suites for comprehensive analysis",
        "~45-60 min"
    )
    table.add_row(
        "Speed & Throughput",
        "Measure latency and tokens per second",
        "~10-15 min"
    )
    table.add_row(
        "Resource Efficiency",
        "Monitor CPU, GPU, memory consumption",
        "~10-15 min"
    )
    table.add_row(
        "Quality & Consistency",
        "Test output consistency across runs",
        "~15-20 min"
    )
    table.add_row(
        "Stress & Endurance",
        "Long-running stability and degradation test",
        "~30-60 min"
    )

    console.print(table)
    console.print()

    # Show current selection if provided
    if current_value:
        console.print(f"[dim]Current selection: {current_value.display_name()}[/dim]")
        console.print()

    # Build arrow-key options with detailed descriptions
    options = [
        ("complete", "[cyan]Complete System Analysis[/cyan]", "All suites (~45-60 min) - Comprehensive performance analysis"),
        ("speed", "[green]Speed & Throughput[/green]", "Latency and tokens/sec (~10-15 min) - Fast execution metrics"),
        ("resources", "[yellow]Resource Efficiency[/yellow]", "CPU/GPU/memory usage (~10-15 min) - Hardware utilization"),
        ("quality", "[magenta]Quality & Consistency[/magenta]", "Output consistency (~15-20 min) - Response reliability"),
        ("stress", "[red]Stress & Endurance[/red]", "Long-running stability (~30-60 min) - Degradation testing")
    ]

    # Add navigation options
    options.append(("__previous", "[blue]← Previous[/blue]", "Go back to model type selection"))
    options.append(("__cancel", "[red]Cancel[/red]", "Cancel benchmark configuration"))

    # Get user selection
    choice = professional_prompt.get_arrow_selection(
        options=options,
        title="Select Benchmark Suite",
        instructions="Use ↑/↓ arrows to navigate, Enter to select"
    )

    # Handle navigation
    if choice is None or choice == "__cancel":
        return (None, "cancel")
    elif choice == "__previous":
        return (None, "back")

    # Convert choice to SuiteType
    suite_type = SuiteType.from_string(choice)

    return (suite_type, "next")


def show_model_selection(model_type: ModelType, session_manager) -> Optional[List[str]]:
    """
    Display model selection screen (multi-select).

    Args:
        model_type: Type of models to show
        session_manager: SessionManager to query available models

    Returns:
        List of selected model IDs or None if cancelled
    """
    from src.cli.tui_manager import tui

    tui.clear_screen()
    _show_step_header(
        3, 7,
        "Model Selection",
        f"Select one or more {model_type.value.upper()} models to benchmark"
    )

    # Get available models from session manager
    try:
        # Discover all models and filter by type
        all_models = session_manager.discover_models()

        # Filter by model type
        available_models = []
        for model in all_models:
            if hasattr(model, 'model_type'):
                if (model_type == ModelType.LLM and str(model.model_type).upper() == 'LLM') or \
                   (model_type == ModelType.VLM and str(model.model_type).upper() == 'VLM'):
                    model_id = model.model_id
                    display_name = _extract_model_display_name(model_id)
                    available_models.append({
                        "model_id": model_id,
                        "display_name": display_name,
                        "provider": model.provider.value if hasattr(model.provider, 'value') else str(model.provider)
                    })

        if not available_models:
            console.print(f"[red]No {model_type.value.upper()} models available![/red]")
            console.print("\nPlease ensure models are loaded before benchmarking.")
            return None

        # Display available models
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("#", style="cyan", width=6, no_wrap=True)
        table.add_column("Model ID", style="green")
        table.add_column("Provider", style="yellow", width=17, no_wrap=True)

        for idx, model in enumerate(available_models, 1):
            display_name = model.get("display_name", model.get("model_id", "unknown"))
            provider = model.get("provider", "unknown")
            table.add_row(str(idx), display_name, provider)

        console.print(table)
        console.print()

        # Multi-select prompt
        console.print("[yellow]Enter model numbers separated by commas (e.g., 1,3,4)[/yellow]")
        console.print("[yellow]Or 'all' to select all models[/yellow]")

        selection = get_text_input("Select models", allow_empty=False)

        if selection is None:
            return None

        # Parse selection
        if selection.lower() == "all":
            selected_models = [m.get("model_id") for m in available_models]
        else:
            try:
                indices = [int(x.strip()) for x in selection.split(",")]
                selected_models = []
                for idx in indices:
                    if 1 <= idx <= len(available_models):
                        selected_models.append(available_models[idx - 1].get("model_id"))
                    else:
                        console.print(f"[red]Invalid index: {idx}[/red]")
                        return None
            except ValueError:
                console.print("[red]Invalid input format![/red]")
                return None

        if not selected_models:
            console.print("[red]No models selected![/red]")
            return None

        console.print(f"\n[green]Selected {len(selected_models)} model(s):[/green]")
        for model_id in selected_models:
            display_name = _extract_model_display_name(model_id)
            console.print(f"  • {display_name}")

        return selected_models

    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        console.print(f"[red]Error getting models: {e}[/red]")
        return None


def show_endpoint_selection(
    models: List[str],
    model_type: ModelType,
    current_value: Optional[Dict[str, List[str]]] = None
) -> Optional[tuple]:
    """
    Display endpoint selection screen with arrow-key navigation.

    Args:
        models: List of selected model IDs
        model_type: Type of models
        current_value: Previously selected endpoints map for pre-filling

    Returns:
        Tuple of (endpoints_map, action) where action is "next", "back", or "cancel"
        Returns (None, action) if cancelled/back
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui

    tui.clear_screen()
    _show_step_header(
        4, 7, "Endpoint Selection",
        f"Select ONE endpoint to test across all {len(models)} model(s)\nThis ensures fair comparison by testing all models on the same task"
    )


    # Extract current endpoint if provided
    current_endpoint = None
    if current_value and len(current_value) > 0:
        # Get the endpoint from the first model (all models use the same endpoint)
        first_model_endpoints = list(current_value.values())[0]
        if first_model_endpoints:
            current_endpoint = first_model_endpoints[0]
            console.print(f"[dim]Current selection: {current_endpoint}[/dim]")
            console.print()

    # Build arrow-key options based on model type
    if model_type == ModelType.LLM:
        console.print("[bold]LLM Endpoints:[/bold]")
        console.print("[dim]• chat: Modern instruct models, multi-turn conversations[/dim]")
        console.print("[dim]• completion: Base models, simple text generation[/dim]")
        console.print()

        options = [
            ("text/chat", "[green]Text/Chat[/green]", "Conversational chat (system/user/assistant roles)"),
            ("text/completion", "[yellow]Text/Completion[/yellow]", "Direct completion (raw prompt → response)")
        ]
    else:  # VLM
        console.print("[bold]VLM Endpoints:[/bold]")
        console.print("[dim]Model-specific implementations used when available[/dim]")
        console.print("[dim]Falls back to generic implementation if needed[/dim]")
        console.print()

        options = [
            ("vision/qa", "[magenta]Visual Q&A[/magenta]", "Ask questions about images"),
            ("vision/caption", "[cyan]Image Captioning[/cyan]", "Describe image content"),
            ("vision/detect", "[yellow]Object Detection[/yellow]", "Find and locate objects"),
            ("vision/point", "[blue]Object Pointing[/blue]", "Get object coordinates")
        ]

    # Add navigation options
    options.append(("__previous", "[blue]← Previous[/blue]", "Go back to model selection"))
    options.append(("__cancel", "[red]Cancel[/red]", "Cancel benchmark configuration"))

    # Get user selection
    choice = professional_prompt.get_arrow_selection(
        options=options,
        title=f"Select Endpoint for {len(models)} Model(s)",
        instructions="Use ↑/↓ arrows to navigate, Enter to select"
    )

    # Handle navigation
    if choice is None or choice == "__cancel":
        return (None, "cancel")
    elif choice == "__previous":
        return (None, "back")

    # Apply the same endpoint to all models
    endpoints_map = {model_id: [choice] for model_id in models}

    console.print(f"\n[green]✓ Endpoint '{choice}' will be used for all {len(models)} model(s)[/green]")

    return (endpoints_map, "next")


def _manual_prompt_entry_wizard(
    num_prompts: int,
    model_type: ModelType,
    selected_endpoint: Optional[str] = None,
    existing_prompts: Optional[List[str]] = None,
    existing_images: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Interactive wizard for manual one-by-one prompt (and image) entry.

    Args:
        num_prompts: Number of prompts to collect
        model_type: LLM or VLM (determines if images are needed)
        selected_endpoint: VLM endpoint (if applicable)
        existing_prompts: Previously entered prompts (for editing)
        existing_images: Previously entered images (for VLM editing)

    Returns:
        Dict with 'prompts' and optionally 'images' lists, or None if cancelled
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui
    import json
    from pathlib import Path

    # Initialize storage
    prompts = existing_prompts[:] if existing_prompts else []
    images = existing_images[:] if existing_images else []
    autosave_path = Path.cwd() / ".ekam" / "prompt_entry_autosave.json"

    # Helper: Auto-save progress
    def autosave():
        autosave_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_prompts": num_prompts,
            "model_type": model_type.value,
            "endpoint": selected_endpoint,
            "prompts": prompts,
            "images": images if model_type == ModelType.VLM else []
        }
        with open(autosave_path, 'w') as f:
            json.dump(data, f, indent=2)

    # Helper: Validate image path
    def validate_image_path(path_str: str) -> Optional[Path]:
        """Validate and resolve image path using CWD -> project root -> absolute."""
        if not path_str:
            return None

        # Try CWD first
        cwd_path = Path.cwd() / path_str
        if cwd_path.exists() and cwd_path.is_file():
            return cwd_path

        # Try project root (parent of src/)
        project_root = Path(__file__).parent.parent.parent.parent
        root_path = project_root / path_str
        if root_path.exists() and root_path.is_file():
            return root_path

        # Try as absolute path
        abs_path = Path(path_str)
        if abs_path.exists() and abs_path.is_file():
            return abs_path

        return None

    # STEP 1: Collect prompts one by one
    current_idx = len(prompts)
    while current_idx < num_prompts:
        tui.clear_screen()
        console.print(f"[bold magenta]Manual Prompt Entry Wizard[/bold magenta]")
        console.print(f"[dim]Progress: {current_idx + 1} of {num_prompts}[/dim]")
        console.print()

        # For VLM, ask for image first
        if model_type == ModelType.VLM:
            console.print(f"[bold cyan]Prompt #{current_idx + 1} - Image Path:[/bold cyan]")
            console.print("[dim]Provide full path to image file[/dim]")
            console.print("[dim]Commands: /back (previous), /runs (restart count), /default (skip to defaults)[/dim]")
            console.print()

            image_input = professional_prompt.get_text_input(
                prompt=f"Image path for prompt {current_idx + 1}",
                default_value="",
                allow_empty=False,
                multiline=False
            )

            if image_input is None:
                if current_idx > 0:
                    current_idx -= 1
                    prompts.pop()
                    if images:
                        images.pop()
                    autosave()
                continue

            # Handle commands
            if image_input.lower() == "/back":
                if current_idx > 0:
                    current_idx -= 1
                    prompts.pop()
                    if images:
                        images.pop()
                    autosave()
                continue
            elif image_input.lower() == "/runs":
                return {"_command": "/runs"}
            elif image_input.lower() == "/default":
                return {"_command": "/default"}

            # Validate image path
            validated_path = validate_image_path(image_input)
            if not validated_path:
                console.print(f"[red]✗ Invalid image path: {image_input}[/red]")
                console.print("[dim]Press Enter to retry...[/dim]")
                input()
                continue

            # Check image format
            valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
            if validated_path.suffix.lower() not in valid_extensions:
                console.print(f"[red]✗ Unsupported image format: {validated_path.suffix}[/red]")
                console.print(f"[dim]Supported: {', '.join(valid_extensions)}[/dim]")
                console.print("[dim]Press Enter to retry...[/dim]")
                input()
                continue

            images.append(str(validated_path))
            console.print(f"[green]✓ Image validated: {validated_path.name}[/green]")
            console.print()

        # Ask for prompt text
        console.print(f"[bold cyan]Prompt #{current_idx + 1} - Text:[/bold cyan]")
        console.print("[dim]Enter your prompt (supports multiline, paste-friendly)[/dim]")
        console.print("[dim]Commands: /back, /runs, /default[/dim]")
        console.print()

        prompt_input = professional_prompt.get_text_input(
            prompt=f"Prompt {current_idx + 1}",
            default_value="",
            allow_empty=False,
            multiline=True
        )

        if prompt_input is None:
            if current_idx > 0:
                current_idx -= 1
                prompts.pop()
                if model_type == ModelType.VLM and images:
                    images.pop()
                autosave()
            continue

        # Handle commands
        if prompt_input.lower() == "/back":
            if current_idx > 0:
                current_idx -= 1
                prompts.pop()
                if model_type == ModelType.VLM and images:
                    images.pop()
                autosave()
            continue
        elif prompt_input.lower() == "/runs":
            autosave()
            return {"_command": "/runs"}
        elif prompt_input.lower() == "/default":
            autosave()
            return {"_command": "/default"}

        # Store prompt
        prompts.append(prompt_input)
        autosave()
        current_idx += 1

    # STEP 2: Show summary and confirm
    while True:
        tui.clear_screen()
        console.print(f"[bold magenta]Prompt Entry Summary[/bold magenta]")
        console.print(f"[dim]Total prompts: {len(prompts)}[/dim]")
        console.print()

        # Show scrollable list
        console.print("[bold]Entered Prompts:[/bold]")
        console.print()

        for idx, prompt in enumerate(prompts, 1):
            # Truncate long prompts for display
            display_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt
            display_prompt = display_prompt.replace("\n", " ")

            if model_type == ModelType.VLM and idx <= len(images):
                image_name = Path(images[idx - 1]).name
                console.print(f"  [cyan]{idx:2d}.[/cyan] [dim]({image_name})[/dim] {display_prompt}")
            else:
                console.print(f"  [cyan]{idx:2d}.[/cyan] {display_prompt}")

        console.print()
        console.print("[bold]What would you like to do?[/bold]")
        console.print()

        # Confirmation menu
        menu_options = [
            ("confirm", "[green]Yes - Continue[/green]", "Accept these prompts and proceed"),
            ("edit", "[yellow]Edit - Modify a prompt[/yellow]", "Edit a specific prompt from the list"),
            ("back", "[blue]No - Go back[/blue]", "Discard and return to previous step"),
        ]

        choice = professional_prompt.get_arrow_selection(
            options=menu_options,
            title="Confirm prompt entry",
            instructions="Y = Yes, E = Edit, N = No"
        )

        if choice == "confirm" or choice == "y":
            # Clean up autosave
            if autosave_path.exists():
                autosave_path.unlink()

            result = {"prompts": prompts}
            if model_type == ModelType.VLM:
                result["images"] = images
            return result

        elif choice == "edit" or choice == "e":
            # Ask which prompt to edit
            console.print()
            console.print(f"[bold cyan]Enter prompt number to edit (1-{len(prompts)}):[/bold cyan]")

            edit_input = professional_prompt.get_text_input(
                prompt="Prompt number",
                default_value="",
                allow_empty=True
            )

            if not edit_input:
                continue

            try:
                edit_idx = int(edit_input) - 1
                if 0 <= edit_idx < len(prompts):
                    # Re-enter wizard at specific index
                    prompts_temp = prompts[:edit_idx]
                    images_temp = images[:edit_idx] if model_type == ModelType.VLM else []

                    result = _manual_prompt_entry_wizard(
                        num_prompts=num_prompts,
                        model_type=model_type,
                        selected_endpoint=selected_endpoint,
                        existing_prompts=prompts_temp,
                        existing_images=images_temp
                    )

                    if result is None or result.get("_command"):
                        return result
                    else:
                        return result
                else:
                    console.print(f"[red]✗ Invalid prompt number[/red]")
                    console.print("[dim]Press Enter to continue...[/dim]")
                    input()
            except ValueError:
                console.print(f"[red]✗ Invalid input: {edit_input}[/red]")
                console.print("[dim]Press Enter to continue...[/dim]")
                input()

        elif choice == "back" or choice == "n":
            autosave()
            return None


def show_num_prompts_config(
    suite_name: str,
    current_value: Optional[int] = None
) -> Optional[tuple]:
    """
    Display number of prompts configuration screen.

    Args:
        suite_name: Name of the benchmark suite being configured
        current_value: Previously selected number of prompts for pre-filling

    Returns:
        Tuple of (num_prompts, action) where action is "next", "back", or "cancel"
        Returns (None, action) if cancelled/back
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui

    # Suite-specific defaults
    suite_defaults = {
        "speed": 20,
        "resources": 10,
        "quality": 20,
        "stress": 3,
        "complete": 10,
    }

    default_value = suite_defaults.get(suite_name.lower(), 10)
    num_prompts = current_value if current_value is not None else default_value

    while True:
        tui.clear_screen()
        _show_step_header(5, 8, "Number of Prompts", f"Configure how many prompts to use for {suite_name.title()} suite")

        console.print("[bold]Prompt Configuration:[/bold]")
        console.print()
        console.print(f"  Each prompt will be run exactly once (1 prompt = 1 run)")
        if suite_name.lower() == "quality":
            console.print(f"  [dim](Quality suite will generate multiple outputs per prompt)[/dim]")
        elif suite_name.lower() == "stress":
            console.print(f"  [dim](Stress suite will cycle through prompts for the duration)[/dim]")
        console.print()
        console.print(f"  [cyan]Current value:[/cyan] {num_prompts} prompts")
        console.print(f"  [dim]Range: 1-50 | Default: {default_value}[/dim]")
        console.print()

        # Menu options
        menu_options = [
            ("edit", "[yellow]Edit number of prompts[/yellow]", "Specify how many prompts to use (1-50)"),
            ("use_default", "[blue]Use default[/blue]", f"Reset to default value ({default_value})"),
            ("__separator", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ""),
            ("continue", "[green]Continue to Step 6 →[/green]", "Proceed to test data selection"),
            ("__previous", "[blue]← Previous[/blue]", "Go back to endpoint selection"),
            ("__cancel", "[red]Cancel[/red]", "Cancel benchmark configuration"),
        ]

        # Get user selection
        choice = professional_prompt.get_arrow_selection(
            options=menu_options,
            title="Configure number of prompts",
            instructions="Use ↑/↓ arrows to navigate, Enter to select, or type /back, /default"
        )

        # Handle navigation
        if choice is None or choice == "__cancel":
            return (None, "cancel")
        elif choice == "__previous":
            return (None, "back")
        elif choice == "__separator":
            continue
        elif choice == "use_default":
            num_prompts = default_value
            continue
        elif choice == "continue":
            return (num_prompts, "next")
        elif choice == "edit":
            # Prompt for new value
            console.print()
            console.print("[bold cyan]Enter number of prompts (1-50):[/bold cyan]")
            console.print("[dim]Type /back to return, /default for default value[/dim]")

            user_input = professional_prompt.get_text_input(
                prompt="Number of prompts",
                default_value=str(num_prompts),
                allow_empty=False
            )

            if user_input is None:
                continue

            # Handle special commands
            if user_input.lower() == "/back":
                return (None, "back")
            elif user_input.lower() == "/default":
                num_prompts = default_value
                continue
            elif user_input.lower() == "/runs":
                # Shortcut to go back to this screen (already here)
                continue

            # Validate numeric input
            try:
                new_value = int(user_input)
                if 1 <= new_value <= 50:
                    num_prompts = new_value
                else:
                    console.print(f"[red]✗ Value must be between 1 and 50[/red]")
                    console.print("[dim]Press Enter to continue...[/dim]")
                    input()
            except ValueError:
                console.print(f"[red]✗ Invalid number: {user_input}[/red]")
                console.print("[dim]Press Enter to continue...[/dim]")
                input()


def show_test_data_selection(
    model_type: ModelType,
    num_prompts: int,
    suite_name: str,
    selected_endpoint: Optional[str] = None,
    current_value: Optional[Dict[str, Any]] = None
) -> Optional[tuple]:
    """
    Display test data configuration screen with arrow-key navigation and manual entry.

    Args:
        model_type: Type of models being tested
        num_prompts: Number of prompts to collect (from step 5)
        suite_name: Name of the suite (for default prompt selection)
        selected_endpoint: Optional VLM endpoint for structured test data
        current_value: Previously selected test data config for pre-filling

    Returns:
        Tuple of (test_data_dict, action) where action is "next", "back", "runs", or "cancel"
        Returns (None, action) if cancelled/back
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui

    tui.clear_screen()
    _show_step_header(6, 8, "Test Data Configuration", f"Choose test data for {num_prompts} prompts")

    # Show current selection if provided
    if current_value:
        source = current_value.get("source", "unknown")
        num_prompts = len(current_value.get("prompts", []))
        num_images = len(current_value.get("images", []))
        console.print(f"[dim]Current: {source} ({num_prompts} prompts, {num_images} images)[/dim]")
        console.print()

    # Build arrow-key options based on model type and available data
    arrow_options = []

    # For VLM with structured test data, show structured option
    if model_type == ModelType.VLM and selected_endpoint:
        from src.benchmarking.datasets.defaults import validate_endpoint_data

        endpoint_status = validate_endpoint_data(selected_endpoint)

        if endpoint_status.get("is_ready"):
            image_count = endpoint_status['image_count']
            arrow_options.append((
                "structured",
                "[green]Structured Test Data[/green]",
                f"Pre-configured {image_count} image-prompt pairs for {selected_endpoint}"
            ))

    # Add default and custom options
    default_available = 20  # We have 20 default prompts per endpoint
    if model_type == ModelType.LLM:
        if num_prompts <= default_available:
            default_desc = f"Use {num_prompts} suite-specific default prompts"
        else:
            default_desc = f"Use {default_available} defaults + manual entry for {num_prompts - default_available} more"

        arrow_options.extend([
            ("default", "[cyan]Default Prompts[/cyan]", default_desc),
            ("custom", "[yellow]Custom Prompts[/yellow]", f"Manually enter all {num_prompts} prompts one by one")
        ])
    else:  # VLM
        if num_prompts <= default_available:
            default_desc = f"Use {num_prompts} default prompts with test images"
        else:
            default_desc = f"Use {default_available} defaults + manual entry for {num_prompts - default_available} more"

        arrow_options.extend([
            ("default", "[cyan]Default Test Data[/cyan]", default_desc),
            ("custom", "[yellow]Custom Prompts[/yellow]", f"Manually enter all {num_prompts} image-prompt pairs")
        ])

    # Add navigation options
    arrow_options.append(("__previous", "[blue]← Previous[/blue]", "Go back to endpoint selection"))
    arrow_options.append(("__cancel", "[red]Cancel[/red]", "Cancel benchmark configuration"))

    # Get user selection
    choice = professional_prompt.get_arrow_selection(
        options=arrow_options,
        title="Select Test Data Source",
        instructions="Use ↑/↓ arrows to navigate, Enter to select"
    )

    # Handle navigation
    if choice is None or choice == "__cancel":
        return (None, "cancel")
    elif choice == "__previous":
        return (None, "back")

    # Process selection
    if choice == "structured":
        from src.benchmarking.datasets.defaults import get_test_data_for_endpoint

        test_data = get_test_data_for_endpoint(selected_endpoint)
        console.print(f"\n[green]✓ Loaded {test_data['num_pairs']} image-prompt pairs[/green]")
        console.print(f"[dim]Endpoint: {selected_endpoint}[/dim]")

        result = {
            "source": "structured",
            "endpoint": selected_endpoint,
            "prompts": test_data["prompts"],
            "images": test_data["images"],
            "pairs": test_data["pairs"],
        }
        return (result, "next")

    elif choice == "default":
        # Use default prompts (up to 20 available)
        from src.benchmarking.datasets.defaults import get_default_prompts_for_suite

        prompts = []
        images = []

        if num_prompts <= 20:
            # Use first N default prompts
            prompts = get_default_prompts_for_suite(suite_name, model_type, count=num_prompts)
            console.print(f"\n[green]✓ Loaded {len(prompts)} suite-specific default prompts[/green]")

            result = {"source": "default", "prompts": prompts, "images": images}
            return (result, "next")
        else:
            # Use all 20 defaults + manual entry for overflow
            prompts = get_default_prompts_for_suite(suite_name, model_type, count=20)
            console.print(f"\n[cyan]✓ Loaded {len(prompts)} default prompts[/cyan]")
            console.print(f"[yellow]→ Need {num_prompts - 20} more prompts via manual entry[/yellow]")
            console.print()
            console.print("[dim]Press Enter to start manual entry wizard...[/dim]")
            input()

            # Launch wizard for remaining prompts
            wizard_result = _manual_prompt_entry_wizard(
                num_prompts=num_prompts - 20,
                model_type=model_type,
                selected_endpoint=selected_endpoint,
                existing_prompts=[],
                existing_images=[]
            )

            # Handle wizard commands
            if wizard_result is None:
                return (None, "back")
            elif wizard_result.get("_command") == "/runs":
                return (None, "runs")
            elif wizard_result.get("_command") == "/default":
                # User wants to skip to defaults - just use the 20 we have
                console.print(f"\n[yellow]Using only {len(prompts)} default prompts[/yellow]")
                result = {"source": "default", "prompts": prompts, "images": images}
                return (result, "next")

            # Combine defaults + manual entries
            prompts.extend(wizard_result["prompts"])
            if model_type == ModelType.VLM and "images" in wizard_result:
                images.extend(wizard_result["images"])

            console.print(f"\n[green]✓ Total: {len(prompts)} prompts ({20} default + {num_prompts - 20} manual)[/green]")

            result = {"source": "mixed", "prompts": prompts, "images": images}
            return (result, "next")

    elif choice == "custom":
        # Full manual entry for all prompts
        console.print(f"\n[yellow]Manual Prompt Entry Mode[/yellow]")
        console.print(f"[dim]You will enter {num_prompts} prompts one by one[/dim]")
        console.print()
        console.print("[dim]Press Enter to start...[/dim]")
        input()

        # Launch wizard
        wizard_result = _manual_prompt_entry_wizard(
            num_prompts=num_prompts,
            model_type=model_type,
            selected_endpoint=selected_endpoint,
            existing_prompts=[],
            existing_images=[]
        )

        # Handle wizard result
        if wizard_result is None:
            return (None, "back")
        elif wizard_result.get("_command") == "/runs":
            return (None, "runs")
        elif wizard_result.get("_command") == "/default":
            # User wants to switch to defaults
            from src.benchmarking.datasets.defaults import get_default_prompts_for_suite
            prompts = get_default_prompts_for_suite(suite_name, model_type, count=min(num_prompts, 20))
            console.print(f"\n[yellow]Switched to {len(prompts)} default prompts[/yellow]")
            result = {"source": "default", "prompts": prompts, "images": []}
            return (result, "next")

        # Successfully collected all prompts
        prompts = wizard_result["prompts"]
        images = wizard_result.get("images", [])

        console.print(f"\n[green]✓ Successfully entered {len(prompts)} custom prompts[/green]")

        result = {"source": "custom", "prompts": prompts, "images": images}
        return (result, "next")


def show_parameters_config(current_value: Optional[Dict[str, Any]] = None) -> Optional[tuple]:
    """
    Display interactive parameters configuration form with arrow-key navigation.

    Args:
        current_value: Previously configured parameters for pre-filling

    Returns:
        Tuple of (params_dict, action) where action is "next", "back", or "cancel"
        Returns (None, action) if cancelled/back
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui

    # Initialize parameters with defaults
    # NOTE: num_runs removed - now configured as num_prompts in Step 5
    params = {
        "num_warmup": current_value.get("num_warmup", 1) if current_value else 1,
        "temperature": current_value.get("temperature", 0.7) if current_value else 0.7,
        "max_tokens": current_value.get("max_tokens", 512) if current_value else 512,
        "export_formats": current_value.get("export_formats", ["json", "csv"]) if current_value else ["json", "csv"],
    }

    while True:
        tui.clear_screen()
        _show_step_header(7, 8, "Benchmark Parameters", "Configure model inference parameters")

        # Show current configuration summary
        console.print("[bold]Current Configuration:[/bold]")
        console.print()
        console.print(f"  [cyan]Warmup runs:[/cyan] {params['num_warmup']}")
        console.print(f"  [cyan]Temperature:[/cyan] {params['temperature']}")
        console.print(f"  [cyan]Max tokens:[/cyan] {params['max_tokens']}")
        console.print(f"  [cyan]Export formats:[/cyan] {', '.join(params['export_formats'])}")
        console.print()
        console.print("[dim]Note: Number of prompts configured in Step 5[/dim]")
        console.print()

        # Build interactive menu options
        menu_options = [
            ("edit_warmup", "[yellow]Warmup runs[/yellow]", f"Currently: {params['num_warmup']} (range: 0-5)"),
            ("edit_temp", "[yellow]Temperature[/yellow]", f"Currently: {params['temperature']} (range: 0.0-1.0)"),
            ("edit_tokens", "[yellow]Max tokens[/yellow]", f"Currently: {params['max_tokens']} (range: 1-4096)"),
            ("edit_formats", "[yellow]Export formats[/yellow]", f"Currently: {', '.join(params['export_formats'])}"),
            ("__separator", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ""),
            ("continue", "[green]Continue to Step 8 →[/green]", "Proceed to execution mode selection"),
            ("__previous", "[blue]← Previous[/blue]", "Go back to test data selection"),
            ("__cancel", "[red]Cancel[/red]", "Cancel benchmark configuration"),
        ]

        # Get user selection
        choice = professional_prompt.get_arrow_selection(
            options=menu_options,
            title="Select parameter to edit or continue",
            instructions="Use ↑/↓ arrows to navigate, Enter to select"
        )

        # Handle navigation
        if choice is None or choice == "__cancel":
            return (None, "cancel")
        elif choice == "__previous":
            return (None, "back")
        elif choice == "__separator":
            # Separator selected, ignore
            continue
        elif choice == "continue":
            # Validate and return
            return (params, "next")

        # Handle parameter editing
        elif choice == "edit_warmup":
            tui.clear_screen()
            console.print()
            console.print("[bold cyan]Edit Warmup Runs[/bold cyan]")
            console.print("[dim]Initial runs to warm up the model (not counted in results)[/dim]")
            console.print()

            new_value = professional_prompt.get_numeric(
                prompt_msg="Warmup runs",
                min_val=0,
                max_val=5,
                default=params["num_warmup"],
                style="cyan"
            )
            params["num_warmup"] = new_value

        elif choice == "edit_temp":
            tui.clear_screen()
            console.print()
            console.print("[bold cyan]Edit Temperature[/bold cyan]")
            console.print("[dim]Controls randomness (0.0=deterministic, 1.0=creative)[/dim]")
            console.print(f"[dim]Current: {params['temperature']}, Range: 0.0-1.0[/dim]")
            console.print()

            temp_str = professional_prompt.get_input(
                f"Temperature (default: {params['temperature']})",
                style="cyan",
                allow_multiline=False,
                show_instructions=False
            )

            if temp_str:
                try:
                    params["temperature"] = max(0.0, min(1.0, float(temp_str)))
                    console.print(f"[green]✓ Temperature set to {params['temperature']}[/green]")
                except ValueError:
                    console.print(f"[red]Invalid value. Keeping {params['temperature']}[/red]")

            console.print()
            tui.prompt("Press Enter to continue...", style="dim")

        elif choice == "edit_tokens":
            tui.clear_screen()
            console.print()
            console.print("[bold cyan]Edit Max Tokens[/bold cyan]")
            console.print("[dim]Maximum number of tokens to generate[/dim]")
            console.print()

            new_value = professional_prompt.get_numeric(
                prompt_msg="Max tokens",
                min_val=1,
                max_val=4096,
                default=params["max_tokens"],
                style="cyan"
            )
            params["max_tokens"] = new_value

        elif choice == "edit_formats":
            tui.clear_screen()
            console.print()
            console.print("[bold cyan]Edit Export Formats[/bold cyan]")
            console.print("[dim]Comma-separated list (e.g., json,csv,html)[/dim]")
            console.print("[dim]Available: json, csv, html[/dim]")
            current_formats = ",".join(params["export_formats"])
            console.print(f"[dim]Current: {current_formats}[/dim]")
            console.print()

            formats_str = professional_prompt.get_input(
                f"Export formats (default: {current_formats})",
                style="cyan",
                allow_multiline=False,
                show_instructions=False
            )

            if formats_str:
                params["export_formats"] = [f.strip() for f in formats_str.split(",") if f.strip()]
                console.print(f"[green]✓ Formats set to: {', '.join(params['export_formats'])}[/green]")

            console.print()
            tui.prompt("Press Enter to continue...", style="dim")


def show_execution_mode(current_value: Optional[ExecutionMode] = None) -> Optional[tuple]:
    """
    Display execution mode selection screen with arrow-key navigation.

    Args:
        current_value: Previously selected ExecutionMode for pre-filling

    Returns:
        Tuple of (ExecutionMode, action) where action is "next", "back", or "cancel"
        Returns (None, action) if cancelled/back
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui

    tui.clear_screen()
    _show_step_header(8, 8, "Execution Mode", "How would you like to run the benchmark?")

    # Show current selection if provided
    if current_value:
        mode_str = "Foreground" if current_value == ExecutionMode.FOREGROUND else "Background"
        console.print(f"[dim]Current selection: {mode_str}[/dim]")
        console.print()

    # Build arrow-key options
    options = [
        (
            "foreground",
            "[green]Foreground Execution[/green]",
            "Watch real-time progress with live updates (blocks other operations)"
        ),
        (
            "background",
            "[yellow]Background Execution[/yellow]",
            "Run benchmark in background, continue using app (non-blocking)"
        )
    ]

    # Add detailed info
    console.print("[bold]Execution Modes:[/bold]")
    console.print("[green]Foreground:[/green] Real-time progress, detailed output, blocking")
    console.print("[yellow]Background:[/yellow] Silent execution, notification on completion, non-blocking")
    console.print()

    # Add navigation options
    options.append(("__previous", "[blue]← Previous[/blue]", "Go back to parameters configuration"))
    options.append(("__cancel", "[red]Cancel[/red]", "Cancel benchmark configuration"))

    # Get user selection
    choice = professional_prompt.get_arrow_selection(
        options=options,
        title="Select Execution Mode",
        instructions="Use ↑/↓ arrows to navigate, Enter to select"
    )

    # Handle navigation
    if choice is None or choice == "__cancel":
        return (None, "cancel")
    elif choice == "__previous":
        return (None, "back")

    # Convert choice to ExecutionMode
    execution_mode = ExecutionMode.FOREGROUND if choice == "foreground" else ExecutionMode.BACKGROUND

    console.print(f"\n[green]✓ {execution_mode.value} mode selected[/green]")

    return (execution_mode, "next")


def show_complete_suite_selection(current_value: Optional[List[str]] = None) -> Optional[tuple]:
    """
    Display Complete suite sub-suite selection screen.

    Args:
        current_value: Previously selected sub-suites

    Returns:
        Tuple of (selected_suites_list, action) where action is "next", "back", or "cancel"
        Returns (None, action) if cancelled/back
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui

    # Default: all suites selected
    selected_suites = set(current_value) if current_value else {"speed", "resources", "quality", "stress"}

    while True:
        tui.clear_screen()
        console.print("[bold magenta]Complete Suite Configuration[/bold magenta]")
        console.print("[dim]Select which sub-suites to run[/dim]")
        console.print()

        # Show current selection
        console.print("[bold]Selected Suites:[/bold]")
        if selected_suites:
            for suite in ["speed", "resources", "quality", "stress"]:
                if suite in selected_suites:
                    console.print(f"  [green]✓[/green] {suite.title()}")
                else:
                    console.print(f"  [dim]○ {suite.title()}[/dim]")
        else:
            console.print("  [yellow]No suites selected[/yellow]")
        console.print()

        # Build menu options
        menu_options = []
        for suite in ["speed", "resources", "quality", "stress"]:
            if suite in selected_suites:
                menu_options.append((f"toggle_{suite}", f"[green]✓ {suite.title()}[/green]", f"Deselect {suite} suite"))
            else:
                menu_options.append((f"toggle_{suite}", f"[dim]○ {suite.title()}[/dim]", f"Select {suite} suite"))

        menu_options.extend([
            ("__separator", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ""),
            ("select_all", "[cyan]Select All[/cyan]", "Enable all sub-suites"),
            ("deselect_all", "[yellow]Deselect All[/yellow]", "Disable all sub-suites"),
            ("__separator2", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ""),
            ("continue", "[green]Continue →[/green]", "Proceed with selected suites"),
            ("__previous", "[blue]← Previous[/blue]", "Go back to suite selection"),
            ("__cancel", "[red]Cancel[/red]", "Cancel benchmark configuration"),
        ])

        # Get user selection
        choice = professional_prompt.get_arrow_selection(
            options=menu_options,
            title="Configure Complete Suite",
            instructions="Use ↑/↓ arrows to navigate, Enter to toggle"
        )

        # Handle navigation
        if choice is None or choice == "__cancel":
            return (None, "cancel")
        elif choice == "__previous":
            return (None, "back")
        elif choice in ["__separator", "__separator2"]:
            continue
        elif choice == "select_all":
            selected_suites = {"speed", "resources", "quality", "stress"}
        elif choice == "deselect_all":
            selected_suites = set()
        elif choice == "continue":
            if not selected_suites:
                console.print()
                console.print("[red]✗ Please select at least one sub-suite[/red]")
                console.print("[dim]Press Enter to continue...[/dim]")
                input()
                continue
            return (list(selected_suites), "next")
        elif choice.startswith("toggle_"):
            suite_name = choice.replace("toggle_", "")
            if suite_name in selected_suites:
                selected_suites.remove(suite_name)
            else:
                selected_suites.add(suite_name)


def show_config_review(config_summary: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> str:
    """
    Display configuration review screen with arrow-key navigation.

    Args:
        config_summary: Dictionary with configuration details for display
        state: Optional state dictionary for saving configuration

    Returns:
        "confirm" to start benchmark
        "save" to save configuration
        "edit" to go back and edit
        "cancel" to cancel
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui

    while True:  # Loop to handle save and return to review
        tui.clear_screen()

        # Show step heading with underline (Ekam-CLI header already shown by clear_screen)
        tui.show_step_heading("Final Step: Configuration Review")
        console.print("[dim]Please review your benchmark configuration[/dim]")
        console.print()

        # Display configuration summary
        table = Table(show_header=True, header_style="bold magenta", border_style="cyan", box=None)
        table.add_column("Setting", style="cyan", width=25)
        table.add_column("Value", style="green")

        table.add_row("Model Type", config_summary.get("model_type", "Unknown"))
        table.add_row("Suite", config_summary.get("suite", "Unknown"))
        table.add_row("Models", str(len(config_summary.get("models", []))))
        table.add_row("Runs per model", str(config_summary.get("num_runs", 0)))
        table.add_row("Warmup runs", str(config_summary.get("num_warmup", 0)))
        table.add_row("Execution mode", config_summary.get("execution_mode", "Unknown"))
        table.add_row("Export formats", ", ".join(config_summary.get("export_formats", [])))

        if "estimated_duration" in config_summary:
            table.add_row(
                "Estimated duration",
                f"{config_summary['estimated_duration']} minutes",
                style="yellow bold"
            )

        console.print(table)

        console.print("\n[bold]Selected models:[/bold]")
        for model_id in config_summary.get("models", []):
            display_name = _extract_model_display_name(model_id)
            console.print(f"  • {display_name}")

        console.print()

        # Build arrow-key navigation options
        review_options = [
            ("confirm", "[green]✓ Start Benchmark[/green]", "Begin benchmark execution with this configuration"),
            ("save", "[cyan]💾 Save Configuration[/cyan]", "Save this configuration to file for later use"),
            ("edit", "[yellow]← Edit Configuration[/yellow]", "Go back to execution mode to make changes"),
            ("cancel", "[red]Cancel[/red]", "Cancel benchmark and return to main menu"),
        ]

        # Get user selection
        choice = professional_prompt.get_arrow_selection(
            options=review_options,
            title="Review Complete - Ready to Start?",
            instructions="Use ↑/↓ arrows to navigate, Enter to select"
        )

        # Handle selection
        if choice is None:
            return "cancel"
        elif choice == "save":
            if state is not None:
                from src.benchmarking.cli.benchmark_menu import save_benchmark_config
                save_benchmark_config(state)
                console.print("[dim]Press Enter to continue...[/dim]")
                input()
                # Return to review screen
                continue
            else:
                console.print("[red]✗ Cannot save: state not available[/red]")
                console.print("[dim]Press Enter to continue...[/dim]")
                input()
                continue
        else:
            return choice


def show_benchmark_error(error_message: str):
    """
    Display benchmark error screen.

    Args:
        error_message: Error message to display
    """
    console.print()
    console.print(Panel.fit(
        f"[bold red]Benchmark Error[/bold red]\n\n"
        f"{error_message}",
        border_style="red"
    ))
    console.print()
