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


def _show_step_header(step_number: int, total_steps: int, title: str, subtitle: str = ""):
    """
    Show step information header.

    Note: Ekam-CLI header is now shown automatically by tui.clear_screen()

    Args:
        step_number: Current step number
        total_steps: Total number of steps
        title: Step title
        subtitle: Optional subtitle text
    """
    # Show step information (header is shown by clear_screen automatically)
    console.print(f"[bold]Step {step_number}/{total_steps}:[/bold] [cyan]{title}[/cyan]")
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
                    available_models.append({
                        "model_id": model.model_id,
                        "provider": model.provider.value if hasattr(model.provider, 'value') else str(model.provider)
                    })

        if not available_models:
            console.print(f"[red]No {model_type.value.upper()} models available![/red]")
            console.print("\nPlease ensure models are loaded before benchmarking.")
            return None

        # Display available models
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Model ID", style="green", width=50)
        table.add_column("Provider", style="yellow", width=15)

        for idx, model in enumerate(available_models, 1):
            model_id = model.get("model_id", "unknown")
            provider = model.get("provider", "unknown")
            table.add_row(str(idx), model_id, provider)

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
            console.print(f"  • {model_id}")

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


def show_test_data_selection(
    model_type: ModelType,
    selected_endpoint: Optional[str] = None,
    current_value: Optional[Dict[str, Any]] = None
) -> Optional[tuple]:
    """
    Display test data configuration screen with arrow-key navigation.

    Args:
        model_type: Type of models being tested
        selected_endpoint: Optional VLM endpoint for structured test data
        current_value: Previously selected test data config for pre-filling

    Returns:
        Tuple of (test_data_dict, action) where action is "next", "back", or "cancel"
        Returns (None, action) if cancelled/back
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui

    tui.clear_screen()
    _show_step_header(5, 7, "Test Data Configuration", "Choose test data for benchmarking")

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
    if model_type == ModelType.LLM:
        arrow_options.extend([
            ("default", "[cyan]Default Prompts[/cyan]", "Use 10 recommended test prompts (text-only)"),
            ("custom", "[yellow]Custom Data[/yellow]", "Provide custom prompts file")
        ])
    else:  # VLM
        arrow_options.extend([
            ("default", "[cyan]Default Prompts[/cyan]", "Use 10 default VLM prompts without images"),
            ("custom", "[yellow]Custom Data[/yellow]", "Provide custom images directory and prompts")
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
        from src.benchmarking.datasets.defaults import get_default_prompts

        model_type_str = "llm" if model_type == ModelType.LLM else "vlm"
        prompts = get_default_prompts(model_type_str, count=10)

        console.print(f"\n[green]✓ Loaded {len(prompts)} default test prompts[/green]")

        result = {"source": "default", "prompts": prompts, "images": []}
        return (result, "next")

    elif choice == "custom":
        # Custom data configuration with text inputs
        console.print("\n[yellow]Custom test data configuration:[/yellow]")
        console.print()

        if model_type == ModelType.LLM:
            console.print("[bold]Provide path to text file with prompts (one per line):[/bold]")
            console.print("[dim]Leave empty to use defaults[/dim]")
            console.print()

            prompts_path = professional_prompt.get_file_path(
                prompt_msg="Prompts file path",
                style="cyan"
            )

            if prompts_path and Path(prompts_path).exists():
                with open(prompts_path, 'r') as f:
                    prompts = [line.strip() for line in f if line.strip()]
                console.print(f"\n[green]✓ Loaded {len(prompts)} prompts[/green]")

                result = {"source": "custom", "prompts": prompts, "images": []}
                return (result, "next")
            else:
                # Fallback to defaults
                from src.benchmarking.datasets.defaults import get_default_prompts

                prompts = get_default_prompts("llm", count=10)
                console.print(f"\n[yellow]File not found. Using {len(prompts)} default prompts[/yellow]")

                result = {"source": "default", "prompts": prompts, "images": []}
                return (result, "next")

        else:  # VLM
            console.print("[bold]Provide path to directory with images:[/bold]")
            console.print("[dim]Supported formats: PNG, JPG, JPEG[/dim]")
            console.print()

            images_dir = professional_prompt.get_file_path(
                prompt_msg="Images directory path",
                style="cyan"
            )

            if images_dir and Path(images_dir).is_dir():
                images = list(Path(images_dir).glob("*.png")) + \
                        list(Path(images_dir).glob("*.jpg")) + \
                        list(Path(images_dir).glob("*.jpeg"))
                console.print(f"\n[green]✓ Found {len(images)} images[/green]")

                console.print("\n[bold]Provide path to prompts file (optional):[/bold]")
                console.print("[dim]Leave empty to skip[/dim]")
                console.print()

                prompts_path = professional_prompt.get_file_path(
                    prompt_msg="Prompts file path (optional)",
                    style="cyan"
                )

                prompts = []
                if prompts_path and Path(prompts_path).exists():
                    with open(prompts_path, 'r') as f:
                        prompts = [line.strip() for line in f if line.strip()]
                    console.print(f"\n[green]✓ Loaded {len(prompts)} prompts[/green]")

                result = {
                    "source": "custom",
                    "prompts": prompts,
                    "images": [str(img) for img in images]
                }
                return (result, "next")
            else:
                # Fallback to defaults
                from src.benchmarking.datasets.defaults import get_default_prompts

                prompts = get_default_prompts("vlm", count=10)
                console.print(f"\n[yellow]Directory not found. Using {len(prompts)} default VLM prompts[/yellow]")
                console.print("[yellow]Note: You'll need to provide images when running VLM benchmarks[/yellow]")

                result = {"source": "default", "prompts": prompts, "images": []}
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
    params = {
        "num_runs": current_value.get("num_runs", 5) if current_value else 5,
        "num_warmup": current_value.get("num_warmup", 1) if current_value else 1,
        "temperature": current_value.get("temperature", 0.7) if current_value else 0.7,
        "max_tokens": current_value.get("max_tokens", 512) if current_value else 512,
        "export_formats": current_value.get("export_formats", ["json", "csv"]) if current_value else ["json", "csv"],
    }

    while True:
        tui.clear_screen()
        _show_step_header(6, 7, "Benchmark Parameters", "Configure benchmark execution parameters")

        # Show current configuration summary
        console.print("[bold]Current Configuration:[/bold]")
        console.print()
        console.print(f"  [cyan]Number of runs:[/cyan] {params['num_runs']}")
        console.print(f"  [cyan]Warmup runs:[/cyan] {params['num_warmup']}")
        console.print(f"  [cyan]Temperature:[/cyan] {params['temperature']}")
        console.print(f"  [cyan]Max tokens:[/cyan] {params['max_tokens']}")
        console.print(f"  [cyan]Export formats:[/cyan] {', '.join(params['export_formats'])}")
        console.print()

        # Build interactive menu options
        menu_options = [
            ("edit_runs", "[yellow]Number of runs[/yellow]", f"Currently: {params['num_runs']} (range: 1-100)"),
            ("edit_warmup", "[yellow]Warmup runs[/yellow]", f"Currently: {params['num_warmup']} (range: 0-5)"),
            ("edit_temp", "[yellow]Temperature[/yellow]", f"Currently: {params['temperature']} (range: 0.0-1.0)"),
            ("edit_tokens", "[yellow]Max tokens[/yellow]", f"Currently: {params['max_tokens']} (range: 1-4096)"),
            ("edit_formats", "[yellow]Export formats[/yellow]", f"Currently: {', '.join(params['export_formats'])}"),
            ("__separator", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ""),
            ("continue", "[green]Continue to Step 7 →[/green]", "Proceed to execution mode selection"),
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
        elif choice == "edit_runs":
            tui.clear_screen()
            console.print()
            console.print("[bold cyan]Edit Number of Runs[/bold cyan]")
            console.print("[dim]How many times to run each benchmark test[/dim]")
            console.print()

            new_value = professional_prompt.get_numeric(
                prompt_msg="Number of runs",
                min_val=1,
                max_val=100,
                default=params["num_runs"],
                style="cyan"
            )
            params["num_runs"] = new_value

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
    _show_step_header(7, 7, "Execution Mode", "How would you like to run the benchmark?")

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


def show_config_review(config_summary: Dict[str, Any]) -> str:
    """
    Display configuration review screen with arrow-key navigation.

    Args:
        config_summary: Dictionary with configuration details

    Returns:
        "confirm" to start benchmark
        "edit" to go back and edit
        "cancel" to cancel
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui

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
        console.print(f"  • {model_id}")

    console.print()

    # Build arrow-key navigation options
    review_options = [
        ("confirm", "[green]✓ Start Benchmark[/green]", "Begin benchmark execution with this configuration"),
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
