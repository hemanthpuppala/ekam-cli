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


def show_model_selection(model_type: ModelType, session_manager) -> Optional[tuple[List[str], str]]:
    """
    Display model selection screen (multi-select).

    Args:
        model_type: Type of models to show
        session_manager: SessionManager to query available models

    Returns:
        Tuple of (selected_model_ids, action) where action is "next", "back", or "cancel";
        returns None on hard cancel
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

        # Multi-select prompt with navigation hints
        console.print("[yellow]Enter model numbers separated by commas (e.g., 1,3,4)[/yellow]")
        console.print("[yellow]Or 'all' to select all models[/yellow]")
        console.print("[dim]Hotkeys: b = Back, h = Home[/dim]")

        selection = get_text_input("Select models", allow_empty=False)

        if selection is None:
            return None

        # Parse selection
        sel = selection.strip().lower()
        if sel == "b":
            return ([], "back")
        if sel == "h":
            return ([], "cancel")

        if sel == "all":
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
                        return ([], "back")
            except ValueError:
                console.print("[red]Invalid input format![/red]")
                return ([], "back")

        if not selected_models:
            console.print("[red]No models selected![/red]")
            return ([], "back")

        console.print(f"\n[green]Selected {len(selected_models)} model(s):[/green]")
        for model_id in selected_models:
            display_name = _extract_model_display_name(model_id)
            console.print(f"  • {display_name}")

        # Step 3.1: For quantized GGUF VLMs, pre-select mmproj if multiple candidates exist
        try:
            from src.models.endpoints import ProviderType
            from src.cli.text_input import professional_prompt
            from src.cli.tui_manager import tui as _tui
            # Map model_id to ModelInfo to check provider types
            discovered = session_manager.discover_models()
            id_to_model = {m.model_id: m for m in discovered}
            quant_provider = session_manager.model_discovery.get_provider(ProviderType.QUANTIZED)

            if quant_provider is not None and hasattr(quant_provider, "_find_mmproj_candidates"):
                # Filter only quantized gguf models from selection
                pending = []
                from pathlib import Path
                for mid in selected_models:
                    model = id_to_model.get(mid)
                    if not model:
                        continue
                    # Determine if this model is provided by the QUANTIZED provider
                    provider_obj = getattr(model, "provider", None)
                    provider_name = str(provider_obj.value if hasattr(provider_obj, "value") else provider_obj).lower()
                    if not (provider_obj == ProviderType.QUANTIZED or provider_name == "quantized"):
                        continue
                    parts = str(mid).split(":", 2)
                    if len(parts) == 3 and parts[0] == "quantized" and parts[1] == "gguf":
                        pending.append(Path(parts[2]))

                if pending:
                    # Clear previous table and show a dedicated Step 3.1 screen
                    from src.cli.tui_manager import tui as __tui
                    __tui.clear_screen()
                    _show_step_header(3, 7, "Vision Encoder Selection", "Select mmproj for quantized VLMs (if multiple are found)")

                    # Show a compact summary of selected models at the top
                    console.print(f"[green]Selected {len(selected_models)} model(s):[/green]")
                    for mid in selected_models:
                        display_name = _extract_model_display_name(mid)
                        console.print(f"  • {display_name}")
                    console.print()

                i = 0
                while i < len(pending):
                    lang_path = pending[i]
                    # Get candidates
                    try:
                        candidates = quant_provider._find_mmproj_candidates(lang_path)
                    except Exception as e:
                        logger.debug(f"mmproj scan failed for {lang_path}: {e}")
                        continue

                    if not candidates:
                        i += 1
                        continue
                    # If only one candidate found, try a broader family match to detect additional mmproj variants
                    if len(candidates) == 1:
                        try:
                            import re
                            base = lang_path.stem.lower()
                            # Strip language/format tokens anywhere in name, not just suffix
                            base_general = re.sub(r"(_language|_text|_model)", "", base)
                            # Remove any quant/precision tokens anywhere
                            base_general = re.sub(r"_q\d[^_]*", "", base_general)
                            base_general = re.sub(r"_(f16|f32|bf16|fp16)", "", base_general)
                            family = base_general.strip('_')

                            broad = []
                            for mmproj in quant_provider.quantized_dir.rglob("mmproj-*.gguf"):
                                name = mmproj.stem[len("mmproj-"):].lower()
                                name = re.sub(r"(_language|_text|_model)", "", name)
                                name = re.sub(r"_q\d[^_]*", "", name)
                                name = re.sub(r"_(f16|f32|bf16|fp16)", "", name)
                                name = name.strip('_')
                                if name == family or name in family or family in name:
                                    if mmproj not in candidates:
                                        broad.append(mmproj)
                            if broad:
                                candidates.extend(broad)
                        except Exception:
                            pass

                    if len(candidates) == 1:
                        quant_provider._session_mmproj_choice[str(lang_path)] = candidates[0]
                        logger.info(f"Auto-selected mmproj for {lang_path.name}: {candidates[0].name}")
                        i += 1
                        continue

                    # Multiple candidates, prompt user once per model
                    options = [("__back", "[blue]← Back[/blue]", "Go back to previous selection or model list")]
                    for p in candidates:
                        try:
                            meta = quant_provider.metadata_cache.get_metadata(str(p), provider="gguf")
                        except Exception:
                            meta = None
                        try:
                            size_gb = p.stat().st_size / (1024 ** 3)
                        except Exception:
                            size_gb = 0.0
                        quant = (meta.quantization.upper() if meta and getattr(meta, "quantization", None) else "UNKNOWN")
                        label = f"[green]{p.name}[/green]"
                        desc = f"{size_gb:.2f} GB • {quant}"
                        options.append((str(p), label, desc))

                    _tui.console.print(f"\n[cyan]Select vision encoder (mmproj) for[/cyan] [yellow]{lang_path.name}[/yellow]")
                    _tui.console.print(
                        f"[dim]Found {len(candidates)} candidate mmproj files in results/quantizations[/dim]"
                    )
                    _tui.console.print(
                        "[dim]Choose the one to pair with this VLM. Different quantizations can impact speed/quality.[/dim]"
                    )
                    _tui.console.print(
                        "[dim]Your choice is remembered for this session and used during loading.[/dim]"
                    )
                    choice = professional_prompt.get_arrow_selection(
                        options=options,
                        title="Select Vision Encoder (mmproj)",
                        instructions="Use ↑/↓ arrows, Enter to select, q to skip",
                    )
                    if not choice:
                        _tui.console.print("[yellow]Skipped mmproj selection for this model (auto-detect later)[/yellow]")
                        i += 1
                        continue
                    if choice == "__back":
                        # If at first model, go back to Step 3 (model selection)
                        if i == 0:
                            return None
                        # Otherwise go back to previous model's mmproj selection
                        i -= 1
                        # Clear any previous remembered choice to force re-selection
                        try:
                            key = str(pending[i])
                            if key in quant_provider._session_mmproj_choice:
                                del quant_provider._session_mmproj_choice[key]
                        except Exception:
                            pass
                        continue
                    sel_path = Path(choice)
                    quant_provider._session_mmproj_choice[str(lang_path)] = sel_path
                    logger.info(f"User selected mmproj for {lang_path.name}: {sel_path.name}")
                    i += 1
        except Exception as mmerr:
            logger.debug(f"mmproj pre-selection step skipped due to: {mmerr}")

        return (selected_models, "next")

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


def _json_pairs_upload_wizard(
    model_type: ModelType,
    selected_endpoint: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Interactive wizard for uploading image-prompt pairs from JSON file.

    Supports two JSON formats:
    1. Dict format: {"image1.jpg": "prompt1", "image2.jpg": "prompt2"}
    2. List format: [{"image": "img.jpg", "prompt": "p1"}, {"image": "img.jpg", "prompt": "p2"}]

    Args:
        model_type: LLM or VLM (VLM requires images)
        selected_endpoint: VLM endpoint (if applicable)

    Returns:
        Dict with 'prompts', 'images', and 'pairs', or None if cancelled
    """
    from src.cli.text_input import professional_prompt
    from src.cli.tui_manager import tui
    from rich.panel import Panel
    import json
    from pathlib import Path

    # Helper: Validate image path (reuse existing logic)
    def validate_image_path(path_str: str) -> Optional[Path]:
        """Validate and resolve image path using CWD -> project root -> absolute."""
        if not path_str:
            return None

        # Strip quotes (single or double) from path
        path_str = path_str.strip().strip('"').strip("'")

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

    while True:
        tui.clear_screen()
        console.print(f"[bold magenta]JSON Image-Prompt Pairs Upload[/bold magenta]")
        console.print()

        # Show example formats in a panel
        example_dict = """{
  "path/to/image1.jpg": "Describe this image",
  "path/to/image2.jpg": "What objects are visible?"
}"""

        example_list = """[
  {"image": "path/to/image1.jpg", "prompt": "Describe this image"},
  {"image": "path/to/image1.jpg", "prompt": "What colors are present?"},
  {"image": "path/to/image2.jpg", "prompt": "Count the objects"}
]"""

        console.print(Panel(
            f"[bold cyan]Supported JSON Formats:[/bold cyan]\n\n"
            f"[yellow]1. Dict Format (1:1 mapping):[/yellow]\n"
            f"[dim]{example_dict}[/dim]\n\n"
            f"[yellow]2. List Format (supports same image with different prompts):[/yellow]\n"
            f"[dim]{example_list}[/dim]\n\n"
            f"[bold]Notes:[/bold]\n"
            f"• Image paths can be relative to project root or absolute\n"
            f"• Same image can appear multiple times with different prompts (list format)\n"
            f"• All image paths will be validated before proceeding",
            title="[cyan]JSON Format Examples[/cyan]",
            border_style="cyan"
        ))
        console.print()

        # Get JSON file path
        console.print("[bold cyan]Enter path to JSON file:[/bold cyan]")
        console.print("[dim]Supports quotes, relative paths, and absolute paths[/dim]")
        console.print("[dim]Commands: /back (return to previous step)[/dim]")
        console.print()

        json_path_input = professional_prompt.get_text_input(
            prompt="JSON file path",
            default_value="",
            allow_empty=False,
            multiline=False
        )

        if json_path_input is None or json_path_input.lower() == "/back":
            return None

        # Strip quotes and resolve path
        json_path_str = json_path_input.strip().strip('"').strip("'")
        json_path = Path(json_path_str)

        # Try resolving relative to CWD or project root
        if not json_path.exists():
            project_root = Path(__file__).parent.parent.parent.parent
            json_path_alt = project_root / json_path_str
            if json_path_alt.exists():
                json_path = json_path_alt

        # Validate JSON file exists
        if not json_path.exists():
            console.print(f"[red]✗ JSON file not found: {json_path}[/red]")
            console.print("[dim]Press Enter to retry...[/dim]")
            input()
            continue

        if not json_path.is_file():
            console.print(f"[red]✗ Not a file: {json_path}[/red]")
            console.print("[dim]Press Enter to retry...[/dim]")
            input()
            continue

        # Load and parse JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            console.print(f"[red]✗ Invalid JSON format: {e}[/red]")
            console.print("[dim]Press Enter to retry...[/dim]")
            input()
            continue
        except Exception as e:
            console.print(f"[red]✗ Failed to read JSON file: {e}[/red]")
            console.print("[dim]Press Enter to retry...[/dim]")
            input()
            continue

        # Parse JSON data into pairs
        pairs = []
        errors = []

        if isinstance(json_data, dict):
            # Dict format: {"image.jpg": "prompt"}
            for img_path, prompt in json_data.items():
                if not isinstance(prompt, str):
                    errors.append(f"Prompt for '{img_path}' is not a string")
                    continue
                pairs.append((img_path, prompt))

        elif isinstance(json_data, list):
            # List format: [{"image": "...", "prompt": "..."}]
            for idx, item in enumerate(json_data):
                if not isinstance(item, dict):
                    errors.append(f"Item {idx} is not a dictionary")
                    continue

                img_path = item.get("image") or item.get("image_path") or item.get("img")
                prompt = item.get("prompt")

                if not img_path:
                    errors.append(f"Item {idx} missing 'image' field")
                    continue
                if not prompt:
                    errors.append(f"Item {idx} missing 'prompt' field")
                    continue
                if not isinstance(prompt, str):
                    errors.append(f"Item {idx} prompt is not a string")
                    continue

                pairs.append((img_path, prompt))
        else:
            console.print(f"[red]✗ JSON must be a dict or list, got {type(json_data).__name__}[/red]")
            console.print("[dim]Press Enter to retry...[/dim]")
            input()
            continue

        if errors:
            console.print(f"[red]✗ Found {len(errors)} error(s) in JSON:[/red]")
            for error in errors[:5]:  # Show first 5 errors
                console.print(f"  • {error}")
            if len(errors) > 5:
                console.print(f"  [dim]... and {len(errors) - 5} more[/dim]")
            console.print("[dim]Press Enter to retry...[/dim]")
            input()
            continue

        if not pairs:
            console.print(f"[red]✗ No valid image-prompt pairs found in JSON[/red]")
            console.print("[dim]Press Enter to retry...[/dim]")
            input()
            continue

        # Validate all image paths
        console.print()
        console.print(f"[cyan]Validating {len(pairs)} image path(s)...[/cyan]")
        console.print()

        validated_pairs = []
        validation_errors = []

        for img_path_str, prompt in pairs:
            validated_path = validate_image_path(img_path_str)
            if not validated_path:
                validation_errors.append(f"Image not found: {img_path_str}")
                continue

            # Check image format
            valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
            if validated_path.suffix.lower() not in valid_extensions:
                validation_errors.append(f"Unsupported format ({validated_path.suffix}): {img_path_str}")
                continue

            validated_pairs.append((str(validated_path), prompt))

        if validation_errors:
            console.print(f"[red]✗ Found {len(validation_errors)} validation error(s):[/red]")
            for error in validation_errors[:10]:  # Show first 10 errors
                console.print(f"  • {error}")
            if len(validation_errors) > 10:
                console.print(f"  [dim]... and {len(validation_errors) - 10} more[/dim]")
            console.print()
            console.print("[yellow]Options:[/yellow]")
            console.print("  [1] Fix JSON file and retry")
            console.print("  [2] Continue with valid pairs only")
            console.print("  [3] Cancel")
            console.print()

            choice = professional_prompt.get_text_input(
                prompt="Choose option (1/2/3)",
                default_value="1",
                allow_empty=False
            )

            if choice == "1" or choice is None:
                continue  # Retry
            elif choice == "3":
                return None  # Cancel
            # else: continue with validated_pairs (choice == "2")

        if not validated_pairs:
            console.print(f"[red]✗ No valid pairs after validation[/red]")
            console.print("[dim]Press Enter to retry...[/dim]")
            input()
            continue

        # Show summary
        console.print()
        console.print(f"[green]✓ Successfully loaded {len(validated_pairs)} image-prompt pair(s)[/green]")
        console.print()
        console.print("[bold]Sample pairs:[/bold]")
        for idx, (img_path, prompt) in enumerate(validated_pairs[:5], 1):
            img_name = Path(img_path).name
            prompt_preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
            console.print(f"  [cyan]{idx}.[/cyan] [dim]{img_name}:[/dim] {prompt_preview}")

        if len(validated_pairs) > 5:
            console.print(f"  [dim]... and {len(validated_pairs) - 5} more pair(s)[/dim]")

        console.print()

        # Confirm
        menu_options = [
            ("confirm", "[green]Yes - Use these pairs[/green]", "Proceed with loaded pairs"),
            ("retry", "[yellow]No - Try another file[/yellow]", "Load a different JSON file"),
            ("cancel", "[red]Cancel[/red]", "Return to previous step"),
        ]

        choice = professional_prompt.get_arrow_selection(
            options=menu_options,
            title="Confirm JSON pairs",
            instructions="Y = Yes, N = No, C = Cancel"
        )

        if choice == "confirm" or choice == "y":
            # Extract images and prompts from validated pairs
            images = [img for img, _ in validated_pairs]
            prompts = [prompt for _, prompt in validated_pairs]

            result = {
                "prompts": prompts,
                "images": images,
                "pairs": validated_pairs,
                "source": "json_upload"
            }
            return result

        elif choice == "retry" or choice == "n":
            continue  # Loop back to file selection

        elif choice == "cancel" or choice == "c" or choice is None:
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
        "speed": 10,
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

        # Guidance
        console.print("Each prompt will be run exactly once (1 prompt = 1 run)")
        if suite_name.lower() == "quality":
            console.print("[dim](Quality suite will generate multiple outputs per prompt)[/dim]")
        elif suite_name.lower() == "stress":
            console.print("[dim](Stress suite will cycle through prompts for the duration)[/dim]")
        console.print()

        # Tabular display of current value
        from rich.table import Table as _Table
        tbl = _Table.grid(padding=(0, 2))
        tbl.add_column("Parameter", style="cyan")
        tbl.add_column("Current Value", style="white")
        tbl.add_row("Prompts", f"{num_prompts}")
        console.print(tbl)
        console.print(f"[dim]Range: 1-50 | Default: {default_value}[/dim]")
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
            user_input = professional_prompt.get_text_input(
                prompt="Number of prompts [1-50]",
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
        # Default now loads image-prompt pairs from assets/test_data for the selected endpoint
        default_desc = (
            f"Load image-prompt pairs from test_data for {selected_endpoint}"
            if selected_endpoint else
            "Load image-prompt pairs from VLM test_data"
        )

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
    if choice == "default":
        # Use default prompts (up to 20 available)
        if model_type == ModelType.VLM and selected_endpoint:
            # For VLM, Default pulls paired images+prompts from assets/test_data for the endpoint
            from src.benchmarking.datasets.defaults import get_test_data_for_endpoint

            test_data = get_test_data_for_endpoint(selected_endpoint, count=num_prompts)
            console.print(f"\n[green]✓ Loaded {test_data['num_pairs']} image-prompt pairs[/green]")
            console.print(f"[dim]Endpoint: {selected_endpoint}[/dim]")

            result = {
                "source": "default",
                "endpoint": selected_endpoint,
                "prompts": test_data["prompts"],
                "images": test_data["images"],
                "pairs": test_data["pairs"],
            }
            return (result, "next")
        else:
            # LLM default behavior (prompts only) and VLM fallback if no endpoint provided
            from src.benchmarking.datasets.defaults import get_default_prompts_for_suite

            prompts = []
            images = []

            if num_prompts <= 20:
                prompts = get_default_prompts_for_suite(suite_name, model_type, count=num_prompts)
                console.print(f"\n[green]✓ Loaded {len(prompts)} suite-specific default prompts[/green]")
                result = {"source": "default", "prompts": prompts, "images": images}
                return (result, "next")
            else:
                prompts = get_default_prompts_for_suite(suite_name, model_type, count=20)
                console.print(f"\n[cyan]✓ Loaded {len(prompts)} default prompts[/cyan]")
                console.print(f"[yellow]→ Need {num_prompts - 20} more prompts via manual entry[/yellow]")
                console.print()
                console.print("[dim]Press Enter to start manual entry wizard...[/dim]")
                input()

                wizard_result = _manual_prompt_entry_wizard(
                    num_prompts=num_prompts - 20,
                    model_type=model_type,
                    selected_endpoint=selected_endpoint,
                    existing_prompts=[],
                    existing_images=[]
                )

                if wizard_result is None:
                    return (None, "back")
                elif wizard_result.get("_command") == "/runs":
                    return (None, "runs")
                elif wizard_result.get("_command") == "/default":
                    console.print(f"\n[yellow]Using only {len(prompts)} default prompts[/yellow]")
                    result = {"source": "default", "prompts": prompts, "images": images}
                    return (result, "next")

                prompts.extend(wizard_result["prompts"])
                if model_type == ModelType.VLM and "images" in wizard_result:
                    images.extend(wizard_result["images"])

                # Build pairs for VLM
                pairs = []
                if model_type == ModelType.VLM and images:
                    pairs = list(zip(images, prompts))

                console.print(f"\n[green]✓ Total: {len(prompts)} prompts ({20} default + {num_prompts - 20} manual)[/green]")

                result = {
                    "source": "mixed",
                    "prompts": prompts,
                    "images": images,
                    "pairs": pairs
                }
                return (result, "next")

    elif choice == "custom":
        # STEP 1: Choose between JSON upload or manual entry
        while True:
            tui.clear_screen()
            _show_step_header(6, 8, "Custom Prompts - Input Method", "Choose how to provide your custom prompts")

            # Build sub-menu for VLM custom prompts
            if model_type == ModelType.VLM:
                custom_options = [
                    ("json", "[cyan]Upload JSON File[/cyan]", f"Load image-prompt pairs from JSON (supports duplicate images)"),
                    ("manual", "[yellow]Enter Manually[/yellow]", f"Enter {num_prompts} image-prompt pairs one by one"),
                ]
            else:  # LLM
                custom_options = [
                    ("manual", "[yellow]Enter Manually[/yellow]", f"Enter {num_prompts} prompts one by one"),
                ]

            custom_options.extend([
                ("__previous", "[blue]← Previous[/blue]", "Go back to test data selection"),
                ("__cancel", "[red]Cancel[/red]", "Cancel benchmark configuration"),
            ])

            custom_choice = professional_prompt.get_arrow_selection(
                options=custom_options,
                title="Select Custom Prompt Input Method",
                instructions="Use ↑/↓ arrows to navigate, Enter to select"
            )

            if custom_choice is None or custom_choice == "__cancel":
                return (None, "cancel")
            elif custom_choice == "__previous":
                return (None, "back")

            # STEP 2: Handle JSON upload
            if custom_choice == "json":
                json_result = _json_pairs_upload_wizard(
                    model_type=model_type,
                    selected_endpoint=selected_endpoint
                )

                if json_result is None:
                    continue  # Go back to custom input method selection

                # Successfully loaded JSON
                prompts = json_result["prompts"]
                images = json_result.get("images", [])
                pairs = json_result.get("pairs", [])

                console.print(f"\n[green]✓ Successfully loaded {len(prompts)} prompts from JSON[/green]")

                result = {
                    "source": "json_upload",
                    "prompts": prompts,
                    "images": images,
                    "pairs": pairs
                }
                return (result, "next")

            # STEP 3: Handle manual entry
            elif custom_choice == "manual":
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
                    continue  # Go back to custom input method selection
                elif wizard_result.get("_command") == "/runs":
                    return (None, "runs")
                elif wizard_result.get("_command") == "/default":
                    # User wants to switch to defaults
                    from src.benchmarking.datasets.defaults import get_default_prompts_for_suite
                    prompts = get_default_prompts_for_suite(suite_name, model_type, count=min(num_prompts, 20))[:num_prompts]
                    console.print(f"\n[yellow]Switched to {len(prompts)} default prompts[/yellow]")
                    result = {"source": "default", "prompts": prompts, "images": []}
                    return (result, "next")

                # Successfully collected all prompts (enforce count limit)
                prompts = wizard_result["prompts"][:num_prompts]
                images = wizard_result.get("images", [])

                # Build pairs for VLM
                pairs = []
                if model_type == ModelType.VLM and images:
                    pairs = list(zip(images, prompts))

                console.print(f"\n[green]✓ Successfully entered {len(prompts)} custom prompts[/green]")

                result = {
                    "source": "custom",
                    "prompts": prompts,
                    "images": images,
                    "pairs": pairs
                }
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

    # Initialize parameters with defaults (common across providers)
    params = {
        "num_warmup": (current_value or {}).get("num_warmup", 1),
        "temperature": (current_value or {}).get("temperature", 0.7),
        "top_p": (current_value or {}).get("top_p", 0.9),
        "top_k": (current_value or {}).get("top_k", 50),
        "max_tokens": (current_value or {}).get("max_tokens", 512),
        "n_ctx": (current_value or {}).get("n_ctx", 2048),
        "image_resolution": (current_value or {}).get("image_resolution", "1280x1024"),
    }

    while True:
        tui.clear_screen()
        _show_step_header(7, 8, "Benchmark Parameters", "Configure model inference parameters")

        # Show current configuration summary as a structured table
        from rich.table import Table as _Table
        from rich.panel import Panel as _Panel
        from rich.columns import Columns as _Columns

        param_table = _Table(padding=(0, 1))
        param_table.add_column("Parameter", style="cyan")
        param_table.add_column("Value", style="white")
        param_table.add_column("Range", style="dim")

        param_table.add_row("Warmup runs", str(params['num_warmup']), "0-5")
        param_table.add_row("Temperature", str(params['temperature']), "0.0-1.0")
        param_table.add_row("Top-p", str(params['top_p']), "0.0-1.0")
        param_table.add_row("Top-k", str(params['top_k']), "0-200")
        param_table.add_row("Max tokens", str(params['max_tokens']), "1-4096")
        param_table.add_row("n_ctx", str(params['n_ctx']), "256-32768")
        param_table.add_row("Image resolution", str(params['image_resolution']), "WIDTHxHEIGHT")

        console.print(param_table)
        console.print("[dim]Note: Number of prompts configured in Step 5[/dim]")
        console.print()

        # Build interactive menu options
        from src.cli.text_input import professional_prompt
        menu_options = [
            ("edit_warmup", "Warmup runs", f"Currently: {params['num_warmup']} (range: 0-5)"),
            ("edit_temp", "Temperature", f"Currently: {params['temperature']} (range: 0.0-1.0)"),
            ("edit_top_p", "Top-p", f"Currently: {params['top_p']} (range: 0.0-1.0)"),
            ("edit_top_k", "Top-k", f"Currently: {params['top_k']} (range: 0-200)"),
            ("edit_tokens", "Max tokens", f"Currently: {params['max_tokens']} (range: 1-4096)"),
            ("edit_nctx", "n_ctx", f"Currently: {params['n_ctx']} (range: 256-32768)"),
            ("edit_img_res", "Image resolution", f"Currently: {params['image_resolution']} (format: WIDTHxHEIGHT)"),
            ("__separator", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ""),
            ("__continue", "Continue to Final Review →", "Proceed to configuration review"),
            ("__previous", "← Previous", "Go back to test data selection"),
            ("__cancel", "Cancel", "Cancel benchmark configuration"),
        ]

        choice = professional_prompt.get_arrow_selection(
            options=menu_options,
            title="Select parameter to edit or continue",
            instructions="Use ↑/↓ arrows to navigate, Enter to select"
        )

        choice_key = choice

        if choice_key is None or choice_key == "__cancel":
            return (None, "cancel")
        if choice_key == "__previous":
            return (None, "back")
        if choice_key == "__continue":
            return (params, "next")

        # Selected a parameter card; open corresponding editor
        if choice_key == "edit_warmup":
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

        elif choice_key == "edit_temp":
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

        elif choice_key == "edit_top_p":
            tui.clear_screen()
            console.print("[bold cyan]Edit Top-p[/bold cyan]")
            tp = professional_prompt.get_input(
                f"Top-p (default: {params['top_p']})",
                style="cyan",
                allow_multiline=False,
                show_instructions=False
            )
            if tp:
                try:
                    params['top_p'] = max(0.0, min(1.0, float(tp)))
                except ValueError:
                    pass

        elif choice_key == "edit_top_k":
            tui.clear_screen()
            console.print("[bold cyan]Edit Top-k[/bold cyan]")
            tk = professional_prompt.get_input(
                f"Top-k (default: {params['top_k']})",
                style="cyan",
                allow_multiline=False,
                show_instructions=False
            )
            if tk:
                try:
                    v = int(tk)
                    params['top_k'] = max(0, min(200, v))
                except ValueError:
                    pass

        elif choice_key == "edit_tokens":
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

        elif choice_key == "edit_nctx":
            tui.clear_screen()
            console.print("[bold cyan]Edit n_ctx[/bold cyan]")
            console.print("[dim]Context window size (tokens kept in memory)[/dim]")
            new_value = professional_prompt.get_numeric(
                prompt_msg="n_ctx",
                min_val=256,
                max_val=32768,
                default=params["n_ctx"],
                style="cyan"
            )
            params["n_ctx"] = new_value

        elif choice_key == "edit_img_res":
            tui.clear_screen()
            console.print()
            console.print("[bold cyan]Edit Image Resolution[/bold cyan]")
            console.print("[dim]VLM input images will be rescaled to this resolution (maintains aspect ratio with padding)[/dim]")
            console.print(f"[dim]Format: WIDTHxHEIGHT (e.g., 1280x1024, 1920x1080)[/dim]")
            console.print(f"[dim]Range: 256x256 to 4096x4096 per dimension[/dim]")
            console.print(f"[dim]Current: {params['image_resolution']}[/dim]")
            console.print()

            resolution_input = professional_prompt.get_input(
                f"Image resolution (default: {params['image_resolution']})",
                style="cyan",
                allow_multiline=False,
                show_instructions=False
            )

            # Use default if user pressed Enter without input
            if not resolution_input.strip():
                resolution_input = params['image_resolution']

            # Parse and validate resolution (WIDTHxHEIGHT)
            import re
            match = re.match(r'(\d+)x(\d+)', resolution_input.strip())
            if match:
                width, height = int(match.group(1)), int(match.group(2))
                # Validate range (256-4096 per dimension)
                if 256 <= width <= 4096 and 256 <= height <= 4096:
                    params["image_resolution"] = f"{width}x{height}"
                    console.print(f"[green]✓ Image resolution set to: {width}x{height}[/green]")
                else:
                    console.print(f"[red]✗ Invalid range. Must be 256-4096 per dimension. Keeping: {params['image_resolution']}[/red]")
            else:
                console.print(f"[red]✗ Invalid format. Use WIDTHxHEIGHT (e.g., 1280x1024). Keeping: {params['image_resolution']}[/red]")

            console.print()
            tui.prompt("Press Enter to continue...", style="dim")

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
