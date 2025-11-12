"""Benchmark menu integration for CLI."""

from typing import Optional, List
from loguru import logger
from rich.table import Table

from ..benchmarking.core.benchmark_runner import BenchmarkRunner
from ..benchmarking.models.benchmark_config import BenchmarkConfig
from ..benchmarking.models.metric_types import ModelType, SuiteType, ExecutionMode
from ..benchmarking.datasets.defaults import get_prompts_for_suite
from ..services.session import SessionManager
from .tui_manager import tui


class BenchmarkMenu:
    """Benchmark menu for selecting and configuring benchmarks."""

    @staticmethod
    def show_main_menu() -> Optional[str]:
        """
        Show main benchmark menu.

        Returns:
            Selected option or None to return
        """
        tui.clear_screen()

        tui.show_panel(
            """[bold cyan]Benchmark Mode[/bold cyan]

[1] [green]Speed & Throughput[/green]
    Measure latency and tokens per second

[2] [yellow]Resource Efficiency[/yellow]
    Monitor CPU, GPU, and memory usage

[3] [magenta]Complete Analysis[/magenta]
    Run all benchmark suites together

[4] [blue]View Previous Results[/blue]
    Browse and compare benchmark history

[bold]Navigation:[/bold]
  [h] Main menu (home)
  [b] Back to main menu
  [q] Quit application""",
            title="Benchmarking",
            border_style="cyan"
        )

        choice = tui.prompt("\nChoose [1-4/h/b/q]:", style="cyan").strip().lower()
        return choice

    @staticmethod
    def _parse_model_selection(input_str: str, max_value: int) -> list[int]:
        """Parse user selection input supporting multiple formats.

        Supports:
        - Single: "5" → [5]
        - Multiple: "1 2 3" or "1,2,3" → [1, 2, 3]
        - Range: "1-5" → [1, 2, 3, 4, 5]
        - Mixed: "1 3-5 7" → [1, 3, 4, 5, 7]

        Args:
            input_str: User input string
            max_value: Maximum valid value

        Returns:
            List of selected indices (1-based)

        Raises:
            ValueError: If input is invalid
        """
        if not input_str or input_str.strip().lower() == 'c':
            return []

        indices = set()
        input_str = input_str.replace(',', ' ')
        parts = input_str.split()

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check for range (e.g., "1-5")
            if '-' in part:
                try:
                    start, end = part.split('-', 1)
                    start_idx = int(start.strip())
                    end_idx = int(end.strip())

                    if start_idx < 1 or end_idx > max_value or start_idx > end_idx:
                        raise ValueError(f"Invalid range: {part}")

                    indices.update(range(start_idx, end_idx + 1))
                except (ValueError, AttributeError) as e:
                    raise ValueError(f"Invalid range format: {part}") from e
            else:
                # Single number
                try:
                    idx = int(part)
                    if idx < 1 or idx > max_value:
                        raise ValueError(f"Index {idx} out of range [1-{max_value}]")
                    indices.add(idx)
                except ValueError as e:
                    raise ValueError(f"Invalid number: {part}") from e

        return sorted(list(indices))

    @staticmethod
    def _show_models_by_type(models: List, model_type: ModelType) -> Optional[List[str]]:
        """
        Display available models filtered by type and let user select multiple.

        Args:
            models: List of available models
            model_type: Filter by this model type (LLM or VLM)

        Returns:
            List of selected model_ids or None if cancelled
        """
        # Filter models by type
        try:
            filtered_models = [m for m in models if hasattr(m, 'model_type') and str(m.model_type).upper() == model_type.value.upper()]
        except Exception as e:
            logger.error(f"Error filtering models: {e}", exc_info=True)
            filtered_models = []

        if not filtered_models:
            tui.show_message(
                f"No {model_type.value.upper()} models found.\n\n"
                f"Please install models in the Inference mode first.",
                title="No Models Available",
                style="yellow"
            )
            return None

        tui.clear_screen()
        tui.console.print(f"\n[bold cyan]Available {model_type.value.upper()} Models[/bold cyan]\n")

        # Create table
        table = Table(show_header=True, header_style="bold cyan", border_style="cyan")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Model Name", style="green")
        table.add_column("Provider", style="yellow")
        table.add_column("Size (GB)", style="magenta")
        table.add_column("RAM Est.", style="blue")

        for idx, model in enumerate(filtered_models, 1):
            try:
                # Safely get model name
                model_name = model.name if hasattr(model, 'name') and model.name else "Unknown"

                # Safely get provider
                provider = "Unknown"
                if hasattr(model, 'provider') and model.provider:
                    provider = str(model.provider).upper()

                # Safely get size
                size_gb = "N/A"
                if hasattr(model, 'size_gb') and model.size_gb is not None:
                    try:
                        size_gb = f"{float(model.size_gb):.1f}"
                    except (ValueError, TypeError):
                        size_gb = "N/A"

                # Safely get RAM estimate
                ram_gb = "N/A"
                if hasattr(model, 'ram_gb') and model.ram_gb is not None:
                    try:
                        ram_gb = f"{float(model.ram_gb):.1f}"
                    except (ValueError, TypeError):
                        ram_gb = "N/A"

                table.add_row(
                    str(idx),
                    model_name,
                    provider,
                    size_gb,
                    ram_gb,
                )
            except Exception as e:
                logger.warning(f"Error displaying model {idx}: {e}")
                table.add_row(str(idx), "Error displaying model", "N/A", "N/A", "N/A")

        tui.show_panel(table)

        # Get user selection
        while True:
            choice = tui.prompt(
                f"\nSelect model(s) [1-{len(filtered_models)}] or 'b' to go back:\n"
                f"[dim]Examples: '5' (single), '1 2 3' (multiple), '1-5' (range), '1 3-5 7' (mixed)[/dim]",
                style="cyan"
            ).strip()

            if choice.lower() == "b":
                logger.info("User cancelled model selection")
                return None

            try:
                selected_indices = BenchmarkMenu._parse_model_selection(choice, len(filtered_models))

                if not selected_indices:
                    tui.show_error("Please select at least one model")
                    continue

                selected_models = [filtered_models[idx - 1].model_id for idx in selected_indices]
                logger.info(f"Selected {len(selected_models)} model(s): {selected_models}")
                return selected_models

            except ValueError as e:
                tui.show_error(f"Invalid input: {e}")

    @staticmethod
    def configure_benchmark(suite_type: SuiteType, session_manager: SessionManager) -> Optional[BenchmarkConfig]:
        """
        Configure a benchmark interactively.

        Args:
            suite_type: Type of benchmark suite
            session_manager: Session manager with model discovery

        Returns:
            BenchmarkConfig or None if cancelled
        """
        tui.clear_screen()
        tui.show_panel(
            f"[bold cyan]Configure {suite_type.value.upper()} Benchmark[/bold cyan]\n\n"
            f"Follow the prompts to set up your benchmark configuration.",
            title="Benchmark Configuration",
            border_style="cyan"
        )

        # Step 1: Model type selection
        from ..cli.text_input import professional_prompt

        tui.console.print("\n[cyan]Step 1: Select Model Type[/cyan]")
        tui.console.print()

        model_type_options = [
            ("llm", "[green]LLM[/green]", "Text-only language models"),
            ("vlm", "[magenta]VLM[/magenta]", "Vision-language models"),
        ]

        model_type_choice = professional_prompt.get_arrow_selection(
            options=model_type_options,
            title="Select Model Type",
            instructions="Use ↑/↓ arrows to navigate, Enter to select"
        )

        if model_type_choice is None:
            return None

        model_type = ModelType.LLM if model_type_choice == "llm" else ModelType.VLM

        # Step 2: Discover and select model
        try:
            tui.console.print(f"\n[cyan]Step 2: Select {model_type.value.upper()} Model[/cyan]")
            logger.info(f"Discovering available {model_type.value.upper()} models...")

            all_models = session_manager.discover_models()

            if not all_models:
                tui.show_error(
                    f"No models found!\n\n"
                    f"Please install models in the Inference mode first.\n"
                    f"Then return to Benchmarking to select them."
                )
                return None

            logger.info(f"Discovered {len(all_models)} total models")
            model_ids = BenchmarkMenu._show_models_by_type(all_models, model_type)

            if not model_ids:
                logger.info("User cancelled model selection")
                return None

            # If VLMs selected, pre-select mmproj for any quantized GGUF VLMs with multiple candidates
            if model_type == ModelType.VLM:
                try:
                    from ..models.endpoints import ProviderType
                    id_to_model = {m.model_id: m for m in all_models}
                    quant_provider = session_manager.model_discovery.get_provider(ProviderType.QUANTIZED)
                    # Only proceed if quantized provider is available and supports mmproj selection helpers
                    if quant_provider is not None and hasattr(quant_provider, "_find_mmproj_candidates"):
                        from ..cli.text_input import professional_prompt
                        from ..cli.tui_manager import tui as _tui
                        from pathlib import Path

                        # Iterate each selected model; if quantized GGUF, ask for mmproj when multiple candidates
                        for mid in model_ids:
                            m = id_to_model.get(mid)
                            if not m:
                                continue
                            provider_obj = getattr(m, "provider", None)
                            provider_name = str(provider_obj.value if hasattr(provider_obj, "value") else provider_obj).lower()
                            if not (provider_obj == ProviderType.QUANTIZED or provider_name == "quantized"):
                                continue
                            # Parse quantized model-id: quantized:gguf:/abs/path or quantized:hf:/path
                            parts = str(mid).split(":", 2)
                            if len(parts) != 3 or parts[1] != "gguf":
                                continue  # mmproj relevant for gguf
                            lang_path = Path(parts[2])

                            # Find mmproj candidates across results/quantizations
                            try:
                                candidates = quant_provider._find_mmproj_candidates(lang_path)
                            except Exception as e:
                                logger.debug(f"Could not enumerate mmproj for {lang_path}: {e}")
                                continue

                            if not candidates:
                                continue

                            # If single candidate, remember it for session to avoid prompt later
                            if len(candidates) == 1:
                                quant_provider._session_mmproj_choice[str(lang_path)] = candidates[0]
                                logger.info(f"Auto-selected mmproj for {lang_path.name}: {candidates[0].name}")
                                continue

                            # Multiple candidates — ask user via arrow menu
                            options = []
                            for p in candidates:
                                try:
                                    meta = quant_provider.metadata_cache.get_metadata(str(p), provider="gguf")
                                except Exception:
                                    meta = None
                                size_gb = 0.0
                                try:
                                    size_gb = p.stat().st_size / (1024 ** 3)
                                except Exception:
                                    pass
                                quant = (meta.quantization.upper() if meta and getattr(meta, "quantization", None) else "UNKNOWN")
                                label = f"[green]{p.name}[/green]"
                                desc = f"{size_gb:.2f} GB • {quant}"
                                options.append((str(p), label, desc))

                            _tui.console.print(f"\n[cyan]Select vision encoder (mmproj) for[/cyan] [yellow]{lang_path.name}[/yellow]")
                            choice = professional_prompt.get_arrow_selection(
                                options=options,
                                title="Select Vision Encoder (mmproj)",
                                instructions="Use ↑/↓ arrows, Enter to select, q to cancel",
                            )
                            if not choice:
                                _tui.console.print("[yellow]Skipping mmproj assignment (will auto-detect later)[/yellow]")
                                continue
                            sel_path = Path(choice)
                            quant_provider._session_mmproj_choice[str(lang_path)] = sel_path
                            logger.info(f"User selected mmproj for {lang_path.name}: {sel_path.name}")
                except Exception as mm_err:
                    logger.debug(f"mmproj pre-selection skipped due to: {mm_err}")

        except Exception as e:
            logger.error(f"Failed to discover models: {e}", exc_info=True)
            tui.show_error(
                f"Failed to discover models:\n\n"
                f"{str(e)}\n\n"
                f"Please check the logs for details."
            )
            return None

        # Step 3: Endpoint selection (auto-determined)
        if model_type == ModelType.LLM:
            endpoint = "text/chat"
        else:
            endpoint = "vision/analyze"

        # Step 3: Test Data
        tui.console.print(f"\n[cyan]Step 3/8: Test Data[/cyan]")
        use_default_choice = tui.prompt("Use default test prompts? [Y/n]:", style="cyan").strip().lower()
        use_default = use_default_choice != "n"

        if use_default:
            model_type_str = "llm" if model_type == ModelType.LLM else "vlm"
            prompts = get_prompts_for_suite(suite_type.value, model_type_str)
            tui.console.print(f"[dim]Using {len(prompts)} default prompts[/dim]")
        else:
            # For fair comparison, use ONE prompt for all models and all runs
            tui.console.print(f"\n[cyan]Enter ONE prompt to test all {len(model_ids)} model(s)[/cyan]")
            tui.console.print("[dim]This prompt will be used consistently for fair comparison across all models and runs[/dim]")

            prompt = tui.prompt("Enter your prompt:", style="cyan").strip()

            if not prompt:
                tui.show_error("No prompt entered. Using defaults.")
                model_type_str = "llm" if model_type == ModelType.LLM else "vlm"
                prompts = get_prompts_for_suite(suite_type.value, model_type_str)
            else:
                # Use the same prompt for all runs (will be repeated internally)
                prompts = [prompt]
                tui.console.print(f"[dim]✓ Prompt accepted ({len(prompt)} chars)[/dim]")

        # Step 4 (VLM only): Select test images
        images: list[str] = []
        if model_type == ModelType.VLM:
            tui.console.print(f"\n[cyan]Step 4/8: Test Images (VLM)[/cyan]")
            tui.console.print("Enter one or more image file paths (comma-separated). Must exist.")
            from pathlib import Path as _P
            while True:
                img_in = tui.prompt("Images:", style="cyan").strip()
                imgs = [p.strip() for p in img_in.split(',') if p.strip()]
                valid = [p for p in imgs if _P(p).exists()]
                if valid:
                    images = valid
                    tui.console.print(f"[dim]✓ Using {len(images)} image(s)[/dim]")
                    break
                tui.show_error("No valid image paths provided. Please enter at least one valid file path.")

        # Step 5: Number of Prompts (Speed only)
        if suite_type == SuiteType.SPEED:
            tui.console.print(f"\n[cyan]Step 5/8: Number of Prompts[/cyan]")
            current_prompts = len(prompts)
            target_prompts = min(10, current_prompts) if current_prompts > 0 else 1
            from rich.table import Table as _Table
            ptbl = _Table.grid(padding=(0, 2))
            ptbl.add_column(style="cyan")
            ptbl.add_column(style="white")
            ptbl.add_row("Parameter", "Current Value")
            ptbl.add_row("Prompts", f"{target_prompts}")
            tui.console.print(ptbl)
            tui.console.print("[dim]Range: 1-50 | Default: 10[/dim]")

            from ..cli.text_input import ProfessionalPrompt as _PP
            _pp = _PP()
            while True:
                choice = _pp.get_arrow_selection(
                    options=[
                        ("edit", "Edit number of prompts", "Specify how many prompts to use (1-50)"),
                        ("default", "Use default", "Reset to default value (10)"),
                        ("continue", "Continue to Step 6 →", "Proceed to parameter selection"),
                        ("prev", "← Previous", "Back to endpoint selection"),
                        ("cancel", "Cancel", "Cancel benchmark configuration"),
                    ],
                    title="Configure number of prompts",
                    instructions="Use ↑/↓ arrows to navigate, Enter to select"
                )
                if choice == "edit":
                    val = tui.prompt("Enter number of prompts [1-50]:", style="cyan").strip()
                    try:
                        n = int(val)
                        if n < 1 or n > 50:
                            raise ValueError
                        target_prompts = max(1, min(50, n))
                    except Exception:
                        tui.show_error("Invalid number. Please enter an integer between 1 and 50.")
                        continue
                elif choice == "default":
                    target_prompts = min(10, current_prompts) if current_prompts > 0 else 1
                elif choice == "continue":
                    break
                elif choice in ("prev", None):
                    return None
                elif choice == "cancel":
                    return None
                # refresh table
                ptbl = _Table.grid(padding=(0, 2))
                ptbl.add_column(style="cyan")
                ptbl.add_column(style="white")
                ptbl.add_row("Parameter", "Current Value")
                ptbl.add_row("Prompts", f"{target_prompts}")
                tui.clear_and_show_with_status(ptbl, "Range: 1-50 | Default: 10")

            if len(prompts) > target_prompts:
                prompts = prompts[:target_prompts]

        # Step 6: Number of Runs
        tui.console.print(f"\n[cyan]Step 6/8: Number of Runs[/cyan]")
        num_runs_input = tui.prompt("Number of runs [1-100] (default: 5):", style="cyan").strip()
        try:
            num_runs = int(num_runs_input) if num_runs_input else 5
            num_runs = max(1, min(100, num_runs))
        except ValueError:
            num_runs = 5
            tui.console.print(f"[dim]Invalid input, using default: {num_runs}[/dim]")

        # Step 7: Warmup runs & Parameters
        warmup_input = tui.prompt("Number of warmup runs [0-10] (default: 2):", style="cyan").strip()
        try:
            num_warmup = int(warmup_input) if warmup_input else 2
            num_warmup = max(0, min(10, num_warmup))
        except ValueError:
            num_warmup = 2
            tui.console.print(f"[dim]Invalid input, using default: {num_warmup}[/dim]")

        # Suite-specific configuration
        suite_config = {}

        if suite_type == SuiteType.STRESS:
            # Stress suite: Ask for duration
            tui.console.print(f"\n[cyan]Stress Suite Configuration[/cyan]")
            tui.console.print("[dim]Configure stress test duration and checkpoints[/dim]")

            # Show clear information about total time
            num_models = len(model_ids)
            if num_models > 1:
                tui.console.print(f"\n[yellow]Note: You have selected {num_models} models.[/yellow]")
                tui.console.print(f"[yellow]Each model will be stress tested sequentially for the duration you specify.[/yellow]")
                tui.console.print(f"[dim]Example: 30 min/model × {num_models} models = {30 * num_models} minutes total[/dim]\n")

            duration_input = tui.prompt(
                f"Duration per model in minutes [5-120] (default: 30):" if num_models > 1
                else "Duration in minutes [5-120] (default: 30):",
                style="cyan"
            ).strip()
            try:
                duration_minutes = int(duration_input) if duration_input else 30
                duration_minutes = max(5, min(120, duration_minutes))
            except ValueError:
                duration_minutes = 30
                tui.console.print(f"[dim]Invalid input, using default: {duration_minutes} minutes[/dim]")

            checkpoint_input = tui.prompt("Checkpoint interval in minutes [1-30] (default: 5):", style="cyan").strip()
            try:
                checkpoint_minutes = int(checkpoint_input) if checkpoint_input else 5
                checkpoint_minutes = max(1, min(30, checkpoint_minutes))
            except ValueError:
                checkpoint_minutes = 5
                tui.console.print(f"[dim]Invalid input, using default: {checkpoint_minutes} minutes[/dim]")

            suite_config["duration_minutes"] = duration_minutes
            suite_config["checkpoint_interval_seconds"] = checkpoint_minutes * 60

            # Show total time calculation
            total_time = duration_minutes * num_models
            if num_models > 1:
                tui.console.print(
                    f"[dim]✓ Each model: {duration_minutes} min with checkpoints every {checkpoint_minutes} min[/dim]"
                )
                tui.console.print(
                    f"[dim]✓ Total execution time: ~{total_time} minutes ({total_time // 60}h {total_time % 60}m)[/dim]"
                )
            else:
                tui.console.print(
                    f"[dim]✓ Stress test will run for {duration_minutes} minutes with checkpoints every {checkpoint_minutes} minutes[/dim]"
                )

        elif suite_type == SuiteType.COMPLETE:
            # Complete suite: Ask for runs per sub-suite
            num_models = len(model_ids)
            tui.console.print(f"\n[cyan]Complete Suite Configuration[/cyan]")
            tui.console.print(f"[dim]This runs all benchmark suites for {num_models} model(s). Configure parameters:[/dim]")

            speed_runs_input = tui.prompt("Runs for Speed suite [1-50] (default: 5):", style="cyan").strip()
            try:
                speed_runs = int(speed_runs_input) if speed_runs_input else 5
                speed_runs = max(1, min(50, speed_runs))
            except ValueError:
                speed_runs = 5

            resources_runs_input = tui.prompt("Runs for Resources suite [1-50] (default: 5):", style="cyan").strip()
            try:
                resources_runs = int(resources_runs_input) if resources_runs_input else 5
                resources_runs = max(1, min(50, resources_runs))
            except ValueError:
                resources_runs = 5

            quality_runs_input = tui.prompt("Runs for Quality suite [1-50] (default: 10):", style="cyan").strip()
            try:
                quality_runs = int(quality_runs_input) if quality_runs_input else 10
                quality_runs = max(1, min(50, quality_runs))
            except ValueError:
                quality_runs = 10

            stress_prompt = (
                f"Duration for Stress suite per model in minutes [5-60] (default: 15):" if num_models > 1
                else "Duration for Stress suite in minutes [5-60] (default: 15):"
            )
            stress_duration_input = tui.prompt(stress_prompt, style="cyan").strip()
            try:
                stress_duration = int(stress_duration_input) if stress_duration_input else 15
                stress_duration = max(5, min(60, stress_duration))
            except ValueError:
                stress_duration = 15

            suite_config["speed_runs"] = speed_runs
            suite_config["resources_runs"] = resources_runs
            suite_config["quality_runs"] = quality_runs
            suite_config["stress_duration_minutes"] = stress_duration

            # Show summary with total stress time
            if num_models > 1:
                total_stress_time = stress_duration * num_models
                tui.console.print(f"[dim]✓ Complete suite configured with custom run counts per sub-suite[/dim]")
                tui.console.print(f"[dim]✓ Stress test: {stress_duration} min/model × {num_models} models = {total_stress_time} min total[/dim]")
            else:
                tui.console.print(f"[dim]✓ Complete suite configured with custom run counts per sub-suite[/dim]")

        # Default to foreground execution
        execution_mode = ExecutionMode.FOREGROUND

        # Create endpoints dictionary for all selected models
        endpoints = {model_id: [endpoint] for model_id in model_ids}

        # Step 7/8: Benchmark Parameters (tabular, editable)
        tui.console.print(f"\n[cyan]Step 7/8: Benchmark Parameters[/cyan]")
        temp_in = tui.prompt("Temperature [0.0-1.0] (default: 0.7):", style="cyan").strip()
        top_p_in = tui.prompt("Top-p [0.0-1.0] (default: 0.9):", style="cyan").strip()
        top_k_in = tui.prompt("Top-k [0-200] (default: 50):", style="cyan").strip()
        max_tok_in = tui.prompt("Max tokens [16-8192] (default: 512):", style="cyan").strip()
        n_ctx_in = tui.prompt("n_ctx [256-32768] (default: 4096):", style="cyan").strip()

        def _parse_float(val: str, d: float, lo: float, hi: float) -> float:
            try:
                x = float(val) if val else d
                return max(lo, min(hi, x))
            except Exception:
                return d

        def _parse_int(val: str, d: int, lo: int, hi: int) -> int:
            try:
                x = int(val) if val else d
                return max(lo, min(hi, x))
            except Exception:
                return d

        temperature = _parse_float(temp_in, 0.7, 0.0, 1.0)
        top_p = _parse_float(top_p_in, 0.9, 0.0, 1.0)
        top_k = _parse_int(top_k_in, 50, 0, 200)
        max_tokens = _parse_int(max_tok_in, 512, 16, 8192)
        n_ctx_val = _parse_int(n_ctx_in, 4096, 256, 32768)

        # Build parameters with inference settings and suite-specific config
        parameters = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "n_ctx": n_ctx_val,
            **suite_config  # Add suite-specific configuration
        }

        config = BenchmarkConfig(
            model_type=model_type,
            suite_type=suite_type,
            models=model_ids,
            endpoints=endpoints,
            test_data=test_data,
            parameters=parameters,
            num_runs=num_runs,
            num_warmup=num_warmup,
            execution_mode=execution_mode,
            export_formats=["json", "csv"]
        )

        # Final Step: Configuration Review
        tui.clear_screen()
        model_list = "\n".join([f"  • {mid}" for mid in model_ids])

        # Build suite-specific summary
        suite_summary = ""
        if suite_type == SuiteType.STRESS:
            duration_per_model = suite_config['duration_minutes']
            total_duration = duration_per_model * len(model_ids)

            if len(model_ids) > 1:
                suite_summary = f"""
[cyan]Duration per model:[/cyan] {duration_per_model} minutes
[cyan]Total duration:[/cyan] {total_duration} minutes ({total_duration // 60}h {total_duration % 60}m)
[cyan]Checkpoints:[/cyan] Every {suite_config['checkpoint_interval_seconds'] // 60} minutes"""
            else:
                suite_summary = f"""
[cyan]Duration:[/cyan] {duration_per_model} minutes
[cyan]Checkpoints:[/cyan] Every {suite_config['checkpoint_interval_seconds'] // 60} minutes"""
        elif suite_type == SuiteType.COMPLETE:
            stress_per_model = suite_config['stress_duration_minutes']
            if len(model_ids) > 1:
                total_stress = stress_per_model * len(model_ids)
                suite_summary = f"""
[cyan]Speed Runs:[/cyan] {suite_config['speed_runs']}
[cyan]Resources Runs:[/cyan] {suite_config['resources_runs']}
[cyan]Quality Runs:[/cyan] {suite_config['quality_runs']}
[cyan]Stress per model:[/cyan] {stress_per_model} min
[cyan]Total stress:[/cyan] {total_stress} min ({total_stress // 60}h {total_stress % 60}m)"""
            else:
                suite_summary = f"""
[cyan]Speed Runs:[/cyan] {suite_config['speed_runs']}
[cyan]Resources Runs:[/cyan] {suite_config['resources_runs']}
[cyan]Quality Runs:[/cyan] {suite_config['quality_runs']}
[cyan]Stress Duration:[/cyan] {stress_per_model} minutes"""
        else:
            suite_summary = f"""
[cyan]Benchmark Runs:[/cyan] {num_runs} runs × {len(model_ids)} model(s) = {num_runs * len(model_ids)} total
[cyan]Warmup Runs:[/cyan] {num_warmup}"""

        from rich.table import Table as _Table
        review_tbl = _Table.grid(padding=(0, 2))
        review_tbl.add_column(style="cyan")
        review_tbl.add_column(style="white")
        review_tbl.add_row("Setting", "Value")
        review_tbl.add_row("Model Type", model_type.value.upper())
        review_tbl.add_row("Suite", suite_type.display_name())
        review_tbl.add_row("Models", str(len(model_ids)))
        runs_per_model = len(prompts) if suite_type == SuiteType.SPEED else num_runs
        review_tbl.add_row("Runs per model", str(runs_per_model))
        review_tbl.add_row("Warmup runs", str(num_warmup))
        review_tbl.add_row("Execution mode", execution_mode.value.title())
        review_tbl.add_row("Temperature", str(temperature))
        review_tbl.add_row("Top-p", str(top_p))
        review_tbl.add_row("Top-k", str(top_k))
        review_tbl.add_row("Max tokens", str(max_tokens))
        review_tbl.add_row("n_ctx", str(n_ctx_val))

        from rich.panel import Panel as _Panel
        tui.show_panel(review_tbl, title="Final Step: Configuration Review", border_style="green")
        tui.console.print(f"Selected models:\n{model_list}")

        from ..cli.text_input import ProfessionalPrompt as _PP2
        _p2 = _PP2()
        choice = _p2.get_arrow_selection(
            options=[
                ("START", "Start Benchmark", "Begin benchmark execution with this configuration"),
                ("SAVE", "Save Configuration", "Save this configuration to file for later use"),
                ("EDIT", "Edit Configuration", "Go back to parameters to make changes"),
                ("CANCEL", "Cancel", "Cancel benchmark and return to main menu"),
            ],
            title="Review Complete - Ready to Start?",
            instructions="Use ↑/↓ arrows to navigate, Enter to select"
        )

        if choice == "START":
            return config
        if choice == "SAVE":
            try:
                from pathlib import Path as _Path
                out_dir = _Path("results/configs")
                out_dir.mkdir(parents=True, exist_ok=True)
                cfg_path = out_dir / f"{config.benchmark_id}.json"
                cfg_path.write_text(config.to_json(), encoding="utf-8")
                tui.show_message(f"Saved configuration to {cfg_path}", title="Saved", style="green")
            except Exception as e:
                tui.show_error(f"Failed to save configuration: {e}")
            return None
        if choice == "EDIT":
            return None
        return None

        return None

    @staticmethod
    def run_benchmark(config: BenchmarkConfig, session_manager: SessionManager) -> None:
        """
        Run benchmark with progress display.

        Args:
            config: BenchmarkConfig
            session_manager: SessionManager for model execution
        """
        runner = BenchmarkRunner(session_manager=session_manager)

        tui.clear_screen()
        tui.console.print(f"\n[bold cyan]Starting {config.suite_type.value.upper()} Benchmark...[/bold cyan]\n")

        try:
            result = runner.run(config)

            # Summary
            tui.console.print(f"\n[bold green]✓ Benchmark Complete![/bold green]")
            tui.console.print(f"[cyan]Status:[/cyan] {result.status.value}")
            tui.console.print(f"[cyan]Duration:[/cyan] {result.duration_seconds:.2f}s")
            tui.console.print(f"[cyan]Successful runs:[/cyan] {result.successful_runs}/{result.total_runs}")

            # Next action prompt
            from ..cli.text_input import ProfessionalPrompt
            prompt = ProfessionalPrompt()
            next_choice = prompt.get_arrow_selection(
                options=[("DETAILS", "[green]Show Detailed Results[/green]", "Open detailed view"), ("BACK", "[dim]◄ Go back[/dim]", "Return")],
                title="Next Action",
                instructions="Use ↑/↓ to navigate, Enter to select"
            )
            if next_choice == "DETAILS":
                runner.print_result(result, detailed=True)

        except Exception as e:
            tui.clear_screen()
            tui.console.print(f"\n[bold red]✗ Benchmark Failed[/bold red]")
            tui.console.print(f"[red]Error: {str(e)}[/red]")
            logger.error(f"Benchmark execution failed: {e}", exc_info=True)

        tui.prompt("\nPress Enter to continue...", style="dim")

    @staticmethod
    def view_results() -> None:
        """View previous benchmark results."""
        tui.clear_screen()
        runner = BenchmarkRunner()
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

        from rich.table import Table
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
                ", ".join(result["models_tested"])[:20],
                result["status"],
                result.get("date", "N/A")
            )

        tui.show_panel(table)
        tui.prompt("\nPress Enter to continue...", style="dim")
