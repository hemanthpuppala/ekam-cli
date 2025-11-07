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

        # Step 4: Test data
        tui.console.print(f"\n[cyan]Step 3: Test Data[/cyan]")
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

        # Step 5: Number of runs
        tui.console.print(f"\n[cyan]Step 4: Number of Runs[/cyan]")
        num_runs_input = tui.prompt("Number of runs [1-100] (default: 5):", style="cyan").strip()
        try:
            num_runs = int(num_runs_input) if num_runs_input else 5
            num_runs = max(1, min(100, num_runs))
        except ValueError:
            num_runs = 5
            tui.console.print(f"[dim]Invalid input, using default: {num_runs}[/dim]")

        # Step 6: Warmup runs
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

        # Step 7: Execution mode
        tui.console.print(f"\n[cyan]Step 5: Execution Mode[/cyan]")
        tui.console.print()

        execution_mode_options = [
            ("foreground", "[green]Foreground[/green]", "Blocking with real-time progress display"),
            ("background", "[yellow]Background[/yellow]", "Non-blocking, run in background"),
        ]

        execution_mode_choice = professional_prompt.get_arrow_selection(
            options=execution_mode_options,
            title="Execution Mode",
            instructions="Use ↑/↓ arrows to navigate, Enter to select"
        )

        if execution_mode_choice is None:
            return None

        execution_mode = ExecutionMode.FOREGROUND if execution_mode_choice == "foreground" else ExecutionMode.BACKGROUND

        # Create configuration
        test_data = {"prompts": prompts}
        if model_type == ModelType.VLM:
            test_data["images"] = []

        # Create endpoints dictionary for all selected models
        endpoints = {model_id: [endpoint] for model_id in model_ids}

        # Build parameters with inference settings and suite-specific config
        parameters = {
            "temperature": 0.7,
            "max_tokens": 512,
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

        # Show summary and confirm
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

        summary = f"""[bold green]Configuration Summary[/bold green]

[cyan]Models ({len(model_ids)}):[/cyan]
{model_list}

[cyan]Type:[/cyan] {model_type.value.upper()}
[cyan]Suite:[/cyan] {suite_type.value.upper()}
[cyan]Test Prompts:[/cyan] {len(prompts)}{suite_summary}
[cyan]Mode:[/cyan] {execution_mode.value.upper()}"""

        tui.show_panel(summary, title="Summary", border_style="green")

        confirm_choice = tui.prompt("\nStart benchmark? [Y/n]:", style="green").strip().lower()
        if confirm_choice == "n":
            logger.info("User cancelled benchmark configuration")
            return None

        return config

    @staticmethod
    def run_benchmark(config: BenchmarkConfig, session_manager: SessionManager) -> None:
        """
        Run benchmark with progress display.

        Args:
            config: BenchmarkConfig
            session_manager: SessionManager for model execution
        """
        runner = BenchmarkRunner(session_manager=session_manager)

        def progress_callback(message: str):
            tui.console.print(f"[dim]{message}[/dim]")

        tui.clear_screen()
        tui.console.print(f"\n[bold cyan]Starting {config.suite_type.value.upper()} Benchmark...[/bold cyan]\n")

        try:
            result = runner.run(config, progress_callback=progress_callback)

            tui.clear_screen()
            tui.console.print(f"\n[bold green]✓ Benchmark Complete![/bold green]")
            tui.console.print(f"[cyan]Status:[/cyan] {result.status.value}")
            tui.console.print(f"[cyan]Duration:[/cyan] {result.duration_seconds:.2f}s")
            tui.console.print(f"[cyan]Successful runs:[/cyan] {result.successful_runs}/{result.total_runs}")

            # Ask if user wants to see detailed results
            show_details = tui.prompt("\nShow detailed results? [Y/n]:", style="green").strip().lower() != "n"

            if show_details:
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
