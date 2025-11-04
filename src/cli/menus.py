"""Menu classes for dynamic TUI interface."""

from typing import Optional

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models.model import ModelInfo
from ..models.system import SystemSpecs
from .tui_manager import tui


class SystemSpecsScreen:
    """Display system specifications screen."""

    @staticmethod
    def show(specs: SystemSpecs) -> None:
        """Display comprehensive system specs in a panel.

        Args:
            specs: SystemSpecs to display
        """
        from ..system_specs import SystemSpecsDetector, SystemMonitor

        # Clear screen first to prevent scrollback visibility
        tui.clear_screen()

        # Get comprehensive specs
        detector = SystemSpecsDetector()
        full_specs = detector.detect_all()

        # Get real-time snapshot for current metrics
        monitor = SystemMonitor()
        snapshot = monitor.get_snapshot()

        # Build content string
        content = f"""[bold cyan]Platform:[/bold cyan] {specs.platform} ({specs.architecture})
[bold cyan]Category:[/bold cyan] {specs.device_category}
[bold cyan]Python:[/bold cyan] {full_specs.get('python_version', 'unknown')}

[bold green]CPU:[/bold green]
  • Physical Cores: {specs.cpu_cores_physical}
  • Logical Cores: {specs.cpu_cores_logical}"""

        # Add CPU frequency if available
        if 'cpu' in full_specs and 'frequency' in full_specs['cpu']:
            freq = full_specs['cpu']['frequency']
            content += f"\n  • Current Frequency: {freq.get('current_ghz', 0):.2f} GHz"
            if freq.get('min_ghz') and freq.get('max_ghz'):
                content += f"\n  • Frequency Range: {freq.get('min_ghz', 0):.2f} - {freq.get('max_ghz', 0):.2f} GHz"

        # Add current CPU usage
        if snapshot.cpu:
            content += f"\n  • Current Usage: {snapshot.cpu.usage_percent:.1f}%"
            if snapshot.cpu.load_avg_1min is not None:
                content += f"\n  • Load Average: {snapshot.cpu.load_avg_1min:.2f} (1m)"

        content += f"""

[bold green]Memory:[/bold green]
  • Total RAM: {specs.total_ram_gb:.1f} GB
  • Available RAM: {specs.available_ram_gb:.1f} GB"""

        # Add current memory usage from snapshot
        if snapshot.memory:
            content += f"\n  • Used (Processes): {snapshot.memory.used_ram_gb:.1f} GB"
            if snapshot.memory.cached_gb:
                content += f"\n  • Cached (File System): {snapshot.memory.cached_gb:.1f} GB"
            content += f"\n  • Available (Free): {snapshot.memory.available_ram_gb:.1f} GB"
            content += f"\n  • Usage: {snapshot.memory.percent_used:.1f}%"
            if snapshot.memory.total_swap_gb > 0:
                content += f"\n  • Swap: {snapshot.memory.used_swap_gb:.1f} GB / {snapshot.memory.total_swap_gb:.1f} GB"

        content += f"\n  • Recommended Model Size: {specs.recommended_model_size_gb:.1f} GB"

        # Storage information
        if 'storage' in full_specs and full_specs['storage'].get('devices'):
            content += f"\n\n[bold green]Storage:[/bold green]"

            # Filter out system volumes (macOS has confusing virtual volumes)
            # Keep only user-relevant storage mounts
            system_mounts = {'/System/Volumes/VM', '/System/Volumes/Preboot', '/System/Volumes/Update',
                           '/System/Volumes/Data', '/private', '/run', '/dev', '/snap', '/boot'}

            shown_count = 0
            for dev in full_specs['storage']['devices']:
                mount = dev['mount_point']

                # Skip system volumes (but allow main / mount)
                if mount != '/' and any(mount.startswith(sys_mount) for sys_mount in system_mounts):
                    continue

                usage_pct = dev['percent_used']
                status = "🟢" if usage_pct < 70 else "🟡" if usage_pct < 90 else "🔴"
                used_gb = dev['total_gb'] - dev['free_gb']
                content += f"\n  • {mount}: {status} {used_gb:.1f} GB used, {dev['free_gb']:.1f} GB free ({usage_pct:.1f}%)"

                shown_count += 1
                if shown_count >= 3:  # Show up to 3 user-relevant devices
                    break

            # Show total I/O if available
            if full_specs['storage'].get('total_read_mb'):
                total_io = full_specs['storage']
                content += f"\n  • Total I/O: {total_io.get('total_read_mb', 0):.1f} MB read, {total_io.get('total_write_mb', 0):.1f} MB written"

        # GPU information
        content += f"\n\n[bold green]GPU:[/bold green]"
        content += f"\n  • Available: {'Yes' if specs.gpu.available else 'No'}"
        content += f"\n  • Type: {specs.gpu.gpu_type.upper()}"

        if specs.gpu.device_name:
            content += f"\n  • Device: {specs.gpu.device_name}"

        # Get GPU memory from full_specs (more complete)
        gpu_total_gb = specs.gpu.memory_gb
        gpu_used_gb = 0.0

        # Add detailed GPU info from full_specs
        if 'gpus' in full_specs and full_specs['gpus']:
            for gpu_info in full_specs['gpus']:
                gpu_total_gb = gpu_info.get('total_memory_gb', specs.gpu.memory_gb)
                gpu_used_gb = gpu_info.get('used_memory_gb', 0.0)

                if gpu_info.get('compute_capability'):
                    content += f"\n  • Compute Capability: {gpu_info['compute_capability']}"

                # Show memory with note for Apple Silicon
                if gpu_total_gb > 0:
                    usage_pct = (gpu_used_gb / gpu_total_gb * 100)
                    content += f"\n  • Total Memory: {gpu_total_gb:.1f} GB"
                    content += f"\n  • Used Memory: {gpu_used_gb:.1f} GB ({usage_pct:.1f}%)"

                    # Add note for Apple Silicon unified memory
                    if specs.gpu.gpu_type.lower() == "mps":
                        content += "\n  • [dim](Unified memory, estimate based on system RAM)[/dim]"

        elif gpu_total_gb > 0:
            # Fallback to specs data if full_specs not available
            content += f"\n  • Total Memory: {gpu_total_gb:.1f} GB"
            if gpu_used_gb > 0:
                usage_pct = (gpu_used_gb / gpu_total_gb * 100)
                content += f"\n  • Used Memory: {gpu_used_gb:.1f} GB ({usage_pct:.1f}%)"

        # Add current GPU metrics from snapshot
        if snapshot.gpus:
            for gpu in snapshot.gpus:
                if gpu.gpu_utilization_percent > 0:
                    content += f"\n  • Current Utilization: {gpu.gpu_utilization_percent:.1f}%"
                if gpu.power_draw_watts:
                    content += f"\n  • Power Draw: {gpu.power_draw_watts:.1f}W"
                    if gpu.power_limit_watts:
                        content += f" / {gpu.power_limit_watts:.1f}W"

        # Temperature information
        if 'temperature' in full_specs and full_specs['temperature']:
            temp_data = full_specs['temperature']
            content += f"\n\n[bold yellow]Temperatures:[/bold yellow]"

            if 'cpu_celsius' in temp_data:
                cpu_temp = temp_data['cpu_celsius']
                temp_status = "🟢" if cpu_temp < 70 else "🟡" if cpu_temp < 85 else "🔴"
                content += f"\n  • CPU: {temp_status} {cpu_temp:.1f}°C"

            if 'cpu_package_celsius' in temp_data and temp_data['cpu_package_celsius'] != temp_data.get('cpu_celsius'):
                pkg_temp = temp_data['cpu_package_celsius']
                content += f"\n  • CPU Package: {pkg_temp:.1f}°C"

            if 'gpu_celsius' in temp_data and temp_data['gpu_celsius']:
                for gpu_id, gpu_temp in temp_data['gpu_celsius'].items():
                    temp_status = "🟢" if gpu_temp < 75 else "🟡" if gpu_temp < 85 else "🔴"
                    content += f"\n  • GPU {gpu_id}: {temp_status} {gpu_temp:.1f}°C"

            if temp_data.get('throttling'):
                content += f"\n  • [red]⚠️ THERMAL THROTTLING DETECTED[/red]"

        # Battery information (if laptop)
        if 'battery' in full_specs and full_specs['battery']:
            battery = full_specs['battery']
            content += f"\n\n[bold magenta]Battery:[/bold magenta]"
            percent = battery.get('percent', 0)
            battery_status = "🔋" if battery.get('plugged') else "🪫"
            content += f"\n  • Level: {battery_status} {percent:.0f}%"
            if battery.get('plugged'):
                content += " (Charging)"
            else:
                if battery.get('seconds_left') and battery['seconds_left'] > 0:
                    hours = battery['seconds_left'] // 3600
                    mins = (battery['seconds_left'] % 3600) // 60
                    content += f"\n  • Time Remaining: {hours}h {mins}m"

        # Network I/O (if tracking enabled)
        if snapshot.network:
            net = snapshot.network
            content += f"\n\n[bold cyan]Network I/O:[/bold cyan]"
            content += f"\n  • Sent: {net.bytes_sent / (1024**2):.1f} MB"
            content += f"\n  • Received: {net.bytes_recv / (1024**2):.1f} MB"
            if net.errors_in + net.errors_out > 0:
                content += f"\n  • Errors: {net.errors_in + net.errors_out}"

        # System uptime
        if snapshot.uptime_seconds:
            hours = int(snapshot.uptime_seconds // 3600)
            mins = int((snapshot.uptime_seconds % 3600) // 60)
            content += f"\n\n[bold dim]System Uptime:[/bold dim] {hours}h {mins}m"

        tui.show_panel(
            content,
            title="Comprehensive System Specifications",
            border_style="blue"
        )

        # Wait for user to continue
        tui.prompt("\n[dim]Press Enter to continue...[/dim]", style="dim")


class ProviderSelectionMenu:
    """Provider selection menu."""

    @staticmethod
    def show(providers: list[str]) -> Optional[str]:
        """Display provider selection menu.

        Args:
            providers: List of enabled provider names

        Returns:
            Selected provider name or None if quit
        """
        from ..cli.text_input import professional_prompt

        tui.clear_screen()

        # Detect system capabilities
        import platform
        system = platform.system()
        machine = platform.machine()
        is_macos = system == "Darwin"
        is_apple_silicon = machine in ["arm64", "aarch64"] and is_macos

        # Check for optional packages
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

        # Show ALL possible providers (not just registered ones)
        from ..models.endpoints import ProviderType
        all_providers = [p.value for p in ProviderType if p != ProviderType.LM_STUDIO]  # Exclude unimplemented

        # Build options list for arrow-key selection
        arrow_options = []

        for provider in all_providers:
            # Check if provider is registered (successfully initialized)
            is_registered = provider in providers
            is_available = is_registered
            reason = ""

            # Platform-specific checks
            if provider == "mlx":
                if not is_macos:
                    is_available = False
                    reason = "macOS only"
                elif not is_apple_silicon:
                    is_available = False
                    reason = "Apple Silicon only"
                elif not has_mlx:
                    is_available = False
                    reason = "pip install mlx mlx-lm mlx-vlm"
                elif not is_registered:
                    is_available = False
                    reason = "Failed to initialize"

            elif provider == "openvino":
                if not has_openvino:
                    is_available = False
                    reason = "pip install openvino optimum[openvino]"
                elif not is_registered:
                    is_available = False
                    reason = "Failed to initialize"

            elif not is_registered:
                # Other providers failed to register
                is_available = False
                reason = "Not available"

            # Add to options (both available and unavailable for visibility)
            if is_available:
                arrow_options.append((
                    provider,  # value to return
                    f"[green]{provider.upper()}[/green]",  # label
                    "✓ Available"  # description
                ))
            else:
                arrow_options.append((
                    f"__unavailable_{provider}",  # special marker
                    f"[dim]{provider.upper()}[/dim]",  # dimmed label
                    f"[dim red]✗ {reason}[/dim red]"  # error description
                ))

        # Add special navigation options
        arrow_options.append(("HOME", "[cyan]Main Menu (Home)[/cyan]", "Return to operation mode selection"))
        arrow_options.append(("QUIT", "[red]Quit Application[/red]", "Exit the application"))

        # Show persistent Ekam-CLI header
        tui.clear_screen()
        tui.console.print("[dim]Select a provider to begin model inference[/dim]")
        tui.console.print()

        # Use arrow-key selection
        choice = professional_prompt.get_arrow_selection(
            options=arrow_options,
            title="Select Provider",
            instructions="Use ↑/↓ arrows to navigate, Enter to select, q to quit, /background to monitor jobs"
        )

        # Handle /background command (not caught by arrow selector)
        # For now, arrow selector doesn't support custom commands, so user would need to press 'q' and use command separately

        # Check if user selected unavailable provider
        if choice and choice.startswith("__unavailable_"):
            provider_name = choice.replace("__unavailable_", "")
            tui.show_error(f"Provider {provider_name.upper()} is not available on this system.")
            tui.prompt("Press Enter to continue...", style="dim")
            return ProviderSelectionMenu.show(providers)  # Re-show menu

        return choice


class ModelSelectionMenu:
    """Model selection menu with categorization by type."""

    @staticmethod
    def _apply_filter(models: list[ModelInfo], filter_type: str) -> list[ModelInfo]:
        """Apply filter to model list.

        Args:
            models: Full model list
            filter_type: Filter type (recommended, vlm, llm, embedding, size_small, size_medium, size_large, all)

        Returns:
            Filtered model list
        """
        if filter_type == "recommended":
            return [m for m in models if m.compatibility == "perfect_fit"]
        elif filter_type == "vlm":
            return [m for m in models if m.model_type == ModelType.VLM]
        elif filter_type == "llm":
            return [m for m in models if m.model_type == ModelType.LLM]
        elif filter_type == "embedding":
            return [m for m in models if m.model_type == ModelType.EMBEDDING]
        elif filter_type == "size_small":
            return [m for m in models if m.size_gb <= 5.0]
        elif filter_type == "size_medium":
            return [m for m in models if 5.0 < m.size_gb <= 10.0]
        elif filter_type == "size_large":
            return [m for m in models if m.size_gb > 10.0]
        elif filter_type == "search":
            return models  # Will use incremental search
        elif filter_type == "all":
            return models
        else:
            return models

    @staticmethod
    def _show_filter_menu(models: list[ModelInfo], provider: str) -> Optional[str]:
        """Show filter selection menu when there are many models.

        Args:
            models: Full model list
            provider: Provider name

        Returns:
            Filter type or None to cancel
        """
        from ..cli.text_input import professional_prompt

        # Count models by category
        recommended_count = len([m for m in models if m.compatibility == "perfect_fit"])
        vlm_count = len([m for m in models if m.model_type == ModelType.VLM])
        llm_count = len([m for m in models if m.model_type == ModelType.LLM])
        embedding_count = len([m for m in models if m.model_type == ModelType.EMBEDDING])
        small_count = len([m for m in models if m.size_gb <= 5.0])
        medium_count = len([m for m in models if 5.0 < m.size_gb <= 10.0])
        large_count = len([m for m in models if m.size_gb > 10.0])

        tui.clear_screen()
        tui.show_panel(
            f"[bold cyan]{provider.upper()} - Filter Models[/bold cyan]\n\n"
            f"Found [bold]{len(models)}[/bold] models. Choose how to filter:",
            title=f"{provider.upper()} Models",
            border_style="cyan"
        )

        filter_options = []

        if recommended_count > 0:
            filter_options.append((
                "recommended",
                "[green]Recommended for your system[/green]",
                f"{recommended_count} models - OPTIMAL fit for your hardware"
            ))

        if vlm_count > 0:
            filter_options.append((
                "vlm",
                "[magenta]VLMs only[/magenta]",
                f"{vlm_count} models - Vision-Language Models"
            ))

        if llm_count > 0:
            filter_options.append((
                "llm",
                "[cyan]LLMs only[/cyan]",
                f"{llm_count} models - Large Language Models"
            ))

        if embedding_count > 0:
            filter_options.append((
                "embedding",
                "[yellow]Embeddings only[/yellow]",
                f"{embedding_count} models - Embedding Models"
            ))

        filter_options.extend([
            ("size_small", "[green]Small models (0-5GB)[/green]", f"{small_count} models"),
            ("size_medium", "[yellow]Medium models (5-10GB)[/yellow]", f"{medium_count} models"),
            ("size_large", "[red]Large models (10GB+)[/red]", f"{large_count} models"),
        ])

        # Convert to numbered menu
        tui.clear_screen()
        tui.console.print(f"[bold]Found {len(models)} models.[/bold] Choose a filter:")
        tui.console.print()

        # Create numbered mapping
        filter_map = {}
        idx = 1

        for value, label, description in filter_options:
            if value == "BACK":
                continue  # Handle separately
            tui.console.print(f"  [{idx}] {label}")
            tui.console.print(f"      [dim]{description}[/dim]")
            tui.console.print()
            filter_map[idx] = value
            idx += 1

        tui.console.print(f"  [b] Back to provider selection")
        tui.console.print(f"  [q] Quit")
        tui.console.print()

        # Get user choice
        while True:
            choice = tui.prompt(
                f"Select filter [1-{len(filter_map)}] or 'b'/'q':",
                style="cyan"
            ).strip().lower()

            if choice == 'b':
                return "BACK"
            elif choice == 'q':
                return None

            try:
                choice_num = int(choice)
                if choice_num in filter_map:
                    return filter_map[choice_num]
                tui.show_error(f"Please enter a number between 1 and {len(filter_map)}")
            except ValueError:
                tui.show_error("Please enter a valid number or 'b'/'q'")

    @staticmethod
    def _show_with_search(models: list[ModelInfo], provider: str) -> Optional[ModelInfo]:
        """Show model selection with incremental search.

        Args:
            models: Full model list
            provider: Provider name

        Returns:
            Selected ModelInfo or navigation command
        """
        from ..cli.text_input import professional_prompt

        # Prepare search items
        search_items = []
        for model in models:
            # Determine color based on compatibility
            if model.compatibility == "perfect_fit":
                color = "green"
                fit_text = "OPTIMAL"
            elif model.compatibility == "tight_fit":
                color = "yellow"
                fit_text = "TIGHT"
            else:
                color = "red"
                fit_text = "LARGE"

            # Build metadata string
            params = f"{model.params_billions:.1f}B" if model.params_billions and model.params_billions > 0 else "?"
            quant = model.quantization if model.quantization and model.quantization != "unknown" else "?"
            size = f"{model.size_gb:.1f}GB"

            label = f"[{color}]{model.name}[/{color}]"
            description = f"[{params}] [{quant}] {size} | {fit_text}"

            search_items.append((model, label, description))

        # Show incremental search
        choice = professional_prompt.get_incremental_search_selection(
            items=search_items,
            title=f"{provider.upper()} - Search Models (Type to filter)",
            instructions="Type to search, ↑/↓ to navigate, Enter to select, Esc to cancel",
            max_display=15
        )

        # Handle special navigation
        if choice is None:
            return "BACK"

        return choice

    @staticmethod
    def show(models: list[ModelInfo], provider: str) -> Optional[ModelInfo]:
        """Display model selection menu with smart filtering.

        Args:
            models: List of available models
            provider: Provider name

        Returns:
            Selected ModelInfo or None
        """
        # If more than 20 models, show filter menu first
        if len(models) > 20:
            filter_choice = ModelSelectionMenu._show_filter_menu(models, provider)

            if filter_choice is None or filter_choice == "BACK":
                return "BACK"

            # Apply filter
            filtered_models = ModelSelectionMenu._apply_filter(models, filter_choice)

            # If filter resulted in 0 models, show error and go back
            if not filtered_models:
                tui.show_error(
                    f"No models match the selected filter.\n\n"
                    f"Try a different filter or select 'Show ALL models'."
                )
                tui.prompt("Press Enter to continue...", style="dim")
                return ModelSelectionMenu.show(models, provider)  # Retry with filter menu

            # If search filter, use incremental search
            if filter_choice == "search":
                return ModelSelectionMenu._show_with_search(models, provider)
        else:
            filtered_models = models

        tui.clear_screen()

        # Group models by type
        categorized = {}
        for model in filtered_models:
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

        # Build numbered model list and display
        model_list = []
        idx = 1

        # Main container
        width, _ = tui.get_terminal_size()

        for category in category_order:
            if category not in categorized:
                continue  # Skip categories with no models

            category_meta = category_info.get(category, {"name": category, "color": "white"})

            # Create table for this category with new metadata columns
            table = Table(
                show_header=True,
                header_style="bold white",
                expand=True,
                border_style=category_meta["color"],
            )

            table.add_column("#", style="dim", width=3, justify="right")
            table.add_column("Model Name", style="white", no_wrap=False, min_width=20)
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
                    # Shorten common quantization names
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

                table.add_row(
                    str(idx),
                    f"[{name_style}]{model.name}[/{name_style}]",
                    params_str,
                    quant_str,
                    ram_str,
                    size_str,
                    status_text,
                )

                model_list.append(model)
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

        # Show legend and action menu
        legend = Panel(
            "[green]OPTIMAL[/green] = Recommended size  |  "
            "[yellow]TIGHT[/yellow] = May impact performance  |  "
            "[red]LARGE[/red] = Risk of memory errors",
            title=f"[bold]{provider.upper()} - Select Model[/bold]",
            border_style="white",
            expand=True,
        )
        tui.console.print(legend)
        tui.console.print()

        # Show action options
        tui.console.print("[bold]Actions:[/bold]")
        tui.console.print("  [i] Install New Model")
        tui.console.print("  [d] Delete a Model")
        tui.console.print("  [r] Refresh Model Registry")
        tui.console.print("  [f] Change Filter")
        tui.console.print("  [b] Back to Provider Selection")
        tui.console.print("  [h] Main Menu (Home)")
        tui.console.print("  [q] Quit")
        tui.console.print()

        # Get user choice
        while True:
            if len(model_list) > 0:
                choice = tui.prompt(
                    f"Select model [1-{len(model_list)}] or action [i/d/r/f/b/h/q]:",
                    style="cyan"
                ).strip().lower()
            else:
                choice = tui.prompt(
                    f"No models to select. Choose action [i/r/f/b/h/q]:",
                    style="yellow"
                ).strip().lower()

            # Handle action commands
            if choice == 'i':
                return "INSTALL"
            elif choice == 'd':
                return "DELETE"
            elif choice == 'r':
                return "REFRESH"
            elif choice == 'f':
                # Return to filter menu
                return ModelSelectionMenu.show(models, provider)
            elif choice == 'b':
                return "BACK"
            elif choice == 'h':
                return "HOME"
            elif choice == 'q':
                return "QUIT"

            # Handle model number selection
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(model_list):
                    selected_model = model_list[choice_num - 1]
                    # Show selected model confirmation
                    tui.console.print(f"\n[green]✓ Selected:[/green] {selected_model.name}")
                    tui.console.print(f"[dim]  Type: {selected_model.model_type}[/dim]")
                    tui.console.print(f"[dim]  Size: {selected_model.size_gb:.1f}GB[/dim]")
                    tui.console.print(f"[dim]  ID: {selected_model.model_id}[/dim]\n")
                    return selected_model
                else:
                    tui.show_error(f"Please enter a number between 1 and {len(model_list)}")
            except ValueError:
                tui.show_error("Invalid input. Enter a model number or action letter.")


class LoadingScreen:
    """Loading screen with spinner."""

    @staticmethod
    def show(message: str = "Loading...") -> None:
        """Display loading message.

        Args:
            message: Loading message
        """
        tui.clear_screen()

        width, height = tui.get_terminal_size()

        # Center the loading message
        padding = "\n" * (height // 3)
        content = f"{padding}[bold cyan]{message}[/bold cyan]\n\n[dim]Please wait...[/dim]"

        panel = Panel(
            content,
            border_style="cyan",
            expand=True
        )

        tui.console.print(panel)


class EndpointMenu:
    """Endpoint selection menu for VLM/LLM models."""

    @staticmethod
    def show(model_name: str, model_type: str) -> Optional[str]:
        """Display endpoint selection menu.

        Args:
            model_name: Name of loaded model
            model_type: Type of model (vlm or llm)

        Returns:
            Selected endpoint name or None if quit
        """
        from ..cli.text_input import professional_prompt

        tui.clear_screen()

        # Available endpoints based on model type
        if model_type.lower() == "vlm":
            endpoints = [
                ("qa", "[magenta]Question Answering[/magenta]", "Ask questions about an image"),
                ("caption", "[cyan]Image Captioning[/cyan]", "Generate captions for an image"),
                ("detect", "[yellow]Object Detection[/yellow]", "Detect objects in an image"),
                ("point", "[blue]Object Pointing[/blue]", "Find object locations in an image"),
                ("chat", "[green]Text Chat[/green]", "Interactive text conversation (no image)"),
            ]
        else:  # LLM
            endpoints = [
                ("chat", "[green]Text Chat[/green]", "Interactive text conversation"),
            ]

        # Build arrow-key selection options
        arrow_options = []

        # Add endpoints
        for endpoint_key, endpoint_label, description in endpoints:
            arrow_options.append((endpoint_key, endpoint_label, description))

        # Add separator and additional options
        arrow_options.append(("STATS", "[blue]View Statistics[/blue]", "View model performance statistics"))
        arrow_options.append(("SWITCH", "[yellow]Switch Model[/yellow]", "Select a different model"))
        arrow_options.append(("BACK", "[cyan]Back to Provider Selection[/cyan]", "Return to provider selection"))
        arrow_options.append(("HOME", "[cyan]Main Menu (Home)[/cyan]", "Return to operation mode selection"))
        arrow_options.append(("QUIT", "[red]Quit Application[/red]", "Exit the application"))

        # Show persistent Ekam-CLI header
        tui.clear_screen()
        tui.console.print(f"[bold]Model:[/bold] {model_name}")
        tui.console.print(f"[dim]Type: {model_type.upper()}[/dim]")
        tui.console.print()

        # Use arrow-key selection
        choice = professional_prompt.get_arrow_selection(
            options=arrow_options,
            title="Select Endpoint",
            instructions="Use ↑/↓ arrows to navigate, Enter to select, q to quit"
        )

        return choice


class WelcomeScreen:
    """Welcome screen for application."""

    @staticmethod
    def show() -> None:
        """Display welcome screen."""
        tui.clear_screen()

        # Show persistent Ekam-CLI header

        content = f"""[green]Welcome to Ekam-CLI![/green]

[bold]What is Ekam?[/bold]
Ekam (एकम्) means "Unity" in Sanskrit - representing the unified interface
for all your AI model testing needs.

[bold cyan]Features:[/bold cyan]
  • Test multiple VLM/LLM providers (Ollama, HuggingFace, GGUF, Quantized)
  • Run vision and text inference with intelligent resource management
  • Benchmark models across multiple suites
  • Quantize models for edge devices
  • Monitor performance and generate detailed reports

[dim]Initializing...[/dim]
"""

        panel = Panel(
            content,
            border_style="cyan",
            expand=True
        )

        tui.console.print(panel)
