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
        available_providers = {}  # Track which are actually selectable

        # Create provider table with compatibility info
        table = Table(title="Select Provider", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Provider", style="cyan", width=20)
        table.add_column("Status", justify="center", width=35)

        for idx, provider in enumerate(all_providers, 1):
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

            # Track available providers for selection
            if is_available:
                available_providers[idx] = provider

            # Render row
            if is_available:
                table.add_row(
                    str(idx),
                    provider.upper(),
                    "[green]✓ Available[/green]"
                )
            else:
                table.add_row(
                    f"[dim]{idx}[/dim]",
                    f"[dim]{provider.upper()}[/dim]",
                    f"[dim red]✗ {reason}[/dim red]"
                )

        # Show menu in panel
        width, _ = tui.get_terminal_size()
        panel = Panel(
            table,
            title="[bold]VLM/LLM CLI - Provider Selection[/bold]",
            border_style="cyan",
            expand=True
        )

        tui.console.print(panel)
        tui.console.print()

        # Show navigation
        tui.console.print("[bold]Navigation:[/bold]")
        tui.console.print("  [cyan bold]h[/cyan bold] = Main Menu (Home)")
        tui.console.print("  [cyan bold]q[/cyan bold] = Quit application")
        tui.console.print()

        # Show special commands
        tui.console.print("[bold]Commands:[/bold]")
        tui.console.print("  [yellow]/background[/yellow] = Monitor quantization jobs")
        tui.console.print()

        # Get user selection
        choice = tui.prompt(f"Select provider [1-{len(all_providers)}/h/q]:", style="cyan")

        # Check for /background command
        if choice.strip().lower().startswith("/background") or choice.strip().lower() == "/bg":
            from ..core import app
            if app.handle_background_command():
                return ProviderSelectionMenu.show(providers)  # Refresh menu

        if choice.lower() == 'q':
            return "QUIT"
        if choice.lower() == 'h':
            return "HOME"

        try:
            idx = int(choice)
            # Check if selection is valid and available
            if idx in available_providers:
                return available_providers[idx]
            elif 1 <= idx <= len(all_providers):
                # User selected an unavailable provider
                tui.show_error(f"Provider {all_providers[idx-1].upper()} is not available on this system.")
                tui.prompt("Press Enter to continue...", style="dim")
                return ProviderSelectionMenu.show(providers)  # Re-show menu
        except ValueError:
            pass

        tui.show_error("Invalid selection. Please try again.")
        tui.prompt("Press Enter to continue...", style="dim")
        return ProviderSelectionMenu.show(providers)


class ModelSelectionMenu:
    """Model selection menu with categorization by type."""

    @staticmethod
    def show(models: list[ModelInfo], provider: str) -> Optional[ModelInfo]:
        """Display categorized model selection menu.

        Args:
            models: List of available models
            provider: Provider name

        Returns:
            Selected ModelInfo or None
        """
        tui.clear_screen()

        # Group models by type
        categorized = {}
        for model in models:
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

        # Show legend with install option
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

        # Show additional options
        tui.console.print("[bold]Options:[/bold]")
        tui.console.print("  [cyan bold]i[/cyan bold] = Install New Model")
        tui.console.print("  [cyan bold]d[/cyan bold] = Delete a Model")
        tui.console.print("  [cyan bold]b[/cyan bold] = Back to Provider Selection")
        tui.console.print("  [cyan bold]bb[/cyan bold] = Back 2 Levels")
        tui.console.print("  [cyan bold]h[/cyan bold] = Main Menu (Home)")
        tui.console.print("  [cyan bold]q[/cyan bold] = Quit Application")
        tui.console.print()

        # Show special commands
        tui.console.print("[bold]Commands:[/bold]")
        tui.console.print("  [yellow]/background[/yellow] = Monitor quantization jobs")
        tui.console.print()

        # Get user selection
        choice = tui.prompt(
            f"Select model [1-{len(model_list)}] or option [i/d/b/bb/h/q]:",
            style="cyan"
        )

        # Check for /background command
        if choice.strip().lower().startswith("/background") or choice.strip().lower() == "/bg":
            from ..core import app
            if app.handle_background_command():
                return ModelSelectionMenu.show(models, provider)  # Refresh menu

        if choice.lower() == 'q':
            return "QUIT"
        if choice.lower() == 'bb':
            return "BACK2"
        if choice.lower() == 'b':
            return "BACK"
        if choice.lower() == 'h':
            return "HOME"
        if choice.lower() == 'i':
            return "INSTALL"
        if choice.lower() == 'd':
            return "DELETE"

        try:
            idx = int(choice)
            if 1 <= idx <= len(model_list):
                return model_list[idx - 1]
        except ValueError:
            pass

        tui.show_error("Invalid selection. Please try again.")
        tui.prompt("Press Enter to continue...", style="dim")
        return ModelSelectionMenu.show(models, provider)


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
        tui.clear_screen()

        # Available endpoints based on model type
        if model_type.lower() == "vlm":
            endpoints = [
                ("qa", "Question Answering", "Ask questions about an image"),
                ("caption", "Image Captioning", "Generate captions for an image"),
                ("detect", "Object Detection", "Detect objects in an image"),
                ("point", "Object Pointing", "Find object locations in an image"),
                ("chat", "Text Chat", "Interactive text conversation (no image)"),
            ]
        else:  # LLM
            endpoints = [
                ("chat", "Text Chat", "Interactive text conversation"),
            ]

        # Create endpoint table
        table = Table(
            title=f"Available Endpoints - {model_name}",
            show_header=True,
            header_style="bold magenta",
            expand=True
        )

        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Endpoint", style="cyan", width=20)
        table.add_column("Description", style="white", min_width=30)

        for idx, (endpoint, name, description) in enumerate(endpoints, 1):
            table.add_row(
                str(idx),
                name,
                description
            )

        # Show menu in panel
        panel = Panel(
            table,
            title=f"[bold]Endpoint Selection - {model_name}[/bold]",
            border_style="cyan",
            expand=True
        )

        tui.console.print(panel)
        tui.console.print()

        # Show additional options with clear formatting
        tui.console.print("[bold]Additional Options:[/bold]")
        tui.console.print("  [cyan bold]s[/cyan bold] = View Statistics")
        tui.console.print("  [cyan bold]m[/cyan bold] = Switch Model")
        tui.console.print("  [cyan bold]b[/cyan bold] = Back to Provider Selection")
        tui.console.print("  [cyan bold]bb[/cyan bold] = Back 2 Levels (to Model Selection)")
        tui.console.print("  [cyan bold]h[/cyan bold] = Main Menu (Home)")
        tui.console.print("  [cyan bold]q[/cyan bold] = Quit Application")
        tui.console.print()

        # Show special commands
        tui.console.print("[bold]Commands:[/bold]")
        tui.console.print("  [yellow]/background[/yellow] = Monitor quantization jobs")
        tui.console.print()

        # Get user selection
        choice = tui.prompt(
            f"[bold]Choose:[/bold] Endpoint number [1-{len(endpoints)}] or option [s/m/b/bb/h/q]:",
            style="cyan"
        )

        # Check for /background command
        if choice.strip().lower().startswith("/background") or choice.strip().lower() == "/bg":
            from ..core import app
            if app.handle_background_command():
                return EndpointMenu.show(model_name, model_type)  # Refresh menu

        if choice.lower() == 'q':
            return "QUIT"
        if choice.lower() == 'bb':
            return "BACK2"
        if choice.lower() == 'b':
            return "BACK"
        if choice.lower() == 's':
            return "STATS"
        if choice.lower() == 'm':
            return "SWITCH"
        if choice.lower() == 'h':
            return "HOME"

        try:
            idx = int(choice)
            if 1 <= idx <= len(endpoints):
                return endpoints[idx - 1][0]  # Return endpoint key
        except ValueError:
            pass

        tui.show_error("Invalid selection. Please try again.")
        tui.prompt("Press Enter to continue...", style="dim")
        return EndpointMenu.show(model_name, model_type)


class WelcomeScreen:
    """Welcome screen for application."""

    @staticmethod
    def show() -> None:
        """Display welcome screen."""
        tui.clear_screen()

        title = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              VLM/LLM CLI Testing Tool                        ║
║         High-Performance Multi-Provider Interface            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

        content = f"""[bold cyan]{title}[/bold cyan]

[green]Welcome to the VLM/LLM CLI![/green]

This tool allows you to:
  • Test multiple VLM/LLM providers (Ollama, HuggingFace, GGUF)
  • Run vision and text inference with resource management
  • Monitor performance and save results

[dim]Initializing...[/dim]
"""

        panel = Panel(
            content,
            border_style="cyan",
            expand=True
        )

        tui.console.print(panel)
