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
        """Display system specs in a panel.

        Args:
            specs: SystemSpecs to display
        """
        content = f"""[bold cyan]Platform:[/bold cyan] {specs.platform} ({specs.architecture})
[bold cyan]Category:[/bold cyan] {specs.device_category}

[bold green]CPU:[/bold green]
  • Physical Cores: {specs.cpu_cores_physical}
  • Logical Cores: {specs.cpu_cores_logical}

[bold green]Memory:[/bold green]
  • Total RAM: {specs.total_ram_gb:.1f} GB
  • Available RAM: {specs.available_ram_gb:.1f} GB
  • Recommended Model Size: {specs.recommended_model_size_gb:.1f} GB

[bold green]GPU:[/bold green]
  • Available: {"Yes" if specs.gpu.available else "No"}
  • Type: {specs.gpu.gpu_type.upper()}"""

        if specs.gpu.device_name:
            content += f"\n  • Device: {specs.gpu.device_name}"
        if specs.gpu.memory_gb:
            content += f"\n  • Memory: {specs.gpu.memory_gb:.1f} GB"

        tui.show_panel(
            content,
            title="System Specifications",
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

        # Create provider table
        table = Table(title="Select Provider", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Provider", style="cyan", width=20)
        table.add_column("Status", justify="center", width=10)

        for idx, provider in enumerate(providers, 1):
            table.add_row(
                str(idx),
                provider.upper(),
                "[green]✓ Available[/green]"
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
        choice = tui.prompt(f"Select provider [1-{len(providers)}/h/q]:", style="cyan")

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
            if 1 <= idx <= len(providers):
                return providers[idx - 1]
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
