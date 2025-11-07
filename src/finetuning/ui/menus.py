"""Menu systems for finetuning pipeline navigation."""

from typing import Any, Callable, Optional

from loguru import logger

from ...cli.tui_manager import tui
from ..config import FinetuneMethod
from .information import MethodInformation


class FinetuneHomeMenu:
    """Home menu for finetuning mode with context-aware navigation."""

    def __init__(self):
        """Initialize home menu."""
        self.current_context = "home"
        self.history = ["home"]
        self.on_navigate: Optional[Callable] = None

    def display(self) -> str:
        """Display finetuning home menu.

        Returns:
            Selected option
        """
        menu_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   FINETUNING MODE - HOME                                    ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Adapt pre-trained LLMs to your custom domain using parameter-efficient    ║
║  finetuning methods. Choose between LoRA, QLoRA, or Full finetuning.       ║
║                                                                              ║
║  [1] Start New Finetuning Run                                              ║
║  [2] Load Previous Configuration                                            ║
║  [3] View Method Information & Comparison                                   ║
║  [4] System Requirements & Prerequisites                                    ║
║  [5] Manage Saved Models & Adapters                                         ║
║  [6] Switch to Inference Pipeline                                           ║
║  [7] Switch to Quantization Pipeline                                        ║
║  [0] Back to Main Menu                                                      ║
║                                                                              ║
║  [i] Information  [?] Help  [q] Quit                                        ║
║                                                                              ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
        tui.console.print(menu_text)
        return tui.prompt(
            "Select option",
            style="cyan",
        )

    def handle_selection(self, choice: str) -> str:
        """Handle menu selection and navigate appropriately.

        Args:
            choice: User's menu selection

        Returns:
            Next navigation target or command
        """
        navigation_map = {
            "1": "new_run",
            "2": "load_config",
            "3": "view_info",
            "4": "requirements",
            "5": "manage_models",
            "6": "inference",
            "7": "quantization",
            "0": "main_menu",
            "i": "show_info",
            "?": "help",
            "q": "quit",
        }

        target = navigation_map.get(choice.lower())

        if target:
            self.current_context = target
            self.history.append(target)
            if self.on_navigate:
                self.on_navigate(target)
            return target
        else:
            tui.show_error("Invalid selection. Please try again.")
            return "home"

    def go_back(self) -> str:
        """Navigate back to previous context.

        Returns:
            Previous context
        """
        if len(self.history) > 1:
            self.history.pop()
            self.current_context = self.history[-1]
            return self.current_context
        return "home"

    def get_context(self) -> dict[str, Any]:
        """Get current navigation context.

        Returns:
            Context dictionary with state info
        """
        return {
            "current": self.current_context,
            "history": self.history,
            "can_go_back": len(self.history) > 1,
        }


class MethodSelectionMenu:
    """Menu for selecting finetuning method with detailed information."""

    def __init__(self):
        """Initialize method selection menu."""
        self.selected_method: Optional[FinetuneMethod] = None
        self.info = MethodInformation()

    def display(self) -> FinetuneMethod:
        """Display method selection with information prompts.

        Returns:
            Selected FinetuneMethod
        """
        while True:
            menu_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   SELECT FINETUNING METHOD                                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Choose the finetuning method based on your hardware and requirements:      ║
║                                                                              ║
║  [1] LoRA - Parameter Efficient (57% memory reduction)                      ║
║      • 0.06% trainable parameters                                            ║
║      • Best: Quick iteration, balanced quality/speed                        ║
║      • Memory: ~13.6GB for 7B model                                          ║
║      • Hardware: RTX 3060 (12GB) minimum                                     ║
║                                                                              ║
║  [2] QLoRA - Quantized LoRA (92% memory reduction!) ⭐ BUDGET FRIENDLY      ║
║      • 4-bit quantization + LoRA                                             ║
║      • Best: Large models on consumer hardware                              ║
║      • Memory: ~6.3GB for 7B model (works with 8GB!)                        ║
║      • Hardware: Any GPU with 8GB VRAM                                       ║
║                                                                              ║
║  [3] FULL - Complete Model Finetuning (Maximum Quality)                     ║
║      • 100% trainable parameters                                             ║
║      • Best: Unlimited data, maximum accuracy needed                        ║
║      • Memory: ~58-60GB for 7B model (!)                                     ║
║      • Hardware: A100 (40GB+) or H100 (80GB+) required                      ║
║                                                                              ║
║  [4] View Detailed Comparison                                               ║
║  [0] Back                                                                    ║
║                                                                              ║
║  Tip: Press [i] after selecting for detailed information                    ║
║                                                                              ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
            tui.console.print(menu_text)
            choice = tui.prompt("Select method", style="cyan")

            if choice == "1":
                self.selected_method = FinetuneMethod.LORA
                self._show_method_info(FinetuneMethod.LORA)
                return FinetuneMethod.LORA

            elif choice == "2":
                self.selected_method = FinetuneMethod.QLORA
                self._show_method_info(FinetuneMethod.QLORA)
                return FinetuneMethod.QLORA

            elif choice == "3":
                self.selected_method = FinetuneMethod.FULL
                self._show_method_info(FinetuneMethod.FULL)
                return FinetuneMethod.FULL

            elif choice == "4":
                tui.show_panel(
                    self.info.get_comparison(),
                    title="Methods Comparison",
                    border_style="cyan",
                )

            elif choice == "0":
                return None

            else:
                tui.show_error("Invalid selection")

    def _show_method_info(self, method: FinetuneMethod) -> None:
        """Show detailed information about selected method.

        Args:
            method: Selected method
        """
        info = self.info.get_method_info(method)
        if not info:
            return

        # Build comprehensive info panel
        info_text = f"""
{info['title']}
{'=' * 70}

DESCRIPTION
{'-' * 70}
{info['description']}

HOW IT WORKS
{'-' * 70}
{info['how_it_works']}

BEST FOR
{'-' * 70}
{info['best_for']}

MEMORY REQUIREMENTS
{'-' * 70}
{info['memory_requirements']}

RECOMMENDED SETTINGS
{'-' * 70}
Dataset Size:        {info['recommendations']['dataset_size']}
Learning Rate:       {info['recommendations']['learning_rate']}
Batch Size:          {info['recommendations']['batch_size']}
Training Epochs:     {info['recommendations']['epochs']}
Minimum Hardware:    {info['recommendations']['hardware']}
Typical Training:    {info['recommendations']['training_time']}

DATA REQUIREMENTS
{'-' * 70}
{info['data_needs']}

QUALITY/SPEED/COST PROFILE
{'-' * 70}
Quality: {info['quality']}
Speed:   {info['speed']}
Cost:    {info['cost']}

TRAINING DURATION ESTIMATES
{'-' * 70}
Small Dataset:    {info['training_duration']['small']}
Medium Dataset:   {info['training_duration']['medium']}
Large Dataset:    {info['training_duration']['large']}
"""

        tui.show_panel(
            info_text,
            title=f"Detailed Information: {method.value.upper()}",
            border_style="green",
            expand=False,
        )


class ConfigurationMenu:
    """Interactive configuration menu for finetuning settings."""

    def __init__(self, method: FinetuneMethod):
        """Initialize configuration menu.

        Args:
            method: Selected finetuning method
        """
        self.method = method

    def configure_interactive(self) -> dict[str, Any]:
        """Interactively configure finetuning settings.

        Returns:
            Configuration dictionary
        """
        config = {}

        # Basic info
        config["run_name"] = tui.prompt(
            "Finetuning run name (e.g., 'medical-domain-v1')",
            style="cyan",
        )

        config["model_id"] = tui.prompt(
            "Model ID from HuggingFace (e.g., 'mistralai/Mistral-7B')",
            style="cyan",
        )

        # Dataset
        config["dataset_path"] = tui.prompt(
            "Path to dataset (JSON/CSV) or HF dataset name",
            style="cyan",
        )

        config["text_column"] = tui.prompt(
            "Column name with text data (default: 'text')",
            style="cyan",
            default="text",
        )

        # Training parameters
        epochs_str = tui.prompt(
            "Number of training epochs (1-10, default: 3):",
            style="cyan",
        ).strip()
        config["num_epochs"] = int(epochs_str) if epochs_str else 3

        batch_str = tui.prompt(
            "Batch size (4-32, default: 8):",
            style="cyan",
        ).strip()
        config["batch_size"] = int(batch_str) if batch_str else 8

        lr_str = tui.prompt(
            "Learning rate (e.g., 1e-4, default: 1e-4):",
            style="cyan",
        ).strip()
        config["learning_rate"] = float(lr_str) if lr_str else 1e-4

        return config


class ProgressMonitor:
    """Live progress monitoring for finetuning execution."""

    def __init__(self):
        """Initialize progress monitor."""
        self.current_step = ""
        self.progress = 0.0
        self.metrics = {}

    def update(self, status: dict[str, Any]) -> None:
        """Update progress display.

        Args:
            status: Status dictionary from pipeline
        """
        self.current_step = status.get("step", "")
        self.progress = status.get("progress", 0.0)
        self.metrics = status.get("metrics", {})

        self._render_progress()

    def _render_progress(self) -> None:
        """Render live progress bar and metrics."""
        progress_bar = self._create_progress_bar()
        metrics_text = self._create_metrics_text()

        display = f"""
{progress_bar}
Current Step: {self.current_step}
{metrics_text}
"""
        tui.console.print(display)

    def _create_progress_bar(self) -> str:
        """Create visual progress bar.

        Returns:
            Formatted progress bar string
        """
        bar_length = 50
        filled = int(bar_length * self.progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        return f"[{bar}] {self.progress:.1f}%"

    def _create_metrics_text(self) -> str:
        """Create formatted metrics display.

        Returns:
            Formatted metrics string
        """
        if not self.metrics:
            return ""

        lines = ["Metrics:"]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class ModelSelectionMenu:
    """Menu for selecting model to finetune from all providers."""

    def __init__(self, model_discovery: "FinetuneModelDiscovery"):
        """Initialize model selection menu.

        Args:
            model_discovery: FinetuneModelDiscovery instance
        """
        self.model_discovery = model_discovery

    def display(self) -> Optional[str]:
        """Display model selection with organized list.

        Returns:
            Selected model ID or None if cancelled
        """
        while True:
            tui.clear_screen()

            # Count models
            counts = self.model_discovery.count_models()

            tui.show_panel(
                f"""[bold cyan]SELECT MODEL FOR FINETUNING[/bold cyan]

Total Models Available: {counts['total']}
  • Language Models (LLM): {counts['llm']}
  • Vision-Language Models (VLM): {counts['vlm']}

[1] Browse all models
[2] Language Models only (LLM)
[3] Vision-Language Models only (VLM)
[4] Search by model ID
[0] Back

[dim]Tip: You can finetune any available model from any provider
(Ollama, HuggingFace, GGUF, Quantized, MLX)[/dim]""",
                title="Model Selection",
                border_style="cyan",
            )

            choice = tui.prompt("Select option [1-4/0]:", style="cyan").strip().lower()

            if choice == "0":
                return None
            elif choice == "1":
                selected = self._show_all_models()
                if selected:
                    return selected
            elif choice == "2":
                selected = self._show_llm_models()
                if selected:
                    return selected
            elif choice == "3":
                selected = self._show_vlm_models()
                if selected:
                    return selected
            elif choice == "4":
                selected = self._search_model()
                if selected:
                    return selected
            else:
                tui.show_error("Invalid selection")
                tui.prompt("Press Enter to continue...", style="dim")

    def _show_all_models(self) -> Optional[str]:
        """Show all models organized by type.

        Returns:
            Selected model ID or None
        """
        tui.clear_screen()
        models_dict = self.model_discovery.discover_all_models()
        all_models = models_dict["llm"] + models_dict["vlm"]

        if not all_models:
            tui.show_error("No models available")
            tui.prompt("Press Enter to continue...", style="dim")
            return None

        # Display formatted list
        formatted_text = self.model_discovery.format_models_for_display("all")
        tui.show_panel(
            formatted_text,
            title="Available Models for Finetuning",
            border_style="green",
        )

        # Get user selection
        model_id = tui.prompt(
            "Enter model ID to select (or 'b' to go back):",
            style="cyan",
        ).strip()

        if model_id.lower() in ["b", "back"]:
            return None

        # Validate selection
        selected = self.model_discovery.get_model_by_id(model_id)
        if selected:
            return model_id
        else:
            tui.show_error(f"Model not found: {model_id}")
            tui.prompt("Press Enter to continue...", style="dim")
            return None

    def _show_llm_models(self) -> Optional[str]:
        """Show LLM models only.

        Returns:
            Selected model ID or None
        """
        tui.clear_screen()
        formatted_text = self.model_discovery.format_models_for_display("llm")
        tui.show_panel(
            formatted_text,
            title="Language Models (LLM) for Finetuning",
            border_style="green",
        )

        model_id = tui.prompt(
            "Enter model ID to select (or 'b' to go back):",
            style="cyan",
        ).strip()

        if model_id.lower() in ["b", "back"]:
            return None

        selected = self.model_discovery.get_model_by_id(model_id)
        if selected:
            return model_id
        else:
            tui.show_error(f"Model not found: {model_id}")
            tui.prompt("Press Enter to continue...", style="dim")
            return None

    def _show_vlm_models(self) -> Optional[str]:
        """Show VLM models only.

        Returns:
            Selected model ID or None
        """
        tui.clear_screen()
        formatted_text = self.model_discovery.format_models_for_display("vlm")
        tui.show_panel(
            formatted_text,
            title="Vision-Language Models (VLM) for Finetuning",
            border_style="green",
        )

        model_id = tui.prompt(
            "Enter model ID to select (or 'b' to go back):",
            style="cyan",
        ).strip()

        if model_id.lower() in ["b", "back"]:
            return None

        selected = self.model_discovery.get_model_by_id(model_id)
        if selected:
            return model_id
        else:
            tui.show_error(f"Model not found: {model_id}")
            tui.prompt("Press Enter to continue...", style="dim")
            return None

    def _search_model(self) -> Optional[str]:
        """Search for model by ID or name fragment.

        Returns:
            Selected model ID or None
        """
        search_term = tui.prompt(
            "Search for model (ID or name fragment):",
            style="cyan",
        ).strip().lower()

        if not search_term:
            return None

        models_dict = self.model_discovery.discover_all_models()
        all_models = models_dict["llm"] + models_dict["vlm"]

        # Search by ID or name
        matching_models = [
            m for m in all_models
            if search_term in m.model_id.lower() or search_term in m.name.lower()
        ]

        if not matching_models:
            tui.show_error(f"No models found matching '{search_term}'")
            tui.prompt("Press Enter to continue...", style="dim")
            return None

        if len(matching_models) == 1:
            return matching_models[0].model_id

        # Display matching models
        tui.clear_screen()
        results_text = "Search Results\n"
        results_text += "━" * 80 + "\n\n"
        for i, model in enumerate(matching_models, 1):
            results_text += f"[{i}] {model.name:<40} ({model.size_gb:.1f}GB)\n"
            results_text += f"    ID: {model.model_id}\n"
            results_text += f"    Provider: {model.provider}\n\n"

        tui.show_panel(
            results_text,
            title=f"Found {len(matching_models)} model(s)",
            border_style="green",
        )

        selection = tui.prompt(
            f"Select [1-{len(matching_models)}] or 'b' to go back:",
            style="cyan",
        ).strip()

        if selection.lower() in ["b", "back"]:
            return None

        try:
            idx = int(selection) - 1
            if 0 <= idx < len(matching_models):
                return matching_models[idx].model_id
            else:
                tui.show_error("Invalid selection")
                return None
        except ValueError:
            tui.show_error("Invalid input")
            return None
