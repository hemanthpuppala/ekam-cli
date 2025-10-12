"""Model information display and parameter configuration."""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from ..models.session import ModelParameters


def display_model_info(console: Console, model_info: dict) -> None:
    """Display detailed model information in rich format.

    Args:
        console: Rich console for output
        model_info: Model metadata dictionary
    """
    # Create main info table
    table = Table(
        title=f"Model Information: {model_info.get('name', 'Unknown')}",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        expand=True,
    )

    table.add_column("Property", style="white", no_wrap=True, width=25)
    table.add_column("Value", style="green")

    # Basic info
    table.add_row("Model ID", model_info.get("model_id", "N/A"))
    table.add_row("Provider", model_info.get("provider", "N/A"))
    table.add_row("Type", model_info.get("model_type", "N/A"))
    table.add_row("Size", f"{model_info.get('size_gb', 0):.2f} GB")

    # Architecture info
    if "architecture" in model_info:
        table.add_row("Architecture", model_info["architecture"])

    if "parameter_count" in model_info:
        table.add_row("Parameters", model_info["parameter_count"])

    # Quantization
    if "quantization" in model_info:
        table.add_row("Quantization", model_info["quantization"])

    # Family/format
    if "family" in model_info:
        table.add_row("Model Family", model_info["family"])

    if "format" in model_info:
        table.add_row("Format", model_info["format"])

    # License
    if "license" in model_info:
        table.add_row("", "")  # Separator
        table.add_row("[bold]License[/bold]", model_info["license"])

    # Default parameters
    if "default_parameters" in model_info:
        table.add_row("", "")  # Separator
        params = model_info["default_parameters"]
        table.add_row("[bold]Default Parameters[/bold]", "")
        if "temperature" in params:
            table.add_row("  Temperature", str(params["temperature"]))
        if "top_p" in params:
            table.add_row("  Top P", str(params["top_p"]))
        if "top_k" in params:
            table.add_row("  Top K", str(params["top_k"]))
        if "max_tokens" in params:
            table.add_row("  Max Tokens", str(params["max_tokens"]))

    # Template info
    if "chat_template" in model_info:
        table.add_row("", "")  # Separator
        table.add_row("Chat Template", model_info["chat_template"][:100] + "..." if len(model_info.get("chat_template", "")) > 100 else model_info.get("chat_template", ""))

    # Capabilities
    if "capabilities" in model_info:
        table.add_row("", "")  # Separator
        table.add_row("[bold]Capabilities[/bold]", ", ".join(model_info["capabilities"]))

    panel = Panel(table, border_style="cyan", expand=True)
    console.print(panel)
    console.print()


def configure_model_parameters(
    console: Console,
    current_params: Optional[ModelParameters] = None,
    defaults: Optional[dict] = None
) -> Optional[ModelParameters]:
    """Interactive parameter configuration.

    Args:
        console: Rich console for output
        current_params: Current parameters (if any)
        defaults: Default values from provider

    Returns:
        ModelParameters if user configured, None if cancelled
    """
    console.print("\n[bold cyan]Model Parameter Configuration[/bold cyan]\n")
    console.print("[dim]Press Enter to use default value • Type value to override[/dim]\n")

    # Get defaults
    if defaults is None:
        defaults = {}

    # Get current values or defaults
    current_max = current_params.max_tokens if current_params and current_params.max_tokens else defaults.get("max_tokens", 512)
    current_temp = current_params.temperature if current_params and current_params.temperature else defaults.get("temperature", 0.7)
    current_top_p = current_params.top_p if current_params and current_params.top_p else defaults.get("top_p", 0.9)
    current_top_k = current_params.top_k if current_params and current_params.top_k else defaults.get("top_k", 40)
    current_repeat = current_params.repeat_penalty if current_params and current_params.repeat_penalty else defaults.get("repeat_penalty", 1.1)
    current_max_history = current_params.max_history_turns if current_params and current_params.max_history_turns else 20  # CLI default

    # Display current/default values
    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Parameter", style="cyan")
    table.add_column("Current/Default", style="green")
    table.add_column("Description", style="dim")

    table.add_row(
        "max_tokens",
        str(current_max),
        "Maximum tokens to generate (1-32768)"
    )
    table.add_row(
        "temperature",
        f"{current_temp:.2f}",
        "Randomness (0.0=deterministic, 2.0=creative)"
    )
    table.add_row(
        "top_p",
        f"{current_top_p:.2f}",
        "Nucleus sampling threshold (0.0-1.0)"
    )
    table.add_row(
        "top_k",
        str(current_top_k),
        "Top-K sampling (number of tokens to consider)"
    )
    table.add_row(
        "repeat_penalty",
        f"{current_repeat:.2f}",
        "Penalize repetition (1.0=none, >1.0=penalize)"
    )
    table.add_row(
        "max_history_turns",
        str(current_max_history),
        "Max history exchanges (1-200, affects context)"
    )

    console.print(table)
    console.print()

    # Ask if user wants to configure
    if not Confirm.ask("[bold]Configure custom parameters?[/bold]", default=False):
        return None

    console.print()

    # Configure each parameter
    params = ModelParameters()

    try:
        # Max tokens
        console.print(f"[cyan]max_tokens[/cyan] (current: {current_max})")
        max_tokens_input = Prompt.ask("  Value", default=str(current_max))
        if max_tokens_input and max_tokens_input != str(current_max):
            params.max_tokens = int(max_tokens_input)

        # Temperature
        console.print(f"[cyan]temperature[/cyan] (current: {current_temp:.2f})")
        temp_input = Prompt.ask("  Value", default=f"{current_temp:.2f}")
        if temp_input and float(temp_input) != current_temp:
            params.temperature = float(temp_input)

        # Top P
        console.print(f"[cyan]top_p[/cyan] (current: {current_top_p:.2f})")
        top_p_input = Prompt.ask("  Value", default=f"{current_top_p:.2f}")
        if top_p_input and float(top_p_input) != current_top_p:
            params.top_p = float(top_p_input)

        # Top K
        console.print(f"[cyan]top_k[/cyan] (current: {current_top_k})")
        top_k_input = Prompt.ask("  Value", default=str(current_top_k))
        if top_k_input and int(top_k_input) != current_top_k:
            params.top_k = int(top_k_input)

        # Repeat penalty
        console.print(f"[cyan]repeat_penalty[/cyan] (current: {current_repeat:.2f})")
        repeat_input = Prompt.ask("  Value", default=f"{current_repeat:.2f}")
        if repeat_input and float(repeat_input) != current_repeat:
            params.repeat_penalty = float(repeat_input)

        # Max history turns
        console.print(f"[cyan]max_history_turns[/cyan] (current: {current_max_history})")
        console.print("  [dim]Controls how many recent exchanges are used for context[/dim]")
        max_history_input = Prompt.ask("  Value", default=str(current_max_history))
        if max_history_input and int(max_history_input) != current_max_history:
            params.max_history_turns = int(max_history_input)

        # Seed (optional)
        console.print("[cyan]seed[/cyan] (optional, for reproducibility)")
        seed_input = Prompt.ask("  Value", default="")
        if seed_input:
            params.seed = int(seed_input)

    except (ValueError, KeyboardInterrupt):
        console.print("\n[yellow]Configuration cancelled[/yellow]")
        return None

    # Show summary
    configured = params.to_dict()
    if configured:
        console.print("\n[bold green]✓ Parameters configured:[/bold green]")
        for key, value in configured.items():
            console.print(f"  [cyan]{key}:[/cyan] {value}")
        console.print()
        return params
    else:
        console.print("\n[dim]No parameters changed[/dim]")
        return None


def show_current_parameters(console: Console, params: Optional[ModelParameters], defaults: dict) -> None:
    """Display current session parameters.

    Args:
        console: Rich console for output
        params: Session parameters (if any)
        defaults: Default values from provider
    """
    console.print("\n[bold cyan]Current Session Parameters[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
    table.add_column("Parameter", style="white")
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    # Get effective values
    max_tokens = params.max_tokens if params and params.max_tokens else defaults.get("max_tokens", 512)
    temp = params.temperature if params and params.temperature else defaults.get("temperature", 0.7)
    top_p = params.top_p if params and params.top_p else defaults.get("top_p", 0.9)
    top_k = params.top_k if params and params.top_k else defaults.get("top_k", 40)
    repeat = params.repeat_penalty if params and params.repeat_penalty else defaults.get("repeat_penalty", 1.1)
    max_history = params.max_history_turns if params and params.max_history_turns else 20  # CLI default

    table.add_row(
        "max_tokens",
        str(max_tokens),
        "Custom" if params and params.max_tokens else "Default"
    )
    table.add_row(
        "temperature",
        f"{temp:.2f}",
        "Custom" if params and params.temperature else "Default"
    )
    table.add_row(
        "top_p",
        f"{top_p:.2f}",
        "Custom" if params and params.top_p else "Default"
    )
    table.add_row(
        "top_k",
        str(top_k),
        "Custom" if params and params.top_k else "Default"
    )
    table.add_row(
        "repeat_penalty",
        f"{repeat:.2f}",
        "Custom" if params and params.repeat_penalty else "Default"
    )
    table.add_row(
        "max_history_turns",
        str(max_history),
        "Custom" if params and params.max_history_turns else "CLI Default (20)"
    )

    if params and params.seed:
        table.add_row("seed", str(params.seed), "Custom")

    console.print(table)
    console.print()
