"""Model information display and parameter configuration."""

from typing import Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from ..models.session import ModelParameters
from .text_input import professional_prompt


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
    """Interactive parameter configuration with validation and multi-edit support."""

    console.print("\n[bold cyan]Model Parameter Configuration[/bold cyan]\n")
    console.print("[dim]Use ↑/↓ to navigate, Enter to edit, q to finish[/dim]\n")

    defaults = defaults or {}

    CANCELLED = object()

    base_fallbacks = {
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "max_history_turns": 20,
    }

    def get_base_value(name: str) -> Union[float, int]:
        value = getattr(current_params, name) if current_params else None
        if value is not None:
            return value
        return defaults.get(name, base_fallbacks[name])

    def current_or_default(value, fallback):
        return value if value is not None else fallback

    current_max = get_base_value("max_tokens")
    current_temp = get_base_value("temperature")
    current_top_p = get_base_value("top_p")
    current_top_k = get_base_value("top_k")
    current_repeat = get_base_value("repeat_penalty")
    current_max_history = get_base_value("max_history_turns")

    def format_display(value):
        if value is None:
            return "Not set"
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, list):
            return ", ".join(str(v) for v in value) if value else "[]"
        if isinstance(value, dict):
            import json as _json
            serialized = _json.dumps(value)
            return serialized if len(serialized) <= 32 else serialized[:29] + "..."
        return str(value)

    # Display current/default values
    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Parameter", style="cyan")
    table.add_column("Current/Default", style="green")
    table.add_column("Description", style="dim")

    table.add_row("max_tokens", str(current_max), "Maximum tokens to generate (1-32768)")
    table.add_row("temperature", f"{float(current_temp):.2f}", "Randomness (0.0-2.0)")
    table.add_row("top_p", f"{float(current_top_p):.2f}", "Nucleus sampling threshold (0.0-1.0)")
    table.add_row("top_k", str(current_top_k), "Top-K sampling (number of tokens to consider)")
    table.add_row("repeat_penalty", f"{float(current_repeat):.2f}", "Penalize repetition (1.0-10.0)")
    table.add_row("max_history_turns", str(current_max_history), "Max history exchanges (1-200)")

    console.print(table)
    console.print()

    # Advanced parameters summary with live values
    adv = Table(show_header=True, header_style="bold magenta", border_style="dim")
    adv.add_column("Advanced Param", style="cyan")
    adv.add_column("Current/Default", style="green")
    adv.add_column("Description", style="dim")

    advanced_fields = [
        ("system_prompt", "System message (prepended to conversation)"),
        ("seed", "Random seed for reproducibility"),
        ("presence_penalty", "Discourage new topic repeats (0.0-2.0)"),
        ("frequency_penalty", "Discourage token frequency (0.0-2.0)"),
        ("typical_p", "Typical decoding p (0.0-1.0)"),
        ("tfs_z", "Tail free sampling z (0.0-1.0)"),
        ("min_p", "Min-p sampling (0.0-1.0)"),
        ("penalty_last_n", "Tokens considered for repetition penalty"),
        ("mirostat", "Mirostat mode (0=off,1,2)"),
        ("mirostat_tau", "Mirostat target entropy"),
        ("mirostat_eta", "Mirostat learning rate"),
        ("n_keep", "Tokens to keep from truncation"),
        ("ignore_eos", "Ignore EOS tokens"),
        ("stop", "Stop sequences (list)"),
        ("n_probs", "Return top-n token probabilities"),
        ("grammar", "GGML grammar (string)"),
        ("logit_bias", "Logit bias map (token/id → bias)"),
    ]

    for field_name, description in advanced_fields:
        value = getattr(current_params, field_name) if current_params else None
        if value is None and field_name in defaults:
            value = defaults[field_name]
        adv.add_row(field_name, format_display(value), description)

    console.print(adv)
    console.print()

    if not Confirm.ask("[bold]Configure custom parameters?[/bold]", default=False):
        return None

    console.print()

    existing_payload = current_params.to_dict() if current_params else {}
    working_params = ModelParameters(**existing_payload) if existing_payload else ModelParameters()

    def effective_values() -> dict:
        base_values = {
            "max_tokens": working_params.max_tokens if working_params.max_tokens is not None else defaults.get("max_tokens", base_fallbacks["max_tokens"]),
            "temperature": working_params.temperature if working_params.temperature is not None else defaults.get("temperature", base_fallbacks["temperature"]),
            "top_p": working_params.top_p if working_params.top_p is not None else defaults.get("top_p", base_fallbacks["top_p"]),
            "top_k": working_params.top_k if working_params.top_k is not None else defaults.get("top_k", base_fallbacks["top_k"]),
            "repeat_penalty": working_params.repeat_penalty if working_params.repeat_penalty is not None else defaults.get("repeat_penalty", base_fallbacks["repeat_penalty"]),
            "max_history_turns": working_params.max_history_turns if working_params.max_history_turns is not None else base_fallbacks["max_history_turns"],
        }

        extra_values = {}
        for key, _ in advanced_fields:
            extra_values[key] = getattr(working_params, key)
            if extra_values[key] is None and key in defaults:
                extra_values[key] = defaults[key]

        return {**base_values, **extra_values}

    def build_options(value_map: dict) -> list[tuple[str, str, str]]:
        def format_option(key: str) -> str:
            val = value_map.get(key)
            return "Not set" if val is None or val == "" else format_display(val)

        return [
            ("max_tokens", f"[cyan]max_tokens[/cyan] — {format_option('max_tokens')}", "Maximum tokens to generate"),
            ("temperature", f"[cyan]temperature[/cyan] — {format_option('temperature')}", "Randomness control (0.0-2.0)"),
            ("top_p", f"[cyan]top_p[/cyan] — {format_option('top_p')}", "Nucleus sampling threshold"),
            ("top_k", f"[cyan]top_k[/cyan] — {format_option('top_k')}", "Top-K sampling"),
            ("repeat_penalty", f"[cyan]repeat_penalty[/cyan] — {format_option('repeat_penalty')}", "Penalize repetition"),
            ("max_history_turns", f"[cyan]max_history_turns[/cyan] — {format_option('max_history_turns')}", "History turns for context"),
            ("seed", f"[cyan]seed[/cyan] — {format_option('seed')}", "Random seed for reproducibility"),
            ("system_prompt", f"[cyan]system_prompt[/cyan] — {format_option('system_prompt')}", "System message (prepended)"),
            ("presence_penalty", f"[cyan]presence_penalty[/cyan] — {format_option('presence_penalty')}", "Discourage new topic repeats"),
            ("frequency_penalty", f"[cyan]frequency_penalty[/cyan] — {format_option('frequency_penalty')}", "Discourage token frequency"),
            ("typical_p", f"[cyan]typical_p[/cyan] — {format_option('typical_p')}", "Typical decoding p"),
            ("tfs_z", f"[cyan]tfs_z[/cyan] — {format_option('tfs_z')}", "Tail free sampling z"),
            ("min_p", f"[cyan]min_p[/cyan] — {format_option('min_p')}", "Min-p sampling"),
            ("penalty_last_n", f"[cyan]penalty_last_n[/cyan] — {format_option('penalty_last_n')}", "Tokens considered for repetition penalty"),
            ("mirostat", f"[cyan]mirostat[/cyan] — {format_option('mirostat')}", "Mirostat mode (0=off,1,2)"),
            ("mirostat_tau", f"[cyan]mirostat_tau[/cyan] — {format_option('mirostat_tau')}", "Mirostat target entropy"),
            ("mirostat_eta", f"[cyan]mirostat_eta[/cyan] — {format_option('mirostat_eta')}", "Mirostat learning rate"),
            ("n_keep", f"[cyan]n_keep[/cyan] — {format_option('n_keep')}", "Tokens to keep from truncation"),
            ("ignore_eos", f"[cyan]ignore_eos[/cyan] — {format_option('ignore_eos')}", "Ignore EOS tokens"),
            ("stop", f"[cyan]stop[/cyan] — {format_option('stop')}", "Stop sequences"),
            ("n_probs", f"[cyan]n_probs[/cyan] — {format_option('n_probs')}", "Return top-n token probabilities"),
            ("grammar", f"[cyan]grammar[/cyan] — {format_option('grammar')}", "GGML grammar (string)"),
            ("logit_bias", f"[cyan]logit_bias[/cyan] — {format_option('logit_bias')}", "Logit bias map (token/id → bias)"),
        ]

    def prompt_float(
        label: str,
        min_val: float,
        max_val: float,
        default_val: Optional[float] = None,
        allow_empty: bool = False,
        multiline: bool = False,
        keep_current: Optional[float] = None,
    ) -> Union[Optional[float], object]:
        while True:
            hint = f"{label} [{min_val}-{max_val}]"
            if keep_current is not None:
                hint += f" (current: {keep_current})"
            if default_val is not None and not allow_empty:
                hint += f" • default: {default_val}"
            if allow_empty:
                hint += " • leave empty to clear"
            console.print(f"[dim]{hint}[/dim]")
            response = professional_prompt.get_text_input(
                label,
                default_value="" if allow_empty else (str(default_val) if default_val is not None else ""),
                allow_empty=allow_empty,
                multiline=multiline,
            )
            if response is None:
                return CANCELLED
            if response == "":
                return None if allow_empty else default_val
            try:
                value = float(response)
            except ValueError:
                console.print("[yellow]Please enter a valid number[/yellow]")
                continue
            if not (min_val <= value <= max_val):
                console.print(f"[yellow]Value must be between {min_val} and {max_val}[/yellow]")
                continue
            return value

    def prompt_int(
        label: str,
        min_val: int,
        max_val: int,
        default_val: Optional[int] = None,
        allow_empty: bool = False,
        keep_current: Optional[int] = None,
    ) -> Union[Optional[int], object]:
        while True:
            hint = f"{label} [{min_val}-{max_val}]"
            if keep_current is not None:
                hint += f" (current: {keep_current})"
            if default_val is not None and not allow_empty:
                hint += f" • default: {default_val}"
            if allow_empty:
                hint += " • leave empty to clear"
            console.print(f"[dim]{hint}[/dim]")
            response = professional_prompt.get_text_input(
                label,
                default_value="" if allow_empty else (str(default_val) if default_val is not None else ""),
                allow_empty=allow_empty,
            )
            if response is None:
                return CANCELLED
            if response == "":
                return None if allow_empty else default_val
            try:
                value = int(response)
            except ValueError:
                console.print("[yellow]Please enter a whole number[/yellow]")
                continue
            if not (min_val <= value <= max_val):
                console.print(f"[yellow]Value must be between {min_val} and {max_val}[/yellow]")
                continue
            return value

    def prompt_text(
        label: str,
        current_value: Optional[str] = None,
        multiline: bool = False,
        allow_empty: bool = True,
    ) -> Union[Optional[str], object]:
        hint = f"{label}"
        if current_value:
            snippet = current_value[:40] + ("…" if len(current_value) > 40 else "")
            hint += f" (current: {snippet})"
        if allow_empty:
            hint += " • leave empty to clear"
        console.print(f"[dim]{hint}[/dim]")
        response = professional_prompt.get_text_input(
            label,
            default_value="",
            allow_empty=allow_empty,
            multiline=multiline,
        )
        if response is None:
            return CANCELLED
        return response

    while True:
        value_map = effective_values()
        options = build_options(value_map)
        selection = professional_prompt.get_arrow_selection(
            options=options,
            title="Select parameter to configure",
            instructions="Use ↑/↓ to navigate, Enter to select, q to finish",
        )

        if selection is None:
            break

        try:
            updated = False
            if selection == "max_tokens":
                default_val = defaults.get("max_tokens", base_fallbacks["max_tokens"])
                value = prompt_int(
                    "max_tokens",
                    1,
                    32768,
                    default_val=default_val,
                    keep_current=current_or_default(working_params.max_tokens, default_val),
                )
                if value is CANCELLED:
                    continue
                if value is not None:
                    working_params.max_tokens = value
                    updated = True
            elif selection == "temperature":
                default_val = defaults.get("temperature", base_fallbacks["temperature"])
                value = prompt_float(
                    "temperature",
                    0.0,
                    2.0,
                    default_val=default_val,
                    keep_current=current_or_default(working_params.temperature, default_val),
                )
                if value is CANCELLED:
                    continue
                if value is not None:
                    working_params.temperature = value
                    updated = True
            elif selection == "top_p":
                default_val = defaults.get("top_p", base_fallbacks["top_p"])
                value = prompt_float(
                    "top_p",
                    0.0,
                    1.0,
                    default_val=default_val,
                    keep_current=current_or_default(working_params.top_p, default_val),
                )
                if value is CANCELLED:
                    continue
                if value is not None:
                    working_params.top_p = value
                    updated = True
            elif selection == "top_k":
                default_val = defaults.get("top_k", base_fallbacks["top_k"])
                value = prompt_int(
                    "top_k",
                    0,
                    1000,
                    default_val=default_val,
                    keep_current=current_or_default(working_params.top_k, default_val),
                )
                if value is CANCELLED:
                    continue
                if value is not None:
                    working_params.top_k = value
                    updated = True
            elif selection == "repeat_penalty":
                default_val = defaults.get("repeat_penalty", base_fallbacks["repeat_penalty"])
                value = prompt_float(
                    "repeat_penalty",
                    0.0,
                    10.0,
                    default_val=default_val,
                    keep_current=current_or_default(working_params.repeat_penalty, default_val),
                )
                if value is CANCELLED:
                    continue
                if value is not None:
                    working_params.repeat_penalty = value
                    updated = True
            elif selection == "max_history_turns":
                value = prompt_int(
                    "max_history_turns",
                    1,
                    200,
                    default_val=base_fallbacks["max_history_turns"],
                    keep_current=current_or_default(working_params.max_history_turns, base_fallbacks["max_history_turns"]),
                )
                if value is CANCELLED:
                    continue
                if value is not None:
                    working_params.max_history_turns = value
                    updated = True
            elif selection == "seed":
                value = prompt_int("seed", 0, 2**32 - 1, allow_empty=True, keep_current=working_params.seed)
                if value is CANCELLED:
                    continue
                if value is None:
                    working_params.seed = None
                    updated = True
                else:
                    working_params.seed = value
                    updated = True
            elif selection == "system_prompt":
                response = prompt_text("system_prompt", current_value=working_params.system_prompt, multiline=True)
                if response is CANCELLED:
                    continue
                working_params.system_prompt = response or None
                updated = True
            elif selection == "presence_penalty":
                value = prompt_float("presence_penalty", 0.0, 2.0, allow_empty=True, keep_current=working_params.presence_penalty)
                if value is CANCELLED:
                    continue
                working_params.presence_penalty = value
                updated = True
            elif selection == "frequency_penalty":
                value = prompt_float("frequency_penalty", 0.0, 2.0, allow_empty=True, keep_current=working_params.frequency_penalty)
                if value is CANCELLED:
                    continue
                working_params.frequency_penalty = value
                updated = True
            elif selection == "typical_p":
                value = prompt_float("typical_p", 0.0, 1.0, allow_empty=True, keep_current=working_params.typical_p)
                if value is CANCELLED:
                    continue
                working_params.typical_p = value
                updated = True
            elif selection == "tfs_z":
                value = prompt_float("tfs_z", 0.0, 1.0, allow_empty=True, keep_current=working_params.tfs_z)
                if value is CANCELLED:
                    continue
                working_params.tfs_z = value
                updated = True
            elif selection == "min_p":
                value = prompt_float("min_p", 0.0, 1.0, allow_empty=True, keep_current=working_params.min_p)
                if value is CANCELLED:
                    continue
                working_params.min_p = value
                updated = True
            elif selection == "penalty_last_n":
                value = prompt_int("penalty_last_n", 0, 4096, allow_empty=True, keep_current=working_params.penalty_last_n)
                if value is CANCELLED:
                    continue
                working_params.penalty_last_n = value
                updated = True
            elif selection == "mirostat":
                value = prompt_int("mirostat (0=off,1,2)", 0, 2, allow_empty=True, keep_current=working_params.mirostat)
                if value is CANCELLED:
                    continue
                working_params.mirostat = value
                updated = True
            elif selection == "mirostat_tau":
                value = prompt_float("mirostat_tau", 0.0, 10.0, allow_empty=True, keep_current=working_params.mirostat_tau)
                if value is CANCELLED:
                    continue
                working_params.mirostat_tau = value
                updated = True
            elif selection == "mirostat_eta":
                value = prompt_float("mirostat_eta", 0.0, 10.0, allow_empty=True, keep_current=working_params.mirostat_eta)
                if value is CANCELLED:
                    continue
                working_params.mirostat_eta = value
                updated = True
            elif selection == "n_keep":
                value = prompt_int("n_keep", 0, 32768, allow_empty=True, keep_current=working_params.n_keep)
                if value is CANCELLED:
                    continue
                working_params.n_keep = value
                updated = True
            elif selection == "ignore_eos":
                console.print("[dim]Enter y/yes to ignore EOS, n/no to honor EOS, leave empty to clear[/dim]")
                response = professional_prompt.get_text_input("ignore_eos", default_value="", allow_empty=True)
                if response is None:
                    updated = False
                elif response.strip() == "":
                    working_params.ignore_eos = None
                    updated = True
                else:
                    normalized = response.strip().lower()
                    if normalized in ["y", "yes", "true", "1"]:
                        working_params.ignore_eos = True
                        updated = True
                    elif normalized in ["n", "no", "false", "0"]:
                        working_params.ignore_eos = False
                        updated = True
                    else:
                        console.print("[yellow]Please answer yes or no[/yellow]")
                        updated = False
                        continue
            elif selection == "stop":
                console.print("[dim]Enter comma-separated stop sequences, leave empty to clear[/dim]")
                response = professional_prompt.get_text_input("stop", default_value="", allow_empty=True)
                if response is None:
                    updated = False
                elif response.strip():
                    working_params.stop = [s.strip() for s in response.split(",") if s.strip()]
                    updated = True
                else:
                    working_params.stop = None
                    updated = True
            elif selection == "n_probs":
                value = prompt_int("n_probs", 0, 10, allow_empty=True, keep_current=working_params.n_probs)
                if value is CANCELLED:
                    continue
                working_params.n_probs = value
                updated = True
            elif selection == "grammar":
                response = prompt_text("grammar", current_value=working_params.grammar)
                if response is CANCELLED:
                    continue
                working_params.grammar = response or None
                updated = True
            elif selection == "logit_bias":
                console.print("[dim]Provide JSON object mapping token/id to bias, leave empty to clear[/dim]")
                response = professional_prompt.get_text_input("logit_bias", default_value="", allow_empty=True, multiline=True)
                if response is None:
                    updated = False
                elif response.strip() == "":
                    working_params.logit_bias = None
                    updated = True
                else:
                    import json as _json
                    try:
                        parsed = _json.loads(response)
                        if isinstance(parsed, dict):
                            working_params.logit_bias = parsed
                            updated = True
                        else:
                            console.print("[yellow]Logit bias must be a JSON object[/yellow]")
                            continue
                    except _json.JSONDecodeError as exc:
                        console.print(f"[yellow]Invalid JSON: {exc}[/yellow]")
                        continue
            else:
                console.print("\n[dim]No parameters changed[/dim]")
                continue

            if updated:
                console.print("[green]✓ Parameter updated[/green]\n")
        except KeyboardInterrupt:
            console.print("\n[yellow]Input cancelled[/yellow]\n")
            continue

    updated_payload = working_params.to_dict()
    if updated_payload == existing_payload:
        console.print("\n[dim]No parameters changed[/dim]")
        return None

    console.print("\n[bold green]✓ Parameters updated[/bold green]\n")
    return working_params


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
