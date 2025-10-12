"""Input validation utilities for user prompts."""

from pathlib import Path

from rich.prompt import Confirm, Prompt

from .tui_manager import tui


class UserExitException(Exception):
    """Raised when user wants to exit/go back from a prompt."""
    pass


def _check_exit_keywords(user_input: str) -> None:
    """Check if user input contains exit keywords and raise exception if so.

    Args:
        user_input: User's input string

    Raises:
        UserExitException: If user wants to exit
    """
    exit_keywords = ["quit", "exit", "back", "q", "b", "/quit", "/exit", "/back"]
    if user_input.lower().strip() in exit_keywords:
        raise UserExitException("User requested to exit")


def prompt_numeric(prompt: str, min_val: int, max_val: int) -> int:
    """Prompt for numeric input with validation.

    Args:
        prompt: Prompt message
        min_val: Minimum valid value
        max_val: Maximum valid value

    Returns:
        Validated integer

    Raises:
        UserExitException: If user wants to exit (type 'back', 'quit', 'q', 'b')
    """
    tui.console.print("[dim]Type 'back' or 'q' to go back[/dim]")
    while True:
        user_input = tui.prompt(prompt, style="cyan").strip()
        _check_exit_keywords(user_input)

        try:
            value = int(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                tui.show_error(f"Please enter a number between {min_val} and {max_val}")
        except ValueError:
            tui.show_error("Please enter a valid number")


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Prompt for yes/no input.

    Args:
        prompt: Prompt message
        default: Default value if user just presses Enter

    Returns:
        True for yes, False for no
    """
    default_str = "Y/n" if default else "y/N"
    response = tui.prompt(f"{prompt} [{default_str}]:", style="cyan").lower().strip()

    if not response:
        return default

    return response in ["y", "yes"]


def prompt_file_path(prompt: str, must_exist: bool = True) -> Path:
    """Prompt for file path with validation.

    Args:
        prompt: Prompt message
        must_exist: If True, file must exist

    Returns:
        Validated Path object

    Raises:
        UserExitException: If user wants to exit (type 'back', 'quit', 'q', 'b')
    """
    tui.console.print("[dim]Type 'back' or 'q' to go back[/dim]")
    while True:
        path_str = tui.prompt(prompt, style="cyan").strip()
        _check_exit_keywords(path_str)

        if not path_str:
            tui.show_error("Please enter a file path")
            continue

        path = Path(path_str).expanduser()

        if must_exist and not path.exists():
            tui.show_error(f"File not found: {path}")
            continue

        if must_exist and not path.is_file():
            tui.show_error(f"Not a file: {path}")
            continue

        return path


def prompt_image_path() -> Path:
    """Prompt for image file path with validation.

    Returns:
        Path to valid image file

    Raises:
        UserExitException: If user wants to exit (type 'back', 'quit', 'q', 'b')
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    while True:
        path = prompt_file_path("Enter image path:", must_exist=True)

        if path.suffix.lower() in valid_extensions:
            return path

        tui.show_error(
            f"Invalid image format. Supported: {', '.join(valid_extensions)}"
        )


def prompt_question() -> str:
    """Prompt for question text.

    Returns:
        Question string

    Raises:
        UserExitException: If user wants to exit (type 'back', 'quit', 'q', 'b')
    """
    tui.console.print("[dim]Type 'back' or 'q' to go back[/dim]")
    while True:
        question = tui.prompt("Enter your question:", style="cyan").strip()
        _check_exit_keywords(question)

        if question:
            return question

        tui.show_error("Please enter a question")


def prompt_text_input(prompt: str = "Enter text:", allow_blank: bool = False) -> str:
    """Prompt for text input.

    Args:
        prompt: Prompt message
        allow_blank: If True, allow blank input

    Returns:
        Text string (may be empty if allow_blank=True)

    Raises:
        UserExitException: If user wants to exit (type 'back', 'quit', 'q', 'b')
    """
    tui.console.print("[dim]Type 'back' or 'q' to go back[/dim]")
    while True:
        text = tui.prompt(prompt, style="cyan").strip()
        _check_exit_keywords(text)

        if text or allow_blank:
            return text

        tui.show_error("Please enter some text")


def prompt_choice(prompt: str, choices: list[str]) -> str:
    """Prompt for choice from list.

    Args:
        prompt: Prompt message
        choices: List of valid choices

    Returns:
        Selected choice
    """
    choices_lower = [c.lower() for c in choices]

    while True:
        choice = tui.prompt(
            f"{prompt} [{'/'.join(choices)}]:", style="cyan"
        ).lower().strip()

        if choice in choices_lower:
            return choices[choices_lower.index(choice)]

        tui.show_error(f"Please choose from: {', '.join(choices)}")


def prompt_model_name(provider: str = "ollama") -> str:
    """Prompt for model name with examples.

    Args:
        provider: Provider name for context-specific examples

    Returns:
        Model name string

    Raises:
        UserExitException: If user wants to exit (type 'back', 'quit', 'q', 'b')
    """
    # Show examples based on provider
    if provider.lower() == "ollama":
        tui.console.print("\n[bold]Examples:[/bold]")
        tui.console.print("  • [cyan]llama3.2:3b[/cyan] - Small LLM (2GB)")
        tui.console.print("  • [cyan]llama3.2:1b[/cyan] - Tiny LLM (1GB)")
        tui.console.print("  • [cyan]llava:7b[/cyan] - Vision model (4GB)")
        tui.console.print("  • [cyan]qwen2.5:7b[/cyan] - Text model (4GB)")
        tui.console.print("\n[dim]See https://ollama.com/library for all models[/dim]\n")

    tui.console.print("[dim]Type 'back' or 'q' to cancel[/dim]")
    while True:
        model_name = tui.prompt("Enter model name to install:", style="cyan").strip()
        _check_exit_keywords(model_name)

        if model_name:
            return model_name

        tui.show_error("Please enter a model name")


def confirm_installation(
    model_name: str,
    size_gb: float,
    compatibility: str,
    available_gb: float
) -> bool:
    """Confirm model installation with warnings for large models.

    Args:
        model_name: Name of model to install
        size_gb: Estimated size in GB
        compatibility: Compatibility status (perfect_fit, tight_fit, too_large)
        available_gb: Available RAM in GB

    Returns:
        True if user confirms installation
    """
    tui.clear_screen()

    # Build confirmation message
    if compatibility == "too_large":
        message = f"""[bold yellow]⚠️  Large Model Warning[/bold yellow]

[bold]Model:[/bold] {model_name}
[bold]Estimated Size:[/bold] {size_gb:.1f} GB
[bold]Available RAM:[/bold] {available_gb:.1f} GB

[yellow]This model may be too large for your system.[/yellow]

Proceeding may cause:
  • System slowdown or freezing
  • Out of memory errors
  • Excessive swap usage
  • Failed installation

[dim]Recommendation: Choose a smaller model variant[/dim]
"""
        tui.show_panel(message, title="Installation Warning", border_style="yellow")

        # Require explicit "yes, proceed anyway"
        response = tui.prompt(
            "Type 'yes, proceed anyway' to continue or press Enter to cancel:",
            style="yellow"
        ).lower().strip()

        return response == "yes, proceed anyway"

    elif compatibility == "tight_fit":
        message = f"""[bold cyan]Model Installation[/bold cyan]

[bold]Model:[/bold] {model_name}
[bold]Estimated Size:[/bold] {size_gb:.1f} GB
[bold]Available RAM:[/bold] {available_gb:.1f} GB

[yellow]⚠️  This model will use most of your available memory.[/yellow]

Performance may be impacted. Consider closing other applications.
"""
        tui.show_panel(message, title="Installation Confirmation", border_style="cyan")

        return prompt_yes_no("Proceed with installation?", default=True)

    else:  # perfect_fit
        message = f"""[bold green]Model Installation[/bold green]

[bold]Model:[/bold] {model_name}
[bold]Estimated Size:[/bold] {size_gb:.1f} GB
[bold]Available RAM:[/bold] {available_gb:.1f} GB

[green]✓ This model should fit comfortably in memory.[/green]
"""
        tui.show_panel(message, title="Installation Confirmation", border_style="green")

        return prompt_yes_no("Proceed with installation?", default=True)


def confirm_deletion(model_name: str, size_gb: float) -> bool:
    """Confirm model deletion with warnings.

    Args:
        model_name: Name of model to delete
        size_gb: Model size in GB

    Returns:
        True if user confirms deletion
    """
    tui.clear_screen()

    message = f"""[bold yellow]⚠️  Model Deletion[/bold yellow]

[bold]Model:[/bold] {model_name}
[bold]Size:[/bold] {size_gb:.1f} GB

[yellow]This will permanently delete the model from disk.[/yellow]

This action cannot be undone.
"""
    tui.show_panel(message, title="Confirm Deletion", border_style="yellow")

    # Require explicit confirmation
    response = tui.prompt(
        "Type 'delete' to confirm or press Enter to cancel:",
        style="yellow"
    ).lower().strip()

    return response == "delete"


def prompt_enable_history(endpoint_name: str) -> bool:
    """Prompt to enable conversation history.

    Args:
        endpoint_name: Name of the endpoint (QA, Caption, Chat)

    Returns:
        True if user wants to enable history

    Raises:
        UserExitException: If user wants to exit (type 'back', 'quit', 'q', 'b')
    """
    message = f"""[bold cyan]Conversation History for {endpoint_name}[/bold cyan]

[bold]Options:[/bold]
  [1] [green]WITH History[/green]  - Model remembers conversation context
  [2] [yellow]NO History[/yellow]   - Each message is independent (faster)

[dim]With history: Better context, more memory. Without: Faster, no memory.[/dim]
"""
    tui.show_panel(message, title="Choose Mode", border_style="cyan")
    tui.console.print("[dim]Type 'back' or 'q' to go back[/dim]")

    while True:
        choice = tui.prompt("Choose [1/2]:", style="cyan").strip()
        _check_exit_keywords(choice)

        if choice == "1" or choice.lower() in ["y", "yes", "with"]:
            return True
        elif choice == "2" or choice.lower() in ["n", "no", "without"]:
            return False
        else:
            tui.show_error("Please enter 1 or 2")


def select_or_create_session(
    endpoint_name: str,
    existing_sessions: list,
    show_image_path: bool = False
) -> tuple[str, str | None]:
    """Prompt user to select existing session or create new one.

    Args:
        endpoint_name: Name of the endpoint
        existing_sessions: List of ConversationSession objects
        show_image_path: Whether to show image paths (for QA/Caption)

    Returns:
        Tuple of (action, session_id) where action is "new" or "continue"

    Raises:
        UserExitException: If user wants to exit (type 'back', 'quit', 'q', 'b')
    """
    tui.clear_screen()

    if not existing_sessions:
        return ("new", None)

    # Show existing sessions with clearer formatting
    tui.show_panel(
        f"[bold cyan]{endpoint_name} Sessions Found[/bold cyan]\n\n"
        f"[bold]Choose an option:[/bold]\n"
        f"  • Select number to [green]CONTINUE[/green] existing session\n"
        f"  • Type [yellow]'n'[/yellow] to start [yellow]NEW[/yellow] session",
        title="Session Selection",
        border_style="cyan"
    )

    tui.console.print()
    tui.console.print("[bold]Existing Sessions:[/bold]")
    for idx, session in enumerate(existing_sessions, 1):
        # Format session info with color coding
        time_ago = _format_time_ago(session.last_active)
        exchanges_text = f"{session.exchange_count} msg" if session.exchange_count == 1 else f"{session.exchange_count} msgs"

        info = f"  [{idx}] [cyan]{session.session_id}[/cyan] • {exchanges_text} • [dim]{time_ago}[/dim]"

        if show_image_path and session.image_path:
            path_short = Path(session.image_path).name
            info += f"\n      [dim]→ Image: {path_short}[/dim]"

        tui.console.print(info)

    tui.console.print(f"\n  [n] [yellow]Start NEW session[/yellow]")
    tui.console.print("\n[dim]Type 'back' or 'q' to go back[/dim]")
    tui.console.print()

    # Get user choice with clearer prompt
    while True:
        choice = tui.prompt(
            f"Your choice [1-{len(existing_sessions)}/n]:",
            style="cyan bold"
        ).lower().strip()
        _check_exit_keywords(choice)

        if choice == "n" or choice == "new":
            return ("new", None)

        try:
            idx = int(choice)
            if 1 <= idx <= len(existing_sessions):
                session = existing_sessions[idx - 1]
                return ("continue", session.session_id)
            else:
                tui.show_error(f"Invalid choice. Enter 1-{len(existing_sessions)} or 'n'")
        except ValueError:
            tui.show_error(f"Invalid input. Enter a number (1-{len(existing_sessions)}) or 'n'")


def _format_time_ago(dt: "datetime") -> str:
    """Format datetime as human-readable 'time ago' string.

    Args:
        dt: Datetime to format

    Returns:
        Human-readable string like "2 minutes ago"
    """
    from datetime import datetime

    now = datetime.now()
    diff = (now - dt).total_seconds()

    if diff < 60:
        return "just now"
    elif diff < 3600:
        mins = int(diff / 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    elif diff < 86400:
        hours = int(diff / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(diff / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
