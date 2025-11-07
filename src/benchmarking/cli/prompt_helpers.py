"""
Prompt helper functions for benchmark CLI.

Wraps the existing CLI prompt functions to provide a consistent API
for benchmark configuration screens.
"""

from typing import Optional, List, Tuple
from rich.console import Console

console = Console()


def get_selection(
    prompt: str,
    options: List[Tuple[str, str]],
    show_back: bool = False
) -> Optional[str]:
    """
    Display a menu and get user selection.

    Args:
        prompt: Prompt message
        options: List of (value, description) tuples
        show_back: Whether to show a back option

    Returns:
        Selected value or None if cancelled/back
    """
    console.print(f"\n[cyan]{prompt}[/cyan]")

    for idx, (value, desc) in enumerate(options, 1):
        console.print(f"  [{idx}] {desc}")

    if show_back:
        console.print("  [b] Back")

    while True:
        choice = console.input("\n[cyan]Choose: [/cyan]").strip().lower()

        if show_back and choice == 'b':
            return None

        try:
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
            else:
                console.print("[red]Invalid choice. Try again.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Enter a number.[/red]")


def get_confirmation(prompt: str, default: bool = True) -> bool:
    """
    Get yes/no confirmation from user.

    Args:
        prompt: Confirmation prompt
        default: Default value

    Returns:
        True for yes, False for no
    """
    default_str = "Y/n" if default else "y/N"
    response = console.input(f"\n[cyan]{prompt} [{default_str}]: [/cyan]").strip().lower()

    if not response:
        return default

    return response in ['y', 'yes']


def get_text_input(
    prompt: str,
    allow_empty: bool = False,
    default: Optional[str] = None
) -> Optional[str]:
    """
    Get text input from user.

    Args:
        prompt: Input prompt
        allow_empty: Whether to allow empty input
        default: Default value if user presses enter

    Returns:
        User input or None if cancelled
    """
    default_str = f" (default: {default})" if default else ""
    full_prompt = f"\n[cyan]{prompt}{default_str}: [/cyan]"

    while True:
        response = console.input(full_prompt).strip()

        if not response:
            if default:
                return default
            elif allow_empty:
                return ""
            else:
                console.print("[red]Input required. Try again.[/red]")
                continue

        return response
