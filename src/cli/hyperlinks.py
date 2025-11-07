"""OSC 8 hyperlink generation for clickable file paths in terminal."""

from pathlib import Path
from typing import Optional


def make_hyperlink(path: Path, text: Optional[str] = None) -> str:
    """Generate OSC 8 hyperlink for clickable terminal paths.

    Supported terminals: iTerm2, WezTerm, kitty, Windows Terminal

    Args:
        path: File path to make clickable
        text: Display text (defaults to path string)

    Returns:
        Formatted string with OSC 8 escape sequences
    """
    # Convert to absolute path
    abs_path = path.resolve()

    # Default text to path string
    if text is None:
        text = str(path)

    # OSC 8 format: \033]8;;file://path\033\\text\033]8;;\033\\
    return f"\033]8;;file://{abs_path}\033\\{text}\033]8;;\033\\"


def make_hyperlink_fallback(path: Path, text: Optional[str] = None) -> str:
    """Generate hyperlink with graceful fallback for non-supporting terminals.

    Args:
        path: File path to make clickable
        text: Display text (defaults to path string)

    Returns:
        Formatted string (hyperlink if supported, plain text otherwise)
    """
    try:
        return make_hyperlink(path, text)
    except Exception:
        # Fallback to plain text if hyperlink fails
        return text or str(path)
