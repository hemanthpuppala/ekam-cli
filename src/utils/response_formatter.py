"""Response formatting utilities for handling reasoning models."""

import re
from typing import Optional

from rich.console import Console
from rich.panel import Panel


def extract_thinking_blocks(text: str) -> tuple[list[str], str]:
    """Extract thinking blocks from response text.

    Handles both closed (<think>...</think>) and unclosed (<think>...) thinking blocks.

    Args:
        text: Raw response text potentially containing <think> blocks

    Returns:
        Tuple of (list of thinking blocks, clean response without thinking)

    Example:
        >>> text = "<think>reasoning</think>Hello world"
        >>> thinking, clean = extract_thinking_blocks(text)
        >>> thinking
        ['reasoning']
        >>> clean
        'Hello world'

        >>> text = "<think>unclosed reasoning"
        >>> thinking, clean = extract_thinking_blocks(text)
        >>> thinking
        ['unclosed reasoning']
        >>> clean
        ''
    """
    # Handle None or empty text
    if not text:
        return [], ""

    thinking_blocks = []
    clean_text = text

    # First, try to match properly closed thinking blocks (handles multiple blocks)
    closed_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
    closed_matches = closed_pattern.findall(text)

    if closed_matches:
        # Filter out empty thinking blocks (from nested tags or empty blocks)
        thinking_blocks.extend([block.strip() for block in closed_matches if block.strip()])
        # Remove all thinking blocks from text
        clean_text = closed_pattern.sub('', text).strip()

    # Then check for unclosed thinking blocks (everything after last <think>)
    # Only look for unclosed if we didn't find closed ones (avoid double-processing)
    if not closed_matches:
        # Find ALL unclosed <think> tags and take the last one
        unclosed_pattern = re.compile(r'<think>\s*(.*)', re.DOTALL | re.IGNORECASE)
        # Use finditer to get all matches, then take the last one
        unclosed_matches = list(unclosed_pattern.finditer(text))

        if unclosed_matches:
            # Take the last unclosed <think> tag
            last_match = unclosed_matches[-1]
            thinking_content = last_match.group(1).strip()
            if thinking_content:
                thinking_blocks.append(thinking_content)
            # Everything before the last <think> is the clean text
            clean_text = text[:last_match.start()].strip()

    return thinking_blocks, clean_text


def has_thinking_blocks(text: str) -> bool:
    """Check if text contains thinking blocks (closed or unclosed).

    Args:
        text: Text to check

    Returns:
        True if text contains <think> tags (with or without closing tags)
    """
    # Check for either closed or unclosed thinking blocks
    return bool(re.search(r'<think>', text, re.IGNORECASE))


def display_response_with_thinking(
    console: Console,
    response: str,
    show_thinking: bool = False,
    user_message: Optional[str] = None,
    elapsed_ms: Optional[float] = None
) -> None:
    """Display AI response with optional thinking blocks.

    Args:
        console: Rich console for output
        response: Raw AI response (may contain thinking blocks)
        show_thinking: Whether to show thinking blocks (default: False)
        user_message: Optional user message to show above response
        elapsed_ms: Optional inference time in milliseconds
    """
    # Handle None or empty response
    if response is None:
        console.print("[bold red]⚠️  Model returned None response[/bold red]")
        console.print("[dim]This usually indicates an error during inference.[/dim]")
        return

    # Handle empty or whitespace-only response
    if not response or not response.strip():
        console.print("[bold yellow]⚠️  Model returned empty response[/bold yellow]")
        console.print("[dim]The model generated no text. Try a different prompt or check model status.[/dim]")
        if elapsed_ms is not None:
            console.print(f"[dim]({elapsed_ms/1000:.2f}s)[/dim]")
        return

    # Extract thinking blocks
    thinking_blocks, clean_response = extract_thinking_blocks(response)

    # Display user message if provided
    if user_message:
        console.print(f"[bold green]You:[/bold green] {user_message}")

    # Display thinking blocks if requested
    if thinking_blocks and show_thinking:
        for i, thinking in enumerate(thinking_blocks, 1):
            console.print(Panel(
                thinking.strip(),
                title=f"[dim]🤔 Thinking {i}/{len(thinking_blocks)}[/dim]",
                border_style="dim",
                style="dim italic"
            ))

    # Display clean response (or warning if only thinking blocks exist)
    if clean_response.strip():
        console.print(f"[bold cyan]AI:[/bold cyan] {clean_response}")
    elif thinking_blocks:
        # Response was ONLY thinking blocks, no actual response
        console.print("[bold yellow]⚠️  Response contained only thinking blocks, no actual response[/bold yellow]")
        console.print("[dim]Model may be misconfigured or prompt needs adjustment.[/dim]")
    else:
        # Empty after extraction (shouldn't happen, but handle it)
        console.print("[bold yellow]⚠️  Model returned empty response after processing[/bold yellow]")

    # Show timing
    if elapsed_ms is not None:
        console.print(f"[dim]({elapsed_ms/1000:.2f}s)[/dim]", end="")

    # Show thinking hint if blocks exist but not shown
    if thinking_blocks and not show_thinking:
        console.print(f" [dim]| {len(thinking_blocks)} thinking block(s) hidden • Type 'show' to reveal[/dim]")
    else:
        console.print()


def prompt_show_thinking() -> bool:
    """Prompt user if they want to see thinking blocks.

    Returns:
        True if user wants to see thinking
    """
    from rich.prompt import Prompt
    response = Prompt.ask(
        "[dim]Show thinking blocks?[/dim]",
        choices=["yes", "no", "y", "n"],
        default="no"
    )
    return response.lower() in ["yes", "y"]
