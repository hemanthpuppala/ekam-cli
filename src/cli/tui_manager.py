"""Dynamic TUI manager for responsive terminal interface."""

import os
import shutil
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console()


class TUIManager:
    """Manage dynamic TUI with screen clearing and responsive layouts."""

    def __init__(self):
        """Initialize TUI manager."""
        self.console = Console()
        self.current_layout: Optional[Layout] = None
        self.live: Optional[Live] = None

    def clear_screen(self, show_header: bool = True) -> None:
        """Clear terminal screen completely (including scrollback) and optionally show permanent Ekam-CLI header.

        This method clears both the visible screen AND the scrollback buffer to ensure
        users cannot scroll up to see previous screens. Only the current screen content
        and the Ekam-CLI header should be visible.

        Args:
            show_header: If True, automatically displays Ekam-CLI header at top (default: True)
        """
        # Clear screen AND scrollback buffer to prevent users from scrolling back to previous screens
        # Use os.system for reliable clearing across all terminals
        if os.name != 'nt':
            # Unix/Linux/macOS: Use printf to clear screen and scrollback
            # \033c is a full terminal reset (clears screen + scrollback)
            os.system('printf "\\033c"')
        else:
            # Windows: cls clears both screen and scrollback
            os.system('cls')

        # Automatically show permanent Ekam-CLI header after clearing
        if show_header:
            terminal_width = self.console.width

            # Top separator
            self.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")

            # Main header - Ekam-CLI centered
            self.console.print("[bold cyan]" + "Ekam-CLI".center(terminal_width) + "[/bold cyan]")

            # Tagline - Multi-Provider AI Interface
            self.console.print("[dim cyan]" + "Multi-Provider AI Interface".center(terminal_width) + "[/dim cyan]")

            # Bottom separator
            self.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
            self.console.print()

    def get_terminal_size(self) -> tuple[int, int]:
        """Get current terminal dimensions.

        Returns:
            (width, height) in characters
        """
        size = shutil.get_terminal_size()
        return size.columns, size.lines

    def show_step_heading(self, heading: str, style: str = "bold cyan") -> None:
        """Show a step/pipeline heading with underline.

        The heading text is displayed with an underline of '═' characters
        matching the text length (not full terminal width).

        Args:
            heading: The heading text to display
            style: Rich style for the heading (default: "bold cyan")
        """
        # Display heading text
        self.console.print(f"[{style}]{heading}[/{style}]")

        # Calculate text length (strip ANSI/Rich formatting)
        # Use Rich's Text to measure the actual displayed width
        from rich.text import Text
        text_obj = Text.from_markup(f"[{style}]{heading}[/{style}]")
        text_length = len(text_obj.plain)

        # Display underline matching text length
        self.console.print(f"[{style}]" + "═" * text_length + f"[/{style}]")
        self.console.print()

    def show_panel(self, content, title: str = "", border_style: str = "blue") -> None:
        """Display content in a panel that fills the screen.

        Args:
            content: Content to display (can be Text, Table, or string)
            title: Panel title
            border_style: Border color style
        """
        self.clear_screen()

        width, _ = self.get_terminal_size()

        # Create panel with dynamic sizing
        panel = Panel(
            content,
            title=title,
            border_style=border_style,
            width=width,
            expand=True
        )

        self.console.print(panel)

    def show_layout(self, layout: Layout) -> None:
        """Display a Rich Layout that auto-adjusts to terminal size.

        Args:
            layout: Rich Layout object
        """
        self.clear_screen()
        self.console.print(layout)

    def start_live_display(self, renderable) -> Live:
        """Start a live-updating display.

        Args:
            renderable: Content to display (auto-updates)

        Returns:
            Live display context
        """
        self.clear_screen()
        self.live = Live(renderable, console=self.console, refresh_per_second=4)
        return self.live

    def stop_live_display(self) -> None:
        """Stop the current live display."""
        if self.live:
            self.live.stop()
            self.live = None

    def show_ekam_header(self, subtitle: Optional[str] = None) -> None:
        """Show persistent Ekam-CLI banner at the top of the screen.

        Args:
            subtitle: Optional subtitle text below the main header
        """
        terminal_width = self.console.width

        # Top separator
        self.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")

        # Main header - Ekam-CLI centered
        self.console.print("[bold cyan]" + "Ekam-CLI".center(terminal_width) + "[/bold cyan]")

        # Tagline - Multi-Provider AI Interface
        self.console.print("[dim cyan]" + "Multi-Provider AI Interface".center(terminal_width) + "[/dim cyan]")

        # Optional subtitle
        if subtitle:
            self.console.print("[dim]" + subtitle.center(terminal_width) + "[/dim]")

        # Bottom separator
        self.console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
        self.console.print()

    def prompt(self, message: str, style: str = "cyan") -> str:
        """Show a prompt and get user input using professional text input handler.

        Args:
            message: Prompt message
            style: Text style

        Returns:
            User input string
        """
        # Use professional prompt handler for proper escape sequence handling
        from .text_input import professional_prompt
        return professional_prompt.get_input(message, style=style, show_instructions=False)

    def prompt_in_box(self, message: str, title: str = "Input", style: str = "cyan") -> str:
        """Show a prompt in a box and get user input.

        Args:
            message: Prompt message
            title: Box title
            style: Text style

        Returns:
            User input string
        """
        width, _ = self.get_terminal_size()

        # Create prompt panel
        panel = Panel(
            f"[{style}]{message}[/{style}]",
            title=title,
            border_style="cyan",
            width=min(width - 4, 80)
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

        # Use professional prompt handler for consistent input handling
        from .text_input import professional_prompt
        return professional_prompt.get_input(f"[{style}]→[/{style}]", style=style, show_instructions=False)

    def show_error(self, message: str) -> None:
        """Display error message in a red panel.

        Args:
            message: Error message
        """
        width, _ = self.get_terminal_size()

        panel = Panel(
            f"[bold red]{message}[/bold red]",
            title="[bold red]Error[/bold red]",
            border_style="red",
            width=min(width - 4, 80)
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def show_message(self, message: str, title: str = "Message", style: str = "green") -> None:
        """Display message in a styled panel.

        Args:
            message: Message text
            title: Panel title
            style: Border style
        """
        width, _ = self.get_terminal_size()

        panel = Panel(
            message,
            title=f"[bold {style}]{title}[/bold {style}]",
            border_style=style,
            width=min(width - 4, 80)
        )

        self.console.print()
        self.console.print(panel)
        self.console.print()

    def create_centered_layout(self, content, title: str = "") -> Layout:
        """Create a centered layout with content.

        Args:
            content: Content to display
            title: Optional title

        Returns:
            Rich Layout object
        """
        layout = Layout()

        # Create 3-row layout: header, content, footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="content"),
            Layout(name="footer", size=3)
        )

        # Add title to header if provided
        if title:
            header_text = Text(title, style="bold cyan", justify="center")
            layout["header"].update(Panel(header_text, border_style="cyan"))
        else:
            layout["header"].update("")

        # Add content
        layout["content"].update(content)

        # Footer
        layout["footer"].update("")

        return layout

    def show_status_bar(self, status_text: str) -> None:
        """Show a persistent status bar at the bottom of the screen.

        Args:
            status_text: Status text to display
        """
        if not status_text:
            return

        width, height = self.get_terminal_size()

        # Move cursor to bottom, show status bar
        status_panel = Panel(
            status_text,
            border_style="yellow",
            width=width,
            expand=False
        )

        # Print status at bottom
        self.console.print(f"\n{status_panel}")

    def clear_and_show_with_status(self, content, status_text: str = "") -> None:
        """Clear screen, show content, and add status bar at bottom.

        Args:
            content: Content to display
            status_text: Status bar text (optional)
        """
        self.clear_screen()
        self.console.print(content)

        if status_text:
            self.show_status_bar(status_text)


# Global TUI instance
tui = TUIManager()
