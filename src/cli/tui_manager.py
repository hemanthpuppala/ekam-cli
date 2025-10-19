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

    def clear_screen(self) -> None:
        """Clear terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')

    def get_terminal_size(self) -> tuple[int, int]:
        """Get current terminal dimensions.

        Returns:
            (width, height) in characters
        """
        size = shutil.get_terminal_size()
        return size.columns, size.lines

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

    def prompt(self, message: str, style: str = "cyan") -> str:
        """Show a prompt and get user input.

        Args:
            message: Prompt message
            style: Text style

        Returns:
            User input string
        """
        self.console.print(f"[{style}]{message}[/{style}]", end="")
        return input(" ")

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
        self.console.print(f"[{style}]→[/{style}] ", end="")
        return input()

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
