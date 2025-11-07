"""Professional text input system using prompt_toolkit for clean TUI."""

from typing import Optional, List, Tuple
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.layout.containers import Window, HSplit
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from .tui_manager import tui


# Custom style for the input prompt
INPUT_STYLE = Style.from_dict({
    "prompt": "ansicyan",
    "question-mark": "ansicyan bold",
    "text": "ansiwhite",
    "instruction": "ansibrightblack",
})


class ProfessionalPrompt:
    """Professional text input handler with prompt_toolkit."""

    def __init__(self):
        """Initialize the prompt handler."""
        self.session = PromptSession(
            history=InMemoryHistory(),
            style=INPUT_STYLE,
        )
        self.slash_commands = {
            "exit": "Exit the application",
            "quit": "Exit the application",
            "back": "Go back",
            "help": "Show available commands",
            "/info": "Show model information",
            "/config": "Show configuration",
            "/status": "Show status",
            "/newimage": "Load a new image",
        }

    def get_input(
        self,
        message: str = "",
        style: str = "cyan",
        allow_multiline: bool = False,
        show_instructions: bool = True,
    ) -> str:
        """Get user input with professional formatting.

        Args:
            message: The prompt message
            style: Color style (cyan, green, yellow, etc.)
            allow_multiline: Allow multi-line input (Ctrl+J to newline)
            show_instructions: Show keyboard instructions

        Returns:
            User input string
        """
        # Build the prompt message
        styled_message = f"[{style}]{message}[/{style}]"

        # Show instructions if requested
        if show_instructions:
            instructions = (
                "Arrow Keys: Navigate | Up/Down: History | "
                "Enter: Submit | Ctrl+J: New line | Ctrl+A: Start | "
                "Ctrl+E: End | Ctrl+C: Cancel"
            )
            tui.console.print(f"[dim]✓ {instructions}[/dim]")
            tui.console.print()

        try:
            # Create custom key bindings for multiline input
            # Makes Enter submit and Ctrl+J create newlines
            kb = None
            if allow_multiline:
                kb = KeyBindings()

                @kb.add(Keys.ControlJ)
                def _(event):
                    """Ctrl+J inserts a newline in multiline mode."""
                    event.current_buffer.insert_text('\n')

                @kb.add(Keys.Enter)
                def _(event):
                    """Enter submits the input in multiline mode."""
                    event.current_buffer.validate_and_handle()

            # Get input using prompt_toolkit
            with patch_stdout():
                user_input = self.session.prompt(
                    message + " ",
                    style=INPUT_STYLE,
                    multiline=allow_multiline,
                    wrap_lines=True,
                    key_bindings=kb,
                )

            return user_input.strip()

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            tui.console.print("\n[yellow]Input cancelled[/yellow]")
            return ""
        except EOFError:
            # Handle Ctrl+D or input termination
            raise KeyboardInterrupt("User cancelled input")

    def get_question(self, style: str = "green") -> str:
        """Get question input for QA.

        Args:
            style: Color style

        Returns:
            Question string
        """
        return self.get_input(
            "Your question:",
            style=style,
            allow_multiline=True,
            show_instructions=True,
        )

    def get_text(
        self,
        prompt_msg: str = "Enter text:",
        style: str = "cyan",
        allow_multiline: bool = True,
        allow_blank: bool = False,
        show_char_count: bool = True,
    ) -> str:
        """Get multi-line text input with character count.

        Args:
            prompt_msg: Prompt message
            style: Color style
            allow_multiline: Allow multiple lines
            allow_blank: Allow empty input
            show_char_count: Show character and line count after input

        Returns:
            Text string
        """
        # Show multiline instructions if enabled
        if allow_multiline:
            tui.console.print("[dim]Press Ctrl+J for new line, Enter to submit[/dim]")
            tui.console.print()

        while True:
            text = self.get_input(
                prompt_msg,
                style=style,
                allow_multiline=allow_multiline,
                show_instructions=False,  # We show custom instructions above
            )

            if text or allow_blank:
                # Show character count if requested
                if text and show_char_count:
                    char_count = len(text)
                    line_count = text.count('\n') + 1 if text else 0
                    tui.console.print(f"[dim]✓ {char_count} characters, {line_count} line(s)[/dim]")
                return text

            tui.show_error("Please enter some text")

    def get_choice(
        self,
        prompt_msg: str,
        choices: list[str],
        style: str = "cyan",
    ) -> str:
        """Get choice from a list.

        Args:
            prompt_msg: Prompt message
            choices: List of valid choices
            style: Color style

        Returns:
            Selected choice
        """
        choices_lower = [c.lower() for c in choices]
        choice_str = "/".join(choices)

        while True:
            choice = self.get_input(
                f"{prompt_msg} [{choice_str}]:",
                style=style,
                show_instructions=False,
            )

            if choice.lower() in choices_lower:
                return choices[choices_lower.index(choice.lower())]

            tui.show_error(f"Please choose from: {', '.join(choices)}")

    def get_numeric(
        self,
        prompt_msg: str,
        min_val: int,
        max_val: int,
        default: Optional[int] = None,
        style: str = "cyan",
    ) -> int:
        """Get numeric input with validation and hints.

        Args:
            prompt_msg: Prompt message
            min_val: Minimum value
            max_val: Maximum value
            default: Default value if user presses Enter
            style: Color style

        Returns:
            Validated integer
        """
        # Show validation hints
        hint = f"[dim]Range: {min_val}-{max_val}"
        if default is not None:
            hint += f", Default: {default}"
        hint += "[/dim]"

        tui.console.print(hint)
        tui.console.print()

        while True:
            prompt = f"{prompt_msg} [{min_val}-{max_val}]"
            if default is not None:
                prompt += f" (default: {default})"
            prompt += ":"

            user_input = self.get_input(
                prompt,
                style=style,
                show_instructions=False,
            )

            # Handle empty input with default
            if not user_input and default is not None:
                tui.console.print(f"[dim]Using default: {default}[/dim]")
                return default

            try:
                value = int(user_input)
                if min_val <= value <= max_val:
                    return value

                tui.show_error(
                    f"Value must be between {min_val} and {max_val}"
                )
            except ValueError:
                tui.show_error("Please enter a valid number")

    def get_file_path(
        self,
        prompt_msg: str = "Enter file path:",
        style: str = "cyan",
    ) -> str:
        """Get file path input.

        Args:
            prompt_msg: Prompt message
            style: Color style

        Returns:
            File path string
        """
        return self.get_input(
            prompt_msg,
            style=style,
            allow_multiline=False,
            show_instructions=False,
        )

    def get_confirmation(
        self,
        prompt_msg: str = "Proceed?",
        default: bool = False,
        style: str = "cyan",
    ) -> bool:
        """Get yes/no confirmation.

        Args:
            prompt_msg: Prompt message
            default: Default value if empty
            style: Color style

        Returns:
            True for yes, False for no
        """
        default_str = "Y/n" if default else "y/N"
        response = self.get_input(
            f"{prompt_msg} [{default_str}]:",
            style=style,
            show_instructions=False,
        )

        if not response:
            return default

        return response.lower() in ["y", "yes"]

    def get_arrow_selection(
        self,
        options: List[Tuple[str, str, str]],
        title: str = "Select an option",
        instructions: str = "Use ↑/↓ arrows to navigate, Enter to select, q to quit",
    ) -> Optional[str]:
        """Get user selection using arrow keys for navigation.

        Args:
            options: List of tuples (value, label, description)
                     - value: returned when selected
                     - label: display name (colored)
                     - description: explanation text
            title: Menu title
            instructions: Help text for navigation

        Returns:
            Selected value or None if cancelled

        Example:
            options = [
                ("inference", "[green]Inference[/green]", "Run model inference"),
                ("quantization", "[yellow]Quantization[/yellow]", "Reduce model size"),
                ("quit", "Quit", "Exit application")
            ]
            selection = prompt.get_arrow_selection(options, "Main Menu")
        """
        selected_index = [0]  # Use list to allow modification in nested function
        result = [None]  # Store the result

        def get_formatted_menu():
            """Generate formatted menu text with current selection highlighted."""
            lines = []
            lines.append(f"[bold cyan]{title}[/bold cyan]\n")

            for idx, (value, label, description) in enumerate(options):
                if idx == selected_index[0]:
                    # Highlighted selection with pointer
                    lines.append(f"[black on cyan]  ▶ {label}[/black on cyan]")
                    lines.append(f"[dim cyan]    {description}[/dim cyan]")
                else:
                    # Normal option
                    lines.append(f"    {label}")
                    lines.append(f"[dim]    {description}[/dim]")

                if idx < len(options) - 1:
                    lines.append("")  # Add spacing between options

            lines.append(f"\n[dim]{instructions}[/dim]")
            return "\n".join(lines)

        # Create key bindings
        kb = KeyBindings()

        @kb.add(Keys.Up)
        def move_up(event):
            """Move selection up."""
            selected_index[0] = (selected_index[0] - 1) % len(options)
            # Refresh display
            event.app.invalidate()

        @kb.add(Keys.Down)
        def move_down(event):
            """Move selection down."""
            selected_index[0] = (selected_index[0] + 1) % len(options)
            # Refresh display
            event.app.invalidate()

        @kb.add(Keys.Enter)
        def select(event):
            """Select current option."""
            result[0] = options[selected_index[0]][0]
            event.app.exit()

        @kb.add('q')
        def quit_menu(event):
            """Quit menu."""
            result[0] = None
            event.app.exit()

        @kb.add(Keys.ControlC)
        def cancel(event):
            """Cancel with Ctrl+C."""
            result[0] = None
            event.app.exit()

        # Create the layout using FormattedTextControl that updates dynamically
        def get_formatted_text():
            """Get the current formatted text for display."""
            import html
            import re

            # Convert rich markup to prompt_toolkit HTML
            menu_text = get_formatted_menu()

            # First, escape HTML entities (but preserve our rich markup tags)
            # We'll temporarily replace rich tags with placeholders, escape, then restore

            # Find all rich markup tags
            rich_tag_pattern = r'\[/?[^\]]+\]'
            rich_tags = re.findall(rich_tag_pattern, menu_text)

            # Replace rich tags with placeholders (using __ to avoid HTML escaping)
            for i, tag in enumerate(rich_tags):
                menu_text = menu_text.replace(tag, f"__RICHTAG{i}__", 1)

            # Escape HTML entities in the remaining text
            menu_text = html.escape(menu_text)

            # Restore rich tags
            for i, tag in enumerate(rich_tags):
                menu_text = menu_text.replace(f"__RICHTAG{i}__", tag)

            # Now convert rich markup to HTML-like format for prompt_toolkit
            menu_text = menu_text.replace("[bold cyan]", "<b><style fg='cyan'>")
            menu_text = menu_text.replace("[/bold cyan]", "</style></b>")
            menu_text = menu_text.replace("[black on cyan]", "<style bg='cyan' fg='black'>")
            menu_text = menu_text.replace("[/black on cyan]", "</style>")
            menu_text = menu_text.replace("[dim cyan]", "<style fg='cyan'>")
            menu_text = menu_text.replace("[/dim cyan]", "</style>")
            menu_text = menu_text.replace("[dim]", "<style fg='#666666'>")
            menu_text = menu_text.replace("[/dim]", "</style>")
            menu_text = menu_text.replace("[green]", "<style fg='green'>")
            menu_text = menu_text.replace("[/green]", "</style>")
            menu_text = menu_text.replace("[yellow]", "<style fg='yellow'>")
            menu_text = menu_text.replace("[/yellow]", "</style>")
            menu_text = menu_text.replace("[magenta]", "<style fg='magenta'>")
            menu_text = menu_text.replace("[/magenta]", "</style>")
            menu_text = menu_text.replace("[blue]", "<style fg='blue'>")
            menu_text = menu_text.replace("[/blue]", "</style>")
            menu_text = menu_text.replace("[cyan]", "<style fg='cyan'>")
            menu_text = menu_text.replace("[/cyan]", "</style>")
            menu_text = menu_text.replace("[red]", "<style fg='red'>")
            menu_text = menu_text.replace("[/red]", "</style>")

            return HTML(menu_text)

        control = FormattedTextControl(
            text=get_formatted_text,
            focusable=True,
            show_cursor=False,
        )

        window = Window(content=control, wrap_lines=True)
        layout = Layout(window)

        # Create and run the application
        app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
        )

        try:
            app.run()
        except KeyboardInterrupt:
            return None

        return result[0]

    def get_incremental_search_selection(
        self,
        items: List[Tuple[any, str, str]],
        title: str = "Search and Select",
        instructions: str = "Type to search, ↑/↓ to navigate, Enter to select, Esc to cancel",
        max_display: int = 10,
    ) -> Optional[any]:
        """Get selection with incremental search (type to filter).

        Args:
            items: List of tuples (value, label, description)
                   - value: returned when selected
                   - label: display name (supports rich markup)
                   - description: explanation text
            title: Menu title
            instructions: Help text for navigation
            max_display: Maximum number of items to display at once

        Returns:
            Selected value or None if cancelled

        Example:
            items = [
                (model_obj, "llama3.2:3b", "3B params, 2.0GB"),
                (model_obj2, "llava:7b", "7B params, 4.5GB"),
            ]
            selection = prompt.get_incremental_search_selection(items, "Select Model")
        """
        search_term = [""]  # Use list to allow modification in nested function
        selected_index = [0]
        result = [None]
        filtered_items = [items]  # Store filtered results

        def update_filtered_items():
            """Update filtered items based on search term."""
            if not search_term[0]:
                filtered_items[0] = items
            else:
                term_lower = search_term[0].lower()
                filtered_items[0] = [
                    item for item in items
                    if term_lower in item[1].lower() or term_lower in item[2].lower()
                ]
            # Reset selection if out of bounds
            if selected_index[0] >= len(filtered_items[0]):
                selected_index[0] = max(0, len(filtered_items[0]) - 1)

        def get_formatted_menu():
            """Generate formatted menu text with current selection and search."""
            lines = []
            lines.append(f"[bold cyan]{title}[/bold cyan]\n")

            # Show search bar
            lines.append(f"[cyan]Search:[/cyan] {search_term[0]}█")
            lines.append(f"[dim]{len(filtered_items[0])} of {len(items)} items[/dim]\n")

            # Show filtered items
            display_items = filtered_items[0][:max_display]

            if not display_items:
                lines.append("[yellow]No matches found[/yellow]")
            else:
                for idx, (value, label, description) in enumerate(display_items):
                    actual_idx = filtered_items[0].index((value, label, description))

                    if actual_idx == selected_index[0]:
                        # Highlighted selection with pointer
                        lines.append(f"[black on cyan]  ▶ {label}[/black on cyan]")
                        lines.append(f"[dim cyan]    {description}[/dim cyan]")
                    else:
                        # Normal option
                        lines.append(f"    {label}")
                        lines.append(f"[dim]    {description}[/dim]")

                    if idx < len(display_items) - 1:
                        lines.append("")  # Add spacing

                # Show pagination info if needed
                if len(filtered_items[0]) > max_display:
                    lines.append(f"\n[dim]... {len(filtered_items[0]) - max_display} more items (scroll to see)[/dim]")

            lines.append(f"\n[dim]{instructions}[/dim]")
            return "\n".join(lines)

        # Create key bindings
        kb = KeyBindings()

        @kb.add(Keys.Up)
        def move_up(event):
            """Move selection up."""
            if filtered_items[0]:
                selected_index[0] = (selected_index[0] - 1) % len(filtered_items[0])
                event.app.invalidate()

        @kb.add(Keys.Down)
        def move_down(event):
            """Move selection down."""
            if filtered_items[0]:
                selected_index[0] = (selected_index[0] + 1) % len(filtered_items[0])
                event.app.invalidate()

        @kb.add(Keys.Enter)
        def select(event):
            """Select current option."""
            if filtered_items[0] and selected_index[0] < len(filtered_items[0]):
                result[0] = filtered_items[0][selected_index[0]][0]
                event.app.exit()

        @kb.add(Keys.Escape)
        def cancel(event):
            """Cancel with Escape."""
            result[0] = None
            event.app.exit()

        @kb.add(Keys.ControlC)
        def ctrl_c_cancel(event):
            """Cancel with Ctrl+C."""
            result[0] = None
            event.app.exit()

        @kb.add(Keys.Backspace)
        def handle_backspace(event):
            """Handle backspace to delete from search."""
            if search_term[0]:
                search_term[0] = search_term[0][:-1]
                update_filtered_items()
                event.app.invalidate()

        # Handle all printable characters for search
        @kb.add(Keys.Any)
        def handle_key(event):
            """Handle character input for search."""
            key = event.key_sequence[0].key
            if isinstance(key, str) and len(key) == 1 and key.isprintable():
                search_term[0] += key
                update_filtered_items()
                event.app.invalidate()

        # Initialize filtered items
        update_filtered_items()

        # Create the layout
        def get_formatted_text():
            """Get the current formatted text for display."""
            import html
            import re

            menu_text = get_formatted_menu()

            # Find all rich markup tags
            rich_tag_pattern = r'\[/?[^\]]+\]'
            rich_tags = re.findall(rich_tag_pattern, menu_text)

            # Replace rich tags with placeholders
            for i, tag in enumerate(rich_tags):
                menu_text = menu_text.replace(tag, f"__RICHTAG{i}__", 1)

            # Escape HTML entities
            menu_text = html.escape(menu_text)

            # Restore rich tags
            for i, tag in enumerate(rich_tags):
                menu_text = menu_text.replace(f"__RICHTAG{i}__", tag)

            # Convert rich markup to HTML
            menu_text = menu_text.replace("[bold cyan]", "<b><style fg='cyan'>")
            menu_text = menu_text.replace("[/bold cyan]", "</style></b>")
            menu_text = menu_text.replace("[black on cyan]", "<style bg='cyan' fg='black'>")
            menu_text = menu_text.replace("[/black on cyan]", "</style>")
            menu_text = menu_text.replace("[dim cyan]", "<style fg='cyan'>")
            menu_text = menu_text.replace("[/dim cyan]", "</style>")
            menu_text = menu_text.replace("[cyan]", "<style fg='cyan'>")
            menu_text = menu_text.replace("[/cyan]", "</style>")
            menu_text = menu_text.replace("[dim]", "<style fg='#666666'>")
            menu_text = menu_text.replace("[/dim]", "</style>")
            menu_text = menu_text.replace("[green]", "<style fg='green'>")
            menu_text = menu_text.replace("[/green]", "</style>")
            menu_text = menu_text.replace("[yellow]", "<style fg='yellow'>")
            menu_text = menu_text.replace("[/yellow]", "</style>")
            menu_text = menu_text.replace("[magenta]", "<style fg='magenta'>")
            menu_text = menu_text.replace("[/magenta]", "</style>")
            menu_text = menu_text.replace("[blue]", "<style fg='blue'>")
            menu_text = menu_text.replace("[/blue]", "</style>")
            menu_text = menu_text.replace("[red]", "<style fg='red'>")
            menu_text = menu_text.replace("[/red]", "</style>")

            return HTML(menu_text)

        control = FormattedTextControl(
            text=get_formatted_text,
            focusable=True,
            show_cursor=False,
        )

        window = Window(content=control, wrap_lines=True)
        layout = Layout(window)

        # Create and run the application
        app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
        )

        try:
            app.run()
        except KeyboardInterrupt:
            return None

        return result[0]

    @staticmethod
    def show_available_commands() -> None:
        """Display available slash commands."""
        tui.console.print("\n[bold cyan]Available Commands:[/bold cyan]")
        tui.console.print("  [cyan]/info[/cyan]     - Show model information")
        tui.console.print("  [cyan]/config[/cyan]   - Show configuration")
        tui.console.print("  [cyan]/status[/cyan]   - Show status")
        tui.console.print("  [cyan]/newimage[/cyan] - Load a new image")
        tui.console.print("  [cyan]/help[/cyan]     - Show this help")
        tui.console.print("  [cyan]/exit[/cyan]     - Exit the application\n")


# Global instance
professional_prompt = ProfessionalPrompt()
