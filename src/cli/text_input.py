"""Professional text input system using prompt_toolkit for clean TUI."""

from typing import Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.patch_stdout import patch_stdout

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
            # Get input using prompt_toolkit
            with patch_stdout():
                user_input = self.session.prompt(
                    message + " ",
                    style=INPUT_STYLE,
                    multiline=allow_multiline,
                    wrap_lines=True,
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
    ) -> str:
        """Get multi-line text input.

        Args:
            prompt_msg: Prompt message
            style: Color style
            allow_multiline: Allow multiple lines
            allow_blank: Allow empty input

        Returns:
            Text string
        """
        while True:
            text = self.get_input(
                prompt_msg,
                style=style,
                allow_multiline=allow_multiline,
                show_instructions=True,
            )

            if text or allow_blank:
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
        style: str = "cyan",
    ) -> int:
        """Get numeric input with validation.

        Args:
            prompt_msg: Prompt message
            min_val: Minimum value
            max_val: Maximum value
            style: Color style

        Returns:
            Validated integer
        """
        while True:
            user_input = self.get_input(
                prompt_msg,
                style=style,
                show_instructions=False,
            )

            try:
                value = int(user_input)
                if min_val <= value <= max_val:
                    return value

                tui.show_error(
                    f"Please enter a number between {min_val} and {max_val}"
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
