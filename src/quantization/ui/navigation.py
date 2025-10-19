"""Navigation state tracking for quantization pipeline."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class NavigationAction(Enum):
    """Navigation actions for workflow control."""

    BACK = "back"  # Go back one step
    MAIN_MENU = "main"  # Go to quantization main menu
    HOME = "home"  # Go to home menu (inference/quantization)
    CONTINUE = "continue"  # Continue to next step
    EXIT = "exit"  # Exit application


class NavigationException(Exception):
    """Exception raised for navigation actions (b/m/h hotkeys).

    This exception is used to control workflow navigation without
    treating navigation as an error condition.
    """

    def __init__(self, action: NavigationAction, message: str = ""):
        """Initialize navigation exception.

        Args:
            action: The navigation action requested
            message: Optional message explaining the action
        """
        self.action = action
        self.message = message or f"Navigation: {action.value}"
        super().__init__(self.message)


@dataclass
class NavigationStep:
    """Represents a step in the quantization workflow."""

    name: str  # Human-readable step name
    step_id: str  # Unique step identifier
    data: Any = None  # Step-specific data to restore state


class NavigationTracker:
    """Tracks navigation state in the quantization pipeline.

    This class maintains a stack of workflow steps to enable:
    - Going back one step (b hotkey)
    - Jumping to main menu (m hotkey)
    - Jumping to home menu (h hotkey)
    - Displaying breadcrumb trail

    Example:
        tracker = NavigationTracker()
        tracker.push_step("Model Selection", "select_model", model_data)
        tracker.push_step("Method Selection", "select_method", method_data)
        print(tracker.get_breadcrumb())  # "Model Selection → Method Selection"
    """

    def __init__(self):
        """Initialize navigation tracker with empty stack."""
        self.steps: list[NavigationStep] = []

    def push_step(self, name: str, step_id: str, data: Any = None) -> None:
        """Add a new step to the navigation stack.

        Args:
            name: Human-readable step name (e.g., "Model Selection")
            step_id: Unique identifier for this step (e.g., "select_model")
            data: Optional data to restore state when going back
        """
        self.steps.append(NavigationStep(name=name, step_id=step_id, data=data))

    def pop_step(self) -> Optional[NavigationStep]:
        """Remove and return the last step.

        Returns:
            The removed step, or None if stack is empty
        """
        if self.steps:
            return self.steps.pop()
        return None

    def get_current_step(self) -> Optional[NavigationStep]:
        """Get the current step without removing it.

        Returns:
            The current step, or None if stack is empty
        """
        if self.steps:
            return self.steps[-1]
        return None

    def get_previous_step(self) -> Optional[NavigationStep]:
        """Get the previous step without removing it.

        Returns:
            The previous step, or None if there is none
        """
        if len(self.steps) >= 2:
            return self.steps[-2]
        return None

    def clear(self) -> None:
        """Clear all navigation state."""
        self.steps.clear()

    def get_breadcrumb(self) -> str:
        """Get a breadcrumb trail of all steps.

        Returns:
            String like "Home → Model Selection → Method Selection"
        """
        if not self.steps:
            return "Home"
        return " → ".join(step.name for step in self.steps)

    def get_step_number(self) -> int:
        """Get the current step number (1-indexed).

        Returns:
            Current step number, or 0 if stack is empty
        """
        return len(self.steps)

    def can_go_back(self) -> bool:
        """Check if there's a previous step to go back to.

        Returns:
            True if there's at least one step in the stack
        """
        return len(self.steps) > 0


def check_navigation_input(user_input: str) -> Optional[NavigationAction]:
    """Check if user input is a navigation command (b/m/h).

    Args:
        user_input: The user's input string (stripped and lowercased)

    Returns:
        NavigationAction if input matches a navigation command, None otherwise
    """
    nav_map = {
        "b": NavigationAction.BACK,
        "back": NavigationAction.BACK,
        "m": NavigationAction.MAIN_MENU,
        "main": NavigationAction.MAIN_MENU,
        "menu": NavigationAction.MAIN_MENU,
        "h": NavigationAction.HOME,
        "home": NavigationAction.HOME,
    }
    return nav_map.get(user_input.strip().lower())


def format_navigation_help(show_back: bool = True, show_main: bool = True, show_home: bool = True) -> str:
    """Format navigation help text for prompts.

    Args:
        show_back: Whether to show 'b' for back
        show_main: Whether to show 'm' for main menu
        show_home: Whether to show 'h' for home

    Returns:
        Formatted navigation help string
    """
    nav_options = []
    if show_back:
        nav_options.append("[cyan bold]b[/cyan bold]=back")
    if show_main:
        nav_options.append("[cyan bold]m[/cyan bold]=main menu")
    if show_home:
        nav_options.append("[cyan bold]h[/cyan bold]=home")

    if nav_options:
        return f"[dim]Navigation: {' | '.join(nav_options)}[/dim]"
    return ""
