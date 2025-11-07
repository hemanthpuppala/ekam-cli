"""
Reusable wizard flow controller for multi-step configuration.

Provides a base class for creating wizard-style configuration flows with:
- Forward/backward navigation between steps
- State preservation and pre-filling
- Step counter display
- Consistent navigation controls
"""

from typing import Optional, Any, Dict, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from loguru import logger
from rich.console import Console


console = Console()


@dataclass
class WizardStep:
    """Configuration for a single wizard step."""

    step_number: int
    total_steps: int
    title: str
    handler: Callable
    state_key: str  # Key to store result in state dict
    allow_back: bool = True  # Can user go back from this step?
    allow_skip: bool = False  # Can user skip this step?


class WizardController(ABC):
    """
    Base class for multi-step wizard flows.

    Handles navigation, state management, and provides consistent UX.
    Subclasses define the steps and implement result building.

    Example:
        class MyWizard(WizardController):
            def define_steps(self):
                return [
                    WizardStep(1, 3, "Select Type", self.step_type, "type"),
                    WizardStep(2, 3, "Configure", self.step_config, "config"),
                    WizardStep(3, 3, "Review", self.step_review, "review"),
                ]

            def build_result(self):
                return MyConfig(**self.state)

            def step_type(self, current_value):
                # Implementation returns (value, action) tuple
                # action can be: "next", "back", "cancel"
                return (selected_type, "next")
    """

    def __init__(self):
        """Initialize wizard controller."""
        self.state: Dict[str, Any] = {}
        self.steps = self.define_steps()
        self.current_step_index = 0

    @abstractmethod
    def define_steps(self) -> list[WizardStep]:
        """
        Define the wizard steps.

        Returns:
            List of WizardStep objects
        """
        pass

    @abstractmethod
    def build_result(self) -> Any:
        """
        Build final result from collected state.

        Returns:
            Configuration object or None if invalid
        """
        pass

    def run(self) -> Optional[Any]:
        """
        Execute the wizard flow.

        Returns:
            Result from build_result() or None if cancelled
        """
        logger.info(f"Starting wizard: {self.__class__.__name__}")

        try:
            while True:
                # Check if we've completed all steps
                if self.current_step_index >= len(self.steps):
                    logger.info("All steps completed, building result")
                    return self.build_result()

                # Get current step
                step = self.steps[self.current_step_index]
                logger.debug(f"Executing step {step.step_number}/{step.total_steps}: {step.title}")

                # Get current value for pre-filling (if step was visited before)
                current_value = self.state.get(step.state_key)

                # Execute step handler
                try:
                    result = step.handler(current_value)

                    # Handle different result formats
                    if isinstance(result, tuple):
                        value, action = result
                    else:
                        # If handler returns single value, assume "next"
                        value, action = result, "next"

                    # Process action
                    if action == "cancel" or value is None:
                        logger.info(f"User cancelled at step {step.step_number}")
                        return None

                    elif action == "back":
                        if self.current_step_index > 0 and step.allow_back:
                            logger.debug(f"Going back from step {step.step_number}")
                            self.current_step_index -= 1
                        else:
                            logger.warning(f"Cannot go back from step {step.step_number}")

                    elif action == "next":
                        # Store value in state
                        self.state[step.state_key] = value
                        logger.debug(f"Stored {step.state_key} = {value}")
                        self.current_step_index += 1

                    else:
                        logger.error(f"Unknown action: {action}")
                        return None

                except Exception as e:
                    logger.error(f"Error in step {step.step_number}: {str(e)}", exc_info=True)
                    console.print(f"\n[red]Error: {str(e)}[/red]")
                    return None

        except KeyboardInterrupt:
            logger.info("User interrupted wizard")
            console.print("\n[yellow]Configuration cancelled by user.[/yellow]")
            return None

    def reset(self):
        """Reset wizard state."""
        self.state.clear()
        self.current_step_index = 0
        logger.debug("Wizard state reset")

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from state with default."""
        return self.state.get(key, default)

    def set_state(self, key: str, value: Any):
        """Set value in state."""
        self.state[key] = value

    def show_step_header(self, step: WizardStep, subtitle: str = ""):
        """
        Show consistent step header with dynamic terminal width.

        Args:
            step: Current WizardStep
            subtitle: Optional subtitle text
        """
        terminal_width = console.width
        console.print()
        console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
        console.print(f"[bold cyan]Step {step.step_number}/{step.total_steps}: {step.title}[/bold cyan]")
        if subtitle:
            console.print(f"[dim]{subtitle}[/dim]")
        console.print("[bold cyan]" + "═" * terminal_width + "[/bold cyan]")
        console.print()

    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"step={self.current_step_index + 1}/{len(self.steps)}, "
                f"state_keys={list(self.state.keys())})")
