"""Step-based workflow implementation for quantization with navigation support.

This module provides a refactored workflow that supports:
- Going back one step (b hotkey)
- Jumping to main menu (m hotkey)
- Jumping to home menu (h hotkey)
- Breadcrumb trail display
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional
from loguru import logger

from ...cli.tui_manager import tui
from ...services.model_discovery import ModelDiscoveryService
from ..manager import QuantizationManager
from ..models import QuantizationTask
from .navigation import NavigationAction, NavigationException, NavigationTracker
from .display import (
    ask_background_mode,
    ask_gpu_preference,
    ask_quantization_method,
    ask_vlm_components,
    ask_vlm_quantization_scope,
    confirm_quantization,
    select_model_to_quantize,
    select_quantization_type,
    show_quantization_intro,
)
from .enhanced_display import show_enhanced_live_progress


@dataclass
class WorkflowState:
    """Stores the state of the quantization workflow."""

    # Step 1-2: Discovery
    all_models: Optional[list] = None
    quantizable_models: Optional[list] = None

    # Step 3: Model selection
    selected_model: Optional[Any] = None

    # Step 4-5: VLM components (if applicable)
    vlm_scope: Optional[str] = None
    vlm_components: Optional[str] = None

    # Step 6: Method selection
    quant_method: Optional[str] = None

    # Step 7: GPU preference
    use_gpu: Optional[bool] = None

    # Step 8: Recommendations
    recommendations: Optional[list] = None

    # Step 9: Quantization type selection
    selected_rec: Optional[Any] = None

    # Step 10: Confirmation (boolean)
    confirmed: Optional[bool] = None

    # Step 11: Background mode
    run_background: Optional[bool] = None

    # Step 12-13: Task
    task: Optional[QuantizationTask] = None
    task_id: Optional[str] = None


class WorkflowStep:
    """Represents a single step in the workflow."""

    def __init__(
        self,
        step_id: str,
        name: str,
        execute_fn: Callable,
        skip_condition: Optional[Callable[[WorkflowState], bool]] = None,
    ):
        """Initialize workflow step.

        Args:
            step_id: Unique identifier for this step
            name: Human-readable step name for breadcrumbs
            execute_fn: Function to execute this step (takes state, returns result)
            skip_condition: Optional function to check if step should be skipped
        """
        self.step_id = step_id
        self.name = name
        self.execute_fn = execute_fn
        self.skip_condition = skip_condition

    def should_skip(self, state: WorkflowState) -> bool:
        """Check if this step should be skipped."""
        if self.skip_condition:
            return self.skip_condition(state)
        return False


def run_step_based_workflow(
    model_discovery: ModelDiscoveryService,
    quantization_manager: QuantizationManager,
) -> Optional[NavigationAction]:
    """Run quantization workflow with step-by-step navigation support.

    Returns:
        NavigationAction.HOME if user wants to go home, None otherwise
    """
    state = WorkflowState()
    tracker = NavigationTracker()

    # Define all workflow steps
    steps = _create_workflow_steps(model_discovery, quantization_manager, state)

    current_step_index = 0

    while current_step_index < len(steps):
        step = steps[current_step_index]

        # Check if step should be skipped
        if step.should_skip(state):
            logger.debug(f"Skipping step: {step.name}")
            current_step_index += 1
            continue

        # Show breadcrumb (if not first step)
        if current_step_index > 0:
            tui.console.print(f"\n[dim]{tracker.get_breadcrumb()}[/dim]\n", style="dim")

        try:
            # Execute step
            logger.info(f"Executing step {current_step_index + 1}/{len(steps)}: {step.name}")
            tracker.push_step(step.name, step.step_id)

            result = step.execute_fn(state)

            # If result is None and not a completion step, user cancelled
            if result is None and step.step_id not in ["execute_task"]:
                # Default behavior: go back one step
                logger.info(f"User cancelled at step: {step.name}")
                if current_step_index > 0:
                    tracker.pop_step()  # Remove current step
                    current_step_index -= 1  # Go back
                    logger.info(f"Going back to step: {steps[current_step_index].name}")
                else:
                    # At first step, exit workflow
                    return None
            else:
                # Success, move to next step
                current_step_index += 1

        except NavigationException as e:
            tracker.pop_step()  # Remove current step from tracker

            if e.action == NavigationAction.BACK:
                # Go back one step
                if current_step_index > 0:
                    current_step_index -= 1
                    logger.info(f"Navigation: Going back to step {current_step_index + 1}: {steps[current_step_index].name}")
                else:
                    # Already at first step, exit workflow
                    logger.info("Navigation: At first step, exiting workflow")
                    return None

            elif e.action == NavigationAction.MAIN_MENU:
                # Exit to main menu
                logger.info("Navigation: Returning to main menu")
                return None

            elif e.action == NavigationAction.HOME:
                # Exit to home
                logger.info("Navigation: Returning to home")
                return NavigationAction.HOME

        except Exception as e:
            logger.error(f"Error in workflow step {step.name}: {e}", exc_info=True)
            tui.show_error(f"Unexpected error: {str(e)}")
            tui.prompt("Press Enter to continue...", style="dim")
            return None

    # Workflow completed successfully
    logger.info("Quantization workflow completed")
    return None


def _create_workflow_steps(
    model_discovery: ModelDiscoveryService,
    quantization_manager: QuantizationManager,
    state: WorkflowState,
) -> list[WorkflowStep]:
    """Create the list of workflow steps.

    Args:
        model_discovery: Model discovery service
        quantization_manager: Quantization manager
        state: Workflow state to populate

    Returns:
        List of WorkflowStep objects
    """
    from ...models.model import ModelType

    def step_intro(s: WorkflowState):
        show_quantization_intro()
        return True

    def step_discover(s: WorkflowState):
        s.all_models = model_discovery.discover_all_models()
        s.quantizable_models = quantization_manager.get_quantizable_models(s.all_models)
        return True

    def step_select_model(s: WorkflowState):
        s.selected_model = select_model_to_quantize(s.quantizable_models, quantization_manager)
        return s.selected_model

    def step_vlm_scope(s: WorkflowState):
        from ...cli.prompts import UserExitException
        from .navigation import NavigationException
        try:
            # Pass quant_method for MLX warning logic
            s.vlm_scope = ask_vlm_quantization_scope(s.selected_model, s.quant_method)
            return s.vlm_scope
        except NavigationException:
            # Let navigation exceptions bubble up
            raise
        except UserExitException:
            # Convert 'b' back to None (go back one step)
            return None

    def step_vlm_components(s: WorkflowState):
        from ...cli.prompts import UserExitException
        from .navigation import NavigationException
        try:
            if s.vlm_scope == "component_level":
                # Pass quant_method for MLX forcing logic
                s.vlm_components = ask_vlm_components(s.selected_model, s.quant_method)
            else:
                s.vlm_components = "both"
            return s.vlm_components
        except NavigationException:
            raise
        except UserExitException:
            return None

    def step_method(s: WorkflowState):
        from ...cli.prompts import UserExitException
        from .navigation import NavigationException
        try:
            s.quant_method = ask_quantization_method(s.selected_model)
            return s.quant_method
        except NavigationException:
            raise
        except UserExitException:
            return None

    def step_gpu(s: WorkflowState):
        from ...cli.prompts import UserExitException
        from .navigation import NavigationException
        try:
            s.use_gpu = ask_gpu_preference()
            return s.use_gpu
        except NavigationException:
            raise
        except UserExitException:
            return None

    def step_recommendations(s: WorkflowState):
        s.recommendations = quantization_manager.get_recommendations(
            s.selected_model,
            use_gpu=s.use_gpu,
            method=s.quant_method,
        )
        if not s.recommendations:
            tui.show_error("No quantization options available for this model.")
            tui.prompt("Press Enter to continue...", style="dim")
            return None
        return s.recommendations

    def step_select_type(s: WorkflowState):
        from ...cli.prompts import UserExitException
        from .navigation import NavigationException
        try:
            s.selected_rec = select_quantization_type(s.recommendations)
            return s.selected_rec
        except NavigationException:
            raise
        except UserExitException:
            return None

    def step_confirm(s: WorkflowState):
        from ...cli.prompts import UserExitException
        from .navigation import NavigationException
        try:
            s.confirmed = confirm_quantization(s.selected_model, s.selected_rec)
            return s.confirmed
        except NavigationException:
            raise
        except UserExitException:
            return None

    def step_background(s: WorkflowState):
        from ...cli.prompts import UserExitException
        from .navigation import NavigationException
        try:
            s.run_background = ask_background_mode()
            return s.run_background
        except NavigationException:
            raise
        except UserExitException:
            return None

    def step_create_task(s: WorkflowState):
        s.task = quantization_manager.create_task(
            model_info=s.selected_model,
            quant_type=s.selected_rec.quant_type,
            module=s.selected_rec.module,
            use_gpu=s.use_gpu,
            background=s.run_background,
            vlm_components=s.vlm_components,
        )
        return s.task

    def step_execute_task(s: WorkflowState):
        s.task_id = quantization_manager.submit_task(s.task)
        logger.info(f"Submitted quantization task: {s.task_id}")

        if s.run_background:
            # Background mode - show confirmation
            tui.clear_screen()
            tui.show_message(
                f"""[bold green]✓ Quantization Started in Background[/bold green]

[bold]Task ID:[/bold] {s.task_id}
[bold]Model:[/bold] {s.selected_model.name}
[bold]Type:[/bold] {s.selected_rec.quant_type.display_name}

A status bar will appear at the bottom of all screens showing progress.

[bold]Commands:[/bold]
  • Type [yellow]/background[/yellow] to monitor progress
  • Type [yellow]/cancel {s.task_id}[/yellow] to cancel

[dim]Estimated completion: {s.selected_rec.estimated_time_minutes:.0f} minutes[/dim]""",
                title="Background Job Started",
                style="green",
            )
            tui.prompt("Press Enter to return to menu...", style="dim")
        else:
            # Live mode - show enhanced real-time progress
            show_enhanced_live_progress(s.task, use_gpu=s.use_gpu)

        return True

    # Build step list
    return [
        WorkflowStep("intro", "Introduction", step_intro),
        WorkflowStep("discover", "Model Discovery", step_discover),
        WorkflowStep("select_model", "Model Selection", step_select_model),
        WorkflowStep(
            "vlm_scope",
            "VLM Scope Selection",
            step_vlm_scope,
            skip_condition=lambda s: s.selected_model.model_type != ModelType.VLM,
        ),
        WorkflowStep(
            "vlm_components",
            "VLM Component Selection",
            step_vlm_components,
            skip_condition=lambda s: s.selected_model.model_type != ModelType.VLM,
        ),
        WorkflowStep("method", "Method Selection", step_method),
        WorkflowStep("gpu", "GPU Preference", step_gpu),
        WorkflowStep("recommendations", "Get Recommendations", step_recommendations),
        WorkflowStep("select_type", "Quantization Type", step_select_type),
        WorkflowStep("confirm", "Confirmation", step_confirm),
        WorkflowStep("background", "Processing Mode", step_background),
        WorkflowStep("create_task", "Create Task", step_create_task),
        WorkflowStep("execute_task", "Execute Task", step_execute_task),
    ]
