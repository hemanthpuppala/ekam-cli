"""Main CLI application entry point with dynamic TUI."""

import atexit
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from ..cli.display import display_conversation_history, display_statistics
from ..cli.menus import (
    EndpointMenu,
    LoadingScreen,
    ModelSelectionMenu,
    ProviderSelectionMenu,
    SystemSpecsScreen,
    WelcomeScreen,
)
from ..cli.diagnostics import SystemDiagnostics
from ..cli.prompts import (
    UserExitException,
    confirm_deletion,
    confirm_installation,
    prompt_enable_history,
    prompt_image_path,
    prompt_model_name,
    prompt_question,
    prompt_text_input,
    prompt_yes_no,
    select_or_create_session,
)
from ..cli.tui_manager import tui
from ..models.endpoints import EndpointType, ProviderType
from ..models.inference import InferenceInput, InferenceOutput, InferenceResult
from ..models.system import SystemSpecs
from ..providers.gguf import GGUFProvider
from ..providers.huggingface import HuggingFaceProvider
from ..providers.ollama import OllamaProvider
from ..providers.quantized import QuantizedProvider
from ..services.session import SessionManager
from ..utils.ollama_manager import ensure_ollama_running
from .config_loader import load_config
from .logging_setup import setup_logging

# Global session manager for cleanup on exit/signals
_global_session_manager: Optional[SessionManager] = None

# Global quantization manager for /background command and notifications
_global_quantization_manager = None

# Global notification tracker for completed jobs
_completed_task_notifications = set()  # Set of task_ids that have been notified


def check_and_show_completion_notifications() -> None:
    """Check for completed quantization jobs and show notifications."""
    global _global_quantization_manager, _completed_task_notifications

    if not _global_quantization_manager:
        return

    all_tasks = _global_quantization_manager.get_all_tasks()

    for task in all_tasks:
        # Only notify for tasks that just completed and haven't been notified yet
        if task.is_finished and task.task_id not in _completed_task_notifications:
            _completed_task_notifications.add(task.task_id)

            # Show notification
            if task.status.value == "completed":
                tui.console.print(
                    f"\n[bold green]🎉 NOTIFICATION: Quantization Complete![/bold green]",
                )
                tui.console.print(
                    f"[green]✓ {task.model_info.name} → {task.quant_type.display_name}[/green]"
                )
                tui.console.print(f"[dim]Output: {task.output_path.name}[/dim]\n")
            elif task.status.value == "failed":
                tui.console.print(
                    f"\n[bold red]❌ NOTIFICATION: Quantization Failed[/bold red]",
                )
                tui.console.print(f"[red]✗ {task.model_info.name}[/red]")
                tui.console.print(f"[dim]Error: {task.error}[/dim]\n")


def handle_background_command() -> bool:
    """Handle /background command if background jobs exist.

    Returns:
        True if command was handled, False if no background jobs
    """
    global _global_quantization_manager

    if not _global_quantization_manager:
        return False

    if not _global_quantization_manager.has_active_jobs():
        tui.show_error(
            "No background quantization jobs are currently running.\n\n"
            "Start a quantization in background mode to use this command."
        )
        tui.prompt("Press Enter to continue...", style="dim")
        return True

    # Show background jobs monitor
    from ..quantization.ui.workflow import show_background_jobs_monitor
    show_background_jobs_monitor(_global_quantization_manager)
    return True


def cleanup_on_exit() -> None:
    """Clean up resources on application exit.

    This function is called:
    - On normal exit (via atexit)
    - On Ctrl+C (SIGINT)
    - On Ctrl+Z followed by kill (SIGTSTP -> SIGCONT -> exit)
    - On termination signal (SIGTERM)
    """
    global _global_session_manager

    if _global_session_manager is None:
        return

    try:
        if _global_session_manager.state.loaded_model:
            model_id = _global_session_manager.state.loaded_model.model_info.model_id
            logger.info(f"Cleaning up: Unloading model {model_id}")
            _global_session_manager.unload_current_model()
            logger.info(f"Successfully unloaded model {model_id}")
        else:
            logger.debug("No model loaded, cleanup not needed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def signal_handler(signum: int, frame) -> None:
    """Handle termination signals (SIGINT, SIGTERM, SIGTSTP).

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    signal_names = {
        signal.SIGINT: "SIGINT (Ctrl+C)",
        signal.SIGTERM: "SIGTERM",
        signal.SIGTSTP: "SIGTSTP (Ctrl+Z)",
    }

    signal_name = signal_names.get(signum, f"Signal {signum}")
    logger.info(f"Received {signal_name}, cleaning up...")

    # Cleanup models
    cleanup_on_exit()

    # Show user message for interactive signals
    if signum == signal.SIGINT:
        tui.clear_screen()
        tui.show_message(
            "Application interrupted by user (Ctrl+C).\nModels unloaded.",
            title="Interrupted",
            style="yellow"
        )
    elif signum == signal.SIGTSTP:
        tui.clear_screen()
        tui.show_message(
            "Application suspended (Ctrl+Z).\nModels unloaded.",
            title="Suspended",
            style="yellow"
        )
        # Actually suspend the process after cleanup
        signal.signal(signal.SIGTSTP, signal.SIG_DFL)
        signal.raise_signal(signal.SIGTSTP)

    sys.exit(0)


def setup_signal_handlers() -> None:
    """Register signal handlers for clean shutdown."""
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command

    # Ctrl+Z handling (only on Unix-like systems)
    if hasattr(signal, 'SIGTSTP'):
        signal.signal(signal.SIGTSTP, signal_handler)

    # Register cleanup on normal exit
    atexit.register(cleanup_on_exit)

    logger.info("Signal handlers registered (SIGINT, SIGTERM, SIGTSTP)")


def run_endpoint_workflow(session_manager: SessionManager, model_info: "ModelInfo") -> Optional[str]:
    """Run endpoint workflow for VLM/LLM inference.

    Args:
        session_manager: Session manager with loaded model
        model_info: Information about the loaded model

    Returns:
        "QUIT" to quit app, "HOME" to return to main menu, None for normal back
    """
    from ..models.model import ModelInfo

    while True:
        # Show endpoint menu
        endpoint = EndpointMenu.show(model_info.name, str(model_info.model_type))

        if endpoint == "QUIT":
            # User quit - propagate signal
            logger.info("User initiated quit from endpoint menu")
            return "QUIT"

        if endpoint == "BACK":
            # Go back to model selection
            logger.info("User pressed b from endpoint menu - going back to model selection")
            return None

        if endpoint == "STATS":
            # Display statistics
            tui.clear_screen()
            display_statistics(session_manager.state.statistics)
            tui.prompt("Press Enter to continue...", style="dim")
            continue

        if endpoint == "SWITCH":
            # Switch to different model (same as BACK)
            logger.info("User chose to switch model")
            return None

        if endpoint == "HOME":
            # Return to main menu
            logger.info("User chose to return to main menu (home)")
            return "HOME"

        # Handle different endpoints
        if endpoint == "qa":
            run_qa_endpoint(session_manager, model_info)
        elif endpoint == "caption":
            run_caption_endpoint(session_manager, model_info)
        elif endpoint == "detect":
            run_detect_endpoint(session_manager, model_info)
        elif endpoint == "point":
            run_point_endpoint(session_manager, model_info)
        elif endpoint == "chat":
            run_chat_endpoint(session_manager, model_info)
        else:
            tui.show_error(f"Endpoint '{endpoint}' not yet implemented")
            tui.prompt("Press Enter to continue...", style="dim")


# Import QA endpoint from modular implementation
from .app_qa import run_qa_endpoint


# Import Caption endpoint from modular implementation
from .app_caption import run_caption_endpoint


def run_detect_endpoint(session_manager: SessionManager, model_info: "ModelInfo") -> None:
    """Run Object Detection endpoint.

    Args:
        session_manager: Session manager with loaded model
        model_info: Model information
    """
    tui.clear_screen()
    tui.show_panel(
        "[bold cyan]Object Detection Endpoint[/bold cyan]\n\n"
        "This endpoint detects objects in an image.",
        title="Detect Endpoint",
        border_style="cyan"
    )

    try:
        # Get image path
        image_path = prompt_image_path()

        # Get object to detect (optional)
        object_name = prompt_text_input(
            "Enter object to detect (or leave blank for all objects):",
            allow_blank=True
        )
    except UserExitException:
        logger.info("User exited from detect endpoint prompts")
        return

    # Run inference
    LoadingScreen.show(f"Detecting objects with {model_info.name}...")

    try:
        start_time = time.perf_counter()

        # Call the provider's detect endpoint
        provider = session_manager.model_discovery.get_provider(ProviderType(model_info.provider))
        if provider is None:
            raise ValueError(f"Provider {model_info.provider} not registered")

        # Load image with PIL for all providers
        from PIL import Image
        image = Image.open(image_path)
        original_width, original_height = image.size

        # Use appropriate method based on provider type to get structured detections
        provider_type_str = str(model_info.provider).lower()
        if provider_type_str == "ollama":
            # Ollama: Call run_detect directly to get structured data
            detections = provider.run_detect(model_info.model_id, image, object_name or "all objects")
        else:
            # HF and GGUF use handle-based approach
            if not session_manager.state.loaded_model:
                raise RuntimeError("No model loaded")
            model_handle = session_manager.state.loaded_model._handle
            detections = provider.run_detect(model_handle, image, object_name or "all objects")

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000

        # Save annotated image with bounding boxes
        from pathlib import Path as PathLib
        from ..utils.image import save_annotated_image

        results_dir = PathLib("results/detections")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotated_filename = f"detect_{timestamp}.png"
        annotated_path = results_dir / annotated_filename

        # Include original image dimensions for proper scaling
        saved_image_path = save_annotated_image(
            image,
            detections,
            annotated_path,
            original_size=(original_width, original_height)
        )
        logger.info(f"Saved annotated image to: {saved_image_path}")

        # Format text response for display
        if detections:
            text_response = f"Detected {len(detections)} instance(s):\n\n"
            for i, det in enumerate(detections, 1):
                text_response += f"{i}. {det.get('label', 'unknown')}\n"
                if 'bbox' in det:
                    bbox = det['bbox']
                    text_response += f"   BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n"
                if 'confidence' in det:
                    text_response += f"   Confidence: {det['confidence']:.2f}\n"
                if 'raw' in det:
                    text_response += f"   Raw: {det.get('raw', 'N/A')}\n"
        else:
            text_response = "No objects detected."

        # Display results
        tui.clear_screen()
        tui.show_panel(
            f"[bold green]✓ Objects Detected[/bold green]\n\n"
            f"[bold]Target:[/bold] {object_name if object_name else 'All objects'}\n\n"
            f"[bold]Results:[/bold]\n{text_response}\n\n"
            f"[bold]Annotated Image:[/bold] {saved_image_path}\n\n"
            f"[dim]Time: {elapsed_ms/1000:.2f}s | Model: {model_info.name}[/dim]",
            title="Detection Results",
            border_style="green"
        )

        logger.info(f"Detection inference completed in {elapsed_ms:.2f}ms")

        # Record statistics with annotated image path
        inference_result = InferenceResult(
            model_id=model_info.model_id,
            provider=model_info.provider,
            device=session_manager.state.loaded_model.device,
            input=InferenceInput(endpoint=EndpointType.DETECT, object_name=object_name, image_path=str(image_path)),
            output=InferenceOutput(
                text_response=text_response,
                detections=detections,
                annotated_image_path=str(saved_image_path)
            ),
            inference_time_ms=elapsed_ms,
        )
        session_manager.state.statistics.record_inference(inference_result)

    except Exception as e:
        logger.error(f"Detection inference failed: {e}")
        tui.show_error(f"Inference failed: {e}")

    tui.prompt("Press Enter to continue...", style="dim")


def run_point_endpoint(session_manager: SessionManager, model_info: "ModelInfo") -> None:
    """Run Object Pointing endpoint.

    Args:
        session_manager: Session manager with loaded model
        model_info: Model information
    """
    tui.clear_screen()
    tui.show_panel(
        "[bold cyan]Object Pointing Endpoint[/bold cyan]\n\n"
        "This endpoint finds the location of objects in an image.",
        title="Point Endpoint",
        border_style="cyan"
    )

    try:
        # Get image path
        image_path = prompt_image_path()

        # Get object to point to
        object_name = prompt_text_input("Enter object to locate:")
    except UserExitException:
        logger.info("User exited from point endpoint prompts")
        return

    # Run inference
    LoadingScreen.show(f"Locating objects with {model_info.name}...")

    try:
        start_time = time.perf_counter()

        # Call the provider's point endpoint
        provider = session_manager.model_discovery.get_provider(ProviderType(model_info.provider))
        if provider is None:
            raise ValueError(f"Provider {model_info.provider} not registered")

        # Load image with PIL for all providers
        from PIL import Image
        image = Image.open(image_path)
        original_width, original_height = image.size

        # Use appropriate method based on provider type to get structured coordinates
        provider_type_str = str(model_info.provider).lower()
        if provider_type_str == "ollama":
            # Ollama: Call run_point directly to get structured data
            coordinates = provider.run_point(model_info.model_id, image, object_name)
        else:
            # HF and GGUF use handle-based approach
            if not session_manager.state.loaded_model:
                raise RuntimeError("No model loaded")
            model_handle = session_manager.state.loaded_model._handle
            coordinates = provider.run_point(model_handle, image, object_name)

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000

        # Save annotated image with crosshair/point marker
        from pathlib import Path as PathLib
        from ..utils.image import save_annotated_image

        results_dir = PathLib("results/pointing")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotated_filename = f"point_{timestamp}.png"
        annotated_path = results_dir / annotated_filename

        # Convert coordinates to detection format for save_annotated_image
        point_detection = [{"coordinates": coordinates}]
        # Include original image dimensions for proper scaling
        saved_image_path = save_annotated_image(
            image,
            point_detection,
            annotated_path,
            original_size=(original_width, original_height)
        )
        logger.info(f"Saved annotated image to: {saved_image_path}")

        # Format text response for display
        x = coordinates.get("x", 0)
        y = coordinates.get("y", 0)
        raw = coordinates.get("raw", "")

        text_response = f"Object location: ({x}, {y})"
        if raw:
            text_response += f"\n\nModel response:\n{raw}"

        # Display results
        tui.clear_screen()
        tui.show_panel(
            f"[bold green]✓ Object Located[/bold green]\n\n"
            f"[bold]Target:[/bold] {object_name}\n\n"
            f"[bold]Location:[/bold]\n{text_response}\n\n"
            f"[bold]Annotated Image:[/bold] {saved_image_path}\n\n"
            f"[dim]Time: {elapsed_ms/1000:.2f}s | Model: {model_info.name}[/dim]",
            title="Pointing Results",
            border_style="green"
        )

        logger.info(f"Pointing inference completed in {elapsed_ms:.2f}ms")

        # Record statistics with annotated image path
        inference_result = InferenceResult(
            model_id=model_info.model_id,
            provider=model_info.provider,
            device=session_manager.state.loaded_model.device,
            input=InferenceInput(endpoint=EndpointType.POINT, object_name=object_name, image_path=str(image_path)),
            output=InferenceOutput(
                text_response=text_response,
                coordinates=coordinates,
                annotated_image_path=str(saved_image_path)
            ),
            inference_time_ms=elapsed_ms,
        )
        session_manager.state.statistics.record_inference(inference_result)

    except Exception as e:
        logger.error(f"Pointing inference failed: {e}")
        tui.show_error(f"Inference failed: {e}")

    tui.prompt("Press Enter to continue...", style="dim")


# Import chat endpoint from modular implementation
from .app_chat import run_chat_endpoint

# Import quantization integration
from .app_quantization import show_quantization_or_inference_menu, run_quantization_mode, run_finetuning_mode


def run_install_workflow(
    session_manager: SessionManager,
    provider_type: ProviderType,
    system_specs: SystemSpecs
) -> bool:
    """Run model installation workflow.

    Args:
        session_manager: Session manager instance
        provider_type: Provider to install model for
        system_specs: System specifications for compatibility check

    Returns:
        True if model was successfully installed
    """
    from ..cli.display import show_download_progress

    tui.clear_screen()

    # Get provider instance
    # Special handling: GGUF, MLX, OpenVINO, and Quantized providers don't support direct installation
    # They rely on HuggingFace to download models first
    install_provider_type = provider_type
    needs_hf_installation = provider_type in [
        ProviderType.GGUF,
        ProviderType.MLX,
        ProviderType.OPENVINO,
        ProviderType.QUANTIZED
    ]

    if needs_hf_installation:
        # Explain to user what will happen
        provider_name = provider_type.value.upper()
        tui.console.print()
        tui.console.print(f"[yellow]ℹ Note:[/yellow] {provider_name} models must be installed via HuggingFace Hub first.")
        tui.console.print(f"[dim]After installation, the model will appear in the HuggingFace provider list.[/dim]")
        tui.console.print()

        # Ask for confirmation
        from ..cli.prompts import prompt_yes_no
        if not prompt_yes_no(
            "Install this model using HuggingFace Hub?",
            default=True
        ):
            logger.info(f"User cancelled HuggingFace installation for {provider_name}")
            tui.show_message("Installation cancelled.", title="Cancelled", style="yellow")
            tui.prompt("Press Enter to continue...", style="dim")
            return False

        # Use HuggingFace provider for installation
        install_provider_type = ProviderType.HUGGINGFACE
        logger.info(f"Using HuggingFace provider to install model for {provider_name}")

    logger.info(f"Getting provider for installation: {install_provider_type.value}")
    provider = session_manager.model_discovery.get_provider(install_provider_type)
    logger.info(f"Got provider: {type(provider).__name__}")
    if provider is None:
        tui.show_error(f"Provider {install_provider_type.value} not registered")
        return False

    # Prompt for model name with examples (use original provider for examples)
    try:
        model_name = prompt_model_name(provider_type.value)
    except UserExitException:
        logger.info("User cancelled model installation")
        return False

    # Estimate model size
    try:
        estimated_size = provider.estimate_model_size(model_name)
        logger.info(f"Estimated size for {model_name}: {estimated_size:.1f} GB")
    except Exception as e:
        logger.warning(f"Could not estimate size for {model_name}: {e}")
        estimated_size = 4.0  # Default estimate

    # Assess compatibility
    memory_percentage = (estimated_size / system_specs.recommended_model_size_gb) * 100

    if memory_percentage > 100:
        compatibility = "too_large"
    elif memory_percentage > 70:
        compatibility = "tight_fit"
    else:
        compatibility = "perfect_fit"

    # Confirm installation with user
    if not confirm_installation(
        model_name,
        estimated_size,
        compatibility,
        system_specs.available_ram_gb
    ):
        logger.info(f"User cancelled installation of {model_name}")
        tui.show_message("Installation cancelled.", title="Cancelled", style="yellow")
        tui.prompt("Press Enter to continue...", style="dim")
        return False

    # Install model with progress bar
    tui.clear_screen()
    tui.console.print(f"\n[bold cyan]Installing {model_name}...[/bold cyan]\n")

    # Create progress bar
    progress, task_id = show_download_progress()

    install_success = False
    last_status = ""

    def progress_callback(status: str, completed: int, total: int):
        """Update progress bar during installation."""
        nonlocal last_status

        # Update task description if status changed
        if status != last_status:
            progress.update(task_id, description=f"[bold blue]{status}")
            last_status = status

        # Update progress if we have total
        if total > 0:
            percentage = (completed / total) * 100
            progress.update(task_id, completed=completed, total=total)

    try:
        progress.start()
        install_success = provider.install_model(model_name, progress_callback=progress_callback)
        progress.stop()

        if install_success:
            tui.console.print()
            tui.show_message(
                f"✓ Successfully installed {model_name}!",
                title="Installation Complete",
                style="green"
            )
            logger.info(f"Successfully installed {model_name}")

            # If installed via HuggingFace for another provider, inform user
            if needs_hf_installation:
                tui.console.print()
                tui.console.print(f"[yellow]ℹ Important:[/yellow] This model is now available in the [bold cyan]HuggingFace[/bold cyan] provider.")
                tui.console.print(f"[dim]To use it with {provider_name}, you can:[/dim]")

                if provider_type == ProviderType.GGUF:
                    tui.console.print(f"[dim]  • Convert it to GGUF format using the quantization feature[/dim]")
                elif provider_type == ProviderType.MLX:
                    tui.console.print(f"[dim]  • Quantize it to MLX format using the quantization feature[/dim]")
                elif provider_type == ProviderType.OPENVINO:
                    tui.console.print(f"[dim]  • Quantize it to OpenVINO format using the quantization feature[/dim]")
                elif provider_type == ProviderType.QUANTIZED:
                    tui.console.print(f"[dim]  • Quantize it using the quantization feature[/dim]")

                tui.console.print()

                # Ask for confirmation that user understands
                prompt_yes_no(
                    "Do you understand where to find this model?",
                    default=True
                )
        else:
            tui.show_error(f"Failed to install {model_name}")
            logger.error(f"Installation failed for {model_name}")

    except Exception as e:
        progress.stop()
        logger.error(f"Installation error for {model_name}: {e}")
        tui.show_error(f"Installation failed: {e}")
        install_success = False

    tui.prompt("Press Enter to continue...", style="dim")
    return install_success


def parse_selection_input(input_str: str, max_value: int) -> list[int]:
    """Parse user selection input supporting multiple formats.

    Supports:
    - Single: "5" → [5]
    - Multiple: "1 2 3" or "1,2,3" → [1, 2, 3]
    - Range: "1-5" → [1, 2, 3, 4, 5]
    - Mixed: "1 3-5 7" → [1, 3, 4, 5, 7]

    Args:
        input_str: User input string
        max_value: Maximum valid value

    Returns:
        List of selected indices (1-based)

    Raises:
        ValueError: If input is invalid
    """
    if not input_str or input_str.strip().lower() == 'c':
        return []

    indices = set()

    # Replace commas with spaces for uniform parsing
    input_str = input_str.replace(',', ' ')

    # Split by whitespace
    parts = input_str.split()

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check for range (e.g., "1-5")
        if '-' in part:
            try:
                start, end = part.split('-', 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())

                if start_idx < 1 or end_idx > max_value or start_idx > end_idx:
                    raise ValueError(f"Invalid range: {part}")

                indices.update(range(start_idx, end_idx + 1))
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid range format: {part}") from e
        else:
            # Single number
            try:
                idx = int(part)
                if idx < 1 or idx > max_value:
                    raise ValueError(f"Index {idx} out of range [1-{max_value}]")
                indices.add(idx)
            except ValueError as e:
                raise ValueError(f"Invalid number: {part}") from e

    return sorted(list(indices))


def run_delete_workflow(
    session_manager: SessionManager,
    provider_type: ProviderType,
    models: list
) -> bool:
    """Run model deletion workflow with multi-selection support.

    Supports multiple selection formats:
    - Single: "5"
    - Multiple: "1 2 3" or "1,2,3"
    - Range: "1-5"
    - Mixed: "1 3-5 7"

    Args:
        session_manager: Session manager instance
        provider_type: Provider to delete model from
        models: List of available models

    Returns:
        True if at least one model was successfully deleted
    """
    from ..models.model import ModelInfo

    tui.clear_screen()

    # Get provider instance
    provider = session_manager.model_discovery.get_provider(provider_type)
    if provider is None:
        tui.show_error(f"Provider {provider_type.value} not registered")
        return False

    if not models:
        tui.show_message(
            "No models available to delete.",
            title="No Models",
            style="yellow"
        )
        tui.prompt("Press Enter to continue...", style="dim")
        return False

    # Show model list for selection
    tui.show_panel(
        "[bold cyan]Select Model(s) to Delete[/bold cyan]\n\n"
        "Choose one or more models to delete permanently.\n"
        "[dim]Examples: '5' (single), '1 2 3' (multiple), '1-5' (range), '1 3-5 7' (mixed)[/dim]",
        title="Model Deletion",
        border_style="cyan"
    )

    # Display simplified model list
    tui.console.print()
    for idx, model in enumerate(models, 1):
        tui.console.print(f"  [{idx}] {model.name} - {model.size_gb:.1f} GB")
    tui.console.print()

    # Get model selection(s)
    try:
        choice = tui.prompt(
            f"Select model(s) [1-{len(models)}] or 'c' to cancel:",
            style="cyan"
        ).strip()

        if choice.lower() == 'c':
            logger.info("User cancelled model deletion")
            return False

        # Parse selection input
        selected_indices = parse_selection_input(choice, len(models))

        if not selected_indices:
            logger.info("User cancelled model deletion (empty selection)")
            return False

        # Get selected models
        selected_models = [models[idx - 1] for idx in selected_indices]

    except ValueError as e:
        tui.show_error(f"Invalid input: {e}")
        tui.prompt("Press Enter to continue...", style="dim")
        return False

    # Calculate total size
    total_size = sum(model.size_gb for model in selected_models)

    # Confirm deletion
    model_names = "\n".join([f"  • {model.name} ({model.size_gb:.1f} GB)" for model in selected_models])
    confirm_msg = (
        f"[bold red]Delete {len(selected_models)} model(s)?[/bold red]\n\n"
        f"{model_names}\n\n"
        f"[bold]Total space to free: {total_size:.1f} GB[/bold]\n\n"
        f"[yellow]This action cannot be undone![/yellow]"
    )

    tui.show_panel(confirm_msg, title="Confirm Deletion", border_style="red")
    confirm = tui.prompt("Type 'yes' to confirm deletion:", style="yellow").strip().lower()

    if confirm != 'yes':
        logger.info(f"User cancelled deletion of {len(selected_models)} models")
        tui.show_message("Deletion cancelled.", title="Cancelled", style="yellow")
        tui.prompt("Press Enter to continue...", style="dim")
        return False

    # Check if any selected model is currently loaded - unload it first
    if session_manager.state.loaded_model:
        loaded_model_id = session_manager.state.loaded_model.model_info.model_id
        if any(model.model_id == loaded_model_id for model in selected_models):
            logger.info("Unloading currently loaded model before deletion")
            tui.show_message(
                "One of the selected models is currently loaded.\nUnloading before deletion...",
                title="Unloading Model",
                style="yellow"
            )
            session_manager.unload_current_model()

    # Delete the models
    tui.clear_screen()
    tui.console.print(f"\n[bold yellow]Deleting {len(selected_models)} model(s)...[/bold yellow]\n")

    successful_deletions = []
    failed_deletions = []

    for i, model in enumerate(selected_models, 1):
        tui.console.print(f"[{i}/{len(selected_models)}] Deleting {model.name}...", style="cyan")

        try:
            delete_success = provider.delete_model(model.model_id)

            if delete_success:
                successful_deletions.append(model)
                tui.console.print(f"  ✓ Deleted {model.name} ({model.size_gb:.1f} GB)", style="green")
                logger.info(f"Successfully deleted {model.name}")
            else:
                failed_deletions.append(model)
                tui.console.print(f"  ✗ Failed to delete {model.name}", style="red")
                logger.error(f"Deletion failed for {model.name}")

        except Exception as e:
            failed_deletions.append(model)
            logger.error(f"Deletion error for {model.name}: {e}")
            tui.console.print(f"  ✗ Error deleting {model.name}: {e}", style="red")

    # Show summary
    tui.console.print()
    freed_space = sum(model.size_gb for model in successful_deletions)

    if successful_deletions and not failed_deletions:
        tui.show_message(
            f"✓ Successfully deleted {len(successful_deletions)} model(s)!\n\n"
            f"Freed {freed_space:.1f} GB of disk space.",
            title="Deletion Complete",
            style="green"
        )
    elif successful_deletions and failed_deletions:
        tui.show_message(
            f"⚠ Partial success:\n\n"
            f"✓ Deleted: {len(successful_deletions)} model(s) ({freed_space:.1f} GB freed)\n"
            f"✗ Failed: {len(failed_deletions)} model(s)\n\n"
            f"Check logs for details.",
            title="Deletion Partially Complete",
            style="yellow"
        )
    else:
        tui.show_error(
            f"Failed to delete all {len(selected_models)} model(s).\n\n"
            "Check logs for details."
        )

    tui.prompt("Press Enter to continue...", style="dim")
    return len(successful_deletions) > 0


def main() -> None:
    """Main application entry point with clean TUI."""
    global _global_session_manager

    # Setup logging (only errors to console, everything to file)
    setup_logging()
    logger.info("Starting VLM/LLM CLI application")

    # Setup signal handlers for cleanup
    setup_signal_handlers()

    # Show welcome screen
    WelcomeScreen.show()
    time.sleep(1.5)  # Brief pause to show welcome

    # Detect system specs
    LoadingScreen.show("Detecting system specifications...")
    try:
        system_specs = SystemSpecs.detect()
        logger.info("System specs detected successfully")
    except Exception as e:
        logger.error(f"System detection failed: {e}")
        tui.show_error(f"Failed to detect system specs: {e}")
        return

    # Show system specs screen
    SystemSpecsScreen.show(system_specs)

    # Main application loop - allows switching between modes
    operation_mode = None
    session_manager = None
    config = None

    while True:
        # Show mode selection if not set
        if operation_mode is None:
            operation_mode = show_quantization_or_inference_menu()
            if operation_mode == "quit":
                logger.info("User quit from operation mode selection")
                cleanup_on_exit()
                tui.clear_screen()
                tui.show_message(
                    "Thank you for using VLM/LLM CLI!",
                    title="Goodbye",
                    style="cyan"
                )
                return
            elif operation_mode == "diagnostics":
                # Run diagnostics
                logger.info("Running system diagnostics")
                from ..cli.diagnostics import SystemDiagnostics
                diag = SystemDiagnostics()
                diag.run_full_diagnostics(output_format="rich")
                tui.prompt("\nPress Enter to return to main menu...", style="dim")
                operation_mode = None  # Return to main menu
                continue

        # Initialize config and session manager if needed
        if session_manager is None:
            LoadingScreen.show("Loading configuration...")
            try:
                config = load_config()
                logger.info("Configuration loaded")
            except Exception as e:
                logger.error(f"Config loading failed: {e}")
                tui.show_error(f"Failed to load configuration: {e}")
                return

            LoadingScreen.show("Initializing session...")
            session_manager = SessionManager(config, system_specs)
            _global_session_manager = session_manager

            # Register all providers ONCE (used by both quantization and inference modes)
            LoadingScreen.show("Registering model providers...")
            registered_providers = register_all_providers(session_manager, config, system_specs)
            if not registered_providers:
                logger.warning("No providers were registered successfully")

        # Run selected mode
        if operation_mode == "quantization":
            logger.info("Running Quantization mode")
            result = run_quantization_mode(session_manager)

            if result == "quit":
                logger.info("User quit from quantization mode")
                cleanup_on_exit()
                tui.clear_screen()
                tui.show_message(
                    "Thank you for using VLM/LLM CLI!",
                    title="Goodbye",
                    style="cyan"
                )
                return
            elif result == "main_menu":
                logger.info("Returning to main menu from quantization")
                operation_mode = None  # Reset to show menu again
                continue
            elif result == "switch_inference":
                logger.info("Switching from quantization to inference mode")
                operation_mode = "inference"
                continue

        elif operation_mode == "inference":
            # Run inference mode
            logger.info("Running Inference mode")
            result = run_inference_mode(session_manager, config, system_specs)

            if result == "quit":
                logger.info("User quit from inference mode")
                cleanup_on_exit()
                tui.clear_screen()
                tui.show_message(
                    "Thank you for using VLM/LLM CLI!",
                    title="Goodbye",
                    style="cyan"
                )
                return
            elif result == "main_menu":
                logger.info("Returning to main menu from inference")
                operation_mode = None  # Reset to show menu again
                continue
            elif result == "switch_quantization":
                logger.info("Switching from inference to quantization mode")
                operation_mode = "quantization"
                continue
            elif result == "switch_finetuning":
                logger.info("Switching from inference to finetuning mode")
                operation_mode = "finetuning"
                continue

        elif operation_mode == "finetuning":
            # Run finetuning mode
            logger.info("Running Finetuning mode")
            result = run_finetuning_mode(session_manager)

            if result == "quit":
                logger.info("User quit from finetuning mode")
                cleanup_on_exit()
                tui.clear_screen()
                tui.show_message(
                    "Thank you for using VLM/LLM CLI!",
                    title="Goodbye",
                    style="cyan"
                )
                return
            elif result == "main_menu":
                logger.info("Returning to main menu from finetuning")
                operation_mode = None  # Reset to show menu again
                continue
            elif result == "switch_inference":
                logger.info("Switching from finetuning to inference mode")
                operation_mode = "inference"
                continue
            elif result == "switch_quantization":
                logger.info("Switching from finetuning to quantization mode")
                operation_mode = "quantization"
                continue

        elif operation_mode == "benchmarking":
            # Run benchmarking mode
            logger.info("Running Benchmarking mode")
            from .app_quantization import run_benchmarking_mode
            result = run_benchmarking_mode(session_manager)

            if result == "quit":
                logger.info("User quit from benchmarking mode")
                cleanup_on_exit()
                tui.clear_screen()
                tui.show_message(
                    "Thank you for using VLM/LLM CLI!",
                    title="Goodbye",
                    style="cyan"
                )
                return
            elif result == "main_menu":
                logger.info("Returning to main menu from benchmarking")
                operation_mode = None  # Reset to show menu again
                continue
            elif result == "switch_inference":
                logger.info("Switching from benchmarking to inference mode")
                operation_mode = "inference"
                continue


def register_all_providers(session_manager: SessionManager, config: dict, system_specs: SystemSpecs) -> list[ProviderType]:
    """Register all enabled providers with the session manager.

    Args:
        session_manager: Session manager
        config: Configuration dict
        system_specs: System specifications

    Returns:
        List of successfully registered provider types
    """
    enabled_providers = [p for p, c in config.items() if c.enabled]
    registered_providers = []

    # Check if Ollama is enabled and ensure it's running
    ollama_config = config.get(ProviderType.OLLAMA)
    if ollama_config and ollama_config.enabled:
        logger.info("Checking Ollama server...")
        if not ensure_ollama_running(str(ollama_config.host)):
            logger.warning("Could not start Ollama server - skipping Ollama provider")
        else:
            logger.info("Ollama server is running")

    # Register enabled providers
    for provider_type, provider_config in config.items():
        if not provider_config.enabled:
            continue

        try:
            if provider_type == ProviderType.OLLAMA:
                provider = OllamaProvider(provider_config)
                session_manager.register_provider(provider_type, provider_config, provider)
                logger.info(f"Registered Ollama provider at {provider_config.host}")
                registered_providers.append(provider_type)

            elif provider_type == ProviderType.HUGGINGFACE:
                provider = HuggingFaceProvider(provider_config)
                session_manager.register_provider(provider_type, provider_config, provider)
                logger.info(f"Registered HuggingFace provider at {provider_config.cache_dir}")
                registered_providers.append(provider_type)

            elif provider_type == ProviderType.GGUF:
                provider = GGUFProvider(provider_config)
                session_manager.register_provider(provider_type, provider_config, provider)
                logger.info(f"Registered GGUF provider at {provider_config.models_dir}")
                registered_providers.append(provider_type)

            elif provider_type == ProviderType.QUANTIZED:
                provider = QuantizedProvider(provider_config, system_specs)
                session_manager.register_provider(provider_type, provider_config, provider)
                logger.info(f"Registered Quantized provider at {provider_config.models_dir}")
                registered_providers.append(provider_type)

            # LM Studio not yet implemented
            elif provider_type == ProviderType.LM_STUDIO:
                logger.info("LM Studio provider not yet implemented")

            # 2025 NEW: MLX provider for Apple Silicon
            elif provider_type == ProviderType.MLX:
                try:
                    from ..providers.mlx import MLXProvider
                    provider = MLXProvider(provider_config)
                    session_manager.register_provider(provider_type, provider_config, provider)
                    logger.info(f"Registered MLX provider at {provider_config.models_dir}")
                    registered_providers.append(provider_type)
                except ImportError:
                    logger.warning("MLX provider not available (requires macOS with Apple Silicon)")
                except Exception as e:
                    logger.error(f"Failed to initialize MLX provider: {e}")

            # 2025 NEW: OpenVINO provider for Intel optimization
            elif provider_type == ProviderType.OPENVINO:
                try:
                    from ..providers.openvino import OpenVINOProvider
                    provider = OpenVINOProvider(provider_config)
                    session_manager.register_provider(provider_type, provider_config, provider)
                    logger.info(f"Registered OpenVINO provider at {provider_config.models_dir}")
                    registered_providers.append(provider_type)
                except ImportError:
                    logger.warning("OpenVINO provider not available (install: pip install openvino optimum[openvino])")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenVINO provider: {e}")

            else:
                logger.warning(f"Unknown provider type: {provider_type.value}")

        except Exception as e:
            logger.error(f"Failed to register {provider_type.value} provider: {e}")

    logger.info(f"Registered {len(registered_providers)} providers successfully")
    return registered_providers


def run_inference_mode(session_manager: SessionManager, config: dict, system_specs: SystemSpecs) -> str:
    """Run inference mode.

    Args:
        session_manager: Session manager
        config: Configuration dict
        system_specs: System specifications

    Returns:
        "quit", "main_menu", or "switch_quantization"
    """
    global _global_session_manager
    _global_session_manager = session_manager

    enabled_providers = [p for p, c in config.items() if c.enabled]
    logger.info(f"Running inference mode with {len(enabled_providers)} providers enabled")

    # Main application loop
    while True:
        # Check for completed quantization notifications
        check_and_show_completion_notifications()

        # Show provider selection menu
        provider_names = [p.value for p in enabled_providers]
        selected_provider = ProviderSelectionMenu.show(provider_names)

        if selected_provider == "QUIT":
            # User quit - cleanup and exit
            logger.info("User initiated quit from provider selection")
            return "quit"

        if selected_provider == "HOME":
            # User wants to return to main menu
            logger.info("User chose to return to main menu from provider selection")
            return "main_menu"

        # Convert provider name to ProviderType
        provider_type = ProviderType(selected_provider)

        # Discover models for selected provider
        LoadingScreen.show(f"Discovering models from {selected_provider.upper()}...")
        try:
            models = session_manager.discover_models(provider=provider_type)
            logger.info(f"Discovered {len(models)} models from {selected_provider}")

            if not models:
                tui.show_message(
                    f"No models found for {selected_provider.upper()}.\n\n"
                    f"Install models using:\n"
                    f"  Ollama: ollama pull llama3.2:3b",
                    title="No Models",
                    style="yellow"
                )
                tui.prompt("Press Enter to continue...", style="dim")
                continue

        except Exception as e:
            logger.error(f"Failed to discover models: {e}")
            tui.show_error(f"Failed to discover models: {e}")
            tui.prompt("Press Enter to continue...", style="dim")
            continue

        # Show model selection menu
        while True:
            # Check for completed quantization notifications
            check_and_show_completion_notifications()

            selected_model = ModelSelectionMenu.show(models, selected_provider)

            if selected_model == "QUIT":
                # User quit - cleanup and exit
                logger.info("User initiated quit from model selection")
                return "quit"

            if selected_model == "HOME":
                # User wants to return to main menu
                logger.info("User chose to return to main menu from model selection")
                return "main_menu"

            if selected_model == "BACK":
                # Go back to provider selection
                logger.info("User pressed b - going back to provider selection")
                break

            if selected_model == "INSTALL":
                # User wants to install a new model
                logger.info(f"User initiated model installation for provider: {provider_type.value}")
                install_success = run_install_workflow(session_manager, provider_type, system_specs)

                if install_success:
                    # Refresh model list after successful installation
                    LoadingScreen.show(f"Refreshing models from {selected_provider.upper()}...")
                    try:
                        models = session_manager.discover_models(provider=provider_type)
                        logger.info(f"Refreshed models: {len(models)} found")
                    except Exception as e:
                        logger.error(f"Failed to refresh models: {e}")
                        tui.show_error(f"Failed to refresh model list: {e}")

                # Continue to show model menu again
                continue

            if selected_model == "DELETE":
                # User wants to delete a model
                logger.info("User initiated model deletion")
                delete_success = run_delete_workflow(session_manager, provider_type, models)

                if delete_success:
                    # Refresh model list after successful deletion
                    LoadingScreen.show(f"Refreshing models from {selected_provider.upper()}...")
                    try:
                        models = session_manager.discover_models(provider=provider_type)
                        logger.info(f"Refreshed models: {len(models)} found")

                        # If no models left, show message and go back
                        if not models:
                            tui.show_message(
                                f"No models remaining for {selected_provider.upper()}.\n\n"
                                "Install a model or select a different provider.",
                                title="No Models",
                                style="yellow"
                            )
                            tui.prompt("Press Enter to continue...", style="dim")
                            break

                    except Exception as e:
                        logger.error(f"Failed to refresh models: {e}")
                        tui.show_error(f"Failed to refresh model list: {e}")

                # Continue to show model menu again
                continue

            if selected_model == "REFRESH":
                # User wants to refresh the model registry
                logger.info(f"User initiated model registry refresh for {selected_provider}")

                # Clear metadata cache to force re-inspection
                from ..models.model_cache import ModelMetadataCache
                cache = ModelMetadataCache()
                cache.clear_cache()
                logger.info("Cleared model metadata cache")

                # Show loading screen while refreshing
                LoadingScreen.show(f"Refreshing model registry for {selected_provider.upper()}...\n\n[dim]Clearing cache and re-discovering models...[/dim]")

                try:
                    # Re-discover models from provider
                    models = session_manager.discover_models(provider=provider_type)
                    logger.info(f"Refreshed models: {len(models)} found")

                    tui.show_message(
                        f"Model registry refreshed successfully!\n\n"
                        f"Found {len(models)} models for {selected_provider.upper()}",
                        title="Refresh Complete",
                        style="green"
                    )
                    tui.prompt("Press Enter to continue...", style="dim")

                except Exception as e:
                    logger.error(f"Failed to refresh model registry: {e}")
                    tui.show_error(
                        f"Failed to refresh model registry:\n\n{e}\n\n"
                        f"Make sure {selected_provider.upper()} is running and accessible.",
                        title="Refresh Failed"
                    )
                    tui.prompt("Press Enter to continue...", style="dim")

                # Continue to show model menu again
                continue

            # Model selected - check compatibility and confirm if needed
            logger.info(f"User selected model: {selected_model.model_id}")

            # Handle TOO_LARGE models with user override
            if selected_model.compatibility == "too_large":
                tui.show_message(
                    f"⚠️  WARNING: Model Size Exceeds Recommended Limit\n\n"
                    f"Model: {selected_model.name}\n"
                    f"Size: {selected_model.size_gb:.1f} GB\n"
                    f"Recommended max: {system_specs.recommended_model_size_gb:.1f} GB\n"
                    f"Available RAM: {system_specs.available_ram_gb:.1f} GB\n\n"
                    f"{selected_model.compatibility_message}\n\n"
                    f"This may cause:\n"
                    f"  • Out-of-memory (OOM) errors\n"
                    f"  • System slowdown or freezing\n"
                    f"  • Application crashes\n"
                    f"  • Excessive swap usage",
                    title="Memory Warning",
                    style="yellow"
                )

                confirm = tui.prompt(
                    "Do you want to proceed anyway? [y/N]:",
                    style="yellow"
                ).lower().strip()

                if confirm not in ["y", "yes"]:
                    logger.info(f"User declined to load too-large model: {selected_model.model_id}")
                    continue

            # Show loading screen
            LoadingScreen.show(f"Loading model: {selected_model.name}...")

            # Attempt to load the model
            try:
                success = session_manager.load_model(selected_model.model_id, provider_type)
                if not success:
                    raise RuntimeError("Model loading failed")
                logger.info(f"Model loaded successfully: {selected_model.model_id}")

                # Show success message
                tui.show_message(
                    f"✓ Model loaded successfully!\n\n"
                    f"Model: {selected_model.name}\n"
                    f"Size: {selected_model.size_gb:.1f} GB\n"
                    f"Type: {selected_model.model_type}\n"
                    f"Provider: {selected_provider.upper()}\n"
                    f"Device: {session_manager.state.loaded_model.device}",
                    title="Model Ready",
                    style="green"
                )
                tui.prompt("Press Enter to continue...", style="dim")

                # Run endpoint workflow loop
                workflow_result = run_endpoint_workflow(session_manager, selected_model)

                # After exiting endpoint workflow, unload model
                logger.info("Unloading model after workflow completion")
                session_manager.unload_current_model()

                # Handle special navigation signals
                if workflow_result == "QUIT":
                    # Propagate quit signal up
                    logger.info("Propagating QUIT signal from endpoint workflow")
                    return "quit"

                if workflow_result == "HOME":
                    # User wants to return to main menu
                    logger.info("Propagating HOME signal - returning to main menu")
                    return "main_menu"

            except Exception as e:
                logger.error(f"Failed to load model {selected_model.model_id}: {e}")
                tui.show_error(
                    f"Failed to load model: {e}\n\n"
                    f"This could be due to:\n"
                    f"  • Insufficient memory\n"
                    f"  • Model not pulled/installed\n"
                    f"  • Provider service unavailable"
                )
                tui.prompt("Press Enter to continue...", style="dim")
                continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Signal handler already handles cleanup, just exit gracefully
        pass
    except Exception as e:
        # Log unexpected errors
        logger.exception(f"Unexpected error: {e}")
        tui.show_error(f"Unexpected error: {e}\n\nCheck logs for details.")
        # Ensure cleanup happens
        cleanup_on_exit()
