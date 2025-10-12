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
from ..services.session import SessionManager
from ..utils.ollama_manager import ensure_ollama_running
from .config_loader import load_config
from .logging_setup import setup_logging

# Global session manager for cleanup on exit/signals
_global_session_manager: Optional[SessionManager] = None


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
        "QUIT" to quit app, "BACK2" to go back 2 levels, None for normal back
    """
    from ..models.model import ModelInfo

    while True:
        # Show endpoint menu
        endpoint = EndpointMenu.show(model_info.name, str(model_info.model_type))

        if endpoint == "QUIT":
            # User quit - propagate signal
            logger.info("User initiated quit from endpoint menu")
            return "QUIT"

        if endpoint == "BACK2":
            # Go back 2 levels - propagate signal
            logger.info("User pressed bb from endpoint menu - going back 2 levels")
            return "BACK2"

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
    provider = session_manager.model_discovery.get_provider(provider_type)
    if provider is None:
        tui.show_error(f"Provider {provider_type.value} not registered")
        return False

    # Prompt for model name with examples
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


def run_delete_workflow(
    session_manager: SessionManager,
    provider_type: ProviderType,
    models: list
) -> bool:
    """Run model deletion workflow.

    Args:
        session_manager: Session manager instance
        provider_type: Provider to delete model from
        models: List of available models

    Returns:
        True if model was successfully deleted
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
        "[bold cyan]Select Model to Delete[/bold cyan]\n\n"
        "Choose a model from the list to delete it permanently.",
        title="Model Deletion",
        border_style="cyan"
    )

    # Display simplified model list
    tui.console.print()
    for idx, model in enumerate(models, 1):
        tui.console.print(f"  [{idx}] {model.name} - {model.size_gb:.1f} GB")
    tui.console.print()

    # Get model selection
    try:
        choice = tui.prompt(
            f"Select model [1-{len(models)}] or 'c' to cancel:",
            style="cyan"
        ).strip().lower()

        if choice == 'c':
            logger.info("User cancelled model deletion")
            return False

        idx = int(choice)
        if not (1 <= idx <= len(models)):
            tui.show_error("Invalid selection")
            tui.prompt("Press Enter to continue...", style="dim")
            return False

        selected_model = models[idx - 1]

    except ValueError:
        tui.show_error("Invalid input")
        tui.prompt("Press Enter to continue...", style="dim")
        return False

    # Confirm deletion
    if not confirm_deletion(selected_model.name, selected_model.size_gb):
        logger.info(f"User cancelled deletion of {selected_model.name}")
        tui.show_message("Deletion cancelled.", title="Cancelled", style="yellow")
        tui.prompt("Press Enter to continue...", style="dim")
        return False

    # Check if model is currently loaded - unload it first
    if (session_manager.state.loaded_model and
        session_manager.state.loaded_model.model_info.model_id == selected_model.model_id):
        logger.info(f"Unloading currently loaded model before deletion: {selected_model.model_id}")
        tui.show_message(
            f"Model {selected_model.name} is currently loaded.\nUnloading before deletion...",
            title="Unloading Model",
            style="yellow"
        )
        session_manager.unload_current_model()

    # Delete the model
    tui.clear_screen()
    tui.console.print(f"\n[bold yellow]Deleting {selected_model.name}...[/bold yellow]\n")

    try:
        delete_success = provider.delete_model(selected_model.model_id)

        if delete_success:
            tui.show_message(
                f"✓ Successfully deleted {selected_model.name}!\n\n"
                f"Freed {selected_model.size_gb:.1f} GB of disk space.",
                title="Deletion Complete",
                style="green"
            )
            logger.info(f"Successfully deleted {selected_model.name}")
        else:
            tui.show_error(f"Failed to delete {selected_model.name}")
            logger.error(f"Deletion failed for {selected_model.name}")

    except Exception as e:
        logger.error(f"Deletion error for {selected_model.name}: {e}")
        tui.show_error(f"Deletion failed: {e}")
        delete_success = False

    tui.prompt("Press Enter to continue...", style="dim")
    return delete_success


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

    # Load configuration
    LoadingScreen.show("Loading configuration...")
    try:
        config = load_config()
        enabled_providers = [p for p, c in config.items() if c.enabled]
        logger.info(f"Configuration loaded: {len(enabled_providers)} providers enabled")
    except Exception as e:
        logger.error(f"Config loading failed: {e}")
        tui.show_error(f"Failed to load configuration: {e}")
        return

    # Initialize session manager
    LoadingScreen.show("Initializing session...")
    session_manager = SessionManager(config, system_specs)

    # Register global session manager for cleanup on exit/signals
    _global_session_manager = session_manager

    # Check if Ollama is enabled and ensure it's running (silently in background)
    ollama_config = config.get(ProviderType.OLLAMA)
    if ollama_config and ollama_config.enabled:
        logger.info("Checking Ollama server...")
        if not ensure_ollama_running(str(ollama_config.host)):
            logger.error("Could not start Ollama server")
            tui.show_error(
                "Could not start Ollama server.\n\n"
                "Please install Ollama from: https://ollama.ai\n"
                "Or start it manually with: ollama serve"
            )
            return
        logger.info("Ollama server is running")

    # Register enabled providers (silently)
    for provider_type, provider_config in config.items():
        if not provider_config.enabled:
            continue

        try:
            if provider_type == ProviderType.OLLAMA:
                provider = OllamaProvider(provider_config)
                session_manager.register_provider(provider_type, provider_config, provider)
                logger.info(f"Registered Ollama provider at {provider_config.host}")

            elif provider_type == ProviderType.HUGGINGFACE:
                provider = HuggingFaceProvider(provider_config)
                session_manager.register_provider(provider_type, provider_config, provider)
                logger.info(f"Registered HuggingFace provider at {provider_config.cache_dir}")

            elif provider_type == ProviderType.GGUF:
                provider = GGUFProvider(provider_config)
                session_manager.register_provider(provider_type, provider_config, provider)
                logger.info(f"Registered GGUF provider at {provider_config.models_dir}")

            # LM Studio not yet implemented
            elif provider_type == ProviderType.LM_STUDIO:
                logger.info("LM Studio provider not yet implemented")

            else:
                logger.warning(f"Unknown provider type: {provider_type.value}")

        except Exception as e:
            logger.error(f"Failed to register {provider_type.value} provider: {e}")

    # Main application loop
    while True:
        # Show provider selection menu
        provider_names = [p.value for p in enabled_providers]
        selected_provider = ProviderSelectionMenu.show(provider_names)

        if selected_provider == "QUIT":
            # User quit - cleanup and exit
            logger.info("User initiated quit from provider selection")
            cleanup_on_exit()
            tui.clear_screen()
            tui.show_message(
                "Thank you for using VLM/LLM CLI!\nAll models unloaded.",
                title="Goodbye",
                style="cyan"
            )
            break

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
            selected_model = ModelSelectionMenu.show(models, selected_provider)

            if selected_model == "QUIT":
                # User quit - cleanup and exit
                logger.info("User initiated quit from model selection")
                cleanup_on_exit()
                tui.clear_screen()
                tui.show_message(
                    "Thank you for using VLM/LLM CLI!\nAll models unloaded.",
                    title="Goodbye",
                    style="cyan"
                )
                return

            if selected_model == "BACK2":
                # Go back 2 levels - exit to provider selection by returning
                logger.info("User pressed bb - going back 2 levels (to provider selection)")
                return

            if selected_model == "BACK":
                # Go back to provider selection
                logger.info("User pressed b - going back to provider selection")
                break

            if selected_model == "INSTALL":
                # User wants to install a new model
                logger.info("User initiated model installation")
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

            # Model selected - check compatibility and confirm if needed
            logger.info(f"User selected model: {selected_model.model_id}")

            # Handle TOO_LARGE models with user override
            if selected_model.compatibility == "too_large":
                tui.show_message(
                    f"⚠️  WARNING: Model May Not Fit in Memory\n\n"
                    f"Model: {selected_model.name}\n"
                    f"Size: {selected_model.size_gb:.1f} GB\n"
                    f"Recommended: {system_specs.recommended_model_size_gb:.1f} GB\n"
                    f"Available RAM: {system_specs.available_ram_gb:.1f} GB\n\n"
                    f"Loading this model may cause:\n"
                    f"  • System slowdown or crashes\n"
                    f"  • Out of memory errors\n"
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
                    cleanup_on_exit()
                    tui.clear_screen()
                    tui.show_message(
                        "Thank you for using VLM/LLM CLI!\nAll models unloaded.",
                        title="Goodbye",
                        style="cyan"
                    )
                    return

                if workflow_result == "BACK2":
                    # Go back 2 levels from endpoint menu = back to provider selection
                    # Break from model selection loop to get to provider selection
                    logger.info("Propagating BACK2 signal - returning to provider selection")
                    break

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
