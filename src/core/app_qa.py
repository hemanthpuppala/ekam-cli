"""QA endpoint implementation with image-aware session management."""

import time
from pathlib import Path

from loguru import logger

from ..cli.display import display_conversation_history
from ..cli.prompts import (
    UserExitException,
    prompt_enable_history,
    prompt_image_path,
    prompt_question,
    select_or_create_session,
)
from ..cli.tui_manager import tui
from ..models.endpoints import EndpointType, ProviderType
from ..models.inference import InferenceInput, InferenceOutput, InferenceResult
from ..services.session import SessionManager


def run_qa_endpoint(session_manager: SessionManager, model_info: "ModelInfo") -> None:
    """Run Question Answering endpoint with image-aware sessions.

    Args:
        session_manager: Session manager with loaded model
        model_info: Model information
    """
    from ..models.model import ModelInfo

    tui.clear_screen()
    tui.show_step_heading("Question Answering Endpoint")

    # Show endpoint information
    tui.show_panel(
        "[bold cyan]Question Answering Endpoint[/bold cyan]\n\n"
        "Ask questions about images with conversation history support.",
        title="QA Endpoint",
        border_style="cyan"
    )
    tui.prompt("Press Enter to start...", style="dim")

    try:
        # Get image path first
        image_path = prompt_image_path()
        image_path_str = str(image_path)

        # Check for existing sessions
        existing_sessions = session_manager.state.loaded_model.get_sessions(EndpointType.QA)

        # Check if there's an existing session for this specific image
        image_sessions = [s for s in existing_sessions if s.image_path == image_path_str]

        # Determine if history should be enabled
        use_history = False
        active_session = None

        if image_sessions:
            # Found session(s) for this image - let user choose
            action, session_id = select_or_create_session(
                "QA",
                image_sessions,
                show_image_path=True
            )

            if action == "continue" and session_id:
                # Continue existing session for this image
                session_manager.state.loaded_model.set_active_session(session_id)
                active_session = session_manager.state.loaded_model.get_active_session()
                use_history = True
                logger.info(f"Continuing QA session {session_id} for image {image_path_str}")
            else:
                # Create new session - ask if user wants history
                use_history = prompt_enable_history("QA")
                if use_history:
                    active_session = session_manager.state.loaded_model.create_session(
                        EndpointType.QA,
                        image_path=image_path_str
                    )
                    session_manager.state.statistics.record_session_created()
                    logger.info(f"Created new QA session {active_session.session_id} with history for image")
                else:
                    logger.info("Starting QA without history (single-turn mode)")
        else:
            # No existing sessions for this image - ask if user wants history
            use_history = prompt_enable_history("QA")
            if use_history:
                active_session = session_manager.state.loaded_model.create_session(
                    EndpointType.QA,
                    image_path=image_path_str
                )
                session_manager.state.statistics.record_session_created()
                logger.info(f"Created new QA session {active_session.session_id} with history for image")
            else:
                logger.info("Starting QA without history (single-turn mode)")
    except UserExitException:
        logger.info("User exited from QA endpoint setup")
        return

    # Main QA conversation loop
    current_image = image_path
    while True:
        tui.clear_screen()

        # Show session info header
        tui.console.print(f"[bold cyan]QA with {model_info.name}[/bold cyan]")
        tui.console.print(f"[dim]Image: {Path(current_image).name}[/dim]")
        if use_history and active_session:
            tui.console.print(f"[dim]Session: {active_session.session_id} • {active_session.exchange_count} questions[/dim]")
        elif use_history:
            tui.console.print("[dim]History enabled • Single turn mode[/dim]")
        else:
            tui.console.print("[dim]Single-turn mode (no history)[/dim]")

        tui.console.print()

        # Display conversation history if enabled
        if use_history and active_session and active_session.exchange_count > 0:
            display_conversation_history(active_session)

        # Show instructions
        tui.console.print("[dim]Commands: /info | /config | /status | /newimage | /exit[/dim]")
        tui.console.print("[dim]✓ Arrow Keys: Navigate | Up/Down: History | Enter: Submit | Ctrl+J: New line[/dim]\n")

        # Get user question using professional prompt handler
        from ..cli.text_input import professional_prompt
        question = professional_prompt.get_input(
            "Your question:",
            style="green",
            allow_multiline=True,
            show_instructions=False
        )

        # Handle special commands
        question_stripped = question.strip()
        question_lower = question_stripped.lower()

        # Exit commands
        if question_lower in ["exit", "quit", "back", "/exit", "/quit", "/back"]:
            logger.info("User exited QA")
            break

        if not question_stripped:
            continue

        # Slash commands
        if question_stripped.startswith("/"):
            command = question_lower[1:]  # Remove the /

            # Model info command
            if command in ["info", "model", "modelinfo"]:
                tui.clear_screen()
                from ..cli.model_config import display_model_info
                provider = session_manager.model_discovery.get_provider(ProviderType(model_info.provider))

                # Get detailed model info (Ollama only for now)
                if hasattr(provider, "get_model_info"):
                    model_details = provider.get_model_info(model_info.model_id)
                    display_model_info(tui.console, model_details)
                else:
                    tui.console.print(f"\n[yellow]Model info not available for {model_info.provider} provider[/yellow]\n")

                tui.prompt("Press Enter to continue...", style="dim")
                continue

            # Parameter status command
            elif command in ["status", "params", "parameters"]:
                tui.clear_screen()
                from ..cli.model_config import show_current_parameters

                # Get default parameters from provider
                provider = session_manager.model_discovery.get_provider(ProviderType(model_info.provider))
                defaults = {}
                if hasattr(provider, "get_model_info"):
                    model_details = provider.get_model_info(model_info.model_id)
                    defaults = model_details.get("default_parameters", {})

                # Show current parameters
                current_params = active_session.model_parameters if (use_history and active_session) else None
                show_current_parameters(tui.console, current_params, defaults)

                tui.prompt("Press Enter to continue...", style="dim")
                continue

            # Parameter configuration command
            elif command in ["config", "configure", "settings"]:
                tui.clear_screen()

                # Only allow config if using history (parameters are session-specific)
                if not use_history or not active_session:
                    tui.console.print("\n[yellow]⚠️  Parameter configuration requires a history-enabled session[/yellow]")
                    tui.console.print("[dim]Start a new session with history enabled to configure parameters[/dim]\n")
                    tui.prompt("Press Enter to continue...", style="dim")
                    continue

                # Get default parameters from provider
                from ..cli.model_config import configure_model_parameters
                provider = session_manager.model_discovery.get_provider(ProviderType(model_info.provider))
                defaults = {}
                if hasattr(provider, "get_model_info"):
                    model_details = provider.get_model_info(model_info.model_id)
                    defaults = model_details.get("default_parameters", {})

                # Configure parameters
                current_params = active_session.model_parameters
                new_params = configure_model_parameters(tui.console, current_params, defaults)

                if new_params:
                    active_session.model_parameters = new_params
                    tui.console.print("\n[bold green]✓ Parameters updated for this session[/bold green]")
                    tui.console.print("[dim]Changes will apply to all future questions in this session[/dim]\n")

                tui.prompt("Press Enter to continue...", style="dim")
                continue

            # New image command
            elif command in ["newimage", "image", "changeimage"]:
                question_lower = "newimage"  # Set to trigger the existing newimage handler below

            # Help command
            elif command in ["help", "h", "?"]:
                tui.clear_screen()
                tui.console.print("\n[bold cyan]Available Commands:[/bold cyan]\n")
                tui.console.print("  [cyan]/info[/cyan]      - Show detailed model information (architecture, license, etc.)")
                tui.console.print("  [cyan]/status[/cyan]    - Display current generation parameters")
                tui.console.print("  [cyan]/config[/cyan]    - Configure generation parameters (temperature, max_tokens, etc.)")
                tui.console.print("  [cyan]/newimage[/cyan] - Change to a different image")
                tui.console.print("  [cyan]/help[/cyan]      - Show this help message")
                tui.console.print("  [cyan]/exit[/cyan]      - Exit QA endpoint\n")
                tui.prompt("Press Enter to continue...", style="dim")
                continue

            else:
                tui.console.print(f"\n[yellow]Unknown command: {question_stripped}[/yellow]")
                tui.console.print("[dim]Type /help for available commands[/dim]\n")
                tui.prompt("Press Enter to continue...", style="dim")
                continue

        if question_lower == "newimage":
            # User wants to switch to a new image
            logger.info("User requested new image")

            try:
                new_image = prompt_image_path()
                new_image_str = str(new_image)
            except UserExitException:
                logger.info("User cancelled new image selection")
                continue

            # Check if this is a different image
            if new_image_str != str(current_image):
                # Different image detected
                if use_history and active_session:
                    # Prompt user about starting new conversation
                    tui.show_panel(
                        "[bold yellow]New Image Detected[/bold yellow]\n\n"
                        f"[bold]Previous:[/bold] {Path(current_image).name}\n"
                        f"[bold]New:[/bold] {Path(new_image).name}\n\n"
                        "Starting a new conversation can help with:\n"
                        "  • Faster responses (less context)\n"
                        "  • Better memory management\n"
                        "  • Clearer conversation focus\n\n"
                        "[dim]Your previous session is saved and can be resumed later.[/dim]",
                        title="New Image",
                        border_style="yellow"
                    )

                    from ..cli.prompts import prompt_yes_no
                    start_new = prompt_yes_no("Start a new conversation for this image?", default=True)

                    if start_new:
                        # Check if session exists for new image
                        image_sessions_new = [s for s in existing_sessions if s.image_path == new_image_str]

                        if image_sessions_new:
                            # Existing session for new image - let user choose
                            action, session_id = select_or_create_session(
                                "QA",
                                image_sessions_new,
                                show_image_path=True
                            )

                            if action == "continue" and session_id:
                                session_manager.state.loaded_model.set_active_session(session_id)
                                active_session = session_manager.state.loaded_model.get_active_session()
                                logger.info(f"Switched to existing session {session_id} for new image")
                            else:
                                # Create new session for new image
                                active_session = session_manager.state.loaded_model.create_session(
                                    EndpointType.QA,
                                    image_path=new_image_str
                                )
                                session_manager.state.statistics.record_session_created()
                                logger.info(f"Created new session {active_session.session_id} for new image")
                        else:
                            # No session for new image - create one
                            active_session = session_manager.state.loaded_model.create_session(
                                EndpointType.QA,
                                image_path=new_image_str
                            )
                            session_manager.state.statistics.record_session_created()
                            logger.info(f"Created new session {active_session.session_id} for new image")
                    else:
                        # Continue with same session but update image reference
                        logger.info("User chose to continue same session with new image")

                current_image = new_image
                logger.info(f"Switched to new image: {new_image_str}")
            else:
                logger.info("Same image selected, continuing")

            continue

        if not question.strip():
            continue

        # Run inference
        try:
            start_time = time.perf_counter()

            provider = session_manager.model_discovery.get_provider(ProviderType(model_info.provider))
            if provider is None:
                raise ValueError(f"Provider {model_info.provider} not registered")

            # Use appropriate method based on provider type
            provider_type_str = str(model_info.provider).lower()

            if provider_type_str == "ollama":
                # Ollama has convenience methods (manages its own history)
                response = provider.qa(model_info.model_id, str(current_image), question)
            else:
                # HF and GGUF use handle-based approach with conversation history
                if not session_manager.state.loaded_model:
                    raise RuntimeError("No model loaded")

                model_handle = session_manager.state.loaded_model._handle

                # Get conversation history if using history
                conversation_history = None
                if use_history and active_session:
                    conversation_history = active_session.get_history_for_provider()

                # Load image with PIL
                from PIL import Image
                image = Image.open(current_image)

                # Pass conversation history to provider
                response = provider.run_qa(model_handle, image, question, conversation_history)

            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000

            # Log raw VLM response (before any cleaning/processing) - FULL response without truncation
            logger.debug(f"Raw VLM QA response ({len(response)} chars): {response}")

            # Store exchange in session if using history
            if use_history and active_session:
                active_session.add_exchange(question, response, elapsed_ms)
                logger.debug(f"Added exchange to session {active_session.session_id}")

            logger.info(f"QA inference completed in {elapsed_ms:.2f}ms")

            # Record statistics
            inference_result = InferenceResult(
                model_id=model_info.model_id,
                provider=model_info.provider,
                device=session_manager.state.loaded_model.device,
                input=InferenceInput(endpoint=EndpointType.QA, prompt=question, image_path=str(current_image)),
                output=InferenceOutput(text_response=response),
                inference_time_ms=elapsed_ms,
            )
            session_manager.state.statistics.record_inference(inference_result, used_history=use_history)
            logger.debug(f"Recorded inference in statistics (history={use_history})")

            # Display just the new response (history already shown above)
            tui.console.print(f"[bold green]You:[/bold green] {question}")
            tui.console.print(f"[bold cyan]AI:[/bold cyan] {response}")
            tui.console.print(f"[dim]({elapsed_ms/1000:.2f}s)[/dim]\n")

            tui.prompt("Press Enter to continue...", style="dim")

        except Exception as e:
            logger.error(f"QA inference failed: {e}")
            tui.show_error(f"Inference failed: {e}")
            tui.prompt("Press Enter to continue...", style="dim")
