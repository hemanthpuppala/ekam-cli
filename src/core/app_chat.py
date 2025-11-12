"""Chat endpoint implementation with session management."""

import time
from loguru import logger

from ..cli.display import display_conversation_history
from ..cli.model_config import configure_model_parameters, display_model_info, show_current_parameters
from ..cli.prompts import UserExitException, prompt_enable_history, select_or_create_session
from ..cli.tui_manager import tui
from ..models.endpoints import EndpointType, ProviderType
from ..models.inference import InferenceInput, InferenceOutput, InferenceResult
from ..services.session import SessionManager
from ..utils.response_formatter import display_response_with_thinking, has_thinking_blocks, extract_thinking_blocks

# CLI Configuration: Max conversation history turns to prevent context overflow
# Library users can override this by calling get_history_for_provider(max_turns=N) directly
MAX_HISTORY_TURNS_CLI = 20  # Last 20 exchanges = ~40 messages, safe for most models


def run_chat_endpoint(session_manager: SessionManager, model_info: "ModelInfo") -> None:
    """Run Text Chat endpoint for LLMs with conversation history.

    Args:
        session_manager: Session manager with loaded model
        model_info: Model information
    """
    from ..models.model import ModelInfo

    tui.clear_screen()
    tui.show_step_heading("Text Chat Endpoint")

    # Warn if using VLM for text-only chat
    if str(model_info.model_type).lower() == "vlm":
        tui.show_panel(
            "[bold yellow]⚠️  VLM Text Chat Warning[/bold yellow]\n\n"
            f"[yellow]{model_info.name}[/yellow] is a Vision-Language Model.\n\n"
            "VLMs are trained on image+text pairs and may produce:\n"
            "  • Inconsistent responses without image context\n"
            "  • Hallucinated or random-seeming answers\n"
            "  • Unexpected behavior in pure text conversations\n\n"
            "[bold]Recommendation:[/bold] Use VLMs with images (QA, Caption endpoints)\n"
            "For text-only chat, use a pure LLM like llama3.2, qwen2.5, etc.\n\n"
            "[dim]Press Enter to continue anyway, or type 'back' to return...[/dim]",
            title="Warning: VLM in Text Mode",
            border_style="yellow"
        )

        choice = tui.prompt("Continue with text chat? [y/back]:", style="yellow").lower().strip()
        if choice in ["back", "b", "n", "no"]:
            return

        tui.clear_screen()

    # Show endpoint information
    tui.show_panel(
        "[bold cyan]Text Chat Endpoint[/bold cyan]\n\n"
        "Interactive text conversation with your model.",
        title="Chat Endpoint",
        border_style="cyan"
    )
    tui.prompt("Press Enter to start...", style="dim")

    try:
        # Check for existing sessions
        existing_sessions = session_manager.state.loaded_model.get_sessions(EndpointType.TEXT)

        # Determine if history should be enabled
        use_history = False
        active_session = None

        if existing_sessions:
            # Let user choose existing session or create new
            action, session_id = select_or_create_session(
                "Chat",
                existing_sessions,
                show_image_path=False
            )

            if action == "continue" and session_id:
                # Continue existing session
                session_manager.state.loaded_model.set_active_session(session_id)
                active_session = session_manager.state.loaded_model.get_active_session()
                use_history = True
                logger.info(f"Continuing chat session {session_id}")
            else:
                # Create new session - ask if user wants history
                use_history = prompt_enable_history("Chat")
                if use_history:
                    active_session = session_manager.state.loaded_model.create_session(EndpointType.TEXT)
                    session_manager.state.statistics.record_session_created()
                    logger.info(f"Created new chat session {active_session.session_id} with history")
                else:
                    logger.info("Starting chat without history (single-turn mode)")
        else:
            # No existing sessions - ask if user wants history
            use_history = prompt_enable_history("Chat")
            if use_history:
                active_session = session_manager.state.loaded_model.create_session(EndpointType.TEXT)
                session_manager.state.statistics.record_session_created()
                logger.info(f"Created new chat session {active_session.session_id} with history")
            else:
                logger.info("Starting chat without history (single-turn mode)")
    except UserExitException:
        logger.info("User exited from chat endpoint setup")
        return

    # Main conversation loop
    while True:
        tui.clear_screen()

        # Show session info header
        tui.console.print(f"[bold cyan]Chat with {model_info.name}[/bold cyan]")
        if use_history and active_session:
            session_info = f"Session: {active_session.session_id} • {active_session.exchange_count} exchanges"

            # Determine effective max history turns
            effective_max = MAX_HISTORY_TURNS_CLI
            if active_session.model_parameters and active_session.model_parameters.max_history_turns:
                effective_max = active_session.model_parameters.max_history_turns

            # Show history truncation warning if session exceeds limit
            if active_session.exchange_count > effective_max:
                session_info += f" (using last {effective_max} for context)"

            tui.console.print(f"[dim]{session_info}[/dim]")
        elif use_history:
            tui.console.print("[dim]History enabled • Single turn mode[/dim]")
        else:
            tui.console.print("[dim]Single-turn mode (no history)[/dim]")

        tui.console.print()

        # Display conversation history if enabled
        if use_history and active_session and active_session.exchange_count > 0:
            display_conversation_history(active_session)

        # Show instructions
        tui.console.print("[dim]Commands: /info | /config | /status | /clear | /exit[/dim]\n")

        # Get user message
        message = tui.prompt("You:", style="green")

        # Handle special commands
        message_stripped = message.strip()
        message_lower = message_stripped.lower()

        # Exit commands
        if message_lower in ["exit", "quit", "back", "/exit", "/quit", "/back"]:
            logger.info("User exited chat")
            break

        if not message_stripped:
            continue

        # Slash commands
        if message_stripped.startswith("/"):
            command = message_lower[1:]  # Remove the /

            # Clear history command
            if command in ["clear"]:
                if use_history and active_session:
                    active_session.exchanges.clear()
                    tui.clear_screen()
                    tui.console.print("[green]✓ Chat history cleared[/green]\n")
                    tui.prompt("Press Enter to continue...", style="dim")
                else:
                    tui.console.print("\n[yellow]No history to clear (single-turn mode)[/yellow]\n")
                    tui.prompt("Press Enter to continue...", style="dim")
                continue

            # Model info command
            if command in ["info", "model", "modelinfo"]:
                tui.clear_screen()
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
                    tui.console.print("[dim]Changes will apply to all future messages in this session[/dim]\n")

                tui.prompt("Press Enter to continue...", style="dim")
                continue

            # Help command
            elif command in ["help", "h", "?"]:
                tui.clear_screen()
                tui.console.print("\n[bold cyan]Available Commands:[/bold cyan]\n")
                tui.console.print("  [cyan]/info[/cyan]     - Show detailed model information (architecture, license, etc.)")
                tui.console.print("  [cyan]/status[/cyan]   - Display current generation parameters")
                tui.console.print("  [cyan]/config[/cyan]   - Configure generation parameters (temperature, max_tokens, etc.)")
                tui.console.print("  [cyan]/clear[/cyan]    - Clear chat history and screen")
                tui.console.print("  [cyan]/help[/cyan]     - Show this help message")
                tui.console.print("  [cyan]/exit[/cyan]     - Exit chat\n")
                tui.prompt("Press Enter to continue...", style="dim")
                continue

            else:
                tui.console.print(f"\n[yellow]Unknown command: {message_stripped}[/yellow]")
                tui.console.print("[dim]Type /help for available commands[/dim]\n")
                tui.prompt("Press Enter to continue...", style="dim")
                continue

        # Run inference
        try:
            start_time = time.perf_counter()

            provider = session_manager.model_discovery.get_provider(ProviderType(model_info.provider))
            if provider is None:
                raise ValueError(f"Provider {model_info.provider} not registered")

            # Get conversation history if using history (limited to prevent context overflow)
            conversation_history = None
            if use_history and active_session:
                # Determine max history turns: use session-specific setting if available, else CLI default
                max_turns = MAX_HISTORY_TURNS_CLI  # Default: 20
                if active_session.model_parameters and active_session.model_parameters.max_history_turns:
                    max_turns = active_session.model_parameters.max_history_turns
                    logger.debug(f"Using session-specific max_history_turns: {max_turns}")

                # Library users can call get_history_for_provider(max_turns=None) for unlimited
                conversation_history = active_session.get_history_for_provider(max_turns=max_turns)
                total_turns = active_session.exchange_count
                used_turns = len(conversation_history)
                if total_turns > used_turns:
                    logger.debug(f"Limiting history: using last {used_turns}/{total_turns} exchanges")
                else:
                    logger.debug(f"Passing {used_turns} history turns to provider")

            # Get custom parameters from session (if any)
            custom_parameters = None
            if use_history and active_session and active_session.model_parameters:
                custom_parameters = active_session.model_parameters.to_dict()
                logger.debug(f"Using session-specific parameters: {custom_parameters}")

            # Determine if we should use KV cache based on device
            from ..utils.kv_cache_policy import should_use_kv_cache, get_kv_cache_status_message
            device = session_manager.state.loaded_model.device
            use_kv_cache = should_use_kv_cache(device, is_benchmark=False)

            # Pass session_id only if using history AND KV cache is safe for this device
            session_id_for_inference = None
            if use_history and active_session and use_kv_cache:
                session_id_for_inference = active_session.session_id
                logger.info(f"💾 {get_kv_cache_status_message(device, False)}")
                logger.debug(f"Using session ID for KV cache: {session_id_for_inference}")
            elif use_history and active_session:
                logger.info(f"⚠️  {get_kv_cache_status_message(device, False)}")
                logger.debug("Session ID not passed - KV cache disabled for this device")

            # Use appropriate method based on provider type
            provider_type_str = str(model_info.provider).lower()

            # Prepare streaming UI and callback
            from rich.text import Text
            from rich.live import Live
            from rich.panel import Panel

            tui.console.print(f"[bold green]You:[/bold green] {message}")

            # Simple streaming - no special handling during stream
            first_token_time = [None]
            streamed_text = [""]

            # Create live display for streaming
            ai_text = Text()
            ai_text.append("AI: ", style="bold cyan")

            def _stream_cb(delta: str, is_first: bool = False):
                if is_first and first_token_time[0] is None:
                    first_token_time[0] = time.perf_counter()
                streamed_text[0] += delta
                ai_text.append(delta)
                live.update(ai_text)

            live = Live(ai_text, console=tui.console, refresh_per_second=24)
            live.start()

            if provider_type_str == "ollama":
                # Ollama: Always use run_text with streaming support
                response = provider.run_text(
                    model_info.model_id,
                    message,
                    conversation_history,
                    system_prompt=None,
                    custom_parameters=custom_parameters,
                    stream_callback=_stream_cb,
                )
            else:
                # HF and GGUF use handle-based approach with conversation history
                if not session_manager.state.loaded_model:
                    raise RuntimeError("No model loaded")

                model_handle = session_manager.state.loaded_model._handle

                # Pass conversation history and custom parameters to provider
                # Ensure quantized transformers path receives the stream callback via custom_parameters
                if custom_parameters is None:
                    custom_parameters = {}
                custom_parameters.setdefault("_stream_callback", _stream_cb)

                try:
                    response = provider.run_text(
                        model_handle,
                        message,
                        conversation_history,
                        custom_parameters=custom_parameters,
                        stream_callback=_stream_cb,
                        session_id=session_id_for_inference,
                    )
                except TypeError:
                    # Older provider signature without stream_callback/session_id
                    try:
                        response = provider.run_text(
                            model_handle,
                            message,
                            conversation_history,
                            custom_parameters=custom_parameters,
                            session_id=session_id_for_inference,
                        )
                    except TypeError:
                        # Oldest signature - no session_id support
                        response = provider.run_text(
                            model_handle,
                            message,
                            conversation_history,
                            custom_parameters=custom_parameters,
                        )

            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000

            # Stop live streaming view
            try:
                live.stop()
            except Exception:
                pass

            # Extract reasoning and clean response from raw output for storage
            thinking_blocks, clean_response = extract_thinking_blocks(response)
            reasoning_str = "\n\n".join(thinking_blocks) if thinking_blocks else None

            # Log raw LLM response (before any cleaning/processing) - FULL response without truncation
            logger.info("="*80)
            logger.info(f"RAW FULL RESPONSE ({len(response)} chars):")
            logger.info(response)  # Print FULL response, no truncation
            logger.info("="*80)
            logger.info(f"Has <think> tags: {'<think>' in response.lower()}")
            logger.info(f"Has </think> tags: {'</think>' in response.lower()}")
            logger.info(f"Extracted thinking blocks: {len(thinking_blocks)}")
            if reasoning_str:
                logger.info(f"Reasoning content (FULL): {reasoning_str}")
                logger.info(f"Clean response (FULL): {clean_response}")
            else:
                logger.info("No thinking blocks found in response")

            # Add newline for spacing
            tui.console.print()

            # Store exchange in session if using history (store reasoning separately)
            if use_history and active_session:
                active_session.add_exchange(
                    user_message=message,
                    ai_response=clean_response,
                    inference_time_ms=elapsed_ms,
                    ai_reasoning=reasoning_str
                )
                logger.debug(f"Added exchange to session {active_session.session_id} (with reasoning: {bool(reasoning_str)})")

            logger.info(f"Chat inference completed in {elapsed_ms:.2f}ms")

            # Record statistics (use clean response for stats)
            inference_result = InferenceResult(
                model_id=model_info.model_id,
                provider=model_info.provider,
                device=session_manager.state.loaded_model.device,
                input=InferenceInput(endpoint=EndpointType.TEXT, prompt=message),
                output=InferenceOutput(text_response=clean_response),
                inference_time_ms=elapsed_ms,
            )
            session_manager.state.statistics.record_inference(inference_result, used_history=use_history)
            logger.debug(f"Recorded inference in statistics (history={use_history})")

            # Print timing and tokens summary
            # Estimate output tokens and n_ctx when possible (based on clean response)
            output_text = clean_response
            approx_tokens = max(1, len(output_text) // 4)
            n_ctx = 4096
            try:
                h = session_manager.state.loaded_model._handle if session_manager.state.loaded_model else None
                if h is not None and hasattr(h, 'n_ctx'):
                    # LlamaServerManager stores n_ctx after start
                    n_ctx = getattr(h, 'n_ctx', n_ctx)
                elif provider_type_str == 'ollama' and hasattr(provider, '_ctx_overrides') and model_info.model_id in provider._ctx_overrides:
                    n_ctx = provider._ctx_overrides[model_info.model_id]
            except Exception:
                pass

            # Build timing info string
            timing_info = f"({elapsed_ms/1000:.2f}s | {approx_tokens}/{n_ctx} tokens)"
            if reasoning_str:
                timing_info += f" | 🤔 {len(thinking_blocks)} reasoning block(s)"

            tui.console.print(f"[dim]{timing_info}[/dim]")

            # If response has thinking blocks, offer to show them
            if reasoning_str:
                tui.console.print()
                show_choice = tui.prompt(
                    "[dim]View thinking? (y/n):[/dim]",
                    style="dim"
                ).lower().strip()

                if show_choice in ["yes", "y"]:
                    tui.console.print()
                    for i, thinking in enumerate(thinking_blocks, 1):
                        tui.console.print(Panel(
                            thinking.strip(),
                            title=f"[dim]🤔 Reasoning {i}/{len(thinking_blocks)}[/dim]",
                            border_style="dim",
                            style="dim italic"
                        ))
                    tui.console.print()

            tui.console.print()  # Add spacing before continue prompt
            tui.prompt("Press Enter to continue...", style="dim")

        except Exception as e:
            logger.error(f"Chat inference failed: {e}", exc_info=True)
            tui.show_error(f"Inference failed: {e}")
            tui.prompt("Press Enter to continue...", style="dim")
