"""HuggingFace provider implementation using transformers library."""

import os
from pathlib import Path
from typing import Any, Optional

import torch
from loguru import logger
from PIL import Image

from importlib import import_module

from ..models.endpoints import CompatibilityStatus, EndpointType, ModelType
from ..models.model import ModelInfo
from ..models.model_cache import ModelMetadataCache
from ..models.provider import ProviderConfig
from ..utils.architecture_registry import get_architecture_registry
from ..utils.history_formatter import format_conversation_history, format_qa_history
from .base import BaseProvider


class HuggingFaceProvider(BaseProvider):
    """HuggingFace provider using transformers library."""

    def __init__(self, config: ProviderConfig):
        """Initialize HuggingFace provider.

        Args:
            config: Provider configuration with cache_dir and device settings
        """
        self.config = config
        self.cache_dir = Path(str(config.cache_dir)).expanduser()
        self.device_preference = config.device_preference or ["cuda", "mps", "cpu"]

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata cache for efficient model inspection
        self.metadata_cache = ModelMetadataCache()
        logger.debug("Initialized model metadata cache")

        # Initialize architecture registry for automatic dependency management
        self.arch_registry = get_architecture_registry()
        logger.debug(f"Initialized architecture registry with {len(self.arch_registry.architectures)} architectures")

        # Initialize HuggingFace authentication
        self.hf_token = self._setup_authentication()

        # Determine best available device
        self.device = self._get_best_device()
        logger.info(f"HuggingFace provider initialized on device: {self.device}")

        # Store current model reference for cleanup
        self.current_model = None
        self.current_processor = None

    def _setup_mps_memory_management(self) -> None:
        """Configure MPS memory management to avoid OOM errors.

        Sets environment variables for better MPS memory handling and clears
        any cached memory before loading a new model.
        """
        if self.device == "mps":
            # Set a reasonable watermark ratio to prevent OOM while allowing growth
            # This is safer than 0.0 which can cause system failure
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"
            logger.debug("Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 for MPS memory management")

            # Clear MPS cache to free up memory before loading new model
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache before model loading")

    def _cleanup_current_model(self) -> None:
        """Clean up the currently loaded model to free memory.

        This is called before loading a new model to ensure maximum
        available memory for the new model.
        """
        try:
            if self.current_model is not None:
                del self.current_model
                self.current_model = None
                logger.debug("Cleaned up previous model from memory")

            if self.current_processor is not None:
                del self.current_processor
                self.current_processor = None
                logger.debug("Cleaned up previous processor from memory")

            # Clear GPU/MPS cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
            elif self.device == "mps":
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    logger.debug("Cleared MPS cache")
        except Exception as e:
            logger.debug(f"Error during model cleanup: {e}")

    def _move_model_to_device(self, model: Any, target_device: str) -> str:
        """Move model to target device with fallback to CPU on OOM.

        Args:
            model: The model to move
            target_device: The target device (cuda, mps, or cpu)

        Returns:
            The actual device the model was moved to
        """
        try:
            model.to(target_device)
            logger.debug(f"Successfully moved model to {target_device}")
            return target_device
        except RuntimeError as e:
            error_msg = str(e)

            # Check for MPS out of memory
            if "MPS backend out of memory" in error_msg and target_device == "mps":
                logger.warning(
                    f"MPS out of memory. Falling back to CPU. "
                    f"Error: {error_msg[:100]}..."
                )
                try:
                    model.to("cpu")
                    return "cpu"
                except Exception as cpu_error:
                    logger.error(f"Failed to move model to CPU: {cpu_error}")
                    raise RuntimeError(f"Failed to load model on any device: {e}")

            # Check for CUDA out of memory
            elif "CUDA out of memory" in error_msg and target_device == "cuda":
                logger.warning(
                    f"CUDA out of memory. Falling back to CPU. "
                    f"Error: {error_msg[:100]}..."
                )
                try:
                    torch.cuda.empty_cache()
                    model.to("cpu")
                    return "cpu"
                except Exception as cpu_error:
                    logger.error(f"Failed to move model to CPU: {cpu_error}")
                    raise RuntimeError(f"Failed to load model on any device: {e}")

            # Other runtime errors
            raise

    def _prompt_and_save_token(self) -> bool:
        """Interactively prompt user for HF token and save to config.yaml.

        Returns:
            True if token was successfully saved, False otherwise
        """
        try:
            from ..cli.tui_manager import tui

            # Show token prompt panel
            tui.show_panel(
                "[bold cyan]HuggingFace Authentication Required[/bold cyan]\n\n"
                "This model requires HuggingFace authentication.\n\n"
                "[bold]To get your token:[/bold]\n"
                "  1. Visit: https://huggingface.co/settings/tokens\n"
                "  2. Create a new token (read access is enough)\n"
                "  3. Copy the token\n\n"
                "[dim]Your token will be saved securely to config.yaml[/dim]",
                title="Get HuggingFace Token",
                border_style="cyan"
            )

            # Prompt for token
            token = tui.prompt(
                "Paste your HuggingFace token here (or press Enter to skip):",
                style="cyan"
            ).strip()

            if not token:
                logger.info("User skipped token input")
                return False

            # Validate token format (basic check)
            if not token.startswith("hf_"):
                tui.show_panel(
                    "[yellow]⚠️  Invalid token format[/yellow]\n\n"
                    "HF tokens should start with 'hf_'\n"
                    "Please check and try again.",
                    title="Invalid Token",
                    border_style="yellow"
                )
                logger.warning("Invalid token format provided by user")
                return False

            # Save token to config.yaml
            if self._save_token_to_config(token):
                # Update current instance token
                self.hf_token = token

                tui.show_panel(
                    "[bold green]✓ Token Saved Successfully![/bold green]\n\n"
                    "Your token has been saved to config.yaml\n"
                    "Restart the application to use the new token.",
                    title="Success",
                    border_style="green"
                )
                logger.info("HF token successfully saved to config.yaml")
                return True
            else:
                tui.show_panel(
                    "[red]✗ Failed to save token[/red]\n\n"
                    "Could not write to config.yaml\n"
                    "Please check file permissions.",
                    title="Error",
                    border_style="red"
                )
                return False

        except ImportError:
            # Fallback to simple input if TUI not available
            print("\nHuggingFace token required.")
            print("Get it from: https://huggingface.co/settings/tokens")
            token = input("Paste your token (starts with 'hf_'): ").strip()

            if token and token.startswith("hf_"):
                if self._save_token_to_config(token):
                    self.hf_token = token
                    print("✓ Token saved to config.yaml")
                    return True
                else:
                    print("✗ Failed to save token to config.yaml")
                    return False
            return False

    def _save_token_to_config(self, token: str) -> bool:
        """Save token to config.yaml file.

        Args:
            token: HuggingFace token to save

        Returns:
            True if successful, False otherwise
        """
        try:
            import yaml

            # Read current config
            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                config = {"providers": {}}

            # Ensure structure exists
            if "providers" not in config:
                config["providers"] = {}
            if "huggingface" not in config["providers"]:
                config["providers"]["huggingface"] = {}

            # Save token
            config["providers"]["huggingface"]["hf_token"] = token

            # Write back to config
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Token saved to {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save token to config: {e}")
            return False

    def _setup_authentication(self) -> Optional[str]:
        """Setup HuggingFace authentication from config or environment.

        Checks in order:
        1. Config file (hf_token field)
        2. Environment variable (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)
        3. Local ~/.huggingface/token file
        4. None if not found

        Returns:
            HuggingFace token string or None
        """
        # Try to get token from config
        hf_token = None
        try:
            if hasattr(self.config, 'hf_token') and self.config.hf_token:
                hf_token = self.config.hf_token
                logger.debug("Using HF token from config")
                return hf_token
        except (AttributeError, KeyError):
            pass

        # Try environment variables
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            logger.debug("Using HF token from environment variable")
            return hf_token

        # Try local token file
        token_file = Path.home() / ".huggingface" / "token"
        if token_file.exists():
            try:
                with open(token_file, 'r') as f:
                    hf_token = f.read().strip()
                    if hf_token:
                        logger.debug("Using HF token from ~/.huggingface/token")
                        return hf_token
            except Exception as e:
                logger.debug(f"Could not read token file: {e}")

        logger.debug("No HuggingFace token found (will use public access)")
        return None

    def _get_best_device(self) -> str:
        """Determine best available device from preferences.

        Returns:
            Device string ("cuda", "mps", or "cpu")
        """
        for device in self.device_preference:
            if device == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif device == "mps" and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def discover_models(self) -> list[ModelInfo]:
        """Discover cached HuggingFace models using metadata cache.

        Scans two locations:
        1. HuggingFace cache directory (~/.cache/huggingface/hub/)
        2. Local models directory (./models/)

        Returns:
            List of ModelInfo objects for cached models
        """
        models = []

        # Location 1: Scan HuggingFace cache directory for model directories
        if self.cache_dir.exists():
            try:
                # Look for model directories (models--<org>--<name> format)
                for model_dir in self.cache_dir.glob("models--*"):
                    if not model_dir.is_dir():
                        continue

                    # Parse model name from directory (models--org--name -> org/name)
                    dir_name = model_dir.name
                    if dir_name.startswith("models--"):
                        parts = dir_name[8:].split("--")
                        if len(parts) >= 2:
                            model_id = "/".join(parts)

                            # Get metadata from cache (reads config.json only, no weight loading)
                            metadata = self.metadata_cache.get_metadata(str(model_dir), provider="huggingface")

                            # Convert metadata to ModelInfo
                            if metadata:
                                # Map metadata model_type to ModelType enum
                                if metadata.model_type == "vlm":
                                    model_type = ModelType.VLM
                                    capabilities = [
                                        EndpointType.QA,
                                        EndpointType.CAPTION,
                                        EndpointType.DETECT,
                                        EndpointType.POINT,
                                        EndpointType.TEXT,
                                    ]
                                elif metadata.model_type == "embedding":
                                    model_type = ModelType.EMBEDDING
                                    capabilities = []
                                else:  # llm or unknown
                                    model_type = ModelType.LLM
                                    capabilities = [EndpointType.TEXT]

                                model_info = ModelInfo(
                                    model_id=model_id,
                                    name=model_id,
                                    provider="huggingface",
                                    size_gb=metadata.file_size_gb,
                                    architecture=metadata.architecture,
                                    quantization=metadata.quantization,
                                    params_billions=metadata.params_billions,
                                    ram_gb=metadata.ram_gb,
                                    vram_gb=metadata.vram_gb,
                                    params_exact=metadata.params_exact,
                                    ram_exact=metadata.ram_exact,
                                    model_type=model_type,
                                    capabilities=capabilities,
                                    compatibility=CompatibilityStatus.PERFECT_FIT,  # Will be assessed later
                                    compatibility_message="Compatibility not yet assessed",
                                    is_installed=True,
                                )
                            else:
                                # Fallback: metadata inspection failed, use basic info
                                logger.warning(f"Could not get metadata for {model_id}, using fallback")
                                size_gb = self._estimate_cached_model_size(model_dir)
                                model_info = ModelInfo(
                                    model_id=model_id,
                                    name=model_id,
                                    provider="huggingface",
                                    size_gb=size_gb,
                                    model_type=ModelType.LLM,
                                    capabilities=[EndpointType.TEXT],
                                    compatibility=CompatibilityStatus.PERFECT_FIT,
                                    compatibility_message="Compatibility not yet assessed",
                                    is_installed=True,
                                )

                            models.append(model_info)
                            logger.debug(f"Discovered HF model: {model_id} ({metadata.model_type if metadata else 'unknown'}, {metadata.quantization if metadata else 'unknown'})")

                logger.info(f"Discovered {len(models)} HuggingFace models from HF cache")
            except Exception as e:
                logger.error(f"Error discovering HuggingFace cache models: {e}")
        else:
            logger.warning(f"HuggingFace cache directory not found: {self.cache_dir}")

        # Location 2: Scan local ./models directory for HuggingFace format models
        local_models_dir = Path.cwd() / "models"
        if local_models_dir.exists():
            try:
                logger.info(f"Scanning local models directory: {local_models_dir}")
                for model_dir in local_models_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    # Check if it's a valid HuggingFace model (has config.json or safetensors)
                    has_config = (model_dir / "config.json").exists()
                    has_weights = list(model_dir.glob("*.safetensors")) or list(model_dir.glob("*.bin"))

                    if has_config or has_weights:
                        # Use directory name as model_id
                        model_id = model_dir.name
                        logger.info(f"Found local model: {model_id}")

                        # Get metadata
                        metadata = self.metadata_cache.get_metadata(str(model_dir), provider="huggingface")

                        if metadata:
                            # Map metadata model_type to ModelType enum
                            if metadata.model_type == "vlm":
                                model_type = ModelType.VLM
                                capabilities = [
                                    EndpointType.QA,
                                    EndpointType.CAPTION,
                                    EndpointType.DETECT,
                                    EndpointType.POINT,
                                    EndpointType.TEXT,
                                ]
                            elif metadata.model_type == "embedding":
                                model_type = ModelType.EMBEDDING
                                capabilities = []
                            else:  # llm or unknown
                                model_type = ModelType.LLM
                                capabilities = [EndpointType.TEXT]

                            model_info = ModelInfo(
                                model_id=str(model_dir),  # Use full path as model_id for local models
                                name=f"{model_id} (local)",
                                provider="huggingface",
                                size_gb=metadata.file_size_gb,
                                architecture=metadata.architecture,
                                quantization=metadata.quantization,
                                params_billions=metadata.params_billions,
                                ram_gb=metadata.ram_gb,
                                vram_gb=metadata.vram_gb,
                                params_exact=metadata.params_exact,
                                ram_exact=metadata.ram_exact,
                                model_type=model_type,
                                capabilities=capabilities,
                                compatibility=CompatibilityStatus.PERFECT_FIT,
                                compatibility_message="Compatibility not yet assessed",
                                is_installed=True,
                            )
                        else:
                            # Fallback: metadata inspection failed, use basic info
                            logger.warning(f"Could not get metadata for local model {model_id}, using fallback")
                            size_gb = self._estimate_cached_model_size(model_dir)
                            model_info = ModelInfo(
                                model_id=str(model_dir),  # Use full path as model_id
                                name=f"{model_id} (local)",
                                provider="huggingface",
                                size_gb=size_gb,
                                model_type=ModelType.LLM,
                                capabilities=[EndpointType.TEXT],
                                compatibility=CompatibilityStatus.PERFECT_FIT,
                                compatibility_message="Compatibility not yet assessed",
                                is_installed=True,
                            )

                        models.append(model_info)
                        logger.debug(f"Discovered local model: {model_id}")

                logger.info(f"Discovered {len([m for m in models if '(local)' in m.name])} models from ./models directory")
            except Exception as e:
                logger.error(f"Error discovering local models: {e}")

        logger.info(f"Total HuggingFace models discovered: {len(models)}")
        return models

    def _estimate_cached_model_size(self, model_dir: Path) -> float:
        """Estimate size of cached model in GB.

        Args:
            model_dir: Path to model cache directory

        Returns:
            Size in GB
        """
        try:
            total_size = 0
            for file in model_dir.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size / (1024**3)  # Convert to GB
        except Exception as e:
            logger.warning(f"Could not estimate size for {model_dir}: {e}")
            return 4.0  # Default estimate

    # DEPRECATED: This method is no longer used - metadata cache provides dynamic detection
    # Keeping for backward compatibility only (used in fallback scenarios)
    def _classify_hf_model(self, model_id: str) -> tuple[ModelType, list[EndpointType]]:
        """[DEPRECATED] Classify HuggingFace model by name patterns.

        This method is deprecated in favor of metadata cache which reads actual model config.
        Only used as a fallback when metadata inspection fails.

        Args:
            model_id: Model identifier (e.g., "llava-hf/llava-1.5-7b-hf")

        Returns:
            (ModelType, list of EndpointType)
        """
        model_lower = model_id.lower()

        # VLM keywords (fallback only)
        vlm_keywords = [
            "llava", "blip", "instructblip", "clip", "vision", "vit",
            "paligemma", "idefics", "fuyu", "kosmos", "qwen-vl",
            "cogvlm", "internvl", "minicpm-v", "phi-3-vision", "moondream"
        ]

        if any(keyword in model_lower for keyword in vlm_keywords):
            return (
                ModelType.VLM,
                [
                    EndpointType.QA,
                    EndpointType.CAPTION,
                    EndpointType.DETECT,
                    EndpointType.POINT,
                    EndpointType.TEXT,
                ],
            )

        # Embedding models
        if "embed" in model_lower or "sentence-transformer" in model_lower:
            return (ModelType.EMBEDDING, [])

        # Default to LLM
        return (ModelType.LLM, [EndpointType.TEXT])

    def load_model(self, model_id: str, device: str) -> Any:
        """Load HuggingFace model with transformers and automatic dependency management.

        Args:
            model_id: Model identifier (e.g., "llava-hf/llava-1.5-7b-hf")
            device: Target device (used self.device instead)

        Returns:
            Tuple of (model, processor/tokenizer)

        Raises:
            RuntimeError: If model cannot be loaded after all attempts
        """
        logger.info(f"Loading HuggingFace model: {model_id}")

        # Clean up previous model to free memory before loading new one
        self._cleanup_current_model()

        # Setup MPS memory management if needed
        self._setup_mps_memory_management()

        # Retry loop for iterative dependency installation
        max_retries = 5  # Prevent infinite loops
        all_installed_deps = []
        all_failed_deps = {}
        all_errors = []

        for attempt in range(max_retries):
            try:
                model, processor = self._load_model_internal(model_id)
                # Store references for cleanup
                self.current_model = model
                self.current_processor = processor
                return (model, processor)
            except Exception as e:
                error_msg = str(e)
                all_errors.append(f"Attempt {attempt + 1}: {error_msg}")

                if attempt == 0:
                    logger.error(f"Failed to load HuggingFace model {model_id}: {error_msg}")
                else:
                    logger.error(f"Retry attempt {attempt} failed: {error_msg}")

                # Log full error traceback for debugging
                import traceback
                full_traceback = traceback.format_exc()
                logger.debug(f"Full error traceback:\n{full_traceback}")

                # Try to handle missing dependencies automatically
                should_retry, installed_deps, failure_info = self.arch_registry.handle_model_load_error(
                    model_id=model_id,
                    error_message=error_msg,
                    auto_install=True
                )

                # Track all failed/skipped dependencies
                all_failed_deps.update(failure_info)

                if should_retry:
                    if installed_deps:
                        all_installed_deps.extend(installed_deps)
                        logger.info(
                            f"Installed dependencies: {installed_deps}. "
                            f"Retrying model load (attempt {attempt + 2}/{max_retries})..."
                        )
                    # Continue to next iteration to retry
                    continue
                else:
                    # No new dependencies to install - this is a real failure
                    self._show_model_incompatibility_error(
                        model_id=model_id,
                        all_errors=all_errors,
                        all_installed_deps=all_installed_deps,
                        all_failed_deps=all_failed_deps
                    )
                    raise RuntimeError(f"Failed to load model {model_id}: {e}")

        # Max retries exceeded
        logger.error(f"Max retries ({max_retries}) exceeded for model {model_id}")
        self._show_model_incompatibility_error(
            model_id=model_id,
            all_errors=all_errors,
            all_installed_deps=all_installed_deps,
            all_failed_deps=all_failed_deps
        )
        raise RuntimeError(
            f"Failed to load model {model_id} after {max_retries} attempts. "
            f"See logs for details."
        )

    def _show_model_incompatibility_error(
        self,
        model_id: str,
        all_errors: list[str],
        all_installed_deps: list[str],
        all_failed_deps: dict[str, str]
    ) -> None:
        """Show detailed error message when model cannot be loaded.

        Args:
            model_id: Model identifier
            all_errors: List of all error messages from attempts
            all_installed_deps: List of successfully installed dependencies
            all_failed_deps: Dict of failed/skipped dependencies with reasons
        """
        # Log comprehensive error information
        logger.error("=" * 80)
        logger.error(f"MODEL INCOMPATIBILITY REPORT: {model_id}")
        logger.error("=" * 80)

        # Check if this is an authentication issue
        has_auth_error = any(
            "gated" in error.lower() or "unauthorized" in error.lower() or
            "authentication" in error.lower() or "token" in error.lower() or
            "access denied" in error.lower() or "permission denied" in error.lower()
            for error in all_errors
        )

        # Check if it's a connection/download issue (might also need token)
        has_connection_error = any(
            "connection" in error.lower() or "cannot find" in error.lower() or
            "locate the file" in error.lower() or "network" in error.lower() or
            "timeout" in error.lower() or "unable to download" in error.lower()
            for error in all_errors
        )

        if has_auth_error or (has_connection_error and not self.hf_token):
            # Could be auth issue or connection that needs token
            logger.error("\n⚠️  POSSIBLE AUTHENTICATION ISSUE")
            logger.error("The model may require HuggingFace authentication.")
            logger.error(f"Current HF token: {'Set (from config/env)' if self.hf_token else 'NOT SET'}")

            # Try to get token from user interactively
            self._prompt_and_save_token()

        if all_installed_deps:
            logger.error(f"\nSuccessfully installed dependencies ({len(all_installed_deps)}):")
            for dep in all_installed_deps:
                logger.error(f"  ✓ {dep}")

        if all_failed_deps:
            logger.error(f"\nFailed/skipped dependencies ({len(all_failed_deps)}):")
            for dep, reason in all_failed_deps.items():
                logger.error(f"  ✗ {dep}: {reason}")

        logger.error(f"\nAll error messages ({len(all_errors)} attempts):")
        for i, error in enumerate(all_errors, 1):
            logger.error(f"  {i}. {error}")

        logger.error("=" * 80)

        # Show user-friendly TUI message
        try:
            from ..cli.tui_manager import tui

            error_panel = "[bold red]Model Loading Failed[/bold red]\n\n"
            error_panel += f"[bold]Model:[/bold] {model_id}\n\n"

            # Check if it's an authentication issue or connection issue
            if has_auth_error or (has_connection_error and not self.hf_token):
                # Auth/connection error already handled by _prompt_and_save_token()
                error_panel += "[bold yellow]Authentication or Connection Issue[/bold yellow]\n\n"
                error_panel += "A HuggingFace token may be needed.\n"
                error_panel += "Check the panel above to add your token.\n\n"
            else:
                error_panel += "[bold]This model is not compatible with your current system.[/bold]\n\n"
                error_panel += "Possible reasons:\n"
                if all_failed_deps:
                    error_panel += "[yellow]Failed dependencies:[/yellow]\n"
                    for dep, reason in all_failed_deps.items():
                        error_panel += f"  • [cyan]{dep}[/cyan]: {reason}\n"
                    error_panel += "\n"
                else:
                    error_panel += "  • Model requires CUDA (NVIDIA GPU) but you're on CPU/MPS\n"
                    error_panel += "  • Platform-specific dependencies unavailable\n"
                    error_panel += "  • Architecture not supported on this OS\n\n"

            if all_installed_deps:
                error_panel += f"[green]Successfully installed:[/green] {', '.join(all_installed_deps)}\n"

            error_panel += f"\n[dim]Full error details logged to: logs/vlm_cli_*.log[/dim]"

            tui.show_panel(error_panel, title="Model Loading Failed", border_style="red")
            tui.prompt("\nPress Enter to return to model selection...", style="dim")

        except ImportError:
            # Fallback if TUI not available
            print("\n" + "=" * 80)
            print(f"ERROR: Model {model_id} is not compatible with this system")
            print("=" * 80)
            if all_failed_deps:
                print("\nFailed dependencies:")
                for dep, reason in all_failed_deps.items():
                    print(f"  • {dep}: {reason}")
            print("\nCheck logs for full error details.")
            print("=" * 80 + "\n")

    def _load_model_internal(self, model_id: str) -> Any:
        """Internal method to load model (called by load_model and retry logic).

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (model, processor/tokenizer)
        """
        # Suppress transformers output during loading (keeps TUI clean)
        from ..utils.output_suppressor import suppress_transformers_output

        try:
            # Import base classes (always available)
            from transformers import AutoModel, AutoTokenizer

            # Setup token for private model access
            token_kwarg = {}
            if self.hf_token:
                token_kwarg = {"token": self.hf_token}
                logger.debug("Using HF token for model access")

            # Determine if this is a VLM using metadata cache
            # This checks the actual model config, not just keywords
            model_dir = self.cache_dir / ("models--" + model_id.replace("/", "--"))
            metadata = self.metadata_cache.get_metadata(str(model_dir), provider="huggingface")

            is_vlm = False
            if metadata:
                is_vlm = metadata.model_type == "vlm"
                logger.debug(f"Metadata indicates model type: {metadata.model_type}")
            else:
                # Fallback to keyword detection if metadata not available
                logger.debug("Metadata not available, using keyword fallback")
                model_lower = model_id.lower()
                vlm_keywords = [
                    "llava", "blip", "instructblip", "clip", "vision", "vl",
                    "paligemma", "idefics", "fuyu", "kosmos", "qwen-vl", "qwen3-vl",
                    "cogvlm", "internvl", "minicpm-v", "phi-3-vision", "moondream"
                ]
                is_vlm = any(keyword in model_lower for keyword in vlm_keywords)

            if is_vlm:
                # Try to load as VLM with processor (only import if needed)
                try:
                    # Import VLM-specific classes (may not be available in all versions)
                    from transformers import AutoConfig, AutoProcessor

                    # Suppress output during loading
                    with suppress_transformers_output():
                        processor = AutoProcessor.from_pretrained(
                            model_id,
                            cache_dir=str(self.cache_dir),
                            trust_remote_code=True,
                            **token_kwarg
                        )

                        # Prepare loading kwargs with proper dtype parameter
                        vlm_load_kwargs = {
                            "cache_dir": str(self.cache_dir),
                            "trust_remote_code": True,
                            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                            "low_cpu_mem_usage": True,  # Stream weights during loading
                            **token_kwarg,  # Add token for private model access
                        }

                        # 2025 OPTIMIZATION: Flash Attention 2/3 support for faster inference
                        # Reduces memory usage and improves speed (requires flash-attn package)
                        try:
                            import flash_attn  # noqa: F401
                            vlm_load_kwargs["attn_implementation"] = "flash_attention_2"
                            logger.info("✓ Flash Attention 2 enabled for faster VLM inference")
                        except ImportError:
                            logger.debug("Flash Attention not available (install flash-attn for 2-4x speedup)")
                            # Fallback to SDPA (Scaled Dot-Product Attention) - PyTorch 2.0+ native
                            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                                vlm_load_kwargs["attn_implementation"] = "sdpa"
                                logger.info("✓ Using PyTorch SDPA for optimized attention")

                        # Inspect config to derive best-fit auto classes dynamically
                        try:
                            config = AutoConfig.from_pretrained(
                                model_id,
                                cache_dir=str(self.cache_dir),
                                trust_remote_code=True,
                                **token_kwarg,
                            )
                        except Exception as config_error:
                            logger.debug(f"Failed to load config for VLM auto-class resolution: {config_error}")
                            config = None

                        model, attempted_classes, last_error = self._resolve_and_load_model(
                            model_id=model_id,
                            base_kwargs=vlm_load_kwargs,
                            processor=processor,
                            config=config,
                            is_vlm=True,
                        )

                        if model is None:
                            error_msg = (
                                f"Could not load VLM {model_id} with any compatible model class.\n"
                                f"Tried: {', '.join(attempted_classes)}\n"
                            )
                            if last_error:
                                error_msg += f"Last error: {str(last_error)[:300]}"
                            raise RuntimeError(error_msg)

                    # Determine target device - some models have MPS compatibility issues
                    target_device = self.device
                    if "qwen3" in model_id.lower() and self.device == "mps":
                        target_device = "cpu"
                        logger.warning(
                            f"Qwen3 has MPS compatibility issues. Using CPU instead. "
                            f"This may be slower but will work correctly."
                        )

                    # Move model to device with OOM fallback
                    actual_device = self._move_model_to_device(model, target_device)
                    logger.info(f"Loaded VLM model {model_id} on {actual_device}")
                    return (model, processor)
                except Exception as e:
                    logger.warning(f"Failed to load as VLM: {e}, trying as LLM...")

            # Load as LLM (suppress output)
            with suppress_transformers_output():
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        cache_dir=str(self.cache_dir),
                        trust_remote_code=True,
                        **token_kwarg
                    )
                except (ValueError, OSError) as e:
                    # Some models (audio, speech, etc.) may not have a tokenizer
                    # Try to find the base language model's tokenizer from config
                    error_str = str(e)
                    if "Unrecognized configuration class" in error_str or "does not appear to have" in error_str:
                        logger.warning(f"Model doesn't have a standard tokenizer: {e}")
                        logger.info("Attempting to find base language model tokenizer from config...")

                        # Try to read config and find the language model component
                        try:
                            from transformers import AutoConfig
                            import json

                            config = AutoConfig.from_pretrained(
                                model_id,
                                cache_dir=str(self.cache_dir),
                                trust_remote_code=True,
                                **token_kwarg
                            )

                            # Look for language_model or llm attribute in config
                            base_model = None
                            if hasattr(config, 'language_model'):
                                base_model = config.language_model
                            elif hasattr(config, 'text_config'):
                                if hasattr(config.text_config, 'model_type'):
                                    base_model = config.text_config.model_type
                            elif hasattr(config, 'llm_config'):
                                if hasattr(config.llm_config, 'model_type'):
                                    base_model = config.llm_config.model_type

                            if base_model:
                                logger.info(f"Found base model type: {base_model}")
                                # Try loading tokenizer for base model type
                                tokenizer = AutoTokenizer.from_pretrained(
                                    base_model if '/' in str(base_model) else f"google/{base_model}",
                                    cache_dir=str(self.cache_dir),
                                    trust_remote_code=True
                                )
                                logger.info(f"✓ Loaded tokenizer from base model: {base_model}")
                            else:
                                # No base model found - this model is not compatible
                                raise ValueError(
                                    f"Model {model_id} doesn't have a compatible tokenizer and "
                                    f"no base language model could be found in config. "
                                    f"This may be an audio-only or vision-only model."
                                )
                        except Exception as fallback_error:
                            logger.error(f"Fallback tokenizer loading failed: {fallback_error}")
                            raise ValueError(
                                f"Could not load tokenizer for {model_id}. "
                                f"Original error: {e}\n"
                                f"Fallback error: {fallback_error}"
                            )
                    else:
                        raise

                # Prepare loading kwargs with proper dtype parameter
                load_kwargs = {
                    "cache_dir": str(self.cache_dir),
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,  # Stream weights during loading
                    **token_kwarg,  # Add token for private model access
                }

                # Use 'dtype' instead of deprecated 'torch_dtype'
                if self.device == "cuda":
                    load_kwargs["torch_dtype"] = torch.float16
                else:
                    load_kwargs["torch_dtype"] = torch.float32

                # 2025 OPTIMIZATION: Flash Attention for LLMs
                try:
                    import flash_attn  # noqa: F401
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("✓ Flash Attention 2 enabled for faster inference")
                except ImportError:
                    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                        load_kwargs["attn_implementation"] = "sdpa"
                        logger.info("✓ Using PyTorch SDPA for optimized attention")

                # Dynamically resolve best auto-class for text models
                from transformers import AutoConfig

                try:
                    config = AutoConfig.from_pretrained(
                        model_id,
                        cache_dir=str(self.cache_dir),
                        trust_remote_code=True,
                        **token_kwarg,
                    )
                except Exception as config_error:
                    logger.debug(f"Failed to load config for LLM auto-class resolution: {config_error}")
                    config = None

                model, attempted_classes, last_error = self._resolve_and_load_model(
                    model_id=model_id,
                    base_kwargs=load_kwargs,
                    processor=tokenizer,
                    config=config,
                    is_vlm=False,
                )

                if model is None:
                    error_msg = (
                        f"Could not load model {model_id} with any compatible model class.\n"
                        f"Tried: {', '.join(attempted_classes)}\n"
                    )
                    if last_error:
                        error_msg += f"Last error: {str(last_error)[:300]}"
                    raise RuntimeError(error_msg)

            # Determine target device - some models have MPS compatibility issues
            target_device = self.device

            # Qwen3 has known MPS compatibility issues with torch 2.5.1
            # Force CPU for Qwen3 on MPS devices to avoid matrix dimension errors
            if "qwen3" in model_id.lower() and self.device == "mps":
                target_device = "cpu"
                logger.warning(
                    f"Qwen3 has MPS compatibility issues. Using CPU instead. "
                    f"This may be slower but will work correctly."
                )

            # Move model to device with OOM fallback
            actual_device = self._move_model_to_device(model, target_device)
            logger.info(f"Loaded LLM model {model_id} on {actual_device}")
            return (model, tokenizer)

        except Exception as e:
            # Let the parent load_model method handle errors and retries
            raise

    def _resolve_and_load_model(
        self,
        model_id: str,
        base_kwargs: dict[str, Any],
        processor: Any,
        config: Optional[Any],
        is_vlm: bool,
    ) -> tuple[Optional[Any], list[str], Optional[Exception]]:
        """Attempt to load a model by dynamically resolving the appropriate Auto* class.

        Args:
            model_id: Hugging Face model identifier.
            base_kwargs: Keyword arguments to pass to `.from_pretrained()`.
            processor: Loaded tokenizer or processor used to infer additional candidates.
            config: Optional `AutoConfig` instance for the model.
            is_vlm: Whether we are loading a vision-language model.

        Returns:
            Tuple of `(model, attempted_classes, last_error)`.
        """

        candidate_classes: list[tuple[str, str]] = []
        attempted_classes: list[str] = []
        last_error: Optional[Exception] = None

        # Helper to append candidate if not already present
        def add_candidate(class_path: str, reason: str) -> None:
            if class_path not in [c[0] for c in candidate_classes]:
                candidate_classes.append((class_path, reason))

        # Baseline candidates depending on model type
        if is_vlm:
            add_candidate(
                "transformers.AutoModelForVision2Seq",
                "Vision-to-text architectures (e.g., Qwen-VL, LFM2-VL)",
            )
            add_candidate(
                "transformers.AutoModelForCausalLM",
                "VLMs with causal decoder heads (e.g., LLaVA, Moondream)",
            )
            add_candidate(
                "transformers.AutoModelForUniversalSeg",
                "Multimodal segmenters with text decoders",
            )
            add_candidate(
                "transformers.AutoModel",
                "Generic fallback for custom VLM architectures",
            )
        else:
            add_candidate(
                "transformers.AutoModelForCausalLM",
                "Causal language models (default)",
            )
            add_candidate(
                "transformers.AutoModelForSeq2SeqLM",
                "Seq2seq text models",
            )
            add_candidate(
                "transformers.AutoModel",
                "Generic fallback",
            )

        # Enrich candidates from config information
        if config is not None:
            architectures = getattr(config, "architectures", None) or []
            model_type = getattr(config, "model_type", None)
            text_config = getattr(config, "text_config", None)
            language_model = getattr(config, "language_model", None)

            derived_types = []

            for arch in architectures:
                if arch:
                    derived_types.append(arch)
            if model_type:
                derived_types.append(model_type)
            if text_config and hasattr(text_config, "arch_type"):
                derived_types.append(text_config.arch_type)
            if isinstance(language_model, str):
                derived_types.append(language_model)

            for derived in derived_types:
                if not derived:
                    continue
                derived_lower = derived.lower()
                if "vision" in derived_lower or "vl" in derived_lower or "multimodal" in derived_lower:
                    add_candidate(
                        "transformers.AutoModelForVision2Seq",
                        f"Detected vision language architecture '{derived}'",
                    )
                if "causal" in derived_lower or derived_lower.endswith("forcausallm"):
                    add_candidate(
                        "transformers.AutoModelForCausalLM",
                        f"Detected causal LM architecture '{derived}'",
                    )
                if "seq2seq" in derived_lower or derived_lower.endswith("forconditionalgeneration"):
                    add_candidate(
                        "transformers.AutoModelForSeq2SeqLM",
                        f"Detected seq2seq architecture '{derived}'",
                    )

                # Specialized handling for LiquidAI LFM2-VL family
                if "lfm2" in derived_lower or derived_lower == "lfm2_vlforconditionalgeneration":
                    add_candidate(
                        "transformers.AutoModelForImageTextToText",
                        "LFM2-VL requires AutoModelForImageTextToText (transformers>=4.57)",
                    )
                    add_candidate(
                        "transformers.models.lfm2_vl.modeling_lfm2_vl.Lfm2VlForConditionalGeneration",
                        "Direct LFM2-VL implementation fallback",
                    )

            if model_type and model_type.lower() == "lfm2_vl":
                add_candidate(
                    "transformers.AutoModelForImageTextToText",
                    "Model type 'lfm2_vl' detected",
                )
                add_candidate(
                    "transformers.models.lfm2_vl.modeling_lfm2_vl.Lfm2VlForConditionalGeneration",
                    "Direct LFM2-VL implementation fallback",
                )

        # Add fallback derived from tokenizer/processor type
        if processor is not None:
            proc_class = processor.__class__.__name__.lower()
            if "processor" in proc_class and is_vlm:
                add_candidate(
                    "transformers.AutoModelForVision2Seq",
                    f"Processor {processor.__class__.__name__} suggests VLM",
                )

        model: Optional[Any] = None

        for class_path, reason in candidate_classes:
            attempted_classes.append(class_path.split(".")[-1])
            try:
                module_name, class_name = class_path.rsplit(".", 1)
                module = import_module(module_name)
                auto_class = getattr(module, class_name)

                logger.debug(f"Trying {class_name} for {model_id}: {reason}")

                try:
                    model = auto_class.from_pretrained(model_id, **base_kwargs)
                except (TypeError, ValueError) as param_error:
                    logger.debug(
                        f"{class_name} full kwargs failed, retrying minimal set: {param_error}"
                    )
                    minimal_kwargs = {
                        "cache_dir": base_kwargs.get("cache_dir"),
                        "trust_remote_code": True,
                    }
                    if "torch_dtype" in base_kwargs:
                        minimal_kwargs["torch_dtype"] = base_kwargs["torch_dtype"]
                    model = auto_class.from_pretrained(model_id, **minimal_kwargs)

                if not hasattr(model, "generate"):
                    logger.debug(f"{class_name} loaded but missing .generate(); skipping")
                    model = None
                    continue

                logger.info(f"✓ Loaded {model_id} with {class_name}")
                return model, attempted_classes, None

            except Exception as load_error:
                last_error = load_error
                logger.debug(f"{class_path} failed for {model_id}: {load_error}")
                continue

        return None, attempted_classes, last_error

    def unload_model(self, handle: Any) -> None:
        """Unload model and free GPU memory universally across all platforms.

        Args:
            handle: Tuple of (model, processor/tokenizer)
        """
        if handle is None:
            return

        try:
            model, _ = handle
            del model

            # Universal GPU memory cleanup - works on all platforms
            if torch.cuda.is_available():
                # NVIDIA CUDA (Windows, Linux, Jetson)
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
            elif torch.backends.mps.is_available():
                # Apple Metal (macOS with M-series)
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache")
            # Note: ROCm (AMD) uses same API as CUDA
            # CPU doesn't need explicit cache clearing

            logger.info("HuggingFace model unloaded")
        except Exception as e:
            logger.warning(f"Error unloading HuggingFace model: {e}")

    def run_qa(
        self,
        handle: Any,
        image: Image.Image,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None
    ) -> str:
        """Run question answering with universal inference handler.

        Supports both VLM (with images) and LLM (text only) models transparently.
        Handles processor/tokenizer differences automatically.

        Args:
            handle: Tuple of (model, processor)
            image: PIL Image (VLM only, can be None for LLM)
            question: Question text
            conversation_history: Optional list of (user_msg, bot_response) tuples

        Returns:
            Answer text
        """
        from ..core.inference_handler import UniversalInferenceHandler
        from ..models.endpoints import ModelType

        model, processor = handle

        try:
            # Determine model type: check BOTH model class AND processor type
            # A model might be a VLM but with wrong processor, so we check both
            model_class_name = type(model).__name__.lower()

            # VLM indicators in model class
            is_vlm_model = any(
                keyword in model_class_name
                for keyword in [
                    "vision", "vlm", "multimodal", "llava", "blip",
                    "qwen2vl", "qwen3vl", "internvl", "minicpm-v",
                    "moondream", "idefics", "clip"
                ]
            )

            # VLM indicators in processor
            is_vlm_processor = (
                hasattr(processor, "apply_chat_template")
                or "processor" in type(processor).__name__.lower()
            )

            # Trust the model class first - it's more reliable
            is_vlm = is_vlm_model or is_vlm_processor
            model_type = ModelType.VLM if is_vlm else ModelType.LLM

            logger.debug(
                f"Model type detection: "
                f"model_class={model_class_name}, is_vlm_model={is_vlm_model}, "
                f"processor={type(processor).__name__}, is_vlm_processor={is_vlm_processor}, "
                f"final={model_type.value}"
            )

            # Use universal handler that works with all model types
            handler = UniversalInferenceHandler(
                model=model,
                processor=processor,
                model_type=model_type
            )

            # Run inference with universal handler
            answer = handler.run_qa(
                image=image if is_vlm else None,
                question=question,
                conversation_history=conversation_history,
                max_new_tokens=1024
            )

            return answer

        except Exception as e:
            logger.error(f"QA inference failed: {e}", exc_info=True)
            raise

    def run_caption(
        self,
        handle: Any,
        image: Image.Image,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        detail_level: str = "detailed"
    ) -> str:
        """Generate image caption.

        Args:
            handle: Tuple of (model, processor)
            image: PIL Image
            conversation_history: Optional list of (user_msg, bot_response) tuples
            detail_level: "detailed" or "short"

        Returns:
            Caption text
        """
        prompt = (
            "Describe this image in detail."
            if detail_level == "detailed"
            else "Describe this image briefly."
        )
        return self.run_qa(handle, image, prompt, conversation_history)

    def run_detect(self, handle: Any, image: Image.Image, object_name: str) -> list[dict]:
        """Detect objects with VLM and parse bounding boxes.

        Args:
            handle: Tuple of (model, processor)
            image: PIL Image
            object_name: Object to detect

        Returns:
            List of detection dicts with parsed bounding boxes
        """
        from ..utils.vlm_response_parser import parse_detection_response

        prompt = (
            f"Detect all instances of '{object_name}' in this image. "
            f"Provide bounding box coordinates in JSON format as: "
            f'[{{"bbox": [x1, y1, x2, y2], "label": "{object_name}"}}]'
        )
        response = self.run_qa(handle, image, prompt)

        # Parse response to extract actual bounding boxes
        detections = parse_detection_response(response, object_name)

        # Add raw response to each detection for debugging
        for detection in detections:
            detection["raw"] = response

        return detections

    def run_point(self, handle: Any, image: Image.Image, object_name: str) -> dict:
        """Point to object location with VLM and parse coordinates.

        Args:
            handle: Tuple of (model, processor)
            image: PIL Image
            object_name: Object to locate

        Returns:
            Coordinates dict with parsed x, y values
        """
        from ..utils.vlm_response_parser import parse_point_response

        prompt = (
            f"Where is the '{object_name}' in this image? "
            f"Provide the center coordinates in JSON format as: "
            f'{{"x": <number>, "y": <number>}}'
        )
        response = self.run_qa(handle, image, prompt)

        # Parse response to extract actual coordinates
        coordinates = parse_point_response(response, object_name)

        # Add raw response for debugging
        coordinates["raw"] = response

        return coordinates

    def run_text(
        self,
        handle: Any,
        prompt: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        custom_parameters: Optional[dict] = None
    ) -> str:
        """Generate text response for LLM using tokenizer's built-in chat template.

        Uses the tokenizer's apply_chat_template() method which automatically
        formats conversations based on the model's native template. This is more
        reliable than manual template detection.

        Args:
            handle: Tuple of (model, tokenizer)
            prompt: Current text prompt
            conversation_history: Optional list of (user_msg, bot_response) tuples
            custom_parameters: Optional dict of custom generation parameters
                              (max_tokens, temperature, top_p, top_k, repeat_penalty, seed)

        Returns:
            Generated text
        """
        model, tokenizer = handle

        try:
            # UNIVERSAL DEVICE DETECTION - works on all platforms
            # Get the actual device the model is on (not self.device which may differ)
            # This handles: CUDA (Windows/Linux), MPS (Mac), CPU (all platforms)
            model_device = next(model.parameters()).device
            logger.debug(f"Model is on device: {model_device}")

            # Set padding token if not present (needed for some models like DialoGPT)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Build messages array (same format as OpenAI/Ollama)
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Provide clear, accurate, and concise responses."}
            ]

            # Add conversation history (limit to last 5 turns)
            if conversation_history:
                recent_history = conversation_history[-5:]
                logger.debug(f"Including {len(recent_history)} previous conversation turns")

                for user_msg, assistant_msg in recent_history:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})

            # Add current user message
            messages.append({"role": "user", "content": prompt})

            # Use tokenizer's built-in chat template (knows model's native format)
            try:
                # Try to use apply_chat_template (available in modern transformers)
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # Adds assistant prefix for response
                )
                logger.debug(f"Using tokenizer's native chat template")
            except Exception as e:
                # Fallback to manual formatting if tokenizer doesn't have chat template
                logger.warning(f"Tokenizer has no chat template, using fallback: {e}")

                # Extract model name for fallback template detection
                model_name = getattr(model.config, "_name_or_path", "")
                formatted_prompt = format_conversation_history(
                    conversation_history=conversation_history,
                    current_prompt=prompt,
                    max_turns=5,
                    system_prompt=None,
                    model_name=model_name
                )

            # Tokenize input and move to the SAME device as the model
            # This works universally: CUDA (NVIDIA), MPS (Apple), CPU (all), ROCm (AMD), etc.
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(model_device)
            
            # Debug: Log what tokenizer produced
            logger.debug(f"Tokenizer outputs: {list(inputs.keys())}")
            logger.debug(f"input_ids shape: {inputs.get('input_ids').shape if inputs.get('input_ids') is not None else 'None'}")
            logger.debug(f"attention_mask shape: {inputs.get('attention_mask').shape if inputs.get('attention_mask') is not None else 'None'}")

            # GENERIC FIX: Filter inputs to only include parameters the model's forward() accepts
            # The generate() method passes **inputs to forward() as model_kwargs
            # Some models (e.g., OLMo) don't accept token_type_ids
            import inspect
            try:
                # CRITICAL: Check model.forward() signature, not generate()
                # generate() passes inputs to forward(), so we need forward's signature
                if hasattr(model, 'forward'):
                    sig = inspect.signature(model.forward)
                    
                    # Get accepted parameter names
                    accepted_params = set(sig.parameters.keys())
                    
                    # Check if forward accepts **kwargs (VAR_KEYWORD)
                    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD 
                                   for p in sig.parameters.values())
                    
                    # ALWAYS keep input_ids and attention_mask (essential for generation)
                    # Filter out parameters not present in the forward signature
                    logger.debug(f"Model forward() signature: {accepted_params}")
                    filtered_inputs = {}

                    essential_params = {'input_ids', 'attention_mask'}
                    allowed_params = set(accepted_params) | essential_params

                    for key, value in inputs.items():
                        if key in allowed_params:
                            filtered_inputs[key] = value
                        else:
                            logger.debug(f"Filtering out unsupported parameter: {key}")

                    if not filtered_inputs:
                        # Safety: never leave inputs empty
                        filtered_inputs = {'input_ids': inputs['input_ids']}

                    inputs = filtered_inputs
                    logger.debug(f"Filtered inputs to: {list(inputs.keys())}")
                else:
                    # Fallback: model has no forward method (unlikely)
                    logger.warning("Model has no forward method, using minimal inputs")
                    inputs = {'input_ids': inputs['input_ids']}
                    
            except Exception as e:
                logger.debug(f"Could not filter inputs, using fallback: {e}")
                # Fallback: try with only essential parameters
                try:
                    essential_params = {'input_ids', 'attention_mask'}
                    filtered_inputs = {k: v for k, v in inputs.items() if k in essential_params}
                    if filtered_inputs:
                        inputs = filtered_inputs
                        logger.debug("Fallback: using only input_ids and attention_mask")
                except:
                    # Last resort: keep all inputs (backward compatible)
                    logger.debug("Fallback failed, passing all inputs")
                    pass

            # Build generation parameters with defaults
            # CRITICAL: Check if model supports use_cache properly
            # Some models (e.g., OLMo) don't handle None past_key_values correctly
            model_name_or_path = getattr(model.config, "_name_or_path", "").lower()
            model_class_name = model.__class__.__name__.lower()
            
            # Disable use_cache for models that don't handle it properly
            supports_use_cache = True
            if "olmo" in model_name_or_path or "olmo" in model_class_name:
                supports_use_cache = False
                logger.debug("Disabled use_cache for OLMo model (doesn't handle None past_key_values)")
            
            gen_params = {
                "max_new_tokens": 1024,  # Increased for more detailed responses
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "use_cache": supports_use_cache,  # Disabled for models that don't support it
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            # Override with custom parameters from session
            if custom_parameters:
                if "max_tokens" in custom_parameters:
                    gen_params["max_new_tokens"] = custom_parameters["max_tokens"]
                if "temperature" in custom_parameters:
                    gen_params["temperature"] = custom_parameters["temperature"]
                if "top_p" in custom_parameters:
                    gen_params["top_p"] = custom_parameters["top_p"]
                if "top_k" in custom_parameters:
                    gen_params["top_k"] = custom_parameters["top_k"]
                if "repeat_penalty" in custom_parameters:
                    gen_params["repetition_penalty"] = custom_parameters["repeat_penalty"]
                if "seed" in custom_parameters:
                    # Set seed for reproducibility
                    import random
                    import numpy as np
                    seed = custom_parameters["seed"]
                    torch.manual_seed(seed)
                    random.seed(seed)
                    np.random.seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                logger.debug(f"Using custom parameters: {custom_parameters}")

            # Debug: Log final inputs going to generate
            logger.debug(f"Final inputs to model.generate(): {list(inputs.keys())}")
            for key in inputs.keys():
                tensor = inputs[key]
                logger.debug(f"  {key}: shape={tensor.shape if tensor is not None else 'None'}, "
                           f"dtype={tensor.dtype if tensor is not None else 'None'}")

            # Generate with parameters
            with torch.no_grad():
                try:
                    output = model.generate(**inputs, **gen_params)
                except Exception as gen_error:
                    logger.error(f"model.generate() failed with inputs: {list(inputs.keys())}")
                    logger.error(f"Generation params: {list(gen_params.keys())}")
                    import traceback
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    raise

            # Decode only the new tokens (exclude input prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_length:]

            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Clean up response - remove any remaining EOS tokens or extra whitespace
            if tokenizer.eos_token:
                response = response.replace(tokenizer.eos_token, "").strip()

            # Fallback: if empty, decode full output and try to remove prompt
            if not response:
                full_response = tokenizer.decode(output[0], skip_special_tokens=True)
                if full_response.startswith(formatted_prompt):
                    response = full_response[len(formatted_prompt):].strip()
                else:
                    response = full_response.strip()

            # Apply response cleaning to remove artifacts and meta-commentary
            from ..utils.response_cleaner import clean_model_response
            if response:
                response = clean_model_response(response, aggressive=True)

            return response if response else "I don't have a response."

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")

    def get_model_info(self, model_id: str) -> dict:
        """Get detailed model information for HuggingFace models.

        Args:
            model_id: Model identifier (e.g., "llava-hf/llava-1.5-7b-hf")

        Returns:
            Dict with model metadata including default_parameters
        """
        # Find the model in discovered models
        models = self.discover_models()
        model = next((m for m in models if m.model_id == model_id), None)

        if not model:
            # Return minimal info if not found
            return {
                "model_id": model_id,
                "name": model_id,
                "provider": "huggingface",
                "default_parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "max_tokens": 1024,  # Increased for more detailed responses
                    "repeat_penalty": 1.0,
                }
            }

        # Return full model info
        return {
            "model_id": model.model_id,
            "name": model.name,
            "provider": "huggingface",
            "architecture": model.architecture or "Unknown",
            "quantization": model.quantization or "None",
            "size_gb": model.size_gb,
            "model_type": str(model.model_type).upper() if hasattr(model.model_type, 'value') else str(model.model_type).upper(),
            "capabilities": [str(cap) for cap in model.capabilities] if model.capabilities else ["text"],
            "default_parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "max_tokens": 1024,  # Increased for more detailed responses
                "repeat_penalty": 1.0,
            }
        }

    def install_model(self, model_name: str, progress_callback=None) -> bool:
        """Download model from HuggingFace Hub with visible progress bars.

        Handles both regular transformers models and GGUF models:
        - GGUF repos: Downloads .gguf files directly (no transformers validation)
        - Regular models: Downloads via transformers (requires valid config.json)
        
        Progress bars are shown during download for transparency.
        No output suppression is used during installation.

        Args:
            model_name: Model identifier (e.g., "llava-hf/llava-1.5-7b-hf" or "TheBloke/Llama-2-7B-GGUF")
            progress_callback: Optional progress callback (not implemented)

        Returns:
            True if successful
        """
        logger.info(f"Downloading HuggingFace model: {model_name}")
        
        # NOTE: Do NOT use suppress_transformers_output() here!
        # We want users to see download progress bars during installation.

        try:
            from huggingface_hub import list_repo_files, hf_hub_download

            # Setup token kwargs for authenticated access
            token_kwarg = {}
            if self.hf_token:
                token_kwarg = {"token": self.hf_token}
                logger.debug("Using HF token for model download")

            # STEP 1: Check if this is a GGUF repository
            # GGUF repos contain .gguf files and should be downloaded directly
            try:
                repo_files = list_repo_files(model_name, **token_kwarg)
                gguf_files = [f for f in repo_files if f.endswith('.gguf')]

                if gguf_files:
                    logger.info(f"Detected GGUF repository with {len(gguf_files)} .gguf file(s)")
                    logger.info(f"Downloading GGUF files directly (skipping transformers validation)")

                    # Get file sizes for better UX
                    from huggingface_hub import HfFileSystem
                    fs = HfFileSystem()
                    file_sizes = {}
                    try:
                        files_info = fs.ls(model_name, detail=True)
                        for finfo in files_info:
                            fname = finfo['name'].split('/')[-1]
                            if fname.endswith('.gguf'):
                                file_sizes[fname] = finfo['size'] / (1024**3)  # GB
                    except:
                        pass

                    # Smart file selection: prioritize smaller quantized versions
                    # Download order: Q4_K_M > Q5_K_M > Q8_0 > F16 (smallest to largest)
                    quantization_priority = {
                        'q2_k': 1, 'q3_k_m': 2, 'q4_k_m': 3, 'q4_k_s': 4,
                        'q5_k_m': 5, 'q5_k_s': 6, 'q6_k': 7, 'q8_0': 8,
                        'f16': 99, 'f32': 100  # Full precision last
                    }

                    def get_priority(filename):
                        fname_lower = filename.lower()
                        for quant_type, priority in quantization_priority.items():
                            if quant_type in fname_lower:
                                return priority
                        return 50  # Unknown quantization

                    # Sort files by priority (download best quantized version first)
                    sorted_files = sorted(gguf_files, key=get_priority)

                    # Determine download directory: use GGUF models directory from config
                    # Read from config.yaml to get the correct GGUF models directory
                    from pathlib import Path
                    import yaml

                    gguf_dir = Path.home() / "models" / "gguf"  # Default
                    try:
                        config_path = Path("config.yaml")
                        if config_path.exists():
                            with open(config_path) as f:
                                cfg = yaml.safe_load(f)
                                if cfg and 'providers' in cfg and 'gguf' in cfg['providers']:
                                    gguf_models_dir = cfg['providers']['gguf'].get('models_dir', '~/models/gguf/')
                                    gguf_dir = Path(gguf_models_dir).expanduser()
                    except Exception as e:
                        logger.debug(f"Could not read GGUF models_dir from config, using default: {e}")

                    # Download to GGUF provider's directory so it can be discovered
                    download_dir = gguf_dir / model_name.replace("/", "--")
                    download_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Downloading to GGUF models directory: {download_dir}")

                    # Download the recommended file (smallest good quantization)
                    best_file = sorted_files[0]
                    file_size_str = f" ({file_sizes.get(best_file, 0):.2f} GB)" if best_file in file_sizes else ""
                    logger.info(f"Downloading recommended quantization: {best_file}{file_size_str}")

                    if len(sorted_files) > 1:
                        logger.info(f"Note: {len(sorted_files)-1} other quantization(s) available but not downloaded to save space")
                        other_files = [f"{f} ({file_sizes.get(f, 0):.1f}GB)" if f in file_sizes else f
                                     for f in sorted_files[1:]]
                        logger.info(f"Other versions: {', '.join(other_files)}")

                    # Download with visible progress bar
                    # hf_hub_download shows tqdm progress by default (not suppressed)
                    downloaded_path = hf_hub_download(
                        repo_id=model_name,
                        filename=best_file,
                        local_dir=str(download_dir),
                        local_dir_use_symlinks=False,  # Direct copy for GGUF compatibility
                        **token_kwarg
                    )
                    logger.info(f"Downloaded {best_file} successfully")
                    logger.info(f"Location: {downloaded_path}")
                    logger.info(f"Use the GGUF or Quantized provider to load this model")

                    # Invalidate metadata cache for GGUF provider (will be re-inspected on discovery)
                    # This ensures the new model shows up with correct metadata
                    self.metadata_cache.invalidate_model(str(download_dir))
                    logger.debug(f"Invalidated GGUF metadata cache for {model_name}")

                    return True

            except Exception as e:
                # If we can't check repo files, assume it's a regular model
                logger.debug(f"Could not check for GGUF files: {e}")

            # STEP 2: Regular transformers model download
            # Import base classes (always available)
            from transformers import AutoModel, AutoTokenizer

            model_lower = model_name.lower()

            # Check if VLM (vision-language model)
            vlm_keywords = [
                "llava", "blip", "instructblip", "vision", "vl", "clip",
                "paligemma", "idefics", "fuyu", "kosmos", "qwen-vl", "qwen3-vl", "moondream"
            ]
            is_vlm = any(keyword in model_lower for keyword in vlm_keywords)

            if is_vlm:
                # Download processor for VLM (with visible progress)
                try:
                    from transformers import AutoProcessor
                    # Progress bars shown during download
                    AutoProcessor.from_pretrained(
                        model_name,
                        cache_dir=str(self.cache_dir),
                        trust_remote_code=True,
                        **token_kwarg
                    )
                    logger.info("Downloaded VLM processor")
                except Exception as e:
                    logger.debug(f"Could not download processor: {e}")

            # Download tokenizer (with visible progress)
            try:
                # Progress bars shown during download
                AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                    **token_kwarg
                )
                logger.info("Downloaded tokenizer")
            except Exception as e:
                logger.debug(f"Could not download tokenizer: {e}")

            # Download model weights (with visible progress)
            # Use snapshot_download for robust handling of all model types:
            # - Models with custom configurations (e.g., moondream2)
            # - Standard transformers models
            # - VLMs and LLMs
            # This avoids configuration class validation errors while still downloading all files
            from huggingface_hub import snapshot_download
            
            # Progress bars shown during download
            snapshot_download(
                repo_id=model_name,
                cache_dir=str(self.cache_dir),
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.py"],
                ignore_patterns=["*.gguf", "*.md", "*.git*"],
                **token_kwarg
            )

            logger.info(f"Successfully downloaded {model_name}")

            # Invalidate cache for this model so it gets re-inspected on next discovery
            model_dir = self.cache_dir / ("models--" + model_name.replace("/", "--"))
            if model_dir.exists():
                self.metadata_cache.invalidate_model(str(model_dir))
                logger.debug(f"Invalidated metadata cache for {model_name}")

            return True

        except Exception as e:
            error_str = str(e)
            logger.error(f"Failed to download {model_name}: {error_str}")
            
            # Check if this is an authentication/access error
            is_auth_error = any([
                "gated" in error_str.lower(),
                "unauthorized" in error_str.lower(),
                "authentication" in error_str.lower(),
                "token" in error_str.lower(),
                "access denied" in error_str.lower(),
                "permission denied" in error_str.lower(),
                "403" in error_str,
                "401" in error_str
            ])
            
            # Check if it's a connection error that might need auth
            is_connection_error = any([
                "connection" in error_str.lower(),
                "cannot find" in error_str.lower(),
                "locate the file" in error_str.lower(),
                "network" in error_str.lower(),
                "timeout" in error_str.lower(),
                "unable to download" in error_str.lower()
            ])
            
            # If auth error OR connection error without token, prompt for token
            if is_auth_error or (is_connection_error and not self.hf_token):
                logger.error("\n⚠️  AUTHENTICATION REQUIRED")
                logger.error(f"Model {model_name} may require HuggingFace authentication.")
                logger.error(f"Current HF token: {'Set' if self.hf_token else 'NOT SET'}")
                
                # Try to get token from user interactively
                token_saved = self._prompt_and_save_token()
                
                # Show error panel with guidance
                try:
                    from ..cli.tui_manager import tui
                    
                    error_panel = "[bold red]Model Download Failed[/bold red]\n\n"
                    error_panel += f"[bold]Model:[/bold] {model_name}\n\n"
                    
                    if token_saved:
                        error_panel += "[bold green]✓ Token Saved![/bold green]\n\n"
                        error_panel += "[yellow]Please restart the application to use the new token.[/yellow]\n\n"
                        error_panel += "Steps:\n"
                        error_panel += "  1. Exit this application (Ctrl+C)\n"
                        error_panel += "  2. Run ./run.sh again\n"
                        error_panel += "  3. Try installing the model again\n"
                    else:
                        error_panel += "[bold yellow]Authentication Issue[/bold yellow]\n\n"
                        error_panel += "This model requires HuggingFace authentication.\n\n"
                        error_panel += "[bold]To fix this:[/bold]\n"
                        error_panel += "  1. Get a token from: https://huggingface.co/settings/tokens\n"
                        error_panel += "  2. Restart the app and try again\n"
                        error_panel += "  3. You'll be prompted to enter your token\n"
                    
                    error_panel += f"\n[dim]Error: {error_str[:200]}...[/dim]"
                    
                    tui.show_panel(error_panel, title="Authentication Required", border_style="yellow")
                    tui.prompt("\nPress Enter to continue...", style="dim")
                    
                except ImportError:
                    print("\n" + "="*80)
                    print(f"ERROR: {model_name} requires authentication")
                    if token_saved:
                        print("Token saved! Please restart the application.")
                    else:
                        print("Get token from: https://huggingface.co/settings/tokens")
                    print("="*80 + "\n")
            
            return False

    def delete_model(self, model_id: str) -> bool:
        """Delete model from disk (dynamically detects local vs cached models).

        Args:
            model_id: Model identifier (can be HF repo name like "org/model" or full path)

        Returns:
            True if successful
        """
        logger.info(f"Deleting HuggingFace model: {model_id}")

        try:
            from pathlib import Path
            import shutil
            import subprocess
            import platform

            # Check if model_id is a path (local model) or a HF repo name (cached model)
            model_path = Path(model_id)

            if model_path.exists() and model_path.is_dir():
                # Local model - model_id is the full path
                # Use rm -rf on macOS/Linux for better handling of AppleDouble files
                if platform.system() in ["Darwin", "Linux"]:
                    try:
                        subprocess.run(
                            ["rm", "-rf", str(model_path)],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        logger.info(f"Deleted local model directory (rm -rf): {model_path}")
                        return True
                    except subprocess.CalledProcessError as e:
                        logger.error(f"rm -rf failed: {e.stderr}")
                        # Fall back to shutil
                        shutil.rmtree(model_path)
                        logger.info(f"Deleted local model directory (shutil): {model_path}")
                        return True
                else:
                    # Windows - use shutil
                    shutil.rmtree(model_path)
                    logger.info(f"Deleted local model directory: {model_path}")
                    return True
            else:
                # HuggingFace cached model - model_id is like "org/name"
                # Convert to cache directory name: org/name -> models--org--name
                dir_name = "models--" + model_id.replace("/", "--")
                cache_dir = self.cache_dir / dir_name

                if cache_dir.exists():
                    # Use rm -rf on macOS/Linux, shutil on Windows
                    if platform.system() in ["Darwin", "Linux"]:
                        try:
                            subprocess.run(
                                ["rm", "-rf", str(cache_dir)],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            logger.info(f"Deleted cached model (rm -rf): {model_id} from {cache_dir}")
                            return True
                        except subprocess.CalledProcessError:
                            # Fall back to shutil
                            shutil.rmtree(cache_dir)
                            logger.info(f"Deleted cached model (shutil): {model_id} from {cache_dir}")
                            return True
                    else:
                        shutil.rmtree(cache_dir)
                        logger.info(f"Deleted cached model: {model_id} from {cache_dir}")
                        return True
                else:
                    logger.warning(f"Model not found. Tried local: {model_path}, cache: {cache_dir}")
                    return False

        except Exception as e:
            logger.error(f"Failed to delete {model_id}: {e}")
            return False

    def estimate_model_size(self, model_name: str) -> float:
        """Estimate model size (rough estimate based on name).

        Args:
            model_name: Model identifier

        Returns:
            Estimated size in GB
        """
        # Rough estimates based on common model sizes
        name_lower = model_name.lower()

        if "7b" in name_lower:
            return 14.0  # 7B models ~14GB fp16
        elif "13b" in name_lower:
            return 26.0
        elif "3b" in name_lower:
            return 6.0
        elif "1b" in name_lower:
            return 2.0
        else:
            return 8.0  # Default estimate

    # Convenience methods for app.py
    def qa(self, model_id: str, image_path: str, question: str) -> str:
        """QA convenience wrapper (requires manual handle management)."""
        raise NotImplementedError("Use load_model() then run_qa() directly")

    def caption(self, model_id: str, image_path: str, detail_level: str = "detailed") -> str:
        """Caption convenience wrapper (requires manual handle management)."""
        raise NotImplementedError("Use load_model() then run_caption() directly")

    def detect(self, model_id: str, image_path: str, prompt: str) -> str:
        """Detect convenience wrapper (requires manual handle management)."""
        raise NotImplementedError("Use load_model() then run_detect() directly")

    def point(self, model_id: str, image_path: str, object_name: str) -> str:
        """Point convenience wrapper (requires manual handle management)."""
        raise NotImplementedError("Use load_model() then run_point() directly")

    def chat(self, model_id: str, message: str) -> str:
        """Chat convenience wrapper (requires manual handle management)."""
        raise NotImplementedError("Use load_model() then run_text() directly")
