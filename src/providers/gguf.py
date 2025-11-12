"""GGUF provider implementation using llama-server for GPU acceleration."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from PIL import Image

from ..models.endpoints import CompatibilityStatus, EndpointType, ModelType
from ..models.model import ModelInfo
from ..models.model_cache import ModelMetadataCache
from ..models.provider import ProviderConfig
from ..models.system import SystemSpecs
from ..services.llama_server_manager import LlamaServerManager
from ..services.llama_cli_server_manager import LlamaCLIServerManager
from ..config.server_config import get_llama_server_type
from ..utils.history_formatter import format_conversation_history, format_qa_history
from .base import BaseProvider


class GGUFProvider(BaseProvider):
    """GGUF provider using llama-cpp-python for local inference."""

    def __init__(self, config: ProviderConfig, system_specs: Optional['SystemSpecs'] = None):
        """Initialize GGUF provider.

        Args:
            config: Provider configuration with models_dir and device settings
            system_specs: System specifications for dynamic GPU/CPU detection
        """
        self.config = config
        self.models_dir = Path(str(config.models_dir)).expanduser()
        self.device_preference = config.device_preference or ["cuda", "mps", "cpu"]

        # Store or detect system specs
        self.system_specs = system_specs or SystemSpecs.detect()

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata cache for efficient model inspection
        self.metadata_cache = ModelMetadataCache()
        logger.debug("Initialized model metadata cache")

        # Determine if GPU is available from system specs
        self.use_gpu = self.system_specs.gpu.available

        # llama-server instances for GPU acceleration + KV cache reuse
        self.llama_servers: Dict[str, LlamaServerManager] = {}  # model_path -> server
        self.use_llama_server = True  # Always use server for GGUF models (GPU + cache)

        # VLM image rescaling preferences (per-model, session-scoped)
        # Maps model_id -> (width, height) or None to disable
        self._rescale_resolution: Dict[str, Optional[tuple[int, int]]] = {}

        logger.info(f"GGUF provider initialized (GPU: {self.use_gpu}, type: {self.system_specs.gpu.gpu_type}, llama-server mode: ON)")

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available.

        Returns:
            True if GPU should be used
        """
        for device in self.device_preference:
            if device in ["cuda", "mps"]:
                try:
                    import llama_cpp
                    # Check if llama-cpp-python was compiled with GPU support
                    return True
                except Exception:
                    continue
        return False

    def discover_models(self) -> list[ModelInfo]:
        """Discover GGUF models in models directory using metadata cache.

        Returns:
            List of ModelInfo objects for found GGUF files
        """
        models = []

        if not self.models_dir.exists():
            logger.warning(f"GGUF models directory not found: {self.models_dir}")
            return models

        try:
            # Search for .gguf files
            for gguf_file in self.models_dir.rglob("*.gguf"):
                if not gguf_file.is_file():
                    continue

                # Skip macOS metadata files (._filename)
                if gguf_file.name.startswith('._'):
                    continue

                # Use file name as model ID (without extension)
                model_id = gguf_file.stem

                # Get metadata from cache (reads GGUF header only, no full model loading)
                metadata = self.metadata_cache.get_metadata(str(gguf_file), provider="gguf")

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
                        model_id=str(gguf_file),  # Use full path as ID
                        name=model_id,  # Use filename as display name
                        provider="gguf",
                        size_gb=metadata.file_size_gb,
                        architecture=metadata.architecture,
                        quantization=metadata.quantization,
                        params_billions=metadata.params_billions,
                        ram_gb=metadata.ram_gb,
                        vram_gb=metadata.vram_gb,
                        model_type=model_type,
                        capabilities=capabilities,
                        compatibility=CompatibilityStatus.PERFECT_FIT,
                        compatibility_message="Compatibility not yet assessed",
                        is_installed=True,
                    )
                else:
                    # Fallback: metadata inspection failed, use basic info
                    logger.warning(f"Could not get metadata for {model_id}, using fallback")
                    size_gb = gguf_file.stat().st_size / (1024**3)
                    model_type, capabilities = self._classify_gguf_model(model_id)
                    model_info = ModelInfo(
                        model_id=str(gguf_file),
                        name=model_id,
                        provider="gguf",
                        size_gb=size_gb,
                        model_type=model_type,
                        capabilities=capabilities,
                        compatibility=CompatibilityStatus.PERFECT_FIT,
                        compatibility_message="Compatibility not yet assessed",
                        is_installed=True,
                    )

                models.append(model_info)
                logger.debug(f"Discovered GGUF model: {model_id} ({metadata.model_type if metadata else 'unknown'}, {metadata.quantization if metadata else 'unknown'})")

            logger.info(f"Discovered {len(models)} GGUF models")
            return models

        except Exception as e:
            logger.error(f"Error discovering GGUF models: {e}")
            return models

    # DEPRECATED: This method is no longer used - metadata cache provides dynamic detection
    # Keeping for backward compatibility only (used in fallback scenarios)
    def _classify_gguf_model(self, model_name: str) -> tuple[ModelType, list[EndpointType]]:
        """[DEPRECATED] Classify GGUF model by name patterns.

        This method is deprecated in favor of metadata cache which reads actual GGUF header.
        Only used as a fallback when metadata inspection fails.

        Args:
            model_name: Model filename (without .gguf)

        Returns:
            (ModelType, list of EndpointType)
        """
        name_lower = model_name.lower()

        # VLM keywords (fallback only)
        vlm_keywords = [
            "llava", "bakllava", "obsidian", "vision", "clip",
            "moondream", "cogvlm", "minicpm-v"
        ]

        if any(keyword in name_lower for keyword in vlm_keywords):
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
        if "embed" in name_lower:
            return (ModelType.EMBEDDING, [])

        # Default to LLM
        return (ModelType.LLM, [EndpointType.TEXT])

    def _find_mmproj_file(self, model_path: str) -> Optional[str]:
        """Find associated mmproj file for VLM support.

        2025 FEATURE: llama-server now supports VLMs via --mmproj flag (April 2025 with libmtmd).

        The mmproj file contains the vision encoder (typically CLIP) that processes images
        and converts them to embeddings that the language model can understand.

        Searches for mmproj files in the same directory as the model:
        - Same name as model with .mmproj extension
        - "mmproj" keyword in filename
        - Common patterns: *-mmproj.gguf, *-mmproj-*.gguf, mmproj-*.gguf

        Example for qwen3-vl:2b-instruct-q4_K_M:
        - Model: qwen3-vl-2b-instruct-q4_k_m.gguf
        - Mmproj: qwen3-vl-2b-instruct-mmproj-f16.gguf

        Args:
            model_path: Path to GGUF model file

        Returns:
            Path to mmproj file if found, None otherwise
        """
        model_file = Path(model_path)
        model_dir = model_file.parent
        model_stem = model_file.stem

        # Search patterns (in order of preference)
        patterns = [
            f"{model_stem}.mmproj",  # exact match with .mmproj extension
            f"{model_stem}-mmproj.gguf",  # model-name-mmproj.gguf
            f"{model_stem}*mmproj*.gguf",  # model-name*mmproj*.gguf
            "mmproj*.gguf",  # any mmproj file in same directory
        ]

        for pattern in patterns:
            matches = list(model_dir.glob(pattern))
            if matches:
                mmproj_file = matches[0]
                logger.info(f"✓ Found mmproj file: {mmproj_file.name}")
                return str(mmproj_file)

        # Not found
        logger.debug(f"No mmproj file found for {model_file.name} (VLM support disabled)")
        return None

    def _extract_model_name_from_path(self, model_path: str) -> str:
        """Extract model name from file path for template detection.

        Args:
            model_path: Full path to GGUF model file (lowercased)

        Returns:
            Model name suitable for chat template detection

        Examples:
            /path/qwen2-0_5b-instruct-q4_k_m.gguf → qwen2-0_5b-instruct
            /path/deepseek-r1-distill-qwen-1.5b_q4_k_m.gguf → deepseek-r1-distill-qwen
        """
        from pathlib import Path

        # Get filename without extension
        filename = Path(model_path).stem

        # Remove quantization suffixes (q4_k_m, q4_k_s, etc.)
        quant_patterns = [
            "_q2_k", "_q3_k_s", "_q3_k_m", "_q3_k_l",
            "_q4_0", "_q4_1", "_q4_k_s", "_q4_k_m",
            "_q5_0", "_q5_1", "_q5_k_s", "_q5_k_m",
            "_q6_k", "_q8_0", "_f16", "_f32"
        ]

        model_name = filename.lower()
        for pattern in quant_patterns:
            model_name = model_name.replace(pattern, "")

        logger.debug(f"Extracted model name: {model_name} from {filename}")
        return model_name

    def load_model(self, model_id: str, device: str) -> Any:
        """Load GGUF model using llama-server for GPU acceleration.

        Uses llama-server instead of direct llama-cpp-python loading to enable:
        - Full GPU utilization (matches llama.cpp WebUI performance)
        - Persistent KV cache across conversation turns
        - 40+ tokens/s performance on follow-up messages
        - VLM support via --mmproj flag (2025 feature with libmtmd)

        For VLMs, automatically detects and loads mmproj files for vision support.

        Args:
            model_id: Path to GGUF file
            device: Target device ("cuda", "mps", or "cpu")

        Returns:
            LlamaServerManager instance (acts as model handle)
        """
        logger.info(f"Loading GGUF model with llama-server: {model_id}")

        try:
            # Check if server already running for this model
            model_key = str(model_id)
            if model_key in self.llama_servers:
                server = self.llama_servers[model_key]
                if server.is_running():
                    logger.info(f"Reusing existing llama-server for {Path(model_id).name}")
                    return server

            # Determine GPU layers based on device and model type
            model_name_lower = str(model_id).lower()
            is_jamba = any(keyword in model_name_lower for keyword in ['jamba', 'mamba'])

            # Jamba/Mamba models: force CPU on MPS, use GPU on CUDA
            if is_jamba and self.use_gpu:
                import torch
                if torch.backends.mps.is_available():
                    logger.warning(
                        "Jamba/Mamba models have MPS compatibility issues. "
                        "Using CPU mode for stability."
                    )
                    n_gpu_layers = 0
                else:
                    n_gpu_layers = -1  # CUDA is stable
            else:
                n_gpu_layers = -1 if self.use_gpu else 0

            # Determine context window (n_ctx)
            # In benchmark mode, suppress interactive prompt and use defaults or override from env/params
            import os
            min_ctx, max_ctx = 256, 32768
            default_ctx = 8192 if is_jamba else 4096
            n_ctx = default_ctx
            try:
                # Allow explicit override via env var
                env_ctx = os.getenv("EKAM_N_CTX")
                if env_ctx:
                    n_ctx = max(min_ctx, min(max_ctx, int(env_ctx)))
                # If not in benchmark mode, preserve interactive prompt
                if not os.getenv("EKAM_BENCHMARK_MODE"):
                    from ..cli.text_input import professional_prompt
                    from ..cli.tui_manager import tui
                    tui.clear_screen()
                    tui.console.print("\n[cyan]Context Window (n_ctx)[/cyan] — tokens kept in memory per request")
                    tui.console.print(
                        f"[dim]Min:[/dim] {min_ctx}   [dim]Max:[/dim] {max_ctx}   "
                        f"[dim]Default:[/dim] {default_ctx}   [dim]Recommended:[/dim] {default_ctx}"
                    )
                    tui.console.print("[dim]Higher values increase RAM/VRAM usage and may reduce throughput.[/dim]")
                    n_ctx = professional_prompt.get_numeric(
                        "Context length (tokens)",
                        min_val=min_ctx,
                        max_val=max_ctx,
                        default=default_ctx,
                        style="cyan",
                    )
            except Exception:
                # Fallback to default if any parsing/IO error
                n_ctx = default_ctx

            # Check for mmproj file for VLM support (needed for rescaling prompt)
            mmproj_path = self._find_mmproj_file(str(model_id))
            is_vlm_model = mmproj_path is not None

            # VLM Image Rescaling Configuration (only for VLM models)
            # Ask user for target resolution to normalize input images
            rescale_width, rescale_height = None, None
            if is_vlm_model and model_id not in self._rescale_resolution:
                # Default rescaling resolution
                default_resolution = "1280x1024"
                try:
                    # In benchmark mode, resolution is set via benchmark config (not here)
                    # In regular inference mode, prompt user
                    if not os.getenv("EKAM_BENCHMARK_MODE"):
                        from ..cli.text_input import professional_prompt
                        from ..cli.tui_manager import tui
                        tui.clear_screen()
                        tui.console.print("\n[cyan]VLM Image Rescaling[/cyan] — normalize input image resolution")
                        tui.console.print(
                            f"[dim]Images will be rescaled to consistent resolution (maintains aspect ratio with padding)[/dim]"
                        )
                        tui.console.print(
                            f"[dim]Default:[/dim] {default_resolution}   [dim]Range:[/dim] 256x256 to 4096x4096"
                        )
                        tui.console.print(
                            "[dim]Enter resolution as WIDTHxHEIGHT (e.g., 1280x1024) or press Enter for default[/dim]"
                        )

                        resolution_input = professional_prompt.get_input(
                            f"Image resolution (default: {default_resolution})",
                            style="cyan",
                            allow_multiline=False,
                            show_instructions=False
                        )

                        # Use default if user pressed Enter without input
                        if not resolution_input.strip():
                            resolution_input = default_resolution

                        # Parse resolution (WIDTHxHEIGHT)
                        import re
                        match = re.match(r'(\d+)x(\d+)', resolution_input.strip())
                        if match:
                            rescale_width, rescale_height = int(match.group(1)), int(match.group(2))
                            # Validate range (256-4096 per dimension)
                            if not (256 <= rescale_width <= 4096 and 256 <= rescale_height <= 4096):
                                logger.warning(
                                    f"Resolution {rescale_width}x{rescale_height} out of range (256-4096). "
                                    f"Using default {default_resolution}"
                                )
                                rescale_width, rescale_height = 1280, 1024
                        else:
                            logger.warning(f"Invalid resolution format '{resolution_input}'. Using default {default_resolution}")
                            rescale_width, rescale_height = 1280, 1024
                    else:
                        # Benchmark mode - use default (will be overridden by benchmark config)
                        rescale_width, rescale_height = 1280, 1024

                    # Store rescaling preference for this model
                    if rescale_width and rescale_height:
                        self._rescale_resolution[model_id] = (rescale_width, rescale_height)
                        logger.info(f"VLM image rescaling enabled: {rescale_width}x{rescale_height}")
                except Exception as e:
                    logger.warning(f"Error configuring image rescaling: {e}. Disabling rescaling.")
                    self._rescale_resolution[model_id] = None

            # Thread configuration
            physical_cores = self.system_specs.cpu_cores_physical
            optimal_threads = min(physical_cores, 8)
            optimal_threads_batch = physical_cores

            # Find available port
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]

            # Show compact loading box only if not in benchmark mode
            if not os.getenv("EKAM_BENCHMARK_MODE"):
                from ..cli.tui_manager import tui
                tui.clear_screen()
                tui.show_message(
                    f"Loading model: {Path(model_id).name}\n\n[dim]Please wait...[/dim]",
                    title="Loading",
                    style="cyan",
                )

            # Determine which server implementation to use
            server_type = get_llama_server_type()

            if server_type == "llama-cli-server":
                # Use custom CLI server with isolated mtmd_context (supports both LLM and VLM)
                if mmproj_path:
                    logger.info("Using custom CLI server with isolated mtmd_context for VLM")
                else:
                    logger.info("Using custom CLI server with isolated mtmd_context for LLM")
                server = LlamaCLIServerManager(
                    model_path=str(model_id),
                    host="127.0.0.1",
                    port=port,
                )
                ServerManagerClass = LlamaCLIServerManager
            else:
                # Use legacy llama-server (has vision model bugs)
                logger.info("Using legacy llama-server (EKAM_USE_LLAMA_SERVER=true)")
                server = LlamaServerManager(
                    model_path=str(model_id),
                    host="127.0.0.1",
                    port=port,
                )
                ServerManagerClass = LlamaServerManager

            success = server.start(
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_batch=2048,
                n_ubatch=512,
                n_threads=optimal_threads,
                n_threads_batch=optimal_threads_batch,
                mmproj_path=mmproj_path,  # Pass mmproj for VLM support
            )

            if not success:
                raise RuntimeError(f"Failed to start {server_type} for {model_id}")

            # Store server instance
            self.llama_servers[model_key] = server

            vlm_status = "VLM enabled" if mmproj_path else "LLM mode"
            logger.info(
                f"✓ llama-server started on port {port} ({vlm_status}) "
                f"(GPU layers={n_gpu_layers}, threads={optimal_threads}, "
                f"ctx={n_ctx}, batch=2048, KV cache=GPU VRAM)"
            )

            # Warn about experimental architecture support
            if is_jamba:
                logger.warning(
                    "Jamba uses hybrid Mamba+Transformer architecture. "
                    "If you encounter generation errors, try shorter prompts or /clear."
                )

            return server

        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            raise RuntimeError(f"Failed to load GGUF model via server: {e}")

    def unload_model(self, handle: Any) -> None:
        """Unload GGUF model and free memory.

        Args:
            handle: LlamaServerManager or LlamaCLIServerManager instance
        """
        if handle is None:
            return

        try:
            # Stop server if this is a server instance (works for both types)
            if isinstance(handle, (LlamaServerManager, LlamaCLIServerManager)):
                model_key = str(handle.model_path)
                if model_key in self.llama_servers:
                    handle.stop()
                    del self.llama_servers[model_key]
                    server_type = "custom CLI server" if isinstance(handle, LlamaCLIServerManager) else "llama-server"
                    logger.info(f"{server_type} stopped for {handle.model_path.name}")
                return

            # Legacy: direct Llama instance
            del handle
            logger.info("GGUF model unloaded")
        except Exception as e:
            logger.warning(f"Error unloading GGUF model: {e}")

    def run_qa(
        self,
        handle: Any,
        image: Image.Image,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        custom_parameters: Optional[dict] = None,
        stream_callback: Optional[callable] = None,
    ) -> str:
        """Run QA with GGUF VLM using llama-server HTTP API.

        2025 UPDATE: Now uses llama-server HTTP API with native file:// format (prevents hallucinations).
        Priority: file:// (native llama.cpp) → OpenAI base64 → [img-1] proprietary (fallback)

        Args:
            handle: LlamaServerManager instance
            image: PIL Image
            question: Question text
            conversation_history: Optional list of (user_msg, bot_response) tuples
            custom_parameters: Optional custom inference parameters
            stream_callback: Optional callback for token streaming (delta: str, is_first: bool)

        Returns:
            Answer text
        """
        from ..services.llama_server_manager import LlamaServerManager
        from ..benchmarking.utils.llama_delay import apply_llama_server_delay
        import base64
        from io import BytesIO
        import tempfile
        import os

        logger.info("Running GGUF VLM inference via llama-server...")

        # Verify model is LlamaServerManager instance
        if not isinstance(handle, LlamaServerManager):
            raise TypeError(f"Expected LlamaServerManager, got {type(handle).__name__}")

        # Apply VLM image rescaling if configured for this model
        model_path = str(handle.model_path) if hasattr(handle, 'model_path') else None
        if model_path and model_path in self._rescale_resolution:
            rescale_config = self._rescale_resolution[model_path]
            if rescale_config:  # Not None (None = disabled)
                target_width, target_height = rescale_config
                original_size = image.size
                # Only rescale if image is different size
                if image.size != (target_width, target_height):
                    from ..utils.image import rescale_image_with_padding
                    image = rescale_image_with_padding(image, target_width, target_height)
                    logger.info(
                        f"Rescaled image from {original_size[0]}x{original_size[1]} to "
                        f"{target_width}x{target_height} (with aspect ratio preservation)"
                    )

        # Prepare image in multiple formats for llama-server API
        # 1. Save to temporary file for file:// URL (native llama.cpp format - RECOMMENDED)
        temp_image_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name, format="JPEG")
                temp_image_path = tmp.name
        except Exception as e:
            logger.warning(f"Could not save temp image file: {e}")

        # 2. Convert to base64 data URL for OpenAI-style format
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"

        # Debug logging
        logger.debug(f"Image encoding: {len(img_bytes)} bytes → {len(img_base64)} base64 chars")
        logger.debug(f"Image size: {image.size}, mode: {image.mode}")
        if temp_image_path:
            logger.debug(f"Temp image saved to: {temp_image_path}")

        # Build messages array with vision content
        messages = []

        # System prompt (from custom_parameters)
        sys_prompt = None
        if custom_parameters and isinstance(custom_parameters.get("system_prompt"), str):
            sys_prompt = custom_parameters.get("system_prompt")
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # Add conversation history (text-only, no images in history)
        if conversation_history:
            for user_msg, bot_msg in conversation_history[-3:]:  # Last 3 turns
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": bot_msg})

        # Add current question with image
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": img_data_url}}
            ]
        })

        # Gather advanced parameters from custom_parameters
        extra = {}
        if custom_parameters:
            for key in [
                "top_k", "min_p", "typical_p", "tfs_z", "repeat_penalty",
                "presence_penalty", "frequency_penalty", "penalty_last_n",
                "mirostat", "mirostat_tau", "mirostat_eta", "seed", "n_keep",
                "ignore_eos", "grammar", "logit_bias", "n_probs",
            ]:
                if key in custom_parameters:
                    extra[key] = custom_parameters[key]

        # ============================================================================
        # 2025 llama.cpp STANDARD FORMAT: OpenAI-compatible image_url content parts
        # ============================================================================
        # Reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/utils.hpp:616-660
        # The llama.cpp server supports OpenAI-compatible format with image_url in content array:
        # - data:image/*;base64,<base64> (standard, portable, proven to work)
        # - http:// or https:// URLs (downloads remote images automatically)
        # - file:// URLs (experimental, may not be supported in all versions)
        #
        # Priority order (as requested by user):
        # 1. PRIMARY: OpenAI base64 data URL format (most compatible with llama.cpp web UI)
        # 2. FALLBACK: Proprietary [img-1] format (for compatibility with older code)
        # ============================================================================

        # Format 1: OpenAI-compatible format with base64 data URL (PRIMARY)
        # This is what the llama.cpp web UI uses internally
        openai_messages = []
        if sys_prompt:
            openai_messages.append({"role": "system", "content": sys_prompt})
        if conversation_history:
            for user_msg, bot_msg in conversation_history[-3:]:
                openai_messages.append({"role": "user", "content": user_msg})
                openai_messages.append({"role": "assistant", "content": bot_msg})
        openai_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": img_data_url}}
            ]
        })

        # Format 2: HTTP URL format (if we support remote images in future)
        # Placeholder for potential HTTP URL support
        # llama.cpp server can download images from http:// or https:// URLs
        # Not implementing now as we're working with local PIL images

        # Format 3: Proprietary llama.cpp format with [img-1] placeholder (FALLBACK for compatibility)
        # This is a custom format not part of standard llama.cpp
        # Keeping as fallback for any edge cases where OpenAI format might not work
        proprietary_messages = []
        if sys_prompt:
            proprietary_messages.append({"role": "system", "content": sys_prompt})
        if conversation_history:
            for user_msg, bot_msg in conversation_history[-3:]:
                proprietary_messages.append({"role": "user", "content": user_msg})
                proprietary_messages.append({"role": "assistant", "content": bot_msg})
        proprietary_messages.append({
            "role": "user",
            "content": f"[img-1]\n{question}"
        })
        proprietary_image_data = [{"id": 1, "data": img_base64}]

        # Try formats in priority order (as requested):
        # 1. OpenAI base64 (PRIMARY - standard llama.cpp web UI format)
        # 2. Proprietary [img-1] (FALLBACK - for compatibility)
        formats_to_try = []
        formats_to_try.append(("openai_base64", openai_messages, None))  # PRIMARY: Standard OpenAI format
        formats_to_try.append(("proprietary", proprietary_messages, proprietary_image_data))  # FALLBACK: Custom format

        last_error = None
        for format_tuple in formats_to_try:
            format_name = format_tuple[0]
            test_messages = format_tuple[1]
            test_image_data = format_tuple[2] if len(format_tuple) > 2 else None

            try:
                logger.info(f"Trying vision format: {format_name}")

                # Build kwargs for chat_completion with streaming support
                chat_kwargs = {
                    "messages": test_messages,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "stream": bool(stream_callback),  # Enable streaming if callback provided
                }
                chat_kwargs.update(extra)
                if test_image_data:
                    chat_kwargs["image_data"] = test_image_data

                response = handle.chat_completion(**chat_kwargs)

                # Handle streaming vs non-streaming responses
                if stream_callback and hasattr(response, '__iter__'):
                    # Streaming response
                    answer = ""
                    is_first_token = True
                    for chunk in response:
                        if isinstance(chunk, dict) and "choices" in chunk:
                            delta = chunk["choices"][0].get("delta", {}).get("content", "")
                            if delta:
                                answer += delta
                                stream_callback(delta, is_first=is_first_token)
                                is_first_token = False
                elif isinstance(response, dict) and "choices" in response:
                    # Non-streaming response
                    answer = response["choices"][0]["message"]["content"]
                else:
                    logger.warning(f"Format {format_name}: Unexpected response format")
                    last_error = RuntimeError(f"Unexpected response format from {format_name}")
                    continue

                # Validate content
                error_keywords = [
                    "not a valid image", "invalid image", "unable to process",
                    "cannot", "no image", "do not see an image"
                ]
                if any(k in answer.lower() for k in error_keywords):
                    logger.warning(f"Format {format_name} produced suspected error response; trying next format")
                    last_error = RuntimeError(f"{format_name} returned suspected error response")
                    continue

                logger.info(f"✓ GGUF VLM inference successful ({format_name} format)")
                # Clean up temp file before returning
                if temp_image_path:
                    try:
                        os.unlink(temp_image_path)
                    except Exception:
                        pass
                # Apply post-inference delay for llama-server state cleanup
                apply_llama_server_delay("gguf")
                return answer.strip()

            except Exception as e:
                logger.error(f"Format {format_name} failed with error: {type(e).__name__}: {e}")
                last_error = e
                continue

        # All formats failed - clean up temp file
        if temp_image_path:
            try:
                os.unlink(temp_image_path)
            except Exception:
                pass

        if last_error:
            logger.error(f"All vision formats failed. Last error: {last_error}")
            raise last_error
        else:
            raise RuntimeError("All vision formats failed with no specific error")

    def run_caption(
        self,
        handle: Any,
        image: Image.Image,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        detail_level: str = "detailed",
        stream_callback: Optional[callable] = None,
    ) -> str:
        """Generate caption with GGUF VLM.

        Args:
            handle: Llama model instance
            image: PIL Image
            conversation_history: Optional list of (user_msg, bot_response) tuples
            detail_level: "detailed" or "short"
            stream_callback: Optional callback for token streaming (delta: str, is_first: bool)

        Returns:
            Caption text
        """
        prompt = (
            "Describe this image in detail:"
            if detail_level == "detailed"
            else "Describe this image briefly:"
        )
        return self.run_qa(handle, image, prompt, conversation_history, stream_callback=stream_callback)

    def run_detect(self, handle: Any, image: Image.Image, object_name: str) -> list[dict]:
        """Detect objects with GGUF VLM and parse bounding boxes.

        Args:
            handle: Llama model instance
            image: PIL Image
            object_name: Object to detect

        Returns:
            List of detection dicts with parsed bounding boxes
        """
        from ..utils.vlm_response_parser import parse_detection_response

        prompt = (
            f"Detect all instances of '{object_name}' in this image.\n\n"
            f"Return ONLY valid JSON in this exact structure:\n"
            f"{{\n"
            f"  \"{object_name}_1\": [x1, y1, x2, y2],\n"
            f"  \"{object_name}_2\": [x1, y1, x2, y2]\n"
            f"}}\n\n"
            f"Rules:\n"
            f"- Coordinates must be normalized (0.0 to 1.0)\n"
            f"- 0.0 is left/top edge, 1.0 is right/bottom edge\n"
            f"- Do not include any text before or after the JSON\n"
            f"- Number each instance sequentially (_1, _2, _3, etc.)"
        )
        response = self.run_qa(handle, image, prompt)

        # Parse response to extract actual bounding boxes
        detections = parse_detection_response(response, object_name)

        # Add raw response to each detection for debugging
        for detection in detections:
            detection["raw"] = response

        return detections

    def run_point(self, handle: Any, image: Image.Image, object_name: str) -> dict:
        """Point to object with GGUF VLM and parse coordinates.

        Args:
            handle: Llama model instance
            image: PIL Image
            object_name: Object to locate

        Returns:
            Coordinates dict with parsed x, y values
        """
        from ..utils.vlm_response_parser import parse_point_response

        prompt = (
            f"Locate the '{object_name}' in this image.\n\n"
            f"Return ONLY valid JSON in this exact structure:\n"
            f"{{\n"
            f"  \"{object_name}\": [x, y]\n"
            f"}}\n\n"
            f"Rules:\n"
            f"- Coordinates must be normalized (0.0 to 1.0)\n"
            f"- 0.0 is left/top edge, 1.0 is right/bottom edge\n"
            f"- Do not include any text before or after the JSON"
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
        custom_parameters: Optional[dict] = None,
        stream_callback: Optional[callable] = None,
    ) -> str:
        """Generate text with GGUF LLM with production-level chat templates.

        Args:
            handle: Llama model instance or LlamaServerManager
            prompt: Text prompt
            conversation_history: Optional list of (user_msg, bot_response) tuples
            custom_parameters: Optional custom generation parameters

        Returns:
            Generated text
        """
        try:
            # Check if using llama-server (HTTP API for GPU acceleration + KV cache reuse)
            if isinstance(handle, LlamaServerManager):
                logger.debug("Using llama-server HTTP API for text generation")

                # Extract model name from server's model path
                model_id_str = str(handle.model_path).lower()
                model_name = self._extract_model_name_from_path(model_id_str) if model_id_str else None

                # Get template-specific stop tokens
                from ..utils.history_formatter import get_stop_tokens
                stop_tokens = get_stop_tokens(model_name=model_name)

                # Build messages format for chat completion API
                sys_prompt = None
                if custom_parameters and isinstance(custom_parameters.get("system_prompt"), str):
                    sys_prompt = custom_parameters.get("system_prompt")
                messages = [
                    {"role": "system", "content": sys_prompt or "You are a helpful AI assistant. Provide accurate, concise, and well-formatted responses."}
                ]

                # Add conversation history (last 5 turns for context)
                if conversation_history:
                    for user_msg, ai_response in conversation_history[-5:]:
                        messages.append({"role": "user", "content": user_msg})
                        messages.append({"role": "assistant", "content": ai_response})

                # Add current prompt
                messages.append({"role": "user", "content": prompt})

                # Get generation parameters
                max_tokens = 1024
                temperature = 0.7
                top_p = 0.9

                # Override with custom parameters if provided
                if custom_parameters:
                    max_tokens = custom_parameters.get("max_tokens", max_tokens)
                    temperature = custom_parameters.get("temperature", temperature)
                    top_p = custom_parameters.get("top_p", top_p)

                # Call llama-server's chat completion API
                # Map advanced parameters from session custom_parameters if provided
                extra = {}
                if custom_parameters:
                    for key in [
                        "top_k","min_p","typical_p","tfs_z","repeat_penalty",
                        "presence_penalty","frequency_penalty","penalty_last_n",
                        "mirostat","mirostat_tau","mirostat_eta","seed","n_keep",
                        "ignore_eos","grammar","logit_bias","n_probs",
                    ]:
                        if key in custom_parameters:
                            extra[key] = custom_parameters[key]

                # Build and log exact request before sending
                try:
                    url, payload = handle.build_chat_payload(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop_tokens,
                        stream=True,
                        **extra,
                    )
                    # Log request
                    logger.debug(f"Request: POST {url} | stream=True")
                except Exception:
                    pass

                response_stream = handle.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_tokens,
                    stream=True,
                    **extra,
                )

                # Accumulate streaming response
                full_text = ""
                is_first_token = True
                for chunk in response_stream:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {}).get('content', '')
                        if delta:
                            full_text += delta
                            if stream_callback:
                                try:
                                    stream_callback(delta, is_first=is_first_token)
                                finally:
                                    is_first_token = False

                # Clean response to remove artifacts and meta-commentary
                from ..utils.response_cleaner import clean_model_response
                from ..benchmarking.utils.llama_delay import apply_llama_server_delay
                cleaned_response = clean_model_response(full_text, aggressive=True)

                logger.debug(f"llama-server response generated ({len(cleaned_response)} chars)")
                # Apply post-inference delay for llama-server state cleanup
                apply_llama_server_delay("gguf")
                return cleaned_response

            # Legacy path: Direct llama-cpp-python inference (fallback)
            # Detect model name from handle for proper template selection
            model_id_str = str(getattr(handle, "model_path", "")).lower()
            model_name = self._extract_model_name_from_path(model_id_str) if model_id_str else None

            # Use production-level history formatter with model name for template detection
            full_prompt = format_conversation_history(
                conversation_history=conversation_history,
                current_prompt=prompt,
                max_turns=5,
                system_prompt=None,
                model_name=model_name  # Pass model name for correct template detection
            )

            # Get model's context window size
            model_ctx_size = getattr(handle, 'n_ctx', lambda: 2048)()

            # PERFORMANCE FIX: Use character-based estimation instead of CPU tokenization
            # Character-based estimation is instant (<1ms) vs CPU tokenization (5-7 seconds)
            # Conservative estimate: 1 token ≈ 4 characters for English text
            # This prevents the 10x slowdown while maintaining safety
            estimated_prompt_tokens = len(full_prompt) // 4

            # Calculate maximum safe tokens for generation
            # Use slightly larger safety margin (15% instead of 10%) for estimation uncertainty
            safety_margin = int(model_ctx_size * 0.15)
            max_safe_tokens = max(1, model_ctx_size - estimated_prompt_tokens - safety_margin)

            logger.debug(
                f"Context window: {model_ctx_size} tokens | "
                f"Estimated prompt: ~{estimated_prompt_tokens} tokens ({len(full_prompt)} chars) | "
                f"Max safe response: {max_safe_tokens} tokens"
            )

            # Get appropriate stop tokens for the detected chat template
            from ..utils.history_formatter import get_stop_tokens
            template_stop_tokens = get_stop_tokens(model_name=model_name)

            # Build generation parameters with defaults
            gen_params = {
                "prompt": full_prompt,
                "max_tokens": min(1024, max_safe_tokens),  # Cap at safe limit
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": template_stop_tokens,  # Use template-specific stop tokens
                "stream": True,  # PERFORMANCE: Enable streaming for progressive output
            }

            logger.debug(f"Using stop tokens: {template_stop_tokens}")

            # KV cache is now enabled at model load time via set_cache()
            # No need to check for cache_prompt parameter (doesn't exist in 0.3.x)

            # Model-specific enhancements for DeepSeek-R1 reasoning models
            if model_name and ("deepseek" in model_name.lower() or "r1" in model_name.lower()):
                # DeepSeek-R1 uses Chain-of-Thought reasoning with <think> tags
                # Add additional stop tokens to prevent reasoning leakage
                gen_params["stop"].extend(["<think>", "</think>"])
                logger.debug("Added DeepSeek-R1 reasoning stop tokens to prevent CoT leakage")

            # Override with custom parameters from session
            if custom_parameters:
                if "max_tokens" in custom_parameters:
                    requested_max = custom_parameters["max_tokens"]
                    # Cap at safe limit to prevent context overflow
                    if requested_max > max_safe_tokens:
                        logger.warning(
                            f"Requested max_tokens ({requested_max}) exceeds available context "
                            f"({max_safe_tokens} tokens available). Capping to {max_safe_tokens}."
                        )
                        gen_params["max_tokens"] = max_safe_tokens
                    else:
                        gen_params["max_tokens"] = requested_max
                if "temperature" in custom_parameters:
                    gen_params["temperature"] = custom_parameters["temperature"]
                if "top_p" in custom_parameters:
                    gen_params["top_p"] = custom_parameters["top_p"]
                if "top_k" in custom_parameters:
                    gen_params["top_k"] = custom_parameters["top_k"]
                if "repeat_penalty" in custom_parameters:
                    gen_params["repeat_penalty"] = custom_parameters["repeat_penalty"]
                if "seed" in custom_parameters:
                    gen_params["seed"] = custom_parameters["seed"]

            try:
                # Handle streaming response
                response_stream = handle.create_completion(**gen_params)

                # Accumulate streaming chunks into full response
                full_text = ""
                for chunk in response_stream:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('text', '')
                        full_text += delta

                # Reconstruct response in expected format
                response = {
                    "choices": [{"text": full_text}]
                }

            except Exception as gen_error:
                # llama_decode errors (-1, -2) or context window errors
                error_str = str(gen_error)

                # Check for context window overflow errors
                if "exceed" in error_str.lower() and "context" in error_str.lower():
                    raise RuntimeError(
                        f"Context window overflow: {error_str}\n\n"
                        f"Context info: {model_ctx_size} tokens total, "
                        f"{prompt_token_count} used by prompt, "
                        f"{gen_params['max_tokens']} requested for response.\n\n"
                        f"Try: (1) Type 'back' to exit and start a new session, "
                        f"(2) Use /config to reduce max_tokens, or "
                        f"(3) Reload the model (model context cannot be changed at runtime)"
                    )

                if "llama_decode" in error_str or "returned -1" in error_str:
                    # Context or generation failure - try with reduced parameters
                    logger.warning(f"Generation failed, retrying with reduced parameters: {gen_error}")

                    # Retry with smaller max_tokens and shorter history
                    retry_prompt = format_conversation_history(
                        conversation_history=conversation_history[-2:] if conversation_history else None,  # Only last 2 turns
                        current_prompt=prompt,
                        max_turns=2,
                        system_prompt=None,
                        model_name=None
                    )

                    # Recalculate safe tokens for retry prompt using character estimation
                    retry_estimated_tokens = len(retry_prompt) // 4
                    retry_max_safe = max(1, model_ctx_size - retry_estimated_tokens - safety_margin)

                    retry_params = {
                        "prompt": retry_prompt,
                        "max_tokens": min(512, retry_max_safe),  # Reduced from 1024
                        "temperature": gen_params["temperature"],
                        "top_p": gen_params["top_p"],
                        "top_k": gen_params["top_k"],
                        "repeat_penalty": gen_params["repeat_penalty"],
                        "stop": gen_params["stop"],
                        "stream": True,  # Enable streaming for retry
                    }

                    # Propagate cache_prompt if it was added
                    if "cache_prompt" in gen_params:
                        retry_params["cache_prompt"] = gen_params["cache_prompt"]

                    try:
                        # Handle streaming response for retry
                        response_stream = handle.create_completion(**retry_params)
                        full_text = ""
                        for chunk in response_stream:
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('text', '')
                                full_text += delta
                        response = {"choices": [{"text": full_text}]}
                        logger.info("Retry successful with reduced parameters")
                    except Exception as retry_error:
                        logger.error(f"Retry also failed: {retry_error}")
                        raise RuntimeError(
                            f"Text generation failed. This model may have compatibility issues with long responses. "
                            f"Try: (1) shorter prompts, (2) start a new session, or (3) a different model. "
                            f"Error: {error_str}"
                        )
                else:
                    # Other error - re-raise
                    raise

            # Clean response to remove artifacts and meta-commentary
            from ..utils.response_cleaner import clean_model_response
            response_text = response["choices"][0]["text"]
            cleaned_response = clean_model_response(response_text, aggressive=True)

            return cleaned_response

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")

    def install_model(self, model_name: str, progress_callback=None) -> bool:
        """Download GGUF model (not implemented - manual download).

        Args:
            model_name: Model identifier
            progress_callback: Optional progress callback

        Returns:
            False (not supported)
        """
        logger.warning("GGUF models must be downloaded manually")
        return False

    def delete_model(self, model_id: str) -> bool:
        """Delete GGUF file from disk (dynamically resolves path).

        Args:
            model_id: Path to GGUF file (can be relative or absolute)

        Returns:
            True if successful
        """
        logger.info(f"Deleting GGUF model: {model_id}")

        try:
            from pathlib import Path

            model_path = Path(model_id)

            # Convert to absolute path if relative
            if not model_path.is_absolute():
                model_path = model_path.absolute()

            if model_path.exists() and model_path.is_file():
                model_path.unlink()
                logger.info(f"Deleted GGUF file: {model_path}")
                return True
            else:
                logger.warning(f"GGUF file not found: {model_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete GGUF model: {e}")
            return False

    def get_model_info(self, model_id: str) -> dict:
        """Get detailed model information for GGUF models.

        Args:
            model_id: Model identifier (file path)

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
                "provider": "gguf",
                "default_parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": 1024,  # Increased for more detailed responses
                    "repeat_penalty": 1.1,
                }
            }

        # Return full model info
        return {
            "model_id": model.model_id,
            "name": model.name,
            "provider": "gguf",
            "architecture": model.architecture or "Unknown",
            "quantization": model.quantization or "Unknown",
            "size_gb": model.size_gb,
            "model_type": str(model.model_type).upper() if hasattr(model.model_type, 'value') else str(model.model_type).upper(),
            "capabilities": [str(cap) for cap in model.capabilities] if model.capabilities else ["text"],
            "default_parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 1024,  # Increased for more detailed responses
                "repeat_penalty": 1.1,
            }
        }

    def estimate_model_size(self, model_name: str) -> float:
        """Estimate GGUF model size.

        Args:
            model_name: Model identifier

        Returns:
            Estimated size in GB
        """
        # Rough estimates for GGUF quantized models
        name_lower = model_name.lower()

        # Check quantization level
        if "q4_0" in name_lower or "q4_k_m" in name_lower:
            if "7b" in name_lower:
                return 4.0
            elif "13b" in name_lower:
                return 7.5
            elif "3b" in name_lower:
                return 2.0
        elif "q5" in name_lower:
            if "7b" in name_lower:
                return 5.0
            elif "13b" in name_lower:
                return 9.0
        elif "q8" in name_lower:
            if "7b" in name_lower:
                return 7.0
            elif "13b" in name_lower:
                return 13.0

        # Default estimate
        return 4.0

    # Convenience methods (not implemented for GGUF)
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
