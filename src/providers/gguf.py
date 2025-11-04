"""GGUF provider implementation using llama-cpp-python."""

import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from PIL import Image

from ..models.endpoints import CompatibilityStatus, EndpointType, ModelType
from ..models.model import ModelInfo
from ..models.model_cache import ModelMetadataCache
from ..models.provider import ProviderConfig
from ..utils.history_formatter import format_conversation_history, format_qa_history
from .base import BaseProvider


class GGUFProvider(BaseProvider):
    """GGUF provider using llama-cpp-python for local inference."""

    def __init__(self, config: ProviderConfig):
        """Initialize GGUF provider.

        Args:
            config: Provider configuration with models_dir and device settings
        """
        self.config = config
        self.models_dir = Path(str(config.models_dir)).expanduser()
        self.device_preference = config.device_preference or ["cuda", "mps", "cpu"]

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata cache for efficient model inspection
        self.metadata_cache = ModelMetadataCache()
        logger.debug("Initialized model metadata cache")

        # Determine if GPU is available
        self.use_gpu = self._check_gpu_availability()
        logger.info(f"GGUF provider initialized (GPU: {self.use_gpu})")

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

        2025 FEATURE: llama.cpp now supports VLMs via mmproj files (April 2025).

        Searches for mmproj files in the same directory as the model:
        - Same name as model with .mmproj extension
        - "mmproj" keyword in filename
        - Common patterns: *-mmproj.gguf, *-mmproj-*.gguf, mmproj-*.gguf

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

    def load_model(self, model_id: str, device: str) -> Any:
        """Load GGUF model with llama-cpp-python.

        2025 UPDATE: Now supports VLMs with mmproj files (vision support added April 2025).

        Args:
            model_id: Path to GGUF file
            device: Device preference (used self.use_gpu instead)

        Returns:
            Llama model instance (with VLM chat handler if mmproj found)
        """
        logger.info(f"Loading GGUF model: {model_id}")

        try:
            from llama_cpp import Llama

            # Check if this is a Jamba/Mamba model - needs special handling
            model_name_lower = str(model_id).lower()
            is_jamba = any(keyword in model_name_lower for keyword in ['jamba', 'mamba'])

            # 2025 FEATURE: Check for VLM support (mmproj file)
            mmproj_path = self._find_mmproj_file(model_id)
            is_vlm = mmproj_path is not None

            # Jamba/Mamba models have compatibility issues with MPS (Apple Metal)
            # Force CPU mode for stability on macOS
            if is_jamba and self.use_gpu:
                # Check if we're on MPS (Apple Silicon)
                import torch
                if torch.backends.mps.is_available():
                    logger.warning(
                        "Jamba/Mamba models have compatibility issues with MPS (Apple Metal). "
                        "Forcing CPU mode for stability. This will be slower but more reliable."
                    )
                    n_gpu_layers = 0  # Force CPU
                else:
                    # CUDA is more stable for Jamba
                    n_gpu_layers = -1 if self.use_gpu else 0
            else:
                # Standard GPU configuration for other models
                n_gpu_layers = -1 if self.use_gpu else 0

            # Determine optimal context size based on model
            # Use larger context windows by default to support conversation history
            # Most modern GGUF models can handle 4k-8k context efficiently
            if is_jamba:
                # Jamba and other hybrid models may need larger context
                n_ctx = 8192
                logger.info(f"Using extended context window (8192) for hybrid architecture model")
            else:
                # Standard context for most models - increased from 2048 to 4096
                # This allows for ~3000 token prompts + 1024 token responses
                n_ctx = 4096
                logger.info(f"Using standard context window (4096)")

            # 2025 FEATURE: Load VLM with vision support
            if is_vlm and mmproj_path:
                try:
                    from llama_cpp.llama_chat_format import Llava15ChatHandler, Llava16ChatHandler

                    # Try Llava 1.6 handler first (newer, more capable)
                    # Falls back to 1.5 if not compatible
                    try:
                        chat_handler = Llava16ChatHandler(clip_model_path=mmproj_path, verbose=False)
                        logger.info(f"✓ Using LLaVA 1.6 chat handler with mmproj: {Path(mmproj_path).name}")
                    except Exception:
                        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)
                        logger.info(f"✓ Using LLaVA 1.5 chat handler with mmproj: {Path(mmproj_path).name}")

                    llama = Llama(
                        model_path=model_id,
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        n_threads=os.cpu_count() or 4,
                        chat_handler=chat_handler,  # Enable vision
                        verbose=False,
                    )
                    logger.info(f"✓ Loaded VLM with vision support (GPU layers: {n_gpu_layers}, context: {n_ctx})")

                except ImportError:
                    logger.warning("VLM chat handlers not available in llama-cpp-python. Update to latest version for vision support.")
                    # Fall back to standard loading
                    llama = Llama(
                        model_path=model_id,
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        n_threads=os.cpu_count() or 4,
                        verbose=False,
                    )
                    logger.info(f"Loaded GGUF model (GPU layers: {n_gpu_layers}, context: {n_ctx})")
            else:
                # Standard LLM loading
                llama = Llama(
                    model_path=model_id,
                    n_ctx=n_ctx,  # Context window (auto-adjusted)
                    n_gpu_layers=n_gpu_layers,
                    n_threads=os.cpu_count() or 4,
                    verbose=False,
                )
                logger.info(f"Loaded GGUF model (GPU layers: {n_gpu_layers}, context: {n_ctx})")

            # Warn about experimental architecture support
            if 'jamba' in model_name_lower:
                logger.warning(
                    "Jamba uses a hybrid Mamba+Transformer architecture. "
                    "If you encounter generation errors, try: (1) shorter prompts, "
                    "(2) /clear to reset history, or (3) CPU mode (set GPU layers to 0)"
                )
            return llama

        except Exception as e:
            logger.error(f"Failed to load GGUF model {model_id}: {e}")
            raise RuntimeError(f"Failed to load GGUF model: {e}")

    def unload_model(self, handle: Any) -> None:
        """Unload GGUF model and free memory.

        Args:
            handle: Llama model instance
        """
        if handle is None:
            return

        try:
            del handle
            logger.info("GGUF model unloaded")
        except Exception as e:
            logger.warning(f"Error unloading GGUF model: {e}")

    def run_qa(
        self,
        handle: Any,
        image: Image.Image,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None
    ) -> str:
        """Run QA with GGUF VLM using unified history formatter.

        2025 UPDATE: Now uses chat handler with image support for true VLM inference.

        Args:
            handle: Llama model instance (with VLM chat handler if available)
            image: PIL Image
            question: Question text
            conversation_history: Optional list of (user_msg, bot_response) tuples

        Returns:
            Answer text
        """
        try:
            # Check if model has chat_handler (VLM support)
            if hasattr(handle, 'chat_handler') and handle.chat_handler is not None:
                # 2025 VLM INFERENCE: Use chat completion with image
                logger.info("Using VLM chat handler for vision-based QA")

                # Save image to temporary file for llama.cpp
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image.save(tmp.name)
                    image_path = tmp.name

                try:
                    # Build messages with image
                    messages = []

                    # Add conversation history if present
                    if conversation_history:
                        for user_msg, bot_response in conversation_history[-5:]:  # Last 5 turns
                            messages.append({"role": "user", "content": user_msg})
                            messages.append({"role": "assistant", "content": bot_response})

                    # Add current question with image
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                            {"type": "text", "text": question}
                        ]
                    })

                    # Generate with VLM
                    response = handle.create_chat_completion(
                        messages=messages,
                        max_tokens=1024,
                        temperature=0.0,  # Deterministic for QA
                    )

                    answer = response["choices"][0]["message"]["content"].strip()
                    logger.info("✓ VLM QA completed")
                    return answer

                finally:
                    # Clean up temp file
                    import os
                    try:
                        os.unlink(image_path)
                    except Exception:
                        pass
            else:
                # Fallback: Text-only inference (no vision)
                logger.warning("Model does not have VLM support - running text-only QA")

                prompt = format_qa_history(
                    conversation_history=conversation_history,
                    current_question=question,
                    max_turns=5
                )

                response = handle.create_completion(
                    prompt=prompt,
                    max_tokens=1024,
                    temperature=0.7,
                    stop=["Q:", "\n\n"]
                )

                return response["choices"][0]["text"].strip()

        except Exception as e:
            logger.error(f"GGUF VLM QA failed: {e}")
            raise NotImplementedError(f"VLM QA not supported: {e}")

    def run_caption(
        self,
        handle: Any,
        image: Image.Image,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        detail_level: str = "detailed"
    ) -> str:
        """Generate caption with GGUF VLM.

        Args:
            handle: Llama model instance
            image: PIL Image
            conversation_history: Optional list of (user_msg, bot_response) tuples
            detail_level: "detailed" or "short"

        Returns:
            Caption text
        """
        prompt = (
            "Describe this image in detail:"
            if detail_level == "detailed"
            else "Describe this image briefly:"
        )
        return self.run_qa(handle, image, prompt, conversation_history)

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
        custom_parameters: Optional[dict] = None
    ) -> str:
        """Generate text with GGUF LLM with production-level chat templates.

        Args:
            handle: Llama model instance
            prompt: Text prompt
            conversation_history: Optional list of (user_msg, bot_response) tuples
            custom_parameters: Optional custom generation parameters

        Returns:
            Generated text
        """
        try:
            # Use production-level history formatter (defaults to ChatML for GGUF)
            # ChatML works well with most GGUF models
            full_prompt = format_conversation_history(
                conversation_history=conversation_history,
                current_prompt=prompt,
                max_turns=5,
                system_prompt=None,
                model_name=None  # Will default to ChatML template
            )

            # Get model's context window size
            model_ctx_size = getattr(handle, 'n_ctx', lambda: 2048)()

            # Tokenize the prompt to get accurate token count
            prompt_tokens = handle.tokenize(full_prompt.encode('utf-8'))
            prompt_token_count = len(prompt_tokens)

            # Calculate maximum safe tokens for generation
            # Reserve some tokens for safety margin (10%)
            safety_margin = int(model_ctx_size * 0.1)
            max_safe_tokens = max(1, model_ctx_size - prompt_token_count - safety_margin)

            logger.debug(
                f"Context window: {model_ctx_size} tokens | "
                f"Prompt: {prompt_token_count} tokens | "
                f"Max safe response: {max_safe_tokens} tokens"
            )

            # Build generation parameters with defaults
            gen_params = {
                "prompt": full_prompt,
                "max_tokens": min(1024, max_safe_tokens),  # Cap at safe limit
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": ["\n\n", "User:"]  # Default stop tokens
            }

            # Model-specific stop tokens for better generation control
            # DeepSeek-R1 reasoning models need special handling
            model_id_str = str(getattr(handle, "model_path", "")).lower()
            if "deepseek" in model_id_str or "r1" in model_id_str:
                # DeepSeek-R1 uses Chain-of-Thought reasoning with <think> tags
                # Add stop tokens to prevent over-generation and reasoning leakage
                gen_params["stop"] = [
                    "\n\n",           # Standard paragraph break
                    "User:",          # Chat template boundary
                    "<think>",        # Start of reasoning (shouldn't appear in output)
                    "</think>",       # End of reasoning (shouldn't appear in output)
                    "\n\nUser:",      # Combined boundary
                    "\n\n---",        # Section break
                    "Human:",         # Alternative template
                    "Assistant:",     # Alternative template boundary
                ]
                logger.debug("Using DeepSeek-R1 stop tokens to prevent over-generation")

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
                response = handle.create_completion(**gen_params)
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

                    # Recalculate safe tokens for retry prompt
                    retry_tokens = handle.tokenize(retry_prompt.encode('utf-8'))
                    retry_token_count = len(retry_tokens)
                    retry_max_safe = max(1, model_ctx_size - retry_token_count - safety_margin)

                    retry_params = {
                        "prompt": retry_prompt,
                        "max_tokens": min(512, retry_max_safe),  # Reduced from 1024
                        "temperature": gen_params["temperature"],
                        "top_p": gen_params["top_p"],
                        "top_k": gen_params["top_k"],
                        "repeat_penalty": gen_params["repeat_penalty"],
                        "stop": gen_params["stop"]
                    }

                    try:
                        response = handle.create_completion(**retry_params)
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
