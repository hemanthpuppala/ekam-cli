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

    def load_model(self, model_id: str, device: str) -> Any:
        """Load GGUF model with llama-cpp-python.

        Args:
            model_id: Path to GGUF file
            device: Device preference (used self.use_gpu instead)

        Returns:
            Llama model instance
        """
        logger.info(f"Loading GGUF model: {model_id}")

        try:
            from llama_cpp import Llama

            # Check if this is a Jamba/Mamba model - needs special handling
            model_name_lower = str(model_id).lower()
            is_jamba = any(keyword in model_name_lower for keyword in ['jamba', 'mamba'])

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
            # Larger context for newer/hybrid architectures that may need more space
            if is_jamba:
                # Jamba and other hybrid models may need larger context
                n_ctx = 4096
                logger.info(f"Using extended context window (4096) for hybrid architecture model")
            else:
                # Standard context for most models
                n_ctx = 2048

            # Load model
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

        Args:
            handle: Llama model instance
            image: PIL Image
            question: Question text
            conversation_history: Optional list of (user_msg, bot_response) tuples

        Returns:
            Answer text
        """
        # Check if model has vision support
        try:
            from llama_cpp import llama_cpp
            # For VLM models, llama.cpp supports image embeddings
            # This is a simplified implementation

            # Use unified Q&A history formatter (shared across all providers)
            prompt = format_qa_history(
                conversation_history=conversation_history,
                current_question=question,
                max_turns=5
            )

            response = handle.create_completion(
                prompt=prompt,
                max_tokens=1024,  # Increased for more detailed responses
                temperature=0.7,
                stop=["Q:", "\n\n"]
            )

            return response["choices"][0]["text"].strip()

        except Exception as e:
            logger.error(f"QA failed: {e}")
            raise NotImplementedError(f"VLM QA not fully supported for GGUF: {e}")

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

            # Build generation parameters with defaults
            gen_params = {
                "prompt": full_prompt,
                "max_tokens": 1024,  # Increased for more detailed responses
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
                    gen_params["max_tokens"] = custom_parameters["max_tokens"]
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
                # llama_decode errors (-1, -2) indicate generation failures
                error_str = str(gen_error)
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

                    retry_params = {
                        "prompt": retry_prompt,
                        "max_tokens": 512,  # Reduced from 1024
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
                            f"Try: (1) shorter prompts, (2) /clear to reset history, or (3) a different model. "
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
        """Delete GGUF file from disk.

        Args:
            model_id: Path to GGUF file

        Returns:
            True if successful
        """
        logger.info(f"Deleting GGUF model: {model_id}")

        try:
            model_path = Path(model_id)
            if model_path.exists() and model_path.is_file():
                model_path.unlink()
                logger.info(f"Deleted GGUF file: {model_path.name}")
                return True
            else:
                logger.warning(f"GGUF file not found: {model_id}")
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
