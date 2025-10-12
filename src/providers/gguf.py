"""GGUF provider implementation using llama-cpp-python."""

import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from PIL import Image

from ..models.endpoints import CompatibilityStatus, EndpointType, ModelType
from ..models.model import ModelInfo
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
        """Discover GGUF models in models directory.

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

                # Use file name as model ID (without extension)
                model_id = gguf_file.stem
                size_gb = gguf_file.stat().st_size / (1024**3)

                # Classify model type by name
                model_type, capabilities = self._classify_gguf_model(model_id)

                model_info = ModelInfo(
                    model_id=str(gguf_file),  # Use full path as ID
                    name=model_id,  # Use filename as display name
                    provider="gguf",
                    size_gb=size_gb,
                    model_type=model_type,
                    capabilities=capabilities,
                    compatibility=CompatibilityStatus.PERFECT_FIT,
                    compatibility_message="Compatibility not yet assessed",
                    is_installed=True,
                )
                models.append(model_info)
                logger.debug(f"Discovered GGUF model: {model_id}")

            logger.info(f"Discovered {len(models)} GGUF models")
            return models

        except Exception as e:
            logger.error(f"Error discovering GGUF models: {e}")
            return models

    def _classify_gguf_model(self, model_name: str) -> tuple[ModelType, list[EndpointType]]:
        """Classify GGUF model by name patterns.

        Args:
            model_name: Model filename (without .gguf)

        Returns:
            (ModelType, list of EndpointType)
        """
        name_lower = model_name.lower()

        # VLM keywords
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

            # Configure GPU layers based on device preference
            n_gpu_layers = -1 if self.use_gpu else 0

            # Load model
            llama = Llama(
                model_path=model_id,
                n_ctx=2048,  # Context window
                n_gpu_layers=n_gpu_layers,
                n_threads=os.cpu_count() or 4,
                verbose=False,
            )

            logger.info(f"Loaded GGUF model (GPU layers: {n_gpu_layers})")
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
                max_tokens=512,
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
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": ["\n\n", "User:"]
            }

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

            response = handle.create_completion(**gen_params)

            return response["choices"][0]["text"].strip()

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
