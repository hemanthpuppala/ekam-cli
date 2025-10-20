"""MLX provider for Apple Silicon optimized VLM inference.

MLX is Apple's native ML framework optimized for Apple Silicon with:
- Metal GPU acceleration
- Unified memory architecture utilization
- Built-in 4-bit quantization support
- Up to 85x faster than standard transformers on Mac
- Native VLM support via mlx-vlm library

Research: https://github.com/ml-explore/mlx
VLM: https://github.com/Blaizzy/mlx-vlm
"""

import os
import platform
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from PIL import Image

from ..models.endpoints import CompatibilityStatus, EndpointType, ModelType
from ..models.model import ModelInfo
from ..models.provider import ProviderConfig
from ..utils.history_formatter import format_qa_history
from .base import BaseProvider


class MLXProvider(BaseProvider):
    """MLX provider for Apple Silicon optimized inference.

    Only available on macOS with Apple Silicon (M1/M2/M3/M4).
    Provides native Metal acceleration and efficient 4-bit quantization.
    """

    def __init__(self, config: ProviderConfig):
        """Initialize MLX provider.

        Args:
            config: Provider configuration

        Raises:
            RuntimeError: If not on macOS or MLX not available
        """
        self.config = config
        self.models_dir = Path(str(config.models_dir or "~/.cache/mlx/models")).expanduser()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Check platform compatibility
        if platform.system() != "Darwin":
            raise RuntimeError("MLX provider only available on macOS")

        # Check for Apple Silicon
        machine = platform.machine()
        if machine != "arm64":
            logger.warning(f"MLX is optimized for Apple Silicon, found: {machine}")

        # Verify MLX is available
        try:
            import mlx.core as mx  # noqa: F401
            logger.info("✓ MLX framework available")
        except ImportError:
            raise RuntimeError(
                "MLX not installed. Install with: pip install mlx mlx-lm mlx-vlm"
            )

        # Check for VLM support
        try:
            import mlx_vlm  # noqa: F401
            self.vlm_available = True
            logger.info("✓ MLX-VLM available for vision-language models")
        except ImportError:
            self.vlm_available = False
            logger.warning("MLX-VLM not installed (pip install mlx-vlm for VLM support)")

        logger.info("MLX provider initialized for Apple Silicon")

    def discover_models(self) -> list[ModelInfo]:
        """Discover cached MLX models and mlx-community models.

        Returns:
            List of ModelInfo objects for available models
        """
        models = []

        # Discover local MLX models
        if self.models_dir.exists():
            try:
                for model_dir in self.models_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    # Check for MLX model files (weights.npz or config.json)
                    has_weights = (model_dir / "weights.npz").exists()
                    has_config = (model_dir / "config.json").exists()

                    if has_weights or has_config:
                        model_id = model_dir.name
                        size_gb = self._estimate_model_size(model_dir)

                        # Detect if VLM based on name or config
                        is_vlm = self._is_vlm_model(model_id, model_dir)

                        if is_vlm:
                            model_type = ModelType.VLM
                            capabilities = [
                                EndpointType.QA,
                                EndpointType.CAPTION,
                                EndpointType.TEXT,
                            ]
                        else:
                            model_type = ModelType.LLM
                            capabilities = [EndpointType.TEXT]

                        models.append(
                            ModelInfo(
                                model_id=f"mlx-community/{model_id}",
                                name=f"{model_id} (MLX)",
                                provider="mlx",
                                size_gb=size_gb,
                                model_type=model_type,
                                capabilities=capabilities,
                                compatibility=CompatibilityStatus.PERFECT_FIT,
                                compatibility_message="Optimized for Apple Silicon",
                                is_installed=True,
                            )
                        )
                        logger.debug(f"Discovered MLX model: {model_id}")

            except Exception as e:
                logger.error(f"Error discovering MLX models: {e}")

        logger.info(f"Total MLX models discovered: {len(models)}")
        return models

    def _estimate_model_size(self, model_dir: Path) -> float:
        """Estimate model size in GB.

        Args:
            model_dir: Path to model directory

        Returns:
            Size in GB
        """
        try:
            total_size = 0
            for file in model_dir.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size / (1024**3)
        except Exception as e:
            logger.warning(f"Could not estimate size for {model_dir}: {e}")
            return 2.0  # MLX models are typically smaller due to quantization

    def _is_vlm_model(self, model_id: str, model_dir: Path) -> bool:
        """Check if model is a VLM.

        Args:
            model_id: Model identifier
            model_dir: Model directory path

        Returns:
            True if VLM, False otherwise
        """
        model_lower = model_id.lower()
        vlm_keywords = [
            "llava", "qwen-vl", "qwen2-vl", "phi-3-vision", "pixtral",
            "idefics", "paligemma", "cogvlm", "internvl", "minicpm-v"
        ]

        return any(keyword in model_lower for keyword in vlm_keywords)

    def load_model(self, model_id: str, device: str) -> Any:
        """Load MLX model into memory.

        Args:
            model_id: Model identifier (e.g., "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit")
            device: Ignored (MLX automatically uses Metal)

        Returns:
            Tuple of (model, processor/tokenizer)

        Raises:
            RuntimeError: If model cannot be loaded
        """
        logger.info(f"Loading MLX model: {model_id}")

        try:
            # Determine if VLM
            is_vlm = self._is_vlm_model(model_id, Path(model_id))

            if is_vlm and self.vlm_available:
                return self._load_vlm(model_id)
            else:
                return self._load_llm(model_id)

        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise RuntimeError(f"Could not load MLX model {model_id}: {e}")

    def _load_vlm(self, model_id: str) -> tuple:
        """Load VLM with mlx-vlm.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (model, processor)
        """
        import mlx_vlm

        logger.info("Loading VLM with mlx-vlm...")

        # mlx-vlm automatically handles model loading and quantization
        model, processor = mlx_vlm.load(model_id)

        logger.info(f"✓ Loaded MLX VLM: {model_id}")
        return (model, processor)

    def _load_llm(self, model_id: str) -> tuple:
        """Load LLM with mlx-lm.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            import mlx_lm
        except ImportError:
            raise RuntimeError("mlx-lm not installed. Install with: pip install mlx-lm")

        logger.info("Loading LLM with mlx-lm...")

        # mlx-lm automatically handles model loading and quantization
        model, tokenizer = mlx_lm.load(model_id)

        logger.info(f"✓ Loaded MLX LLM: {model_id}")
        return (model, tokenizer)

    def unload_model(self, handle: Any) -> None:
        """Unload MLX model from memory.

        Args:
            handle: Model handle (model, processor/tokenizer tuple)
        """
        # MLX uses unified memory, no explicit unloading needed
        # Garbage collection handles cleanup automatically
        logger.debug("MLX model will be garbage collected")

    def run_qa(
        self,
        handle: Any,
        image: Image.Image,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Run question answering on image with MLX-VLM.

        Args:
            handle: Loaded model handle (model, processor)
            image: PIL Image to analyze
            question: Question to answer
            conversation_history: Optional conversation history

        Returns:
            Answer text

        Raises:
            NotImplementedError: If not a VLM
        """
        if not self.vlm_available:
            raise NotImplementedError("MLX-VLM not installed")

        model, processor = handle

        try:
            import mlx_vlm

            # Format question with conversation history
            if conversation_history:
                formatted_question = format_qa_history(
                    conversation_history=conversation_history,
                    current_question=question,
                    max_turns=5
                )
            else:
                formatted_question = question

            # Generate response
            logger.info("Generating response with MLX-VLM...")
            response = mlx_vlm.generate(
                model=model,
                processor=processor,
                image=image,
                prompt=formatted_question,
                max_tokens=1024,
                temp=0.0,  # Deterministic for QA
            )

            logger.info("✓ Response generated")
            
            # Extract response text from GenerationResult
            response_text = response.text if hasattr(response, 'text') else str(response)
            response_text = response_text.strip()
            
            # Apply response cleaning to remove artifacts, reasoning, and <think> tags
            from ..utils.response_cleaner import clean_model_response
            from ..utils.response_formatter import extract_thinking_blocks
            
            # Extract and remove thinking blocks
            thinking_blocks, clean_response = extract_thinking_blocks(response_text)
            if thinking_blocks:
                logger.debug(f"Filtered {len(thinking_blocks)} thinking block(s) from response")
            
            # Clean the response for artifacts and meta-commentary
            if clean_response:
                clean_response = clean_model_response(clean_response, aggressive=True)
            
            return clean_response if clean_response else "I don't have a response."

        except Exception as e:
            logger.error(f"MLX-VLM QA failed: {e}")
            raise NotImplementedError(f"QA not supported: {e}")

    def run_caption(
        self,
        handle: Any,
        image: Image.Image,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        detail_level: str = "detailed",
    ) -> str:
        """Generate image caption with MLX-VLM.

        Args:
            handle: Loaded model handle (model, processor)
            image: PIL Image to caption
            conversation_history: Optional conversation history
            detail_level: "detailed" or "short"

        Returns:
            Caption text
        """
        if detail_level == "detailed":
            prompt = "Provide a detailed description of this image, including all visible objects, actions, and context."
        else:
            prompt = "Provide a brief caption for this image in one sentence."

        return self.run_qa(handle, image, prompt, conversation_history)

    def run_detect(
        self,
        handle: Any,
        image: Image.Image,
        object_name: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Run object detection.

        Args:
            handle: Loaded model handle
            image: PIL Image
            object_name: Object to detect
            conversation_history: Optional conversation history

        Returns:
            Detection response
        """
        prompt = (
            f"Detect all instances of '{object_name}' in this image. "
            f"For each instance, provide bounding box coordinates in the format: "
            f"[x_min, y_min, x_max, y_max]"
        )
        return self.run_qa(handle, image, prompt, conversation_history)

    def run_point(
        self,
        handle: Any,
        image: Image.Image,
        object_name: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Run object pointing/localization.

        Args:
            handle: Loaded model handle
            image: PIL Image
            object_name: Object to locate
            conversation_history: Optional conversation history

        Returns:
            Pointing response
        """
        prompt = (
            f"Point to the '{object_name}' in this image. "
            f"Provide the center coordinates as [x, y] where x and y are between 0 and 1."
        )
        return self.run_qa(handle, image, prompt, conversation_history)

    def run_text(
        self,
        handle: Any,
        prompt: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        custom_parameters: Optional[dict] = None,
    ) -> str:
        """Run text generation with MLX-LM.

        Args:
            handle: Loaded model handle (model, tokenizer)
            prompt: Input prompt
            conversation_history: Optional conversation history
            custom_parameters: Optional generation parameters

        Returns:
            Generated text
        """
        model, tokenizer = handle

        try:
            # Check if this is a VLM (has processor instead of tokenizer)
            if self.vlm_available and hasattr(handle[1], "__module__"):
                if "processor" in handle[1].__module__.lower():
                    # Use VLM text generation
                    import mlx_vlm

                    # Format prompt with conversation history
                    if conversation_history:
                        from ..utils.history_formatter import format_conversation_history
                        formatted_prompt = format_conversation_history(
                            conversation_history=conversation_history,
                            current_prompt=prompt,
                            max_turns=5
                        )
                    else:
                        formatted_prompt = prompt

                    response = mlx_vlm.generate(
                        model=model,
                        processor=handle[1],
                        prompt=formatted_prompt,
                        max_tokens=custom_parameters.get("max_tokens", 1024) if custom_parameters else 1024,
                        temp=custom_parameters.get("temperature", 0.7) if custom_parameters else 0.7,
                    )
                    
                    # Extract response text and clean it
                    response_text = response.text if hasattr(response, 'text') else str(response)
                    response_text = response_text.strip()
                    
                    # Apply response cleaning
                    from ..utils.response_cleaner import clean_model_response
                    from ..utils.response_formatter import extract_thinking_blocks
                    
                    thinking_blocks, clean_response = extract_thinking_blocks(response_text)
                    if thinking_blocks:
                        logger.debug(f"Filtered {len(thinking_blocks)} thinking block(s)")
                    
                    if clean_response:
                        clean_response = clean_model_response(clean_response, aggressive=True)
                    
                    return clean_response if clean_response else "I don't have a response."

            # Standard LLM text generation
            import mlx_lm

            # Format prompt with conversation history
            if conversation_history:
                from ..utils.history_formatter import format_conversation_history
                formatted_prompt = format_conversation_history(
                    conversation_history=conversation_history,
                    current_prompt=prompt,
                    max_turns=5
                )
            else:
                formatted_prompt = prompt

            # Build generation parameters
            max_tokens = custom_parameters.get("max_tokens", 1024) if custom_parameters else 1024
            temperature = custom_parameters.get("temperature", 0.7) if custom_parameters else 0.7

            logger.info("Generating text with MLX-LM...")
            response = mlx_lm.generate(
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )

            logger.info("✓ Text generated")
            response_text = response.strip()
            
            # Apply response cleaning to remove artifacts, reasoning, and <think> tags
            from ..utils.response_cleaner import clean_model_response
            from ..utils.response_formatter import extract_thinking_blocks
            
            # Extract and remove thinking blocks
            thinking_blocks, clean_response = extract_thinking_blocks(response_text)
            if thinking_blocks:
                logger.debug(f"Filtered {len(thinking_blocks)} thinking block(s) from response")
            
            # Clean the response for artifacts and meta-commentary
            if clean_response:
                clean_response = clean_model_response(clean_response, aggressive=True)
            
            return clean_response if clean_response else "I don't have a response."

        except Exception as e:
            logger.error(f"MLX text generation failed: {e}")
            raise NotImplementedError(f"Text generation not supported: {e}")

    def run_chat(
        self,
        handle: Any,
        messages: list[dict],
        custom_parameters: Optional[dict] = None,
    ) -> str:
        """Run chat completion.

        Args:
            handle: Loaded model handle
            messages: List of message dicts with 'role' and 'content'
            custom_parameters: Optional generation parameters

        Returns:
            Assistant response
        """
        # Convert messages to conversation history format
        conversation_history = []
        current_prompt = ""

        for msg in messages:
            if msg["role"] == "user":
                current_prompt = msg["content"]
            elif msg["role"] == "assistant":
                if current_prompt:
                    conversation_history.append((current_prompt, msg["content"]))
                    current_prompt = ""

        return self.run_text(handle, current_prompt, conversation_history, custom_parameters)
