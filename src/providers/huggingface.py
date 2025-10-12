"""HuggingFace provider implementation using transformers library."""

import os
from pathlib import Path
from typing import Any, Optional

import torch
from loguru import logger
from PIL import Image

from ..models.endpoints import CompatibilityStatus, EndpointType, ModelType
from ..models.model import ModelInfo
from ..models.provider import ProviderConfig
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

        # Determine best available device
        self.device = self._get_best_device()
        logger.info(f"HuggingFace provider initialized on device: {self.device}")

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
        """Discover cached HuggingFace models.

        Returns:
            List of ModelInfo objects for cached models
        """
        models = []

        # Scan cache directory for model directories
        if not self.cache_dir.exists():
            logger.warning(f"HuggingFace cache directory not found: {self.cache_dir}")
            return models

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

                        # Estimate size and classify model
                        size_gb = self._estimate_cached_model_size(model_dir)
                        model_type, capabilities = self._classify_hf_model(model_id)

                        model_info = ModelInfo(
                            model_id=model_id,
                            name=model_id,
                            provider="huggingface",
                            size_gb=size_gb,
                            model_type=model_type,
                            capabilities=capabilities,
                            compatibility=CompatibilityStatus.PERFECT_FIT,  # Will be assessed later
                            compatibility_message="Compatibility not yet assessed",
                            is_installed=True,
                        )
                        models.append(model_info)
                        logger.debug(f"Discovered HF model: {model_id}")

            logger.info(f"Discovered {len(models)} HuggingFace models from cache")
            return models

        except Exception as e:
            logger.error(f"Error discovering HuggingFace models: {e}")
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

    def _classify_hf_model(self, model_id: str) -> tuple[ModelType, list[EndpointType]]:
        """Classify HuggingFace model by name patterns.

        Args:
            model_id: Model identifier (e.g., "llava-hf/llava-1.5-7b-hf")

        Returns:
            (ModelType, list of EndpointType)
        """
        model_lower = model_id.lower()

        # VLM keywords
        vlm_keywords = [
            "llava", "blip", "instructblip", "clip", "vision", "vit",
            "paligemma", "idefics", "fuyu", "kosmos", "qwen-vl",
            "cogvlm", "internvl", "minicpm-v", "phi-3-vision"
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
        """Load HuggingFace model with transformers.

        Args:
            model_id: Model identifier (e.g., "llava-hf/llava-1.5-7b-hf")
            device: Target device (used self.device instead)

        Returns:
            Tuple of (model, processor/tokenizer)
        """
        logger.info(f"Loading HuggingFace model: {model_id}")

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForVision2Seq,
                AutoProcessor,
                AutoTokenizer,
            )

            model_lower = model_id.lower()

            # Determine if this is a VLM based on model name
            vlm_keywords = [
                "llava", "blip", "instructblip", "clip", "vision",
                "paligemma", "idefics", "fuyu", "kosmos", "qwen-vl",
                "cogvlm", "internvl", "minicpm-v", "phi-3-vision"
            ]
            is_vlm = any(keyword in model_lower for keyword in vlm_keywords)

            if is_vlm:
                # Try to load as VLM with processor
                try:
                    processor = AutoProcessor.from_pretrained(
                        model_id,
                        cache_dir=str(self.cache_dir),
                        trust_remote_code=True
                    )
                    # Try vision-to-seq model first
                    try:
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_id,
                            cache_dir=str(self.cache_dir),
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                        )
                    except Exception:
                        # Fall back to generic causal LM for some VLMs
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            cache_dir=str(self.cache_dir),
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                        )
                    model.to(self.device)
                    logger.info(f"Loaded VLM model {model_id} on {self.device}")
                    return (model, processor)
                except Exception as e:
                    logger.warning(f"Failed to load as VLM: {e}, trying as LLM...")

            # Load as LLM with causal language modeling head
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            model.to(self.device)
            logger.info(f"Loaded LLM model {model_id} on {self.device}")
            return (model, tokenizer)

        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {model_id}: {e}")
            raise RuntimeError(f"Failed to load model {model_id}: {e}")

    def unload_model(self, handle: Any) -> None:
        """Unload model and free GPU memory.

        Args:
            handle: Tuple of (model, processor/tokenizer)
        """
        if handle is None:
            return

        try:
            model, _ = handle
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
        """Run question answering with VLM using unified history formatter.

        Args:
            handle: Tuple of (model, processor)
            image: PIL Image
            question: Question text
            conversation_history: Optional list of (user_msg, bot_response) tuples

        Returns:
            Answer text
        """
        model, processor = handle

        try:
            # Use unified Q&A history formatter (shared across all providers)
            full_question = format_qa_history(
                conversation_history=conversation_history,
                current_question=question,
                max_turns=5
            )

            # Prepare inputs
            inputs = processor(
                text=full_question,
                images=image,
                return_tensors="pt"
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=512)

            # Decode response
            response = processor.batch_decode(output, skip_special_tokens=True)[0]

            # If we used history format, try to extract just the answer
            if conversation_history and "A:" in response:
                # Try to extract the last answer after "A:"
                parts = response.split("A:")
                if len(parts) > 1:
                    response = parts[-1].strip()

            return response.strip()

        except Exception as e:
            logger.error(f"QA inference failed: {e}")
            raise NotImplementedError(f"QA not supported for this model: {e}")

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

            # Tokenize input
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(self.device)

            # Build generation parameters with defaults
            gen_params = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
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

            # Generate with parameters
            with torch.no_grad():
                output = model.generate(**inputs, **gen_params)

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

            return response if response else "I don't have a response."

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")

    def install_model(self, model_name: str, progress_callback=None) -> bool:
        """Download model from HuggingFace Hub.

        Args:
            model_name: Model identifier (e.g., "llava-hf/llava-1.5-7b-hf")
            progress_callback: Optional progress callback (not implemented)

        Returns:
            True if successful
        """
        logger.info(f"Downloading HuggingFace model: {model_name}")

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoProcessor,
                AutoTokenizer,
            )

            model_lower = model_name.lower()

            # Check if VLM
            vlm_keywords = [
                "llava", "blip", "instructblip", "vision",
                "paligemma", "idefics", "fuyu", "kosmos", "qwen-vl"
            ]
            is_vlm = any(keyword in model_lower for keyword in vlm_keywords)

            if is_vlm:
                # Download processor for VLM
                try:
                    AutoProcessor.from_pretrained(
                        model_name,
                        cache_dir=str(self.cache_dir),
                        trust_remote_code=True
                    )
                except Exception:
                    pass

            # Download tokenizer (for all models)
            AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )

            # Download model with causal LM head
            AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )

            logger.info(f"Successfully downloaded {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False

    def delete_model(self, model_id: str) -> bool:
        """Delete cached model from disk.

        Args:
            model_id: Model identifier

        Returns:
            True if successful
        """
        logger.info(f"Deleting HuggingFace model: {model_id}")

        try:
            # Convert model_id to directory name (org/name -> models--org--name)
            dir_name = "models--" + model_id.replace("/", "--")
            model_dir = self.cache_dir / dir_name

            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                logger.info(f"Deleted {model_id}")
                return True
            else:
                logger.warning(f"Model directory not found: {model_dir}")
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
