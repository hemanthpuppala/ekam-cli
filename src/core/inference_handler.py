"""Universal inference handler for all model types (LLM, VLM, and future architectures).

This module provides a unified interface for running inference on different model types
without worrying about processor/tokenizer differences. Supports:
- Vision-Language Models (VLM) with images
- Language Models (LLM) with text only
- Hybrid models with multiple modalities
- Custom model APIs (moondream2, etc.)
- Graceful fallbacks and error handling
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union
from PIL import Image
from loguru import logger
import torch

from ..models.endpoints import ModelType


class ProcessorWrapper:
    """Unified wrapper for both VLM processors and LLM tokenizers.

    This allows treating them uniformly in inference code, handling the differences
    in their APIs transparently.
    """

    def __init__(self, processor: Any, processor_type: str = "unknown"):
        """Initialize processor wrapper.

        Args:
            processor: Either a VLM processor or LLM tokenizer
            processor_type: One of 'vlm_processor', 'llm_tokenizer', or 'unknown'
        """
        self.processor = processor
        self.processor_type = processor_type
        self.name = type(processor).__name__

        # Detect actual type if unknown
        if processor_type == "unknown":
            self._detect_processor_type()

        logger.debug(
            f"Initialized ProcessorWrapper: {self.name} "
            f"(type={self.processor_type})"
        )

    def _detect_processor_type(self) -> None:
        """Detect whether we have a processor or tokenizer."""
        processor_class_name = type(self.processor).__name__.lower()

        # VLM Processor indicators
        if "processor" in processor_class_name:
            self.processor_type = "vlm_processor"
        elif "tokenizer" in processor_class_name:
            self.processor_type = "llm_tokenizer"
        elif hasattr(self.processor, "apply_chat_template"):
            # Advanced processors have chat template support
            self.processor_type = "vlm_processor"
        else:
            # Default to tokenizer if uncertain
            self.processor_type = "llm_tokenizer"

        logger.debug(
            f"Auto-detected processor type: {self.processor_type} "
            f"(class={processor_class_name})"
        )

    def is_vlm_processor(self) -> bool:
        """Check if this is a VLM processor."""
        return self.processor_type == "vlm_processor"

    def is_llm_tokenizer(self) -> bool:
        """Check if this is an LLM tokenizer."""
        return self.processor_type == "llm_tokenizer"

    def prepare_inputs(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Prepare inputs for inference, handling both VLM and LLM cases.

        Args:
            text: Input text/question
            image: Optional PIL Image (for VLMs)
            device: Device to move inputs to (cpu, cuda, mps, etc.)

        Returns:
            Dictionary of inputs ready for model.generate()

        Raises:
            RuntimeError: If inputs cannot be prepared
        """
        if self.is_llm_tokenizer():
            # LLM case: text only
            return self._prepare_llm_inputs(text, device)
        else:
            # VLM case: text and optional image
            return self._prepare_vlm_inputs(text, image, device)

    def _prepare_llm_inputs(self, text: str, device: str) -> dict[str, Any]:
        """Prepare inputs for LLM tokenizer.

        Args:
            text: Input text
            device: Device to move to

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        try:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
            )
            logger.debug("✓ Prepared LLM inputs (text only)")
            return self._move_to_device(inputs, device)
        except Exception as e:
            logger.error(f"Failed to prepare LLM inputs: {e}")
            raise RuntimeError(f"Failed to prepare inputs: {e}")

    def _prepare_vlm_inputs(
        self, text: str, image: Optional[Image.Image], device: str
    ) -> dict[str, Any]:
        """Prepare inputs for VLM processor.

        Args:
            text: Input text
            image: PIL Image
            device: Device to move to

        Returns:
            Dictionary with image data + text inputs

        Raises:
            RuntimeError: If image processing fails
        """
        if image is None:
            logger.warning("VLM processor requires image but none provided - using text only")
            return self._prepare_llm_inputs(text, device)

        inputs = None
        last_error = None

        # Try multiple input preparation strategies
        strategies = [
            ("messages + chat_template", self._try_messages_format),
            ("standard text+images", self._try_standard_format),
        ]

        for strategy_name, strategy_func in strategies:
            try:
                inputs = strategy_func(text, image)
                if inputs and self._has_image_data(inputs):
                    logger.debug(f"✓ Successfully used {strategy_name} format")
                    return self._move_to_device(inputs, device)
            except Exception as e:
                last_error = e
                logger.debug(f"  {strategy_name} failed: {e}")

        # All strategies failed
        error_msg = (
            f"Failed to prepare VLM inputs after {len(strategies)} strategies\n"
            f"Processor: {self.name}\n"
            f"Last error: {last_error}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _try_messages_format(self, text: str, image: Image.Image) -> Optional[dict]:
        """Try messages format with chat template.

        Args:
            text: Input text
            image: PIL Image

        Returns:
            Prepared inputs or None if strategy not applicable
        """
        if not hasattr(self.processor, "apply_chat_template"):
            return None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

    def _try_standard_format(self, text: str, image: Image.Image) -> Optional[dict]:
        """Try standard text+images format.

        Args:
            text: Input text
            image: PIL Image

        Returns:
            Prepared inputs
        """
        # Try with list first
        try:
            return self.processor(
                text=text,
                images=[image],
                return_tensors="pt",
            )
        except TypeError:
            # Some processors don't want list
            return self.processor(
                text=text,
                images=image,
                return_tensors="pt",
            )

    def _has_image_data(self, inputs: dict) -> bool:
        """Check if inputs contain image data.

        Args:
            inputs: Dictionary of model inputs

        Returns:
            True if image data is present
        """
        image_keys = {
            "pixel_values",
            "image_embeds",
            "pixel_values_videos",
            "images",
        }
        return any(key in inputs for key in image_keys)

    def _move_to_device(
        self, inputs: dict[str, Any], device: str
    ) -> dict[str, Any]:
        """Move tensor inputs to device.

        Args:
            inputs: Dictionary of inputs
            device: Target device

        Returns:
            Dictionary with tensors moved to device
        """
        return {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

    def batch_decode(self, output: torch.Tensor, **kwargs) -> list[str]:
        """Decode model output.

        Args:
            output: Model output tensor
            **kwargs: Additional arguments for decoder

        Returns:
            List of decoded strings
        """
        return self.processor.batch_decode(output, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to underlying processor.

        Args:
            name: Attribute name

        Returns:
            Attribute from underlying processor
        """
        return getattr(self.processor, name)


class UniversalInferenceHandler:
    """Universal handler for running inference on any model type.

    Abstracts away differences between LLMs, VLMs, and future model architectures.
    Provides a unified interface and handles error cases gracefully.
    """

    def __init__(self, model: Any, processor: Any, model_type: ModelType = ModelType.LLM):
        """Initialize inference handler.

        Args:
            model: Loaded model instance
            processor: Either VLM processor or LLM tokenizer
            model_type: Type of model (LLM or VLM)
        """
        self.model = model
        self.model_type = model_type

        # Check for processor/model mismatch and try to fix
        self.processor_wrapper = self._get_or_fix_processor(processor, model)

        logger.info(
            f"Initialized UniversalInferenceHandler for {model_type.value} "
            f"(processor={self.processor_wrapper.name})"
        )

    def _get_or_fix_processor(self, processor: Any, model: Any) -> "ProcessorWrapper":
        """Get or fix processor if there's a model/processor mismatch.

        If model is a VLM but processor is a tokenizer, try to load proper processor.

        Args:
            processor: Initial processor/tokenizer
            model: Model instance

        Returns:
            ProcessorWrapper with correct processor
        """
        wrapper = ProcessorWrapper(processor)

        # Check if there's a mismatch: VLM model but tokenizer processor
        is_vlm_model = self.model_type == ModelType.VLM
        is_tokenizer = wrapper.is_llm_tokenizer()

        if is_vlm_model and is_tokenizer:
            logger.warning(
                f"Detected VLM model ({type(model).__name__}) with LLM tokenizer "
                f"({type(processor).__name__}). Attempting to load proper VLM processor..."
            )

            try:
                from transformers import AutoProcessor

                # Get model ID from model.config if available
                model_id = None
                if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                    # Try to infer from model
                    if hasattr(model, '_model_id'):
                        model_id = model._model_id

                if model_id:
                    logger.info(f"Loading proper VLM processor for {model_id}")
                    try:
                        proper_processor = AutoProcessor.from_pretrained(
                            model_id,
                            trust_remote_code=True
                        )
                        wrapper = ProcessorWrapper(proper_processor)
                        logger.info(f"✓ Loaded proper VLM processor: {type(proper_processor).__name__}")
                        return wrapper
                    except Exception as e:
                        logger.debug(f"Failed to load VLM processor: {e}")
            except ImportError:
                logger.debug("AutoProcessor not available")

            logger.warning(
                "Could not load proper VLM processor. Will attempt inference with tokenizer. "
                "This may fail for vision-dependent operations."
            )

        return wrapper

    def run_qa(
        self,
        image: Optional[Image.Image],
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        max_new_tokens: int = 1024,
        stream_callback: Optional[callable] = None,
    ) -> str:
        """Run Q&A inference with optional image and conversation history.

        Handles both VLM (with image) and LLM (text only) cases.

        Args:
            image: Optional PIL Image (VLMs only)
            question: Question text
            conversation_history: Optional conversation history for context
            max_new_tokens: Maximum tokens to generate
            stream_callback: Optional callback for token streaming (delta: str, is_first: bool)

        Returns:
            Generated answer text

        Raises:
            RuntimeError: If inference fails
        """
        try:
            # Get model device
            model_device = next(self.model.parameters()).device

            # Format question with history if provided
            full_question = self._format_question(question, conversation_history)

            # Check for special model APIs
            if self._try_special_api(image, full_question, model_device):
                return self._special_api_result

            # Prepare inputs
            logger.debug(f"Preparing inputs for {self.model_type.value}...")
            inputs = self.processor_wrapper.prepare_inputs(
                text=full_question,
                image=image if self.model_type == ModelType.VLM else None,
                device=str(model_device),
            )

            logger.debug(f"Input keys: {list(inputs.keys())}")

            # Warn if on CPU
            if str(model_device) == "cpu":
                logger.warning(
                    "⏳ Running inference on CPU - this will be SLOW (30s-2min). "
                    "Consider using smaller model or GPU."
                )
                max_new_tokens = min(max_new_tokens, 512)

            # Generate response with optional streaming
            logger.info("Generating response...")
            
            if stream_callback:
                # Streaming mode using TextIteratorStreamer
                response = self._generate_streaming(
                    inputs, max_new_tokens, model_device, stream_callback
                )
            else:
                # Non-streaming mode (traditional)
                output = self._generate(inputs, max_new_tokens, model_device)
                # Decode response
                response = self.processor_wrapper.batch_decode(
                    output, skip_special_tokens=True
                )[0]

            # Extract answer if using history format
            if conversation_history and "A:" in response:
                parts = response.split("A:")
                if len(parts) > 1:
                    response = parts[-1].strip()

            logger.info("✓ Response generated successfully")
            return response.strip()

        except Exception as e:
            logger.error(f"QA inference failed: {e}", exc_info=True)
            raise

    def _format_question(
        self,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Format question with conversation history.

        Args:
            question: Current question
            conversation_history: Optional history

        Returns:
            Formatted question with history
        """
        if not conversation_history:
            return question

        # Format history as Q&A pairs
        from ..utils.history_formatter import format_qa_history
        return format_qa_history(
            conversation_history=conversation_history,
            current_question=question,
            max_turns=5,
        )

    def _try_special_api(
        self, image: Optional[Image.Image], question: str, device: torch.device
    ) -> bool:
        """Try special model APIs (moondream2, etc.).

        Args:
            image: Optional PIL Image
            question: Question text
            device: Model device

        Returns:
            True if special API was successfully used
        """
        # Moondream2 special API
        if hasattr(self.model, "encode_image") and hasattr(
            self.model, "answer_question"
        ):
            try:
                logger.debug("Detected moondream2 custom API")
                image_embeds = self.model.encode_image(image)
                answer = self.model.answer_question(
                    image_embeds=image_embeds,
                    question=question,
                    tokenizer=self.processor_wrapper.processor,
                )
                self._special_api_result = answer.strip()
                logger.info("✓ Response generated with moondream2 API")
                return True
            except Exception as e:
                logger.debug(f"Moondream2 API failed: {e}")
                return False

        return False

    def _generate(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate model output with error handling.

        Args:
            inputs: Prepared model inputs
            max_new_tokens: Maximum new tokens
            device: Model device

        Returns:
            Generated output tensor
        """
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": False,
        }

        try:
            with torch.no_grad():
                return self.model.generate(**inputs, **generation_config)

        except AssertionError as e:
            # InternVL and similar may have special requirements
            logger.error(f"Model assertion failed: {e}")
            if "img_context_token_id" in str(e):
                raise RuntimeError(
                    "Model requires proper image context tokens. "
                    "Try with different image or use different VLM."
                )
            raise

        except RuntimeError as e:
            # MPS missing operator fallback
            error_msg = str(e)
            if (
                "aten::_upsample_bilinear2d_aa" in error_msg
                and "mps" in str(device).lower()
                and "not currently implemented" in error_msg
            ):
                logger.warning(
                    "MPS missing upsample operator - falling back to CPU"
                )
                return self._generate_on_cpu(inputs, generation_config)
            raise

    def _generate_on_cpu(
        self, inputs: dict[str, Any], generation_config: dict
    ) -> torch.Tensor:
        """Fall back to CPU generation.

        Args:
            inputs: Model inputs
            generation_config: Generation configuration

        Returns:
            Generated output tensor
        """
        try:
            torch.mps.empty_cache()  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            pass

        # Move model and inputs to CPU
        self.model.to("cpu")
        cpu_inputs = {
            k: v.to("cpu") if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        # Reduce tokens for faster CPU inference
        cpu_config = dict(generation_config)
        cpu_config["max_new_tokens"] = min(cpu_config.get("max_new_tokens", 512), 512)

        with torch.no_grad():
            return self.model.generate(**cpu_inputs, **cpu_config)

    def _generate_streaming(
        self,
        inputs: dict[str, Any],
        max_new_tokens: int,
        device: Any,
        stream_callback: callable
    ) -> str:
        """Generate text with token-by-token streaming.
        
        Args:
            inputs: Model inputs
            max_new_tokens: Maximum tokens to generate
            device: Device to run on
            stream_callback: Callback function(delta: str, is_first: bool)
            
        Returns:
            Full generated text
        """
        try:
            from transformers import TextIteratorStreamer
            import threading
        except ImportError:
            logger.warning("TextIteratorStreamer not available, falling back to non-streaming")
            output = self._generate(inputs, max_new_tokens, device)
            response = self.processor_wrapper.batch_decode(output, skip_special_tokens=True)[0]
            return response
        
        try:
            # Create streamer
            streamer = TextIteratorStreamer(
                self.processor_wrapper.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Prepare generation config
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "streamer": streamer,
            }
            
            # Start generation in background thread
            generation_kwargs = dict(inputs, **generation_config)
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream tokens to callback
            full_response = ""
            is_first_token = True
            for new_text in streamer:
                if new_text:
                    full_response += new_text
                    stream_callback(new_text, is_first=is_first_token)
                    is_first_token = False
            
            thread.join()
            return full_response
            
        except Exception as e:
            logger.warning(f"Streaming failed: {e}, falling back to non-streaming")
            output = self._generate(inputs, max_new_tokens, device)
            response = self.processor_wrapper.batch_decode(output, skip_special_tokens=True)[0]
            return response

    def run_text_generation(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
    ) -> str:
        """Run text generation (for LLMs).

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text

        Raises:
            RuntimeError: If generation fails
        """
        try:
            model_device = next(self.model.parameters()).device

            inputs = self.processor_wrapper.prepare_inputs(
                text=prompt, device=str(model_device)
            )

            if str(model_device) == "cpu":
                max_new_tokens = min(max_new_tokens, 512)

            logger.info("Generating text...")
            output = self._generate(inputs, max_new_tokens, model_device)

            response = self.processor_wrapper.batch_decode(
                output, skip_special_tokens=True
            )[0]

            logger.info("✓ Text generation complete")
            return response.strip()

        except Exception as e:
            logger.error(f"Text generation failed: {e}", exc_info=True)
            raise
