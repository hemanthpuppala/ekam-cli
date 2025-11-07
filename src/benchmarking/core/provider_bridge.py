"""
Provider bridge for connecting benchmarking to real model providers.

Routes model requests to appropriate providers (Ollama, HuggingFace, GGUF, etc.)
"""

from typing import Any, Optional, Tuple
from pathlib import Path
from PIL import Image
from loguru import logger

from ...models.endpoints import ProviderType
from ...services.session import SessionManager
from ..utils import managed_image


class ProviderBridge:
    """
    Bridge between benchmarking system and application providers.

    Responsibilities:
    - Resolve model_id to provider
    - Load models via SessionManager
    - Execute inference via appropriate provider
    - Handle provider-specific quirks
    """

    def __init__(self, session_manager: SessionManager):
        """
        Initialize provider bridge.

        Args:
            session_manager: Application's session manager with registered providers
        """
        self.session_manager = session_manager
        self._loaded_models = {}  # model_id → (provider_type, handle)
        self._inference_methods = {}  # (model_id, endpoint) → successful_method_signature

    def resolve_provider(self, model_id: str) -> Optional[ProviderType]:
        """
        Resolve model_id to provider type.

        Args:
            model_id: Model identifier

        Returns:
            ProviderType or None if not found
        """
        # Check if model is in discovered models
        model_info = self.session_manager.state.get_model(model_id)
        if model_info:
            return model_info.provider

        # Fallback: parse from model_id format
        # FIX: All quantized models (regardless of format) should route to QUANTIZED provider
        # The QUANTIZED provider will parse the format and delegate appropriately
        if model_id.startswith("quantized:"):
            return ProviderType.QUANTIZED
        elif ":" in model_id and not "/" in model_id.split(":")[0]:
            # Format like "gemma3:270m" → Ollama
            return ProviderType.OLLAMA
        else:
            # Format like "user/model" → HuggingFace
            return ProviderType.HUGGINGFACE

    def load_model(self, model_id: str) -> Tuple[Optional[ProviderType], Optional[Any]]:
        """
        Load model and return provider + handle.

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (provider_type, handle) or (None, None) on failure
        """
        # Check if already loaded
        if model_id in self._loaded_models:
            logger.debug(f"Model {model_id} already loaded")
            return self._loaded_models[model_id]

        # Resolve provider
        provider_type = self.resolve_provider(model_id)
        if not provider_type:
            logger.error(f"Could not resolve provider for model {model_id}")
            return None, None

        # Get provider instance
        provider = self.session_manager.model_discovery.get_provider(provider_type)
        if not provider:
            logger.error(f"Provider {provider_type} not available")
            return None, None

        # Get device
        provider_config = self.session_manager.state.provider_configs.get(provider_type)
        device = provider_config.get_primary_device() if provider_config else "cpu"

        try:
            logger.info(f"Loading model {model_id} via {provider_type} on {device}")
            handle = provider.load_model(model_id, device)
            self._loaded_models[model_id] = (provider_type, handle)
            logger.info(f"Successfully loaded {model_id}")
            return provider_type, handle
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}", exc_info=True)
            return None, None

    def _try_vision_method(
        self,
        provider: Any,
        method_name: str,
        handle: Any,
        image: Any,
        prompt: str,
        model_id: str,
        endpoint: str,
        timeout: Optional[float] = None
    ) -> Optional[str]:
        """
        Try calling a vision method with different parameter conventions.

        Attempts multiple calling strategies and caches the successful one.

        Args:
            provider: Provider instance
            method_name: Method name (e.g., 'run_qa', 'run_caption')
            handle: Model handle
            image: PIL Image
            prompt: Text prompt
            model_id: Model identifier for caching
            endpoint: Endpoint name for caching
            timeout: Optional timeout in seconds for inference

        Returns:
            Output string if successful, None otherwise
        """
        if not hasattr(provider, method_name):
            return None

        method = getattr(provider, method_name)
        cache_key = (model_id, endpoint)

        # If we've already found a working method, use it
        if cache_key in self._inference_methods:
            cached_signature = self._inference_methods[cache_key]
            try:
                logger.debug(f"Using cached inference method: {cached_signature}")
                return self._call_with_signature(method, cached_signature, handle, image, prompt, timeout)
            except Exception as e:
                logger.warning(f"Cached method failed: {e}, trying fallbacks")
                # Continue to fallbacks if cached method fails

        # Try different calling conventions in order of preference
        # Now include timeout parameter where supported
        calling_strategies = []

        if method_name == "run_qa":
            calling_strategies = [
                ("positional_3_timeout", lambda: method(handle, image, prompt, timeout=timeout)),
                ("positional_3", lambda: method(handle, image, prompt)),
                ("positional_4_none_timeout", lambda: method(handle, image, prompt, None, timeout=timeout)),
                ("positional_4_none", lambda: method(handle, image, prompt, None)),
                ("keyword_handle_timeout", lambda: method(handle=handle, image=image, question=prompt, timeout=timeout)),
                ("keyword_handle", lambda: method(handle=handle, image=image, question=prompt)),
                ("keyword_model_handle_timeout", lambda: method(model_handle=handle, image=image, question=prompt, timeout=timeout)),
                ("keyword_model_handle", lambda: method(model_handle=handle, image=image, question=prompt)),
            ]
        elif method_name == "run_caption":
            calling_strategies = [
                ("positional_2_timeout", lambda: method(handle, image, timeout=timeout)),
                ("positional_2", lambda: method(handle, image)),
                ("positional_3_none_timeout", lambda: method(handle, image, None, timeout=timeout)),
                ("positional_3_none", lambda: method(handle, image, None)),
                ("keyword_handle_timeout", lambda: method(handle=handle, image=image, timeout=timeout)),
                ("keyword_handle", lambda: method(handle=handle, image=image)),
                ("keyword_model_handle_timeout", lambda: method(model_handle=handle, image=image, timeout=timeout)),
                ("keyword_model_handle", lambda: method(model_handle=handle, image=image)),
            ]
        elif method_name == "run_point":
            calling_strategies = [
                ("positional_3_timeout", lambda: method(handle, image, prompt, timeout=timeout)),
                ("positional_3", lambda: method(handle, image, prompt)),
                ("keyword_handle_timeout", lambda: method(handle=handle, image=image, object_name=prompt, timeout=timeout)),
                ("keyword_handle", lambda: method(handle=handle, image=image, object_name=prompt)),
                ("keyword_model_handle_timeout", lambda: method(model_handle=handle, image=image, object_name=prompt, timeout=timeout)),
                ("keyword_model_handle", lambda: method(model_handle=handle, image=image, object_name=prompt)),
            ]
        elif method_name == "run_detect":
            calling_strategies = [
                ("positional_3_timeout", lambda: method(handle, image, prompt, timeout=timeout)),
                ("positional_3", lambda: method(handle, image, prompt)),
                ("keyword_handle_timeout", lambda: method(handle=handle, image=image, object_name=prompt, timeout=timeout)),
                ("keyword_handle", lambda: method(handle=handle, image=image, object_name=prompt)),
                ("keyword_model_handle_timeout", lambda: method(model_handle=handle, image=image, object_name=prompt, timeout=timeout)),
                ("keyword_model_handle", lambda: method(model_handle=handle, image=image, object_name=prompt)),
            ]

        # Try each strategy
        for signature, call_func in calling_strategies:
            try:
                logger.debug(f"Trying {method_name} with signature: {signature}")
                output = call_func()

                if output is not None:
                    # Success! Cache this signature
                    logger.info(f"✓ Found working method for {model_id}:{endpoint} - {method_name}:{signature}")
                    self._inference_methods[cache_key] = signature
                    return output

            except TypeError as e:
                logger.debug(f"Signature {signature} failed with TypeError: {e}")
                continue
            except Exception as e:
                # Other exceptions might be real errors, log but continue trying
                logger.debug(f"Signature {signature} failed: {e}")
                continue

        return None

    def _call_with_signature(self, method, signature: str, handle, image, prompt, timeout: Optional[float] = None):
        """Call method using cached signature."""
        if signature == "positional_3_timeout":
            return method(handle, image, prompt, timeout=timeout)
        elif signature == "positional_3":
            return method(handle, image, prompt)
        elif signature == "positional_4_none_timeout":
            return method(handle, image, prompt, None, timeout=timeout)
        elif signature == "positional_4_none":
            return method(handle, image, prompt, None)
        elif signature == "positional_2_timeout":
            return method(handle, image, timeout=timeout)
        elif signature == "positional_2":
            return method(handle, image)
        elif signature == "positional_3_none_timeout":
            return method(handle, image, None, timeout=timeout)
        elif signature == "positional_3_none":
            return method(handle, image, None)
        elif signature == "keyword_handle_timeout":
            if "question" in str(method.__code__.co_varnames):
                return method(handle=handle, image=image, question=prompt, timeout=timeout)
            elif "object_name" in str(method.__code__.co_varnames):
                return method(handle=handle, image=image, object_name=prompt, timeout=timeout)
            else:
                return method(handle=handle, image=image, timeout=timeout)
        elif signature == "keyword_handle":
            if "question" in str(method.__code__.co_varnames):
                return method(handle=handle, image=image, question=prompt)
            elif "object_name" in str(method.__code__.co_varnames):
                return method(handle=handle, image=image, object_name=prompt)
            else:
                return method(handle=handle, image=image)
        elif signature == "keyword_model_handle_timeout":
            if "question" in str(method.__code__.co_varnames):
                return method(model_handle=handle, image=image, question=prompt, timeout=timeout)
            elif "object_name" in str(method.__code__.co_varnames):
                return method(model_handle=handle, image=image, object_name=prompt, timeout=timeout)
            else:
                return method(model_handle=handle, image=image, timeout=timeout)
        elif signature == "keyword_model_handle":
            if "question" in str(method.__code__.co_varnames):
                return method(model_handle=handle, image=image, question=prompt)
            elif "object_name" in str(method.__code__.co_varnames):
                return method(model_handle=handle, image=image, object_name=prompt)
            else:
                return method(model_handle=handle, image=image)
        else:
            raise ValueError(f"Unknown signature: {signature}")

    def execute_text(
        self,
        model_id: str,
        prompt: str,
        parameters: Optional[dict] = None
    ) -> Tuple[Optional[str], Optional[dict]]:
        """
        Execute text inference using appropriate provider.

        Args:
            model_id: Model identifier
            prompt: Input prompt
            parameters: Inference parameters

        Returns:
            Tuple of (output_text, metadata) or (None, None) on failure
        """
        # Load model if needed
        provider_type, handle = self.load_model(model_id)
        if not provider_type or not handle:
            return None, None

        # Get provider
        provider = self.session_manager.model_discovery.get_provider(provider_type)
        if not provider:
            logger.error(f"Provider {provider_type} not available")
            return None, None

        try:
            # Execute via provider
            # Different providers have different signatures:
            # - OllamaProvider: run_text(handle, prompt, conversation_history, system_prompt, custom_parameters)
            # - QuantizedProvider: run_text(model_handle, prompt, conversation_history, custom_parameters)
            logger.debug(f"Executing text inference: {model_id}")

            # Try with system_prompt first (Ollama), fall back without it (Quantized)
            try:
                output = provider.run_text(
                    handle,      # Positional: handle/model_handle
                    prompt,      # Positional: prompt
                    None,        # Positional: conversation_history
                    None,        # Positional: system_prompt (Ollama only)
                    parameters   # Positional: custom_parameters
                )
            except TypeError as te:
                # If too many positional args, try without system_prompt
                if 'system_prompt' in str(te) or 'positional argument' in str(te):
                    logger.debug(f"Provider doesn't support system_prompt, retrying without it")
                    output = provider.run_text(
                        handle,      # Positional: handle/model_handle
                        prompt,      # Positional: prompt
                        None,        # Positional: conversation_history
                        parameters   # Positional: custom_parameters
                    )
                else:
                    raise

            metadata = {
                "provider": str(provider_type),
                "model_id": model_id,
                "prompt_length": len(prompt),
                "output_length": len(output)
            }

            return output, metadata

        except Exception as e:
            logger.error(f"Text inference failed for {model_id}: {e}", exc_info=True)
            return None, None

    def execute_vision(
        self,
        model_id: str,
        image_path: str,
        prompt: str,
        parameters: Optional[dict] = None,
        endpoint: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Tuple[Optional[str], Optional[dict]]:
        """
        Execute vision inference using appropriate provider.

        Args:
            model_id: Model identifier
            image_path: Path to image file
            prompt: Text prompt
            parameters: Inference parameters
            endpoint: VLM endpoint name (e.g., 'vision/qa', 'vision/point')
            timeout: Optional timeout in seconds (default: 600)

        Returns:
            Tuple of (output_text, metadata) or (None, None) on failure
        """
        # Load model if needed
        provider_type, handle = self.load_model(model_id)
        if not provider_type or not handle:
            return None, None

        # Get provider
        provider = self.session_manager.model_discovery.get_provider(provider_type)
        if not provider:
            logger.error(f"Provider {provider_type} not available")
            return None, None

        # Default timeout for VLM inference: 600 seconds (10 minutes)
        request_timeout = timeout if timeout is not None else 600.0
        logger.debug(f"VLM inference timeout: {request_timeout}s")

        try:
            # Check if image exists first (O(1) check)
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                logger.error(f"Image not found: {image_path}")
                return None, None

            # Use managed_image context for automatic cleanup (O(1) overhead)
            with managed_image(image_path) as image:
                logger.debug(f"Executing vision inference: {model_id}")

                # Map endpoint to method name
                endpoint_to_method = {
                    "vision/qa": "run_qa",
                    "vision/caption": "run_caption",
                    "vision/point": "run_point",
                    "vision/detect": "run_detect"
                }

                output = None

                # Try the primary method for this endpoint
                if endpoint in endpoint_to_method:
                    method_name = endpoint_to_method[endpoint]
                    output = self._try_vision_method(
                        provider, method_name, handle, image, prompt, model_id, endpoint, request_timeout
                    )

                # If primary method didn't work, try fallback methods
                if output is None:
                    logger.warning(f"Primary method failed for {endpoint}, trying fallbacks")
                    fallback_methods = ["run_qa", "run_caption", "run_point", "run_detect"]

                    for method_name in fallback_methods:
                        if method_name == endpoint_to_method.get(endpoint):
                            continue  # Already tried this one

                        output = self._try_vision_method(
                            provider, method_name, handle, image, prompt, model_id, endpoint, request_timeout
                        )

                        if output is not None:
                            logger.info(f"✓ Fallback method {method_name} succeeded for {endpoint}")
                            break

                if output is None:
                    logger.error(f"All inference methods failed for {model_id}:{endpoint}")
                    return None, None

                metadata = {
                    "provider": str(provider_type),
                    "model_id": model_id,
                    "image_path": image_path,
                    "endpoint": endpoint,
                    "prompt_length": len(prompt),
                    "output_length": len(output)
                }

                return output, metadata
            # Image automatically closed and memory freed here

        except Exception as e:
            logger.error(f"Vision inference failed for {model_id}: {e}", exc_info=True)
            return None, None

    def unload_model(self, model_id: str) -> bool:
        """
        Unload model from memory.

        Args:
            model_id: Model identifier

        Returns:
            True if unloaded successfully
        """
        if model_id in self._loaded_models:
            provider_type, handle = self._loaded_models[model_id]
            provider = self.session_manager.model_discovery.get_provider(provider_type)

            if provider and hasattr(provider, "unload_model"):
                try:
                    provider.unload_model(handle)
                    logger.info(f"Unloaded model {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to unload {model_id}: {e}")

            del self._loaded_models[model_id]
            return True
        return False

    def unload_all(self):
        """Unload all loaded models."""
        model_ids = list(self._loaded_models.keys())
        for model_id in model_ids:
            self.unload_model(model_id)
