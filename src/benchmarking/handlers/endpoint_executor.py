"""
Endpoint executor for coordinating model inference.

Provides unified interface for executing inference across different
model types and endpoints.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import signal
from contextlib import contextmanager
from loguru import logger

from src.benchmarking.handlers.llm_handler import LLMHandler
from src.benchmarking.handlers.vlm_handler import VLMHandler
from src.benchmarking.models.metric_types import ModelType


class TimeoutException(Exception):
    """Raised when inference times out."""
    pass


class EndpointExecutor:
    """
    Coordinates model inference execution across different handlers.

    Responsibilities:
    - Route requests to appropriate handler (LLM/VLM)
    - Handle timeouts
    - Provide unified interface
    - Error handling and retry logic
    """

    def __init__(
        self,
        provider_bridge: Optional[Any] = None,
        llm_handler: Optional[LLMHandler] = None,
        vlm_handler: Optional[VLMHandler] = None
    ):
        """
        Initialize endpoint executor.

        Args:
            provider_bridge: Optional ProviderBridge for real model execution
            llm_handler: Optional LLMHandler instance
            vlm_handler: Optional VLMHandler instance
        """
        self.llm_handler = llm_handler or LLMHandler(provider_bridge=provider_bridge)
        self.vlm_handler = vlm_handler or VLMHandler(provider_bridge=provider_bridge)

    def execute(
        self,
        model_type: ModelType,
        model_id: str,
        endpoint: str,
        input_data: Union[str, Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = 600  # 10 minutes - generous for large models on CPU/MPS
    ) -> Dict[str, Any]:
        """
        Execute inference through appropriate handler.

        Args:
            model_type: Type of model (LLM or VLM)
            model_id: Model identifier
            endpoint: Endpoint name
            input_data: Input data (prompt string for LLM, dict with prompt+image for VLM)
            parameters: Inference parameters
            timeout: Timeout in seconds

        Returns:
            Dictionary with output, metadata, and execution info

        Raises:
            TimeoutException: If inference exceeds timeout
            ValueError: If parameters are invalid
            RuntimeError: If inference fails
        """
        try:
            logger.debug(
                f"Executing inference: type={model_type.value}, "
                f"model={model_id}, endpoint={endpoint}"
            )

            with self._timeout_context(timeout):
                if model_type == ModelType.LLM:
                    result = self._execute_llm(
                        model_id, input_data, parameters, timeout
                    )
                elif model_type == ModelType.VLM:
                    result = self._execute_vlm(
                        model_id, input_data, parameters, timeout, endpoint
                    )
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

            logger.debug(f"Inference complete for {model_id}")
            return result

        except TimeoutException:
            logger.error(f"Inference timeout after {timeout}s for {model_id}")
            raise
        except Exception as e:
            logger.error(f"Inference execution failed for {model_id}: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    def _execute_llm(
        self,
        model_id: str,
        input_data: Union[str, Dict],
        parameters: Optional[Dict[str, Any]],
        timeout: int
    ) -> Dict[str, Any]:
        """
        Execute LLM inference.

        Args:
            model_id: Model identifier
            input_data: Prompt string or dict with 'prompt' key
            parameters: Inference parameters
            timeout: Timeout in seconds

        Returns:
            Result dictionary
        """
        # Extract prompt
        if isinstance(input_data, str):
            prompt = input_data
        elif isinstance(input_data, dict) and "prompt" in input_data:
            prompt = input_data["prompt"]
        else:
            raise ValueError(f"Invalid LLM input data: {type(input_data)}")

        # Execute inference
        output, token_count, metadata = self.llm_handler.execute(
            model_id=model_id,
            prompt=prompt,
            parameters=parameters,
            timeout=timeout
        )

        return {
            "output": output,
            "token_count": token_count,
            "metadata": metadata,
            "success": True
        }

    def _execute_vlm(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]],
        timeout: int,
        endpoint: str = None
    ) -> Dict[str, Any]:
        """
        Execute VLM inference.

        Args:
            model_id: Model identifier
            input_data: Dict with 'prompt' and 'image_path' keys
            parameters: Inference parameters
            timeout: Timeout in seconds
            endpoint: VLM endpoint name (e.g., 'vision/qa', 'vision/caption')

        Returns:
            Result dictionary
        """
        if not isinstance(input_data, dict):
            raise ValueError("VLM input data must be a dictionary")

        if "prompt" not in input_data:
            raise ValueError("VLM input data missing 'prompt' key")

        if "image_path" not in input_data and "image" not in input_data:
            raise ValueError("VLM input data missing 'image_path' or 'image' key")

        prompt = input_data["prompt"]
        image_path = input_data.get("image_path") or input_data.get("image")

        # Execute inference
        output, metadata = self.vlm_handler.execute(
            model_id=model_id,
            prompt=prompt,
            image_path=image_path,
            parameters=parameters,
            timeout=timeout,
            endpoint=endpoint
        )

        return {
            "output": output,
            "metadata": metadata,
            "success": True
        }

    @contextmanager
    def _timeout_context(self, timeout: int):
        """
        Context manager for enforcing timeout.

        Args:
            timeout: Timeout in seconds

        Yields:
            None
        """
        def timeout_handler(signum, frame):
            raise TimeoutException(f"Operation timed out after {timeout} seconds")

        # Set up signal handler for timeout
        # Note: This only works on Unix-like systems
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except AttributeError:
            # Windows doesn't have SIGALRM, just yield without timeout enforcement
            # Production implementation would use threading.Timer or asyncio
            logger.warning("Timeout enforcement not available on this platform")
            yield

    def validate_input(
        self,
        model_type: ModelType,
        input_data: Union[str, Dict[str, Any]]
    ) -> bool:
        """
        Validate input data for a model type.

        Args:
            model_type: Model type
            input_data: Input data to validate

        Returns:
            True if valid, False otherwise
        """
        if model_type == ModelType.LLM:
            if isinstance(input_data, str):
                return len(input_data) > 0
            elif isinstance(input_data, dict):
                return "prompt" in input_data and len(input_data["prompt"]) > 0
            return False

        elif model_type == ModelType.VLM:
            if not isinstance(input_data, dict):
                return False
            if "prompt" not in input_data:
                return False
            if "image_path" not in input_data and "image" not in input_data:
                return False
            return True

        return False

    def supports_streaming(self, model_type: ModelType) -> bool:
        """
        Check if streaming is supported for model type.

        Args:
            model_type: Model type

        Returns:
            True if streaming is supported
        """
        if model_type == ModelType.LLM:
            return self.llm_handler.supports_streaming()
        elif model_type == ModelType.VLM:
            return False  # VLM typically doesn't support streaming
        return False

    def estimate_inference_time(
        self,
        model_type: ModelType,
        input_data: Union[str, Dict],
        parameters: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Estimate inference time.

        Args:
            model_type: Model type
            input_data: Input data
            parameters: Inference parameters

        Returns:
            Estimated time in seconds
        """
        params = parameters or {}
        max_tokens = params.get("max_tokens", 512)

        if model_type == ModelType.LLM:
            if isinstance(input_data, str):
                prompt_len = len(input_data)
            else:
                prompt_len = len(input_data.get("prompt", ""))

            return self.llm_handler.estimate_inference_time(
                prompt_length=prompt_len,
                max_tokens=max_tokens
            )

        elif model_type == ModelType.VLM:
            # Default dimensions if not specified
            width = 1024
            height = 1024
            prompt_len = len(input_data.get("prompt", ""))

            if "image_path" in input_data:
                try:
                    image = self.vlm_handler.load_image(input_data["image_path"])
                    width = image.width
                    height = image.height
                except Exception:
                    pass  # Use defaults

            return self.vlm_handler.estimate_inference_time(
                image_width=width,
                image_height=height,
                prompt_length=prompt_len,
                max_tokens=max_tokens
            )

        return 5.0  # Default estimate

    def __repr__(self) -> str:
        """String representation."""
        return f"EndpointExecutor(llm={self.llm_handler}, vlm={self.vlm_handler})"
