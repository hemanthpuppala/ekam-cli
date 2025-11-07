"""Inference service with timing and result tracking."""

import time
from typing import Any, Optional

from loguru import logger
from PIL import Image

from ..models.endpoints import EndpointType
from ..models.inference import InferenceInput, InferenceOutput, InferenceResult
from ..models.model import ModelInfo
from ..providers.base import BaseProvider
from ..utils.image import preprocess_image


class InferenceService:
    """Service for running inference with timing and result tracking."""

    def __init__(self, provider: BaseProvider, model_info: ModelInfo, device: str):
        """Initialize inference service.

        Args:
            provider: Provider instance to use for inference
            model_info: Loaded model metadata
            device: Device where model is loaded (cpu, cuda, mps)
        """
        self.provider = provider
        self.model_info = model_info
        self.device = device
        self.model_handle = model_info.model_id  # For Ollama, handle is model_id

    def run_inference(self, inference_input: InferenceInput) -> InferenceResult:
        """Run inference with timing and return complete result.

        Args:
            inference_input: Input parameters for inference

        Returns:
            InferenceResult with output and timing

        Raises:
            ValueError: If endpoint not supported or required inputs missing
        """
        logger.info(
            f"Running {inference_input.endpoint} inference with "
            f"{self.model_info.name} on {self.device}"
        )

        # Validate inputs
        self._validate_input(inference_input)

        # Preprocess image if needed
        image = None
        if inference_input.image_path:
            image = preprocess_image(inference_input.image_path, max_dim=1920)
            logger.debug(f"Loaded image: {inference_input.image_path}")

        # Time the inference
        start_time = time.perf_counter()

        try:
            output = self._dispatch_endpoint(inference_input, image)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

        logger.info(f"Inference completed in {inference_time_ms:.2f}ms")

        # Build result
        result = InferenceResult(
            model_id=self.model_info.model_id,
            provider=self.model_info.provider,
            device=self.device,
            input=inference_input,
            output=output,
            inference_time_ms=inference_time_ms,
        )

        return result

    def _validate_input(self, inference_input: InferenceInput) -> None:
        """Validate inference input based on endpoint type.

        Args:
            inference_input: Input to validate

        Raises:
            ValueError: If required inputs are missing
        """
        endpoint = inference_input.endpoint

        # VLM endpoints require images
        if endpoint in [EndpointType.QA, EndpointType.CAPTION, EndpointType.DETECT, EndpointType.POINT]:
            if not inference_input.image_path:
                raise ValueError(f"{endpoint} endpoint requires image_path")

        # QA requires question
        if endpoint == EndpointType.QA:
            if not inference_input.prompt:
                raise ValueError("QA endpoint requires prompt (question)")

        # Detect/Point require object name
        if endpoint in [EndpointType.DETECT, EndpointType.POINT]:
            if not inference_input.object_name:
                raise ValueError(f"{endpoint} endpoint requires object_name")

        # Text endpoint requires prompt
        if endpoint == EndpointType.TEXT:
            if not inference_input.prompt:
                raise ValueError("Text endpoint requires prompt")

    def _dispatch_endpoint(
        self, inference_input: InferenceInput, image: Optional[Image.Image]
    ) -> InferenceOutput:
        """Dispatch to appropriate provider endpoint method.

        Args:
            inference_input: Input parameters
            image: Preprocessed image (None for text endpoint)

        Returns:
            InferenceOutput with results

        Raises:
            ValueError: If endpoint not supported
        """
        endpoint = inference_input.endpoint

        if endpoint == EndpointType.QA:
            text_response = self.provider.run_qa(
                self.model_handle, image, inference_input.prompt
            )
            return InferenceOutput(text_response=text_response)

        elif endpoint == EndpointType.CAPTION:
            text_response = self.provider.run_caption(
                self.model_handle, image, inference_input.detail_level
            )
            return InferenceOutput(text_response=text_response)

        elif endpoint == EndpointType.DETECT:
            detections = self.provider.run_detect(
                self.model_handle, image, inference_input.object_name
            )
            return InferenceOutput(detections=detections)

        elif endpoint == EndpointType.POINT:
            coordinates = self.provider.run_point(
                self.model_handle, image, inference_input.object_name
            )
            return InferenceOutput(coordinates=coordinates)

        elif endpoint == EndpointType.TEXT:
            text_response = self.provider.run_text(
                self.model_handle, inference_input.prompt
            )
            return InferenceOutput(text_response=text_response)

        else:
            raise ValueError(f"Unsupported endpoint: {endpoint}")
