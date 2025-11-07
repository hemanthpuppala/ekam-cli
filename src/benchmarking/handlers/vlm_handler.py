"""
VLM inference handler for benchmarking.

Handles vision-language model inference with image loading and multimodal input.
"""

from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
from PIL import Image
import io
import json
from loguru import logger
from src.utils.image import save_annotated_image
from src.utils.vlm_response_parser import parse_detection_response, parse_point_response


class VLMHandler:
    """
    Handler for VLM (Vision-Language Model) inference execution.

    Responsibilities:
    - Load and preprocess images
    - Execute multimodal inference (image + text)
    - Validate outputs
    - Handle image formats and conversions
    """

    def __init__(self, provider_bridge: Optional[Any] = None, result_dir: Optional[Path] = None):
        """
        Initialize VLM handler.

        Args:
            provider_bridge: Optional ProviderBridge instance for real model execution
            result_dir: Optional directory for saving visualization outputs
        """
        self.provider_bridge = provider_bridge
        self.result_dir = Path(result_dir) if result_dir else None
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    def set_result_dir(self, result_dir: Union[str, Path]) -> None:
        """
        Set the result directory for saving visualizations.

        Args:
            result_dir: Path to results directory
        """
        self.result_dir = Path(result_dir) if result_dir else None
        logger.debug(f"VLMHandler result_dir set to: {self.result_dir}")

    def execute(
        self,
        model_id: str,
        prompt: str,
        image_path: Union[str, Path],
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = 600,  # 10 minutes for large models
        endpoint: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute VLM inference.

        Args:
            model_id: Model identifier
            prompt: Text prompt/question about the image
            image_path: Path to image file
            parameters: Inference parameters
            timeout: Timeout in seconds
            endpoint: VLM endpoint name (e.g., 'vision/qa', 'vision/caption')

        Returns:
            Tuple of (output_text, metadata)

        Raises:
            FileNotFoundError: If image not found
            ValueError: If image format not supported
            TimeoutError: If inference exceeds timeout
            RuntimeError: If inference fails
        """
        params = parameters or {}

        try:
            logger.debug(f"Executing VLM inference: model={model_id}, image={image_path}")

            # Load image
            image_data = self.load_image(image_path)

            # Prepare parameters
            temperature = params.get("temperature", 0.5)
            max_tokens = params.get("max_tokens", 512)

            # Execute inference via provider bridge
            if self.provider_bridge:
                output, bridge_metadata = self.provider_bridge.execute_vision(
                    model_id=model_id,
                    image_path=str(image_path),
                    prompt=prompt,
                    parameters=params,
                    endpoint=endpoint,
                    timeout=timeout
                )
                if output is None:
                    raise RuntimeError("Provider bridge returned None output")
            else:
                # Mock output for testing (when no bridge provided)
                logger.warning("No provider_bridge configured, using mock data")
                output = f"Mock VLM response: Analyzing image with prompt '{prompt[:30]}...'"

            # Validate output (log warning but don't fail - some models may return short responses)
            if not self.validate_output(output):
                logger.warning(
                    f"Output validation warning for {model_id}: "
                    f"output={repr(output[:100] if output else output)}, "
                    f"length={len(output) if output else 0}"
                )
                # Don't fail - allow benchmark to continue with the output
                # This handles models that return minimal responses

            # Prepare metadata
            image_info = self._get_image_info(image_data)
            metadata = {
                "image_width": image_info["width"],
                "image_height": image_info["height"],
                "image_format": image_info["format"],
                "image_size_bytes": image_info["size"],
                "prompt_length": len(prompt),
                "output_length": len(output),
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Generate visualizations for detection/point endpoints
            output_image_path = self._generate_visualization(
                endpoint=endpoint,
                image_path=image_path,
                output=output,
                prompt=prompt
            )

            if output_image_path:
                metadata["output_image_path"] = str(output_image_path)

            logger.debug(f"VLM inference complete: output_len={len(output)}")
            return output, metadata

        except FileNotFoundError:
            logger.error(f"Image not found: {image_path}")
            raise
        except Exception as e:
            logger.error(f"VLM inference failed for {model_id}: {str(e)}")
            raise RuntimeError(f"VLM inference failed: {str(e)}") from e

    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load and validate image from path.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image object

        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If format not supported
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported: {self.supported_formats}"
            )

        try:
            image = Image.open(path)
            # Convert to RGB if needed
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}") from e

    def _execute_with_provider(
        self,
        model_id: str,
        prompt: str,
        image: Image.Image,
        parameters: Dict[str, Any],
        timeout: int
    ) -> str:
        """
        Execute inference using the model provider.

        Args:
            model_id: Model identifier
            prompt: Text prompt
            image: PIL Image object
            parameters: Inference parameters
            timeout: Timeout in seconds

        Returns:
            Generated text output
        """
        # This would integrate with actual VLM providers
        # For now, it's a placeholder
        if hasattr(self.model_provider, "generate_with_image"):
            return self.model_provider.generate_with_image(
                model=model_id,
                prompt=prompt,
                image=image,
                **parameters
            )
        else:
            raise NotImplementedError("Model provider does not support VLM inference")

    def validate_output(self, output: str) -> bool:
        """
        Validate generated output.

        Args:
            output: Generated text to validate

        Returns:
            True if valid, False otherwise
        """
        if not output:
            return False

        if not isinstance(output, str):
            return False

        # Check for reasonable length
        if len(output) < 1 or len(output) > 100000:
            return False

        return True

    def _generate_visualization(
        self,
        endpoint: Optional[str],
        image_path: Union[str, Path],
        output: str,
        prompt: str
    ) -> Optional[Path]:
        """
        Generate visualization for detection/point outputs.

        Args:
            endpoint: VLM endpoint name (e.g., 'vision/detect', 'vision/point')
            image_path: Path to input image
            output: Model output (JSON string)
            prompt: Input prompt/object name

        Returns:
            Path to saved visualization image, or None if not applicable

        """
        # Only visualize for detection and point endpoints
        if not endpoint or endpoint not in ["vision/detect", "vision/point"]:
            return None

        # Skip if no result_dir configured
        if not self.result_dir:
            logger.debug("No result_dir configured, skipping visualization")
            return None

        try:
            # Create detections subdirectory
            detections_dir = self.result_dir / "detections"
            detections_dir.mkdir(parents=True, exist_ok=True)

            # Generate output filename
            input_filename = Path(image_path).stem
            output_filename = f"{input_filename}_output.png"
            output_path = detections_dir / output_filename

            # Load original image
            image = Image.open(image_path)

            # Parse output and generate visualizations
            if endpoint == "vision/detect":
                # Parse detection response
                detections = parse_detection_response(output, prompt)

                if not detections:
                    logger.warning(f"No detections parsed from output for {image_path}")
                    # Still save image with "No detections" overlay
                    detections = []

                # Convert to format expected by save_annotated_image
                detection_dicts = [{
                    "bbox": det["bbox"],
                    "label": det.get("label", prompt)
                } for det in detections]

                # Save annotated image
                save_annotated_image(
                    image=image,
                    detections=detection_dicts,
                    output_path=output_path
                )

                logger.info(f"✓ Saved detection visualization: {output_path}")
                return output_path

            elif endpoint == "vision/point":
                # Parse point response
                point_data = parse_point_response(output, prompt)

                if not point_data or ("x" not in point_data and "y" not in point_data):
                    logger.warning(f"No valid point coordinates parsed from output for {image_path}")
                    # Still save with message
                    point_data = {}

                # Convert to format expected by save_annotated_image
                if "x" in point_data and "y" in point_data:
                    point_dict = [{
                        "coordinates": {
                            "x": point_data["x"],
                            "y": point_data["y"]
                        },
                        "label": prompt
                    }]
                else:
                    point_dict = []

                # Save annotated image
                save_annotated_image(
                    image=image,
                    detections=point_dict,
                    output_path=output_path
                )

                logger.info(f"✓ Saved point visualization: {output_path}")
                return output_path

        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            # Don't fail the benchmark - just log and continue
            return None

    def _get_image_info(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract image metadata.

        Args:
            image: PIL Image object

        Returns:
            Dictionary with image information
        """
        # Get file size in memory
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        size_bytes = buffer.tell()

        return {
            "width": image.width,
            "height": image.height,
            "format": image.format or "unknown",
            "mode": image.mode,
            "size": size_bytes
        }

    def resize_image(
        self,
        image: Image.Image,
        max_width: int = 1024,
        max_height: int = 1024
    ) -> Image.Image:
        """
        Resize image to fit within maximum dimensions while preserving aspect ratio.

        Args:
            image: PIL Image object
            max_width: Maximum width in pixels
            max_height: Maximum height in pixels

        Returns:
            Resized PIL Image object
        """
        if image.width <= max_width and image.height <= max_height:
            return image

        # Calculate scaling factor
        width_ratio = max_width / image.width
        height_ratio = max_height / image.height
        scale_factor = min(width_ratio, height_ratio)

        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def preprocess_image(
        self,
        image_path: Union[str, Path],
        resize: bool = True,
        max_size: Tuple[int, int] = (1024, 1024)
    ) -> Image.Image:
        """
        Load and preprocess image for inference.

        Args:
            image_path: Path to image
            resize: Whether to resize large images
            max_size: Maximum dimensions (width, height)

        Returns:
            Preprocessed PIL Image
        """
        image = self.load_image(image_path)

        if resize:
            image = self.resize_image(image, max_size[0], max_size[1])

        return image

    def estimate_inference_time(
        self,
        image_width: int,
        image_height: int,
        prompt_length: int,
        max_tokens: int
    ) -> float:
        """
        Estimate inference time based on image size and prompt.

        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            prompt_length: Prompt character count
            max_tokens: Maximum output tokens

        Returns:
            Estimated time in seconds
        """
        # Rough estimation based on image resolution and output length
        # This is a placeholder - actual estimation would be model-specific
        image_pixels = image_width * image_height
        base_time = 2.0  # Base processing time
        image_factor = (image_pixels / 1000000) * 0.5  # 0.5s per megapixel
        token_factor = (max_tokens / 50.0)  # ~50 tokens/sec

        return base_time + image_factor + token_factor

    def supports_batch_processing(self) -> bool:
        """
        Check if handler supports batch processing of multiple images.

        Returns:
            True if batch processing is supported
        """
        return False  # Not implemented yet

    def __repr__(self) -> str:
        """String representation."""
        has_bridge = self.provider_bridge is not None
        return f"VLMHandler(bridge={'configured' if has_bridge else 'none'})"
