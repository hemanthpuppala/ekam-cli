"""Generic quantization implementation (FP16, INT8, INT4).

Generic quantization using PyTorch and HuggingFace Transformers.
Supports FP16 (half precision), INT8, and INT4 quantization for
HuggingFace models.

References:
    - https://huggingface.co/docs/transformers/main_classes/quantization
    - https://pytorch.org/docs/stable/quantization.html
"""

from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ...models.model import ModelInfo
from ...models.provider import ProviderType
from ..models import QuantizationTask, QuantizationType, TaskStatus
from .base import BaseQuantizer


class GenericQuantizer(BaseQuantizer):
    """Generic quantization for FP16, INT8, and INT4.

    Uses PyTorch and HuggingFace Transformers for standard quantization.
    """

    def check_availability(self) -> tuple[bool, str]:
        """Check if PyTorch and Transformers are available.

        Returns:
            (available, message)
        """
        try:
            import torch
            import transformers
            return True, f"PyTorch {torch.__version__} and Transformers {transformers.__version__} available"
        except ImportError as e:
            missing = "torch" if "torch" in str(e) else "transformers"
            return False, f"{missing} not installed. Run: pip install {missing}"

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported quantization types.

        Returns:
            List of supported types
        """
        return [
            QuantizationType.FP16,
            QuantizationType.INT8,
            QuantizationType.INT4,
        ]

    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for quantization.

        Generic quantization works with HuggingFace models.

        Args:
            model_info: Model to quantize

        Returns:
            Path to model directory, or None if not compatible
        """
        # Generic quantization works with HuggingFace models
        if model_info.provider != ProviderType.HUGGINGFACE:
            return None

        # HuggingFace models use file_path as the model directory
        if model_info.file_path:
            return Path(model_info.file_path)

        return None

    def estimate_output_size(
        self, model_info: ModelInfo, quant_type: QuantizationType
    ) -> float:
        """Estimate output file size in GB.

        Args:
            model_info: Source model
            quant_type: Quantization type

        Returns:
            Estimated size in GB
        """
        original_size = model_info.size_gb

        # Size reduction factors
        size_factors = {
            QuantizationType.FP16: 0.5,  # 50% of original
            QuantizationType.INT8: 0.25,  # 25% of original
            QuantizationType.INT4: 0.125,  # 12.5% of original
        }

        factor = size_factors.get(quant_type, 0.5)
        return original_size * factor

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Execute generic quantization.

        Args:
            task: Quantization task
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            task.status = TaskStatus.RUNNING
            logger.info(f"Starting {task.quant_type.display_name} quantization")

            # Update progress: Loading model
            if progress_callback:
                progress_callback(10.0, None)
            task.progress = 10.0

            # Get model path from model_info
            model_path = task.model_info.file_path
            if not model_path or not Path(model_path).exists():
                raise ValueError(f"Model file not found: {model_path}")

            logger.info(f"Loading model from {model_path}")

            # Load model based on quantization type
            if task.quant_type == QuantizationType.FP16:
                # FP16: Load in half precision
                logger.info("Loading model in FP16 (half precision)")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

            elif task.quant_type == QuantizationType.INT8:
                # INT8: Use built-in 8-bit quantization
                logger.info("Loading model in INT8 quantization")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    load_in_8bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

            elif task.quant_type == QuantizationType.INT4:
                # INT4: Use 4-bit quantization (requires bitsandbytes)
                logger.info("Loading model in INT4 quantization")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    load_in_4bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

            else:
                raise ValueError(f"Unsupported quantization type: {task.quant_type}")

            # Update progress: Model loaded
            if progress_callback:
                progress_callback(50.0, None)
            task.progress = 50.0

            # Load tokenizer
            logger.info("Loading tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Update progress: Saving
            if progress_callback:
                progress_callback(75.0, None)
            task.progress = 75.0

            # Create output directory
            task.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save quantized model
            logger.info(f"Saving quantized model to {task.output_path}")
            model.save_pretrained(task.output_path)
            tokenizer.save_pretrained(task.output_path)

            # Update progress: Complete
            if progress_callback:
                progress_callback(100.0, 0)
            task.progress = 100.0
            task.status = TaskStatus.COMPLETED

            logger.info(f"Quantization completed successfully: {task.output_path}")
            return True

        except ImportError as e:
            error_msg = f"Missing dependency: {str(e)}"
            logger.error(error_msg)
            task.status = TaskStatus.FAILED
            task.error = error_msg
            return False

        except Exception as e:
            error_msg = f"Quantization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = error_msg
            return False
