"""BitsAndBytes quantization implementation (Phase 2).

BitsAndBytes provides HuggingFace-integrated 4-bit and 8-bit quantization
with easy integration into transformers models.

References:
    - https://github.com/TimDettmers/bitsandbytes
    - https://huggingface.co/docs/transformers/main_classes/quantization
"""

from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ...models.model import ModelInfo
from ...models.provider import ProviderType
from ..models import QuantizationTask, QuantizationType
from .base import BaseQuantizer


class BitsAndBytesQuantizer(BaseQuantizer):
    """BitsAndBytes quantization implementation.

    Phase 2 implementation will support:
    - BnB 4-bit (NF4, FP4)
    - BnB 8-bit
    - Double quantization
    - Direct HuggingFace integration
    """

    def check_availability(self) -> tuple[bool, str]:
        """Check if BitsAndBytes tools are available.

        Returns:
            (available, message)
        """
        # Phase 2: Check for bitsandbytes library
        try:
            import bitsandbytes as bnb
            return True, "BitsAndBytes available"
        except ImportError:
            return False, "BitsAndBytes not installed. Run: pip install bitsandbytes"

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported quantization types.

        Returns:
            List of supported types
        """
        return [
            QuantizationType.BNB_8BIT,
            QuantizationType.BNB_4BIT_NF4,
            QuantizationType.BNB_4BIT_FP4,
        ]

    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for quantization.

        BitsAndBytes works with HuggingFace models.

        Args:
            model_info: Model to quantize

        Returns:
            Path to model directory, or None if not compatible
        """
        # BitsAndBytes works with HuggingFace models
        if model_info.provider != ProviderType.HUGGINGFACE:
            return None

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

        size_factors = {
            QuantizationType.BNB_8BIT: 0.5,  # 50% of original
            QuantizationType.BNB_4BIT_NF4: 0.35,  # 35% of original
            QuantizationType.BNB_4BIT_FP4: 0.35,  # 35% of original
        }

        factor = size_factors.get(quant_type, 0.35)
        return original_size * factor

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Execute BitsAndBytes quantization.

        Args:
            task: Quantization task
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful
        """
        logger.warning("BitsAndBytes quantization not yet implemented (Phase 2)")

        # Phase 2 implementation will:
        # 1. Load model with transformers
        # 2. Configure BitsAndBytesConfig (nf4, fp4, etc.)
        # 3. Load model with quantization config
        # 4. Save quantized model
        # 5. Update progress via callback

        from ..models import TaskStatus
        task.status = TaskStatus.FAILED
        task.error = "BitsAndBytes quantization not yet implemented (Phase 2)"
        return False
