"""AWQ quantization implementation (Phase 2).

AWQ (Activation-aware Weight Quantization) provides better quality than GPTQ
by preserving salient weights based on activation patterns.

References:
    - https://github.com/mit-han-lab/llm-awq
    - https://arxiv.org/abs/2306.00978
"""

from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ...models.model import ModelInfo
from ...models.provider import ProviderType
from ..models import QuantizationTask, QuantizationType
from .base import BaseQuantizer


class AWQQuantizer(BaseQuantizer):
    """AWQ quantization implementation.

    Phase 2 implementation will support:
    - AWQ 4-bit (int4)
    - Activation-aware weight protection
    - AutoAWQ library integration
    - GPU-optimized inference
    """

    def check_availability(self) -> tuple[bool, str]:
        """Check if AWQ tools are available.

        Returns:
            (available, message)
        """
        # Phase 2: Check for autoawq library
        try:
            import awq
            return True, "AutoAWQ available"
        except ImportError:
            return False, "AutoAWQ not installed. Run: pip install autoawq"

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported quantization types.

        Returns:
            List of supported types
        """
        return [
            QuantizationType.AWQ_4BIT,
        ]

    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for quantization.

        AWQ works with HuggingFace models.

        Args:
            model_info: Model to quantize

        Returns:
            Path to model directory, or None if not compatible
        """
        # AWQ works with HuggingFace models
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

        # AWQ 4-bit is ~30% of original size
        return original_size * 0.3

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Execute AWQ quantization.

        Args:
            task: Quantization task
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful
        """
        logger.warning("AWQ quantization not yet implemented (Phase 2)")

        # Phase 2 implementation will:
        # 1. Load model with transformers
        # 2. Prepare calibration dataset
        # 3. Compute activation statistics
        # 4. Run AWQ quantization with autoawq
        # 5. Save quantized model
        # 6. Update progress via callback

        from ..models import TaskStatus
        task.status = TaskStatus.FAILED
        task.error = "AWQ quantization not yet implemented (Phase 2)"
        return False
