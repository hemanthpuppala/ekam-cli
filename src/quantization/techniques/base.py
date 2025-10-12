"""Base quantizer interface for all quantization techniques."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from ..models import QuantizationTask, QuantizationType
from ...models.model import ModelInfo


class BaseQuantizer(ABC):
    """Abstract base class for quantization implementations."""

    @abstractmethod
    def check_availability(self) -> tuple[bool, str]:
        """Check if this quantization technique is available.

        Returns:
            (available, message) tuple where:
            - available: True if technique can be used
            - message: Description of availability status
        """
        pass

    @abstractmethod
    def get_supported_types(self) -> list[QuantizationType]:
        """Get list of supported quantization types.

        Returns:
            List of QuantizationType enums this quantizer supports
        """
        pass

    @abstractmethod
    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for quantization.

        Args:
            model_info: Model to quantize

        Returns:
            Path to model file, or None if not available/compatible
        """
        pass

    @abstractmethod
    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Perform quantization.

        Args:
            task: Quantization task with all parameters
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
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
        pass

    def validate_compatibility(self, model_info: ModelInfo) -> tuple[bool, str]:
        """Validate if model is compatible with this quantizer.

        Args:
            model_info: Model to check

        Returns:
            (compatible, reason) tuple
        """
        source_path = self.get_source_model_path(model_info)
        if source_path is None:
            return False, "Model not available or incompatible format"
        if not source_path.exists():
            return False, f"Model file not found: {source_path}"
        return True, "Model is compatible"
