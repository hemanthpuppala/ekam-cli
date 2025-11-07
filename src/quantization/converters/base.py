"""Base converter interface for model format conversions.

All converters must inherit from BaseConverter and implement the required methods.
This ensures a consistent interface across all format conversion operations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable

from loguru import logger


class BaseConverter(ABC):
    """Abstract base class for model format converters.

    Each converter handles conversion from one format to another.
    Examples:
        - GGUFDequantizer: GGUF Q4 → GGUF FP16
        - GGUFToHFConverter: GGUF → HuggingFace Safetensors
        - HFToGGUFConverter: HuggingFace → GGUF (already exists)
    """

    @abstractmethod
    def can_convert(self, source_format: str, target_format: str) -> bool:
        """Check if this converter can handle the conversion.

        Args:
            source_format: Source format (e.g., "gguf", "huggingface")
            target_format: Target format (e.g., "fp16", "safetensors")

        Returns:
            True if this converter supports the conversion
        """
        pass

    @abstractmethod
    def check_availability(self) -> tuple[bool, str]:
        """Check if converter dependencies are available.

        Returns:
            (available, message) tuple where:
                - available: True if converter can be used
                - message: Description or error message
        """
        pass

    @abstractmethod
    def convert(
        self,
        source_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Convert model from source format to target format.

        Args:
            source_path: Path to source model file or directory
            output_path: Path for converted model output
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if conversion succeeded
        """
        pass

    @abstractmethod
    def estimate_output_size(self, source_path: Path) -> float:
        """Estimate output size in GB.

        Args:
            source_path: Path to source model

        Returns:
            Estimated output size in gigabytes
        """
        pass

    def validate_paths(self, source_path: Path, output_path: Path) -> tuple[bool, str]:
        """Validate source and output paths.

        Args:
            source_path: Source model path
            output_path: Output path

        Returns:
            (valid, message) tuple
        """
        if not source_path.exists():
            return False, f"Source path does not exist: {source_path}"

        if output_path.exists():
            logger.warning(f"Output path already exists: {output_path}")

        return True, "Paths valid"

    def get_conversion_description(self) -> str:
        """Get human-readable description of this converter.

        Returns:
            Description string
        """
        return self.__class__.__name__
