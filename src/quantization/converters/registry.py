"""Converter registry for managing all format converters.

Provides a central registry to find and use the appropriate converter
for any source→target format conversion.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple

from loguru import logger

from .base import BaseConverter


class ConverterRegistry:
    """Central registry for model format converters.

    Manages all available converters and provides lookup functionality
    to find the right converter for a given conversion task.
    """

    def __init__(self):
        """Initialize empty converter registry."""
        self._converters: list[BaseConverter] = []
        logger.debug("Initialized ConverterRegistry")

    def register(self, converter: BaseConverter) -> None:
        """Register a converter.

        Args:
            converter: Converter instance to register
        """
        self._converters.append(converter)
        logger.debug(f"Registered converter: {converter.get_conversion_description()}")

    def find_converter(
        self, source_format: str, target_format: str
    ) -> Optional[BaseConverter]:
        """Find a converter that can handle the conversion.

        Args:
            source_format: Source format (e.g., "gguf", "huggingface")
            target_format: Target format (e.g., "fp16", "safetensors")

        Returns:
            Converter instance if found, None otherwise
        """
        for converter in self._converters:
            if converter.can_convert(source_format, target_format):
                # Check if converter is available
                available, message = converter.check_availability()
                if available:
                    logger.debug(
                        f"Found converter for {source_format}→{target_format}: "
                        f"{converter.get_conversion_description()}"
                    )
                    return converter
                else:
                    logger.warning(
                        f"Converter {converter.get_conversion_description()} "
                        f"not available: {message}"
                    )

        logger.debug(f"No converter found for {source_format}→{target_format}")
        return None

    def get_available_conversions(self) -> Dict[Tuple[str, str], BaseConverter]:
        """Get all available conversions.

        Returns:
            Dictionary mapping (source_format, target_format) to converter
        """
        available = {}
        for converter in self._converters:
            is_available, _ = converter.check_availability()
            if is_available:
                # This is a simplified version - converters should expose
                # their supported format pairs explicitly
                desc = converter.get_conversion_description()
                available[desc] = converter

        return available


# Global registry instance
_global_registry = ConverterRegistry()


def register_converter(converter: BaseConverter) -> None:
    """Register a converter in the global registry.

    Args:
        converter: Converter instance
    """
    _global_registry.register(converter)


def get_converter(source_format: str, target_format: str) -> Optional[BaseConverter]:
    """Get a converter from the global registry.

    Args:
        source_format: Source format
        target_format: Target format

    Returns:
        Converter instance if found
    """
    return _global_registry.find_converter(source_format, target_format)


def get_registry() -> ConverterRegistry:
    """Get the global converter registry.

    Returns:
        Global ConverterRegistry instance
    """
    return _global_registry
