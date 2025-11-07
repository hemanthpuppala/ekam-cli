"""Format converters for quantization pipeline.

This module provides converters to transform models between different formats,
enabling Ollama models to be quantized using any quantization technique.

Architecture:
    - BaseConverter: Abstract interface for all converters
    - Specific converters: GGUFDequantizer, GGUFToHFConverter, etc.
    - ConverterRegistry: Central registry for format conversion

Example:
    >>> from src.quantization.converters import get_converter
    >>> converter = get_converter("gguf", "huggingface")
    >>> converter.convert(source_path, output_path)
"""

from .base import BaseConverter
from .registry import ConverterRegistry, get_converter

__all__ = [
    "BaseConverter",
    "ConverterRegistry",
    "get_converter",
]
