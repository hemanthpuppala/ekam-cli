"""Quantization techniques."""

from .base import BaseQuantizer
from .generic import GenericQuantizer
from .gguf import GGUFQuantizer
from .gguf_converter import GGUFConverter

# Phase 2 quantizers
from .gptq import GPTQQuantizer
from .awq import AWQQuantizer
from .bnb import BitsAndBytesQuantizer

# 2025 NEW: Platform-specific quantizers
# MLX: Apple Silicon (Mac M1/M2/M3/M4)
# OpenVINO: Intel CPU/iGPU
try:
    from .mlx import MLXQuantizer
except ImportError:
    MLXQuantizer = None  # Only available on macOS

try:
    from .openvino import OpenVINOQuantizer
except ImportError:
    OpenVINOQuantizer = None  # Only available if OpenVINO installed

__all__ = [
    "BaseQuantizer",
    "GenericQuantizer",
    "GGUFQuantizer",
    "GGUFConverter",
    # Phase 2
    "GPTQQuantizer",
    "AWQQuantizer",
    "BitsAndBytesQuantizer",
    # 2025 Platform-specific
    "MLXQuantizer",
    "OpenVINOQuantizer",
]
