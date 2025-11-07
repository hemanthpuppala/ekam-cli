"""Provider implementations."""

from .base import BaseProvider
from .ollama import OllamaProvider
from .huggingface import HuggingFaceProvider
from .gguf import GGUFProvider
from .quantized import QuantizedProvider

# 2025 NEW PROVIDERS
# MLX: Apple Silicon optimized (Mac M1/M2/M3/M4)
# OpenVINO: Intel CPU/iGPU optimized
try:
    from .mlx import MLXProvider
except ImportError:
    MLXProvider = None  # Only available on macOS

try:
    from .openvino import OpenVINOProvider
except ImportError:
    OpenVINOProvider = None  # Only available if OpenVINO installed

__all__ = [
    "BaseProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "GGUFProvider",
    "QuantizedProvider",
    "MLXProvider",
    "OpenVINOProvider",
]
