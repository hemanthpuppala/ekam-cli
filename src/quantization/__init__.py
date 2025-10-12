"""Model quantization module for reducing model size and improving inference speed."""

from .models import (
    QuantizationModule,
    QuantizationRecommendation,
    QuantizationTask,
    QuantizationType,
    TaskStatus,
)
from .manager import QuantizationManager
from .core import BackgroundJobManager, get_quantization_recommendations, check_can_quantize_multiple
from .techniques import BaseQuantizer, GGUFQuantizer

__all__ = [
    # Data models
    "QuantizationType",
    "QuantizationModule",
    "QuantizationTask",
    "QuantizationRecommendation",
    "TaskStatus",
    # Core
    "QuantizationManager",
    "BackgroundJobManager",
    "get_quantization_recommendations",
    "check_can_quantize_multiple",
    # Techniques
    "BaseQuantizer",
    "GGUFQuantizer",
]
