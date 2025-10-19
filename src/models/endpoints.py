"""Enumerations for providers, endpoints, model types, and compatibility status."""

from enum import Enum


class ProviderType(str, Enum):
    """Provider category (all local per CL-001)."""

    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LM_STUDIO = "lm_studio"
    GGUF = "gguf"
    QUANTIZED = "quantized"  # Quantized models from results/quantizations/


class EndpointType(str, Enum):
    """Supported inference endpoints."""

    QA = "qa"  # Question answering (VLM only)
    CAPTION = "caption"  # Image captioning (VLM only)
    DETECT = "detect"  # Object detection (VLM only)
    POINT = "point"  # Object pointing (VLM only)
    TEXT = "text"  # Text generation (LLM and VLM)


class ModelType(str, Enum):
    """Model category based on capabilities."""

    VLM = "vlm"  # Vision-language model
    LLM = "llm"  # Text-only language model
    EMBEDDING = "embedding"  # Embedding model (not supported for inference)


class CompatibilityStatus(str, Enum):
    """Memory compatibility assessment per FR-014."""

    PERFECT_FIT = "perfect_fit"  # <70% of recommended size
    TIGHT_FIT = "tight_fit"  # 70-100% of recommended size
    TOO_LARGE = "too_large"  # >100% of recommended size
