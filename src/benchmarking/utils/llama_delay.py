"""
Llama.cpp Server Post-Inference Delay Utility

Provides configurable delay after llama-server inference to allow proper state cleanup
and reduce race conditions in mtmd (multimodal) encoder.

This helps work around the intermittent mtmd_encode_chunk() failure bug in llama.cpp.
"""

import os
import time
from typing import Optional
from ...core.logging_setup import logger


def get_llama_server_delay() -> float:
    """
    Get the configured post-inference delay for llama-server in seconds.

    Returns:
        Delay in seconds (default: 0.5s)

    Environment Variables:
        LLAMA_SERVER_POST_INFERENCE_DELAY_MS: Delay in milliseconds (default: 500)
    """
    try:
        delay_ms = int(os.getenv("LLAMA_SERVER_POST_INFERENCE_DELAY_MS", "500"))
        return delay_ms / 1000.0
    except (ValueError, TypeError):
        logger.warning("Invalid LLAMA_SERVER_POST_INFERENCE_DELAY_MS, using default 500ms")
        return 0.5


def apply_llama_server_delay(provider_type: Optional[str] = None) -> None:
    """
    Apply post-inference delay for llama-server to allow state cleanup.

    This delay happens AFTER inference completes and latency is recorded,
    so it does not affect benchmark timing measurements.

    Args:
        provider_type: Optional provider type hint (e.g., "gguf", "quantized")
                      Currently applies to all llama-server providers.
    """
    # Check if we should apply delay (currently applies to all llama-server providers)
    if provider_type and not should_apply_delay(provider_type):
        return

    delay_seconds = get_llama_server_delay()

    if delay_seconds > 0:
        logger.debug(f"Applying llama-server post-inference delay: {delay_seconds:.3f}s")
        time.sleep(delay_seconds)


def should_apply_delay(provider_type: str) -> bool:
    """
    Check if delay should be applied for the given provider type.

    Args:
        provider_type: Provider type (e.g., "gguf", "quantized", "huggingface", "ollama")

    Returns:
        True if delay should be applied (llama-server based providers)
    """
    # Apply delay for providers that use llama-server
    llama_server_providers = ["gguf", "quantized"]
    return provider_type.lower() in llama_server_providers
