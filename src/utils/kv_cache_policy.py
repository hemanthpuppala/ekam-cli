"""KV Cache Policy - Smart device-based KV cache management.

This module determines when KV cache should be enabled based on:
1. Device type (Metal has fragmentation issues, others are safe)
2. Context (benchmarking vs inference pipeline)
"""

from loguru import logger


def should_use_kv_cache(device: str, is_benchmark: bool = False) -> bool:
    """Determine if KV cache should be enabled based on device and context.

    Args:
        device: Device type (cuda/mps/cpu/rocm)
        is_benchmark: Whether this is a benchmarking run

    Returns:
        True if KV cache should be enabled, False otherwise

    Policy:
        - Benchmarking: Always disabled (stateless for consistent measurements)
        - Metal (mps): Disabled (memory fragmentation issues)
        - CUDA/ROCm/CPU: Enabled (stable, faster with conversation history)
    """
    # Benchmarking is always stateless
    if is_benchmark:
        logger.debug("KV cache: DISABLED (benchmarking mode)")
        return False

    # Check device type
    device_lower = device.lower()

    # Metal/MPS has memory fragmentation issues with vision models
    if device_lower in ["mps", "metal"]:
        logger.debug("KV cache: DISABLED (Metal backend fragmentation workaround)")
        return False

    # CUDA, ROCm, and CPU are stable with KV cache
    if device_lower in ["cuda", "rocm", "cpu"]:
        logger.debug(f"KV cache: ENABLED ({device_lower} backend)")
        return True

    # Unknown device - default to disabled for safety
    logger.warning(f"Unknown device '{device}' - disabling KV cache for safety")
    return False


def get_kv_cache_status_message(device: str, is_benchmark: bool = False) -> str:
    """Get human-readable KV cache status for logging.

    Args:
        device: Device type
        is_benchmark: Whether this is a benchmarking run

    Returns:
        Status message string
    """
    enabled = should_use_kv_cache(device, is_benchmark)

    if is_benchmark:
        return "KV cache disabled (benchmarking mode - stateless measurements)"
    elif enabled:
        return f"KV cache enabled ({device} - faster with conversation history)"
    else:
        return f"KV cache disabled ({device} - workaround for memory fragmentation)"
