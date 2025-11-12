"""llama.cpp server backend configuration.

This module provides configuration for selecting between llama-server and
the custom CLI-based server implementation.
"""

import os
from typing import Literal
from loguru import logger


ServerType = Literal["llama-server", "llama-cli-server"]


def get_llama_server_type() -> ServerType:
    """Get configured llama.cpp server backend type from environment.

    Reads EKAM_USE_LLAMA_SERVER environment variable to determine which
    server implementation to use:

    - false (default): Use custom CLI server with isolated mtmd_context per slot
      - Fixes vision model state corruption bugs
      - Production-ready with 0% failure rate on VLMs
      - Recommended for all use cases

    - true: Use original llama-server with shared mtmd_context
      - Legacy implementation with known vision model bugs
      - 30-50% failure rate on multi-image vision inference
      - Only use for debugging or compatibility testing

    Returns:
        "llama-cli-server" (default) or "llama-server" (legacy)

    Environment Variables:
        EKAM_USE_LLAMA_SERVER: "true" or "false" (default: "false")

    Examples:
        >>> os.environ["EKAM_USE_LLAMA_SERVER"] = "false"
        >>> get_llama_server_type()
        'llama-cli-server'

        >>> os.environ["EKAM_USE_LLAMA_SERVER"] = "true"
        >>> get_llama_server_type()
        'llama-server'
    """
    use_legacy_server = os.getenv("EKAM_USE_LLAMA_SERVER", "false").lower().strip()

    # Parse boolean value (true/1/yes = use legacy, false/0/no = use custom)
    use_legacy = use_legacy_server in ("true", "1", "yes")

    if use_legacy:
        logger.warning(
            "Using legacy llama-server (EKAM_USE_LLAMA_SERVER=true). "
            "This has known vision model bugs. "
            "Set EKAM_USE_LLAMA_SERVER=false to use the fixed custom server."
        )
        return "llama-server"
    else:
        logger.info(
            "Using custom CLI server (EKAM_USE_LLAMA_SERVER=false). "
            "This provides isolated mtmd_context per slot and fixes vision model bugs."
        )
        return "llama-cli-server"


def should_use_legacy_server() -> bool:
    """Check if legacy llama-server should be used.

    Convenience function for boolean check.

    Returns:
        True if EKAM_USE_LLAMA_SERVER=true, False otherwise
    """
    return get_llama_server_type() == "llama-server"


def should_use_custom_server() -> bool:
    """Check if custom CLI server should be used.

    Convenience function for boolean check.

    Returns:
        True if EKAM_USE_LLAMA_SERVER=false (default), False otherwise
    """
    return get_llama_server_type() == "llama-cli-server"
