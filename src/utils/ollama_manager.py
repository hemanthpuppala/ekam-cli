"""Ollama process management utilities."""

import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger


def is_ollama_running(host: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
    """Check if Ollama server is running and accessible.

    Args:
        host: Ollama server URL
        timeout: Connection timeout in seconds

    Returns:
        True if Ollama is accessible
    """
    try:
        response = httpx.get(f"{host}/api/tags", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def start_ollama_background() -> bool:
    """Start Ollama server in the background.

    Returns:
        True if Ollama started successfully or was already running
    """
    # Check if already running
    if is_ollama_running():
        logger.info("Ollama is already running")
        return True

    logger.info("Starting Ollama server...")

    try:
        # Try to start Ollama in background
        # On macOS/Linux, Ollama runs as a service or background process
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Detach from parent process
        )

        # Wait for Ollama to start (up to 10 seconds)
        for i in range(20):
            time.sleep(0.5)
            if is_ollama_running():
                logger.info("Ollama started successfully")
                return True

        logger.error("Ollama did not start within 10 seconds")
        return False

    except FileNotFoundError:
        logger.error("Ollama executable not found. Please install Ollama from https://ollama.ai")
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False


def ensure_ollama_running(host: str = "http://localhost:11434") -> bool:
    """Ensure Ollama server is running, start it if needed.

    Args:
        host: Ollama server URL

    Returns:
        True if Ollama is running and accessible
    """
    if is_ollama_running(host):
        return True

    logger.info("Ollama not running, attempting to start...")
    return start_ollama_background()


def get_ollama_version() -> Optional[str]:
    """Get Ollama version if available.

    Returns:
        Version string or None if not available
    """
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None
