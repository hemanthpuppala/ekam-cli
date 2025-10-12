"""Loguru logging configuration for production use."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: Path = Path("logs"), console_level: str = "ERROR") -> None:
    """Configure loguru logging with console and file handlers.

    Console shows only ERROR and CRITICAL (clean TUI, no log spam).
    File logs everything at DEBUG level for troubleshooting.

    Args:
        log_dir: Directory for log files (default: logs/)
        console_level: Logging level for console (default: ERROR - only errors shown)
    """
    # Remove default handler
    logger.remove()

    # Console handler (ERROR only) - clean TUI, only show errors
    # User won't see INFO/DEBUG/WARNING logs in terminal
    logger.add(
        sys.stderr,
        format="<red>[ERROR]</red> {message}",
        level=console_level,
        colorize=True,
    )

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler (DEBUG+) with rotation - everything goes to log file
    logger.add(
        log_dir / "vlm_cli_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    # Log initialization (goes to file only, not console)
    logger.info("Logging initialized - console level: {}, file level: DEBUG", console_level)
