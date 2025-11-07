"""Loguru logging configuration for production use."""

import os
import sys
import warnings
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: Path = Path("logs"), console_level: str = "ERROR") -> None:
    """Configure loguru logging with console and file handlers.

    Console shows only ERROR and CRITICAL (clean TUI, no log spam).
    File logs everything at DEBUG level for troubleshooting.
    
    Also suppresses all library output (transformers, torch, etc.) to keep TUI clean.

    Args:
        log_dir: Directory for log files (default: logs/)
        console_level: Logging level for console (default: ERROR - only errors shown)
    """
    # ========================================
    # SUPPRESS ALL THIRD-PARTY LIBRARY OUTPUT
    # ========================================
    
    # 1. Suppress Python warnings (deprecation, etc.)
    warnings.filterwarnings("ignore")
    
    # 2. Suppress HuggingFace transformers info messages (but KEEP progress bars for downloads)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Only show errors
    # NOTE: Do NOT set HF_HUB_DISABLE_PROGRESS_BARS - we want to see download progress!
    # os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # <-- Commented out
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizer warnings
    
    # 3. Suppress HuggingFace datasets/accelerate output
    os.environ["DATASETS_VERBOSITY"] = "error"
    os.environ["ACCELERATE_VERBOSITY"] = "error"
    
    # 4. Suppress TensorFlow messages (if used by any dependency)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 3 = ERROR only
    
    # 5. Suppress PyTorch CUDA initialization messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # 6. Suppress Hugging Face Hub symlink warnings
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # 7. Suppress bitsandbytes compilation messages
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    
    # 8. Configure Python's logging module (used by transformers, torch, etc.)
    import logging
    
    # Set root logger to WARNING (suppresses INFO/DEBUG from all libraries)
    logging.getLogger().setLevel(logging.WARNING)
    
    # Suppress specific noisy loggers
    noisy_loggers = [
        "transformers",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
        "transformers.tokenization_utils_base",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "accelerate",
        "torch",
        "torch.nn",
        "torch.cuda",
        "bitsandbytes",
        "urllib3",
        "filelock",
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
        logging.getLogger(logger_name).propagate = False
    
    # ========================================
    # CONFIGURE LOGURU (OUR LOGGING)
    # ========================================
    
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
        log_dir / "ekam_cli_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    # Log initialization (goes to file only, not console)
    logger.info("Logging initialized - console level: {}, file level: DEBUG", console_level)
    logger.debug("Third-party library output suppressed for clean TUI")
