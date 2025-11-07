"""Output suppression utilities for clean TUI."""

import os
import sys
from contextlib import contextmanager
from typing import IO, Optional


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr.
    
    Use this when loading models or running operations that might
    print unwanted output to the terminal.
    
    Example:
        with suppress_stdout_stderr():
            model = load_model()  # No output to terminal
    """
    # Save original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Redirect to devnull
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        # Restore original stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@contextmanager
def suppress_transformers_output():
    """Context manager to suppress HuggingFace transformers INFO messages.
    
    Suppresses info/warning messages but KEEPS download progress bars.
    More targeted than suppress_stdout_stderr.
    
    Example:
        with suppress_transformers_output():
            model = AutoModel.from_pretrained(...)  # Shows progress, hides spam
    """
    import logging
    
    # Save original log levels
    original_levels = {}
    transformers_loggers = [
        "transformers",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
        "transformers.tokenization_utils_base",
        "huggingface_hub",
        "huggingface_hub.file_download",
    ]
    
    for logger_name in transformers_loggers:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.ERROR)
    
    # Save original environment variables
    original_env = {}
    env_vars = [
        "TRANSFORMERS_VERBOSITY",
        # NOTE: Do NOT suppress HF_HUB_DISABLE_PROGRESS_BARS here
        # We want to see download progress during model installation!
        "TOKENIZERS_PARALLELISM",
    ]
    
    for var in env_vars:
        original_env[var] = os.environ.get(var)
    
    # Set suppression environment variables (but keep progress bars)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    # os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # <-- Don't disable progress!
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        yield
    finally:
        # Restore original log levels
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)
        
        # Restore original environment variables
        for var, value in original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


@contextmanager
def capture_output() -> tuple[IO, IO]:
    """Context manager to capture stdout and stderr to strings.
    
    Returns:
        Tuple of (stdout_capture, stderr_capture) StringIO objects
        
    Example:
        with capture_output() as (out, err):
            print("This will be captured")
        
        captured_stdout = out.getvalue()
        captured_stderr = err.getvalue()
    """
    from io import StringIO
    
    # Save original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Create capture buffers
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        yield stdout_capture, stderr_capture
    finally:
        # Restore original stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def configure_quiet_imports():
    """Configure all imports to be quiet.
    
    Call this at the start of the application to suppress all
    library initialization messages.
    """
    import os
    import warnings
    
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # Environment variables for quiet operation
    quiet_env = {
        # HuggingFace
        "TRANSFORMERS_VERBOSITY": "error",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "DATASETS_VERBOSITY": "error",
        "ACCELERATE_VERBOSITY": "error",
        
        # PyTorch
        "TORCH_LOGS": "error",
        
        # TensorFlow
        "TF_CPP_MIN_LOG_LEVEL": "3",
        
        # BitsAndBytes
        "BITSANDBYTES_NOWELCOME": "1",
        
        # General
        "PYTHONWARNINGS": "ignore",
    }
    
    for var, value in quiet_env.items():
        os.environ[var] = value
