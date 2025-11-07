"""JSON persistence utilities for inference results."""

import errno
import json
import shutil
from pathlib import Path

from loguru import logger

from ..models.inference import InferenceResult


def check_disk_space(path: Path, required_mb: float = 10.0) -> tuple[bool, str]:
    """Check if sufficient disk space is available.

    Args:
        path: Path to check disk space for
        required_mb: Required space in MB (default: 10MB)

    Returns:
        Tuple of (has_space, message)
    """
    try:
        stat = shutil.disk_usage(path.parent if path.is_file() or not path.exists() else path)
        free_mb = stat.free / (1024 ** 2)

        if free_mb < required_mb:
            return False, f"Insufficient disk space: {free_mb:.1f}MB available, {required_mb:.1f}MB required"
        return True, f"{free_mb:.1f}MB available"
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True, "Could not verify disk space"  # Continue anyway


def save_result(result: InferenceResult, directory: Path = Path("results")) -> Path:
    """Save inference result to JSON file with timestamp.

    Args:
        result: InferenceResult to save
        directory: Directory to save results (default: results/)

    Returns:
        Path to saved JSON file

    Raises:
        OSError: If directory cannot be created or file cannot be saved
        IOError: If disk is full (ENOSPC)
    """
    # Normalize path
    directory = Path(directory).expanduser().resolve()

    # Check disk space before attempting save (estimate 10MB needed)
    has_space, space_msg = check_disk_space(directory, required_mb=10.0)
    if not has_space:
        logger.error(space_msg)
        raise IOError(space_msg)

    # Create directory if it doesn't exist
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create results directory: {directory}")
        if e.errno == errno.ENOSPC:
            raise IOError(f"Disk full: Cannot create directory {directory}")
        raise OSError(f"Failed to create results directory: {directory}\nError: {str(e)}")

    # Generate filename from result ID
    file_path = directory / f"{result.result_id}.json"

    # Save to JSON with error handling
    try:
        with open(file_path, "w") as f:
            json.dump(result.to_json_dict(), f, indent=2)
        logger.debug(f"Saved result to: {file_path}")
        return file_path
    except OSError as e:
        logger.error(f"Cannot save result: {file_path}")
        if e.errno == errno.ENOSPC:
            raise IOError(f"Disk full: Cannot save result to {file_path}")
        raise OSError(f"Failed to save result: {file_path}\nError: {str(e)}")


def load_result(file_path: Path) -> InferenceResult:
    """Load inference result from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        InferenceResult instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    return InferenceResult.from_json_dict(data)


def list_results(directory: Path = Path("results")) -> list[Path]:
    """List all saved result files in directory.

    Args:
        directory: Directory containing results

    Returns:
        List of Path objects for JSON files, sorted by name (timestamp)
    """
    if not directory.exists():
        return []

    return sorted(directory.glob("*.json"))
