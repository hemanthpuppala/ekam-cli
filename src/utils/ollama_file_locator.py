"""Ollama file locator - Find GGUF blobs for locally installed Ollama models.

This module provides utilities to locate the actual GGUF files stored by Ollama
in the ~/.ollama/models/blobs/ directory. Ollama stores models using content-
addressable storage with SHA256 hashes.

Architecture:
    1. Query `ollama show <model> --modelfile` to get blob reference
    2. Parse the FROM line: "FROM blob:sha256-abc123..."
    3. Locate file: ~/.ollama/models/blobs/sha256-abc123...
    4. Cache mappings for performance

Example:
    >>> locator = OllamaFileLocator()
    >>> path = locator.get_model_path("llama3.2:3b")
    >>> print(path)
    PosixPath('/Users/user/.ollama/models/blobs/sha256-abc123...')

Future-proof design:
    - Handles multiple blob formats (sha256, sha512, etc.)
    - Caches results to minimize subprocess calls
    - Validates blob files before returning
    - Supports custom Ollama home directories
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Dict
from loguru import logger


class OllamaFileLocator:
    """Locate GGUF files for Ollama models in blob storage."""

    def __init__(self, ollama_home: Optional[Path] = None, cache_enabled: bool = True):
        """Initialize Ollama file locator.

        Args:
            ollama_home: Custom Ollama home directory (default: ~/.ollama)
            cache_enabled: Enable caching of model→path mappings
        """
        self.ollama_home = ollama_home or Path.home() / ".ollama"
        self.blobs_dir = self.ollama_home / "models" / "blobs"
        self.cache_enabled = cache_enabled
        self._path_cache: Dict[str, Path] = {}

        logger.debug(f"Initialized OllamaFileLocator (home: {self.ollama_home})")

    def get_model_path(self, model_name: str, force_refresh: bool = False) -> Optional[Path]:
        """Get file path for an Ollama model.

        This is the main entry point. Returns the path to the GGUF file
        stored in Ollama's blob storage.

        Args:
            model_name: Ollama model name (e.g., "llama3.2:3b", "mistral:7b")
            force_refresh: Force cache refresh even if cached

        Returns:
            Path to GGUF file, or None if not found

        Example:
            >>> locator = OllamaFileLocator()
            >>> path = locator.get_model_path("llama3.2:3b")
            >>> print(path.exists())
            True
        """
        # Check cache first
        if self.cache_enabled and not force_refresh and model_name in self._path_cache:
            cached_path = self._path_cache[model_name]
            if cached_path.exists():
                logger.debug(f"Cache hit for {model_name}: {cached_path}")
                return cached_path
            else:
                # Cache entry is stale, remove it
                del self._path_cache[model_name]

        # Get modelfile to extract blob reference
        modelfile = self._get_modelfile(model_name)
        if not modelfile:
            logger.warning(f"Could not get modelfile for {model_name}")
            return None

        # Parse blob hash from modelfile
        blob_hash = self._parse_blob_hash(modelfile)
        if not blob_hash:
            logger.warning(f"Could not parse blob hash from modelfile for {model_name}")
            return None

        # Locate blob file
        blob_path = self._find_blob_file(blob_hash)
        if not blob_path:
            logger.warning(f"Could not locate blob file for {model_name} (hash: {blob_hash})")
            return None

        # Validate it's a GGUF file
        if not self._validate_gguf_file(blob_path):
            logger.warning(f"Blob file is not a valid GGUF file: {blob_path}")
            return None

        # Cache the result
        if self.cache_enabled:
            self._path_cache[model_name] = blob_path

        logger.info(f"Located Ollama model {model_name} at {blob_path}")
        return blob_path

    def _get_modelfile(self, model_name: str) -> Optional[str]:
        """Get modelfile content for an Ollama model.

        Runs: ollama show <model> --modelfile

        Args:
            model_name: Model name

        Returns:
            Modelfile content as string, or None if failed
        """
        try:
            result = subprocess.run(
                ["ollama", "show", model_name, "--modelfile"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.debug(f"ollama show failed for {model_name}: {result.stderr}")
                return None

            return result.stdout

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout getting modelfile for {model_name}")
            return None
        except FileNotFoundError:
            logger.error("ollama command not found - is Ollama installed?")
            return None
        except Exception as e:
            logger.error(f"Error getting modelfile for {model_name}: {e}")
            return None

    def _parse_blob_hash(self, modelfile: str) -> Optional[str]:
        """Parse blob hash from modelfile content.

        Expected format:
            FROM blob:sha256-abc123def456...

        Also supports:
            FROM @sha256:abc123def456...  (alternative format)

        Args:
            modelfile: Modelfile content

        Returns:
            Blob hash (e.g., "sha256-abc123..."), or None if not found
        """
        # Pattern 1: blob:sha256-abc123...
        pattern1 = r"FROM\s+blob:(sha256-[a-f0-9]+)"
        match = re.search(pattern1, modelfile, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 2: @sha256:abc123... (convert to sha256-abc123...)
        pattern2 = r"FROM\s+@(sha256):([a-f0-9]+)"
        match = re.search(pattern2, modelfile, re.IGNORECASE)
        if match:
            hash_type = match.group(1)
            hash_value = match.group(2)
            return f"{hash_type}-{hash_value}"

        # Pattern 3: Just the hash (fallback)
        pattern3 = r"FROM\s+(sha256-[a-f0-9]+)"
        match = re.search(pattern3, modelfile, re.IGNORECASE)
        if match:
            return match.group(1)

        logger.debug(f"No blob hash found in modelfile:\n{modelfile}")
        return None

    def _find_blob_file(self, blob_hash: str) -> Optional[Path]:
        """Find blob file in Ollama's blob storage.

        Args:
            blob_hash: Blob hash (e.g., "sha256-abc123...")

        Returns:
            Path to blob file, or None if not found
        """
        if not self.blobs_dir.exists():
            logger.error(f"Ollama blobs directory not found: {self.blobs_dir}")
            return None

        # Direct lookup: blobs/sha256-abc123...
        blob_path = self.blobs_dir / blob_hash
        if blob_path.exists() and blob_path.is_file():
            return blob_path

        # Alternative format: blobs/sha256/abc123... (directory structure)
        if "-" in blob_hash:
            hash_type, hash_value = blob_hash.split("-", 1)
            alt_path = self.blobs_dir / hash_type / hash_value
            if alt_path.exists() and alt_path.is_file():
                return alt_path

        logger.debug(f"Blob file not found: {blob_hash}")
        return None

    def _validate_gguf_file(self, file_path: Path) -> bool:
        """Validate that a file is a valid GGUF file.

        Checks the magic number at the beginning of the file.
        GGUF files start with: "GGUF" (0x47 0x47 0x55 0x46)

        Args:
            file_path: Path to file

        Returns:
            True if valid GGUF file
        """
        try:
            with open(file_path, "rb") as f:
                magic = f.read(4)
                return magic == b"GGUF"
        except Exception as e:
            logger.debug(f"Error validating GGUF file {file_path}: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear the path cache.

        Useful when models are updated or removed.
        """
        self._path_cache.clear()
        logger.debug("Cleared OllamaFileLocator cache")

    def get_cached_models(self) -> Dict[str, Path]:
        """Get all cached model→path mappings.

        Returns:
            Dictionary of model_name → path
        """
        return self._path_cache.copy()

    def list_available_blobs(self) -> list[Path]:
        """List all blob files in Ollama's storage.

        Returns:
            List of paths to blob files
        """
        if not self.blobs_dir.exists():
            return []

        blobs = []
        for item in self.blobs_dir.iterdir():
            if item.is_file() and item.name.startswith("sha"):
                blobs.append(item)

        return blobs


def get_ollama_model_path(model_name: str) -> Optional[Path]:
    """Convenience function to get Ollama model path.

    This is a simple wrapper around OllamaFileLocator for quick usage.

    Args:
        model_name: Ollama model name (e.g., "llama3.2:3b")

    Returns:
        Path to GGUF file, or None if not found

    Example:
        >>> from src.utils.ollama_file_locator import get_ollama_model_path
        >>> path = get_ollama_model_path("mistral:7b")
        >>> if path:
        ...     print(f"Found at: {path}")
    """
    locator = OllamaFileLocator()
    return locator.get_model_path(model_name)
