"""Lightweight GGUF inspector utilities.

Provides helpers to read tensor shapes from GGUF files without depending on
external system installations. Uses the bundled llama.cpp/gguf-py when present.
Gracefully degrades when unavailable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from loguru import logger


def _add_local_gguf_to_syspath() -> Optional[Path]:
    """Attempt to add local llama.cpp/gguf-py to sys.path.

    Returns:
        Path to gguf-py root if found, else None
    """
    # Try typical relative locations
    candidates = [
        Path.cwd() / "llama.cpp" / "gguf-py",
        Path(__file__).parent.parent.parent / "llama.cpp" / "gguf-py",
        Path("./llama.cpp/gguf-py").resolve(),
    ]

    for p in candidates:
        if (p / "gguf" / "gguf_reader.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            logger.debug(f"Using bundled gguf library at: {p}")
            return p

    logger.debug("Bundled gguf library not found; will try system gguf if available")
    return None


def _import_gguf_reader():
    """Import GGUFReader from gguf, preferring bundled one.

    Returns:
        GGUFReader class or None if not available
    """
    try:
        _add_local_gguf_to_syspath()
        from gguf.gguf_reader import GGUFReader  # type: ignore
        return GGUFReader
    except Exception as e:
        logger.warning(f"gguf reader not available: {e}")
        return None


def list_tensors(gguf_path: Path) -> List[Tuple[str, List[int]]]:
    """List tensor names and shapes from a GGUF file.

    Args:
        gguf_path: Path to GGUF file

    Returns:
        List of (name, shape) pairs. Empty list if reader unavailable.
    """
    GGUFReader = _import_gguf_reader()
    if GGUFReader is None:
        return []

    try:
        reader = GGUFReader(str(gguf_path))
        tensors = []
        for t in reader.tensors:
            # t.shape is np array of uint32 [n_dims]
            shape = [int(x) for x in list(t.shape)]
            tensors.append((t.name, shape))
        return tensors
    except Exception as e:
        logger.error(f"Failed to read GGUF tensors from {gguf_path}: {e}")
        return []


def find_tensor_shapes_by_prefix(gguf_path: Path, prefixes: Iterable[str]) -> Dict[str, List[int]]:
    """Find tensor shapes whose names start with any of the given prefixes.

    Args:
        gguf_path: GGUF file to inspect
        prefixes: Iterable of tensor name prefixes

    Returns:
        Dict of tensor_name -> shape list[int]
    """
    prefixes = tuple(prefixes)
    result: Dict[str, List[int]] = {}
    for name, shape in list_tensors(gguf_path):
        if name.startswith(prefixes):
            result[name] = shape
    return result


def detect_mmproj_input_dim(gguf_mmproj_path: Path) -> Optional[int]:
    """Detect the expected input dimension for the mmproj MLP.

    For LLaVA-style projectors this is typically the first linear weight
    'mm.0.weight' with shape [in_dim, hidden]. For Moondream2 multi-crop models
    this is often 2304 (global 1152 + regional 1152).

    Args:
        gguf_mmproj_path: Path to the mmproj GGUF file

    Returns:
        in_dim if found (e.g., 1152 or 2304), else None
    """
    try:
        matches = find_tensor_shapes_by_prefix(gguf_mmproj_path, ["mm.0.weight"])  # exact prefix
        if not matches:
            logger.debug(f"mm.0.weight not found in {gguf_mmproj_path}")
            return None
        # Take the first match deterministically
        name = sorted(matches.keys())[0]
        shape = matches[name]
        if not shape:
            return None
        in_dim = int(shape[0])  # GGML uses [rows, cols] with rows = input dim
        logger.info(f"mmproj detected: {name} shape={shape} → in_dim={in_dim}")
        return in_dim
    except Exception as e:
        logger.error(f"Failed to detect mmproj input dim: {e}")
        return None

