"""
Helper functions for Quality Suite.

Provides utility functions for prompt selection and metric computation.
"""

from typing import List
import random
from loguru import logger


def select_prompts_first_n(prompts: List[str], n: int) -> List[str]:
    """
    Select first N prompts from the list.

    Args:
        prompts: List of available prompts
        n: Number of prompts to select (-1 for all)

    Returns:
        Selected prompts
    """
    if n == -1 or n >= len(prompts):
        return prompts
    return prompts[:n]


def select_prompts_random_n(prompts: List[str], n: int, seed: int = 42) -> List[str]:
    """
    Randomly select N prompts from the list.

    Args:
        prompts: List of available prompts
        n: Number of prompts to select (-1 for all)
        seed: Random seed for reproducibility

    Returns:
        Selected prompts
    """
    if n == -1 or n >= len(prompts):
        return prompts

    # Use seed for reproducibility
    random.seed(seed)
    return random.sample(prompts, n)


def select_prompts_evenly_spaced(prompts: List[str], n: int) -> List[str]:
    """
    Select N prompts evenly spaced across the list.

    Args:
        prompts: List of available prompts
        n: Number of prompts to select (-1 for all)

    Returns:
        Selected prompts
    """
    if n == -1 or n >= len(prompts):
        return prompts

    # Calculate step size
    total = len(prompts)
    step = total / n

    # Select evenly spaced indices
    selected = []
    for i in range(n):
        idx = int(i * step)
        selected.append(prompts[idx])

    return selected
