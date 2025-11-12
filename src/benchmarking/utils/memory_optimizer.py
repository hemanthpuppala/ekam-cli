"""
Memory optimization utilities for benchmarking.

Provides O(1) memory checks, efficient cleanup, and resource management
to prevent OOM errors during benchmark runs.
"""

import gc
import os
import sys
from typing import Optional, Dict, Any
from contextlib import contextmanager
from loguru import logger

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available, memory monitoring will be limited")


class MemoryOptimizer:
    """
    O(1) memory monitoring and cleanup utilities.

    Features:
    - Constant-time memory availability checks
    - Aggressive garbage collection
    - Pre-emptive cleanup strategies
    - Memory pressure detection
    """

    # Singleton cache for O(1) repeated calls
    _last_check_result: Optional[Dict[str, float]] = None
    _check_counter: int = 0
    _cache_invalidate_frequency: int = 10  # Invalidate every N calls

    @classmethod
    def get_available_memory_mb(cls) -> float:
        """
        Get available system memory in MB.

        O(1) amortized through caching with periodic invalidation.

        Returns:
            Available memory in megabytes
        """
        # Cache invalidation for freshness vs performance tradeoff
        cls._check_counter += 1
        if cls._last_check_result and cls._check_counter % cls._cache_invalidate_frequency != 0:
            return cls._last_check_result['available_mb']

        if HAS_PSUTIL:
            # O(1) system call
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 ** 2)

            cls._last_check_result = {
                'available_mb': available_mb,
                'percent_used': mem.percent,
                'total_mb': mem.total / (1024 ** 2)
            }
            return available_mb
        else:
            # Fallback: estimate from /proc/meminfo on Linux (O(1) read)
            if sys.platform == 'linux':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemAvailable' in line:
                            # Parse: "MemAvailable:   12345 kB"
                            available_kb = int(line.split()[1])
                            return available_kb / 1024  # Convert to MB

            # Last resort: assume we have memory (unsafe)
            logger.warning("Cannot determine available memory, assuming 1GB free")
            return 1024.0

    @classmethod
    def get_memory_stats(cls) -> Dict[str, float]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dict with memory stats in MB
        """
        if not HAS_PSUTIL:
            return {'available_mb': cls.get_available_memory_mb()}

        # Refresh cache
        cls._check_counter = 0  # Force refresh
        cls.get_available_memory_mb()

        return cls._last_check_result or {'available_mb': 0.0}

    @classmethod
    def has_sufficient_memory(cls, required_mb: float = 300.0) -> bool:
        """
        Check if system has sufficient free memory.

        O(1) amortized through caching.

        Args:
            required_mb: Minimum required memory in MB (default: 300MB)

        Returns:
            True if sufficient memory available
        """
        available = cls.get_available_memory_mb()
        return available >= required_mb

    @classmethod
    def aggressive_cleanup(cls) -> Dict[str, Any]:
        """
        Perform aggressive memory cleanup.

        Complexity: O(n) where n = number of unreachable objects
        In practice: Very fast, typically <10ms for small heaps

        Returns:
            Dict with cleanup statistics
        """
        stats_before = cls.get_memory_stats()

        # Force full garbage collection across all generations
        # Generation 0: young objects (fast)
        # Generation 1: mid-age objects
        # Generation 2: old objects (slowest but most thorough)
        collected = [
            gc.collect(0),  # O(young_objects)
            gc.collect(1),  # O(mid_objects)
            gc.collect(2),  # O(old_objects)
        ]

        stats_after = cls.get_memory_stats()

        freed_mb = stats_after.get('available_mb', 0) - stats_before.get('available_mb', 0)

        result = {
            'objects_collected': sum(collected),
            'freed_mb': freed_mb,
            'available_before_mb': stats_before.get('available_mb', 0),
            'available_after_mb': stats_after.get('available_mb', 0)
        }

        logger.debug(
            f"Memory cleanup: {result['objects_collected']} objects collected, "
            f"{result['freed_mb']:.1f}MB freed, "
            f"{result['available_after_mb']:.0f}MB now available"
        )

        return result

    @classmethod
    def smart_cleanup(cls, threshold_mb: float = 500.0) -> bool:
        """
        Conditionally perform cleanup only if memory is low.

        O(1) check + conditional O(n) cleanup.

        Args:
            threshold_mb: Trigger cleanup if available memory below this

        Returns:
            True if cleanup was performed
        """
        available = cls.get_available_memory_mb()

        if available < threshold_mb:
            logger.info(f"Low memory detected ({available:.0f}MB < {threshold_mb:.0f}MB), running cleanup")
            cls.aggressive_cleanup()
            return True

        return False

    @classmethod
    def memory_pressure_level(cls) -> str:
        """
        Determine current memory pressure level.

        O(1) operation.

        Returns:
            "low", "medium", "high", or "critical"
        """
        available = cls.get_available_memory_mb()

        if available >= 1000:
            return "low"
        elif available >= 500:
            return "medium"
        elif available >= 200:
            return "high"
        else:
            return "critical"

    @classmethod
    def clear_cache(cls):
        """Clear internal cache to force fresh memory reads."""
        cls._last_check_result = None
        cls._check_counter = 0


@contextmanager
def managed_image(image_path: str):
    """
    Context manager for PIL Images with automatic cleanup.

    Ensures images are properly closed and memory is freed.
    O(1) overhead beyond image loading.

    Args:
        image_path: Path to image file

    Yields:
        PIL Image object

    Example:
        ```python
        with managed_image("path/to/image.jpg") as img:
            result = model.predict(img)
        # Image automatically closed, memory freed
        ```
    """
    from PIL import Image

    img = None
    try:
        img = Image.open(image_path)

        # Convert to RGB if needed (common requirement for VLMs)
        if img.mode not in ['RGB', 'RGBA']:
            old_img = img
            img = img.convert('RGB')
            old_img.close()  # Close original to free memory

        # Attach source path for downstream debugging/logging
        try:
            setattr(img, "_source_path", str(image_path))
        except Exception:
            pass

        yield img

    finally:
        # Ensure cleanup even on exceptions
        if img:
            try:
                img.close()
            except Exception as e:
                logger.warning(f"Error closing image: {e}")

        # Force cleanup of image object
        del img


@contextmanager
def memory_tracked_operation(
    operation_name: str,
    min_required_mb: float = 300.0,
    auto_cleanup: bool = True
):
    """
    Context manager for memory-tracked operations.

    Features:
    - Pre-check for sufficient memory
    - Auto cleanup before operation
    - Post-operation cleanup
    - Memory usage reporting

    Args:
        operation_name: Name for logging
        min_required_mb: Minimum required free memory
        auto_cleanup: Whether to auto-cleanup if memory is low

    Yields:
        Dict with memory stats

    Raises:
        MemoryError: If insufficient memory and cleanup doesn't help

    Example:
        ```python
        with memory_tracked_operation("Inference Run 3", min_required_mb=250) as stats:
            result = model.inference(input)
        # Auto cleanup performed, stats available
        ```
    """
    stats = {}

    try:
        # Pre-operation check
        mem_before = MemoryOptimizer.get_memory_stats()
        stats['before'] = mem_before

        available_mb = mem_before.get('available_mb', 0)

        # Auto cleanup if needed
        if available_mb < min_required_mb and auto_cleanup:
            logger.info(f"{operation_name}: Low memory ({available_mb:.0f}MB), running cleanup")
            cleanup_result = MemoryOptimizer.aggressive_cleanup()
            stats['cleanup'] = cleanup_result

            # Re-check after cleanup
            available_mb = MemoryOptimizer.get_available_memory_mb()

        # Final check
        if available_mb < min_required_mb:
            raise MemoryError(
                f"Insufficient memory for {operation_name}: "
                f"{available_mb:.0f}MB available, {min_required_mb:.0f}MB required"
            )

        logger.debug(f"{operation_name}: Starting with {available_mb:.0f}MB available")

        yield stats

    finally:
        # Post-operation cleanup
        if auto_cleanup:
            MemoryOptimizer.aggressive_cleanup()

        mem_after = MemoryOptimizer.get_memory_stats()
        stats['after'] = mem_after

        # Log memory usage delta
        delta_mb = mem_after.get('available_mb', 0) - mem_before.get('available_mb', 0)
        logger.debug(
            f"{operation_name}: Complete. "
            f"Memory delta: {delta_mb:+.1f}MB, "
            f"Available: {mem_after.get('available_mb', 0):.0f}MB"
        )


class RetryWithCleanup:
    """
    Retry mechanism with progressive memory cleanup.

    O(1) space complexity - reuses same retry state.
    """

    def __init__(
        self,
        max_retries: int = 2,
        cleanup_between_retries: bool = True,
        backoff_seconds: float = 2.0
    ):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum retry attempts (default: 2)
            cleanup_between_retries: Run cleanup between retries
            backoff_seconds: Base backoff time (exponential: 2^attempt)
        """
        self.max_retries = max_retries
        self.cleanup_between_retries = cleanup_between_retries
        self.backoff_seconds = backoff_seconds

    def execute(self, operation, *args, **kwargs):
        """
        Execute operation with retry and cleanup.

        Args:
            operation: Callable to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result from operation

        Raises:
            Last exception if all retries fail
        """
        import time

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Attempt operation
                result = operation(*args, **kwargs)

                if attempt > 0:
                    logger.info(f"Operation succeeded on retry {attempt}")

                return result

            except (TimeoutError, MemoryError, Exception) as e:
                last_exception = e
                error_type = type(e).__name__

                if attempt < self.max_retries:
                    wait_time = self.backoff_seconds * (2 ** attempt)

                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{error_type}: {str(e)[:100]}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )

                    # Cleanup before retry
                    if self.cleanup_between_retries:
                        MemoryOptimizer.aggressive_cleanup()

                    # Exponential backoff
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Operation failed after {self.max_retries + 1} attempts: "
                        f"{error_type}: {str(e)}"
                    )

        # All retries exhausted
        raise last_exception
