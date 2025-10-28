"""Base backend interface for metrics collection."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from ..base import (
    CPUMetrics,
    MemoryMetrics,
    StorageMetrics,
    GPUMetrics,
    NetworkMetrics,
    TemperatureInfo,
    FrequencyInfo,
    ProcessMetric,
)


class MetricsBackend(ABC):
    """Abstract base class for platform-specific metrics backends."""

    def __init__(self):
        """Initialize backend."""
        self.available = self._check_availability()

    @abstractmethod
    def _check_availability(self) -> bool:
        """Check if this backend is available on current platform.

        Returns:
            True if backend can be used
        """
        pass

    def is_available(self) -> bool:
        """Check if backend is available."""
        return self.available

    # Core metrics (should be implemented by all backends)

    def get_cpu_metrics(self) -> Optional[CPUMetrics]:
        """Get CPU metrics. Return None if not supported."""
        return None

    def get_memory_metrics(self) -> Optional[MemoryMetrics]:
        """Get memory metrics. Return None if not supported."""
        return None

    def get_storage_metrics(self) -> Optional[StorageMetrics]:
        """Get storage metrics. Return None if not supported."""
        return None

    def get_gpu_metrics(self, device_id: int = 0) -> Optional[GPUMetrics]:
        """Get GPU metrics for specified device. Return None if not supported."""
        return None

    def get_all_gpu_metrics(self) -> List[GPUMetrics]:
        """Get metrics for all GPUs. Return empty list if not supported."""
        return []

    # Extended metrics (optional)

    def get_temperature_info(self) -> Optional[TemperatureInfo]:
        """Get temperature information. Return None if not supported."""
        return None

    def get_frequency_info(self) -> Optional[FrequencyInfo]:
        """Get frequency information. Return None if not supported."""
        return None

    def get_network_metrics(self) -> Optional[NetworkMetrics]:
        """Get network metrics. Return None if not supported."""
        return None

    def get_top_processes(self, limit: int = 10) -> List[ProcessMetric]:
        """Get top processes by resource usage. Return empty list if not supported."""
        return []

    def get_battery_info(self) -> Optional[Dict]:
        """Get battery information. Return None if not supported."""
        return None

    def get_system_uptime(self) -> Optional[float]:
        """Get system uptime in seconds. Return None if not supported."""
        return None
