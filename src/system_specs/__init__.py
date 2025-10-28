"""System specifications and real-time monitoring module.

Provides comprehensive hardware detection and runtime monitoring for:
- CPU, GPU, memory, storage specs
- Temperature monitoring
- Real-time resource usage
- Inference performance metrics
"""

from .base import (
    TemperatureInfo,
    StorageInfo,
    FrequencyInfo,
    PerCoreMetrics,
    CPUMetrics,
    MemoryMetrics,
    StorageMetrics,
    GPUMetrics,
    NetworkMetrics,
    ProcessMetric,
    SystemHealthSnapshot,
    InferenceMetrics,
    MonitoringConfig,
    MonitoringLevel,
)
from .detector import SystemSpecsDetector
from .monitor import SystemMonitor, InferenceTracker
from .exporters import MetricsExporter

__all__ = [
    # Data models
    "TemperatureInfo",
    "StorageInfo",
    "FrequencyInfo",
    "PerCoreMetrics",
    "CPUMetrics",
    "MemoryMetrics",
    "StorageMetrics",
    "GPUMetrics",
    "NetworkMetrics",
    "ProcessMetric",
    "SystemHealthSnapshot",
    "InferenceMetrics",
    "MonitoringConfig",
    "MonitoringLevel",
    # Main classes
    "SystemSpecsDetector",
    "SystemMonitor",
    "InferenceTracker",
    "MetricsExporter",
]
