"""Extended data models for rich quantization monitoring."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QuantizationMetrics:
    """Real-time metrics for quantization monitoring."""

    # Progress
    progress_percent: float = 0.0
    eta_seconds: Optional[float] = None

    # Current operation
    stage: str = "Initializing"  # Loading, Quantizing, Saving
    substage: Optional[str] = None  # e.g., "Layer 24/32"

    # Resource usage
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None

    # Throughput
    disk_write_speed_mbps: Optional[float] = None  # MB/s when saving
    layers_per_second: Optional[float] = None  # For layer-by-layer quantization

    # Size tracking
    current_output_size_mb: Optional[float] = None
    estimated_final_size_gb: Optional[float] = None
    size_reduction_percent: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "progress_percent": self.progress_percent,
            "eta_seconds": self.eta_seconds,
            "stage": self.stage,
            "substage": self.substage,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_memory_percent": self.gpu_memory_percent,
            "disk_write_speed_mbps": self.disk_write_speed_mbps,
            "layers_per_second": self.layers_per_second,
            "current_output_size_mb": self.current_output_size_mb,
            "estimated_final_size_gb": self.estimated_final_size_gb,
            "size_reduction_percent": self.size_reduction_percent,
        }
