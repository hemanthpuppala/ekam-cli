"""Base data models for system specifications and monitoring."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum


class MonitoringLevel(Enum):
    """Level of monitoring detail."""
    MINIMAL = "minimal"  # Only basic CPU/memory
    STANDARD = "standard"  # + GPU, storage
    DETAILED = "detailed"  # + per-core, temperatures
    FULL = "full"  # Everything including network, processes


@dataclass
class MonitoringConfig:
    """Configuration for system monitoring."""
    level: MonitoringLevel = MonitoringLevel.STANDARD
    track_temperature: bool = True
    track_per_core: bool = False
    track_network: bool = False
    track_processes: bool = False
    track_gpu: bool = True
    interval_sec: float = 1.0


@dataclass
class TemperatureInfo:
    """Temperature information for CPU and GPU."""
    cpu_temp_celsius: Optional[float] = None
    cpu_package_temp: Optional[float] = None  # Package/die temp
    gpu_temps: Dict[int, float] = field(default_factory=dict)  # Per GPU device
    throttling: bool = False
    max_safe_temp_cpu: float = 85.0
    max_safe_temp_gpu: float = 90.0

    @property
    def is_overheating(self) -> bool:
        """Check if any component is overheating."""
        if self.cpu_temp_celsius and self.cpu_temp_celsius >= self.max_safe_temp_cpu:
            return True
        for temp in self.gpu_temps.values():
            if temp >= self.max_safe_temp_gpu:
                return True
        return False


@dataclass
class StorageInfo:
    """Storage/disk information."""
    device: str  # /dev/sda1, C:, etc.
    mount_point: str = "/"
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    percent_used: float = 0.0
    filesystem: Optional[str] = None

    # I/O stats
    read_mb: float = 0.0
    write_mb: float = 0.0
    read_count: int = 0
    write_count: int = 0


@dataclass
class FrequencyInfo:
    """CPU and GPU frequency information."""
    cpu_current_ghz: float = 0.0
    cpu_min_ghz: float = 0.0
    cpu_max_ghz: float = 0.0
    cpu_per_core_ghz: List[float] = field(default_factory=list)

    # GPU frequencies in MHz
    gpu_core_mhz: Optional[int] = None
    gpu_memory_mhz: Optional[int] = None


@dataclass
class PerCoreMetrics:
    """Per-core CPU metrics."""
    core_id: int
    cpu_percent: float = 0.0
    frequency_ghz: float = 0.0
    temperature_c: Optional[float] = None


@dataclass
class CPUMetrics:
    """Comprehensive CPU metrics."""
    # Basic info
    architecture: str = ""
    cores_physical: int = 0
    cores_logical: int = 0

    # Usage
    usage_percent: float = 0.0
    per_core_usage: List[float] = field(default_factory=list)

    # Frequency
    frequency: Optional[FrequencyInfo] = None

    # Per-core details (optional)
    per_core_metrics: List[PerCoreMetrics] = field(default_factory=list)

    # Context switches and interrupts (Linux)
    context_switches: Optional[int] = None
    interrupts: Optional[int] = None

    # Load average (Linux/Mac)
    load_avg_1min: Optional[float] = None
    load_avg_5min: Optional[float] = None
    load_avg_15min: Optional[float] = None


@dataclass
class MemoryMetrics:
    """Memory metrics."""
    # RAM
    total_ram_gb: float = 0.0
    used_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    percent_used: float = 0.0

    # Swap
    total_swap_gb: float = 0.0
    used_swap_gb: float = 0.0
    swap_percent_used: float = 0.0

    # Detailed (Linux)
    buffers_gb: Optional[float] = None
    cached_gb: Optional[float] = None
    shared_gb: Optional[float] = None


@dataclass
class StorageMetrics:
    """Storage metrics for all mounted devices."""
    devices: List[StorageInfo] = field(default_factory=list)

    # Aggregate I/O stats
    total_read_mb: float = 0.0
    total_write_mb: float = 0.0
    total_read_count: int = 0
    total_write_count: int = 0


@dataclass
class GPUMetrics:
    """GPU metrics."""
    device_id: int = 0
    device_name: str = ""

    # Memory
    total_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    memory_percent: float = 0.0

    # Utilization
    gpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0

    # Temperature
    temperature_celsius: Optional[float] = None

    # Power
    power_draw_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None

    # Frequency
    frequency: Optional[FrequencyInfo] = None

    # Compute capability (NVIDIA)
    compute_capability: Optional[str] = None  # e.g., "8.6"


@dataclass
class NetworkMetrics:
    """Network I/O metrics."""
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    errors_in: int = 0
    errors_out: int = 0
    drops_in: int = 0
    drops_out: int = 0


@dataclass
class ProcessMetric:
    """Individual process metrics."""
    pid: int
    name: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    status: str = ""


@dataclass
class SystemHealthSnapshot:
    """Complete system health snapshot at a point in time."""
    timestamp: float

    # Core metrics
    cpu: CPUMetrics
    memory: MemoryMetrics
    storage: StorageMetrics

    # GPU metrics (can be multiple GPUs)
    gpus: List[GPUMetrics] = field(default_factory=list)

    # Temperatures
    temperatures: Optional[TemperatureInfo] = None

    # Network
    network: Optional[NetworkMetrics] = None

    # Top processes
    top_processes: List[ProcessMetric] = field(default_factory=list)

    # Battery (for laptops)
    battery_percent: Optional[float] = None
    battery_plugged: Optional[bool] = None

    # System uptime
    uptime_seconds: Optional[float] = None


@dataclass
class InferenceMetrics:
    """Metrics collected during model inference."""

    # Session info
    model_name: str
    start_time: float
    end_time: float
    inference_time_sec: float

    # Throughput
    tokens_generated: Optional[int] = None
    tokens_per_second: Optional[float] = None

    # Peak resource usage
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    peak_gpu_utilization: float = 0.0

    # Average resource usage
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    avg_gpu_memory_mb: float = 0.0
    avg_gpu_utilization: float = 0.0

    # Timelines (sampled data)
    cpu_usage_timeline: List[float] = field(default_factory=list)
    memory_usage_timeline: List[float] = field(default_factory=list)
    gpu_usage_timeline: List[float] = field(default_factory=list)
    gpu_memory_timeline: List[float] = field(default_factory=list)
    temperature_timeline: List[float] = field(default_factory=list)

    # Thermal info
    cpu_temp_start: Optional[float] = None
    cpu_temp_end: Optional[float] = None
    cpu_temp_delta: Optional[float] = None
    cpu_temp_peak: Optional[float] = None

    gpu_temp_start: Optional[float] = None
    gpu_temp_end: Optional[float] = None
    gpu_temp_delta: Optional[float] = None
    gpu_temp_peak: Optional[float] = None

    throttling_occurred: bool = False

    # I/O during inference
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    network_bytes: int = 0

    # Power consumption (if available)
    avg_power_draw_watts: Optional[float] = None
    total_energy_joules: Optional[float] = None

    # Full snapshots (for detailed analysis)
    snapshots: List[SystemHealthSnapshot] = field(default_factory=list)
