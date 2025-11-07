"""NVIDIA GPU backend using nvidia-ml-py (pynvml) and PyTorch."""

from typing import Optional, List
from .base import MetricsBackend
from ..base import GPUMetrics, TemperatureInfo, FrequencyInfo

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NvidiaBackend(MetricsBackend):
    """NVIDIA GPU metrics backend using pynvml and PyTorch."""

    def __init__(self):
        """Initialize NVIDIA backend."""
        self.nvml_initialized = False
        self.device_count = 0
        super().__init__()

    def _check_availability(self) -> bool:
        """Check if NVIDIA GPUs are available."""
        if not PYNVML_AVAILABLE and not TORCH_AVAILABLE:
            return False

        # Try to initialize NVML
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.device_count = pynvml.nvmlDeviceGetCount()
                return self.device_count > 0
            except Exception:
                pass

        # Fallback to PyTorch
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    self.device_count = torch.cuda.device_count()
                    return self.device_count > 0
            except Exception:
                pass

        return False

    def __del__(self):
        """Clean up NVML."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def get_gpu_metrics(self, device_id: int = 0) -> Optional[GPUMetrics]:
        """Get metrics for specific NVIDIA GPU."""
        if not self.available or device_id >= self.device_count:
            return None

        # Try NVML first (more detailed)
        if self.nvml_initialized:
            return self._get_gpu_metrics_nvml(device_id)

        # Fallback to PyTorch
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return self._get_gpu_metrics_torch(device_id)

        return None

    def _get_gpu_metrics_nvml(self, device_id: int) -> Optional[GPUMetrics]:
        """Get GPU metrics using NVML."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # Device name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = mem_info.total / (1024 ** 3)
            used_gb = mem_info.used / (1024 ** 3)
            free_gb = mem_info.free / (1024 ** 3)
            memory_percent = (mem_info.used / mem_info.total * 100) if mem_info.total > 0 else 0.0

            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                mem_util = util.memory
            except Exception:
                gpu_util = 0.0
                mem_util = 0.0

            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None

            # Power
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except Exception:
                power_draw = None
                power_limit = None

            # Clock frequencies
            freq_info = None
            try:
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                freq_info = FrequencyInfo(
                    gpu_core_mhz=gpu_clock,
                    gpu_memory_mhz=mem_clock,
                )
            except Exception:
                pass

            # Compute capability
            compute_cap = None
            try:
                major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                compute_cap = f"{major}.{minor}"
            except Exception:
                pass

            return GPUMetrics(
                device_id=device_id,
                device_name=name,
                total_memory_gb=total_gb,
                used_memory_gb=used_gb,
                free_memory_gb=free_gb,
                memory_percent=memory_percent,
                gpu_utilization_percent=gpu_util,
                memory_utilization_percent=mem_util,
                temperature_celsius=temp,
                power_draw_watts=power_draw,
                power_limit_watts=power_limit,
                frequency=freq_info,
                compute_capability=compute_cap,
            )

        except Exception:
            return None

    def _get_gpu_metrics_torch(self, device_id: int) -> Optional[GPUMetrics]:
        """Get GPU metrics using PyTorch (fallback)."""
        try:
            # Device name
            name = torch.cuda.get_device_name(device_id)

            # Memory info
            props = torch.cuda.get_device_properties(device_id)
            total_gb = props.total_memory / (1024 ** 3)

            allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
            free_gb = total_gb - reserved
            memory_percent = (reserved / total_gb * 100) if total_gb > 0 else 0.0

            # Compute capability
            compute_cap = f"{props.major}.{props.minor}"

            return GPUMetrics(
                device_id=device_id,
                device_name=name,
                total_memory_gb=total_gb,
                used_memory_gb=reserved,
                free_memory_gb=free_gb,
                memory_percent=memory_percent,
                gpu_utilization_percent=0.0,  # Not available via PyTorch
                memory_utilization_percent=0.0,
                compute_capability=compute_cap,
            )

        except Exception:
            return None

    def get_all_gpu_metrics(self) -> List[GPUMetrics]:
        """Get metrics for all NVIDIA GPUs."""
        if not self.available:
            return []

        metrics = []
        for i in range(self.device_count):
            gpu_metric = self.get_gpu_metrics(i)
            if gpu_metric:
                metrics.append(gpu_metric)

        return metrics

    def get_temperature_info(self) -> Optional[TemperatureInfo]:
        """Get temperature info for all NVIDIA GPUs."""
        if not self.available or not self.nvml_initialized:
            return None

        try:
            temp_info = TemperatureInfo()

            for i in range(self.device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temp_info.gpu_temps[i] = temp

                    # Check for throttling
                    if temp >= 85.0:
                        temp_info.throttling = True
                except Exception:
                    continue

            return temp_info if temp_info.gpu_temps else None

        except Exception:
            return None
