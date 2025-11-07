"""System specifications detector - one-time detection at startup."""

import platform
from typing import List, Dict, Optional
from loguru import logger

from .base import (
    CPUMetrics,
    MemoryMetrics,
    StorageMetrics,
    GPUMetrics,
    TemperatureInfo,
)
from .backends import (
    PSUtilBackend,
    NvidiaBackend,
    LinuxBackend,
    MacOSBackend,
    WindowsBackend,
    ROCmBackend,
    EdgeDeviceBackend,
)


class SystemSpecsDetector:
    """Detect system specifications at startup."""

    def __init__(self):
        """Initialize detector with all available backends."""
        self.platform = platform.system().lower()
        self.backends = self._initialize_backends()

    def _initialize_backends(self) -> Dict:
        """Initialize and check availability of all backends."""
        all_backends = {
            'psutil': PSUtilBackend(),
            'nvidia': NvidiaBackend(),
            'linux': LinuxBackend(),
            'macos': MacOSBackend(),
            'windows': WindowsBackend(),
            'rocm': ROCmBackend(),
            'edge': EdgeDeviceBackend(),
        }

        # Filter to only available backends
        available_backends = {
            name: backend
            for name, backend in all_backends.items()
            if backend.is_available()
        }

        logger.info(f"Available backends: {list(available_backends.keys())}")
        return available_backends

    def detect_all(self) -> Dict:
        """Detect all system specifications.

        Returns:
            Dictionary with complete system specs
        """
        specs = {
            'platform': self.platform,
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
        }

        # CPU metrics
        cpu_metrics = self._detect_cpu()
        if cpu_metrics:
            specs['cpu'] = cpu_metrics

        # Memory metrics
        memory_metrics = self._detect_memory()
        if memory_metrics:
            specs['memory'] = memory_metrics

        # Storage metrics
        storage_metrics = self._detect_storage()
        if storage_metrics:
            specs['storage'] = storage_metrics

        # GPU metrics
        gpu_metrics = self._detect_gpus()
        if gpu_metrics:
            specs['gpus'] = gpu_metrics

        # Temperature info
        temp_info = self._detect_temperature()
        if temp_info:
            specs['temperature'] = temp_info

        # Battery info (laptops)
        battery = self._detect_battery()
        if battery:
            specs['battery'] = battery

        # Platform-specific info
        platform_info = self._detect_platform_specific()
        if platform_info:
            specs['platform_info'] = platform_info

        return specs

    def _detect_cpu(self) -> Optional[Dict]:
        """Detect CPU specifications."""
        # Try psutil first (baseline)
        if 'psutil' in self.backends:
            cpu_metrics = self.backends['psutil'].get_cpu_metrics()
            if cpu_metrics:
                result = {
                    'architecture': cpu_metrics.architecture,
                    'cores_physical': cpu_metrics.cores_physical,
                    'cores_logical': cpu_metrics.cores_logical,
                    'usage_percent': cpu_metrics.usage_percent,
                }

                if cpu_metrics.frequency:
                    result['frequency'] = {
                        'current_ghz': cpu_metrics.frequency.cpu_current_ghz,
                        'min_ghz': cpu_metrics.frequency.cpu_min_ghz,
                        'max_ghz': cpu_metrics.frequency.cpu_max_ghz,
                    }

                if cpu_metrics.load_avg_1min:
                    result['load_avg'] = {
                        '1min': cpu_metrics.load_avg_1min,
                        '5min': cpu_metrics.load_avg_5min,
                        '15min': cpu_metrics.load_avg_15min,
                    }

                return result

        return None

    def _detect_memory(self) -> Optional[Dict]:
        """Detect memory specifications."""
        if 'psutil' in self.backends:
            mem_metrics = self.backends['psutil'].get_memory_metrics()
            if mem_metrics:
                return {
                    'total_ram_gb': mem_metrics.total_ram_gb,
                    'available_ram_gb': mem_metrics.available_ram_gb,
                    'used_ram_gb': mem_metrics.used_ram_gb,
                    'percent_used': mem_metrics.percent_used,
                    'total_swap_gb': mem_metrics.total_swap_gb,
                    'used_swap_gb': mem_metrics.used_swap_gb,
                }

        return None

    def _detect_storage(self) -> Optional[Dict]:
        """Detect storage specifications."""
        if 'psutil' in self.backends:
            storage_metrics = self.backends['psutil'].get_storage_metrics()
            if storage_metrics and storage_metrics.devices:
                devices = []
                for device in storage_metrics.devices:
                    devices.append({
                        'device': device.device,
                        'mount_point': device.mount_point,
                        'total_gb': device.total_gb,
                        'used_gb': device.used_gb,
                        'free_gb': device.free_gb,
                        'percent_used': device.percent_used,
                        'filesystem': device.filesystem,
                    })

                return {
                    'devices': devices,
                    'total_read_mb': storage_metrics.total_read_mb,
                    'total_write_mb': storage_metrics.total_write_mb,
                }

        return None

    def _detect_gpus(self) -> Optional[List[Dict]]:
        """Detect all GPUs."""
        gpus = []

        # Try NVIDIA backend
        if 'nvidia' in self.backends:
            nvidia_gpus = self.backends['nvidia'].get_all_gpu_metrics()
            for gpu in nvidia_gpus:
                gpus.append(self._gpu_metrics_to_dict(gpu, 'NVIDIA'))

        # Try ROCm backend
        if 'rocm' in self.backends:
            rocm_gpus = self.backends['rocm'].get_all_gpu_metrics()
            for gpu in rocm_gpus:
                gpus.append(self._gpu_metrics_to_dict(gpu, 'AMD'))

        # Try macOS Metal
        if 'macos' in self.backends:
            metal_gpu = self.backends['macos'].get_gpu_metrics()
            if metal_gpu:
                gpus.append(self._gpu_metrics_to_dict(metal_gpu, 'Apple'))

        # Try edge devices (Jetson)
        if 'edge' in self.backends:
            edge_gpu = self.backends['edge'].get_gpu_metrics()
            if edge_gpu:
                gpus.append(self._gpu_metrics_to_dict(edge_gpu, 'Jetson'))

        return gpus if gpus else None

    def _gpu_metrics_to_dict(self, gpu: GPUMetrics, vendor: str) -> Dict:
        """Convert GPU metrics to dictionary."""
        result = {
            'vendor': vendor,
            'device_id': gpu.device_id,
            'device_name': gpu.device_name,
            'total_memory_gb': gpu.total_memory_gb,
            'used_memory_gb': gpu.used_memory_gb,
            'free_memory_gb': gpu.free_memory_gb,
        }

        if gpu.temperature_celsius:
            result['temperature_celsius'] = gpu.temperature_celsius

        if gpu.power_draw_watts:
            result['power_draw_watts'] = gpu.power_draw_watts
            result['power_limit_watts'] = gpu.power_limit_watts

        if gpu.compute_capability:
            result['compute_capability'] = gpu.compute_capability

        if gpu.frequency:
            result['frequency'] = {
                'core_mhz': gpu.frequency.gpu_core_mhz,
                'memory_mhz': gpu.frequency.gpu_memory_mhz,
            }

        return result

    def _detect_temperature(self) -> Optional[Dict]:
        """Detect temperature information."""
        temp_info = None

        # Try platform-specific backends first
        for backend_name in ['linux', 'macos', 'windows', 'edge']:
            if backend_name in self.backends:
                temp = self.backends[backend_name].get_temperature_info()
                if temp:
                    temp_info = temp
                    break

        # Try psutil as fallback
        if not temp_info and 'psutil' in self.backends:
            temp_info = self.backends['psutil'].get_temperature_info()

        # Add GPU temperatures from NVIDIA backend
        if temp_info and 'nvidia' in self.backends:
            nvidia_temp = self.backends['nvidia'].get_temperature_info()
            if nvidia_temp and nvidia_temp.gpu_temps:
                if not temp_info.gpu_temps:
                    temp_info.gpu_temps = {}
                temp_info.gpu_temps.update(nvidia_temp.gpu_temps)

        if temp_info:
            result = {}
            if temp_info.cpu_temp_celsius:
                result['cpu_celsius'] = temp_info.cpu_temp_celsius
            if temp_info.cpu_package_temp:
                result['cpu_package_celsius'] = temp_info.cpu_package_temp
            if temp_info.gpu_temps:
                result['gpu_celsius'] = temp_info.gpu_temps
            if temp_info.throttling:
                result['throttling'] = True

            return result if result else None

        return None

    def _detect_battery(self) -> Optional[Dict]:
        """Detect battery information (laptops)."""
        if 'psutil' in self.backends:
            return self.backends['psutil'].get_battery_info()
        return None

    def _detect_platform_specific(self) -> Optional[Dict]:
        """Detect platform-specific information."""
        info = {}

        # Edge device info
        if 'edge' in self.backends:
            backend = self.backends['edge']
            if backend.is_raspberry_pi:
                rpi_info = backend.get_raspberry_pi_info()
                if rpi_info:
                    info['raspberry_pi'] = rpi_info
            elif backend.is_jetson:
                jetson_info = backend.get_jetson_info()
                if jetson_info:
                    info['jetson'] = jetson_info

        # Linux CPU info
        if 'linux' in self.backends:
            cpu_info = self.backends['linux'].get_cpu_model_info()
            if cpu_info:
                info['linux_cpu'] = cpu_info

        # macOS system info
        if 'macos' in self.backends:
            macos_info = self.backends['macos'].get_system_info()
            if macos_info:
                info['macos'] = macos_info

        # Windows system info
        if 'windows' in self.backends:
            win_info = self.backends['windows'].get_system_info()
            if win_info:
                info['windows'] = win_info

        return info if info else None

    def get_summary(self, specs: Dict) -> str:
        """Generate a human-readable summary of specs.

        Args:
            specs: System specifications dictionary

        Returns:
            Summary string
        """
        lines = []
        lines.append(f"Platform: {specs.get('platform', 'unknown').title()}")
        lines.append(f"Architecture: {specs.get('architecture', 'unknown')}")

        if 'cpu' in specs:
            cpu = specs['cpu']
            lines.append(f"CPU: {cpu.get('cores_physical', 0)} physical cores, "
                        f"{cpu.get('cores_logical', 0)} logical cores")
            if 'frequency' in cpu:
                freq = cpu['frequency']
                lines.append(f"CPU Frequency: {freq.get('current_ghz', 0):.2f} GHz")

        if 'memory' in specs:
            mem = specs['memory']
            lines.append(f"RAM: {mem.get('total_ram_gb', 0):.1f} GB total, "
                        f"{mem.get('available_ram_gb', 0):.1f} GB available")

        if 'gpus' in specs:
            for i, gpu in enumerate(specs['gpus']):
                lines.append(f"GPU {i}: {gpu.get('device_name', 'unknown')} "
                           f"({gpu.get('total_memory_gb', 0):.1f} GB)")

        if 'temperature' in specs:
            temp = specs['temperature']
            if 'cpu_celsius' in temp:
                lines.append(f"CPU Temperature: {temp['cpu_celsius']:.1f}°C")

        return '\n'.join(lines)
