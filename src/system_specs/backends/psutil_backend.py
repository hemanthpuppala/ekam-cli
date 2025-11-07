"""Cross-platform metrics backend using psutil."""

import platform
from typing import Optional, List, Dict
from .base import MetricsBackend
from ..base import (
    CPUMetrics,
    MemoryMetrics,
    StorageMetrics,
    StorageInfo,
    NetworkMetrics,
    TemperatureInfo,
    FrequencyInfo,
    ProcessMetric,
)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PSUtilBackend(MetricsBackend):
    """Cross-platform backend using psutil for baseline metrics."""

    def _check_availability(self) -> bool:
        """Check if psutil is available."""
        return PSUTIL_AVAILABLE

    def get_cpu_metrics(self) -> Optional[CPUMetrics]:
        """Get CPU metrics using psutil."""
        if not self.available:
            return None

        try:
            # Basic CPU info
            metrics = CPUMetrics(
                architecture=platform.machine(),
                cores_physical=psutil.cpu_count(logical=False) or 1,
                cores_logical=psutil.cpu_count(logical=True) or 1,
                usage_percent=psutil.cpu_percent(interval=0.1),
                per_core_usage=psutil.cpu_percent(interval=0.1, percpu=True),
            )

            # Frequency
            freq = psutil.cpu_freq()
            if freq:
                freq_info = FrequencyInfo(
                    cpu_current_ghz=freq.current / 1000.0 if freq.current else 0.0,
                    cpu_min_ghz=freq.min / 1000.0 if freq.min else 0.0,
                    cpu_max_ghz=freq.max / 1000.0 if freq.max else 0.0,
                )
                metrics.frequency = freq_info

                # Per-core frequencies if available
                try:
                    per_core_freq = psutil.cpu_freq(percpu=True)
                    if per_core_freq:
                        freq_info.cpu_per_core_ghz = [
                            f.current / 1000.0 if f.current else 0.0
                            for f in per_core_freq
                        ]
                except (AttributeError, NotImplementedError):
                    pass

            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                metrics.load_avg_1min = load_avg[0]
                metrics.load_avg_5min = load_avg[1]
                metrics.load_avg_15min = load_avg[2]
            except (AttributeError, OSError):
                pass

            return metrics
        except Exception as e:
            return None

    def get_memory_metrics(self) -> Optional[MemoryMetrics]:
        """Get memory metrics using psutil."""
        if not self.available:
            return None

        try:
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()

            metrics = MemoryMetrics(
                total_ram_gb=vm.total / (1024 ** 3),
                used_ram_gb=vm.used / (1024 ** 3),
                available_ram_gb=vm.available / (1024 ** 3),
                percent_used=vm.percent,
                total_swap_gb=swap.total / (1024 ** 3),
                used_swap_gb=swap.used / (1024 ** 3),
                swap_percent_used=swap.percent,
            )

            # Additional fields on Linux
            if hasattr(vm, 'buffers'):
                metrics.buffers_gb = vm.buffers / (1024 ** 3)
            if hasattr(vm, 'cached'):
                metrics.cached_gb = vm.cached / (1024 ** 3)
            if hasattr(vm, 'shared'):
                metrics.shared_gb = vm.shared / (1024 ** 3)

            return metrics
        except Exception:
            return None

    def get_storage_metrics(self) -> Optional[StorageMetrics]:
        """Get storage metrics using psutil."""
        if not self.available:
            return None

        try:
            devices = []
            partitions = psutil.disk_partitions()

            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    device_info = StorageInfo(
                        device=partition.device,
                        mount_point=partition.mountpoint,
                        total_gb=usage.total / (1024 ** 3),
                        used_gb=usage.used / (1024 ** 3),
                        free_gb=usage.free / (1024 ** 3),
                        percent_used=usage.percent,
                        filesystem=partition.fstype,
                    )
                    devices.append(device_info)
                except (PermissionError, OSError):
                    continue

            # Disk I/O counters
            metrics = StorageMetrics(devices=devices)
            try:
                io_counters = psutil.disk_io_counters()
                if io_counters:
                    metrics.total_read_mb = io_counters.read_bytes / (1024 ** 2)
                    metrics.total_write_mb = io_counters.write_bytes / (1024 ** 2)
                    metrics.total_read_count = io_counters.read_count
                    metrics.total_write_count = io_counters.write_count
            except (AttributeError, RuntimeError):
                pass

            return metrics
        except Exception:
            return None

    def get_temperature_info(self) -> Optional[TemperatureInfo]:
        """Get temperature information using psutil (Linux/Mac only)."""
        if not self.available:
            return None

        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return None

            temp_info = TemperatureInfo()

            # CPU temperature
            # Try different sensor names based on platform
            cpu_sensor_names = ['coretemp', 'cpu_thermal', 'cpu-thermal', 'k10temp', 'zenpower']

            for sensor_name in cpu_sensor_names:
                if sensor_name in temps:
                    entries = temps[sensor_name]
                    for entry in entries:
                        if 'package' in entry.label.lower() or 'tctl' in entry.label.lower():
                            temp_info.cpu_package_temp = entry.current
                        elif not temp_info.cpu_temp_celsius:
                            temp_info.cpu_temp_celsius = entry.current

                        # Check for high temperature (throttling indicator)
                        if entry.current >= 85.0:
                            temp_info.throttling = True

            # If we still don't have CPU temp, try first available
            if not temp_info.cpu_temp_celsius and temps:
                first_sensor = list(temps.values())[0]
                if first_sensor:
                    temp_info.cpu_temp_celsius = first_sensor[0].current

            return temp_info if temp_info.cpu_temp_celsius else None

        except (AttributeError, OSError):
            # sensors_temperatures not available on this platform
            return None

    def get_frequency_info(self) -> Optional[FrequencyInfo]:
        """Get frequency information using psutil."""
        if not self.available:
            return None

        try:
            freq = psutil.cpu_freq()
            if not freq:
                return None

            freq_info = FrequencyInfo(
                cpu_current_ghz=freq.current / 1000.0 if freq.current else 0.0,
                cpu_min_ghz=freq.min / 1000.0 if freq.min else 0.0,
                cpu_max_ghz=freq.max / 1000.0 if freq.max else 0.0,
            )

            # Per-core frequencies if available
            try:
                per_core_freq = psutil.cpu_freq(percpu=True)
                if per_core_freq:
                    freq_info.cpu_per_core_ghz = [
                        f.current / 1000.0 if f.current else 0.0
                        for f in per_core_freq
                    ]
            except (AttributeError, NotImplementedError):
                pass

            return freq_info
        except Exception:
            return None

    def get_network_metrics(self) -> Optional[NetworkMetrics]:
        """Get network metrics using psutil."""
        if not self.available:
            return None

        try:
            net_io = psutil.net_io_counters()
            if not net_io:
                return None

            return NetworkMetrics(
                bytes_sent=net_io.bytes_sent,
                bytes_recv=net_io.bytes_recv,
                packets_sent=net_io.packets_sent,
                packets_recv=net_io.packets_recv,
                errors_in=net_io.errin,
                errors_out=net_io.errout,
                drops_in=net_io.dropin,
                drops_out=net_io.dropout,
            )
        except Exception:
            return None

    def get_top_processes(self, limit: int = 10) -> List[ProcessMetric]:
        """Get top processes by CPU and memory usage."""
        if not self.available:
            return []

        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'status']):
                try:
                    info = proc.info
                    memory_mb = info['memory_info'].rss / (1024 ** 2) if info.get('memory_info') else 0.0
                    processes.append(ProcessMetric(
                        pid=info['pid'],
                        name=info['name'] or 'unknown',
                        cpu_percent=info.get('cpu_percent', 0.0) or 0.0,
                        memory_mb=memory_mb,
                        status=info.get('status', 'unknown'),
                    ))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage and take top N
            processes.sort(key=lambda p: p.cpu_percent, reverse=True)
            return processes[:limit]
        except Exception:
            return []

    def get_battery_info(self) -> Optional[Dict]:
        """Get battery information."""
        if not self.available:
            return None

        try:
            battery = psutil.sensors_battery()
            if not battery:
                return None

            return {
                'percent': battery.percent,
                'plugged': battery.power_plugged,
                'seconds_left': battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None,
            }
        except (AttributeError, OSError):
            return None

    def get_system_uptime(self) -> Optional[float]:
        """Get system uptime in seconds."""
        if not self.available:
            return None

        try:
            import time
            boot_time = psutil.boot_time()
            return time.time() - boot_time
        except Exception:
            return None
