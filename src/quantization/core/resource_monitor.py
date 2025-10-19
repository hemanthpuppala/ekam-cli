"""Resource monitoring utilities for quantization tasks."""

import os
import time
from collections import deque
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ResourceMonitor:
    """Monitor system resources during quantization with minimal overhead."""

    def __init__(self, track_gpu: bool = False):
        """Initialize resource monitor.

        Args:
            track_gpu: Whether to monitor GPU resources
        """
        self.track_gpu = track_gpu and TORCH_AVAILABLE
        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None

        # History for calculating trends (last 10 measurements)
        self.cpu_history = deque(maxlen=10)
        self.memory_history = deque(maxlen=10)

        # Baseline memory (to track delta during quantization)
        self.baseline_memory_mb = 0.0
        self.baseline_system_memory_mb = 0.0
        if self.process:
            try:
                # Get initial memory footprint
                self.baseline_memory_mb = self.process.memory_info().rss / (1024 ** 2)
            except:
                pass

        # Track system-wide memory for more accurate reporting
        if PSUTIL_AVAILABLE:
            try:
                vm = psutil.virtual_memory()
                self.baseline_system_memory_mb = vm.used / (1024 ** 2)
            except:
                pass

        # Timing for throughput calculation
        self.last_measurement_time = time.time()
        self.last_bytes_written = 0

    def get_cpu_percent(self) -> float:
        """Get current CPU usage percentage.

        Returns:
            CPU usage as percentage (0-100)
        """
        if not self.process:
            return 0.0

        try:
            cpu = self.process.cpu_percent(interval=0.1)
            self.cpu_history.append(cpu)
            return cpu
        except Exception:
            return 0.0

    def get_memory_usage(self) -> tuple[float, float]:
        """Get memory usage in MB and percentage (including children).

        Returns:
            (memory_mb, memory_percent) tuple
        """
        if not self.process:
            return 0.0, 0.0

        try:
            # Get memory for main process
            mem_info = self.process.memory_info()
            total_memory_bytes = mem_info.rss

            # Include all child processes recursively
            try:
                children = self.process.children(recursive=True)
                for child in children:
                    if child.is_running():
                        try:
                            total_memory_bytes += child.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except Exception:
                pass  # Continue with just parent process memory

            memory_mb = total_memory_bytes / (1024 ** 2)

            # Calculate percentage based on total system memory
            total_system_memory = psutil.virtual_memory().total
            memory_percent = (total_memory_bytes / total_system_memory) * 100

            self.memory_history.append(memory_mb)

            return memory_mb, memory_percent
        except Exception:
            return 0.0, 0.0

    def get_gpu_memory(self) -> Optional[tuple[float, float]]:
        """Get GPU memory usage in MB and percentage.

        Returns:
            (gpu_memory_mb, gpu_memory_percent) tuple or None
        """
        if not self.track_gpu:
            return None

        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

                # Use reserved memory (more accurate for monitoring)
                percent = (reserved / total) * 100 if total > 0 else 0.0
                return reserved, percent
            elif torch.backends.mps.is_available():
                # MPS (Apple Silicon) - use allocated memory
                allocated = torch.mps.current_allocated_memory() / (1024 ** 2)
                # Note: MPS doesn't expose total memory, so percentage is approximate
                return allocated, 0.0
        except Exception:
            pass

        return None

    def get_cpu_sparkline(self) -> str:
        """Get a sparkline representation of CPU usage.

        Returns:
            Unicode sparkline string
        """
        if not self.cpu_history or len(self.cpu_history) < 2:
            return "─" * 8

        # Normalize to 0-7 range for sparkline characters
        max_cpu = max(self.cpu_history) if self.cpu_history else 1.0
        if max_cpu == 0:
            max_cpu = 1.0

        sparkline_chars = "▁▂▃▄▅▆▇█"
        sparkline = ""

        for cpu in list(self.cpu_history)[-8:]:  # Last 8 measurements
            index = int((cpu / max_cpu) * 7)
            index = min(7, max(0, index))
            sparkline += sparkline_chars[index]

        return sparkline

    def get_memory_sparkline(self) -> str:
        """Get a sparkline representation of memory usage.

        Returns:
            Unicode sparkline string
        """
        if not self.memory_history or len(self.memory_history) < 2:
            return "─" * 8

        # Normalize to 0-7 range for sparkline characters
        max_mem = max(self.memory_history) if self.memory_history else 1.0
        if max_mem == 0:
            max_mem = 1.0

        sparkline_chars = "▁▂▃▄▅▆▇█"
        sparkline = ""

        for mem in list(self.memory_history)[-8:]:  # Last 8 measurements
            index = int((mem / max_mem) * 7)
            index = min(7, max(0, index))
            sparkline += sparkline_chars[index]

        return sparkline

    def calculate_write_speed(self, bytes_written: int) -> Optional[float]:
        """Calculate disk write speed in MB/s.

        Args:
            bytes_written: Total bytes written so far

        Returns:
            Write speed in MB/s or None
        """
        current_time = time.time()
        time_delta = current_time - self.last_measurement_time

        if time_delta < 0.1:  # Too soon to calculate
            return None

        bytes_delta = bytes_written - self.last_bytes_written

        if bytes_delta > 0:
            mbps = (bytes_delta / (1024 ** 2)) / time_delta
            self.last_measurement_time = current_time
            self.last_bytes_written = bytes_written
            return mbps

        return None

    def get_system_memory_delta(self) -> tuple[float, float]:
        """Get system-wide memory change since baseline.

        This is more accurate for large model loading as it captures
        memory-mapped files, shared memory, and all processes.

        Returns:
            (delta_mb, current_used_percent) tuple
        """
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0

        try:
            vm = psutil.virtual_memory()
            current_used_mb = vm.used / (1024 ** 2)
            delta_mb = current_used_mb - self.baseline_system_memory_mb
            return delta_mb, vm.percent
        except Exception:
            return 0.0, 0.0

    def get_all_metrics(self) -> dict:
        """Get all resource metrics at once.

        Returns:
            Dictionary with all current metrics
        """
        cpu = self.get_cpu_percent()
        memory_mb, memory_percent = self.get_memory_usage()
        system_delta_mb, system_percent = self.get_system_memory_delta()
        gpu_metrics = self.get_gpu_memory()

        metrics = {
            "cpu_percent": cpu,
            "cpu_sparkline": self.get_cpu_sparkline(),
            "memory_mb": memory_mb,
            "memory_percent": memory_percent,
            "memory_sparkline": self.get_memory_sparkline(),
            # System-wide memory (more accurate for large models)
            "system_memory_delta_mb": system_delta_mb,
            "system_memory_percent": system_percent,
        }

        if gpu_metrics:
            metrics["gpu_memory_mb"] = gpu_metrics[0]
            metrics["gpu_memory_percent"] = gpu_metrics[1]

        return metrics
