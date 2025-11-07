"""macOS-specific backend using sysctl and IOKit."""

import platform
import subprocess
from typing import Optional
from .base import MetricsBackend
from ..base import TemperatureInfo, GPUMetrics


class MacOSBackend(MetricsBackend):
    """macOS-specific backend."""

    def _check_availability(self) -> bool:
        """Check if running on macOS."""
        return platform.system().lower() == "darwin"

    def get_temperature_info(self) -> Optional[TemperatureInfo]:
        """Get temperature info on macOS (limited without third-party tools)."""
        if not self.available:
            return None

        # Note: macOS doesn't expose temperature sensors easily
        # Would require IOKit access or third-party tools like osx-cpu-temp
        # For now, return None - can be extended later

        try:
            # Try powermetrics (requires sudo, so likely won't work)
            result = subprocess.run(
                ['powermetrics', '--samplers', 'thermal', '-n', '1', '-i', '1000'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # Parse output for temperature
                # This is complex and requires parsing
                pass
        except Exception:
            pass

        return None

    def get_gpu_metrics(self, device_id: int = 0) -> Optional[GPUMetrics]:
        """Get Metal GPU metrics (Apple Silicon)."""
        if not self.available:
            return None

        try:
            import torch
            import psutil

            if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
                return None

            # Apple Silicon uses unified memory
            # Estimate total GPU memory from system RAM (typically 75-95% of system RAM is available for GPU)
            total_system_ram = psutil.virtual_memory().total / (1024 ** 3)

            metrics = GPUMetrics(
                device_id=device_id,
                device_name="Apple Silicon GPU (Metal)",
                total_memory_gb=total_system_ram * 0.75,  # Conservative estimate
            )

            # Try to get allocated memory stats if available
            try:
                if hasattr(torch.mps, 'current_allocated_memory'):
                    allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
                    metrics.used_memory_gb = allocated
                    metrics.free_memory_gb = metrics.total_memory_gb - allocated

                    # Calculate percentage if total is known
                    if metrics.total_memory_gb > 0:
                        metrics.memory_percent = (allocated / metrics.total_memory_gb) * 100
            except Exception:
                pass

            return metrics

        except ImportError:
            return None

    def get_system_info(self) -> Optional[dict]:
        """Get system info using sysctl."""
        if not self.available:
            return None

        try:
            info = {}

            # CPU info
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                info['cpu_brand'] = result.stdout.strip()

            # Memory info
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                info['memory_bytes'] = int(result.stdout.strip())

            return info

        except Exception:
            return None
