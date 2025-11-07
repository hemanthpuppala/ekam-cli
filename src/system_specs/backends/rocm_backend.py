"""AMD ROCm GPU backend."""

import subprocess
from typing import Optional, List
from .base import MetricsBackend
from ..base import GPUMetrics, TemperatureInfo


class ROCmBackend(MetricsBackend):
    """AMD ROCm GPU metrics backend using rocm-smi."""

    def _check_availability(self) -> bool:
        """Check if ROCm is available."""
        try:
            # Check if rocm-smi is available
            result = subprocess.run(['rocm-smi', '--showid'],
                                  capture_output=True,
                                  text=True,
                                  timeout=2)
            return result.returncode == 0
        except Exception:
            return False

    def get_gpu_metrics(self, device_id: int = 0) -> Optional[GPUMetrics]:
        """Get metrics for AMD GPU using rocm-smi."""
        if not self.available:
            return None

        try:
            # Get GPU info
            result = subprocess.run(
                ['rocm-smi', '--showid', '--showmeminfo', 'vram', '--showuse', '--showtemp', '--json'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode != 0:
                return None

            # Parse JSON output
            import json
            data = json.loads(result.stdout)

            # Extract metrics for device_id
            # Note: rocm-smi JSON format may vary by version
            # This is a simplified implementation

            metrics = GPUMetrics(
                device_id=device_id,
                device_name="AMD GPU (ROCm)",
            )

            return metrics

        except Exception:
            return None

    def get_all_gpu_metrics(self) -> List[GPUMetrics]:
        """Get metrics for all AMD GPUs."""
        if not self.available:
            return []

        # Would need to parse rocm-smi output for all devices
        return []

    def get_temperature_info(self) -> Optional[TemperatureInfo]:
        """Get temperature info for AMD GPUs."""
        if not self.available:
            return None

        try:
            result = subprocess.run(
                ['rocm-smi', '--showtemp', '--json'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode != 0:
                return None

            # Parse temperature data
            # This is a simplified implementation
            temp_info = TemperatureInfo()

            return temp_info

        except Exception:
            return None
