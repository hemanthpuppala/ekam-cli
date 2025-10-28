"""Edge device backend for Raspberry Pi, Jetson, etc."""

from pathlib import Path
from typing import Optional
from .base import MetricsBackend
from ..base import TemperatureInfo, GPUMetrics


class EdgeDeviceBackend(MetricsBackend):
    """Backend for edge devices (Raspberry Pi, Jetson, etc.)."""

    def __init__(self):
        """Initialize edge device backend."""
        self.is_raspberry_pi = False
        self.is_jetson = False
        super().__init__()

    def _check_availability(self) -> bool:
        """Check if running on an edge device."""
        # Check for Raspberry Pi
        model_file = Path("/proc/device-tree/model")
        if model_file.exists():
            try:
                model = model_file.read_text().lower()
                if "raspberry pi" in model:
                    self.is_raspberry_pi = True
                    return True
                elif "jetson" in model:
                    self.is_jetson = True
                    return True
            except Exception:
                pass

        # Check for Jetson
        jetson_files = [
            "/etc/nv_tegra_release",
            "/sys/module/tegra_fuse",
        ]
        for filepath in jetson_files:
            if Path(filepath).exists():
                self.is_jetson = True
                return True

        return False

    def get_temperature_info(self) -> Optional[TemperatureInfo]:
        """Get temperature info for edge devices."""
        if not self.available:
            return None

        temp_info = TemperatureInfo()

        # Raspberry Pi temperature
        if self.is_raspberry_pi:
            try:
                # Read from /sys/class/thermal/thermal_zone0/temp
                temp_file = Path("/sys/class/thermal/thermal_zone0/temp")
                if temp_file.exists():
                    temp_millicelsius = int(temp_file.read_text().strip())
                    temp_celsius = temp_millicelsius / 1000.0
                    temp_info.cpu_temp_celsius = temp_celsius

                    # RPi throttles at 80°C
                    if temp_celsius >= 80.0:
                        temp_info.throttling = True

                return temp_info if temp_info.cpu_temp_celsius else None

            except Exception:
                pass

        # Jetson temperature
        if self.is_jetson:
            try:
                # Jetson has multiple thermal zones
                thermal_path = Path("/sys/class/thermal")
                for zone_dir in thermal_path.glob("thermal_zone*"):
                    try:
                        type_file = zone_dir / "type"
                        temp_file = zone_dir / "temp"

                        if not type_file.exists() or not temp_file.exists():
                            continue

                        zone_type = type_file.read_text().strip().lower()
                        temp_millicelsius = int(temp_file.read_text().strip())
                        temp_celsius = temp_millicelsius / 1000.0

                        # CPU or GPU temperature
                        if 'cpu' in zone_type:
                            temp_info.cpu_temp_celsius = temp_celsius
                        elif 'gpu' in zone_type:
                            temp_info.gpu_temps[0] = temp_celsius

                        # Jetson throttles at 90°C typically
                        if temp_celsius >= 85.0:
                            temp_info.throttling = True

                    except Exception:
                        continue

                return temp_info if (temp_info.cpu_temp_celsius or temp_info.gpu_temps) else None

            except Exception:
                pass

        return None

    def get_gpu_metrics(self, device_id: int = 0) -> Optional[GPUMetrics]:
        """Get GPU metrics for Jetson."""
        if not self.available or not self.is_jetson:
            return None

        try:
            # Jetson GPU metrics would come from tegrastats or similar
            # This is a placeholder - full implementation would parse tegrastats output

            metrics = GPUMetrics(
                device_id=device_id,
                device_name="Jetson GPU",
            )

            return metrics

        except Exception:
            return None

    def get_jetson_info(self) -> Optional[dict]:
        """Get Jetson-specific information."""
        if not self.is_jetson:
            return None

        try:
            info = {}

            # Read Jetson release info
            release_file = Path("/etc/nv_tegra_release")
            if release_file.exists():
                info['jetson_release'] = release_file.read_text().strip()

            # Read model from device tree
            model_file = Path("/proc/device-tree/model")
            if model_file.exists():
                info['model'] = model_file.read_text().strip()

            return info

        except Exception:
            return None

    def get_raspberry_pi_info(self) -> Optional[dict]:
        """Get Raspberry Pi-specific information."""
        if not self.is_raspberry_pi:
            return None

        try:
            info = {}

            # Read model
            model_file = Path("/proc/device-tree/model")
            if model_file.exists():
                info['model'] = model_file.read_text().strip()

            # Read revision
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.exists():
                with open(cpuinfo, 'r') as f:
                    for line in f:
                        if line.startswith('Revision'):
                            info['revision'] = line.split(':')[1].strip()
                        elif line.startswith('Serial'):
                            info['serial'] = line.split(':')[1].strip()

            return info

        except Exception:
            return None
