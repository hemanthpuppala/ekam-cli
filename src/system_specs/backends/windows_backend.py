"""Windows-specific backend using WMI and platform APIs."""

import platform
from typing import Optional
from .base import MetricsBackend
from ..base import TemperatureInfo


class WindowsBackend(MetricsBackend):
    """Windows-specific backend."""

    def __init__(self):
        """Initialize Windows backend."""
        self.wmi_available = False
        super().__init__()

    def _check_availability(self) -> bool:
        """Check if running on Windows."""
        if platform.system().lower() != "windows":
            return False

        # Try to import wmi
        try:
            import wmi
            self.wmi_available = True
        except ImportError:
            pass

        return True

    def get_temperature_info(self) -> Optional[TemperatureInfo]:
        """Get temperature info on Windows using WMI."""
        if not self.available or not self.wmi_available:
            return None

        try:
            import wmi
            w = wmi.WMI(namespace="root\\WMI")

            temp_info = TemperatureInfo()

            # Try MSAcpi_ThermalZoneTemperature
            temperature_info = w.MSAcpi_ThermalZoneTemperature()
            if temperature_info:
                for temp in temperature_info:
                    # Temperature is in tenths of Kelvin
                    temp_kelvin = temp.CurrentTemperature / 10.0
                    temp_celsius = temp_kelvin - 273.15

                    if not temp_info.cpu_temp_celsius:
                        temp_info.cpu_temp_celsius = temp_celsius

                    if temp_celsius >= 85.0:
                        temp_info.throttling = True

            return temp_info if temp_info.cpu_temp_celsius else None

        except Exception:
            return None

    def get_system_info(self) -> Optional[dict]:
        """Get system info using WMI."""
        if not self.available or not self.wmi_available:
            return None

        try:
            import wmi
            w = wmi.WMI()

            info = {}

            # CPU info
            for processor in w.Win32_Processor():
                info['cpu_name'] = processor.Name
                info['cpu_cores'] = processor.NumberOfCores
                info['cpu_threads'] = processor.NumberOfLogicalProcessors
                break

            # Memory info
            for mem in w.Win32_PhysicalMemory():
                if 'total_memory' not in info:
                    info['total_memory'] = 0
                info['total_memory'] += int(mem.Capacity)

            return info

        except Exception:
            return None
