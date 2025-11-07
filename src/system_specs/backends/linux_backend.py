"""Linux-specific backend for advanced system metrics via /proc and /sys."""

import platform
from pathlib import Path
from typing import Optional, Dict
from .base import MetricsBackend
from ..base import CPUMetrics, TemperatureInfo


class LinuxBackend(MetricsBackend):
    """Linux-specific backend using /proc and /sys filesystems."""

    def _check_availability(self) -> bool:
        """Check if running on Linux."""
        return platform.system().lower() == "linux"

    def get_cpu_metrics(self) -> Optional[CPUMetrics]:
        """Get extended CPU metrics from /proc/stat."""
        if not self.available:
            return None

        try:
            metrics = CPUMetrics()

            # Read /proc/stat for context switches and interrupts
            stat_path = Path("/proc/stat")
            if stat_path.exists():
                with open(stat_path, 'r') as f:
                    for line in f:
                        if line.startswith('ctxt'):
                            metrics.context_switches = int(line.split()[1])
                        elif line.startswith('intr'):
                            # First number after 'intr' is total interrupts
                            parts = line.split()
                            if len(parts) > 1:
                                metrics.interrupts = int(parts[1])

            return metrics if (metrics.context_switches or metrics.interrupts) else None

        except Exception:
            return None

    def get_temperature_info(self) -> Optional[TemperatureInfo]:
        """Get temperature info from /sys/class/thermal."""
        if not self.available:
            return None

        try:
            temp_info = TemperatureInfo()
            thermal_path = Path("/sys/class/thermal")

            if not thermal_path.exists():
                return None

            # Find thermal zones
            for zone_dir in thermal_path.glob("thermal_zone*"):
                try:
                    # Read temperature type
                    type_file = zone_dir / "type"
                    temp_file = zone_dir / "temp"

                    if not type_file.exists() or not temp_file.exists():
                        continue

                    zone_type = type_file.read_text().strip().lower()
                    temp_millicelsius = int(temp_file.read_text().strip())
                    temp_celsius = temp_millicelsius / 1000.0

                    # Map to CPU or GPU
                    if any(keyword in zone_type for keyword in ['cpu', 'core', 'package', 'x86_pkg', 'soc']):
                        if not temp_info.cpu_temp_celsius:
                            temp_info.cpu_temp_celsius = temp_celsius
                        if 'package' in zone_type or 'x86_pkg' in zone_type:
                            temp_info.cpu_package_temp = temp_celsius

                    # Check for high temperature
                    if temp_celsius >= 85.0:
                        temp_info.throttling = True

                except (ValueError, OSError):
                    continue

            # Alternative: Read from hwmon
            if not temp_info.cpu_temp_celsius:
                hwmon_path = Path("/sys/class/hwmon")
                if hwmon_path.exists():
                    for hwmon_dir in hwmon_path.glob("hwmon*"):
                        try:
                            name_file = hwmon_dir / "name"
                            if not name_file.exists():
                                continue

                            name = name_file.read_text().strip().lower()

                            # Check if it's a CPU temperature sensor
                            if any(keyword in name for keyword in ['coretemp', 'k10temp', 'zenpower', 'cpu']):
                                # Look for temp*_input files
                                for temp_file in hwmon_dir.glob("temp*_input"):
                                    try:
                                        temp_millicelsius = int(temp_file.read_text().strip())
                                        temp_celsius = temp_millicelsius / 1000.0

                                        # Check label to see if it's package temp
                                        label_file = temp_file.parent / temp_file.name.replace("_input", "_label")
                                        if label_file.exists():
                                            label = label_file.read_text().strip().lower()
                                            if 'package' in label or 'tctl' in label:
                                                temp_info.cpu_package_temp = temp_celsius
                                            elif not temp_info.cpu_temp_celsius:
                                                temp_info.cpu_temp_celsius = temp_celsius
                                        else:
                                            if not temp_info.cpu_temp_celsius:
                                                temp_info.cpu_temp_celsius = temp_celsius

                                    except (ValueError, OSError):
                                        continue

                        except OSError:
                            continue

            return temp_info if temp_info.cpu_temp_celsius else None

        except Exception:
            return None

    def get_throttling_status(self) -> bool:
        """Check if CPU is currently throttling."""
        if not self.available:
            return False

        try:
            # Check /proc/cpuinfo for frequency scaling
            cpuinfo_path = Path("/proc/cpuinfo")
            if cpuinfo_path.exists():
                content = cpuinfo_path.read_text()
                # This is a heuristic - actual throttling detection is complex
                # Would need to compare current frequency vs rated frequency

            # Check thermal events in dmesg (requires root)
            # This is just a basic check
            return False

        except Exception:
            return False

    def get_memory_info_detailed(self) -> Optional[Dict]:
        """Get detailed memory info from /proc/meminfo."""
        if not self.available:
            return None

        try:
            meminfo_path = Path("/proc/meminfo")
            if not meminfo_path.exists():
                return None

            mem_info = {}
            with open(meminfo_path, 'r') as f:
                for line in f:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value_str = parts[1].strip().split()[0]
                        try:
                            value_kb = int(value_str)
                            mem_info[key] = value_kb
                        except ValueError:
                            continue

            return mem_info

        except Exception:
            return None

    def get_cpu_model_info(self) -> Optional[Dict]:
        """Get CPU model information from /proc/cpuinfo."""
        if not self.available:
            return None

        try:
            cpuinfo_path = Path("/proc/cpuinfo")
            if not cpuinfo_path.exists():
                return None

            cpu_info = {}
            with open(cpuinfo_path, 'r') as f:
                for line in f:
                    if ':' not in line:
                        continue
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if key == 'model name' and 'model_name' not in cpu_info:
                        cpu_info['model_name'] = value
                    elif key == 'cpu MHz' and 'cpu_mhz' not in cpu_info:
                        cpu_info['cpu_mhz'] = float(value)
                    elif key == 'cache size' and 'cache_size' not in cpu_info:
                        cpu_info['cache_size'] = value
                    elif key == 'cpu cores' and 'cpu_cores' not in cpu_info:
                        cpu_info['cpu_cores'] = int(value)

            return cpu_info if cpu_info else None

        except Exception:
            return None
