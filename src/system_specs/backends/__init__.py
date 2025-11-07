"""Platform-specific backends for system metrics collection."""

from .base import MetricsBackend
from .psutil_backend import PSUtilBackend
from .nvidia_backend import NvidiaBackend
from .linux_backend import LinuxBackend
from .macos_backend import MacOSBackend
from .windows_backend import WindowsBackend
from .rocm_backend import ROCmBackend
from .edge_backend import EdgeDeviceBackend

__all__ = [
    "MetricsBackend",
    "PSUtilBackend",
    "NvidiaBackend",
    "LinuxBackend",
    "MacOSBackend",
    "WindowsBackend",
    "ROCmBackend",
    "EdgeDeviceBackend",
]
