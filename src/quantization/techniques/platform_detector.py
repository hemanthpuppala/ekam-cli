"""Platform and device capability detection for cross-platform quantization.

Detects hardware capabilities across all platforms:
- Mac (M1/M2/Intel)
- Windows (x86, ARM)
- Linux (x86, ARM64, RISC-V)
- Raspberry Pi (ARM)
- Jetson devices (ARM + CUDA)
- Other edge devices

Provides intelligent backend selection based on available hardware.
"""

import platform
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger


class DeviceType(Enum):
    """Available device types."""
    CPU = "cpu"
    CUDA = "cuda"  # NVIDIA GPU
    METAL = "metal"  # Apple Silicon GPU
    ROCM = "rocm"  # AMD GPU
    OPENCL = "opencl"  # Generic GPU
    VULKAN = "vulkan"  # Cross-platform GPU


class PlatformType(Enum):
    """Platform types."""
    MAC_M1 = "mac_m1"  # Apple Silicon
    MAC_INTEL = "mac_intel"  # Intel Mac
    WINDOWS = "windows"
    LINUX_X86 = "linux_x86"
    LINUX_ARM = "linux_arm"  # Raspberry Pi, Jetson, etc.
    RASPBERRY_PI = "raspberry_pi"
    JETSON = "jetson"
    UNKNOWN = "unknown"


@dataclass
class DeviceCapabilities:
    """Hardware capabilities of the current device."""

    platform: PlatformType
    device_type: DeviceType

    # CPU info
    cpu_arch: str  # x86_64, arm64, aarch64, etc.
    cpu_cores: int

    # Memory
    total_ram_gb: float
    available_ram_gb: float

    # GPU info
    has_gpu: bool
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None

    # Compute capabilities
    supports_cuda: bool = False
    supports_metal: bool = False
    supports_rocm: bool = False
    supports_fp16: bool = True
    supports_int8: bool = False
    supports_int4: bool = False

    # Edge device detection
    is_edge_device: bool = False
    is_raspberry_pi: bool = False
    is_jetson: bool = False

    # Backend availability
    backends: list[str] = None

    def __post_init__(self):
        if self.backends is None:
            self.backends = []


class PlatformDetector:
    """Detects platform and device capabilities."""

    @staticmethod
    def detect_platform() -> PlatformType:
        """Detect the current platform.

        Returns:
            Platform type
        """
        system = platform.system().lower()
        machine = platform.machine().lower()

        # Mac detection
        if system == "darwin":
            # Check if Apple Silicon (M1/M2/M3)
            if machine in ["arm64", "aarch64"]:
                return PlatformType.MAC_M1
            else:
                return PlatformType.MAC_INTEL

        # Windows
        elif system == "windows":
            return PlatformType.WINDOWS

        # Linux variants
        elif system == "linux":
            # Check for Jetson
            if PlatformDetector._is_jetson():
                return PlatformType.JETSON

            # Check for Raspberry Pi
            elif PlatformDetector._is_raspberry_pi():
                return PlatformType.RASPBERRY_PI

            # ARM-based Linux
            elif machine in ["arm64", "aarch64", "armv7l", "armv8"]:
                return PlatformType.LINUX_ARM

            # x86/x64 Linux
            else:
                return PlatformType.LINUX_X86

        return PlatformType.UNKNOWN

    @staticmethod
    def _is_jetson() -> bool:
        """Check if running on NVIDIA Jetson device."""
        # Check for Jetson-specific files
        jetson_files = [
            "/etc/nv_tegra_release",
            "/sys/module/tegra_fuse",
            "/proc/device-tree/model",
        ]

        for filepath in jetson_files:
            if Path(filepath).exists():
                try:
                    if "jetson" in Path(filepath).read_text().lower():
                        return True
                except:
                    pass

        return False

    @staticmethod
    def _is_raspberry_pi() -> bool:
        """Check if running on Raspberry Pi."""
        # Check for Raspberry Pi-specific files
        model_file = Path("/proc/device-tree/model")

        if model_file.exists():
            try:
                model = model_file.read_text().lower()
                return "raspberry pi" in model
            except:
                pass

        # Check /proc/cpuinfo
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            try:
                content = cpuinfo.read_text().lower()
                return "raspberry pi" in content or "bcm2" in content
            except:
                pass

        return False

    @staticmethod
    def detect_device_type() -> DeviceType:
        """Detect the best available device type.

        Returns:
            Primary device type to use
        """
        try:
            import torch

            # Check CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                return DeviceType.CUDA

            # Check Metal (Apple Silicon GPU)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return DeviceType.METAL

            # Check ROCm (AMD GPU)
            if hasattr(torch, "hip") and torch.hip.is_available():
                return DeviceType.ROCM

        except ImportError:
            pass

        # Default to CPU
        return DeviceType.CPU

    @staticmethod
    def get_cpu_info() -> tuple[str, int]:
        """Get CPU architecture and core count.

        Returns:
            (architecture, core_count)
        """
        arch = platform.machine()
        cores = 1

        try:
            import psutil
            cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        except ImportError:
            import os
            cores = os.cpu_count() or 1

        return arch, cores

    @staticmethod
    def get_memory_info() -> tuple[float, float]:
        """Get total and available RAM in GB.

        Returns:
            (total_gb, available_gb)
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            available_gb = mem.available / (1024 ** 3)
            return total_gb, available_gb
        except ImportError:
            # Fallback: assume modest specs for edge devices
            return 4.0, 2.0

    @staticmethod
    def get_gpu_info() -> tuple[bool, Optional[str], Optional[float]]:
        """Get GPU information.

        Returns:
            (has_gpu, gpu_name, gpu_memory_gb)
        """
        try:
            import torch

            # CUDA GPU
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                try:
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                except:
                    gpu_mem = None
                return True, gpu_name, gpu_mem

            # Metal GPU (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Apple Silicon shares memory with CPU
                total_gb, _ = PlatformDetector.get_memory_info()
                return True, "Apple Silicon GPU", total_gb

            # ROCm GPU (AMD)
            if hasattr(torch, "hip") and torch.hip.is_available():
                return True, "AMD GPU (ROCm)", None

        except ImportError:
            pass

        return False, None, None

    @staticmethod
    def detect_quantization_support() -> tuple[bool, bool, bool]:
        """Detect quantization support.

        Returns:
            (supports_fp16, supports_int8, supports_int4)
        """
        supports_fp16 = True  # Always available with PyTorch
        supports_int8 = False
        supports_int4 = False

        try:
            import torch

            # INT8/INT4 support depends on backend
            device_type = PlatformDetector.detect_device_type()

            # Check for bitsandbytes (CUDA-only currently)
            if device_type == DeviceType.CUDA:
                try:
                    import bitsandbytes
                    supports_int8 = True
                    supports_int4 = True
                except ImportError:
                    # CUDA available but bitsandbytes not installed
                    pass

            # Metal and CPU: INT8/INT4 not supported for persistent quantization
            # PyTorch's dynamic quantization is runtime-only and can't be saved
            # elif device_type in [DeviceType.CPU, DeviceType.METAL]:
            #     supports_int8 = False  # Not available (can't save to disk)
            #     supports_int4 = False

        except ImportError:
            pass

        return supports_fp16, supports_int8, supports_int4

    @staticmethod
    def detect_available_backends() -> list[str]:
        """Detect available quantization backends.

        Returns:
            List of available backend names
        """
        backends = []

        # Always have CPU
        backends.append("cpu")

        try:
            import torch

            if torch.cuda.is_available():
                backends.append("cuda")

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                backends.append("mps")  # Metal Performance Shaders

            if hasattr(torch, "hip") and torch.hip.is_available():
                backends.append("rocm")

        except ImportError:
            pass

        # Check for optional backends
        try:
            import bitsandbytes
            backends.append("bitsandbytes")
        except ImportError:
            pass

        try:
            import optimum
            backends.append("optimum")
        except ImportError:
            pass

        return backends

    @classmethod
    def detect_all(cls) -> DeviceCapabilities:
        """Detect all platform and device capabilities.

        Returns:
            Complete device capabilities
        """
        platform_type = cls.detect_platform()
        device_type = cls.detect_device_type()
        cpu_arch, cpu_cores = cls.get_cpu_info()
        total_ram, available_ram = cls.get_memory_info()
        has_gpu, gpu_name, gpu_memory = cls.get_gpu_info()
        supports_fp16, supports_int8, supports_int4 = cls.detect_quantization_support()
        backends = cls.detect_available_backends()

        # Determine edge device status
        is_raspberry_pi = platform_type == PlatformType.RASPBERRY_PI
        is_jetson = platform_type == PlatformType.JETSON
        is_edge_device = is_raspberry_pi or is_jetson or total_ram < 8.0

        capabilities = DeviceCapabilities(
            platform=platform_type,
            device_type=device_type,
            cpu_arch=cpu_arch,
            cpu_cores=cpu_cores,
            total_ram_gb=total_ram,
            available_ram_gb=available_ram,
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory,
            supports_cuda=device_type == DeviceType.CUDA,
            supports_metal=device_type == DeviceType.METAL,
            supports_rocm=device_type == DeviceType.ROCM,
            supports_fp16=supports_fp16,
            supports_int8=supports_int8,
            supports_int4=supports_int4,
            is_edge_device=is_edge_device,
            is_raspberry_pi=is_raspberry_pi,
            is_jetson=is_jetson,
            backends=backends,
        )

        logger.info(f"Platform detected: {platform_type.value}")
        logger.info(f"Device type: {device_type.value}")
        logger.info(f"CPU: {cpu_arch} ({cpu_cores} cores)")
        logger.info(f"RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available")

        if has_gpu:
            logger.info(f"GPU: {gpu_name}" + (f" ({gpu_memory:.1f}GB)" if gpu_memory else ""))

        logger.info(f"Backends: {', '.join(backends)}")
        logger.info(f"Quantization support: FP16={supports_fp16}, INT8={supports_int8}, INT4={supports_int4}")

        if is_edge_device:
            logger.info("Edge device detected - will use optimized settings")

        return capabilities
