"""System specifications and hardware detection."""

import platform
from typing import Optional

import psutil
from pydantic import BaseModel, Field, computed_field


class GPUInfo(BaseModel):
    """GPU-specific information."""

    available: bool
    gpu_type: str  # "cuda", "mps", "none"
    device_name: Optional[str] = None  # "NVIDIA RTX 3090", "Apple M1", etc.
    memory_gb: Optional[float] = None


class SystemSpecs(BaseModel):
    """Hardware specifications for resource management."""

    # Platform
    platform: str = Field(description="OS name (Darwin, Linux, Windows)")
    architecture: str = Field(description="CPU arch (x86_64, arm64)")

    # CPU
    cpu_cores_physical: int = Field(ge=1)
    cpu_cores_logical: int = Field(ge=1)

    # Memory
    total_ram_gb: float = Field(gt=0, description="Total system RAM")
    available_ram_gb: float = Field(gt=0, description="Available RAM at startup")

    # GPU
    gpu: GPUInfo

    # Device category (for user display)
    device_category: str = Field(
        description="LAPTOP, DESKTOP, EDGE_DEVICE (RPi/Jetson), or SERVER"
    )

    @computed_field
    @property
    def recommended_model_size_gb(self) -> float:
        """Calculate recommended maximum model size based on actual system resources.

        Formula considers:
        - GPU memory (if available): 50-70% of GPU VRAM
        - CPU + RAM: 40% of available RAM
        - Apple Silicon (MPS): 50% of available RAM (unified memory)
        - Minimum: 0.5 GB, Maximum: total_ram - 2GB (keep 2GB for OS)

        Returns:
            Recommended max model size in GB
        """
        # If GPU is available with dedicated memory, use GPU memory as basis
        if self.gpu.available and self.gpu.memory_gb:
            if self.gpu.gpu_type == "cuda":
                # NVIDIA GPU: use 60% of VRAM (leave room for activations)
                recommended = self.gpu.memory_gb * 0.6
            elif self.gpu.gpu_type == "mps":
                # Apple Silicon: unified memory, use 50% of available RAM
                recommended = self.available_ram_gb * 0.5
            else:
                # Other GPU: conservative estimate
                recommended = min(self.gpu.memory_gb * 0.5, self.available_ram_gb * 0.4)
        else:
            # CPU only: use 40% of available RAM
            recommended = self.available_ram_gb * 0.4

        # Ensure minimum 0.5GB
        recommended = max(0.5, recommended)

        # Cap at (total_ram - 2GB) to keep OS responsive
        max_allowed = max(0.5, self.total_ram_gb - 2.0)
        recommended = min(recommended, max_allowed)

        return round(recommended, 1)

    @classmethod
    def detect(cls) -> "SystemSpecs":
        """Detect system specifications at runtime.

        Returns:
            SystemSpecs instance with current hardware info
        """
        mem = psutil.virtual_memory()

        # Detect GPU
        gpu_info = GPUInfo(available=False, gpu_type="none")
        try:
            import torch

            if torch.cuda.is_available():
                gpu_info.available = True
                gpu_info.gpu_type = "cuda"
                gpu_info.device_name = torch.cuda.get_device_name(0)
                gpu_info.memory_gb = (
                    torch.cuda.get_device_properties(0).total_memory / (1024**3)
                )
            elif torch.backends.mps.is_available():
                gpu_info.available = True
                gpu_info.gpu_type = "mps"
                gpu_info.device_name = "Apple Silicon GPU"
                # MPS doesn't expose memory directly, estimate from system RAM
                gpu_info.memory_gb = mem.total / (1024**3) * 0.75  # Heuristic
        except ImportError:
            pass

        # Categorize device
        total_gb = mem.total / (1024**3)
        if total_gb < 8:
            category = "EDGE_DEVICE"
        elif total_gb < 16:
            category = "LAPTOP"
        elif total_gb < 64:
            category = "DESKTOP"
        else:
            category = "SERVER"

        return cls(
            platform=platform.system(),
            architecture=platform.machine(),
            cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
            cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
            total_ram_gb=total_gb,
            available_ram_gb=mem.available / (1024**3),
            gpu=gpu_info,
            device_category=category,
        )
