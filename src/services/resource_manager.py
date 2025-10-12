"""Resource management for memory-aware model loading."""

import psutil
from loguru import logger

from ..models.endpoints import CompatibilityStatus
from ..models.model import ModelInfo
from ..models.system import SystemSpecs


class ResourceManager:
    """Manage system resources and assess model compatibility."""

    def __init__(self, system_specs: SystemSpecs):
        """Initialize ResourceManager.

        Args:
            system_specs: System specifications from startup
        """
        self.system_specs = system_specs

    def assess_model_compatibility(self, model: ModelInfo) -> tuple[CompatibilityStatus, str]:
        """Assess if model fits in available memory (per FR-014).

        Formula:
        - <70% of recommended size: PERFECT_FIT
        - 70-100% of recommended size: TIGHT_FIT
        - >100% of recommended size: TOO_LARGE

        Args:
            model: ModelInfo to assess

        Returns:
            (CompatibilityStatus, message) tuple
        """
        recommended_size = self.system_specs.recommended_model_size_gb
        return model.assess_compatibility(recommended_size)

    def check_memory_available(self, required_gb: float) -> bool:
        """Check if sufficient memory is available for model.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if memory is available
        """
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)

        logger.debug(f"Memory check: {available_gb:.2f}GB available, {required_gb:.2f}GB required")

        return available_gb >= required_gb

    def prompt_user_override(self, model: ModelInfo) -> bool:
        """Prompt user to override TOO_LARGE model warning (per CL-008).

        Args:
            model: ModelInfo with TOO_LARGE compatibility

        Returns:
            True if user confirms override
        """
        print()
        print("⚠️  WARNING: Model Size Exceeds Recommended Limit")
        print("=" * 60)
        print(f"Model: {model.name}")
        print(f"Size: {model.size_gb:.1f}GB")
        print(f"Recommended max: {self.system_specs.recommended_model_size_gb:.1f}GB")
        print()
        print(model.compatibility_message)
        print()
        print("This may cause:")
        print("  • Out-of-memory (OOM) errors")
        print("  • System slowdown or freezing")
        print("  • Application crashes")
        print()

        while True:
            response = input("Proceed anyway? Type 'yes, proceed anyway' to continue: ").strip().lower()

            if response == "yes, proceed anyway":
                logger.warning(f"User overrode TOO_LARGE warning for {model.model_id}")
                return True
            elif response in ["no", "n", "exit", "quit", "back"]:
                logger.info(f"User declined to load TOO_LARGE model {model.model_id}")
                return False
            else:
                print("Invalid response. Type 'yes, proceed anyway' to continue or 'no' to cancel.")

    def get_current_memory_usage(self) -> dict:
        """Get current memory usage statistics.

        Returns:
            Dict with memory statistics in GB
        """
        mem = psutil.virtual_memory()

        stats = {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
        }

        # Add GPU memory if available
        if self.system_specs.gpu.available:
            try:
                import torch

                if self.system_specs.gpu.gpu_type == "cuda":
                    stats["gpu_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
                    stats["gpu_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
                    stats["gpu_total_gb"] = self.system_specs.gpu.memory_gb
            except ImportError:
                pass

        return stats
