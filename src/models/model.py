"""Model information and metadata."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, computed_field

from .endpoints import CompatibilityStatus, EndpointType, ModelType, ProviderType


class ModelInfo(BaseModel):
    """Complete metadata for a discoverable model."""

    # Identity
    model_id: str = Field(
        description="Unique identifier (e.g., 'llama3.2-vision:11b', 'llava-hf/llava-1.5-7b-hf')"
    )
    name: str = Field(description="Display name for UI")
    provider: ProviderType

    # Resource requirements
    size_gb: float = Field(gt=0, description="Disk size in gigabytes")

    # Type and capabilities
    model_type: ModelType
    capabilities: list[EndpointType] = Field(
        description="Dynamically detected endpoints this model supports (per FR-005a)"
    )

    # Compatibility assessment (computed at discovery time)
    compatibility: CompatibilityStatus
    compatibility_message: str = Field(
        description="Human-readable explanation of compatibility status"
    )

    # Metadata
    discovered_at: datetime = Field(default_factory=datetime.now)
    is_installed: bool = Field(
        default=True, description="False if model is available but not downloaded"
    )

    # Optional extended metadata (from metadata cache)
    architecture: Optional[str] = None  # "llava", "moondream", "llama", etc.
    quantization: Optional[str] = None  # "Q4_K_M", "Q8_0", "fp16", etc.
    parameter_count: Optional[str] = None  # "7B", "13B", etc. (deprecated, use params_billions)
    params_billions: Optional[float] = None  # Parameter count in billions (e.g., 3.2, 7.0)
    ram_gb: Optional[float] = None  # Estimated RAM requirement in GB
    vram_gb: Optional[float] = None  # Estimated VRAM requirement in GB
    params_exact: Optional[bool] = None  # True if params_billions is exact, False if estimated
    ram_exact: Optional[bool] = None  # True if ram_gb is exact, False if estimated

    # Quantization support - path to actual model file/directory
    source_path: Optional[Path] = Field(
        default=None,
        description="Path to actual model file or directory for quantization. "
                    "For GGUF: path to .gguf file. "
                    "For HuggingFace: path to model directory. "
                    "For Ollama: path to blob file in ~/.ollama/models/blobs/"
    )

    @computed_field
    @property
    def compatibility_icon(self) -> str:
        """Return icon for terminal display."""
        return {
            CompatibilityStatus.PERFECT_FIT: "✅",
            CompatibilityStatus.TIGHT_FIT: "⚠️",
            CompatibilityStatus.TOO_LARGE: "❌",
        }[self.compatibility]

    @computed_field
    @property
    def supports_vision(self) -> bool:
        """Check if model supports vision endpoints."""
        vision_endpoints = {
            EndpointType.QA,
            EndpointType.CAPTION,
            EndpointType.DETECT,
            EndpointType.POINT,
        }
        return bool(vision_endpoints.intersection(self.capabilities))

    def assess_compatibility(
        self, recommended_size_gb: float
    ) -> tuple[CompatibilityStatus, str]:
        """Compute compatibility status based on recommended model size (per FR-014).

        Args:
            recommended_size_gb: Recommended maximum model size for system

        Returns:
            (status, message) tuple
        """
        # O(1) time and space complexity - simple division and comparison
        ratio = self.size_gb / recommended_size_gb

        if ratio < 0.7:
            return (
                CompatibilityStatus.PERFECT_FIT,
                f"Model uses {ratio*100:.0f}% of recommended size "
                f"({self.size_gb:.1f}GB / {recommended_size_gb:.1f}GB)",
            )
        elif ratio <= 1.0:
            return (
                CompatibilityStatus.TIGHT_FIT,
                f"Model uses {ratio*100:.0f}% of recommended size "
                f"({self.size_gb:.1f}GB / {recommended_size_gb:.1f}GB). "
                "May impact performance.",
            )
        else:
            return (
                CompatibilityStatus.TOO_LARGE,
                f"Model is {ratio:.1f}x larger than recommended "
                f"({self.size_gb:.1f}GB vs {recommended_size_gb:.1f}GB). "
                "May cause OOM errors.",
            )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
