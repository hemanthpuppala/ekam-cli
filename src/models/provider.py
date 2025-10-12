"""Provider configuration models."""

from typing import Optional

from pydantic import BaseModel, Field, HttpUrl

from .endpoints import ProviderType


class ProviderConfig(BaseModel):
    """Configuration for a single provider."""

    name: ProviderType
    enabled: bool = True

    # API-based providers (Ollama, LM Studio)
    host: Optional[HttpUrl] = None

    # File-based providers (HuggingFace, GGUF)
    cache_dir: Optional[str] = None
    models_dir: Optional[str] = None

    # Device preferences (HuggingFace, GGUF)
    device_preference: list[str] = Field(
        default=["cuda", "mps", "cpu"], description="Priority order for device selection"
    )

    # Connection settings (API providers)
    timeout_seconds: int = Field(default=120, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def get_primary_device(self) -> str:
        """Return first available device from preference list.

        Returns:
            Device name: "cuda", "mps", or "cpu"
        """
        try:
            import torch

            for device in self.device_preference:
                if device == "cuda" and torch.cuda.is_available():
                    return "cuda"
                elif device == "mps" and torch.backends.mps.is_available():
                    return "mps"
        except ImportError:
            pass

        return "cpu"
