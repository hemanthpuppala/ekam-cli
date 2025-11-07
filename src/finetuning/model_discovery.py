"""Unified model discovery for finetuning across all providers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from loguru import logger

from ..models.endpoints import ModelType
from ..models.model import ModelInfo
from ..services.session import SessionManager

if TYPE_CHECKING:
    from ..services.session import SessionManager


@dataclass
class FinetuneModelInfo:
    """Model information for finetuning selection."""

    model_id: str
    name: str
    provider: str
    model_type: ModelType
    size_gb: float
    available: bool = True
    compatibility_message: Optional[str] = None


class FinetuneModelDiscovery:
    """Discover and organize models from all providers for finetuning.

    Supports:
    - HuggingFace models
    - Ollama models
    - GGUF models
    - Quantized models
    - MLX models (Apple Silicon)
    """

    def __init__(self, session_manager):
        """Initialize model discovery.

        Args:
            session_manager: Session manager with registered providers
        """
        self.session_manager = session_manager
        self.model_discovery = session_manager.model_discovery
        self._models_cache = None

    def discover_all_models(self, refresh: bool = False) -> dict:
        """Discover all models from all providers organized by type.

        Returns:
            Dictionary with keys: "llm", "vlm", containing lists of FinetuneModelInfo
        """
        if self._models_cache and not refresh:
            return self._models_cache

        try:
            llm_models = []
            vlm_models = []

            # Get models from each provider
            providers = self.model_discovery._providers

            for provider_type, provider_instance in providers.items():
                try:
                    logger.debug(f"Discovering models from {provider_type.value}")
                    models = provider_instance.discover_models()

                    for model in models:
                        finetune_model = FinetuneModelInfo(
                            model_id=model.model_id,
                            name=model.name,
                            provider=str(provider_type.value),
                            model_type=model.model_type,
                            size_gb=model.size_gb,
                            available=model.compatibility != "unavailable",
                            compatibility_message=getattr(model, "compatibility_message", None),
                        )

                        # Organize by type
                        if model.model_type == ModelType.LLM:
                            llm_models.append(finetune_model)
                        elif model.model_type == ModelType.VLM:
                            vlm_models.append(finetune_model)

                except Exception as e:
                    logger.warning(f"Failed to discover models from {provider_type.value}: {e}")

            # Sort by name
            llm_models.sort(key=lambda x: x.name)
            vlm_models.sort(key=lambda x: x.name)

            result = {
                "llm": llm_models,
                "vlm": vlm_models,
            }

            self._models_cache = result
            logger.info(f"Discovered {len(llm_models)} LLMs and {len(vlm_models)} VLMs for finetuning")

            return result

        except Exception as e:
            logger.error(f"Model discovery failed: {e}", exc_info=True)
            return {"llm": [], "vlm": []}

    def get_llm_models(self, refresh: bool = False):
        """Get all LLM models available for finetuning.

        Args:
            refresh: Force refresh cache

        Returns:
            List of LLM models
        """
        models = self.discover_all_models(refresh)
        return models["llm"]

    def get_vlm_models(self, refresh: bool = False):
        """Get all VLM models available for finetuning.

        Args:
            refresh: Force refresh cache

        Returns:
            List of VLM models
        """
        models = self.discover_all_models(refresh)
        return models["vlm"]

    def get_model_by_id(self, model_id: str) -> Optional[FinetuneModelInfo]:
        """Get model by ID.

        Args:
            model_id: Model identifier

        Returns:
            FinetuneModelInfo or None
        """
        models = self.discover_all_models()
        for model in models["llm"] + models["vlm"]:
            if model.model_id == model_id:
                return model
        return None

    def format_models_for_display(self, model_type: str = "all") -> str:
        """Format models for display in UI.

        Args:
            model_type: "llm", "vlm", or "all"

        Returns:
            Formatted string for display
        """
        models_dict = self.discover_all_models()

        if model_type == "all":
            llm_text = self._format_model_section("Language Models (LLM)", models_dict["llm"])
            vlm_text = self._format_model_section("Vision-Language Models (VLM)", models_dict["vlm"])
            return f"{llm_text}\n\n{vlm_text}"
        elif model_type == "llm":
            return self._format_model_section("Language Models (LLM)", models_dict["llm"])
        elif model_type == "vlm":
            return self._format_model_section("Vision-Language Models (VLM)", models_dict["vlm"])
        else:
            return ""

    def _format_model_section(self, title: str, models) -> str:
        """Format a section of models for display.

        Args:
            title: Section title
            models: List of models

        Returns:
            Formatted string
        """
        text = f"{title}\n"
        text += "━" * 80 + "\n\n"

        if not models:
            text += "[dim]No models found[/dim]\n"
            return text

        # Group by provider
        providers_dict = {}
        for model in models:
            if model.provider not in providers_dict:
                providers_dict[model.provider] = []
            providers_dict[model.provider].append(model)

        # Display grouped by provider
        for provider, provider_models in sorted(providers_dict.items()):
            text += f"[bold cyan]📦 {provider.upper()}[/bold cyan]\n"

            for i, model in enumerate(provider_models, 1):
                status = "✓" if model.available else "✗"
                text += f"  [{status}] {model.name:<40} ({model.size_gb:.1f}GB) - {model.model_id}\n"

            text += "\n"

        return text

    def count_models(self) -> dict:
        """Get count of available models.

        Returns:
            Dictionary with counts: {"llm": int, "vlm": int, "total": int}
        """
        models = self.discover_all_models()
        return {
            "llm": len(models["llm"]),
            "vlm": len(models["vlm"]),
            "total": len(models["llm"]) + len(models["vlm"]),
        }
