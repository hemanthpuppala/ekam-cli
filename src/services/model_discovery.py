"""Model discovery service for aggregating models from all providers."""

from typing import Optional

from loguru import logger

from ..models.endpoints import CompatibilityStatus, ProviderType
from ..models.model import ModelInfo
from ..providers.base import BaseProvider


class ModelDiscoveryService:
    """Aggregate and manage models from all providers."""

    def __init__(self):
        """Initialize ModelDiscoveryService."""
        self.providers: dict[ProviderType, BaseProvider] = {}
        self.model_registry: dict[str, ModelInfo] = {}  # O(1) lookup by model_id

    def register_provider(self, provider_type: ProviderType, provider: BaseProvider) -> None:
        """Register a provider for model discovery.

        Args:
            provider_type: Type of provider
            provider: Provider instance
        """
        self.providers[provider_type] = provider
        logger.info(f"Registered provider: {provider_type}")

    def discover_all_models(self) -> list[ModelInfo]:
        """Discover models from all registered providers.

        Returns:
            List of all discovered models
        """
        all_models = []

        for provider_type, provider in self.providers.items():
            try:
                models = provider.discover_models()
                all_models.extend(models)
                logger.info(f"Discovered {len(models)} models from {provider_type}")
            except Exception as e:
                logger.error(f"Failed to discover models from {provider_type}: {e}")
                # Continue with other providers

        # Update registry
        self.model_registry = {model.model_id: model for model in all_models}

        logger.info(f"Total models discovered: {len(all_models)}")
        return all_models

    def discover_models_by_provider(self, provider: ProviderType) -> list[ModelInfo]:
        """Discover models from specific provider.

        Args:
            provider: Provider type to query

        Returns:
            List of models from provider
        """
        if provider not in self.providers:
            logger.warning(f"Provider {provider} not registered")
            return []

        try:
            models = self.providers[provider].discover_models()
            # Update registry with discovered models
            for model in models:
                self.model_registry[model.model_id] = model
            return models
        except Exception as e:
            logger.error(f"Failed to discover models from {provider}: {e}")
            return []

    def sort_models(
        self, models: list[ModelInfo], sort_by: str = "compatibility"
    ) -> list[ModelInfo]:
        """Sort models by specified criteria.

        Args:
            models: List of models to sort
            sort_by: Sort key ("compatibility", "size", "name")

        Returns:
            Sorted list of models
        """
        if sort_by == "compatibility":
            # Sort: PERFECT_FIT > TIGHT_FIT > TOO_LARGE
            order = {
                CompatibilityStatus.PERFECT_FIT: 0,
                CompatibilityStatus.TIGHT_FIT: 1,
                CompatibilityStatus.TOO_LARGE: 2,
            }
            return sorted(models, key=lambda m: (order[m.compatibility], m.size_gb))

        elif sort_by == "size":
            return sorted(models, key=lambda m: m.size_gb)

        elif sort_by == "name":
            return sorted(models, key=lambda m: m.name.lower())

        else:
            return models

    def find_model(self, model_id: str, provider: Optional[ProviderType] = None) -> Optional[ModelInfo]:
        """Find model by ID with O(1) lookup.

        Args:
            model_id: Model identifier
            provider: Optional provider filter

        Returns:
            ModelInfo if found, None otherwise
        """
        model = self.model_registry.get(model_id)

        if model and provider and model.provider != provider:
            return None

        return model

    def filter_models(
        self,
        models: list[ModelInfo],
        provider: Optional[ProviderType] = None,
        min_compatibility: Optional[CompatibilityStatus] = None,
        installed_only: bool = False,
    ) -> list[ModelInfo]:
        """Filter models by criteria.

        Args:
            models: List of models to filter
            provider: Filter by provider type
            min_compatibility: Minimum compatibility status
            installed_only: Only show installed models

        Returns:
            Filtered list of models
        """
        filtered = models

        if provider:
            filtered = [m for m in filtered if m.provider == provider]

        if min_compatibility:
            compat_order = {
                CompatibilityStatus.PERFECT_FIT: 0,
                CompatibilityStatus.TIGHT_FIT: 1,
                CompatibilityStatus.TOO_LARGE: 2,
            }
            min_level = compat_order[min_compatibility]
            filtered = [m for m in filtered if compat_order[m.compatibility] <= min_level]

        if installed_only:
            filtered = [m for m in filtered if m.is_installed]

        return filtered

    def get_provider(self, provider_type: ProviderType) -> Optional[BaseProvider]:
        """Get registered provider by type.

        Args:
            provider_type: Provider type

        Returns:
            Provider instance if registered
        """
        return self.providers.get(provider_type)
