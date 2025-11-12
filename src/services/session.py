"""Session management service for coordinating application state."""

from typing import Optional

from loguru import logger

from ..models.endpoints import CompatibilityStatus, ProviderType
from ..models.model import ModelInfo
from ..models.provider import ProviderConfig
from ..models.session import SessionState
from ..models.system import SystemSpecs
from .model_discovery import ModelDiscoveryService
from .resource_manager import ResourceManager


class SessionManager:
    """Coordinate session state, providers, and resource management."""

    def __init__(self, config: dict, system_specs: SystemSpecs):
        """Initialize SessionManager.

        Args:
            config: Configuration dict with provider configs
            system_specs: System specifications
        """
        self.config = config
        self.system_specs = system_specs

        # Initialize state
        self.state = SessionState(system_specs=system_specs)

        # Initialize services
        self.resource_manager = ResourceManager(system_specs)
        self.model_discovery = ModelDiscoveryService()

        logger.info("SessionManager initialized")

    def register_provider(self, provider_type: ProviderType, provider_config: ProviderConfig, provider_instance) -> None:
        """Register provider with session.

        Args:
            provider_type: Type of provider
            provider_config: Provider configuration
            provider_instance: Provider instance
        """
        self.state.provider_configs[provider_type] = provider_config
        self.model_discovery.register_provider(provider_type, provider_instance)
        logger.info(f"Registered provider {provider_type}")

    def discover_models(self, provider: Optional[ProviderType] = None) -> list[ModelInfo]:
        """Discover models from all or specific provider.

        Args:
            provider: Optional provider to query (None = all providers)

        Returns:
            List of discovered models with compatibility assessed
        """
        if provider:
            models = self.model_discovery.discover_models_by_provider(provider)
        else:
            models = self.model_discovery.discover_all_models()

        # Assess compatibility for all models
        for model in models:
            status, message = self.resource_manager.assess_model_compatibility(model)
            model.compatibility = status
            model.compatibility_message = message

        # Register models in state
        for model in models:
            self.state.register_model(model)

        # Sort by compatibility
        return self.model_discovery.sort_models(models)

    def load_model(self, model_id: str, provider: ProviderType) -> bool:
        """Load model with compatibility checks and user confirmation.

        Args:
            model_id: Model identifier
            provider: Provider type

        Returns:
            True if model loaded successfully
        """
        # Find model
        model = self.state.get_model(model_id)
        if not model:
            logger.error(f"Model {model_id} not found")
            return False

        # Compatibility check already done in app.py with user confirmation
        # Skip duplicate confirmation here (user already confirmed in the UI layer)

        # Unload current model if any
        if self.state.loaded_model:
            logger.info(f"Unloading current model: {self.state.loaded_model.model_info.model_id}")
            self.unload_current_model()

        # Get provider
        provider_instance = self.model_discovery.get_provider(provider)
        if not provider_instance:
            logger.error(f"Provider {provider} not available")
            return False

        # Get device
        provider_config = self.state.provider_configs.get(provider)
        device = provider_config.get_primary_device() if provider_config else "cpu"

        # Load model
        try:
            logger.info(f"Loading model {model_id} on {device}")
            handle = provider_instance.load_model(model_id, device)
            self.state.load_model(model, device, handle)
            logger.info(f"Successfully loaded {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False

    def unload_current_model(self) -> None:
        """Unload currently loaded model."""
        if self.state.loaded_model:
            model_id = self.state.loaded_model.model_info.model_id
            logger.info(f"Unloading model {model_id}")

            # Call provider's unload_model to properly cleanup resources (e.g., stop llama-server)
            provider = self.get_current_provider()
            if provider and self.state.loaded_model._handle is not None:
                try:
                    provider.unload_model(self.state.loaded_model._handle)
                except Exception as e:
                    logger.warning(f"Error during provider unload: {e}")

            # Clear model from session state
            self.state.unload_model()
            logger.info(f"Model {model_id} unloaded")

    def get_current_provider(self) -> Optional:
        """Get provider for currently loaded model.

        Returns:
            Provider instance if model is loaded
        """
        if not self.state.loaded_model:
            return None

        provider_type = self.state.loaded_model.model_info.provider
        return self.model_discovery.get_provider(provider_type)

    def get_loaded_model_handle(self):
        """Get handle for currently loaded model.

        Returns:
            Model handle if loaded, None otherwise
        """
        if self.state.loaded_model:
            return self.state.loaded_model._handle
        return None

    def record_inference(self, result) -> None:
        """Record inference statistics.

        Args:
            result: InferenceResult to record
        """
        self.state.statistics.record_inference(result)

    def get_statistics(self) -> dict:
        """Get session statistics for display.

        Returns:
            Dict formatted for Rich table display
        """
        return self.state.statistics.to_display_dict()
