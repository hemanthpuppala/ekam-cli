"""
Model context manager for benchmark execution.

Provides safe model loading/unloading with resource tracking and cleanup.
Implements context manager protocol for automatic cleanup on errors.
"""

from typing import Optional, Any, TYPE_CHECKING
from datetime import datetime
from enum import Enum
from contextlib import contextmanager

from loguru import logger
from pydantic import BaseModel, Field

from src.models.endpoints import ProviderType

if TYPE_CHECKING:
    from .provider_bridge import ProviderBridge


class ModelState(str, Enum):
    """Model state during benchmark execution."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    UNLOADING = "unloading"


class ModelContext(BaseModel):
    """
    Tracks state of a model during benchmark execution.

    Attributes:
        model_id: Model identifier
        provider: Provider type
        state: Current model state
        loaded_at: When model was loaded (if loaded)
        failed_reason: Error message if loading failed
        handle: Model handle from provider (if loaded)
    """

    model_id: str
    provider: ProviderType
    state: ModelState = Field(default=ModelState.UNLOADED)
    loaded_at: Optional[datetime] = None
    failed_reason: Optional[str] = None

    # Private handle (not serialized)
    _handle: Any = None

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        use_enum_values = True


class ModelContextManager:
    """
    Context manager for safe model loading during benchmarks.

    Responsibilities:
    - Load models with error handling
    - Track model states across benchmark runs
    - Ensure proper cleanup (unload) even on errors
    - Provide resource usage tracking
    - Support persistent loading (keep model loaded across runs)

    Usage:
        ```python
        manager = ModelContextManager(provider_bridge)

        # Load model with automatic cleanup
        with manager.load_model("model-id") as context:
            if context.state == ModelState.LOADED:
                # Use model handle
                handle = context._handle
                # ... run inference ...
        # Model automatically unloaded on exit

        # Or use persistent mode (don't unload between runs)
        context = manager.load_model_persistent("model-id")
        # ... run multiple benchmarks ...
        manager.unload_model(context)
        ```
    """

    def __init__(self, provider_bridge: "ProviderBridge"):
        """
        Initialize model context manager.

        Args:
            provider_bridge: ProviderBridge for model operations
        """
        self.provider_bridge = provider_bridge

        # Track all contexts created (for resource tracking)
        self._contexts: dict[str, ModelContext] = {}

        logger.debug("ModelContextManager initialized")

    @contextmanager
    def load_model(self, model_id: str, unload_on_exit: bool = True):
        """
        Load model with context manager protocol.

        Args:
            model_id: Model identifier
            unload_on_exit: Whether to unload model on context exit (default: True)

        Yields:
            ModelContext with loaded model or error state

        Example:
            ```python
            with manager.load_model("llama-3") as ctx:
                if ctx.state == ModelState.LOADED:
                    # Use model via ProviderBridge
                    result = provider_bridge.execute_text(model_id, prompt)
            # Model automatically unloaded
            ```
        """
        # Resolve provider first
        provider = self.provider_bridge.resolve_provider(model_id)
        if not provider:
            logger.error(f"Could not resolve provider for {model_id}")
            context = ModelContext(model_id=model_id, provider=ProviderType.OLLAMA, state=ModelState.FAILED)
            context.failed_reason = "Provider resolution failed"
            yield context
            return

        context = self._create_or_get_context(model_id, provider)

        try:
            # Load model
            self._load_model_internal(context)
            yield context

        except Exception as e:
            logger.error(f"Error in model context: {e}")
            context.state = ModelState.FAILED
            context.failed_reason = str(e)
            yield context

        finally:
            # Cleanup on exit if requested
            if unload_on_exit and context.state == ModelState.LOADED:
                self._unload_model_internal(context)

    def load_model_persistent(self, model_id: str) -> ModelContext:
        """
        Load model without automatic unload (persistent across runs).

        Useful for speed benchmarks where we want to keep model loaded
        across multiple runs to avoid reload overhead.

        Args:
            model_id: Model identifier

        Returns:
            ModelContext with loaded model or error state

        Note:
            Caller must explicitly call unload_model() when done
        """
        # Resolve provider first
        provider = self.provider_bridge.resolve_provider(model_id)
        if not provider:
            logger.error(f"Could not resolve provider for {model_id}")
            context = ModelContext(model_id=model_id, provider=ProviderType.OLLAMA, state=ModelState.FAILED)
            context.failed_reason = "Provider resolution failed"
            return context

        context = self._create_or_get_context(model_id, provider)

        try:
            self._load_model_internal(context)
        except Exception as e:
            logger.error(f"Failed to load model persistently: {e}")
            context.state = ModelState.FAILED
            context.failed_reason = str(e)

        return context

    def unload_model(self, context: ModelContext) -> None:
        """
        Explicitly unload a model.

        Args:
            context: ModelContext to unload
        """
        if context.state == ModelState.LOADED:
            self._unload_model_internal(context)

    def unload_all(self) -> None:
        """Unload all loaded models."""
        loaded_contexts = [ctx for ctx in self._contexts.values() if ctx.state == ModelState.LOADED]
        for context in loaded_contexts:
            self._unload_model_internal(context)

    def get_context(self, model_id: str) -> Optional[ModelContext]:
        """
        Get context for a model.

        Args:
            model_id: Model identifier

        Returns:
            ModelContext if exists, None otherwise
        """
        return self._contexts.get(model_id)

    def get_loaded_contexts(self) -> list[ModelContext]:
        """
        Get all currently loaded model contexts.

        Returns:
            List of ModelContext objects in LOADED state
        """
        return [ctx for ctx in self._contexts.values() if ctx.state == ModelState.LOADED]

    def is_model_loaded(self, model_id: str) -> bool:
        """
        Check if a model is currently loaded.

        Args:
            model_id: Model identifier

        Returns:
            True if model is loaded
        """
        context = self._contexts.get(model_id)
        return context is not None and context.state == ModelState.LOADED

    def cleanup_all(self) -> None:
        """
        Unload all models and cleanup resources.

        Should be called at the end of benchmark execution.
        """
        logger.info("Cleaning up all model contexts")

        # Unload all loaded models
        self.unload_all()

        # Clear all contexts
        self._contexts.clear()

        # Also cleanup ProviderBridge
        self.provider_bridge.unload_all()

        logger.debug("All model contexts cleaned up")

    def _create_or_get_context(self, model_id: str, provider: ProviderType) -> ModelContext:
        """
        Create new context or get existing one.

        Args:
            model_id: Model identifier
            provider: Provider type

        Returns:
            ModelContext
        """
        if model_id in self._contexts:
            context = self._contexts[model_id]
            # Reset state if it was failed
            if context.state == ModelState.FAILED:
                context.state = ModelState.UNLOADED
                context.failed_reason = None
            return context

        # Create new context
        context = ModelContext(
            model_id=model_id,
            provider=provider
        )
        self._contexts[model_id] = context

        return context

    def _load_model_internal(self, context: ModelContext) -> None:
        """
        Internal method to load a model.

        Args:
            context: ModelContext to load

        Raises:
            RuntimeError: If loading fails
        """
        if context.state == ModelState.LOADED:
            logger.debug(f"Model {context.model_id} already loaded")
            return

        context.state = ModelState.LOADING
        logger.info(f"Loading model: {context.model_id} via {context.provider.value}")

        try:
            # Use provider bridge to load model
            provider_type, handle = self.provider_bridge.load_model(context.model_id)

            if not provider_type or not handle:
                raise RuntimeError(f"ProviderBridge failed to load {context.model_id}")

            # Update context
            context._handle = handle
            context.provider = provider_type  # Update with resolved provider
            context.state = ModelState.LOADED
            context.loaded_at = datetime.now()

            logger.info(f"Successfully loaded model: {context.model_id}")

        except Exception as e:
            logger.error(f"Failed to load model {context.model_id}: {e}")
            context.state = ModelState.FAILED
            context.failed_reason = str(e)
            raise

    def _unload_model_internal(self, context: ModelContext) -> None:
        """
        Internal method to unload a model.

        Args:
            context: ModelContext to unload
        """
        if context.state != ModelState.LOADED:
            logger.debug(f"Model {context.model_id} not loaded, skipping unload")
            return

        context.state = ModelState.UNLOADING
        logger.info(f"Unloading model: {context.model_id}")

        try:
            # Unload via provider bridge
            success = self.provider_bridge.unload_model(context.model_id)

            # Update context
            context._handle = None
            context.state = ModelState.UNLOADED
            context.loaded_at = None

            if success:
                logger.info(f"Successfully unloaded model: {context.model_id}")
            else:
                logger.warning(f"Model {context.model_id} was not in ProviderBridge's loaded models")

        except Exception as e:
            logger.error(f"Error unloading model {context.model_id}: {e}")
            # Still mark as unloaded to avoid stuck state
            context.state = ModelState.UNLOADED
            context._handle = None

    def get_resource_summary(self) -> dict:
        """
        Get summary of resource usage.

        Returns:
            Dict with resource statistics
        """
        total_contexts = len(self._contexts)
        loaded_count = sum(1 for ctx in self._contexts.values() if ctx.state == ModelState.LOADED)
        failed_count = sum(1 for ctx in self._contexts.values() if ctx.state == ModelState.FAILED)
        loaded_models = [ctx.model_id for ctx in self._contexts.values() if ctx.state == ModelState.LOADED]

        return {
            "total_contexts": total_contexts,
            "loaded_models": loaded_count,
            "failed_models": failed_count,
            "loaded_model_ids": loaded_models
        }

    def __repr__(self) -> str:
        """String representation."""
        summary = self.get_resource_summary()
        return (
            f"ModelContextManager("
            f"contexts={summary['total_contexts']}, "
            f"loaded={summary['loaded_models']}, "
            f"failed={summary['failed_models']}"
            f")"
        )
