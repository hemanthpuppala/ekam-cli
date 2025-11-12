"""Session state, statistics, and loaded model tracking."""

import gc
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr, computed_field

from .endpoints import EndpointType, ProviderType
from .inference import InferenceResult
from .model import ModelInfo
from .provider import ProviderConfig
from .system import SystemSpecs


class ConversationExchange(BaseModel):
    """A single user-AI exchange in a conversation."""

    user_message: str = Field(description="User's message")
    ai_response: str = Field(description="AI's response (clean response without reasoning)")
    ai_reasoning: Optional[str] = Field(default=None, description="AI's reasoning/thinking blocks (if present)")
    timestamp: datetime = Field(default_factory=datetime.now)
    inference_time_ms: float = Field(default=0.0, description="Time taken for this response")


class ModelParameters(BaseModel):
    """Model generation parameters for session-specific customization.

    All parameters are optional. When None, provider defaults are used.
    Validation ensures parameters are within safe, reasonable ranges.
    """

    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=32768,
        description="Maximum tokens to generate (1-32768, None = use default)"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0, lower = more deterministic)"
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold (0.0-1.0)"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=0,
        le=1000,
        description="Top-K sampling (0-1000, 0 = disabled)"
    )
    repeat_penalty: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Repetition penalty (0.0-10.0, 1.0 = no penalty)"
    )
    # Llama.cpp advanced sampling parameters
    presence_penalty: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Presence penalty (0.0-2.0, 0 = none)"
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Frequency penalty (0.0-2.0, 0 = none)"
    )
    typical_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Typical decoding p (0.0-1.0)"
    )
    tfs_z: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Tail free sampling z (0.0-1.0)"
    )
    min_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Min-p sampling (0.0-1.0)"
    )
    penalty_last_n: Optional[int] = Field(
        default=None,
        ge=0,
        le=4096,
        description="Tokens to consider for repetition penalty"
    )
    mirostat: Optional[int] = Field(
        default=None,
        ge=0,
        le=2,
        description="Mirostat mode (0=off,1,2)"
    )
    mirostat_tau: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Mirostat target entropy"
    )
    mirostat_eta: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Mirostat learning rate"
    )
    n_keep: Optional[int] = Field(
        default=None,
        ge=0,
        le=32768,
        description="Number of initial tokens to keep from being truncated"
    )
    ignore_eos: Optional[bool] = Field(
        default=None,
        description="Ignore end-of-sequence tokens"
    )
    stop: Optional[list[str]] = Field(
        default=None,
        description="Stop sequences"
    )
    n_probs: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description="Return top-n token probabilities"
    )
    grammar: Optional[str] = Field(
        default=None,
        description="GGML grammar (string)"
    )
    logit_bias: Optional[dict] = Field(
        default=None,
        description="Logit bias map (token or id -> bias)"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to prepend as role=system"
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2**32 - 1,
        description="Random seed for reproducibility (0 to 2^32-1)"
    )
    max_history_turns: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Max conversation history turns (1-200, None = use CLI default of 20, 0 = unlimited for API)"
    )

    def to_dict(self) -> dict:
        """Convert to dict with only non-None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class ConversationSession(BaseModel):
    """A conversation session with history tracking."""

    session_id: str = Field(default_factory=lambda: str(uuid4())[:12], description="Unique session ID (12-char UUID prefix)")
    endpoint_type: EndpointType = Field(description="Endpoint this session belongs to")
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)

    # Conversation exchanges
    exchanges: list[ConversationExchange] = Field(default_factory=list)

    # Metadata for image-based endpoints (QA, Caption)
    image_path: Optional[str] = Field(default=None, description="Associated image path for QA/Caption")

    # Session-specific model parameters (None = use provider defaults)
    model_parameters: Optional[ModelParameters] = Field(
        default=None,
        description="Custom model parameters for this session (persists across session reloads)"
    )

    @computed_field
    @property
    def exchange_count(self) -> int:
        """Number of exchanges in this session."""
        return len(self.exchanges)

    @computed_field
    @property
    def total_inference_time_ms(self) -> float:
        """Total inference time for all exchanges."""
        return sum(ex.inference_time_ms for ex in self.exchanges)

    @computed_field
    @property
    def session_duration_minutes(self) -> float:
        """Duration of session in minutes."""
        return (self.last_active - self.created_at).total_seconds() / 60

    def add_exchange(
        self,
        user_message: str,
        ai_response: str,
        inference_time_ms: float = 0.0,
        ai_reasoning: Optional[str] = None
    ) -> None:
        """Add a new exchange to the session.

        Args:
            user_message: User's input message
            ai_response: AI's clean response (without reasoning/thinking blocks)
            inference_time_ms: Time taken for inference
            ai_reasoning: Optional reasoning/thinking blocks extracted from response
        """
        exchange = ConversationExchange(
            user_message=user_message,
            ai_response=ai_response,
            ai_reasoning=ai_reasoning,
            inference_time_ms=inference_time_ms
        )
        self.exchanges.append(exchange)
        self.last_active = datetime.now()

    def get_history_for_provider(self, max_turns: Optional[int] = None) -> list[tuple[str, str]]:
        """Get conversation history in (user_msg, ai_response) format for providers.

        Args:
            max_turns: Maximum number of recent exchanges to include (None = all).
                      For CLI usage, recommended value is 20 to prevent context overflow.
                      For API usage, can be set to any value or None for unlimited.

        Returns:
            List of (user_message, ai_response) tuples, most recent last
            Note: Only returns clean responses (without reasoning blocks) for context
        """
        exchanges_to_use = self.exchanges

        # Apply max_turns limit if specified
        if max_turns is not None and max_turns > 0:
            exchanges_to_use = self.exchanges[-max_turns:]

        return [(ex.user_message, ex.ai_response) for ex in exchanges_to_use]

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class LoadedModel(BaseModel):
    """Active model state."""

    model_info: ModelInfo
    device: str = Field(description="Device model is loaded on (cuda/mps/cpu)")
    loaded_at: datetime = Field(default_factory=datetime.now)

    # Memory tracking (estimated)
    estimated_memory_gb: float = Field(
        description="Estimated RAM/VRAM usage (model_info.size_gb * 1.2 for safety)"
    )

    # Conversation sessions (multiple sessions per endpoint)
    conversation_sessions: dict[EndpointType, list[ConversationSession]] = Field(
        default_factory=dict,
        description="Conversation sessions organized by endpoint type"
    )

    # Currently active session (if any)
    active_session_id: Optional[str] = Field(
        default=None,
        description="ID of currently active conversation session"
    )

    # Provider-specific handle (not serialized) - using PrivateAttr for Pydantic v2
    _handle: Any = PrivateAttr(default=None)

    def create_session(self, endpoint_type: EndpointType, image_path: Optional[str] = None) -> ConversationSession:
        """Create a new conversation session for an endpoint.

        Args:
            endpoint_type: Type of endpoint (TEXT, QA, CAPTION)
            image_path: Optional image path for QA/Caption endpoints

        Returns:
            Created ConversationSession

        Note:
            Session IDs are 12-char UUID prefixes. Collision detection with retry.
        """
        max_retries = 5
        for attempt in range(max_retries):
            session = ConversationSession(
                endpoint_type=endpoint_type,
                image_path=image_path
            )

            # Check for ID collision across all endpoints
            collision = False
            for sessions in self.conversation_sessions.values():
                if any(s.session_id == session.session_id for s in sessions):
                    collision = True
                    break

            if not collision:
                # No collision, use this session
                if endpoint_type not in self.conversation_sessions:
                    self.conversation_sessions[endpoint_type] = []

                self.conversation_sessions[endpoint_type].append(session)
                self.active_session_id = session.session_id
                return session

        # Extremely unlikely: all retries had collisions, use longer ID
        from uuid import uuid4
        session = ConversationSession(endpoint_type=endpoint_type, image_path=image_path)
        session.session_id = str(uuid4())[:16]  # Use 16 chars as fallback

        if endpoint_type not in self.conversation_sessions:
            self.conversation_sessions[endpoint_type] = []

        self.conversation_sessions[endpoint_type].append(session)
        self.active_session_id = session.session_id
        return session

    def get_sessions(self, endpoint_type: EndpointType) -> list[ConversationSession]:
        """Get all sessions for an endpoint.

        Args:
            endpoint_type: Endpoint to get sessions for

        Returns:
            List of ConversationSession objects
        """
        return self.conversation_sessions.get(endpoint_type, [])

    def get_active_session(self) -> Optional[ConversationSession]:
        """Get currently active session.

        Returns:
            Active ConversationSession or None
        """
        if not self.active_session_id:
            return None

        # Search through all endpoint sessions
        for sessions in self.conversation_sessions.values():
            for session in sessions:
                if session.session_id == self.active_session_id:
                    return session

        return None

    def set_active_session(self, session_id: str) -> bool:
        """Set active session by ID.

        Args:
            session_id: Session ID to activate

        Returns:
            True if session found and activated
        """
        for sessions in self.conversation_sessions.values():
            for session in sessions:
                if session.session_id == session_id:
                    self.active_session_id = session_id
                    return True
        return False

    def clear_active_session(self) -> None:
        """Clear the currently active session reference."""
        self.active_session_id = None

    @computed_field
    @property
    def load_duration_seconds(self) -> float:
        """Seconds since model was loaded."""
        return (datetime.now() - self.loaded_at).total_seconds()

    def unload(self) -> None:
        """Unload model from memory and clear caches (per FR-017)."""
        if self._handle is not None:
            del self._handle
            self._handle = None

        # Clear device-specific caches
        if self.device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except ImportError:
                pass
        elif self.device == "mps":
            try:
                import torch

                torch.mps.empty_cache()
            except ImportError:
                pass

        gc.collect()

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True  # Allow Any for _handle


class SessionStatistics(BaseModel):
    """Aggregated inference statistics."""

    session_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"),
        description="Unique session identifier",
    )

    # Current state
    current_model_id: Optional[str] = None
    current_provider: Optional[ProviderType] = None

    # Cumulative metrics
    total_inferences: int = Field(default=0, ge=0)
    total_time_ms: float = Field(default=0.0, ge=0.0)

    # Breakdown by endpoint
    inferences_by_endpoint: dict[EndpointType, int] = Field(default_factory=dict)

    # Conversation session metrics
    total_conversation_sessions: int = Field(default=0, description="Total sessions created")
    history_enabled_inferences: int = Field(default=0, description="Inferences with history enabled")
    single_turn_inferences: int = Field(default=0, description="Single-turn inferences")

    # Session timing
    session_started_at: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def average_time_ms(self) -> float:
        """Average inference time across all endpoints."""
        if self.total_inferences == 0:
            return 0.0
        return self.total_time_ms / self.total_inferences

    @computed_field
    @property
    def session_duration_seconds(self) -> float:
        """Total session runtime."""
        return (datetime.now() - self.session_started_at).total_seconds()

    def record_inference(self, result: InferenceResult, used_history: bool = False) -> None:
        """Update statistics after inference (per FR-030).

        Args:
            result: Completed inference result
            used_history: Whether conversation history was used
        """
        self.total_inferences += 1
        self.total_time_ms += result.inference_time_ms

        endpoint = result.input.endpoint
        self.inferences_by_endpoint[endpoint] = self.inferences_by_endpoint.get(endpoint, 0) + 1

        # Track history usage
        if used_history:
            self.history_enabled_inferences += 1
        else:
            self.single_turn_inferences += 1

        self.current_model_id = result.model_id
        self.current_provider = result.provider

    def record_session_created(self) -> None:
        """Record that a new conversation session was created."""
        self.total_conversation_sessions += 1

    def to_display_dict(self) -> dict[str, str]:
        """Format for Rich table display (per FR-037)."""
        result = {
            "Session Duration": f"{self.session_duration_seconds / 60:.1f} minutes",
            "Current Model": self.current_model_id or "None",
            "Current Provider": self.current_provider or "None",
            "Total Inferences": str(self.total_inferences),
            "Total Time": f"{self.total_time_ms / 1000:.2f} seconds",
            "Average Time": f"{self.average_time_ms:.0f}ms per inference",
            "Breakdown": (
                ", ".join(f"{ep.value}: {count}" for ep, count in self.inferences_by_endpoint.items())
                if self.inferences_by_endpoint
                else "No inferences yet"
            ),
        }

        # Add conversation session metrics if any sessions exist
        if self.total_conversation_sessions > 0:
            result["Conversation Sessions"] = str(self.total_conversation_sessions)
            result["With History"] = f"{self.history_enabled_inferences} inferences"
            result["Single Turn"] = f"{self.single_turn_inferences} inferences"

        return result

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class SessionState(BaseModel):
    """Global application state."""

    # System information
    system_specs: SystemSpecs

    # Provider registry
    provider_configs: dict[ProviderType, ProviderConfig] = Field(default_factory=dict)

    # Model registry (O(1) lookups per FR-049)
    models: dict[str, ModelInfo] = Field(default_factory=dict)
    models_by_provider: dict[ProviderType, list[str]] = Field(default_factory=dict)

    # Active model
    loaded_model: Optional[LoadedModel] = None

    # Statistics
    statistics: SessionStatistics = Field(default_factory=SessionStatistics)

    # Session metadata
    session_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def register_model(self, model: ModelInfo) -> None:
        """Add model to registry with O(1) access (per FR-049).

        Args:
            model: Discovered model to register
        """
        self.models[model.model_id] = model

        provider_models = self.models_by_provider.setdefault(model.provider, [])
        if model.model_id not in provider_models:
            provider_models.append(model.model_id)

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """O(1) model lookup."""
        return self.models.get(model_id)

    def get_models_by_provider(self, provider: ProviderType) -> list[ModelInfo]:
        """O(1) filter by provider + O(N) model object retrieval."""
        model_ids = self.models_by_provider.get(provider, [])
        return [self.models[mid] for mid in model_ids]

    def load_model(self, model_info: ModelInfo, device: str, handle: Any) -> LoadedModel:
        """Load model into memory (enforces single-model constraint per FR-016).

        Args:
            model_info: Model to load
            device: Target device (cuda/mps/cpu)
            handle: Provider-specific model handle

        Returns:
            LoadedModel instance

        Raises:
            RuntimeError: If another model is already loaded
        """
        if self.loaded_model is not None:
            raise RuntimeError(
                f"Model {self.loaded_model.model_info.model_id} already loaded. "
                "Unload current model first."
            )

        self.loaded_model = LoadedModel(
            model_info=model_info,
            device=device,
            estimated_memory_gb=model_info.size_gb * 1.2,  # 20% safety margin
        )
        # Set private attribute after creation (Pydantic v2)
        self.loaded_model._handle = handle

        return self.loaded_model

    def unload_model(self) -> None:
        """Unload current model and free memory (per FR-016, FR-017)."""
        if self.loaded_model is not None:
            self.loaded_model.unload()
            self.loaded_model = None

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True  # For provider handles
        use_enum_values = True
