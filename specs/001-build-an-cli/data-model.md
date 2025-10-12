# Data Model

**Feature**: High-Performance Multi-Provider VLM/LLM CLI with Resource Management
**Branch**: `001-build-an-cli`
**Date**: 2025-10-11

## Overview

This document defines all core entities, their attributes, relationships, validation rules, and state transitions for the VLM/LLM CLI application. All models use **Pydantic v2** for type-safe validation and serialization.

---

## Entity Relationship Diagram

```
┌─────────────────┐       1:N      ┌──────────────────┐
│  ProviderConfig │◄────────────────┤   ModelInfo      │
│                 │                 │                  │
│ - name          │                 │ - name           │
│ - type          │                 │ - provider       │
│ - host/cache    │                 │ - size_gb        │
│ - enabled       │                 │ - model_type     │
└─────────────────┘                 │ - capabilities   │
                                    │ - compatibility  │
        ▲                           └──────────────────┘
        │                                     │
        │                                     │ N:1
        │                                     ▼
        │                           ┌──────────────────┐
        │                           │  LoadedModel     │
        │                           │                  │
        │                           │ - model_info     │
        │                           │ - device         │
        │                           │ - loaded_at      │
        │                           │ - memory_usage   │
        │                           └──────────────────┘
        │                                     │
        │                                     │ 1:N
        │                                     ▼
        │                           ┌──────────────────┐
        │                           │  InferenceResult │
        │                           │                  │
        │                           │ - endpoint_type  │
        │                           │ - input          │
        │                           │ - output         │
        │                           │ - inference_time │
        │                           │ - timestamp      │
        │                           └──────────────────┘
        │
        │ 1:1
        ▼
┌─────────────────┐       1:1      ┌──────────────────┐
│  SystemSpecs    │◄────────────────┤  SessionState    │
│                 │                 │                  │
│ - platform      │                 │ - current_model  │
│ - cpu_cores     │                 │ - provider_reg   │
│ - total_ram     │                 │ - statistics     │
│ - available_ram │                 │ - started_at     │
│ - gpu_info      │                 └──────────────────┘
└─────────────────┘
```

---

## Core Entities

### 1. ProviderConfig

**Purpose**: Configuration for each provider (Ollama, HuggingFace, LM Studio, GGUF)

**Source**: Loaded from `config.yaml`, immutable during runtime

**Schema**:
```python
from pydantic import BaseModel, Field, HttpUrl, DirectoryPath
from enum import Enum
from typing import Optional

class ProviderType(str, Enum):
    """Provider category (all local per CL-001)"""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LM_STUDIO = "lm_studio"
    GGUF = "gguf"

class ProviderConfig(BaseModel):
    """Configuration for a single provider"""

    name: ProviderType
    enabled: bool = True

    # API-based providers (Ollama, LM Studio)
    host: Optional[HttpUrl] = None

    # File-based providers (HuggingFace, GGUF)
    cache_dir: Optional[DirectoryPath] = None
    models_dir: Optional[DirectoryPath] = None

    # Device preferences (HuggingFace, GGUF)
    device_preference: list[str] = Field(
        default=["cuda", "mps", "cpu"],
        description="Priority order for device selection"
    )

    # Connection settings (API providers)
    timeout_seconds: int = Field(default=120, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)

    class Config:
        use_enum_values = True

    def get_primary_device(self) -> str:
        """Return first available device from preference list"""
        import torch

        for device in self.device_preference:
            if device == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif device == "mps" and torch.backends.mps.is_available():
                return "mps"

        return "cpu"
```

**Validation Rules**:
- `host` required for OLLAMA and LM_STUDIO types
- `cache_dir` or `models_dir` required for HUGGINGFACE and GGUF types
- `timeout_seconds` must be >0 and ≤600 (10 minutes)
- `device_preference` must contain at least one valid device

**Relationships**:
- 1:N with ModelInfo (one provider has many models)

---

### 2. ModelInfo

**Purpose**: Metadata for discovered models, including type, size, and capabilities

**Source**: Discovered dynamically from providers on startup or refresh

**Schema**:
```python
from pydantic import BaseModel, Field, computed_field
from enum import Enum
from datetime import datetime

class ModelType(str, Enum):
    """Model category based on capabilities"""
    VLM = "vlm"              # Vision-language model
    LLM = "llm"              # Text-only language model
    EMBEDDING = "embedding"  # Embedding model (not supported for inference)

class CompatibilityStatus(str, Enum):
    """Memory compatibility assessment per FR-014"""
    PERFECT_FIT = "perfect_fit"     # <70% of recommended size
    TIGHT_FIT = "tight_fit"         # 70-100% of recommended size
    TOO_LARGE = "too_large"         # >100% of recommended size

class EndpointType(str, Enum):
    """Supported inference endpoints"""
    QA = "qa"                # Question answering (VLM only)
    CAPTION = "caption"      # Image captioning (VLM only)
    DETECT = "detect"        # Object detection (VLM only)
    POINT = "point"          # Object pointing (VLM only)
    TEXT = "text"            # Text generation (LLM and VLM)

class ModelInfo(BaseModel):
    """Complete metadata for a discoverable model"""

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
        default=True,
        description="False if model is available but not downloaded"
    )

    # Optional extended metadata
    architecture: Optional[str] = None  # "llava", "moondream", "llama", etc.
    quantization: Optional[str] = None  # "Q4_K_M", "Q8_0", "fp16", etc.
    parameter_count: Optional[str] = None  # "7B", "13B", etc.

    @computed_field
    @property
    def compatibility_icon(self) -> str:
        """Return icon for terminal display"""
        return {
            CompatibilityStatus.PERFECT_FIT: "✅",
            CompatibilityStatus.TIGHT_FIT: "⚠️",
            CompatibilityStatus.TOO_LARGE: "❌"
        }[self.compatibility]

    @computed_field
    @property
    def supports_vision(self) -> bool:
        """Check if model supports vision endpoints"""
        vision_endpoints = {EndpointType.QA, EndpointType.CAPTION, EndpointType.DETECT, EndpointType.POINT}
        return bool(vision_endpoints.intersection(self.capabilities))

    def assess_compatibility(self, recommended_size_gb: float) -> tuple[CompatibilityStatus, str]:
        """
        Compute compatibility status based on recommended model size (per FR-014)

        Returns:
            (status, message) tuple
        """
        ratio = self.size_gb / recommended_size_gb

        if ratio < 0.7:
            return (
                CompatibilityStatus.PERFECT_FIT,
                f"Model uses {ratio*100:.0f}% of recommended size ({self.size_gb:.1f}GB / {recommended_size_gb:.1f}GB)"
            )
        elif ratio <= 1.0:
            return (
                CompatibilityStatus.TIGHT_FIT,
                f"Model uses {ratio*100:.0f}% of recommended size ({self.size_gb:.1f}GB / {recommended_size_gb:.1f}GB). May impact performance."
            )
        else:
            return (
                CompatibilityStatus.TOO_LARGE,
                f"Model exceeds recommended size by {(ratio-1)*100:.0f}% ({self.size_gb:.1f}GB vs {recommended_size_gb:.1f}GB). May cause OOM errors."
            )

    class Config:
        use_enum_values = True
```

**Validation Rules**:
- `size_gb` must be >0 and <200 (sanity check)
- `capabilities` must not be empty
- VLM models must include at least one vision endpoint in capabilities
- LLM models must include TEXT in capabilities
- `model_id` must be unique within provider

**Relationships**:
- N:1 with ProviderConfig (many models from one provider)
- 1:1 with LoadedModel (when model is active)

---

### 3. SystemSpecs

**Purpose**: Hardware capabilities detected at startup (per FR-012)

**Source**: Queried via `psutil` and `torch` on startup, cached for session

**Schema**:
```python
from pydantic import BaseModel, Field, computed_field
import psutil
import platform

class GPUInfo(BaseModel):
    """GPU-specific information"""
    available: bool
    gpu_type: str  # "cuda", "mps", "none"
    device_name: Optional[str] = None  # "NVIDIA RTX 3090", "Apple M1", etc.
    memory_gb: Optional[float] = None

class SystemSpecs(BaseModel):
    """Hardware specifications for resource management"""

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
        """
        Calculate recommended maximum model size (per FR-013)

        Formula: min(2.0, max(0.5, total_ram * 0.1))
        """
        size = max(0.5, self.total_ram_gb * 0.1)
        return min(2.0, size)

    @classmethod
    def detect(cls) -> "SystemSpecs":
        """
        Detect system specifications at runtime

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
                gpu_info.memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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
            device_category=category
        )
```

**Validation Rules**:
- `available_ram_gb` must be ≤ `total_ram_gb`
- `cpu_cores_logical` must be ≥ `cpu_cores_physical`
- If `gpu.available` is True, `gpu_type` must not be "none"

**Relationships**:
- 1:1 with SessionState (single system spec per session)

---

### 4. LoadedModel

**Purpose**: Represents currently loaded model in memory (per FR-015, FR-016)

**Source**: Created when model is loaded, destroyed on unload/switch

**Schema**:
```python
from pydantic import BaseModel, Field
from datetime import datetime

class LoadedModel(BaseModel):
    """Active model state"""

    model_info: ModelInfo
    device: str = Field(description="Device model is loaded on (cuda/mps/cpu)")
    loaded_at: datetime = Field(default_factory=datetime.now)

    # Memory tracking (estimated)
    estimated_memory_gb: float = Field(
        description="Estimated RAM/VRAM usage (model_info.size_gb * 1.2 for safety)"
    )

    # Provider-specific handle (not serialized)
    _handle: Any = Field(default=None, exclude=True, repr=False)

    @computed_field
    @property
    def load_duration_seconds(self) -> float:
        """Seconds since model was loaded"""
        return (datetime.now() - self.loaded_at).total_seconds()

    def unload(self):
        """
        Unload model from memory and clear caches (per FR-017)
        """
        if self._handle is not None:
            del self._handle
            self._handle = None

        # Clear device-specific caches
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.device == "mps":
            import torch
            torch.mps.empty_cache()

        import gc
        gc.collect()

    class Config:
        arbitrary_types_allowed = True  # Allow Any for _handle
```

**State Transitions**:
```
    [No Model Loaded]
          │
          │ load_model()
          ▼
    [Model Loading]
          │
          ├──[Success]──► [Model Loaded] ───switch_model()──► [Model Unloading] ──► [No Model Loaded]
          │                     │
          │                     │ unload_model()
          │                     ▼
          │              [Model Unloading]
          │                     │
          │                     ▼
          │              [No Model Loaded]
          │
          └──[Failure]──► [No Model Loaded]
```

**Validation Rules**:
- Only one LoadedModel can exist at a time (enforced by SessionState)
- `estimated_memory_gb` should be ≤ SystemSpecs.available_ram_gb (warning if exceeded)
- `device` must match `model_info.provider` device preferences

**Relationships**:
- 1:1 with ModelInfo (loaded instance of a model)
- 1:N with InferenceResult (generates multiple inferences)

---

### 5. InferenceResult

**Purpose**: Output from a single inference operation with timing and metadata (per FR-029, FR-042)

**Source**: Created after each successful endpoint execution

**Schema**:
```python
from pydantic import BaseModel, Field, FilePath
from datetime import datetime
from typing import Any

class InferenceInput(BaseModel):
    """Input parameters for inference"""

    endpoint: EndpointType

    # Text inputs (all endpoints)
    prompt: Optional[str] = None        # LLM text or VLM question
    object_name: Optional[str] = None   # detect/point target

    # Image inputs (VLM endpoints)
    image_path: Optional[FilePath] = None

    # Endpoint-specific parameters
    detail_level: Optional[str] = Field(default="detailed", pattern="^(detailed|short)$")  # caption

class InferenceOutput(BaseModel):
    """Output from inference"""

    # Text outputs
    text_response: Optional[str] = None  # QA answer, caption, text generation

    # Vision outputs
    detections: Optional[list[dict]] = None  # detect: [{"label": "car", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}]
    coordinates: Optional[dict] = None       # point: {"x": 120, "y": 450} or {"bbox": [x1,y1,x2,y2]}

    # Annotated image paths (detect/point)
    annotated_image_path: Optional[FilePath] = None

class InferenceResult(BaseModel):
    """Complete record of a single inference"""

    # Identity
    result_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        description="Unique ID for result persistence"
    )

    # Model context
    model_id: str
    provider: ProviderType
    device: str

    # Inference details
    input: InferenceInput
    output: InferenceOutput

    # Performance metrics (per FR-029)
    inference_time_ms: float = Field(gt=0, description="Inference duration in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Persistence
    saved_to_file: Optional[FilePath] = None  # JSON file path if user saved

    def to_json_dict(self) -> dict:
        """Serialize for file persistence (per FR-042)"""
        return {
            "result_id": self.result_id,
            "model": {
                "id": self.model_id,
                "provider": self.provider,
                "device": self.device
            },
            "endpoint": self.input.endpoint,
            "input": self.input.model_dump(exclude_none=True),
            "output": self.output.model_dump(exclude_none=True),
            "performance": {
                "inference_time_ms": self.inference_time_ms,
                "timestamp": self.timestamp.isoformat()
            },
            "metadata": {
                "generated_by": "VLM CLI v0.1.0"
            }
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "InferenceResult":
        """Deserialize from saved JSON"""
        return cls(
            result_id=data["result_id"],
            model_id=data["model"]["id"],
            provider=ProviderType(data["model"]["provider"]),
            device=data["model"]["device"],
            input=InferenceInput(**data["input"]),
            output=InferenceOutput(**data["output"]),
            inference_time_ms=data["performance"]["inference_time_ms"],
            timestamp=datetime.fromisoformat(data["performance"]["timestamp"])
        )
```

**Validation Rules**:
- `inference_time_ms` must be >0
- For VLM endpoints (QA, Caption, Detect, Point): `input.image_path` required
- For TEXT endpoint: `input.prompt` required
- For Detect/Point endpoints: `output.annotated_image_path` required

**Relationships**:
- N:1 with LoadedModel (many results from one loaded model)

---

### 6. SessionStatistics

**Purpose**: Cumulative performance metrics for current session (per FR-030, FR-031)

**Source**: Updated after each inference, reset on application restart

**Schema**:
```python
from pydantic import BaseModel, Field, computed_field
from datetime import datetime

class SessionStatistics(BaseModel):
    """Aggregated inference statistics"""

    session_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"),
        description="Unique session identifier"
    )

    # Current state
    current_model_id: Optional[str] = None
    current_provider: Optional[ProviderType] = None

    # Cumulative metrics
    total_inferences: int = Field(default=0, ge=0)
    total_time_ms: float = Field(default=0.0, ge=0.0)

    # Breakdown by endpoint
    inferences_by_endpoint: dict[EndpointType, int] = Field(default_factory=dict)

    # Session timing
    session_started_at: datetime = Field(default_factory=datetime.now)

    @computed_field
    @property
    def average_time_ms(self) -> float:
        """Average inference time across all endpoints"""
        if self.total_inferences == 0:
            return 0.0
        return self.total_time_ms / self.total_inferences

    @computed_field
    @property
    def session_duration_seconds(self) -> float:
        """Total session runtime"""
        return (datetime.now() - self.session_started_at).total_seconds()

    def record_inference(self, result: InferenceResult):
        """
        Update statistics after inference (per FR-030)

        Args:
            result: Completed inference result
        """
        self.total_inferences += 1
        self.total_time_ms += result.inference_time_ms

        endpoint = result.input.endpoint
        self.inferences_by_endpoint[endpoint] = self.inferences_by_endpoint.get(endpoint, 0) + 1

        self.current_model_id = result.model_id
        self.current_provider = result.provider

    def to_display_dict(self) -> dict:
        """Format for Rich table display (per FR-037)"""
        return {
            "Session Duration": f"{self.session_duration_seconds / 60:.1f} minutes",
            "Current Model": self.current_model_id or "None",
            "Current Provider": self.current_provider or "None",
            "Total Inferences": str(self.total_inferences),
            "Total Time": f"{self.total_time_ms / 1000:.2f} seconds",
            "Average Time": f"{self.average_time_ms:.0f}ms per inference",
            "Breakdown": ", ".join(
                f"{ep.value}: {count}" for ep, count in self.inferences_by_endpoint.items()
            ) if self.inferences_by_endpoint else "No inferences yet"
        }
```

**Validation Rules**:
- `total_time_ms` must match sum of all recorded inference times (within floating point error)
- `total_inferences` must equal sum of `inferences_by_endpoint` values

**Relationships**:
- 1:1 with SessionState (single statistics object per session)

---

### 7. SessionState

**Purpose**: Global application state, coordinates all components

**Source**: Singleton created at application startup, persists for session lifetime

**Schema**:
```python
from pydantic import BaseModel, Field
from typing import Optional

class SessionState(BaseModel):
    """Global application state"""

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
    session_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    def register_model(self, model: ModelInfo):
        """
        Add model to registry with O(1) access (per FR-049)

        Args:
            model: Discovered model to register
        """
        self.models[model.model_id] = model

        provider_models = self.models_by_provider.setdefault(model.provider, [])
        if model.model_id not in provider_models:
            provider_models.append(model.model_id)

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """O(1) model lookup"""
        return self.models.get(model_id)

    def get_models_by_provider(self, provider: ProviderType) -> list[ModelInfo]:
        """O(1) filter by provider + O(N) model object retrieval"""
        model_ids = self.models_by_provider.get(provider, [])
        return [self.models[mid] for mid in model_ids]

    def load_model(self, model_info: ModelInfo, device: str, handle: Any) -> LoadedModel:
        """
        Load model into memory (enforces single-model constraint per FR-016)

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
            _handle=handle
        )

        return self.loaded_model

    def unload_model(self):
        """
        Unload current model and free memory (per FR-016, FR-017)
        """
        if self.loaded_model is not None:
            self.loaded_model.unload()
            self.loaded_model = None

    class Config:
        arbitrary_types_allowed = True  # For provider handles
```

**Validation Rules**:
- Only one `loaded_model` allowed at a time (enforced by `load_model()`)
- `models` dict keys must match `ModelInfo.model_id` values
- `system_specs` required and immutable after creation

**Relationships**:
- 1:1 with SystemSpecs
- 1:1 with SessionStatistics
- 1:N with ProviderConfig
- 1:N with ModelInfo
- 1:1 with LoadedModel (when model active)

---

## Serialization & Persistence

### JSON Schema Export

All Pydantic models support automatic JSON schema generation for documentation and validation:

```python
from pydantic import TypeAdapter

# Export schema for external tools
model_schema = TypeAdapter(ModelInfo).json_schema()
```

### File Persistence

**InferenceResult Persistence** (per FR-041, FR-042):
```
results/
├── 20251011_143052_001234.json
├── 20251011_143125_002345.json
└── annotated_images/
    ├── 20251011_143052_detect_car.jpg
    └── 20251011_143125_point_sign.jpg
```

**Config Persistence** (`config.yaml`):
```yaml
providers:
  ollama:
    name: ollama
    enabled: true
    host: "http://localhost:11434"
    timeout_seconds: 120

  huggingface:
    name: huggingface
    enabled: true
    cache_dir: "~/.cache/huggingface/hub/"
    device_preference: ["cuda", "mps", "cpu"]
```

---

## Validation Summary

| Entity | Key Constraints | Error Handling |
|--------|----------------|----------------|
| ProviderConfig | `host` XOR `cache_dir` required | Raise `ValidationError` on load |
| ModelInfo | `size_gb` >0, `capabilities` non-empty | Log warning, skip model |
| SystemSpecs | `available_ram` ≤ `total_ram` | Raise error on detection failure |
| LoadedModel | Single instance only | Raise `RuntimeError` if violated |
| InferenceResult | `inference_time_ms` >0 | Raise `ValidationError` on record |
| SessionStatistics | Cumulative sums consistent | Auto-correct on mismatch (log warning) |
| SessionState | Model registry consistency | Raise error on duplicate IDs |

---

## Constitution Compliance

✅ **I. Provider Abstraction**: All providers produce `ModelInfo` with standardized `capabilities`
✅ **II. Configuration-Driven**: `ProviderConfig` loaded from YAML, no hardcoded values
✅ **III. Production-Ready Reliability**: Pydantic validation prevents invalid states
✅ **IV. Interactive User Experience**: Models include `@computed_field` for display formatting
✅ **V. Extensibility First**: Add new `EndpointType` enum values without breaking existing code

---

## Next Steps

1. Implement provider interfaces in `/contracts/provider.py`
2. Create base classes for provider implementations
3. Generate quickstart documentation with example model workflows

**Phase 1 Data Model Complete** ✅
