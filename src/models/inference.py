"""Inference input, output, and result models."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .endpoints import EndpointType, ProviderType


class InferenceInput(BaseModel):
    """Input parameters for inference."""

    endpoint: EndpointType

    # Text inputs (all endpoints)
    prompt: Optional[str] = None  # LLM text or VLM question
    object_name: Optional[str] = None  # detect/point target

    # Image inputs (VLM endpoints)
    image_path: Optional[str] = None

    # Endpoint-specific parameters
    detail_level: Optional[str] = Field(
        default="detailed", pattern="^(detailed|short)$"
    )  # caption


class InferenceOutput(BaseModel):
    """Output from inference."""

    # Text outputs
    text_response: Optional[str] = None  # QA answer, caption, text generation

    # Vision outputs
    detections: Optional[list[dict[str, Any]]] = (
        None  # detect: [{"label": "car", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}]
    )
    coordinates: Optional[dict[str, Any]] = (
        None  # point: {"x": 120, "y": 450} or {"bbox": [x1,y1,x2,y2]}
    )

    # Annotated image paths (detect/point)
    annotated_image_path: Optional[str] = None


class InferenceResult(BaseModel):
    """Complete record of a single inference."""

    # Identity
    result_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        description="Unique ID for result persistence",
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
    saved_to_file: Optional[str] = None  # JSON file path if user saved

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize for file persistence (per FR-042)."""
        return {
            "result_id": self.result_id,
            "model": {"id": self.model_id, "provider": self.provider, "device": self.device},
            "endpoint": self.input.endpoint,
            "input": self.input.model_dump(exclude_none=True),
            "output": self.output.model_dump(exclude_none=True),
            "performance": {
                "inference_time_ms": self.inference_time_ms,
                "timestamp": self.timestamp.isoformat(),
            },
            "metadata": {"generated_by": "VLM CLI v0.1.0"},
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "InferenceResult":
        """Deserialize from saved JSON."""
        return cls(
            result_id=data["result_id"],
            model_id=data["model"]["id"],
            provider=ProviderType(data["model"]["provider"]),
            device=data["model"]["device"],
            input=InferenceInput(**data["input"]),
            output=InferenceOutput(**data["output"]),
            inference_time_ms=data["performance"]["inference_time_ms"],
            timestamp=datetime.fromisoformat(data["performance"]["timestamp"]),
        )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
