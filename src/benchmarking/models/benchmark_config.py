"""
Benchmark configuration data model.

Defines the BenchmarkConfig pydantic model for validating and managing
benchmark execution parameters.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json

from src.benchmarking.models.metric_types import (
    SuiteType,
    ModelType,
    ExecutionMode,
    DEFAULT_NUM_RUNS,
    DEFAULT_NUM_WARMUP,
)


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark execution."""

    # Identification
    benchmark_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this benchmark execution"
    )

    # Model and suite selection
    model_type: ModelType = Field(..., description="Type of models being benchmarked (LLM or VLM)")
    suite_type: SuiteType = Field(..., description="Benchmark suite to execute")

    # Models and endpoints
    models: List[str] = Field(..., min_length=1, description="List of model IDs to benchmark")
    endpoints: Dict[str, List[str]] = Field(
        ...,
        description="Endpoints per model: {model_id: [endpoint_names]}"
    )

    # Test data
    test_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Test data configuration (prompts, images, etc.)"
    )

    # Inference parameters
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model inference parameters (temperature, max_tokens, etc.)"
    )

    # Execution configuration
    num_runs: int = Field(
        default=DEFAULT_NUM_RUNS,
        ge=1,
        le=1000,
        description="Number of inference runs per model (counted runs)"
    )
    num_warmup: int = Field(
        default=DEFAULT_NUM_WARMUP,
        ge=0,
        le=10,
        description="Number of warmup runs before counted runs"
    )

    # Execution mode
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.FOREGROUND,
        description="Execution mode (foreground or background)"
    )

    # Export configuration
    export_formats: List[str] = Field(
        default_factory=lambda: ["json"],
        description="Output formats for results (json, csv, html)"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this configuration was created"
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of this benchmark"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for organizing benchmarks"
    )

    model_config = ConfigDict(use_enum_values=False, arbitrary_types_allowed=True)

    @field_validator("models")
    @classmethod
    def validate_models_not_empty(cls, v):
        """Ensure models list is not empty."""
        if not v or len(v) == 0:
            raise ValueError("At least one model must be specified")
        return v

    @field_validator("endpoints")
    @classmethod
    def validate_endpoints_match_models(cls, v, info):
        """Ensure all models have endpoint definitions."""
        if "models" in info.data:
            models = info.data["models"]
            for model_id in models:
                if model_id not in v:
                    raise ValueError(f"No endpoints defined for model: {model_id}")
                if not v[model_id] or len(v[model_id]) == 0:
                    raise ValueError(f"At least one endpoint must be defined for model: {model_id}")
        return v

    @field_validator("export_formats")
    @classmethod
    def validate_export_formats(cls, v):
        """Validate export format values."""
        valid_formats = {"json", "csv", "html"}
        for fmt in v:
            if fmt.lower() not in valid_formats:
                raise ValueError(f"Invalid export format: {fmt}. Must be one of {valid_formats}")
        return [fmt.lower() for fmt in v]

    @field_validator("num_runs")
    @classmethod
    def validate_num_runs_reasonable(cls, v):
        """Ensure num_runs is reasonable."""
        if v < 1:
            raise ValueError("num_runs must be at least 1")
        if v > 1000:
            raise ValueError("num_runs should not exceed 1000 (performance concern)")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper enum serialization."""
        data = self.model_dump(exclude_unset=False)

        # Format datetime
        data["created_at"] = self.created_at.isoformat()

        # Format enums
        data["model_type"] = self.model_type.value
        data["suite_type"] = self.suite_type.value
        data["execution_mode"] = self.execution_mode.value

        return data

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        """Create BenchmarkConfig from dictionary."""
        # Convert string enums back to enum types
        if isinstance(data.get("model_type"), str):
            data["model_type"] = ModelType.from_string(data["model_type"])
        if isinstance(data.get("suite_type"), str):
            data["suite_type"] = SuiteType.from_string(data["suite_type"])
        if isinstance(data.get("execution_mode"), str):
            data["execution_mode"] = ExecutionMode.from_string(data["execution_mode"])

        # Convert datetime string back to datetime
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        return cls(**data)

    def get_total_runs_per_model(self) -> int:
        """Get total number of runs per model (warmup + counted)."""
        return self.num_warmup + self.num_runs

    def get_model_count(self) -> int:
        """Get number of models being benchmarked."""
        return len(self.models)

    def get_total_inferences(self) -> int:
        """Calculate total number of inferences that will be executed."""
        total = 0
        for model_id in self.models:
            num_endpoints = len(self.endpoints.get(model_id, []))
            total += num_endpoints * self.get_total_runs_per_model()
        return total

    def summary(self) -> str:
        """Generate human-readable summary of this configuration."""
        return (
            f"Benchmark: {self.benchmark_id}\n"
            f"Suite: {self.suite_type.display_name()}\n"
            f"Model Type: {self.model_type.display_name()}\n"
            f"Models: {', '.join(self.models)}\n"
            f"Runs per model: {self.num_runs} (+ {self.num_warmup} warmup)\n"
            f"Total inferences: {self.get_total_inferences()}\n"
            f"Execution mode: {self.execution_mode.value}\n"
            f"Export formats: {', '.join(self.export_formats)}"
        )
