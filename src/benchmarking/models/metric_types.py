"""
Enum types and constants for benchmarking module.

Defines all enumeration types used throughout the benchmarking system,
including suite types, model types, execution modes, result statuses,
endpoint types, and metric units.
"""

from enum import Enum


class SuiteType(Enum):
    """Benchmark suite types available for execution."""

    SPEED = "speed"  # Latency and throughput metrics
    RESOURCES = "resources"  # CPU, GPU, memory efficiency
    QUALITY = "quality"  # Output consistency and similarity
    STRESS = "stress"  # Continuous running and degradation
    COMPLETE = "complete"  # All suites combined

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "SuiteType":
        """Convert string to SuiteType enum."""
        for suite in cls:
            if suite.value == value.lower():
                return suite
        raise ValueError(f"Unknown suite type: {value}")

    def display_name(self) -> str:
        """Return user-friendly display name."""
        names = {
            "speed": "Speed & Throughput",
            "resources": "Resource Efficiency",
            "quality": "Quality & Consistency",
            "stress": "Stress & Endurance",
            "complete": "Complete System Analysis",
        }
        return names.get(self.value, self.value)


class ModelType(Enum):
    """Type of model being benchmarked."""

    LLM = "llm"  # Large Language Model (text-only)
    VLM = "vlm"  # Vision Language Model (image + text)

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        """Convert string to ModelType enum."""
        for model_type in cls:
            if model_type.value == value.lower():
                return model_type
        raise ValueError(f"Unknown model type: {value}")

    def display_name(self) -> str:
        """Return user-friendly display name."""
        return {
            "llm": "Large Language Model (LLM)",
            "vlm": "Vision Language Model (VLM)",
        }.get(self.value, self.value)


class ExecutionMode(Enum):
    """How the benchmark is executed."""

    FOREGROUND = "foreground"  # Blocking, real-time progress display
    BACKGROUND = "background"  # Non-blocking, background thread

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "ExecutionMode":
        """Convert string to ExecutionMode enum."""
        for mode in cls:
            if mode.value == value.lower():
                return mode
        raise ValueError(f"Unknown execution mode: {value}")


class ResultStatus(Enum):
    """Status of benchmark execution result."""

    SUCCESS = "success"  # All runs completed successfully
    PARTIAL = "partial"  # Some runs completed, some failed
    FAILED = "failed"  # No runs completed successfully
    CANCELLED = "cancelled"  # User cancelled execution
    TIMEOUT = "timeout"  # Exceeded global timeout

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "ResultStatus":
        """Convert string to ResultStatus enum."""
        for status in cls:
            if status.value == value.lower():
                return status
        raise ValueError(f"Unknown result status: {value}")

    def is_success(self) -> bool:
        """Check if status indicates successful completion."""
        return self == ResultStatus.SUCCESS

    def is_partial(self) -> bool:
        """Check if status indicates partial completion."""
        return self == ResultStatus.PARTIAL


class EndpointType(Enum):
    """Type of model endpoint."""

    OLLAMA = "ollama"  # Ollama local inference
    OPENAI_COMPATIBLE = "openai_compatible"  # OpenAI-compatible API
    CUSTOM = "custom"  # Custom endpoint
    HUGGINGFACE = "huggingface"  # HuggingFace Inference API

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "EndpointType":
        """Convert string to EndpointType enum."""
        for endpoint in cls:
            if endpoint.value == value.lower():
                return endpoint
        raise ValueError(f"Unknown endpoint type: {value}")


class MetricUnit(Enum):
    """Units for various metrics."""

    # Time metrics
    MILLISECONDS = "ms"
    SECONDS = "s"
    MS = "ms"  # Alias for MILLISECONDS
    S = "s"  # Alias for SECONDS

    # Throughput metrics
    TOKENS_PER_SECOND = "tokens/s"
    REQUESTS_PER_SECOND = "req/s"

    # Resource metrics
    PERCENT = "%"
    MEGABYTES = "MB"
    GIGABYTES = "GB"
    CELSIUS = "°C"
    WATTS = "W"
    JOULES = "J"
    MB = "MB"  # Alias for MEGABYTES
    GB = "GB"  # Alias for GIGABYTES

    # Quality metrics
    RATIO = "ratio"
    SCORE = "score"

    # Generic
    COUNT = "count"
    BOOL = "bool"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "MetricUnit":
        """Convert string to MetricUnit enum."""
        for unit in cls:
            if unit.value == value.lower():
                return unit
        raise ValueError(f"Unknown metric unit: {value}")


# Constants for benchmarking configuration
DEFAULT_NUM_RUNS = 5
DEFAULT_NUM_WARMUP = 1
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_REFRESH_INTERVAL_MS = 5000  # 5 seconds for TUI refresh

# Timeout constants
INFERENCE_TIMEOUT_SECONDS = 180
SUITE_TIMEOUT_SECONDS = 3600  # 1 hour per suite

# Metric thresholds for alerts
MEMORY_WARNING_MB = 8000
CPU_WARNING_PERCENT = 90.0
GPU_WARNING_PERCENT = 95.0
TEMPERATURE_WARNING_CELSIUS = 80.0
