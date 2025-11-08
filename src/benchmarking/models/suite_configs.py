"""
Suite-specific configuration models.

Provides typed configuration classes for each benchmark suite with detailed
parameter validation and documentation.
"""

from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator


class SpeedConfig(BaseModel):
    """
    Configuration for Speed & Throughput benchmark suite.

    Controls timing measurements, retry behavior, and resource requirements
    for latency and throughput testing.
    """

    # Core execution parameters
    num_runs: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of timed inference runs per model/endpoint"
    )
    num_warmup: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of warmup runs before measurement starts"
    )

    # Timeout and retry configuration
    timeout_seconds: int = Field(
        default=600,
        ge=10,
        le=3600,
        description="Maximum time to wait for a single inference (10s-1hr)"
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Number of retry attempts on inference failure"
    )
    retry_backoff_seconds: float = Field(
        default=2.0,
        ge=0.1,
        le=60.0,
        description="Delay between retry attempts"
    )

    # Memory management
    min_memory_mb: float = Field(
        default=250.0,
        ge=50.0,
        le=32000.0,
        description="Minimum available memory required before starting a run"
    )

    # Timing metrics selection
    timing_metrics_to_capture: List[str] = Field(
        default_factory=lambda: [
            "latency_ms",
            "ttft_ms",
            "itl_ms",
            "prefill_latency_ms",
            "decode_latency_ms"
        ],
        description="Which timing metrics to record"
    )

    @field_validator("timing_metrics_to_capture")
    @classmethod
    def validate_timing_metrics(cls, v):
        """Ensure timing metrics are valid."""
        valid_metrics = {
            "latency_ms",  # Total inference time (always required)
            "ttft_ms",  # Time to First Token
            "itl_ms",  # Inter-Token Latency
            "prefill_latency_ms",  # Prompt processing time
            "decode_latency_ms",  # Token generation time
        }

        # latency_ms is always required
        if "latency_ms" not in v:
            v.insert(0, "latency_ms")

        # Validate all metrics are known
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Invalid timing metric: {metric}. "
                    f"Valid options: {', '.join(sorted(valid_metrics))}"
                )

        return v

    def should_capture_metric(self, metric_name: str) -> bool:
        """Check if a specific timing metric should be captured."""
        return metric_name in self.timing_metrics_to_capture


class ResourcesConfig(BaseModel):
    """
    Configuration for Resource Efficiency benchmark suite.

    Controls CPU/GPU/memory monitoring, sampling rates, and capability selection.
    """

    # Core execution parameters
    num_runs: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of inference runs to monitor"
    )
    num_warmup: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of warmup runs before monitoring starts"
    )

    # Timeout configuration
    timeout_seconds: int = Field(
        default=600,
        ge=10,
        le=3600,
        description="Maximum time to wait for a single inference"
    )

    # Monitoring configuration
    monitor_sampling_rate_ms: int = Field(
        default=100,
        ge=10,
        le=5000,
        description="How often to sample resource metrics during inference (milliseconds)"
    )

    # Capability selection
    capabilities_to_monitor: List[str] = Field(
        default_factory=lambda: ["cpu", "memory", "gpu", "temperature", "power"],
        description="Which hardware metrics to track"
    )

    @field_validator("capabilities_to_monitor")
    @classmethod
    def validate_capabilities(cls, v):
        """Ensure capabilities are valid."""
        valid_capabilities = {"cpu", "memory", "gpu", "temperature", "power"}

        # cpu and memory are always required
        required = {"cpu", "memory"}
        for cap in required:
            if cap not in v:
                v.insert(0, cap)

        # Validate all capabilities are known
        for cap in v:
            if cap not in valid_capabilities:
                raise ValueError(
                    f"Invalid capability: {cap}. "
                    f"Valid options: {', '.join(sorted(valid_capabilities))}"
                )

        return v

    def should_monitor_capability(self, capability_name: str) -> bool:
        """Check if a specific capability should be monitored."""
        return capability_name in self.capabilities_to_monitor


class QualityConfig(BaseModel):
    """
    Configuration for Quality & Consistency benchmark suite.

    Controls output consistency testing, quality metric computation,
    and semantic analysis parameters.

    Note: This suite has two distinct dimensions:
    - outputs_per_prompt: How many outputs to generate for EACH prompt
    - num_prompts_to_test: How many DIFFERENT prompts to test
    """

    # Core execution parameters
    outputs_per_prompt: int = Field(
        default=10,
        ge=5,
        le=100,
        description="Number of outputs to generate PER PROMPT for consistency analysis"
    )
    num_warmup: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of warmup runs before quality testing starts"
    )

    # Prompt selection
    num_prompts_to_test: int = Field(
        default=-1,
        ge=-1,
        le=1000,
        description="Number of different prompts to test (-1 = all available prompts)"
    )
    prompt_selection_strategy: Literal["first_n", "random_n", "evenly_spaced"] = Field(
        default="first_n",
        description="How to select which prompts to test from the available list"
    )

    # Quality metrics selection
    quality_metrics_to_compute: List[str] = Field(
        default_factory=lambda: [
            "bleu",
            "rouge",
            "semantic_similarity",
            "length_variance"
        ],
        description="Which quality metrics to calculate"
    )

    # Semantic analysis configuration
    semantic_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for semantic similarity"
    )
    semantic_outlier_threshold_sigma: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="Standard deviation threshold for semantic outlier detection"
    )

    # Timeout configuration
    timeout_seconds: int = Field(
        default=600,
        ge=10,
        le=3600,
        description="Maximum time to wait for a single inference"
    )

    @field_validator("quality_metrics_to_compute")
    @classmethod
    def validate_quality_metrics(cls, v):
        """Ensure quality metrics are valid."""
        valid_metrics = {
            "bleu",  # BLEU score (NLTK)
            "rouge",  # ROUGE score
            "semantic_similarity",  # Embedding-based similarity
            "length_variance",  # Output length consistency
        }

        # Validate all metrics are known
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Invalid quality metric: {metric}. "
                    f"Valid options: {', '.join(sorted(valid_metrics))}"
                )

        return v

    @field_validator("semantic_embedding_model")
    @classmethod
    def validate_embedding_model(cls, v):
        """Ensure embedding model is supported."""
        supported_models = {
            "all-MiniLM-L6-v2",  # 90MB, fast, good quality (default)
            "all-mpnet-base-v2",  # 420MB, slower, best quality
            "paraphrase-MiniLM-L3-v2",  # 60MB, fastest, lower quality
        }

        if v not in supported_models:
            raise ValueError(
                f"Unsupported embedding model: {v}. "
                f"Supported models: {', '.join(sorted(supported_models))}"
            )

        return v

    def should_compute_metric(self, metric_name: str) -> bool:
        """Check if a specific quality metric should be computed."""
        return metric_name in self.quality_metrics_to_compute

    def get_total_outputs_per_prompt(self) -> int:
        """Get total outputs per prompt (warmup + counted)."""
        return self.num_warmup + self.outputs_per_prompt


class StressConfig(BaseModel):
    """
    Configuration for Stress & Endurance benchmark suite.

    Controls continuous stress testing duration, stability thresholds,
    and degradation analysis parameters.
    """

    # Core test duration (PER MODEL)
    duration_minutes: int = Field(
        default=15,
        ge=1,
        le=1440,  # Max 24 hours
        description="How long to run continuous stress testing per model (minutes)"
    )

    # Baseline measurement
    baseline_measurement_runs: int = Field(
        default=5,
        ge=3,
        le=50,
        description="Number of initial runs used to establish baseline performance"
    )

    # Memory management
    cleanup_frequency_runs: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Perform memory cleanup every N runs to prevent creep"
    )

    # Progress tracking
    checkpoint_interval_seconds: int = Field(
        default=300,  # 5 minutes
        ge=30,
        le=3600,
        description="How often to log progress checkpoints during stress test"
    )

    # Timeout configuration
    timeout_seconds: int = Field(
        default=600,
        ge=10,
        le=3600,
        description="Maximum time to wait for a single inference"
    )

    # Stability assessment thresholds
    max_latency_degradation_percent: float = Field(
        default=20.0,
        ge=0.0,
        le=200.0,
        description="Fail if average latency increases >X% from baseline"
    )
    max_memory_growth_percent: float = Field(
        default=50.0,
        ge=0.0,
        le=500.0,
        description="Fail if average memory usage increases >X% from baseline"
    )
    max_error_rate_percent: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Fail if >X% of runs encounter errors"
    )

    def get_stability_thresholds(self) -> Dict[str, float]:
        """Get stability thresholds as a dictionary."""
        return {
            "max_latency_degradation_percent": self.max_latency_degradation_percent,
            "max_memory_growth_percent": self.max_memory_growth_percent,
            "max_error_rate_percent": self.max_error_rate_percent,
        }

    def get_checkpoint_interval_minutes(self) -> float:
        """Get checkpoint interval in minutes."""
        return self.checkpoint_interval_seconds / 60.0


class CompleteConfig(BaseModel):
    """
    Configuration for Complete System Analysis suite.

    Orchestrates all benchmark suites with hierarchical configuration,
    allowing customization of suite selection, execution order, and
    per-suite parameters.
    """

    # Suite orchestration
    suites_to_run: List[str] = Field(
        default_factory=lambda: ["speed", "resources", "quality", "stress"],
        description="Which sub-suites to execute"
    )
    suite_execution_order: List[str] = Field(
        default_factory=lambda: ["speed", "resources", "quality", "stress"],
        description="Order to run suites in"
    )
    stop_on_suite_failure: bool = Field(
        default=False,
        description="Whether to abort if a sub-suite fails"
    )

    # Per-suite configurations (optional - will use defaults if not specified)
    speed: Optional[SpeedConfig] = Field(
        default=None,
        description="Speed suite configuration (uses defaults if not specified)"
    )
    resources: Optional[ResourcesConfig] = Field(
        default=None,
        description="Resources suite configuration (uses defaults if not specified)"
    )
    quality: Optional[QualityConfig] = Field(
        default=None,
        description="Quality suite configuration (uses defaults if not specified)"
    )
    stress: Optional[StressConfig] = Field(
        default=None,
        description="Stress suite configuration (uses defaults if not specified)"
    )

    @field_validator("suites_to_run")
    @classmethod
    def validate_suites_to_run(cls, v):
        """Ensure suites are valid."""
        valid_suites = {"speed", "resources", "quality", "stress"}

        for suite in v:
            if suite not in valid_suites:
                raise ValueError(
                    f"Invalid suite: {suite}. "
                    f"Valid options: {', '.join(sorted(valid_suites))}"
                )

        if not v:
            raise ValueError("At least one suite must be selected")

        return v

    @field_validator("suite_execution_order")
    @classmethod
    def validate_suite_execution_order(cls, v, info):
        """Ensure execution order matches selected suites."""
        if "suites_to_run" in info.data:
            selected = set(info.data["suites_to_run"])
            ordered = set(v)

            if selected != ordered:
                raise ValueError(
                    f"suite_execution_order must contain exactly the same suites "
                    f"as suites_to_run. Selected: {selected}, Ordered: {ordered}"
                )

        return v

    def get_suite_config(self, suite_name: str) -> Optional[BaseModel]:
        """Get configuration for a specific suite."""
        suite_map = {
            "speed": self.speed,
            "resources": self.resources,
            "quality": self.quality,
            "stress": self.stress,
        }
        return suite_map.get(suite_name)

    def get_or_create_suite_config(self, suite_name: str) -> BaseModel:
        """Get configuration for a suite, creating default if not specified."""
        config = self.get_suite_config(suite_name)
        if config is None:
            # Create default config
            if suite_name == "speed":
                return SpeedConfig()
            elif suite_name == "resources":
                return ResourcesConfig()
            elif suite_name == "quality":
                return QualityConfig()
            elif suite_name == "stress":
                return StressConfig()
        return config


# Helper functions for backward compatibility with parameters dict

def speed_config_from_params(params: Dict[str, Any]) -> SpeedConfig:
    """Create SpeedConfig from parameters dict (backward compatibility)."""
    speed_params = params.get("speed", {})

    # Support top-level params for backward compatibility
    config_data = {
        "num_runs": params.get("num_runs", speed_params.get("num_runs", 10)),
        "num_warmup": params.get("num_warmup", speed_params.get("num_warmup", 3)),
        "timeout_seconds": speed_params.get("timeout_seconds", 600),
        "max_retries": speed_params.get("max_retries", 2),
        "retry_backoff_seconds": speed_params.get("retry_backoff_seconds", 2.0),
        "min_memory_mb": speed_params.get("min_memory_mb", 250.0),
        "timing_metrics_to_capture": speed_params.get(
            "timing_metrics_to_capture",
            ["latency_ms", "ttft_ms", "itl_ms", "prefill_latency_ms", "decode_latency_ms"]
        ),
    }

    return SpeedConfig(**config_data)


def resources_config_from_params(params: Dict[str, Any]) -> ResourcesConfig:
    """Create ResourcesConfig from parameters dict (backward compatibility)."""
    resources_params = params.get("resources", {})

    config_data = {
        "num_runs": params.get("num_runs", resources_params.get("num_runs", 10)),
        "num_warmup": params.get("num_warmup", resources_params.get("num_warmup", 3)),
        "timeout_seconds": resources_params.get("timeout_seconds", 600),
        "monitor_sampling_rate_ms": resources_params.get("monitor_sampling_rate_ms", 100),
        "capabilities_to_monitor": resources_params.get(
            "capabilities_to_monitor",
            ["cpu", "memory", "gpu", "temperature", "power"]
        ),
    }

    return ResourcesConfig(**config_data)


def quality_config_from_params(params: Dict[str, Any]) -> QualityConfig:
    """Create QualityConfig from parameters dict (backward compatibility)."""
    quality_params = params.get("quality", {})

    config_data = {
        # Map old num_runs to new outputs_per_prompt
        "outputs_per_prompt": quality_params.get(
            "outputs_per_prompt",
            params.get("num_runs", 5)  # Fallback to top-level num_runs
        ),
        "num_warmup": params.get("num_warmup", quality_params.get("num_warmup", 3)),
        "num_prompts_to_test": quality_params.get("num_prompts_to_test", -1),
        "prompt_selection_strategy": quality_params.get("prompt_selection_strategy", "first_n"),
        "quality_metrics_to_compute": quality_params.get(
            "quality_metrics_to_compute",
            ["bleu", "rouge", "semantic_similarity", "length_variance"]
        ),
        "semantic_embedding_model": quality_params.get("semantic_embedding_model", "all-MiniLM-L6-v2"),
        "semantic_outlier_threshold_sigma": quality_params.get("semantic_outlier_threshold_sigma", 2.0),
        "timeout_seconds": quality_params.get("timeout_seconds", 600),
    }

    return QualityConfig(**config_data)


def stress_config_from_params(params: Dict[str, Any]) -> StressConfig:
    """Create StressConfig from parameters dict (backward compatibility)."""
    stress_params = params.get("stress", {})

    # Support old parameter names
    duration = stress_params.get(
        "duration_minutes",
        params.get("duration_minutes", 30)  # Fallback to top-level
    )

    config_data = {
        "duration_minutes": duration,
        "baseline_measurement_runs": stress_params.get("baseline_measurement_runs", 5),
        "cleanup_frequency_runs": stress_params.get("cleanup_frequency_runs", 10),
        "checkpoint_interval_seconds": stress_params.get(
            "checkpoint_interval_seconds",
            params.get("checkpoint_interval_seconds", 300)  # Fallback to top-level
        ),
        "timeout_seconds": stress_params.get("timeout_seconds", 600),
        "max_latency_degradation_percent": stress_params.get("max_latency_degradation_percent", 20.0),
        "max_memory_growth_percent": stress_params.get("max_memory_growth_percent", 50.0),
        "max_error_rate_percent": stress_params.get("max_error_rate_percent", 10.0),
    }

    return StressConfig(**config_data)


def complete_config_from_params(params: Dict[str, Any]) -> CompleteConfig:
    """Create CompleteConfig from parameters dict (backward compatibility)."""
    complete_params = params.get("complete", {})

    # Build per-suite configs
    speed_cfg = None
    if "speed" in params or "speed_runs" in params:
        speed_cfg = speed_config_from_params(params)
        # Override num_runs if speed_runs is specified
        if "speed_runs" in params:
            speed_cfg.num_runs = params["speed_runs"]

    resources_cfg = None
    if "resources" in params or "resources_runs" in params:
        resources_cfg = resources_config_from_params(params)
        if "resources_runs" in params:
            resources_cfg.num_runs = params["resources_runs"]

    quality_cfg = None
    if "quality" in params or "quality_runs" in params:
        quality_cfg = quality_config_from_params(params)
        if "quality_runs" in params:
            quality_cfg.outputs_per_prompt = params["quality_runs"]

    stress_cfg = None
    if "stress" in params or "stress_duration_minutes" in params:
        stress_cfg = stress_config_from_params(params)
        if "stress_duration_minutes" in params:
            stress_cfg.duration_minutes = params["stress_duration_minutes"]

    config_data = {
        "suites_to_run": complete_params.get("suites_to_run", ["speed", "resources", "quality", "stress"]),
        "suite_execution_order": complete_params.get(
            "suite_execution_order",
            ["speed", "resources", "quality", "stress"]
        ),
        "stop_on_suite_failure": complete_params.get("stop_on_suite_failure", False),
        "speed": speed_cfg,
        "resources": resources_cfg,
        "quality": quality_cfg,
        "stress": stress_cfg,
    }

    return CompleteConfig(**config_data)
