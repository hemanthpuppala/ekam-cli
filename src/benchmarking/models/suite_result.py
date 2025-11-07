"""
Data models for benchmark results and metrics.

Defines Pydantic models for:
- PerformanceMetric: Individual metric observations
- AggregateStats: Computed statistics from multiple observations
- SuiteResult: Complete result from a benchmark suite execution
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import json

from src.benchmarking.models.metric_types import (
    SuiteType,
    ModelType,
    ResultStatus,
    MetricUnit,
)


class PerformanceMetric(BaseModel):
    """Individual performance metric observation."""

    name: str = Field(..., description="Metric name (e.g., 'latency_ms', 'cpu_percent')")
    value: float = Field(..., description="Metric value")
    unit: MetricUnit = Field(..., description="Unit of measurement")
    run_number: int = Field(..., ge=1, description="Run number this metric is from")
    model_id: str = Field(..., description="Model ID being benchmarked")
    endpoint: str = Field(..., description="Endpoint used for inference")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When metric was recorded")
    is_warmup: bool = Field(default=False, description="Whether this is from a warmup run")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic configuration."""
        use_enum_values = False
        arbitrary_types_allowed = True


class AggregateStats(BaseModel):
    """Computed statistics from multiple metric observations."""

    metric_name: str = Field(..., description="Name of the metric being aggregated")
    unit: MetricUnit = Field(..., description="Unit of measurement")
    count: int = Field(..., ge=0, description="Number of observations")

    # Central tendency
    mean: float = Field(..., description="Arithmetic mean")
    median: float = Field(..., description="Median value")
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")

    # Spread
    std_dev: Optional[float] = Field(default=None, description="Standard deviation")
    variance: Optional[float] = Field(default=None, description="Variance")

    # Percentiles
    p25: Optional[float] = Field(default=None, description="25th percentile")
    p50: Optional[float] = Field(default=None, description="50th percentile (median)")
    p75: Optional[float] = Field(default=None, description="75th percentile")
    p95: Optional[float] = Field(default=None, description="95th percentile")
    p99: Optional[float] = Field(default=None, description="99th percentile")

    class Config:
        """Pydantic configuration."""
        use_enum_values = False
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper formatting."""
        data = self.model_dump()
        data['unit'] = self.unit.value if isinstance(self.unit, MetricUnit) else self.unit
        return data


class SuiteResult(BaseModel):
    """Complete result from a benchmark suite execution."""

    # Identification
    suite_type: SuiteType = Field(..., description="Type of benchmark suite executed")
    benchmark_id: str = Field(..., description="Unique benchmark execution ID")
    model_type: ModelType = Field(..., description="Type of models benchmarked")

    # Execution info
    start_time: datetime = Field(..., description="When the suite started")
    end_time: Optional[datetime] = Field(default=None, description="When the suite completed")
    status: ResultStatus = Field(..., description="Final execution status")
    error_message: Optional[str] = Field(default=None, description="Error message if status is FAILED")

    # Models and runs
    models_tested: List[str] = Field(..., description="List of model IDs tested")
    endpoints_tested: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Endpoints per model"
    )
    total_runs: int = Field(..., ge=0, description="Total runs attempted")
    successful_runs: int = Field(..., ge=0, description="Total runs that succeeded")
    failed_runs: int = Field(..., ge=0, description="Total runs that failed")

    # Raw metrics
    metrics: List[PerformanceMetric] = Field(
        default_factory=list,
        description="All individual metric observations"
    )

    # Aggregated statistics - three levels
    aggregates_global: Dict[str, AggregateStats] = Field(
        default_factory=dict,
        description="Global aggregates across all runs"
    )
    aggregates_per_model: Dict[str, Dict[str, AggregateStats]] = Field(
        default_factory=dict,
        description="Aggregates per model {model_id: {metric: AggregateStats}}"
    )
    aggregates_per_endpoint: Dict[str, Dict[str, AggregateStats]] = Field(
        default_factory=dict,
        description="Aggregates per endpoint {endpoint: {metric: AggregateStats}}"
    )

    # Separate warmup tracking
    aggregates_warmup: Dict[str, AggregateStats] = Field(
        default_factory=dict,
        description="Aggregates from warmup runs only"
    )
    aggregates_counted: Dict[str, AggregateStats] = Field(
        default_factory=dict,
        description="Aggregates from counted (non-warmup) runs only"
    )

    # Configuration snapshot
    config_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Copy of BenchmarkConfig used"
    )

    # Additional info
    duration_seconds: Optional[float] = Field(default=None, description="Total execution time")
    system_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="System specs at execution time"
    )
    notes: Optional[str] = Field(default=None, description="Human-readable notes about the result")

    class Config:
        """Pydantic configuration."""
        use_enum_values = False
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper formatting."""
        data = self.model_dump(exclude_unset=False)

        # Format datetime fields
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()

        # Format enums
        data['suite_type'] = self.suite_type.value
        data['model_type'] = self.model_type.value
        data['status'] = self.status.value

        # Format nested AggregateStats
        data['aggregates_global'] = {
            k: v.to_dict() for k, v in self.aggregates_global.items()
        }
        data['aggregates_warmup'] = {
            k: v.to_dict() for k, v in self.aggregates_warmup.items()
        }
        data['aggregates_counted'] = {
            k: v.to_dict() for k, v in self.aggregates_counted.items()
        }

        # Format per-model and per-endpoint aggregates
        data['aggregates_per_model'] = {
            model_id: {k: v.to_dict() for k, v in metrics.items()}
            for model_id, metrics in self.aggregates_per_model.items()
        }
        data['aggregates_per_endpoint'] = {
            endpoint: {k: v.to_dict() for k, v in metrics.items()}
            for endpoint, metrics in self.aggregates_per_endpoint.items()
        }

        # Format metrics
        data['metrics'] = [m.model_dump() for m in self.metrics]

        return data

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        def json_serializer(obj):
            """Handle non-standard types."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Type {type(obj)} not serializable")

        return json.dumps(self.to_dict(), indent=indent, default=json_serializer)

    def compute_duration(self) -> float:
        """Compute duration from start/end times."""
        if self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds()
        return 0.0

    def success_rate(self) -> float:
        """Compute success rate percentage."""
        if self.total_runs == 0:
            return 0.0
        return (self.successful_runs / self.total_runs) * 100


# Import at end to avoid circular imports
from enum import Enum
