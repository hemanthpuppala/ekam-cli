"""
Metrics collection and aggregation.

Provides the MetricsCollector class for recording individual metrics
and computing aggregate statistics (mean, median, percentiles, etc.).
"""

from typing import Dict, List, Optional
import numpy as np
from scipy import stats as scipy_stats

from src.benchmarking.models.suite_result import PerformanceMetric, AggregateStats
from src.benchmarking.models.metric_types import MetricUnit


class MetricsCollector:
    """
    Collects and aggregates performance metrics during benchmark execution.

    Supports:
    - Recording individual metric observations
    - Separate tracking of warmup vs counted runs
    - Hierarchical aggregation (global, per-model, per-endpoint)
    - Statistical computations (mean, median, percentiles, std dev)
    """

    def __init__(self):
        """Initialize empty metrics collector."""
        self._metrics: List[PerformanceMetric] = []

    def record_metric(
        self,
        name: str,
        value: float,
        unit: MetricUnit,
        run_number: int,
        model_id: str,
        endpoint: str,
        is_warmup: bool = False,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Record a single metric observation.

        Args:
            name: Metric name (e.g., 'latency_ms', 'tokens_per_sec')
            value: Metric value
            unit: Unit of measurement
            run_number: Run number this metric is from
            model_id: Model being benchmarked
            endpoint: Endpoint used
            is_warmup: Whether this is from a warmup run
            metadata: Additional metadata to attach
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            run_number=run_number,
            model_id=model_id,
            endpoint=endpoint,
            is_warmup=is_warmup,
            metadata=metadata or {}
        )
        self._metrics.append(metric)

    def get_metrics(
        self,
        name: Optional[str] = None,
        model_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        warmup_only: bool = False,
        counted_only: bool = False
    ) -> List[PerformanceMetric]:
        """
        Filter and retrieve metrics by criteria.

        Args:
            name: Filter by metric name
            model_id: Filter by model ID
            endpoint: Filter by endpoint
            warmup_only: Only warmup runs
            counted_only: Only counted (non-warmup) runs

        Returns:
            List of matching PerformanceMetric objects
        """
        filtered = self._metrics

        if name is not None:
            filtered = [m for m in filtered if m.name == name]

        if model_id is not None:
            filtered = [m for m in filtered if m.model_id == model_id]

        if endpoint is not None:
            filtered = [m for m in filtered if m.endpoint == endpoint]

        if warmup_only:
            filtered = [m for m in filtered if m.is_warmup]

        if counted_only:
            filtered = [m for m in filtered if not m.is_warmup]

        return filtered

    def get_aggregates(
        self,
        for_warmup: bool = False,
        model_id: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> Dict[str, AggregateStats]:
        """
        Compute aggregate statistics for metrics.

        Args:
            for_warmup: Compute for warmup runs (True) or counted runs (False)
            model_id: Compute for specific model only
            endpoint: Compute for specific endpoint only

        Returns:
            Dictionary mapping metric names to AggregateStats
        """
        # Get relevant metrics
        if for_warmup:
            metrics = self.get_metrics(warmup_only=True, model_id=model_id, endpoint=endpoint)
        else:
            metrics = self.get_metrics(counted_only=True, model_id=model_id, endpoint=endpoint)

        if not metrics:
            return {}

        # Group by metric name
        by_name: Dict[str, List[PerformanceMetric]] = {}
        for metric in metrics:
            if metric.name not in by_name:
                by_name[metric.name] = []
            by_name[metric.name].append(metric)

        # Compute aggregates for each metric
        aggregates = {}
        for metric_name, metric_list in by_name.items():
            aggregates[metric_name] = self._compute_aggregate(metric_name, metric_list)

        return aggregates

    def get_global_aggregates(
        self,
        other_collectors: Optional[List["MetricsCollector"]] = None
    ) -> Dict[str, AggregateStats]:
        """
        Compute global aggregates across all models/endpoints.

        Args:
            other_collectors: Other MetricsCollector instances to include

        Returns:
            Global aggregate statistics
        """
        all_metrics = list(self._metrics)

        if other_collectors:
            for collector in other_collectors:
                all_metrics.extend(collector._metrics)

        # Group by metric name
        by_name: Dict[str, List[PerformanceMetric]] = {}
        for metric in all_metrics:
            if not metric.is_warmup:  # Only counted runs for global
                if metric.name not in by_name:
                    by_name[metric.name] = []
                by_name[metric.name].append(metric)

        # Compute aggregates
        aggregates = {}
        for metric_name, metric_list in by_name.items():
            aggregates[metric_name] = self._compute_aggregate(metric_name, metric_list)

        return aggregates

    def _compute_aggregate(
        self,
        metric_name: str,
        metrics: List[PerformanceMetric]
    ) -> AggregateStats:
        """
        Compute aggregate statistics for a set of metrics.

        Args:
            metric_name: Name of the metric
            metrics: List of PerformanceMetric observations

        Returns:
            AggregateStats with computed statistics
        """
        if not metrics:
            raise ValueError(f"No metrics provided for {metric_name}")

        values = np.array([m.value for m in metrics])
        unit = metrics[0].unit

        # Basic statistics
        mean_val = float(np.mean(values))
        median_val = float(np.median(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        count = len(values)

        # Spread statistics
        std_dev = float(np.std(values, ddof=1)) if count > 1 else 0.0
        variance = float(np.var(values, ddof=1)) if count > 1 else 0.0

        # Percentiles (only if enough data)
        p25 = p50 = p75 = p95 = p99 = None
        if count >= 2:
            percentiles = np.percentile(values, [25, 50, 75, 95, 99])
            p25 = float(percentiles[0])
            p50 = float(percentiles[1])
            p75 = float(percentiles[2])
            p95 = float(percentiles[3])
            p99 = float(percentiles[4])

        return AggregateStats(
            metric_name=metric_name,
            unit=unit,
            count=count,
            mean=mean_val,
            median=median_val,
            min=min_val,
            max=max_val,
            std_dev=std_dev,
            variance=variance,
            p25=p25,
            p50=p50,
            p75=p75,
            p95=p95,
            p99=p99
        )

    def metrics_list(self) -> List[PerformanceMetric]:
        """Return all collected metrics."""
        return list(self._metrics)

    def to_dict(self) -> Dict:
        """Convert all metrics to dictionary format."""
        return {
            "metrics": [m.model_dump() for m in self._metrics],
            "total_count": len(self._metrics)
        }

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()

    def __len__(self) -> int:
        """Return number of collected metrics."""
        return len(self._metrics)

    def __repr__(self) -> str:
        """String representation."""
        return f"MetricsCollector(metrics={len(self._metrics)})"
