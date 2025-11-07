"""
Result aggregation engine for benchmark analysis.

Provides:
- Statistical aggregation across multiple runs
- Cross-model comparison
- Ranking and scoring
- Percentile calculations
- Dimension-based grouping (model, endpoint, suite)
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict

from loguru import logger

from ..models.suite_result import SuiteResult
from ..models.run_metrics import RunMetrics
from ..models.metric_types import ModelType, SuiteType


class AggregationDimension(str, Enum):
    """Dimensions for aggregating results."""

    MODEL = "model"
    ENDPOINT = "endpoint"
    SUITE = "suite"
    MODEL_TYPE = "model_type"


class RankingMetric(str, Enum):
    """Metrics for ranking models."""

    LATENCY = "latency"                # Lower is better
    THROUGHPUT = "throughput"          # Higher is better
    MEMORY_PEAK = "memory_peak"        # Lower is better
    CPU_USAGE = "cpu_usage"            # Lower is better
    GPU_USAGE = "gpu_usage"            # Lower is better
    SUCCESS_RATE = "success_rate"      # Higher is better


@dataclass
class AggregatedMetrics:
    """
    Statistical aggregation of metrics.

    Attributes:
        mean: Average value
        median: Median value
        std_dev: Standard deviation
        min_val: Minimum value
        max_val: Maximum value
        p25: 25th percentile
        p75: 75th percentile
        p95: 95th percentile
        p99: 99th percentile
        count: Number of samples
    """

    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    count: int = 0

    @classmethod
    def from_values(cls, values: List[float]) -> "AggregatedMetrics":
        """
        Create aggregated metrics from list of values.

        Args:
            values: List of numeric values

        Returns:
            AggregatedMetrics instance
        """
        if not values:
            return cls()

        sorted_values = sorted(values)
        count = len(sorted_values)

        return cls(
            mean=statistics.mean(sorted_values),
            median=statistics.median(sorted_values),
            std_dev=statistics.stdev(sorted_values) if count > 1 else 0.0,
            min_val=min(sorted_values),
            max_val=max(sorted_values),
            p25=cls._percentile(sorted_values, 25),
            p75=cls._percentile(sorted_values, 75),
            p95=cls._percentile(sorted_values, 95),
            p99=cls._percentile(sorted_values, 99),
            count=count
        )

    @staticmethod
    def _percentile(sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0

        k = (len(sorted_values) - 1) * (percentile / 100.0)
        f = int(k)
        c = f + 1

        if c >= len(sorted_values):
            return sorted_values[-1]

        d0 = sorted_values[f]
        d1 = sorted_values[c]

        return d0 + (d1 - d0) * (k - f)

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "mean": round(self.mean, 3),
            "median": round(self.median, 3),
            "std_dev": round(self.std_dev, 3),
            "min": round(self.min_val, 3),
            "max": round(self.max_val, 3),
            "p25": round(self.p25, 3),
            "p75": round(self.p75, 3),
            "p95": round(self.p95, 3),
            "p99": round(self.p99, 3),
            "count": self.count
        }


@dataclass
class ModelComparison:
    """
    Comparison between models.

    Attributes:
        model_id: Model identifier
        metrics: Dict of metric_name -> AggregatedMetrics
        rankings: Dict of metric_name -> rank (1 = best)
        scores: Dict of metric_name -> normalized score (0-100)
    """

    model_id: str
    metrics: Dict[str, AggregatedMetrics] = field(default_factory=dict)
    rankings: Dict[str, int] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "model_id": self.model_id,
            "metrics": {
                name: agg.to_dict()
                for name, agg in self.metrics.items()
            },
            "rankings": self.rankings,
            "scores": {k: round(v, 2) for k, v in self.scores.items()}
        }


class ResultAggregator:
    """
    Engine for aggregating and analyzing benchmark results.

    Provides:
    - Statistical aggregation across runs
    - Cross-model comparison
    - Ranking and scoring
    - Grouping by dimensions
    """

    def __init__(self):
        """Initialize result aggregator."""
        logger.debug("ResultAggregator initialized")

    def aggregate_suite_result(self, result: SuiteResult) -> Dict[str, AggregatedMetrics]:
        """
        Aggregate metrics within a single suite result.

        Args:
            result: SuiteResult to aggregate

        Returns:
            Dict of metric_name -> AggregatedMetrics
        """
        # Extract all run metrics
        all_runs: List[RunMetrics] = []
        for model_metrics in result.model_results.values():
            all_runs.extend(model_metrics.runs)

        if not all_runs:
            logger.warning(f"No runs found in result {result.benchmark_id}")
            return {}

        # Group values by metric
        metric_values: Dict[str, List[float]] = defaultdict(list)

        for run in all_runs:
            # Latency metrics
            if run.latency_ms is not None:
                metric_values["latency_ms"].append(run.latency_ms)
            if run.ttft_ms is not None:
                metric_values["ttft_ms"].append(run.ttft_ms)

            # Throughput metrics
            if run.tokens_per_second is not None:
                metric_values["tokens_per_second"].append(run.tokens_per_second)

            # Resource metrics
            if run.memory_peak_mb is not None:
                metric_values["memory_peak_mb"].append(run.memory_peak_mb)
            if run.cpu_usage_percent is not None:
                metric_values["cpu_usage_percent"].append(run.cpu_usage_percent)
            if run.gpu_usage_percent is not None:
                metric_values["gpu_usage_percent"].append(run.gpu_usage_percent)

        # Aggregate each metric
        aggregated = {}
        for metric_name, values in metric_values.items():
            aggregated[metric_name] = AggregatedMetrics.from_values(values)

        return aggregated

    def compare_models(
        self,
        results: List[SuiteResult],
        ranking_metric: RankingMetric = RankingMetric.LATENCY
    ) -> List[ModelComparison]:
        """
        Compare models across multiple benchmark results.

        Args:
            results: List of SuiteResult to compare
            ranking_metric: Primary metric for ranking

        Returns:
            List of ModelComparison sorted by ranking
        """
        # Group runs by model
        model_runs: Dict[str, List[RunMetrics]] = defaultdict(list)

        for result in results:
            for model_id, model_result in result.model_results.items():
                model_runs[model_id].extend(model_result.runs)

        # Aggregate metrics for each model
        comparisons: List[ModelComparison] = []

        for model_id, runs in model_runs.items():
            comparison = ModelComparison(model_id=model_id)

            # Group values by metric
            metric_values: Dict[str, List[float]] = defaultdict(list)

            for run in runs:
                if run.latency_ms is not None:
                    metric_values["latency_ms"].append(run.latency_ms)
                if run.ttft_ms is not None:
                    metric_values["ttft_ms"].append(run.ttft_ms)
                if run.tokens_per_second is not None:
                    metric_values["tokens_per_second"].append(run.tokens_per_second)
                if run.memory_peak_mb is not None:
                    metric_values["memory_peak_mb"].append(run.memory_peak_mb)
                if run.cpu_usage_percent is not None:
                    metric_values["cpu_usage_percent"].append(run.cpu_usage_percent)
                if run.gpu_usage_percent is not None:
                    metric_values["gpu_usage_percent"].append(run.gpu_usage_percent)

            # Aggregate
            for metric_name, values in metric_values.items():
                comparison.metrics[metric_name] = AggregatedMetrics.from_values(values)

            comparisons.append(comparison)

        # Calculate rankings and scores
        self._calculate_rankings(comparisons, ranking_metric)
        self._calculate_scores(comparisons)

        # Sort by primary ranking
        metric_key = self._ranking_metric_to_key(ranking_metric)
        comparisons.sort(key=lambda c: c.rankings.get(metric_key, 999))

        return comparisons

    def group_by_dimension(
        self,
        results: List[SuiteResult],
        dimension: AggregationDimension
    ) -> Dict[str, List[SuiteResult]]:
        """
        Group results by a specific dimension.

        Args:
            results: List of SuiteResult
            dimension: Dimension to group by

        Returns:
            Dict mapping dimension_value -> List[SuiteResult]
        """
        grouped: Dict[str, List[SuiteResult]] = defaultdict(list)

        for result in results:
            if dimension == AggregationDimension.SUITE:
                key = result.suite_type.value
            elif dimension == AggregationDimension.MODEL_TYPE:
                key = result.model_type.value
            elif dimension == AggregationDimension.MODEL:
                # Group by each model tested
                for model_id in result.models_tested:
                    grouped[model_id].append(result)
                continue
            elif dimension == AggregationDimension.ENDPOINT:
                # Group by each endpoint tested
                for endpoint in result.endpoints_tested:
                    grouped[endpoint].append(result)
                continue
            else:
                logger.warning(f"Unknown dimension: {dimension}")
                continue

            grouped[key].append(result)

        return dict(grouped)

    def calculate_success_rate(self, result: SuiteResult) -> float:
        """
        Calculate overall success rate for a benchmark result.

        Args:
            result: SuiteResult

        Returns:
            Success rate as percentage (0-100)
        """
        if result.total_runs == 0:
            return 0.0

        return (result.successful_runs / result.total_runs) * 100.0

    def _calculate_rankings(
        self,
        comparisons: List[ModelComparison],
        primary_metric: RankingMetric
    ) -> None:
        """
        Calculate rankings for all metrics.

        Args:
            comparisons: List of ModelComparison to rank
            primary_metric: Primary ranking metric
        """
        # Define metrics and their sorting order (lower is better / higher is better)
        metrics_config = {
            "latency_ms": ("lower", RankingMetric.LATENCY),
            "ttft_ms": ("lower", RankingMetric.LATENCY),
            "tokens_per_second": ("higher", RankingMetric.THROUGHPUT),
            "memory_peak_mb": ("lower", RankingMetric.MEMORY_PEAK),
            "cpu_usage_percent": ("lower", RankingMetric.CPU_USAGE),
            "gpu_usage_percent": ("lower", RankingMetric.GPU_USAGE)
        }

        for metric_key, (order, _) in metrics_config.items():
            # Get all models with this metric
            models_with_metric = [
                (comp, comp.metrics[metric_key].mean)
                for comp in comparisons
                if metric_key in comp.metrics
            ]

            if not models_with_metric:
                continue

            # Sort by metric value
            if order == "lower":
                models_with_metric.sort(key=lambda x: x[1])  # Ascending
            else:
                models_with_metric.sort(key=lambda x: x[1], reverse=True)  # Descending

            # Assign rankings
            for rank, (comp, _) in enumerate(models_with_metric, start=1):
                comp.rankings[metric_key] = rank

    def _calculate_scores(self, comparisons: List[ModelComparison]) -> None:
        """
        Calculate normalized scores (0-100) for all metrics.

        Args:
            comparisons: List of ModelComparison
        """
        # For each metric, normalize to 0-100 scale
        all_metrics = set()
        for comp in comparisons:
            all_metrics.update(comp.metrics.keys())

        for metric_key in all_metrics:
            # Get min/max values for normalization
            values = [
                comp.metrics[metric_key].mean
                for comp in comparisons
                if metric_key in comp.metrics
            ]

            if not values or len(values) < 2:
                continue

            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val

            if range_val == 0:
                # All values are the same
                for comp in comparisons:
                    if metric_key in comp.metrics:
                        comp.scores[metric_key] = 100.0
                continue

            # Determine if lower or higher is better
            lower_is_better = metric_key in [
                "latency_ms", "ttft_ms", "memory_peak_mb",
                "cpu_usage_percent", "gpu_usage_percent"
            ]

            # Calculate normalized scores
            for comp in comparisons:
                if metric_key not in comp.metrics:
                    continue

                value = comp.metrics[metric_key].mean

                if lower_is_better:
                    # Lower is better: max gets 100, min gets 0
                    score = 100.0 * (1.0 - (value - min_val) / range_val)
                else:
                    # Higher is better: max gets 100, min gets 0
                    score = 100.0 * (value - min_val) / range_val

                comp.scores[metric_key] = score

    def _ranking_metric_to_key(self, metric: RankingMetric) -> str:
        """Convert RankingMetric to metric key."""
        mapping = {
            RankingMetric.LATENCY: "latency_ms",
            RankingMetric.THROUGHPUT: "tokens_per_second",
            RankingMetric.MEMORY_PEAK: "memory_peak_mb",
            RankingMetric.CPU_USAGE: "cpu_usage_percent",
            RankingMetric.GPU_USAGE: "gpu_usage_percent"
        }
        return mapping.get(metric, "latency_ms")

    def generate_comparison_table(
        self,
        comparisons: List[ModelComparison],
        include_metrics: Optional[List[str]] = None
    ) -> dict:
        """
        Generate comparison table data.

        Args:
            comparisons: List of ModelComparison
            include_metrics: Metrics to include (None = all)

        Returns:
            Dict with table structure
        """
        if include_metrics is None:
            # Get all available metrics
            all_metrics = set()
            for comp in comparisons:
                all_metrics.update(comp.metrics.keys())
            include_metrics = sorted(all_metrics)

        rows = []
        for comp in comparisons:
            row = {"model_id": comp.model_id}

            for metric_name in include_metrics:
                if metric_name in comp.metrics:
                    agg = comp.metrics[metric_name]
                    row[f"{metric_name}_mean"] = round(agg.mean, 2)
                    row[f"{metric_name}_median"] = round(agg.median, 2)
                    row[f"{metric_name}_p95"] = round(agg.p95, 2)

                if metric_name in comp.rankings:
                    row[f"{metric_name}_rank"] = comp.rankings[metric_name]

                if metric_name in comp.scores:
                    row[f"{metric_name}_score"] = round(comp.scores[metric_name], 1)

            rows.append(row)

        return {
            "metrics": include_metrics,
            "models": [comp.model_id for comp in comparisons],
            "rows": rows
        }
