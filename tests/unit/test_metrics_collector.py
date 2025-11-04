"""
Unit tests for MetricsCollector.

Tests metric recording, filtering, and aggregation.
"""

import pytest
import numpy as np

from src.benchmarking.core.metrics_collector import MetricsCollector
from src.benchmarking.models.metric_types import MetricUnit


class TestMetricsCollector:
    """Test suite for MetricsCollector."""

    def test_initialization(self):
        """Test collector initializes empty."""
        collector = MetricsCollector()
        assert len(collector) == 0
        assert len(collector.metrics_list()) == 0

    def test_record_metric(self):
        """Test recording a single metric."""
        collector = MetricsCollector()

        collector.record_metric(
            name="latency_ms",
            value=100.5,
            unit=MetricUnit.MILLISECONDS,
            run_number=1,
            model_id="test_model",
            endpoint="test_endpoint"
        )

        assert len(collector) == 1
        metrics = collector.metrics_list()
        assert metrics[0].name == "latency_ms"
        assert metrics[0].value == 100.5
        assert metrics[0].unit == MetricUnit.MILLISECONDS

    def test_record_multiple_metrics(self):
        """Test recording multiple metrics."""
        collector = MetricsCollector()

        for i in range(5):
            collector.record_metric(
                name="latency_ms",
                value=100.0 + i,
                unit=MetricUnit.MILLISECONDS,
                run_number=i + 1,
                model_id="test_model",
                endpoint="test_endpoint"
            )

        assert len(collector) == 5

    def test_warmup_filtering(self):
        """Test filtering warmup vs counted runs."""
        collector = MetricsCollector()

        # Record warmup runs
        for i in range(2):
            collector.record_metric(
                name="latency_ms",
                value=150.0,
                unit=MetricUnit.MILLISECONDS,
                run_number=i + 1,
                model_id="test_model",
                endpoint="test_endpoint",
                is_warmup=True
            )

        # Record counted runs
        for i in range(5):
            collector.record_metric(
                name="latency_ms",
                value=100.0,
                unit=MetricUnit.MILLISECONDS,
                run_number=i + 1,
                model_id="test_model",
                endpoint="test_endpoint",
                is_warmup=False
            )

        warmup_metrics = collector.get_metrics(warmup_only=True)
        counted_metrics = collector.get_metrics(counted_only=True)

        assert len(warmup_metrics) == 2
        assert len(counted_metrics) == 5

    def test_model_filtering(self):
        """Test filtering by model ID."""
        collector = MetricsCollector()

        # Record for model A
        for i in range(3):
            collector.record_metric(
                name="latency_ms",
                value=100.0,
                unit=MetricUnit.MILLISECONDS,
                run_number=i + 1,
                model_id="model_a",
                endpoint="test_endpoint"
            )

        # Record for model B
        for i in range(2):
            collector.record_metric(
                name="latency_ms",
                value=200.0,
                unit=MetricUnit.MILLISECONDS,
                run_number=i + 1,
                model_id="model_b",
                endpoint="test_endpoint"
            )

        model_a_metrics = collector.get_metrics(model_id="model_a")
        model_b_metrics = collector.get_metrics(model_id="model_b")

        assert len(model_a_metrics) == 3
        assert len(model_b_metrics) == 2

    def test_compute_aggregates(self):
        """Test aggregate statistics computation."""
        collector = MetricsCollector()

        # Record known values
        values = [100.0, 110.0, 120.0, 130.0, 140.0]
        for i, value in enumerate(values):
            collector.record_metric(
                name="latency_ms",
                value=value,
                unit=MetricUnit.MILLISECONDS,
                run_number=i + 1,
                model_id="test_model",
                endpoint="test_endpoint"
            )

        aggregates = collector.get_aggregates(for_warmup=False)

        assert "latency_ms" in aggregates
        agg = aggregates["latency_ms"]

        # Check statistics
        assert agg.count == 5
        assert agg.mean == pytest.approx(120.0)
        assert agg.median == pytest.approx(120.0)
        assert agg.min == pytest.approx(100.0)
        assert agg.max == pytest.approx(140.0)
        assert agg.std_dev is not None

    def test_percentiles(self):
        """Test percentile calculations."""
        collector = MetricsCollector()

        # Record values from 1 to 100
        for i in range(1, 101):
            collector.record_metric(
                name="latency_ms",
                value=float(i),
                unit=MetricUnit.MILLISECONDS,
                run_number=i,
                model_id="test_model",
                endpoint="test_endpoint"
            )

        aggregates = collector.get_aggregates(for_warmup=False)
        agg = aggregates["latency_ms"]

        # Check percentiles are reasonable
        assert agg.p25 is not None
        assert agg.p50 is not None
        assert agg.p75 is not None
        assert agg.p95 is not None
        assert agg.p99 is not None

        # P50 should be close to median
        assert agg.p50 == pytest.approx(agg.median)

    def test_global_aggregates(self):
        """Test global aggregation across models."""
        collector1 = MetricsCollector()
        collector2 = MetricsCollector()

        # Record for collector 1
        for i in range(3):
            collector1.record_metric(
                name="latency_ms",
                value=100.0,
                unit=MetricUnit.MILLISECONDS,
                run_number=i + 1,
                model_id="model_a",
                endpoint="test_endpoint"
            )

        # Record for collector 2
        for i in range(3):
            collector2.record_metric(
                name="latency_ms",
                value=200.0,
                unit=MetricUnit.MILLISECONDS,
                run_number=i + 1,
                model_id="model_b",
                endpoint="test_endpoint"
            )

        global_agg = collector1.get_global_aggregates([collector2])

        assert "latency_ms" in global_agg
        agg = global_agg["latency_ms"]
        assert agg.count == 6
        assert agg.mean == pytest.approx(150.0)  # Average of 100 and 200

    def test_clear(self):
        """Test clearing metrics."""
        collector = MetricsCollector()

        collector.record_metric(
            name="latency_ms",
            value=100.0,
            unit=MetricUnit.MILLISECONDS,
            run_number=1,
            model_id="test_model",
            endpoint="test_endpoint"
        )

        assert len(collector) == 1

        collector.clear()

        assert len(collector) == 0
        assert len(collector.metrics_list()) == 0

    def test_to_dict(self):
        """Test dictionary export."""
        collector = MetricsCollector()

        collector.record_metric(
            name="latency_ms",
            value=100.0,
            unit=MetricUnit.MILLISECONDS,
            run_number=1,
            model_id="test_model",
            endpoint="test_endpoint"
        )

        data = collector.to_dict()

        assert "metrics" in data
        assert "total_count" in data
        assert data["total_count"] == 1
        assert len(data["metrics"]) == 1
