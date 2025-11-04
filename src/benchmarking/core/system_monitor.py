"""
System monitoring wrapper for benchmarking.

Thin wrapper extending InferenceTracker from src/system_specs/
with benchmark-specific aggregations and metric recording.
"""

from typing import Dict, Optional, Any
from contextlib import contextmanager

from src.system_specs.monitor import InferenceTracker, SystemMonitor
from src.benchmarking.core.metrics_collector import MetricsCollector
from src.benchmarking.models.metric_types import MetricUnit


class BenchmarkSystemMonitor:
    """
    System monitoring wrapper for benchmarking.

    Extends InferenceTracker with benchmark-specific functionality:
    - Automatic metric recording to MetricsCollector
    - Simplified interface for benchmark suites
    - Aggregation helpers for resource metrics
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize benchmark system monitor.

        Args:
            metrics_collector: MetricsCollector instance to record metrics
        """
        self.metrics_collector = metrics_collector
        self._system_monitor = SystemMonitor()  # Create SystemMonitor instance
        self._tracker: Optional[InferenceTracker] = None

    @contextmanager
    def track_inference(
        self,
        model_id: str,
        endpoint: str,
        run_number: int,
        is_warmup: bool = False
    ):
        """
        Context manager for tracking system metrics during inference.

        Usage:
            with monitor.track_inference("llama2:7b", "text/chat", run_num=1):
                # Run inference
                output = model.generate(prompt)

        Args:
            model_id: Model being benchmarked
            endpoint: Endpoint being used
            run_number: Current run number
            is_warmup: Whether this is a warmup run

        Yields:
            InferenceTracker context manager
        """
        # Create InferenceTracker instance using SystemMonitor
        self._tracker = self._system_monitor.start_inference_tracking(model_name=model_id)

        try:
            # Start tracking
            with self._tracker as tracker:
                yield tracker

            # After inference completes, record metrics
            metrics = self._tracker.get_metrics()
            self._record_system_metrics(
                metrics,
                model_id,
                endpoint,
                run_number,
                is_warmup
            )

        finally:
            self._tracker = None

    def _record_system_metrics(
        self,
        metrics: Any,  # InferenceMetrics dataclass
        model_id: str,
        endpoint: str,
        run_number: int,
        is_warmup: bool
    ) -> None:
        """
        Record system metrics from InferenceTracker to MetricsCollector.

        Args:
            metrics: InferenceMetrics object from InferenceTracker
            model_id: Model ID
            endpoint: Endpoint name
            run_number: Run number
            is_warmup: Whether this is a warmup run
        """
        # CPU metrics (peak)
        if hasattr(metrics, 'peak_cpu_percent') and metrics.peak_cpu_percent > 0:
            self.metrics_collector.record_metric(
                name="cpu_percent_peak",
                value=metrics.peak_cpu_percent,
                unit=MetricUnit.PERCENT,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup
            )

        # CPU metrics (avg)
        if hasattr(metrics, 'avg_cpu_percent') and metrics.avg_cpu_percent > 0:
            self.metrics_collector.record_metric(
                name="cpu_percent_avg",
                value=metrics.avg_cpu_percent,
                unit=MetricUnit.PERCENT,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup
            )

        # GPU utilization (peak)
        if hasattr(metrics, 'peak_gpu_utilization') and metrics.peak_gpu_utilization > 0:
            self.metrics_collector.record_metric(
                name="gpu_percent_peak",
                value=metrics.peak_gpu_utilization,
                unit=MetricUnit.PERCENT,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup
            )

        # GPU utilization (avg)
        if hasattr(metrics, 'avg_gpu_utilization') and metrics.avg_gpu_utilization > 0:
            self.metrics_collector.record_metric(
                name="gpu_percent_avg",
                value=metrics.avg_gpu_utilization,
                unit=MetricUnit.PERCENT,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup
            )

        # Memory metrics (peak)
        if hasattr(metrics, 'peak_memory_mb') and metrics.peak_memory_mb > 0:
            self.metrics_collector.record_metric(
                name="memory_mb_peak",
                value=metrics.peak_memory_mb,
                unit=MetricUnit.MEGABYTES,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup
            )

        # Memory metrics (avg)
        if hasattr(metrics, 'avg_memory_mb') and metrics.avg_memory_mb > 0:
            self.metrics_collector.record_metric(
                name="memory_mb_avg",
                value=metrics.avg_memory_mb,
                unit=MetricUnit.MEGABYTES,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup
            )

        # GPU Memory/VRAM metrics (peak)
        if hasattr(metrics, 'peak_gpu_memory_mb') and metrics.peak_gpu_memory_mb > 0:
            self.metrics_collector.record_metric(
                name="vram_mb_peak",
                value=metrics.peak_gpu_memory_mb,
                unit=MetricUnit.MEGABYTES,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup
            )

        # GPU Memory/VRAM metrics (avg)
        if hasattr(metrics, 'avg_gpu_memory_mb') and metrics.avg_gpu_memory_mb > 0:
            self.metrics_collector.record_metric(
                name="vram_mb_avg",
                value=metrics.avg_gpu_memory_mb,
                unit=MetricUnit.MEGABYTES,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup
            )

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get latest metrics from the tracker.

        Returns:
            Latest metrics dictionary or None if no tracking active
        """
        if self._tracker:
            metrics_obj = self._tracker.get_metrics()
            # Convert InferenceMetrics dataclass to dict for backward compatibility
            return {
                "cpu_percent_peak": metrics_obj.peak_cpu_percent,
                "cpu_percent_avg": metrics_obj.avg_cpu_percent,
                "memory_mb_peak": metrics_obj.peak_memory_mb,
                "memory_mb_avg": metrics_obj.avg_memory_mb,
                "gpu_percent_peak": metrics_obj.peak_gpu_utilization,
                "gpu_percent_avg": metrics_obj.avg_gpu_utilization,
                "vram_mb_peak": metrics_obj.peak_gpu_memory_mb,
                "vram_mb_avg": metrics_obj.avg_gpu_memory_mb,
            }
        return None

    def __repr__(self) -> str:
        """String representation."""
        return f"BenchmarkSystemMonitor(active={self._tracker is not None})"
