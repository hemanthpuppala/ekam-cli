"""
Base suite template for all benchmark suites.

Provides abstract base class with common infrastructure for:
- Metric collection
- System monitoring
- Run execution
- Error handling
- Result aggregation
- Memory optimization (O(1) checks, aggressive cleanup)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import time

from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.suite_result import SuiteResult, PerformanceMetric
from src.benchmarking.models.metric_types import SuiteType, ResultStatus, MetricUnit
from src.benchmarking.core.metrics_collector import MetricsCollector
from src.benchmarking.core.system_monitor import BenchmarkSystemMonitor
from src.benchmarking.core.error_handler import ErrorHandler, ErrorCategory, BenchmarkError
from src.benchmarking.core.progress_display import ProgressDisplay, ProgressPhase
from src.benchmarking.utils import MemoryOptimizer, memory_tracked_operation, RetryWithCleanup
from loguru import logger


class BaseSuite(ABC):
    """
    Abstract base class for benchmark suites.

    All benchmark suites must extend this class and implement:
    - suite_type: Property returning the SuiteType enum
    - run(): Method executing the benchmark and returning SuiteResult

    Common functionality provided:
    - Metrics collection via MetricsCollector
    - System monitoring via BenchmarkSystemMonitor
    - Error handling and recovery
    - Result aggregation and formatting
    """

    def __init__(self, enable_progress: bool = True):
        """
        Initialize base suite with collectors.

        Args:
            enable_progress: Whether to enable progress display (default: True)
        """
        self.metrics_collector = MetricsCollector()
        self.system_monitor = BenchmarkSystemMonitor(self.metrics_collector)
        self.error_handler = ErrorHandler()
        self.progress_display: Optional[ProgressDisplay] = ProgressDisplay() if enable_progress else None
        self._config: Optional[BenchmarkConfig] = None
        self._result: Optional[SuiteResult] = None
        self._errors: list = []
        self._progress_callback: Optional[Any] = None  # External progress callback

    @property
    @abstractmethod
    def suite_type(self) -> SuiteType:
        """Return the suite type. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def run(self, config: BenchmarkConfig) -> SuiteResult:
        """
        Execute the benchmark suite.

        Must be implemented by subclasses.

        Args:
            config: BenchmarkConfig with execution parameters

        Returns:
            SuiteResult with metrics and aggregates
        """
        pass

    def setup(self, config: BenchmarkConfig) -> None:
        """
        Setup before benchmark execution with memory optimization.

        Called automatically before run(). Override to add custom setup.

        Args:
            config: BenchmarkConfig
        """
        self._config = config
        self.metrics_collector.clear()
        self._errors.clear()

        # Initial memory check and cleanup
        mem_stats = MemoryOptimizer.get_memory_stats()
        logger.info(
            f"Setting up {self.suite_type.value} suite for {len(config.models)} model(s). "
            f"Available memory: {mem_stats.get('available_mb', 0):.0f}MB"
        )

        # Aggressive cleanup before starting
        cleanup_result = MemoryOptimizer.aggressive_cleanup()
        logger.debug(
            f"Pre-suite cleanup: {cleanup_result['objects_collected']} objects collected, "
            f"{cleanup_result.get('freed_mb', 0):.1f}MB freed"
        )

    def set_progress_callback(self, callback):
        """
        Set external progress callback for emitting progress messages.

        Args:
            callback: Callable that accepts a string message
        """
        self._progress_callback = callback

    def _emit_progress(self, message: str):
        """
        Emit progress message to external callback if set.

        Args:
            message: Progress message
        """
        if self._progress_callback:
            try:
                self._progress_callback(message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def teardown(self) -> None:
        """
        Cleanup after benchmark execution with memory optimization.

        Called automatically after run(). Override to add custom cleanup.
        """
        # Log error handler statistics
        error_summary = self.get_error_handler_summary()
        if error_summary["total_errors"] > 0:
            logger.info(f"Error handler summary: {error_summary['total_errors']} total errors")
            logger.debug(f"Errors by category: {error_summary['errors_by_category']}")
            logger.debug(f"Circuit breaker states: {error_summary['circuit_breaker_states']}")

        # Final cleanup
        final_cleanup = MemoryOptimizer.aggressive_cleanup()
        final_mem = MemoryOptimizer.get_memory_stats()

        logger.info(
            f"{self.suite_type.value} suite teardown complete. "
            f"Final cleanup: {final_cleanup['objects_collected']} objects collected. "
            f"Available memory: {final_mem.get('available_mb', 0):.0f}MB"
        )

    def _execute_single_run(
        self,
        model_id: str,
        endpoint: str,
        run_number: int,
        is_warmup: bool,
        input_data: Any,
        parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a single inference run with monitoring and retry logic.

        Args:
            model_id: Model being tested
            endpoint: Endpoint being used
            run_number: Current run number
            is_warmup: Whether this is a warmup run
            input_data: Input data (prompt, image, etc.)
            parameters: Inference parameters

        Returns:
            Dictionary with output, metrics, or None if failed
        """
        # Define the inference operation to be retried
        def inference_operation():
            start_time = time.time()

            # Track system metrics during inference
            with self.system_monitor.track_inference(
                model_id=model_id,
                endpoint=endpoint,
                run_number=run_number,
                is_warmup=is_warmup
            ):
                # Subclasses should implement actual model execution
                output = self._execute_inference(model_id, endpoint, input_data, parameters)

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Record latency metric
            self.metrics_collector.record_metric(
                name="latency_ms",
                value=latency_ms,
                unit=MetricUnit.MILLISECONDS,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup
            )

            return {
                "output": output,
                "latency_ms": latency_ms,
                "success": True
            }

        # Execute with retry logic and circuit breaker
        try:
            result, error = self.error_handler.execute_with_retry(
                operation=inference_operation,
                operation_name=f"{model_id}/{endpoint}",
                max_retries=2  # Allow 2 retries for transient failures
            )

            if result:
                return result
            else:
                # Circuit breaker open or max retries exceeded
                if error:
                    self._record_error(model_id, endpoint, run_number, error.message)
                    logger.warning(
                        f"Run {run_number} failed after {error.retry_count} retries: "
                        f"{error.category.value}"
                    )
                return None

        except Exception as e:
            # Unexpected error not caught by error handler
            logger.error(f"Unexpected error in run {run_number} for {model_id}/{endpoint}: {str(e)}")
            self._record_error(model_id, endpoint, run_number, str(e))
            return None

    def _execute_inference(
        self,
        model_id: str,
        endpoint: str,
        input_data: Any,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute the actual inference.

        This is a hook method that subclasses should override to implement
        actual model inference. By default, returns a mock output.

        Args:
            model_id: Model ID
            endpoint: Endpoint name
            input_data: Input data
            parameters: Inference parameters

        Returns:
            Model output
        """
        # Default implementation - subclasses should override
        raise NotImplementedError("Subclasses must implement _execute_inference()")

    def _record_metric(
        self,
        name: str,
        value: float,
        unit: MetricUnit,
        run_number: int,
        model_id: str,
        endpoint: str,
        is_warmup: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a custom metric with optional metadata.

        Args:
            name: Metric name
            value: Metric value
            unit: Metric unit
            run_number: Run number
            model_id: Model ID
            endpoint: Endpoint
            is_warmup: Whether this is a warmup run
            metadata: Optional metadata dict (e.g., input_prompt, raw_response)
        """
        self.metrics_collector.record_metric(
            name=name,
            value=value,
            unit=unit,
            run_number=run_number,
            model_id=model_id,
            endpoint=endpoint,
            is_warmup=is_warmup,
            metadata=metadata
        )

    def _record_error(
        self,
        model_id: str,
        endpoint: str,
        run_number: int,
        error_message: str
    ) -> None:
        """
        Record an error that occurred during execution.

        Args:
            model_id: Model where error occurred
            endpoint: Endpoint where error occurred
            run_number: Run number when error occurred
            error_message: Error message
        """
        error_entry = {
            "model_id": model_id,
            "endpoint": endpoint,
            "run_number": run_number,
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._errors.append(error_entry)
        logger.warning(f"Error recorded: {error_entry}")

    def _execute_with_memory_optimization(
        self,
        operation_name: str,
        operation_callable,
        *args,
        min_required_mb: float = 250.0,
        max_retries: int = 2,
        skip_on_critical_memory: bool = True,
        **kwargs
    ) -> Optional[Any]:
        """
        Execute operation with memory optimization and retry logic.

        This is a helper method that ALL suites can use for memory-safe execution.
        Provides:
        - O(1) memory pressure check
        - Automatic cleanup on memory pressure
        - Retry logic with backoff
        - Memory tracking and logging

        Args:
            operation_name: Name for logging (e.g., "Run 3 warmup")
            operation_callable: Function to execute
            *args: Positional arguments for operation
            min_required_mb: Minimum required free memory (default: 250MB)
            max_retries: Maximum retry attempts (default: 2)
            skip_on_critical_memory: Skip if memory critical (default: True)
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result, or None if skipped/failed

        Example:
            ```python
            result = self._execute_with_memory_optimization(
                operation_name="Inference Run 3",
                operation_callable=self._run_inference,
                model_id="qwen3-vl",
                prompt="What is this?",
                min_required_mb=300.0
            )
            ```
        """
        # O(1) memory pressure check
        pressure = MemoryOptimizer.memory_pressure_level()
        available_mb = MemoryOptimizer.get_available_memory_mb()

        logger.debug(
            f"{operation_name}: {available_mb:.0f}MB available, "
            f"pressure: {pressure}"
        )

        # Critical: skip if memory too low
        if skip_on_critical_memory and pressure == "critical":
            logger.warning(
                f"Skipping {operation_name}: Critical memory pressure "
                f"({available_mb:.0f}MB < 200MB)"
            )
            MemoryOptimizer.aggressive_cleanup()
            return None

        # Smart cleanup if memory is tight
        if pressure in ["high", "critical"]:
            logger.info(f"Memory pressure {pressure}, cleaning up before {operation_name}")
            MemoryOptimizer.aggressive_cleanup()

        # Execute with retry and memory tracking
        retry_handler = RetryWithCleanup(
            max_retries=max_retries,
            cleanup_between_retries=True,
            backoff_seconds=2.0
        )

        def wrapped_operation():
            """Wrapper for memory tracking."""
            with memory_tracked_operation(
                operation_name=operation_name,
                min_required_mb=min_required_mb,
                auto_cleanup=True
            ):
                return operation_callable(*args, **kwargs)

        try:
            result = retry_handler.execute(wrapped_operation)
            return result

        except Exception as e:
            logger.error(f"{operation_name} failed after retries: {str(e)}")
            return None

    def _compute_aggregates(self) -> Dict[str, Dict]:
        """
        Compute aggregate statistics for all collected metrics.

        Returns:
            Dictionary with warmup, counted, and global aggregates
        """
        aggregates = {
            "warmup": self.metrics_collector.get_aggregates(for_warmup=True),
            "counted": self.metrics_collector.get_aggregates(for_warmup=False),
            "global": self.metrics_collector.get_global_aggregates()
        }

        # Per-model aggregates
        per_model = {}
        if self._config:
            for model_id in self._config.models:
                per_model[model_id] = self.metrics_collector.get_aggregates(
                    for_warmup=False,
                    model_id=model_id
                )

        aggregates["per_model"] = per_model

        # Per-endpoint aggregates
        per_endpoint = {}
        if self._config:
            for model_id in self._config.models:
                for endpoint in self._config.endpoints.get(model_id, []):
                    key = f"{model_id}/{endpoint}"
                    per_endpoint[key] = self.metrics_collector.get_aggregates(
                        for_warmup=False,
                        model_id=model_id,
                        endpoint=endpoint
                    )

        aggregates["per_endpoint"] = per_endpoint

        return aggregates

    def _create_result(
        self,
        start_time: datetime,
        end_time: datetime,
        status: ResultStatus
    ) -> SuiteResult:
        """
        Create SuiteResult from collected metrics.

        Args:
            start_time: When suite started
            end_time: When suite ended
            status: Final execution status

        Returns:
            SuiteResult with all metrics and aggregates
        """
        if not self._config:
            raise ValueError("Config not set - call setup() first")

        # Compute aggregates
        aggregates = self._compute_aggregates()

        # Count successful and failed runs
        all_metrics = self.metrics_collector.metrics_list()

        # Get unique run attempts from successful metrics
        successful_run_keys = set((m.model_id, m.endpoint, m.run_number) for m in all_metrics if not m.is_warmup)

        # Get unique run attempts from errors
        failed_run_keys = set((e['model_id'], e['endpoint'], e['run_number']) for e in self._errors)

        # Total runs = successful + failed
        all_run_keys = successful_run_keys | failed_run_keys
        total_runs = len(all_run_keys)
        failed_runs = len(failed_run_keys)
        successful_runs = len(successful_run_keys)

        # Create result
        result = SuiteResult(
            suite_type=self.suite_type,
            benchmark_id=self._config.benchmark_id,
            model_type=self._config.model_type,
            start_time=start_time,
            end_time=end_time,
            status=status,
            models_tested=self._config.models,
            endpoints_tested=self._config.endpoints,
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            metrics=all_metrics,
            aggregates_warmup=aggregates["warmup"],
            aggregates_counted=aggregates["counted"],
            aggregates_global=aggregates["global"],
            aggregates_per_model=aggregates["per_model"],
            aggregates_per_endpoint=aggregates["per_endpoint"],
            config_snapshot=self._config.to_dict(),
            duration_seconds=(end_time - start_time).total_seconds()
        )

        # Add error messages if any
        if self._errors:
            result.error_message = f"{len(self._errors)} errors occurred during execution"

        return result

    def _determine_status(self, total_runs: int, failed_runs: int) -> ResultStatus:
        """
        Determine final execution status based on success/failure counts.

        Args:
            total_runs: Total number of runs attempted
            failed_runs: Number of failed runs

        Returns:
            ResultStatus enum value
        """
        if failed_runs == 0:
            return ResultStatus.SUCCESS
        elif failed_runs == total_runs:
            return ResultStatus.FAILED
        else:
            return ResultStatus.PARTIAL

    def on_run_error(self, error: Exception) -> None:
        """
        Handle error during run execution.

        Override to add custom error handling.

        Args:
            error: Exception that occurred
        """
        logger.error(f"Error in {self.suite_type.value} suite: {str(error)}")

    def get_error_handler_summary(self) -> Dict[str, Any]:
        """
        Get error handler statistics summary.

        Returns:
            Dictionary with error handler statistics
        """
        summary = self.error_handler.get_error_summary()

        return {
            "total_errors": summary.get("total_errors", 0),
            "errors_by_category": summary.get("by_category", {}),
            "errors_by_severity": summary.get("by_severity", {}),
            "circuit_breaker_states": summary.get("circuit_breaker_states", {})
        }

    # Progress display helper methods

    def _progress_start(self, total_runs: int, num_models: int) -> None:
        """
        Start progress display.

        Args:
            total_runs: Total number of runs per model
            num_models: Number of models being tested
        """
        if self.progress_display:
            self.progress_display.start(
                total_models=num_models,
                total_runs_per_model=total_runs,
                suite_name=self.suite_type.display_name()
            )

    def _progress_update_phase(self, phase: ProgressPhase) -> None:
        """
        Update current progress phase.

        Args:
            phase: New phase to enter
        """
        if self.progress_display:
            self.progress_display.update_phase(phase)

    def _progress_update_model(self, model_id: str, model_index: int) -> None:
        """
        Update current model being tested.

        Args:
            model_id: Model identifier
            model_index: Model index (0-based)
        """
        if self.progress_display:
            self.progress_display.update_model(model_id, model_index)

    def _progress_update_run(self, run_number: int, is_warmup: bool = False) -> None:
        """
        Update current run progress.

        Args:
            run_number: Current run number
            is_warmup: Whether this is a warmup run
        """
        if self.progress_display:
            self.progress_display.update_run(run_number, is_warmup)

    def _progress_update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update current metrics display.

        Args:
            metrics: Dictionary of metrics to display
        """
        if self.progress_display:
            self.progress_display.update_metrics(metrics)

    def _progress_complete(self, status: ResultStatus) -> None:
        """
        Mark progress as complete.

        Args:
            status: Final result status
        """
        if self.progress_display:
            if status == ResultStatus.SUCCESS:
                self.progress_display.mark_complete()
            else:
                self.progress_display.mark_failed(f"Benchmark {status.value}")

    def _progress_stop(self) -> None:
        """Stop progress display."""
        if self.progress_display:
            self.progress_display.stop()

    def __repr__(self) -> str:
        """String representation."""
        progress_enabled = self.progress_display is not None
        return f"{self.__class__.__name__}(suite_type={self.suite_type.value}, progress={progress_enabled})"
