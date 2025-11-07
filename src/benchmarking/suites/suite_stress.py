"""
Stress & Endurance benchmark suite.

Measures model stability under extended load:
- Performance degradation over time
- Memory leak detection
- Error rate tracking
- Stability validation
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import time

from src.benchmarking.suites.base_suite import BaseSuite
from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.metric_types import SuiteType, ResultStatus, MetricUnit, ModelType
from src.benchmarking.models.suite_configs import StressConfig, stress_config_from_params
from src.benchmarking.handlers.endpoint_executor import EndpointExecutor
from src.benchmarking.datasets.defaults import get_prompts_for_suite
from src.benchmarking.utils import MemoryOptimizer
from loguru import logger


class StressSuite(BaseSuite):
    """
    Stress & Endurance benchmark suite.

    Runs continuous inference for an extended period to detect:
    - Latency degradation (performance decline over time)
    - Memory leaks (continuously increasing memory usage)
    - Error rate increases
    - Overall stability assessment
    """

    @property
    def suite_type(self) -> SuiteType:
        """Return suite type."""
        return SuiteType.STRESS

    def __init__(self, endpoint_executor: EndpointExecutor = None, duration_minutes: int = 30, enable_progress: bool = True):
        """
        Initialize Stress suite.

        Args:
            endpoint_executor: Optional EndpointExecutor instance
            duration_minutes: Duration to run stress test (default 30 minutes)
            enable_progress: Whether to enable progress display (default: True)
        """
        super().__init__(enable_progress=enable_progress)
        self.executor = endpoint_executor or EndpointExecutor()
        self.duration_minutes = duration_minutes
        self.checkpoint_interval_seconds = 300  # Save checkpoint every 5 minutes

    def run(self, config: BenchmarkConfig) -> SuiteResult:
        """
        Execute stress and endurance benchmark.

        Args:
            config: BenchmarkConfig with execution parameters

        Returns:
            SuiteResult with stability and degradation metrics
        """
        start_time = datetime.utcnow()

        try:
            # Setup
            self.setup(config)

            # Load suite-specific configuration
            suite_config = stress_config_from_params(config.parameters)
            logger.info(f"Starting Stress suite for {len(config.models)} model(s)")
            logger.info(
                f"Configuration: {suite_config.duration_minutes} min duration, "
                f"{suite_config.baseline_measurement_runs} baseline runs, "
                f"cleanup every {suite_config.cleanup_frequency_runs} runs"
            )
            logger.info(
                f"Stability thresholds: {suite_config.max_latency_degradation_percent}% latency, "
                f"{suite_config.max_memory_growth_percent}% memory, "
                f"{suite_config.max_error_rate_percent}% error rate"
            )

            # Prepare test data
            test_prompts = self._get_test_data(config)
            if not test_prompts:
                raise ValueError("No test prompts available")

            logger.info(f"Using {len(test_prompts)} test prompts for stress testing")

            # Run benchmarks for each model
            for model_id in config.models:
                logger.info(f"Stress testing model: {model_id}")

                try:
                    for endpoint in config.endpoints.get(model_id, []):
                        logger.info(f"  Endpoint: {endpoint}")

                        # Run stress test
                        self._run_stress_test(
                            model_id=model_id,
                            endpoint=endpoint,
                            prompts=test_prompts,
                            config=config,
                            suite_config=suite_config
                        )

                finally:
                    # Unload model after stress test
                    if hasattr(self.executor, 'llm_handler') and \
                       hasattr(self.executor.llm_handler, 'provider_bridge') and \
                       self.executor.llm_handler.provider_bridge:
                        logger.info(f"  Unloading model: {model_id}")
                        self.executor.llm_handler.provider_bridge.unload_model(model_id)
                    elif hasattr(self.executor, 'vlm_handler') and \
                         hasattr(self.executor.vlm_handler, 'provider_bridge') and \
                         self.executor.vlm_handler.provider_bridge:
                        logger.info(f"  Unloading model: {model_id}")
                        self.executor.vlm_handler.provider_bridge.unload_model(model_id)

            # Create result
            end_time = datetime.utcnow()
            result = self._create_result(start_time, end_time, ResultStatus.SUCCESS)

            logger.info("Stress suite complete")

            return result

        except Exception as e:
            logger.error(f"Stress suite failed: {str(e)}")
            self.on_run_error(e)

            end_time = datetime.utcnow()
            return self._create_result(start_time, end_time, ResultStatus.FAILED)

        finally:
            self.teardown()

    def _run_stress_test(
        self,
        model_id: str,
        endpoint: str,
        prompts: List[str],
        config: BenchmarkConfig,
        suite_config: StressConfig
    ):
        """
        Run continuous stress test for configured duration.

        Args:
            model_id: Model being tested
            endpoint: Endpoint being used
            prompts: List of test prompts (rotated)
            config: Benchmark configuration
        """
        test_start = time.time()
        test_duration_seconds = suite_config.duration_minutes * 60
        next_checkpoint = test_start + suite_config.checkpoint_interval_seconds

        run_number = 0
        prompt_idx = 0
        total_errors = 0

        # Track baseline metrics (configurable number of runs)
        baseline_latencies = []
        baseline_memory = []

        logger.info(f"  Running stress test for {suite_config.duration_minutes} minutes...")
        logger.info(f"  Checkpoint every {suite_config.get_checkpoint_interval_minutes():.1f} minutes")

        while (time.time() - test_start) < test_duration_seconds:
            run_number += 1
            current_time = time.time()
            elapsed_minutes = (current_time - test_start) / 60

            # Rotate through prompts
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # Execute run with system monitoring and memory optimization
            try:
                # Memory optimization for stress test
                pressure = MemoryOptimizer.memory_pressure_level()
                if pressure == "critical":
                    logger.warning(f"  Stress run {run_number}: Critical memory, cleaning up")
                    MemoryOptimizer.aggressive_cleanup()
                    # Don't skip in stress test - we want to see how it handles pressure
                elif pressure in ["high"]:
                    MemoryOptimizer.aggressive_cleanup()

                run_start = time.time()

                # Prepare input data and track for CSV output
                if config.model_type == ModelType.LLM:
                    input_data = prompt
                    input_image_path = "-"  # LLMs don't use images
                else:  # VLM
                    images = config.test_data.get("images", [])
                    if not images:
                        logger.warning("No images provided for VLM stress test")
                        continue

                    input_image_path = images[run_number % len(images)]
                    input_data = {
                        "prompt": prompt,
                        "image_path": input_image_path
                    }

                # Execute with system monitoring
                with self.system_monitor.track_inference(
                    model_id=model_id,
                    endpoint=endpoint,
                    run_number=run_number,
                    is_warmup=False
                ):
                    result = self.executor.execute(
                        model_type=config.model_type,
                        model_id=model_id,
                        endpoint=endpoint,
                        input_data=input_data,
                        parameters=config.parameters,
                        timeout=suite_config.timeout_seconds
                    )

                # Post-run cleanup at configurable frequency to avoid memory creep
                if run_number % suite_config.cleanup_frequency_runs == 0:
                    MemoryOptimizer.aggressive_cleanup()

                run_end = time.time()
                latency_ms = (run_end - run_start) * 1000

                # Extract raw response (untruncated)
                raw_response = result.get("output", "")

                # Create metadata dict with full input/output details for CSV
                run_metadata = {
                    "input_prompt": prompt,
                    "input_image_path": input_image_path,
                    "raw_response": raw_response
                }

                # Record latency with full I/O metadata
                latency_metadata = run_metadata.copy()
                latency_metadata["elapsed_minutes"] = elapsed_minutes

                self._record_metric(
                    name="latency_ms",
                    value=latency_ms,
                    unit=MetricUnit.MS,
                    run_number=run_number,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata=latency_metadata
                )

                # Get system metrics
                latest_metrics = self.system_monitor.get_latest_metrics()
                if latest_metrics:
                    memory_mb = latest_metrics.get("memory_mb_peak", 0)

                    memory_metadata = run_metadata.copy()
                    memory_metadata["elapsed_minutes"] = elapsed_minutes

                    self._record_metric(
                        name="memory_mb",
                        value=memory_mb,
                        unit=MetricUnit.MB,
                        run_number=run_number,
                        model_id=model_id,
                        endpoint=endpoint,
                        is_warmup=False,
                        metadata=memory_metadata
                    )

                    # Track baseline (configurable number of runs)
                    if run_number <= suite_config.baseline_measurement_runs:
                        baseline_latencies.append(latency_ms)
                        baseline_memory.append(memory_mb)

                # Record output length
                output_length = len(raw_response)
                self._record_metric(
                    name="output_length",
                    value=float(output_length),
                    unit=MetricUnit.COUNT,
                    run_number=run_number,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata=run_metadata
                )

            except Exception as e:
                total_errors += 1
                logger.error(f"  Stress run {run_number} failed: {str(e)}")
                self._record_error(model_id, endpoint, run_number, str(e))

                # Record error
                self._record_metric(
                    name="error_occurred",
                    value=1.0,
                    unit=MetricUnit.COUNT,
                    run_number=run_number,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata={"error_message": str(e)}
                )

            # Checkpoint logging
            if current_time >= next_checkpoint:
                error_rate = (total_errors / run_number) * 100 if run_number > 0 else 0
                logger.info(
                    f"  Checkpoint: {elapsed_minutes:.1f} min elapsed, "
                    f"{run_number} runs, {total_errors} errors ({error_rate:.1f}%)"
                )
                next_checkpoint = current_time + suite_config.checkpoint_interval_seconds

        # Compute degradation metrics after stress test completes
        self._compute_degradation_metrics(
            model_id=model_id,
            endpoint=endpoint,
            baseline_latencies=baseline_latencies,
            baseline_memory=baseline_memory,
            total_runs=run_number,
            total_errors=total_errors,
            suite_config=suite_config
        )

        logger.info(
            f"  Stress test complete: {run_number} runs in {suite_config.duration_minutes} minutes, "
            f"{total_errors} errors"
        )

    def _compute_degradation_metrics(
        self,
        model_id: str,
        endpoint: str,
        baseline_latencies: List[float],
        baseline_memory: List[float],
        total_runs: int,
        total_errors: int,
        suite_config: StressConfig
    ):
        """
        Compute degradation and stability metrics across ALL runs.

        Analyzes the entire test duration to detect:
        - Overall latency changes (baseline vs overall average)
        - Trend/slope of latency over time (degradation curves)
        - Memory growth patterns and leak detection
        - Error rate distribution and patterns
        - CPU/GPU degradation tracking

        Args:
            model_id: Model ID
            endpoint: Endpoint name
            baseline_latencies: Latencies from first 5 runs
            baseline_memory: Memory usage from first 5 runs
            total_runs: Total number of runs executed
            total_errors: Total number of errors encountered
        """
        # Get all latencies and memory metrics
        all_metrics = self.metrics_collector.metrics_list()
        latencies = [m.value for m in all_metrics
                    if m.name == "latency_ms" and m.model_id == model_id and not m.is_warmup]
        memory_values = [m.value for m in all_metrics
                        if m.name == "memory_mb" and m.model_id == model_id and not m.is_warmup]
        cpu_values = [m.value for m in all_metrics
                     if m.name == "cpu_percent_avg" and m.model_id == model_id and not m.is_warmup]
        gpu_values = [m.value for m in all_metrics
                     if m.name == "gpu_percent_avg" and m.model_id == model_id and not m.is_warmup]

        if not latencies or not baseline_latencies:
            logger.warning("Insufficient data to compute degradation metrics")
            return

        # 1. Latency degradation analysis (baseline vs ALL runs)
        baseline_avg = sum(baseline_latencies) / len(baseline_latencies)

        # Overall average across ALL runs
        overall_avg = sum(latencies) / len(latencies) if latencies else baseline_avg

        # Last 5 runs average for comparison
        final_latencies = latencies[-5:] if len(latencies) >= 5 else latencies
        final_avg = sum(final_latencies) / len(final_latencies) if final_latencies else baseline_avg

        # Worst-case: Maximum latency observed
        max_latency = max(latencies) if latencies else baseline_avg

        # Calculate degradation metrics
        overall_delta_ms = overall_avg - baseline_avg
        overall_delta_percent = (overall_delta_ms / baseline_avg * 100) if baseline_avg > 0 else 0

        final_delta_ms = final_avg - baseline_avg
        final_delta_percent = (final_delta_ms / baseline_avg * 100) if baseline_avg > 0 else 0

        max_delta_percent = ((max_latency - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0

        # Record comprehensive latency degradation
        self._record_metric(
            name="latency_degradation_ms",
            value=overall_delta_ms,  # Overall average vs baseline
            unit=MetricUnit.MS,
            run_number=total_runs,
            model_id=model_id,
            endpoint=endpoint,
            is_warmup=False,
            metadata={
                "baseline_avg_ms": baseline_avg,
                "overall_avg_ms": overall_avg,
                "final_avg_ms": final_avg,
                "max_latency_ms": max_latency,
                "degradation_overall_percent": overall_delta_percent,
                "degradation_final_percent": final_delta_percent,
                "degradation_max_percent": max_delta_percent,
                "total_runs_analyzed": len(latencies)
            }
        )

        # 2. Memory trend analysis (leak detection across ALL runs)
        if memory_values and baseline_memory:
            baseline_mem_avg = sum(baseline_memory) / len(baseline_memory)

            # Overall average memory across ALL runs
            overall_mem_avg = sum(memory_values) / len(memory_values) if memory_values else baseline_mem_avg

            # Last 5 measurements
            final_memory = memory_values[-5:] if len(memory_values) >= 5 else memory_values
            final_mem_avg = sum(final_memory) / len(final_memory) if final_memory else baseline_mem_avg

            # Peak memory usage
            max_memory = max(memory_values) if memory_values else baseline_mem_avg

            # Calculate memory growth metrics
            overall_mem_delta = overall_mem_avg - baseline_mem_avg
            overall_mem_delta_percent = (overall_mem_delta / baseline_mem_avg * 100) if baseline_mem_avg > 0 else 0

            final_mem_delta = final_mem_avg - baseline_mem_avg
            final_mem_delta_percent = (final_mem_delta / baseline_mem_avg * 100) if baseline_mem_avg > 0 else 0

            max_mem_delta_percent = ((max_memory - baseline_mem_avg) / baseline_mem_avg * 100) if baseline_mem_avg > 0 else 0

            self._record_metric(
                name="memory_growth_mb",
                value=overall_mem_delta,  # Overall average vs baseline
                unit=MetricUnit.MB,
                run_number=total_runs,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=False,
                metadata={
                    "baseline_avg_mb": baseline_mem_avg,
                    "overall_avg_mb": overall_mem_avg,
                    "final_avg_mb": final_mem_avg,
                    "max_memory_mb": max_memory,
                    "growth_overall_percent": overall_mem_delta_percent,
                    "growth_final_percent": final_mem_delta_percent,
                    "growth_max_percent": max_mem_delta_percent,
                    "total_measurements": len(memory_values)
                }
            )

        # 3. Error rate
        error_rate_percent = (total_errors / total_runs * 100) if total_runs > 0 else 0

        self._record_metric(
            name="error_rate_percent",
            value=error_rate_percent,
            unit=MetricUnit.PERCENT,
            run_number=total_runs,
            model_id=model_id,
            endpoint=endpoint,
            is_warmup=False,
            metadata={"total_errors": total_errors, "total_runs": total_runs}
        )

        # 4. Stability assessment (pass/fail) - based on configurable thresholds
        thresholds = suite_config.get_stability_thresholds()
        stability_passed = (
            overall_delta_percent < thresholds["max_latency_degradation_percent"] and
            (overall_mem_delta_percent < thresholds["max_memory_growth_percent"] if memory_values else True) and
            error_rate_percent < thresholds["max_error_rate_percent"]
        )

        self._record_metric(
            name="stability_passed",
            value=1.0 if stability_passed else 0.0,
            unit=MetricUnit.BOOL,
            run_number=total_runs,
            model_id=model_id,
            endpoint=endpoint,
            is_warmup=False,
            metadata={
                "latency_degradation_overall_percent": overall_delta_percent,
                "latency_degradation_final_percent": final_delta_percent,
                "memory_growth_overall_percent": overall_mem_delta_percent if memory_values else 0,
                "memory_growth_final_percent": final_mem_delta_percent if memory_values else 0,
                "error_rate_percent": error_rate_percent
            }
        )

        # 5. Performance degradation curves (trend analysis via linear regression)
        latency_slope = self._compute_trend_slope(latencies)
        if latency_slope is not None:
            self._record_metric(
                name="latency_degradation_slope_ms_per_run",
                value=latency_slope,
                unit=MetricUnit.MS,
                run_number=total_runs,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=False,
                metadata={
                    "interpretation": "Positive = degrading, Negative = improving, ~0 = stable",
                    "total_runs_analyzed": len(latencies)
                }
            )

        # Memory growth curve
        if memory_values:
            memory_slope = self._compute_trend_slope(memory_values)
            if memory_slope is not None:
                self._record_metric(
                    name="memory_growth_slope_mb_per_run",
                    value=memory_slope,
                    unit=MetricUnit.MB,
                    run_number=total_runs,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata={
                        "interpretation": "Positive = memory leak, Negative = improving, ~0 = stable",
                        "total_measurements": len(memory_values)
                    }
                )

        # CPU degradation curve
        if cpu_values and len(cpu_values) >= 5:
            baseline_cpu = sum(cpu_values[:5]) / len(cpu_values[:5])
            final_cpu = sum(cpu_values[-5:]) / len(cpu_values[-5:])
            cpu_delta_percent = ((final_cpu - baseline_cpu) / baseline_cpu * 100) if baseline_cpu > 0 else 0

            cpu_slope = self._compute_trend_slope(cpu_values)
            if cpu_slope is not None:
                self._record_metric(
                    name="cpu_degradation_slope_percent_per_run",
                    value=cpu_slope,
                    unit=MetricUnit.PERCENT,
                    run_number=total_runs,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata={
                        "baseline_cpu_percent": baseline_cpu,
                        "final_cpu_percent": final_cpu,
                        "cpu_delta_percent": cpu_delta_percent
                    }
                )

        # GPU degradation curve
        if gpu_values and len(gpu_values) >= 5:
            baseline_gpu = sum(gpu_values[:5]) / len(gpu_values[:5])
            final_gpu = sum(gpu_values[-5:]) / len(gpu_values[-5:])
            gpu_delta_percent = ((final_gpu - baseline_gpu) / baseline_gpu * 100) if baseline_gpu > 0 else 0

            gpu_slope = self._compute_trend_slope(gpu_values)
            if gpu_slope is not None:
                self._record_metric(
                    name="gpu_degradation_slope_percent_per_run",
                    value=gpu_slope,
                    unit=MetricUnit.PERCENT,
                    run_number=total_runs,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=False,
                    metadata={
                        "baseline_gpu_percent": baseline_gpu,
                        "final_gpu_percent": final_gpu,
                        "gpu_delta_percent": gpu_delta_percent
                    }
                )

        # 6. Error pattern analysis
        if total_errors > 0:
            error_patterns = self._analyze_error_patterns(all_metrics, model_id, total_runs)

            self._record_metric(
                name="error_pattern_score",
                value=error_patterns["clustering_score"],
                unit=MetricUnit.SCORE,
                run_number=total_runs,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=False,
                metadata=error_patterns
            )

        logger.info(
            f"  Degradation analysis across ALL {len(latencies)} runs:\n"
            f"    Latency: overall {overall_delta_percent:+.1f}%, final {final_delta_percent:+.1f}%, max {max_delta_percent:+.1f}%"
            f" (slope: {latency_slope:+.3f} ms/run)" if latency_slope is not None else "" + "\n"
            f"    Memory: overall {overall_mem_delta_percent:+.1f}%, final {final_mem_delta_percent:+.1f}%, max {max_mem_delta_percent:+.1f}%"
            if memory_values else f"    Memory: N/A\n"
            f"    Error rate: {error_rate_percent:.1f}%\n"
            f"    Stability: {'PASS ✓' if stability_passed else 'FAIL ✗'}"
        )

    def _get_test_data(self, config: BenchmarkConfig) -> List[str]:
        """
        Get test prompts from config or defaults.

        Args:
            config: Benchmark configuration

        Returns:
            List of test prompts
        """
        # Try to get from config
        if "prompts" in config.test_data and config.test_data["prompts"]:
            return config.test_data["prompts"]

        # Fall back to defaults
        model_type_str = "llm" if config.model_type == ModelType.LLM else "vlm"
        return get_prompts_for_suite("stress", model_type_str)

    def _execute_inference(
        self,
        model_id: str,
        endpoint: str,
        input_data: Any,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute actual inference (required by BaseSuite).

        This is delegated to the endpoint executor.

        Args:
            model_id: Model ID
            endpoint: Endpoint name
            input_data: Input data
            parameters: Inference parameters

        Returns:
            Model output
        """
        result = self.executor.execute(
            model_type=self._config.model_type if self._config else ModelType.LLM,
            model_id=model_id,
            endpoint=endpoint,
            input_data=input_data,
            parameters=parameters,
            timeout=600
        )

        return result.get("output", "")

    def _compute_trend_slope(self, values: List[float]) -> Optional[float]:
        """
        Compute linear regression slope for degradation curve analysis.

        Uses least squares method to fit a line through the data points.
        Positive slope = degrading, negative = improving, ~0 = stable.

        Args:
            values: Time series of metric values

        Returns:
            Slope value (change per run), or None if insufficient data
        """
        if not values or len(values) < 3:
            return None

        try:
            n = len(values)
            # X values are run numbers (0, 1, 2, ...)
            x_values = list(range(n))

            # Calculate means
            x_mean = sum(x_values) / n
            y_mean = sum(values) / n

            # Calculate slope using least squares
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)

            if denominator == 0:
                return 0.0

            slope = numerator / denominator
            return slope

        except Exception as e:
            logger.warning(f"Failed to compute trend slope: {e}")
            return None

    def _analyze_error_patterns(self, all_metrics: List[Any], model_id: str, total_runs: int) -> Dict[str, Any]:
        """
        Analyze error patterns to detect clustering and timing.

        Analyzes:
        - When errors occur (beginning/middle/end of test)
        - Error clustering (are errors consecutive or distributed?)
        - Error type distribution (if available)

        Args:
            all_metrics: All metrics from the test
            model_id: Model ID
            total_runs: Total number of runs

        Returns:
            Dictionary with error pattern analysis
        """
        # Get error metrics
        error_metrics = [m for m in all_metrics
                        if m.name == "error_occurred" and m.model_id == model_id and not m.is_warmup]

        if not error_metrics:
            return {
                "total_errors": 0,
                "clustering_score": 0.0,
                "errors_in_first_third": 0,
                "errors_in_middle_third": 0,
                "errors_in_last_third": 0,
                "max_consecutive_errors": 0,
                "error_types": {}
            }

        # Extract error run numbers
        error_run_numbers = sorted([m.run_number for m in error_metrics])
        total_errors = len(error_run_numbers)

        # Analyze temporal distribution (beginning/middle/end)
        third = total_runs // 3
        errors_first = sum(1 for r in error_run_numbers if r <= third)
        errors_middle = sum(1 for r in error_run_numbers if third < r <= 2 * third)
        errors_last = sum(1 for r in error_run_numbers if r > 2 * third)

        # Analyze clustering (consecutive errors)
        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(error_run_numbers)):
            if error_run_numbers[i] == error_run_numbers[i-1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        # Clustering score: ratio of max consecutive to total errors
        # High score (close to 1) = errors are clustered
        # Low score (close to 0) = errors are distributed
        clustering_score = max_consecutive / total_errors if total_errors > 0 else 0.0

        # Extract error types from metadata if available
        error_types = {}
        for m in error_metrics:
            if hasattr(m, 'metadata') and m.metadata and 'error_message' in m.metadata:
                error_msg = m.metadata['error_message']
                # Categorize error types by first few words
                error_type = ' '.join(error_msg.split()[:3]) if error_msg else "Unknown"
                error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": total_errors,
            "clustering_score": clustering_score,
            "errors_in_first_third": errors_first,
            "errors_in_middle_third": errors_middle,
            "errors_in_last_third": errors_last,
            "max_consecutive_errors": max_consecutive,
            "error_types": error_types,
            "interpretation": f"Clustering: {'high' if clustering_score > 0.5 else 'distributed'}, "
                            f"Timing: {'early' if errors_first > errors_last else 'late' if errors_last > errors_first else 'uniform'}"
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"StressSuite(executor={self.executor}, duration={self.duration_minutes}min)"
