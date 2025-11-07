"""
Stress & Endurance benchmark suite.

Measures model stability under extended load:
- Performance degradation over time
- Memory leak detection
- Error rate tracking
- Stability validation
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import time

from src.benchmarking.suites.base_suite import BaseSuite
from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.metric_types import SuiteType, ResultStatus, MetricUnit, ModelType
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

            # Read suite-specific configuration from parameters
            if "duration_minutes" in config.parameters:
                self.duration_minutes = config.parameters["duration_minutes"]
                logger.info(f"Using custom duration: {self.duration_minutes} minutes")

            if "checkpoint_interval_seconds" in config.parameters:
                self.checkpoint_interval_seconds = config.parameters["checkpoint_interval_seconds"]
                logger.info(f"Using custom checkpoint interval: {self.checkpoint_interval_seconds // 60} minutes")

            logger.info(
                f"Starting Stress suite for {len(config.models)} model(s) "
                f"(duration: {self.duration_minutes} minutes)"
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
                            config=config
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
        config: BenchmarkConfig
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
        test_duration_seconds = self.duration_minutes * 60
        next_checkpoint = test_start + self.checkpoint_interval_seconds

        run_number = 0
        prompt_idx = 0
        total_errors = 0

        # Track baseline metrics (first 5 runs)
        baseline_latencies = []
        baseline_memory = []

        logger.info(f"  Running stress test for {self.duration_minutes} minutes...")
        logger.info(f"  Checkpoint every {self.checkpoint_interval_seconds // 60} minutes")

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
                        timeout=600
                    )

                # Post-run cleanup every 10 runs to avoid memory creep
                if run_number % 10 == 0:
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

                    # Track baseline (first 5 runs)
                    if run_number <= 5:
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
                next_checkpoint = current_time + self.checkpoint_interval_seconds

        # Compute degradation metrics after stress test completes
        self._compute_degradation_metrics(
            model_id=model_id,
            endpoint=endpoint,
            baseline_latencies=baseline_latencies,
            baseline_memory=baseline_memory,
            total_runs=run_number,
            total_errors=total_errors
        )

        logger.info(
            f"  Stress test complete: {run_number} runs in {self.duration_minutes} minutes, "
            f"{total_errors} errors"
        )

    def _compute_degradation_metrics(
        self,
        model_id: str,
        endpoint: str,
        baseline_latencies: List[float],
        baseline_memory: List[float],
        total_runs: int,
        total_errors: int
    ):
        """
        Compute degradation and stability metrics across ALL runs.

        Analyzes the entire test duration to detect:
        - Overall latency changes (baseline vs overall average)
        - Trend/slope of latency over time
        - Memory growth patterns
        - Error rate distribution

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

        # 4. Stability assessment (pass/fail) - based on OVERALL metrics
        # Thresholds: <20% overall latency degradation, <50% overall memory growth, <10% error rate
        stability_passed = (
            overall_delta_percent < 20 and
            (overall_mem_delta_percent < 50 if memory_values else True) and
            error_rate_percent < 10
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

        logger.info(
            f"  Degradation analysis across ALL {len(latencies)} runs:\n"
            f"    Latency: overall {overall_delta_percent:+.1f}%, final {final_delta_percent:+.1f}%, max {max_delta_percent:+.1f}%\n"
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

    def __repr__(self) -> str:
        """String representation."""
        return f"StressSuite(executor={self.executor}, duration={self.duration_minutes}min)"
