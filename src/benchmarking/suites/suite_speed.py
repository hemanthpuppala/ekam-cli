"""
Speed & Throughput benchmark suite.

Measures model inference speed and throughput:
- Latency per request (milliseconds)
- Throughput (tokens/second or inferences/second)
- Time to first token (TTFT)
"""

from typing import Dict, Any, List
from datetime import datetime
import time

from src.benchmarking.suites.base_suite import BaseSuite
from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.metric_types import SuiteType, ResultStatus, MetricUnit, ModelType
from src.benchmarking.handlers.endpoint_executor import EndpointExecutor
from src.benchmarking.datasets.defaults import get_prompts_for_suite
from src.benchmarking.utils import MemoryOptimizer, memory_tracked_operation, RetryWithCleanup
from loguru import logger


class SpeedSuite(BaseSuite):
    """
    Speed & Throughput benchmark suite.

    Focuses on measuring:
    - Request latency (time per inference)
    - Throughput (tokens or inferences per second)
    - Consistency of speed across runs
    - Impact of warmup on speed
    """

    @property
    def suite_type(self) -> SuiteType:
        """Return suite type."""
        return SuiteType.SPEED

    def __init__(self, endpoint_executor: EndpointExecutor = None, enable_progress: bool = True):
        """
        Initialize Speed suite.

        Args:
            endpoint_executor: Optional EndpointExecutor instance
            enable_progress: Whether to enable progress display (default: True)
        """
        super().__init__(enable_progress=enable_progress)
        self.executor = endpoint_executor or EndpointExecutor()

    def run(self, config: BenchmarkConfig) -> SuiteResult:
        """
        Execute speed benchmark.

        Args:
            config: BenchmarkConfig with execution parameters

        Returns:
            SuiteResult with speed metrics
        """
        start_time = datetime.utcnow()

        try:
            # Setup
            self.setup(config)
            logger.info(f"Starting Speed suite for {len(config.models)} model(s)")

            # Prepare test data
            test_prompts = self._get_test_data(config)
            if not test_prompts:
                raise ValueError("No test prompts available")

            logger.info(f"Using {len(test_prompts)} test prompts")

            # Run benchmarks for each model
            total_runs_attempted = 0
            total_runs_successful = 0

            for model_id in config.models:
                logger.info(f"Benchmarking model: {model_id}")
                self._emit_progress(f"Benchmarking model: {model_id}")

                try:
                    for endpoint in config.endpoints.get(model_id, []):
                        logger.info(f"  Endpoint: {endpoint}")

                        # Warmup runs - critical for discovering working inference methods
                        warmup_success = True
                        if config.num_warmup > 0:
                            logger.info(f"  Running {config.num_warmup} warmup runs...")
                            self._emit_progress(f"Running {config.num_warmup} warmup runs...")
                            warmup_failures = 0

                            for warmup_num in range(1, config.num_warmup + 1):
                                prompt = test_prompts[warmup_num % len(test_prompts)]
                                success, latency_ms = self._execute_speed_run(
                                    model_id=model_id,
                                    endpoint=endpoint,
                                    prompt=prompt,
                                    run_number=warmup_num,
                                    is_warmup=True,
                                    config=config
                                )

                                if success:
                                    self._emit_progress(f"Warmup run {warmup_num}/{config.num_warmup} completed in {latency_ms:.2f}ms")
                                    # At least one warmup succeeded - we found a working method
                                    logger.info(f"✓ Warmup {warmup_num} succeeded - inference method confirmed")
                                    break
                                else:
                                    warmup_failures += 1

                            # If all warmup runs failed, skip this model/endpoint
                            if warmup_failures == config.num_warmup:
                                warmup_success = False
                                logger.error(
                                    f"✗ All {config.num_warmup} warmup runs failed for {model_id}:{endpoint}. "
                                    f"Skipping model - no compatible inference method found."
                                )
                                self._record_error(
                                    model_id, endpoint, 0,
                                    "Model skipped: All warmup runs failed - no compatible inference method"
                                )
                                continue  # Skip to next endpoint

                        # Skip benchmark runs if warmup failed
                        if not warmup_success:
                            logger.warning(f"Skipping benchmark runs for {model_id}:{endpoint} due to warmup failure")
                            continue

                        # Counted runs
                        logger.info(f"  Running {config.num_runs} benchmark runs...")
                        for run_num in range(1, config.num_runs + 1):
                            total_runs_attempted += 1
                            prompt = test_prompts[run_num % len(test_prompts)]

                            success, latency_ms = self._execute_speed_run(
                                model_id=model_id,
                                endpoint=endpoint,
                                prompt=prompt,
                                run_number=run_num,
                                is_warmup=False,
                                config=config
                            )

                            if success:
                                total_runs_successful += 1
                                # Emit progress with latency for each completed run
                                self._emit_progress(f"Run {run_num}/{config.num_runs} completed in {latency_ms:.2f}ms")

                finally:
                    # Unload model after all runs complete to free memory
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

            # Determine status
            if total_runs_successful == total_runs_attempted:
                status = ResultStatus.SUCCESS
            elif total_runs_successful == 0:
                status = ResultStatus.FAILED
            else:
                status = ResultStatus.PARTIAL

            # Create result
            end_time = datetime.utcnow()
            result = self._create_result(start_time, end_time, status)

            logger.info(
                f"Speed suite complete: {total_runs_successful}/{total_runs_attempted} successful"
            )

            return result

        except Exception as e:
            logger.error(f"Speed suite failed: {str(e)}")
            self.on_run_error(e)

            end_time = datetime.utcnow()
            return self._create_result(start_time, end_time, ResultStatus.FAILED)

        finally:
            self.teardown()

    def _execute_speed_run(
        self,
        model_id: str,
        endpoint: str,
        prompt: str,
        run_number: int,
        is_warmup: bool,
        config: BenchmarkConfig
    ) -> tuple[bool, float]:
        """
        Execute a single speed benchmark run with memory optimization.

        Features:
        - O(1) memory check before run
        - Automatic cleanup between runs
        - Retry logic for OOM/timeout errors
        - Memory pressure monitoring

        Args:
            model_id: Model being tested
            endpoint: Endpoint being used
            prompt: Input prompt
            run_number: Run number
            is_warmup: Whether this is a warmup run
            config: Benchmark configuration

        Returns:
            Tuple of (success: bool, latency_ms: float)
        """
        try:
            # O(1) memory pressure check
            pressure = MemoryOptimizer.memory_pressure_level()
            available_mb = MemoryOptimizer.get_available_memory_mb()

            logger.debug(
                f"Run {run_number} starting: {available_mb:.0f}MB available, "
                f"pressure: {pressure}"
            )

            # Critical: skip run if memory is too low
            if pressure == "critical":
                logger.warning(
                    f"Skipping run {run_number}: Critical memory pressure "
                    f"({available_mb:.0f}MB < 200MB threshold)"
                )
                # Aggressive cleanup and return failure
                MemoryOptimizer.aggressive_cleanup()
                return (False, 0.0)

            # Smart cleanup: only if memory is getting tight
            if pressure in ["high", "critical"]:
                logger.info(f"Memory pressure {pressure}, performing cleanup before run")
                MemoryOptimizer.aggressive_cleanup()

            # Prepare input data and track for CSV output
            if config.model_type == ModelType.LLM:
                input_data = prompt
                input_image_path = "-"  # LLMs don't use images
            else:  # VLM
                # For VLM, need image path
                images = config.test_data.get("images", [])
                if not images:
                    logger.warning("No images provided for VLM benchmark")
                    return (False, 0.0)

                input_image_path = images[run_number % len(images)]
                input_data = {
                    "prompt": prompt,
                    "image_path": input_image_path
                }

            # Execute with memory tracking and retry logic
            operation_name = f"Run {run_number} ({'warmup' if is_warmup else 'benchmark'})"

            # Use memory-tracked context + retry mechanism
            retry_handler = RetryWithCleanup(
                max_retries=2,
                cleanup_between_retries=True,
                backoff_seconds=2.0
            )

            def execute_with_timing():
                """Inner function for retry wrapper."""
                with memory_tracked_operation(
                    operation_name=operation_name,
                    min_required_mb=250.0,  # Require at least 250MB for VLM inference
                    auto_cleanup=True
                ):
                    start_time = time.time()

                    result = self.executor.execute(
                        model_type=config.model_type,
                        model_id=model_id,
                        endpoint=endpoint,
                        input_data=input_data,
                        parameters=config.parameters,
                        timeout=600  # 10 minute timeout for large models
                    )

                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000

                    return result, latency_ms

            # Execute with retry
            result, latency_ms = retry_handler.execute(execute_with_timing)

            # Extract raw response (untruncated)
            raw_response = result.get("output", "")

            # Create metadata dict with full input/output details for CSV
            run_metadata = {
                "input_prompt": prompt,
                "input_image_path": input_image_path,
                "raw_response": raw_response
            }

            # Record latency metric with full I/O metadata
            self._record_metric(
                name="latency_ms",
                value=latency_ms,
                unit=MetricUnit.MILLISECONDS,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup,
                metadata=run_metadata
            )

            # Calculate and record throughput if token count available
            if "token_count" in result.get("metadata", {}):
                token_count = result["metadata"]["token_count"]
                duration_seconds = latency_ms / 1000.0

                if duration_seconds > 0:
                    tokens_per_sec = token_count / duration_seconds

                    self._record_metric(
                        name="tokens_per_sec",
                        value=tokens_per_sec,
                        unit=MetricUnit.TOKENS_PER_SECOND,
                        run_number=run_number,
                        model_id=model_id,
                        endpoint=endpoint,
                        is_warmup=is_warmup,
                        metadata=run_metadata
                    )

            # Record output length
            output_length = len(raw_response)
            self._record_metric(
                name="output_length",
                value=float(output_length),
                unit=MetricUnit.COUNT,
                run_number=run_number,
                model_id=model_id,
                endpoint=endpoint,
                is_warmup=is_warmup,
                metadata=run_metadata
            )

            return (True, latency_ms)

        except Exception as e:
            logger.error(f"Speed run {run_number} failed: {str(e)}")
            self._record_error(model_id, endpoint, run_number, str(e))
            return (False, 0.0)

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
        return get_prompts_for_suite("speed", model_type_str)

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
        return f"SpeedSuite(executor={self.executor})"
