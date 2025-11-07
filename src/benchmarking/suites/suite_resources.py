"""
Resource Efficiency benchmark suite.

Measures resource consumption during model inference:
- CPU usage (peak and average)
- GPU usage (peak and average)
- Memory consumption (peak and average)
- Temperature
- VRAM usage
"""

from typing import Dict, Any, List
from datetime import datetime
import time

from src.benchmarking.suites.base_suite import BaseSuite
from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.metric_types import SuiteType, ResultStatus, MetricUnit, ModelType
from src.benchmarking.core.progress_display import ProgressPhase
from src.benchmarking.handlers.endpoint_executor import EndpointExecutor
from src.benchmarking.datasets.defaults import get_prompts_for_suite
from src.benchmarking.utils import MemoryOptimizer
from loguru import logger


class ResourcesSuite(BaseSuite):
    """
    Resource Efficiency benchmark suite.

    Focuses on measuring resource consumption:
    - CPU utilization during inference
    - GPU utilization (if available)
    - Memory usage (RAM and VRAM)
    - Temperature monitoring
    - Resource efficiency per token/inference
    """

    @property
    def suite_type(self) -> SuiteType:
        """Return suite type."""
        return SuiteType.RESOURCES

    def __init__(self, endpoint_executor: EndpointExecutor = None, enable_progress: bool = True):
        """
        Initialize Resources suite.

        Args:
            endpoint_executor: Optional EndpointExecutor instance
            enable_progress: Whether to enable progress display (default: True)
        """
        super().__init__(enable_progress=enable_progress)
        self.executor = endpoint_executor or EndpointExecutor()

    def run(self, config: BenchmarkConfig) -> SuiteResult:
        """
        Execute resource efficiency benchmark.

        Args:
            config: BenchmarkConfig with execution parameters

        Returns:
            SuiteResult with resource metrics
        """
        start_time = datetime.utcnow()

        try:
            # Setup
            self.setup(config)
            logger.info(f"Starting Resources suite for {len(config.models)} model(s)")

            # Start progress tracking
            self._progress_start(config.num_runs, len(config.models))
            self._progress_update_phase(ProgressPhase.INITIALIZING)

            # Diagnostic logging
            logger.info(f"Configuration:")
            logger.info(f"  Models: {config.models}")
            logger.info(f"  Endpoints: {config.endpoints}")
            logger.info(f"  Model type: {config.model_type}")
            logger.info(f"  Num runs: {config.num_runs}")
            logger.info(f"  Num warmup: {config.num_warmup}")
            logger.info(f"  Parameters: {config.parameters}")
            logger.info(f"  Test data keys: {list(config.test_data.keys())}")

            # Prepare test data
            test_prompts = self._get_test_data(config)
            if not test_prompts:
                raise ValueError("No test prompts available")

            logger.info(f"Using {len(test_prompts)} test prompts")
            logger.info(f"First prompt (truncated): {test_prompts[0][:100]}...")

            # Run benchmarks for each model
            total_runs_attempted = 0
            total_runs_successful = 0

            for model_index, model_id in enumerate(config.models):
                logger.info(f"Benchmarking model: {model_id}")
                self._emit_progress(f"Benchmarking model: {model_id}")
                self._progress_update_model(model_id, model_index)
                self._progress_update_phase(ProgressPhase.LOADING_MODEL)

                try:
                    for endpoint in config.endpoints.get(model_id, []):
                        logger.info(f"  Endpoint: {endpoint}")

                        # Warmup runs
                        if config.num_warmup > 0:
                            logger.info(f"  Running {config.num_warmup} warmup runs...")
                            self._emit_progress(f"Running {config.num_warmup} warmup runs...")
                            self._progress_update_phase(ProgressPhase.WARMUP)
                            for warmup_num in range(1, config.num_warmup + 1):
                                self._progress_update_run(warmup_num, is_warmup=True)
                                prompt = test_prompts[warmup_num % len(test_prompts)]
                                self._execute_resource_run(
                                    model_id=model_id,
                                    endpoint=endpoint,
                                    prompt=prompt,
                                    run_number=warmup_num,
                                    is_warmup=True,
                                    config=config
                                )

                        # Counted runs
                        logger.info(f"  Running {config.num_runs} benchmark runs...")
                        self._progress_update_phase(ProgressPhase.RUNNING)
                        for run_num in range(1, config.num_runs + 1):
                            total_runs_attempted += 1
                            self._progress_update_run(run_num, is_warmup=False)
                            prompt = test_prompts[run_num % len(test_prompts)]

                            success = self._execute_resource_run(
                                model_id=model_id,
                                endpoint=endpoint,
                                prompt=prompt,
                                run_number=run_num,
                                is_warmup=False,
                                config=config
                            )

                            if success:
                                total_runs_successful += 1

                                # Update progress with latest metrics
                                latest_metrics = self.system_monitor.get_latest_metrics()
                                if latest_metrics:
                                    self._progress_update_metrics(latest_metrics)

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

            # Aggregate results
            self._progress_update_phase(ProgressPhase.AGGREGATING)
            end_time = datetime.utcnow()
            result = self._create_result(start_time, end_time, status)

            # Mark complete
            self._progress_complete(status)
            logger.info(
                f"Resources suite complete: {total_runs_successful}/{total_runs_attempted} successful"
            )

            return result

        except Exception as e:
            logger.error(f"Resources suite failed: {str(e)}")
            self.on_run_error(e)

            # Mark failed
            self._progress_update_phase(ProgressPhase.FAILED)
            self._progress_complete(ResultStatus.FAILED)

            end_time = datetime.utcnow()
            return self._create_result(start_time, end_time, ResultStatus.FAILED)

        finally:
            self._progress_stop()
            self.teardown()

    def _execute_resource_run(
        self,
        model_id: str,
        endpoint: str,
        prompt: str,
        run_number: int,
        is_warmup: bool,
        config: BenchmarkConfig
    ) -> bool:
        """
        Execute a single resource benchmark run with monitoring and memory optimization.

        Features:
        - O(1) memory pressure check before run
        - Automatic cleanup between runs
        - System resource monitoring
        - Efficiency metrics calculation

        Args:
            model_id: Model being tested
            endpoint: Endpoint being used
            prompt: Input prompt
            run_number: Run number
            is_warmup: Whether this is a warmup run
            config: Benchmark configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            # O(1) memory pressure check
            pressure = MemoryOptimizer.memory_pressure_level()
            available_mb = MemoryOptimizer.get_available_memory_mb()

            logger.debug(
                f"Resource run {run_number}: {available_mb:.0f}MB available, pressure: {pressure}"
            )

            # Critical: skip run if memory too low
            if pressure == "critical":
                logger.warning(
                    f"Skipping resource run {run_number}: Critical memory pressure ({available_mb:.0f}MB)"
                )
                MemoryOptimizer.aggressive_cleanup()
                return False

            # Smart cleanup if memory is getting tight
            if pressure in ["high", "critical"]:
                logger.info(f"Memory pressure {pressure}, cleaning up before resource run")
                MemoryOptimizer.aggressive_cleanup()

            # Prepare input data and track for CSV output
            if config.model_type == ModelType.LLM:
                input_data = prompt
                input_image_path = "-"  # LLMs don't use images
            else:  # VLM
                images = config.test_data.get("images", [])
                if not images:
                    logger.warning("No images provided for VLM benchmark")
                    return False

                input_image_path = images[run_number % len(images)]
                input_data = {
                    "prompt": prompt,
                    "image_path": input_image_path
                }

            # Execute with system monitoring
            # The system monitor wrapper automatically records resource metrics
            with self.system_monitor.track_inference(
                model_id=model_id,
                endpoint=endpoint,
                run_number=run_number,
                is_warmup=is_warmup
            ):
                result = self.executor.execute(
                    model_type=config.model_type,
                    model_id=model_id,
                    endpoint=endpoint,
                    input_data=input_data,
                    parameters=config.parameters,
                    timeout=600
                )

            # Post-run cleanup
            MemoryOptimizer.aggressive_cleanup()

            # Extract raw response (untruncated)
            raw_response = result.get("output", "")

            # Create metadata dict with full input/output details for CSV
            run_metadata = {
                "input_prompt": prompt,
                "input_image_path": input_image_path,
                "raw_response": raw_response
            }

            # System metrics are automatically recorded by the monitor
            # Additional metrics can be recorded here if needed

            # Record output size for efficiency analysis
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

            # Calculate efficiency metrics if token count available
            if "token_count" in result.get("metadata", {}):
                token_count = result["metadata"]["token_count"]

                # Get latest system metrics
                latest_metrics = self.system_monitor.get_latest_metrics()
                if latest_metrics:
                    # CPU efficiency (tokens per CPU percentage point)
                    if "cpu_percent_avg" in latest_metrics:
                        cpu_avg = latest_metrics["cpu_percent_avg"]
                        if cpu_avg > 0:
                            cpu_efficiency = token_count / cpu_avg
                            self._record_metric(
                                name="tokens_per_cpu_percent",
                                value=cpu_efficiency,
                                unit=MetricUnit.RATIO,
                                run_number=run_number,
                                model_id=model_id,
                                endpoint=endpoint,
                                is_warmup=is_warmup,
                                metadata=run_metadata
                            )

                    # Memory efficiency (tokens per MB)
                    if "memory_mb_avg" in latest_metrics:
                        memory_avg = latest_metrics["memory_mb_avg"]
                        if memory_avg > 0:
                            memory_efficiency = token_count / memory_avg
                            self._record_metric(
                                name="tokens_per_mb",
                                value=memory_efficiency,
                                unit=MetricUnit.RATIO,
                                run_number=run_number,
                                model_id=model_id,
                                endpoint=endpoint,
                                is_warmup=is_warmup,
                                metadata=run_metadata
                            )

            return True

        except Exception as e:
            logger.error(f"Resource run {run_number} failed for model {model_id}: {str(e)}")
            logger.error(f"Endpoint: {endpoint}, Prompt length: {len(prompt)}")
            logger.error(f"Full error details:", exc_info=True)
            self._record_error(model_id, endpoint, run_number, str(e))
            return False

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
        return get_prompts_for_suite("resources", model_type_str)

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
            timeout=180
        )

        return result.get("output", "")

    def __repr__(self) -> str:
        """String representation."""
        return f"ResourcesSuite(executor={self.executor})"
