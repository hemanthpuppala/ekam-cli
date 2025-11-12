"""
Speed & Throughput benchmark suite.

Measures model inference speed and throughput:
- Latency per request (milliseconds)
- Throughput (tokens/second or inferences/second)
- Time to first token (TTFT)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import time

from src.benchmarking.suites.base_suite import BaseSuite
from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.metric_types import SuiteType, ResultStatus, MetricUnit, ModelType
from src.benchmarking.models.suite_configs import SpeedConfig, speed_config_from_params
from src.benchmarking.handlers.endpoint_executor import EndpointExecutor
from src.benchmarking.datasets.defaults import get_prompts_for_suite
from src.benchmarking.utils import MemoryOptimizer, memory_tracked_operation, RetryWithCleanup
from src.benchmarking.utils.llama_delay import apply_llama_server_delay
from src.benchmarking.core.progress_display import ProgressPhase
from src.benchmarking.core.model_context import ModelContextManager, ModelState
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

    def __init__(self, endpoint_executor: EndpointExecutor = None, enable_progress: bool = True, model_context_manager: Optional[ModelContextManager] = None):
        """
        Initialize Speed suite.

        Args:
            endpoint_executor: Optional EndpointExecutor instance
            enable_progress: Whether to enable progress display (default: True)
            model_context_manager: Optional ModelContextManager instance for model lifecycle management
        """
        super().__init__(enable_progress=enable_progress)
        self.executor = endpoint_executor or EndpointExecutor()
        self.model_context_manager = model_context_manager

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

            # Load suite-specific configuration
            suite_config = speed_config_from_params(config.parameters)
            logger.info(f"Starting Speed suite for {len(config.models)} model(s)")
            logger.info(f"Configuration: {suite_config.num_runs} runs, {suite_config.num_warmup} warmup, {suite_config.timeout_seconds}s timeout")
            logger.info(f"Timing metrics: {', '.join(suite_config.timing_metrics_to_capture)}")

            # Prepare test data
            test_prompts = self._get_test_data(config)
            if not test_prompts:
                raise ValueError("No test prompts available")

            logger.info(f"Using {len(test_prompts)} test prompts")

            # Progress display start
            total_runs_per_model = suite_config.num_warmup + len(test_prompts)
            self._progress_start(total_runs=total_runs_per_model, num_models=len(config.models))

            # Run benchmarks for each model
            total_runs_attempted = 0
            total_runs_successful = 0

            for model_id in config.models:
                logger.info(f"Benchmarking model: {model_id}")
                self._emit_progress(f"Benchmarking model: {model_id}")

                if not self.model_context_manager:
                    logger.error("ModelContextManager not initialized. Cannot run benchmark.")
                    self._record_error(model_id, "N/A", 0, "ModelContextManager not initialized")
                    continue

                with self.model_context_manager.load_model(model_id) as model_context:
                    if model_context.state != ModelState.LOADED:
                        logger.error(f"Failed to load model {model_id}: {model_context.failed_reason}")
                        self._record_error(model_id, "N/A", 0, f"Failed to load model: {model_context.failed_reason}")
                        continue

                    # The rest of the loop for endpoints and runs will go here
                    for endpoint in config.endpoints.get(model_id, []):
                        logger.info(f"  Endpoint: {endpoint}")
                        # Update model with endpoint context for UI
                        if self.progress_display:
                            try:
                                self.progress_display.update_model(model_id, config.models.index(model_id) + 1, endpoint)
                            except Exception:
                                self._progress_update_model(model_id, config.models.index(model_id) + 1)
                        else:
                            self._progress_update_model(model_id, config.models.index(model_id) + 1)

                        # Warmup runs - critical for discovering working inference methods
                        # Always use the first prompt for warmup
                        warmup_success = True
                        if suite_config.num_warmup > 0:
                            logger.info(f"  Running {suite_config.num_warmup} warmup runs...")
                            self._emit_progress(f"Running {suite_config.num_warmup} warmup runs...")
                            warmup_failures = 0

                            warmup_prompt = test_prompts[0]  # Always use first prompt for warmup
                            for warmup_num in range(1, suite_config.num_warmup + 1):
                                self._progress_update_run(warmup_num, is_warmup=True)
                                success, latency_ms, timings = self._execute_speed_run(
                                    model_id=model_id,
                                    endpoint=endpoint,
                                    prompt=warmup_prompt,
                                    run_number=warmup_num,
                                    is_warmup=True,
                                    config=config,
                                    suite_config=suite_config
                                )

                                if success:
                                    self._emit_progress(f"Warmup run {warmup_num}/{suite_config.num_warmup} completed in {latency_ms:.2f}ms")
                                    # At least one warmup succeeded - we found a working method
                                    logger.info(f"✓ Warmup {warmup_num} succeeded - inference method confirmed")
                                metrics_update = {"latency_ms": latency_ms}
                                if "ttft_ms" in timings:
                                    metrics_update["ttft_ms"] = timings["ttft_ms"]
                                if "tokens_per_sec" in timings:
                                    metrics_update["tokens_per_sec"] = timings["tokens_per_sec"]
                                if "itl_ms" in timings:
                                    metrics_update["itl_ms"] = timings["itl_ms"]
                                if "decode_latency_ms" in timings:
                                    metrics_update["decode_latency_ms"] = timings["decode_latency_ms"]
                                self._progress_update_metrics(metrics_update)
                                if self.progress_display:
                                    try:
                                        self.progress_display.report_success()
                                    except Exception:
                                        pass
                                    break
                                else:
                                    warmup_failures += 1

                            # If all warmup runs failed, skip this model/endpoint
                            if warmup_failures == suite_config.num_warmup:
                                warmup_success = False
                                logger.error(
                                    f"✗ All {suite_config.num_warmup} warmup runs failed for {model_id}:{endpoint}. "
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

                        # Counted runs - 1 prompt = 1 run paradigm
                        # Each prompt is run exactly once (no cycling)
                        num_prompts = len(test_prompts)
                        logger.info(f"  Running {num_prompts} benchmark runs (1 prompt = 1 run)...")
                        for prompt_idx, prompt in enumerate(test_prompts, 1):
                            total_runs_attempted += 1
                            # Update run number accounting for warmup offset
                            self._progress_update_run(suite_config.num_warmup + prompt_idx, is_warmup=False)
                            success, latency_ms, timings = self._execute_speed_run(
                                model_id=model_id,
                                endpoint=endpoint,
                                prompt=prompt,
                                run_number=prompt_idx,
                                is_warmup=False,
                                config=config,
                                suite_config=suite_config
                            )

                            if success:
                                total_runs_successful += 1
                                # Emit progress with latency for each completed run
                                self._emit_progress(f"Run {prompt_idx}/{num_prompts} completed in {latency_ms:.2f}ms")
                                metrics_update = {"latency_ms": latency_ms}
                                if "ttft_ms" in timings:
                                    metrics_update["ttft_ms"] = timings["ttft_ms"]
                                if "tokens_per_sec" in timings:
                                    metrics_update["tokens_per_sec"] = timings["tokens_per_sec"]
                                if "itl_ms" in timings:
                                    metrics_update["itl_ms"] = timings["itl_ms"]
                                if "decode_latency_ms" in timings:
                                    metrics_update["decode_latency_ms"] = timings["decode_latency_ms"]
                                self._progress_update_metrics(metrics_update)
                                if self.progress_display:
                                    try:
                                        self.progress_display.report_success()
                                    except Exception:
                                        pass

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
        config: BenchmarkConfig,
        suite_config: SpeedConfig
    ) -> tuple[bool, float, dict]:
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
            Tuple of (success: bool, latency_ms: float, timings: dict)
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
                return (False, 0.0, {})

            # Smart cleanup: only if memory is getting tight
            if pressure in ["high", "critical"]:
                logger.info(f"Memory pressure {pressure}, performing cleanup before run")
                MemoryOptimizer.aggressive_cleanup()

            # Prepare input data and track for CSV output
            if config.model_type == ModelType.LLM:
                input_data = prompt
                input_image_path = "-"  # LLMs don't use images
                used_prompt = prompt
            else:  # VLM
                # For VLM, prefer explicit image-prompt pairs mapping if provided
                pairs = config.test_data.get("pairs")
                effective_prompt = prompt
                input_image_path = None

                def _normalize_pairs(p):
                    if p is None:
                        return []
                    if isinstance(p, dict):
                        return list(p.items())
                    out = []
                    for item in p:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            out.append((item[0], item[1]))
                        elif isinstance(item, dict):
                            img = item.get("image") or item.get("image_path") or item.get("img")
                            pr = item.get("prompt")
                            if img is not None and pr is not None:
                                out.append((img, pr))
                    return out

                norm_pairs = _normalize_pairs(pairs)
                if norm_pairs:
                    # For warmup runs, always use the first pair (index 0)
                    # For counted runs, use the appropriate pair based on run_number
                    if is_warmup:
                        idx = 0
                    else:
                        idx = max(0, (run_number - 1) % len(norm_pairs))
                    input_image_path, effective_prompt = norm_pairs[idx]
                else:
                    images = config.test_data.get("images", [])
                    if not images:
                        logger.warning("No images provided for VLM benchmark")
                        return (False, 0.0, {})
                    input_image_path = images[run_number % len(images)]

                used_prompt = effective_prompt
                input_data = {
                    "prompt": effective_prompt,
                    "image_path": input_image_path
                }

            # Execute with memory tracking and retry logic
            operation_name = f"Run {run_number} ({'warmup' if is_warmup else 'benchmark'})"

            # Use memory-tracked context + retry mechanism
            retry_handler = RetryWithCleanup(
                max_retries=suite_config.max_retries,
                cleanup_between_retries=True,
                backoff_seconds=suite_config.retry_backoff_seconds
            )

            def execute_with_timing():
                """Inner function for retry wrapper with advanced timing metrics."""
                with memory_tracked_operation(
                    operation_name=operation_name,
                    min_required_mb=suite_config.min_memory_mb,
                    auto_cleanup=True
                ):
                    # Advanced timing tracking
                    timing_data = {
                        'start_time': time.time(),
                        'first_token_time': None,
                        'token_times': [],
                        'end_time': None
                    }

                    # Streaming callback to capture TTFT and ITL
                    def token_callback(token: str, is_first: bool = False):
                        """Callback for streaming tokens."""
                        current_time = time.time()
                        if is_first and timing_data['first_token_time'] is None:
                            timing_data['first_token_time'] = current_time
                        timing_data['token_times'].append(current_time)

                    # Add streaming callback to parameters if not present
                    exec_params = config.parameters.copy() if config.parameters else {}
                    exec_params['_stream_callback'] = token_callback
                    exec_params['_enable_timing'] = True

                    # Log request before inference
                    logger.info(f"Run {run_number} | Request: prompt='{used_prompt[:50]}...' image={Path(input_image_path).name}")

                    result = self.executor.execute(
                        model_type=config.model_type,
                        model_id=model_id,
                        endpoint=endpoint,
                        input_data=input_data,
                        parameters=exec_params,
                        timeout=suite_config.timeout_seconds
                    )

                    timing_data['end_time'] = time.time()

                    # Calculate metrics
                    total_latency_ms = (timing_data['end_time'] - timing_data['start_time']) * 1000

                    # Calculate TTFT if we got first token timing
                    ttft_ms = None
                    if timing_data['first_token_time']:
                        ttft_ms = (timing_data['first_token_time'] - timing_data['start_time']) * 1000

                    # Calculate inter-token latency (ITL) and decode latency
                    itl_ms = None
                    decode_latency_ms = None
                    if len(timing_data['token_times']) > 1:
                        # ITL = average time between tokens
                        inter_token_gaps = [
                            (timing_data['token_times'][i] - timing_data['token_times'][i-1]) * 1000
                            for i in range(1, len(timing_data['token_times']))
                        ]
                        if inter_token_gaps:
                            itl_ms = sum(inter_token_gaps) / len(inter_token_gaps)

                        # Decode latency = time from first token to last token
                        if timing_data['first_token_time']:
                            decode_latency_ms = (timing_data['token_times'][-1] - timing_data['first_token_time']) * 1000

                    # Log response after inference
                    raw_after = result.get("output", "") if isinstance(result, dict) else ""
                    response_preview = raw_after[:100].replace('\n', ' ') if raw_after else "Empty"
                    logger.info(f"Run {run_number} | Response: {response_preview}..." if len(raw_after) > 100 else f"Run {run_number} | Response: {response_preview}")

                    return result, total_latency_ms, ttft_ms, itl_ms, decode_latency_ms

            # Execute with retry
            result, latency_ms, ttft_ms, itl_ms, decode_latency_ms = retry_handler.execute(execute_with_timing)

            # Extract raw response (untruncated)
            raw_response = result.get("output", "")

            # Extract reasoning from response and format
            from src.utils.response_formatter import extract_thinking_blocks
            thinking_blocks, clean_response = extract_thinking_blocks(raw_response)

            # Format response with reasoning if present
            if thinking_blocks:
                reasoning_str = "\n\n".join(thinking_blocks)
                formatted_response = f"Reasoning: {reasoning_str}\n\nResponse: {clean_response}"
            else:
                formatted_response = raw_response

            # Create metadata dict with full input/output details for CSV
            run_metadata = {
                "input_prompt": used_prompt,
                "input_image_path": input_image_path,
                "raw_response": formatted_response  # Store formatted version with reasoning
            }

            # Record total latency metric with full I/O metadata
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

            # Record Time to First Token (TTFT) - prefill phase (if configured)
            if ttft_ms is not None and suite_config.should_capture_metric("ttft_ms"):
                self._record_metric(
                    name="ttft_ms",
                    value=ttft_ms,
                    unit=MetricUnit.MILLISECONDS,
                    run_number=run_number,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=is_warmup,
                    metadata=run_metadata
                )

            # Also record prefill latency (TTFT is the prefill phase) (if configured)
            if ttft_ms is not None and suite_config.should_capture_metric("prefill_latency_ms"):
                self._record_metric(
                    name="prefill_latency_ms",
                    value=ttft_ms,
                    unit=MetricUnit.MILLISECONDS,
                    run_number=run_number,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=is_warmup,
                    metadata=run_metadata
                )

            # Record Inter-Token Latency (ITL) - decode phase smoothness (if configured)
            if itl_ms is not None and suite_config.should_capture_metric("itl_ms"):
                self._record_metric(
                    name="inter_token_latency_ms",
                    value=itl_ms,
                    unit=MetricUnit.MILLISECONDS,
                    run_number=run_number,
                    model_id=model_id,
                    endpoint=endpoint,
                    is_warmup=is_warmup,
                    metadata=run_metadata
                )

            # Record Decode latency (total decode phase time) (if configured)
            if decode_latency_ms is not None and suite_config.should_capture_metric("decode_latency_ms"):
                self._record_metric(
                    name="decode_latency_ms",
                    value=decode_latency_ms,
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

            # Build timings dict for UI updates (O(1))
            timings = {}
            if ttft_ms is not None:
                timings["ttft_ms"] = ttft_ms
            if itl_ms is not None:
                timings["itl_ms"] = itl_ms
            if decode_latency_ms is not None:
                timings["decode_latency_ms"] = decode_latency_ms
            if "token_count" in result.get("metadata", {}):
                # tokens_per_sec computed above if duration_seconds>0; recompute safely here
                token_count = result["metadata"]["token_count"]
                duration_seconds = latency_ms / 1000.0
                if duration_seconds > 0:
                    timings["tokens_per_sec"] = token_count / duration_seconds

            # Apply post-inference delay for llama-server to allow state cleanup
            # This delay is OUTSIDE latency measurement and does not affect benchmark timing
            provider_type = model_id.split(":")[0] if ":" in model_id else model_id
            apply_llama_server_delay(provider_type)

            return (True, latency_ms, timings)

        except Exception as e:
            logger.error(f"Speed run {run_number} failed: {str(e)}")
            self._record_error(model_id, endpoint, run_number, str(e))
            return (False, 0.0, {})

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
