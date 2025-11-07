"""
Quality & Consistency benchmark suite.

Measures output quality and consistency across multiple runs:
- Consistency (exact match percentage)
- Output variation
- Similarity scoring
"""

from typing import Dict, Any, List
from datetime import datetime
from collections import Counter
import difflib

from src.benchmarking.suites.base_suite import BaseSuite
from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.metric_types import SuiteType, ResultStatus, MetricUnit, ModelType
from src.benchmarking.handlers.endpoint_executor import EndpointExecutor
from src.benchmarking.datasets.defaults import get_prompts_for_suite
from src.benchmarking.utils import MemoryOptimizer
from loguru import logger


class QualitySuite(BaseSuite):
    """
    Quality & Consistency benchmark suite.

    Measures output consistency by running the same prompt multiple times
    and analyzing variation:
    - Exact match consistency (% identical outputs)
    - Average similarity (fuzzy matching)
    - Output variation (% unique outputs)
    """

    @property
    def suite_type(self) -> SuiteType:
        """Return suite type."""
        return SuiteType.QUALITY

    def __init__(self, endpoint_executor: EndpointExecutor = None, enable_progress: bool = True):
        """
        Initialize Quality suite.

        Args:
            endpoint_executor: Optional EndpointExecutor instance
            enable_progress: Whether to enable progress display (default: True)
        """
        super().__init__(enable_progress=enable_progress)
        self.executor = endpoint_executor or EndpointExecutor()

    def run(self, config: BenchmarkConfig) -> SuiteResult:
        """
        Execute quality and consistency benchmark.

        Args:
            config: BenchmarkConfig with execution parameters

        Returns:
            SuiteResult with consistency metrics
        """
        start_time = datetime.utcnow()

        try:
            # Setup
            self.setup(config)
            logger.info(f"Starting Quality suite for {len(config.models)} model(s)")

            # Prepare test data
            test_prompts = self._get_test_data(config)
            if not test_prompts:
                raise ValueError("No test prompts available")

            logger.info(f"Using {len(test_prompts)} test prompts for consistency measurement")

            # Run benchmarks for each model
            total_runs_attempted = 0
            total_runs_successful = 0

            for model_id in config.models:
                logger.info(f"Benchmarking model: {model_id}")

                try:
                    for endpoint in config.endpoints.get(model_id, []):
                        logger.info(f"  Endpoint: {endpoint}")

                        # For quality, we run the same prompt multiple times
                        for prompt_idx, prompt in enumerate(test_prompts):
                            logger.info(f"  Testing prompt {prompt_idx + 1}/{len(test_prompts)}: '{prompt[:50]}...'")

                            # Get image path for VLMs (use first image for consistency testing)
                            if config.model_type == ModelType.VLM:
                                images = config.test_data.get("images", [])
                                image_path = images[prompt_idx % len(images)] if images else "-"
                            else:
                                image_path = "-"

                            # Collect outputs for this prompt
                            outputs = []

                            # Warmup runs (if configured)
                            if config.num_warmup > 0:
                                logger.info(f"  Running {config.num_warmup} warmup runs...")
                                for warmup_num in range(1, config.num_warmup + 1):
                                    output = self._execute_quality_run(
                                        model_id=model_id,
                                        endpoint=endpoint,
                                        prompt=prompt,
                                        run_number=warmup_num,
                                        is_warmup=True,
                                        config=config
                                    )
                                    if output:
                                        outputs.append(output)

                            # Counted runs - measure consistency
                            logger.info(f"  Running {config.num_runs} consistency measurement runs...")
                            counted_outputs = []
                            for run_num in range(1, config.num_runs + 1):
                                total_runs_attempted += 1

                                output = self._execute_quality_run(
                                    model_id=model_id,
                                    endpoint=endpoint,
                                    prompt=prompt,
                                    run_number=run_num,
                                    is_warmup=False,
                                    config=config
                                )

                                if output:
                                    counted_outputs.append(output)
                                    total_runs_successful += 1

                            # Compute consistency metrics for counted runs
                            if counted_outputs:
                                self._compute_consistency_metrics(
                                    outputs=counted_outputs,
                                    model_id=model_id,
                                    endpoint=endpoint,
                                    prompt=prompt,
                                    image_path=image_path,
                                    prompt_idx=prompt_idx,
                                    is_warmup=False
                                )

                finally:
                    # Unload model after all runs complete
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
                f"Quality suite complete: {total_runs_successful}/{total_runs_attempted} successful"
            )

            return result

        except Exception as e:
            logger.error(f"Quality suite failed: {str(e)}")
            self.on_run_error(e)

            end_time = datetime.utcnow()
            return self._create_result(start_time, end_time, ResultStatus.FAILED)

        finally:
            self.teardown()

    def _execute_quality_run(
        self,
        model_id: str,
        endpoint: str,
        prompt: str,
        run_number: int,
        is_warmup: bool,
        config: BenchmarkConfig
    ) -> str:
        """
        Execute a single quality benchmark run with memory optimization.

        Args:
            model_id: Model being tested
            endpoint: Endpoint being used
            prompt: Input prompt
            run_number: Run number
            is_warmup: Whether this is a warmup run
            config: Benchmark configuration

        Returns:
            Output text (or None if failed)
        """
        try:
            # Memory optimization: Check and cleanup
            pressure = MemoryOptimizer.memory_pressure_level()
            if pressure == "critical":
                logger.warning(f"Skipping quality run {run_number}: Critical memory")
                MemoryOptimizer.aggressive_cleanup()
                return None
            elif pressure in ["high", "critical"]:
                MemoryOptimizer.aggressive_cleanup()

            # Prepare input data and track for CSV output
            if config.model_type == ModelType.LLM:
                input_data = prompt
                input_image_path = "-"  # LLMs don't use images
            else:  # VLM
                images = config.test_data.get("images", [])
                if not images:
                    logger.warning("No images provided for VLM benchmark")
                    return None

                input_image_path = images[run_number % len(images)]
                input_data = {
                    "prompt": prompt,
                    "image_path": input_image_path
                }

            # Execute inference
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

            # Get output text (untruncated)
            output = result.get("output", "")

            # Create metadata dict with full input/output details for CSV
            run_metadata = {
                "input_prompt": prompt,
                "input_image_path": input_image_path,
                "raw_response": output
            }

            # Record output length for reference
            output_length = len(output)
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

            return output

        except Exception as e:
            logger.error(f"Quality run {run_number} failed: {str(e)}")
            self._record_error(model_id, endpoint, run_number, str(e))
            return None

    def _compute_consistency_metrics(
        self,
        outputs: List[str],
        model_id: str,
        endpoint: str,
        prompt: str,
        image_path: str,
        prompt_idx: int,
        is_warmup: bool
    ):
        """
        Compute consistency metrics from multiple outputs.

        Args:
            outputs: List of output strings from same prompt
            model_id: Model ID
            endpoint: Endpoint name
            prompt: Input prompt text
            image_path: Input image path (or "-" for LLMs)
            prompt_idx: Prompt index
            is_warmup: Whether these are warmup runs
        """
        if not outputs:
            logger.warning("No outputs to compute consistency metrics")
            return

        # Create base metadata with input/output details for CSV
        base_metadata = {
            "input_prompt": prompt,
            "input_image_path": image_path,
            "raw_response": "; ".join(outputs),  # Join all outputs for context
            "total_outputs": len(outputs),
            "unique_outputs": len(set(outputs))
        }

        # 1. Exact match consistency (% identical outputs)
        output_counts = Counter(outputs)
        most_common_output, most_common_count = output_counts.most_common(1)[0]
        exact_match_percent = (most_common_count / len(outputs)) * 100.0

        self._record_metric(
            name="consistency_exact_match_percent",
            value=exact_match_percent,
            unit=MetricUnit.PERCENT,
            run_number=prompt_idx + 1,
            model_id=model_id,
            endpoint=endpoint,
            is_warmup=is_warmup,
            metadata=base_metadata
        )

        # 2. Output variation (% unique outputs)
        variation_percent = (len(output_counts) / len(outputs)) * 100.0

        variation_metadata = base_metadata.copy()
        variation_metadata["unique_count"] = len(output_counts)
        variation_metadata["total_count"] = len(outputs)

        self._record_metric(
            name="output_variation_percent",
            value=variation_percent,
            unit=MetricUnit.PERCENT,
            run_number=prompt_idx + 1,
            model_id=model_id,
            endpoint=endpoint,
            is_warmup=is_warmup,
            metadata=variation_metadata
        )

        # 3. Average similarity (fuzzy matching)
        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                similarity = difflib.SequenceMatcher(None, outputs[i], outputs[j]).ratio()
                similarities.append(similarity)

        if similarities:
            avg_similarity = (sum(similarities) / len(similarities)) * 100.0
        else:
            avg_similarity = 100.0  # Single output = perfect consistency

        similarity_metadata = base_metadata.copy()
        similarity_metadata["comparisons_made"] = len(similarities)

        self._record_metric(
            name="consistency_average_similarity",
            value=avg_similarity,
            unit=MetricUnit.PERCENT,
            run_number=prompt_idx + 1,
            model_id=model_id,
            endpoint=endpoint,
            is_warmup=is_warmup,
            metadata=similarity_metadata
        )

        logger.info(
            f"  Consistency: {exact_match_percent:.1f}% exact, "
            f"{avg_similarity:.1f}% avg similarity, "
            f"{len(output_counts)} unique outputs"
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
        return get_prompts_for_suite("quality", model_type_str)

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
        return f"QualitySuite(executor={self.executor})"
