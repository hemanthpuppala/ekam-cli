"""
Complete System Analysis benchmark suite.

Orchestrates all benchmark suites to provide comprehensive performance analysis:
- Speed & Throughput
- Resource Efficiency
- Quality & Consistency (if implemented)
- Stress & Endurance (if implemented)
"""

from typing import Dict, List
from datetime import datetime

from src.benchmarking.suites.base_suite import BaseSuite
from src.benchmarking.suites.suite_speed import SpeedSuite
from src.benchmarking.suites.suite_resources import ResourcesSuite
from src.benchmarking.suites.suite_quality import QualitySuite
from src.benchmarking.suites.suite_stress import StressSuite
from src.benchmarking.handlers.endpoint_executor import EndpointExecutor
from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.suite_result import SuiteResult, AggregateStats
from src.benchmarking.models.metric_types import SuiteType, ResultStatus
from src.benchmarking.utils import MemoryOptimizer
from loguru import logger


class CompleteSuite(BaseSuite):
    """
    Complete System Analysis benchmark suite.

    Runs all available benchmark suites and aggregates results
    to provide a comprehensive performance profile.

    Includes:
    - Speed metrics (latency, throughput)
    - Resource metrics (CPU, GPU, memory)
    - Quality metrics (consistency, accuracy) - if available
    - Stress metrics (endurance, stability) - if available
    """

    @property
    def suite_type(self) -> SuiteType:
        """Return suite type."""
        return SuiteType.COMPLETE

    def __init__(self, endpoint_executor: EndpointExecutor = None, enable_progress: bool = True):
        """
        Initialize Complete suite with sub-suites.

        Args:
            endpoint_executor: Optional EndpointExecutor instance to pass to sub-suites
            enable_progress: Whether to enable progress display (default: True)
        """
        super().__init__(enable_progress=enable_progress)

        # Initialize sub-suites with shared executor
        # Disable progress on sub-suites since CompleteSuite will handle overall progress
        self.speed_suite = SpeedSuite(endpoint_executor=endpoint_executor, enable_progress=False)
        self.resources_suite = ResourcesSuite(endpoint_executor=endpoint_executor, enable_progress=False)
        self.quality_suite = QualitySuite(endpoint_executor=endpoint_executor, enable_progress=False)
        self.stress_suite = StressSuite(endpoint_executor=endpoint_executor, enable_progress=False)

    def run(self, config: BenchmarkConfig) -> SuiteResult:
        """
        Execute complete benchmark suite.

        Runs all sub-suites sequentially and aggregates results.

        Args:
            config: BenchmarkConfig with execution parameters

        Returns:
            SuiteResult with aggregated metrics from all suites
        """
        start_time = datetime.utcnow()

        try:
            # Setup
            self.setup(config)
            logger.info(
                f"Starting Complete System Analysis for {len(config.models)} model(s)"
            )

            # Store individual suite results
            suite_results = {}

            # Read suite-specific run configurations from parameters
            speed_runs = config.parameters.get("speed_runs", config.num_runs)
            resources_runs = config.parameters.get("resources_runs", config.num_runs)
            quality_runs = config.parameters.get("quality_runs", config.num_runs)
            stress_duration = config.parameters.get("stress_duration_minutes", 15)

            logger.info(f"Complete suite configuration:")
            logger.info(f"  Speed runs: {speed_runs}")
            logger.info(f"  Resources runs: {resources_runs}")
            logger.info(f"  Quality runs: {quality_runs}")
            logger.info(f"  Stress duration: {stress_duration} minutes")

            # Run Speed suite
            logger.info("=" * 60)
            logger.info("Running Speed & Throughput suite...")
            logger.info("=" * 60)
            try:
                speed_config = config.model_copy(update={"num_runs": speed_runs})
                speed_result = self.speed_suite.run(speed_config)
                suite_results["speed"] = speed_result
                logger.info(f"Speed suite completed with status: {speed_result.status.value}")
            except Exception as e:
                logger.error(f"Speed suite failed: {str(e)}")
                suite_results["speed"] = None

            # Run Resources suite
            logger.info("=" * 60)
            logger.info("Running Resource Efficiency suite...")
            logger.info("=" * 60)
            try:
                resources_config = config.model_copy(update={"num_runs": resources_runs})
                resources_result = self.resources_suite.run(resources_config)
                suite_results["resources"] = resources_result
                logger.info(f"Resources suite completed with status: {resources_result.status.value}")
            except Exception as e:
                logger.error(f"Resources suite failed: {str(e)}")
                suite_results["resources"] = None

            # Run Quality suite
            logger.info("=" * 60)
            logger.info("Running Quality & Consistency suite...")
            logger.info("=" * 60)
            try:
                quality_config = config.model_copy(update={"num_runs": quality_runs})
                quality_result = self.quality_suite.run(quality_config)
                suite_results["quality"] = quality_result
                logger.info(f"Quality suite completed with status: {quality_result.status.value}")
            except Exception as e:
                logger.error(f"Quality suite failed: {str(e)}")
                suite_results["quality"] = None

            # Run Stress suite
            logger.info("=" * 60)
            logger.info("Running Stress & Endurance suite...")
            logger.info("=" * 60)
            try:
                # Create custom config for stress suite with duration parameter
                stress_params = config.parameters.copy()
                stress_params["duration_minutes"] = stress_duration
                stress_config = config.model_copy(update={"parameters": stress_params})
                stress_result = self.stress_suite.run(stress_config)
                suite_results["stress"] = stress_result
                logger.info(f"Stress suite completed with status: {stress_result.status.value}")
            except Exception as e:
                logger.error(f"Stress suite failed: {str(e)}")
                suite_results["stress"] = None

            # Aggregate all metrics from sub-suites
            self._aggregate_suite_results(suite_results)

            # Determine overall status
            status = self._determine_overall_status(suite_results)

            # Create aggregated result
            end_time = datetime.utcnow()
            result = self._create_result(start_time, end_time, status)

            # Add sub-suite summaries to metadata
            result.system_info["sub_suites"] = {
                suite_name: {
                    "status": sr.status.value if sr else "failed",
                    "total_runs": sr.total_runs if sr else 0,
                    "successful_runs": sr.successful_runs if sr else 0,
                }
                for suite_name, sr in suite_results.items()
            }

            logger.info("=" * 60)
            logger.info("Complete System Analysis finished")
            logger.info(f"Overall status: {status.value}")
            logger.info(f"Total duration: {result.duration_seconds:.2f}s")
            logger.info("=" * 60)

            return result

        except Exception as e:
            logger.error(f"Complete suite failed: {str(e)}")
            self.on_run_error(e)

            end_time = datetime.utcnow()
            return self._create_result(start_time, end_time, ResultStatus.FAILED)

        finally:
            self.teardown()

    def _aggregate_suite_results(self, suite_results: Dict[str, SuiteResult]) -> None:
        """
        Aggregate metrics from all sub-suite results.

        Collects all metrics from sub-suites into this suite's metrics collector.

        Args:
            suite_results: Dictionary of suite name to SuiteResult
        """
        for suite_name, result in suite_results.items():
            if result is None:
                continue

            # Add all metrics from this suite
            for metric in result.metrics:
                self.metrics_collector.record_metric(
                    name=f"{suite_name}_{metric.name}",
                    value=metric.value,
                    unit=metric.unit,
                    run_number=metric.run_number,
                    model_id=metric.model_id,
                    endpoint=metric.endpoint,
                    is_warmup=metric.is_warmup,
                    metadata={"source_suite": suite_name, **metric.metadata}
                )

    def _determine_overall_status(self, suite_results: Dict[str, SuiteResult]) -> ResultStatus:
        """
        Determine overall status from sub-suite results.

        Args:
            suite_results: Dictionary of suite results

        Returns:
            Overall ResultStatus
        """
        if not suite_results:
            return ResultStatus.FAILED

        # Count statuses
        success_count = 0
        failed_count = 0
        partial_count = 0

        for result in suite_results.values():
            if result is None:
                failed_count += 1
            elif result.status == ResultStatus.SUCCESS:
                success_count += 1
            elif result.status == ResultStatus.FAILED:
                failed_count += 1
            elif result.status == ResultStatus.PARTIAL:
                partial_count += 1

        # Determine overall
        total = len(suite_results)
        if success_count == total:
            return ResultStatus.SUCCESS
        elif failed_count == total:
            return ResultStatus.FAILED
        else:
            return ResultStatus.PARTIAL

    def _create_result(
        self,
        start_time: datetime,
        end_time: datetime,
        status: ResultStatus
    ) -> SuiteResult:
        """
        Override base _create_result to properly count runs across sub-suites.

        The Complete suite aggregates metrics from multiple sub-suites where
        run_numbers overlap (Speed run 1, Quality run 1, Stress run 1 are all different).
        We need to include source_suite in the uniqueness key.

        Args:
            start_time: When suite started
            end_time: When suite ended
            status: Final execution status

        Returns:
            SuiteResult with correctly counted runs
        """
        if not self._config:
            raise ValueError("Config not set - call setup() first")

        # Compute aggregates
        aggregates = self._compute_aggregates()

        # Count successful and failed runs
        all_metrics = self.metrics_collector.metrics_list()

        # Get unique run attempts INCLUDING source_suite to avoid collisions
        successful_run_keys = set()
        for m in all_metrics:
            if not m.is_warmup:
                source_suite = m.metadata.get('source_suite', 'unknown')
                successful_run_keys.add((m.model_id, m.endpoint, source_suite, m.run_number))

        # Get unique run attempts from errors
        failed_run_keys = set()
        for e in self._errors:
            source_suite = e.get('metadata', {}).get('source_suite', 'unknown')
            failed_run_keys.add((e['model_id'], e['endpoint'], source_suite, e['run_number']))

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

    def setup(self, config: BenchmarkConfig) -> None:
        """
        Setup before benchmark execution.

        Args:
            config: BenchmarkConfig
        """
        super().setup(config)
        logger.info("Setting up Complete suite with all sub-suites")

    def teardown(self) -> None:
        """Cleanup after benchmark execution."""
        super().teardown()
        logger.info("Complete suite teardown finished")

    def _execute_inference(self, model_id: str, endpoint: str, input_data, parameters):
        """
        Not used for Complete suite - delegates to sub-suites.

        Args:
            model_id: Model ID
            endpoint: Endpoint
            input_data: Input data
            parameters: Parameters

        Returns:
            None
        """
        raise NotImplementedError("Complete suite delegates to sub-suites")

    def get_suite_summary(self, result: SuiteResult) -> str:
        """
        Generate human-readable summary of complete analysis.

        Args:
            result: SuiteResult from complete suite

        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            "Complete System Analysis Summary",
            "=" * 60,
            f"Status: {result.status.value.upper()}",
            f"Duration: {result.duration_seconds:.2f}s",
            f"Models tested: {', '.join(result.models_tested)}",
            f"Total runs: {result.total_runs}",
            f"Successful: {result.successful_runs}",
            f"Failed: {result.failed_runs}",
            "",
            "Sub-Suite Results:",
        ]

        # Add sub-suite summaries
        if "sub_suites" in result.system_info:
            for suite_name, suite_info in result.system_info["sub_suites"].items():
                lines.append(
                    f"  {suite_name.title()}: {suite_info['status']} "
                    f"({suite_info['successful_runs']}/{suite_info['total_runs']} runs)"
                )

        lines.append("=" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return "CompleteSuite(speed, resources)"
