"""
Benchmark runner orchestrator.

Main entry point for executing benchmarks with support for:
- Foreground execution (blocking with progress)
- Background execution (non-blocking with threading)
- Result management and export
"""

from typing import Optional, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
import threading
import queue

from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.metric_types import SuiteType, ExecutionMode
from src.benchmarking.core.config_validator import ConfigValidator
from src.benchmarking.core.results_manager import ResultsManager
from src.benchmarking.core.provider_bridge import ProviderBridge
from src.benchmarking.core.model_context import ModelContextManager
from src.benchmarking.suites.suite_speed import SpeedSuite
from src.benchmarking.suites.suite_resources import ResourcesSuite
from src.benchmarking.suites.suite_complete import CompleteSuite
from src.benchmarking.reporters.json_reporter import JSONReporter
from src.benchmarking.reporters.csv_reporter import CSVReporter
from src.benchmarking.reporters.console_formatter import ConsoleFormatter
from loguru import logger
try:
    from src.benchmarking.reporters.graph_generator import GraphGenerator
except Exception as _graph_err:
    # Minimal fallback to avoid crashing when matplotlib style library
    # cannot be decoded on some macOS setups with AppleDouble files.
    logger.warning(f"Graph generator unavailable, disabling graphs: {_graph_err}")
    class GraphGenerator:  # type: ignore
        def generate_graphs(self, *args, **kwargs):
            return []
from src.benchmarking.reporters.html_reporter import HTMLReporter
from src.benchmarking.reporters.dashboard_generator import DashboardGenerator
from src.services.session import SessionManager


class BenchmarkRunner:
    """
    Main orchestrator for benchmark execution.

    Responsibilities:
    - Validate configurations
    - Execute benchmarks (foreground or background)
    - Manage results storage
    - Export results in multiple formats
    - Provide progress callbacks
    """

    def __init__(self, session_manager: SessionManager, results_dir: Optional[Path] = None):
        """
        Initialize benchmark runner.

        Args:
            session_manager: SessionManager for model discovery and execution
            results_dir: Directory for storing results (default: ./results)
        """
        self.session_manager = session_manager
        self.provider_bridge = ProviderBridge(session_manager)
        self.model_context_manager = ModelContextManager(self.provider_bridge)
        self.results_manager = ResultsManager(results_dir)
        self.json_reporter = JSONReporter()
        self.csv_reporter = CSVReporter()
        self.console_formatter = ConsoleFormatter()
        self.graph_generator = GraphGenerator()
        self.html_reporter = HTMLReporter()
        self.dashboard_generator = DashboardGenerator()

        # Background execution state
        self._background_threads: Dict[str, threading.Thread] = {}
        self._background_results: Dict[str, Optional[SuiteResult]] = {}
        self._progress_queues: Dict[str, queue.Queue] = {}

    def run(
        self,
        config: BenchmarkConfig,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> SuiteResult:
        """
        Execute benchmark based on configuration.

        Args:
            config: BenchmarkConfig with execution parameters
            progress_callback: Optional callback for progress updates

        Returns:
            SuiteResult with benchmark metrics

        Raises:
            ConfigValidationError: If configuration is invalid
            RuntimeError: If execution fails
        """
        # Validate configuration
        ConfigValidator.validate_or_raise(config)

        # Log warnings
        warnings = ConfigValidator.get_warnings(config)
        for warning in warnings:
            logger.warning(warning)

        # Execute in foreground with live updates (background disabled)
        return self._run_foreground(config, progress_callback)

    def _run_foreground(
        self,
        config: BenchmarkConfig,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> SuiteResult:
        """
        Execute benchmark in foreground (blocking).

        Args:
            config: BenchmarkConfig
            progress_callback: Optional progress callback

        Returns:
            SuiteResult
        """
        logger.info(f"Starting foreground benchmark: {config.benchmark_id}")

        # Enable non-interactive provider behavior for benchmarks
        import os
        os.environ["EKAM_BENCHMARK_MODE"] = "1"
        # Allow global override of context window via parameters; default to 2048
        try:
            n_ctx_param = None
            if config.parameters:
                n_ctx_param = config.parameters.get("n_ctx")
            os.environ["EKAM_N_CTX"] = str(int(n_ctx_param)) if n_ctx_param is not None else "2048"
        except Exception:
            os.environ["EKAM_N_CTX"] = "2048"

        if progress_callback:
            progress_callback(f"Starting {config.suite_type.display_name()}...")

        # Construct result directory path for this benchmark
        # Format: results/YYYY-MM-DD/benchmark_id/
        date_str = datetime.now().strftime("%Y-%m-%d")
        result_dir = self.results_manager.base_dir / date_str / config.benchmark_id

        # Get appropriate suite with progress callback
        suite = self._get_suite(config.suite_type, progress_callback=progress_callback, model_context_manager=self.model_context_manager)

        # Set result directory on executor's VLM handler for visualization outputs
        if hasattr(suite, 'executor') and hasattr(suite.executor, 'vlm_handler'):
            suite.executor.vlm_handler.set_result_dir(result_dir)
            logger.debug(f"Set VLM handler result_dir to: {result_dir}")

        # Start live UI dashboard
        failed_models_for_ui = getattr(self.provider_bridge, "_recent_load_failures", set())
        try:
            from .live_dashboard import BenchmarkLiveUI
            ui = BenchmarkLiveUI(
                progress=suite.progress_display if hasattr(suite, 'progress_display') else None,
                suite_name=config.suite_type.display_name(),
                config_params=config.parameters or {},
                models=list(config.models),
                endpoints=dict(config.endpoints),
                num_warmup=config.num_warmup,
                num_runs=config.num_runs,
                failed_models_ref=failed_models_for_ui,
            )
        except Exception:
            ui = None

        if ui and suite.progress_display:
            try:
                ui.start()
            except Exception:
                ui = None

        # Execute benchmark
        result = suite.run(config)

        # Stop UI
        if ui:
            try:
                ui.stop()
            except Exception:
                pass

        if progress_callback:
            progress_callback(f"Benchmark complete: {result.status.value}")

        # Cleanup models after benchmark
        logger.debug("Cleaning up model contexts after benchmark")
        self.model_context_manager.cleanup_all()

        # Save results
        self._save_results(result, config)

        # After first run, retry models that failed to load (once)
        try:
            failed_models = []
            if hasattr(self.provider_bridge, "get_and_clear_load_failures"):
                failed_models = self.provider_bridge.get_and_clear_load_failures()

            failed_models = [m for m in failed_models if m in config.models]
            if failed_models:
                logger.info(f"Retrying failed-to-load models at end: {failed_models}")
                retry_config = BenchmarkConfig(
                    model_type=config.model_type,
                    suite_type=config.suite_type,
                    models=failed_models,
                    endpoints={k: v for k, v in config.endpoints.items() if k in failed_models},
                    test_data=config.test_data,
                    parameters=config.parameters,
                    num_runs=config.num_runs,
                    num_warmup=config.num_warmup,
                    execution_mode=config.execution_mode,
                    export_formats=config.export_formats,
                )
                # Best-effort retry run (no separate UI)
                _ = suite.run(retry_config)
        except Exception as e:
            logger.warning(f"Retry pass for failed models skipped due to error: {e}")

        # Export in requested formats
        self._export_results(result, config)

        logger.info(f"Foreground benchmark complete: {config.benchmark_id}")
        return result

    def _run_background(
        self,
        config: BenchmarkConfig,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> SuiteResult:
        """
        Execute benchmark in background (non-blocking).

        Args:
            config: BenchmarkConfig
            progress_callback: Optional progress callback

        Returns:
            SuiteResult (placeholder - actual result available via get_background_result)
        """
        logger.info(f"Starting background benchmark: {config.benchmark_id}")

        # Create progress queue
        progress_queue = queue.Queue()
        self._progress_queues[config.benchmark_id] = progress_queue

        # Create and start background thread
        def background_worker():
            try:
                # Construct result directory path for this benchmark
                date_str = datetime.now().strftime("%Y-%m-%d")
                result_dir = self.results_manager.base_dir / date_str / config.benchmark_id

                suite = self._get_suite(config.suite_type, model_context_manager=self.model_context_manager)

                # Set result directory on executor's VLM handler
                if hasattr(suite, 'executor') and hasattr(suite.executor, 'vlm_handler'):
                    suite.executor.vlm_handler.set_result_dir(result_dir)

                result = suite.run(config)

                # Cleanup models after benchmark
                logger.debug("Cleaning up model contexts after background benchmark")
                self.model_context_manager.cleanup_all()

                # Save and export
                self._save_results(result, config)
                self._export_results(result, config)

                # Store result
                self._background_results[config.benchmark_id] = result

                progress_queue.put(("complete", result.status.value))

            except Exception as e:
                logger.error(f"Background benchmark failed: {str(e)}")
                # Cleanup even on error
                self.model_context_manager.cleanup_all()
                progress_queue.put(("error", str(e)))
                self._background_results[config.benchmark_id] = None

        thread = threading.Thread(
            target=background_worker,
            name=f"benchmark-{config.benchmark_id}",
            daemon=True
        )
        thread.start()

        self._background_threads[config.benchmark_id] = thread

        logger.info(f"Background benchmark started: {config.benchmark_id}")

        # Return placeholder result
        from src.benchmarking.models.metric_types import ResultStatus
        placeholder = SuiteResult(
            suite_type=config.suite_type,
            benchmark_id=config.benchmark_id,
            model_type=config.model_type,
            start_time=datetime.utcnow(),
            status=ResultStatus.SUCCESS,
            models_tested=config.models,
            endpoints_tested=config.endpoints,
            total_runs=0,
            successful_runs=0,
            failed_runs=0
        )

        return placeholder

    def get_background_result(self, benchmark_id: str) -> Optional[SuiteResult]:
        """
        Get result from background benchmark.

        Args:
            benchmark_id: Benchmark ID

        Returns:
            SuiteResult if complete, None if still running or not found
        """
        return self._background_results.get(benchmark_id)

    def is_background_complete(self, benchmark_id: str) -> bool:
        """
        Check if background benchmark is complete.

        Args:
            benchmark_id: Benchmark ID

        Returns:
            True if complete, False if still running
        """
        if benchmark_id not in self._background_threads:
            return False

        thread = self._background_threads[benchmark_id]
        return not thread.is_alive()

    def cancel_background(self, benchmark_id: str) -> bool:
        """
        Attempt to cancel a background benchmark.

        Note: This sets a flag but doesn't forcefully terminate the thread.

        Args:
            benchmark_id: Benchmark ID

        Returns:
            True if cancellation initiated, False if not found
        """
        if benchmark_id not in self._background_threads:
            return False

        logger.info(f"Cancellation requested for: {benchmark_id}")
        # Note: Python threads cannot be forcefully stopped
        # This would require implementing cooperative cancellation in suites
        return True

    def _get_suite(self, suite_type: SuiteType, progress_callback: Optional[Callable[[str], None]] = None, model_context_manager: Optional[ModelContextManager] = None):
        """
        Get suite instance for suite type.

        Args:
            suite_type: SuiteType enum
            progress_callback: Optional progress callback to propagate to suite

        Returns:
            Suite instance
        """
        # Import here to avoid circular imports
        from src.benchmarking.handlers.endpoint_executor import EndpointExecutor

        # Create executor with provider bridge for real model inference
        executor = EndpointExecutor(provider_bridge=self.provider_bridge)

        # Import suites
        from src.benchmarking.suites.suite_quality import QualitySuite
        from src.benchmarking.suites.suite_stress import StressSuite

        if suite_type == SuiteType.SPEED:
            suite = SpeedSuite(endpoint_executor=executor, model_context_manager=model_context_manager)
        elif suite_type == SuiteType.RESOURCES:
            suite = ResourcesSuite(endpoint_executor=executor)
        elif suite_type == SuiteType.COMPLETE:
            suite = CompleteSuite(endpoint_executor=executor)
        elif suite_type == SuiteType.QUALITY:
            suite = QualitySuite(endpoint_executor=executor)
        elif suite_type == SuiteType.STRESS:
            suite = StressSuite(endpoint_executor=executor)
        else:
            raise ValueError(f"Unknown suite type: {suite_type}")

        # Set progress callback on suite if provided
        if progress_callback:
            suite.set_progress_callback(progress_callback)

        return suite

    def _save_results(self, result: SuiteResult, config: BenchmarkConfig) -> None:
        """
        Save results to filesystem.

        Args:
            result: SuiteResult to save
            config: BenchmarkConfig used
        """
        try:
            result_dir = self.results_manager.save_result(result, config)
            logger.info(f"Results saved to: {result_dir}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")

    def _export_results(self, result: SuiteResult, config: BenchmarkConfig) -> None:
        """
        Export results in requested formats.

        Args:
            result: SuiteResult to export
            config: BenchmarkConfig with export_formats
        """
        result_dir = self.results_manager.get_result_path(
            result.benchmark_id,
            result.start_time.strftime("%Y-%m-%d")
        )

        if not result_dir:
            logger.warning("Result directory not found for export")
            return

        for format_name in config.export_formats:
            try:
                if format_name == "json":
                    output_path = result_dir / "result.json"
                    self.json_reporter.export_single(result, output_path)
                    logger.info(f"Exported JSON to: {output_path}")

                elif format_name == "csv":
                    output_path = result_dir / "result.csv"
                    self.csv_reporter.export([result], output_path)
                    logger.info(f"Exported CSV to: {output_path}")

                elif format_name == "html":
                    output_path = result_dir / "report.html"

                    # Load CSV data for charts
                    import pandas as pd
                    metrics_df = None
                    aggregates_df = None

                    metrics_csv = result_dir / "result_metrics.csv"
                    if metrics_csv.exists():
                        metrics_df = pd.read_csv(metrics_csv)

                    aggregates_csv = result_dir / "result_aggregates.csv"
                    if aggregates_csv.exists():
                        aggregates_df = pd.read_csv(aggregates_csv)

                    self.html_reporter.generate_html_report(
                        result,
                        output_path,
                        metrics_df=metrics_df,
                        aggregates_df=aggregates_df
                    )
                    logger.info(f"Exported HTML report to: {output_path}")

            except Exception as e:
                logger.error(f"Failed to export {format_name}: {str(e)}")

        # Generate graphs after all exports complete
        try:
            logger.info("Generating benchmark graphs...")
            graph_files = self.graph_generator.generate_graphs(result, result_dir)

            if graph_files:
                logger.info(f"Generated {len(graph_files)} graph files:")
                for graph_file in graph_files:
                    logger.info(f"  - {graph_file.name}")
            else:
                logger.info("No graphs generated (data may not be available)")
        except Exception as e:
            logger.error(f"Failed to generate graphs: {str(e)}")
            logger.debug(f"Graph generation error details:", exc_info=True)

        # Generate interactive dashboard
        try:
            logger.info("Generating interactive dashboard...")
            dashboard_path = self.dashboard_generator.generate_dashboard(result_dir, result)

            if dashboard_path and dashboard_path.exists():
                logger.info(f"Dashboard generated: {dashboard_path}")
                # Show user-friendly message about opening the dashboard
                self.dashboard_generator.show_dashboard_message(dashboard_path)
            else:
                logger.warning("Failed to generate dashboard")
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {str(e)}")
            logger.debug(f"Dashboard generation error details:", exc_info=True)

    def list_results(
        self,
        suite_type: Optional[str] = None,
        limit: int = 10
    ) -> list:
        """
        List available benchmark results.

        Args:
            suite_type: Optional filter by suite type
            limit: Maximum results to return

        Returns:
            List of result metadata
        """
        return self.results_manager.list_results(
            suite_type=suite_type,
            limit=limit
        )

    def load_result(self, benchmark_id: str, date: Optional[str] = None) -> Optional[SuiteResult]:
        """
        Load a previous benchmark result.

        Args:
            benchmark_id: Benchmark ID
            date: Optional date (YYYY-MM-DD)

        Returns:
            SuiteResult if found, None otherwise
        """
        return self.results_manager.load_result(benchmark_id, date)

    def print_result(self, result: SuiteResult, detailed: bool = True) -> None:
        """
        Print result to console.

        Args:
            result: SuiteResult to display
            detailed: Whether to show detailed metrics
        """
        self.console_formatter.print_result(result, detailed)

    def __repr__(self) -> str:
        """String representation."""
        active_background = len([t for t in self._background_threads.values() if t.is_alive()])
        model_summary = self.model_context_manager.get_resource_summary()
        return (
            f"BenchmarkRunner("
            f"active_background={active_background}, "
            f"loaded_models={model_summary['loaded_models']}"
            f")"
        )
