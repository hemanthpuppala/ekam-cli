"""
CSV reporter for benchmark results.

Exports benchmark results to CSV format for spreadsheet analysis.
"""

from typing import List, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.benchmarking.models.suite_result import SuiteResult
from loguru import logger


class CSVReporter:
    """
    Exports benchmark results to CSV format.

    Features:
    - Tabular format for spreadsheet tools
    - One row per metric observation
    - Aggregate statistics in separate sheet
    - Compatible with Excel, Google Sheets
    """

    def __init__(self):
        """Initialize CSV reporter."""
        pass

    def export(
        self,
        results: List[SuiteResult],
        output_path: Path,
        include_metrics: bool = True,
        include_aggregates: bool = True
    ) -> Path:
        """
        Export results to CSV file(s).

        Creates multiple CSV files:
        - {name}_metrics.csv: Individual metric observations
        - {name}_aggregates.csv: Aggregate statistics

        Args:
            results: List of SuiteResult objects to export
            output_path: Base path for output CSV files
            include_metrics: Whether to export individual metrics
            include_aggregates: Whether to export aggregate statistics

        Returns:
            Path to output directory

        Raises:
            ValueError: If results list is empty
            IOError: If file write fails
        """
        if not results:
            raise ValueError("No results to export")

        logger.info(f"Exporting {len(results)} result(s) to CSV: {output_path}")

        try:
            # Create output directory
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            base_name = output_path.stem

            # Export individual metrics
            if include_metrics:
                metrics_path = output_dir / f"{base_name}_metrics.csv"
                self._export_metrics(results, metrics_path)

            # Export aggregate statistics
            if include_aggregates:
                aggregates_path = output_dir / f"{base_name}_aggregates.csv"
                self._export_aggregates(results, aggregates_path)

            logger.info(f"Successfully exported to {output_dir}")
            return output_dir

        except Exception as e:
            logger.error(f"Failed to export CSV: {str(e)}")
            raise IOError(f"CSV export failed: {str(e)}") from e

    def _export_metrics(self, results: List[SuiteResult], output_path: Path) -> None:
        """
        Export individual metrics to CSV.

        Args:
            results: List of SuiteResult objects
            output_path: Output CSV file path
        """
        rows = []

        for result in results:
            for metric in result.metrics:
                # Extract input/output data from metadata (if available)
                input_prompt = metric.metadata.get("input_prompt", "")
                input_image_path = metric.metadata.get("input_image_path", "-")
                output_image_path = metric.metadata.get("output_image_path", "-")
                raw_response = metric.metadata.get("raw_response", "")

                row = {
                    "benchmark_id": result.benchmark_id,
                    "suite_type": result.suite_type.value,
                    "model_type": result.model_type.value,
                    "model_id": metric.model_id,
                    "endpoint": metric.endpoint,
                    "metric_name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit.value,
                    "run_number": metric.run_number,
                    "is_warmup": metric.is_warmup,
                    "timestamp": metric.timestamp.isoformat(),
                    # NEW: Input/Output columns for full visibility
                    "input_prompt": input_prompt,
                    "input_image_path": input_image_path,
                    "output_image_path": output_image_path,
                    "raw_response": raw_response
                }

                # Add remaining metadata fields (excluding the ones we already extracted)
                for key, value in metric.metadata.items():
                    if key not in ["input_prompt", "input_image_path", "output_image_path", "raw_response"]:
                        row[f"meta_{key}"] = value

                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(rows)} metrics to {output_path}")
        else:
            logger.warning("No metrics to export")

    def _export_aggregates(self, results: List[SuiteResult], output_path: Path) -> None:
        """
        Export aggregate statistics to CSV.

        Exports 3 levels of aggregation:
        1. Global (all models combined)
        2. Per-model (each model separately)
        3. Per-endpoint (each endpoint separately)

        Args:
            results: List of SuiteResult objects
            output_path: Output CSV file path
        """
        rows = []

        for result in results:
            # 1. GLOBAL AGGREGATES (all models combined)

            # Global counted aggregates
            for metric_name, aggregate in result.aggregates_counted.items():
                row = {
                    "benchmark_id": result.benchmark_id,
                    "suite_type": result.suite_type.value,
                    "model_type": result.model_type.value,
                    "aggregation_level": "global",
                    "aggregation_type": "counted",
                    "model_id": "ALL",
                    "endpoint": "ALL",
                    "metric_name": aggregate.metric_name,
                    "unit": aggregate.unit.value,
                    "count": aggregate.count,
                    "mean": aggregate.mean,
                    "median": aggregate.median,
                    "min": aggregate.min,
                    "max": aggregate.max,
                    "std_dev": aggregate.std_dev,
                    "variance": aggregate.variance,
                    "p25": aggregate.p25,
                    "p50": aggregate.p50,
                    "p75": aggregate.p75,
                    "p95": aggregate.p95,
                    "p99": aggregate.p99
                }
                rows.append(row)

            # Global warmup aggregates
            for metric_name, aggregate in result.aggregates_warmup.items():
                row = {
                    "benchmark_id": result.benchmark_id,
                    "suite_type": result.suite_type.value,
                    "model_type": result.model_type.value,
                    "aggregation_level": "global",
                    "aggregation_type": "warmup",
                    "model_id": "ALL",
                    "endpoint": "ALL",
                    "metric_name": aggregate.metric_name,
                    "unit": aggregate.unit.value,
                    "count": aggregate.count,
                    "mean": aggregate.mean,
                    "median": aggregate.median,
                    "min": aggregate.min,
                    "max": aggregate.max,
                    "std_dev": aggregate.std_dev,
                    "variance": aggregate.variance,
                    "p25": aggregate.p25,
                    "p50": aggregate.p50,
                    "p75": aggregate.p75,
                    "p95": aggregate.p95,
                    "p99": aggregate.p99
                }
                rows.append(row)

            # 2. PER-MODEL AGGREGATES

            if hasattr(result, 'aggregates_per_model') and result.aggregates_per_model:
                for model_id, model_aggregates in result.aggregates_per_model.items():
                    for metric_name, aggregate in model_aggregates.items():
                        # Determine if warmup or counted based on metric metadata
                        is_warmup = getattr(aggregate, 'is_warmup', False)
                        agg_type = "warmup" if is_warmup else "counted"

                        row = {
                            "benchmark_id": result.benchmark_id,
                            "suite_type": result.suite_type.value,
                            "model_type": result.model_type.value,
                            "aggregation_level": "per_model",
                            "aggregation_type": agg_type,
                            "model_id": model_id,
                            "endpoint": "ALL",
                            "metric_name": aggregate.metric_name,
                            "unit": aggregate.unit.value,
                            "count": aggregate.count,
                            "mean": aggregate.mean,
                            "median": aggregate.median,
                            "min": aggregate.min,
                            "max": aggregate.max,
                            "std_dev": aggregate.std_dev,
                            "variance": aggregate.variance,
                            "p25": aggregate.p25,
                            "p50": aggregate.p50,
                            "p75": aggregate.p75,
                            "p95": aggregate.p95,
                            "p99": aggregate.p99
                        }
                        rows.append(row)

            # 3. PER-ENDPOINT AGGREGATES

            if hasattr(result, 'aggregates_per_endpoint') and result.aggregates_per_endpoint:
                for endpoint, endpoint_aggregates in result.aggregates_per_endpoint.items():
                    for metric_name, aggregate in endpoint_aggregates.items():
                        is_warmup = getattr(aggregate, 'is_warmup', False)
                        agg_type = "warmup" if is_warmup else "counted"

                        row = {
                            "benchmark_id": result.benchmark_id,
                            "suite_type": result.suite_type.value,
                            "model_type": result.model_type.value,
                            "aggregation_level": "per_endpoint",
                            "aggregation_type": agg_type,
                            "model_id": "ALL",
                            "endpoint": endpoint,
                            "metric_name": aggregate.metric_name,
                            "unit": aggregate.unit.value,
                            "count": aggregate.count,
                            "mean": aggregate.mean,
                            "median": aggregate.median,
                            "min": aggregate.min,
                            "max": aggregate.max,
                            "std_dev": aggregate.std_dev,
                            "variance": aggregate.variance,
                            "p25": aggregate.p25,
                            "p50": aggregate.p50,
                            "p75": aggregate.p75,
                            "p95": aggregate.p95,
                            "p99": aggregate.p99
                        }
                        rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(rows)} aggregates ({len([r for r in rows if r['aggregation_level']=='per_model'])} per-model, {len([r for r in rows if r['aggregation_level']=='per_endpoint'])} per-endpoint) to {output_path}")
        else:
            logger.warning("No aggregates to export")

    def export_comparison(
        self,
        results: List[SuiteResult],
        output_path: Path,
        metric_name: str = "latency_ms"
    ) -> Path:
        """
        Export comparison table for a specific metric across models.

        Args:
            results: List of SuiteResult objects
            output_path: Output CSV file path
            metric_name: Metric to compare

        Returns:
            Path to created file
        """
        logger.info(f"Exporting comparison for metric: {metric_name}")

        rows = []

        for result in results:
            for model_id in result.models_tested:
                # Get per-model aggregate for this metric
                if model_id in result.aggregates_per_model:
                    model_aggregates = result.aggregates_per_model[model_id]

                    if metric_name in model_aggregates:
                        aggregate = model_aggregates[metric_name]

                        row = {
                            "benchmark_id": result.benchmark_id,
                            "suite_type": result.suite_type.value,
                            "model_id": model_id,
                            "metric": metric_name,
                            "unit": aggregate.unit.value,
                            "mean": aggregate.mean,
                            "median": aggregate.median,
                            "min": aggregate.min,
                            "max": aggregate.max,
                            "std_dev": aggregate.std_dev
                        }
                        rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported comparison to {output_path}")
            return output_path
        else:
            logger.warning(f"No data found for metric: {metric_name}")
            return output_path

    def export_summary(
        self,
        results: List[SuiteResult],
        output_path: Path
    ) -> Path:
        """
        Export high-level summary table.

        Args:
            results: List of SuiteResult objects
            output_path: Output CSV file path

        Returns:
            Path to created file
        """
        rows = []

        for result in results:
            row = {
                "benchmark_id": result.benchmark_id,
                "suite_type": result.suite_type.value,
                "model_type": result.model_type.value,
                "status": result.status.value,
                "start_time": result.start_time.isoformat(),
                "duration_seconds": result.duration_seconds,
                "models_tested": ", ".join(result.models_tested),
                "total_runs": result.total_runs,
                "successful_runs": result.successful_runs,
                "failed_runs": result.failed_runs,
                "success_rate_percent": result.success_rate()
            }
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported summary to {output_path}")
            return output_path
        else:
            logger.warning("No results to export")
            return output_path

    def __repr__(self) -> str:
        """String representation."""
        return "CSVReporter()"
