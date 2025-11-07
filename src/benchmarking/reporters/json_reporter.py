"""
JSON reporter for benchmark results.

Exports complete benchmark results to JSON format with full precision.
"""

from typing import List, Optional
from pathlib import Path
import json
from datetime import datetime

from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.benchmark_config import BenchmarkConfig
from loguru import logger


class JSONReporter:
    """
    Exports benchmark results to JSON format.

    Features:
    - Complete metric preservation (no data loss)
    - Full precision for floating-point values
    - Human-readable formatting (indented)
    - Includes configuration snapshot
    - Supports single or multiple results
    """

    def __init__(self, indent: int = 2):
        """
        Initialize JSON reporter.

        Args:
            indent: Number of spaces for indentation (default: 2)
        """
        self.indent = indent

    def export(
        self,
        results: List[SuiteResult],
        output_path: Path,
        include_config: bool = True
    ) -> Path:
        """
        Export results to JSON file.

        Args:
            results: List of SuiteResult objects to export
            output_path: Path to output JSON file
            include_config: Whether to include config snapshot

        Returns:
            Path to created file

        Raises:
            ValueError: If results list is empty
            IOError: If file write fails
        """
        if not results:
            raise ValueError("No results to export")

        logger.info(f"Exporting {len(results)} result(s) to JSON: {output_path}")

        try:
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data
            export_data = {
                "export_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "format_version": "1.0",
                    "result_count": len(results)
                },
                "results": []
            }

            # Add each result
            for result in results:
                result_data = result.to_dict()

                # Optionally exclude config to reduce size
                if not include_config:
                    result_data.pop("config_snapshot", None)

                export_data["results"].append(result_data)

            # Write to file
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=self.indent, default=self._json_serializer)

            logger.info(f"Successfully exported to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export JSON: {str(e)}")
            raise IOError(f"JSON export failed: {str(e)}") from e

    def export_single(
        self,
        result: SuiteResult,
        output_path: Path,
        include_config: bool = True
    ) -> Path:
        """
        Export single result to JSON file.

        Args:
            result: SuiteResult to export
            output_path: Path to output JSON file
            include_config: Whether to include config snapshot

        Returns:
            Path to created file
        """
        return self.export([result], output_path, include_config)

    def export_summary(
        self,
        results: List[SuiteResult],
        output_path: Path
    ) -> Path:
        """
        Export lightweight summary (no individual metrics).

        Args:
            results: List of SuiteResult objects
            output_path: Path to output JSON file

        Returns:
            Path to created file
        """
        if not results:
            raise ValueError("No results to export")

        logger.info(f"Exporting summary for {len(results)} result(s) to {output_path}")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            summary_data = {
                "export_info": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "format_version": "1.0",
                    "result_count": len(results),
                    "type": "summary"
                },
                "summaries": []
            }

            for result in results:
                summary = {
                    "benchmark_id": result.benchmark_id,
                    "suite_type": result.suite_type.value,
                    "model_type": result.model_type.value,
                    "status": result.status.value,
                    "start_time": result.start_time.isoformat(),
                    "end_time": result.end_time.isoformat() if result.end_time else None,
                    "duration_seconds": result.duration_seconds,
                    "models_tested": result.models_tested,
                    "total_runs": result.total_runs,
                    "successful_runs": result.successful_runs,
                    "failed_runs": result.failed_runs,
                    "success_rate": result.success_rate(),
                    "aggregate_metrics": {}
                }

                # Include aggregate statistics (not individual metrics)
                for metric_name, aggregate in result.aggregates_counted.items():
                    summary["aggregate_metrics"][metric_name] = {
                        "mean": aggregate.mean,
                        "median": aggregate.median,
                        "min": aggregate.min,
                        "max": aggregate.max,
                        "std_dev": aggregate.std_dev,
                        "unit": aggregate.unit.value
                    }

                summary_data["summaries"].append(summary)

            with open(output_path, "w") as f:
                json.dump(summary_data, f, indent=self.indent)

            logger.info(f"Successfully exported summary to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export summary: {str(e)}")
            raise IOError(f"Summary export failed: {str(e)}") from e

    def load(self, input_path: Path) -> List[SuiteResult]:
        """
        Load results from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            List of SuiteResult objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        logger.info(f"Loading results from {input_path}")

        try:
            with open(input_path, "r") as f:
                data = json.load(f)

            # Validate format
            if "results" not in data:
                raise ValueError("Invalid JSON format: missing 'results' key")

            # TODO: Implement full deserialization
            # This would reconstruct SuiteResult objects from JSON
            logger.warning("Full deserialization not yet implemented")
            return []

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {str(e)}")
            raise ValueError(f"Invalid JSON: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to load JSON: {str(e)}")
            raise

    @staticmethod
    def _json_serializer(obj):
        """
        Custom JSON serializer for non-standard types.

        Args:
            obj: Object to serialize

        Returns:
            Serialized representation

        Raises:
            TypeError: If object is not serializable
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'value'):  # Enum types
            return obj.value
        raise TypeError(f"Type {type(obj)} not serializable")

    def validate_output(self, output_path: Path) -> bool:
        """
        Validate that exported JSON is valid.

        Args:
            output_path: Path to JSON file

        Returns:
            True if valid, False otherwise
        """
        try:
            with open(output_path, "r") as f:
                data = json.load(f)

            # Basic validation
            if "export_info" not in data:
                return False
            if "results" not in data:
                return False

            return True

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False

    def __repr__(self) -> str:
        """String representation."""
        return f"JSONReporter(indent={self.indent})"
