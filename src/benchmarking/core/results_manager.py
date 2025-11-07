"""
Results management and persistence.

Handles saving, loading, and organizing benchmark results on the filesystem.
"""

from typing import Optional, List
from pathlib import Path
from datetime import datetime
import json

from src.benchmarking.models.suite_result import SuiteResult
from src.benchmarking.models.benchmark_config import BenchmarkConfig


class ResultsManager:
    """
    Manages benchmark results persistence.

    Responsibilities:
    - Save results to structured directory (results/YYYY-MM-DD/benchmark_id/)
    - Load previous results
    - List available results
    - Organize results by date and benchmark ID
    """

    def __init__(self, base_results_dir: Optional[Path] = None):
        """
        Initialize results manager.

        Args:
            base_results_dir: Base directory for results (default: ./results)
        """
        self.base_dir = base_results_dir or Path("results")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_result(
        self,
        result: SuiteResult,
        config: Optional[BenchmarkConfig] = None
    ) -> Path:
        """
        Save benchmark result to filesystem.

        Creates directory structure: results/YYYY-MM-DD/benchmark_id/

        Args:
            result: SuiteResult to save
            config: Optional BenchmarkConfig to save alongside result

        Returns:
            Path to saved result directory
        """
        # Create dated directory
        date_str = result.start_time.strftime("%Y-%m-%d")
        result_dir = self.base_dir / date_str / result.benchmark_id
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save result
        result_path = result_dir / "result.json"
        with open(result_path, "w") as f:
            f.write(result.to_json(indent=2))

        # Save config if provided
        if config:
            config_path = result_dir / "config.json"
            with open(config_path, "w") as f:
                f.write(config.to_json(indent=2))

        # Save metadata
        metadata = {
            "benchmark_id": result.benchmark_id,
            "suite_type": result.suite_type.value,
            "model_type": result.model_type.value,
            "status": result.status.value,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "models_tested": result.models_tested,
            "total_runs": result.total_runs,
            "successful_runs": result.successful_runs,
            "failed_runs": result.failed_runs,
        }
        metadata_path = result_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return result_dir

    def load_result(self, benchmark_id: str, date: Optional[str] = None) -> Optional[SuiteResult]:
        """
        Load a benchmark result by ID.

        Args:
            benchmark_id: Benchmark ID to load
            date: Optional date string (YYYY-MM-DD) to narrow search

        Returns:
            SuiteResult if found, None otherwise
        """
        if date:
            result_path = self.base_dir / date / benchmark_id / "result.json"
            if result_path.exists():
                return self._load_result_from_file(result_path)
        else:
            # Search all dates
            for date_dir in sorted(self.base_dir.iterdir(), reverse=True):
                if date_dir.is_dir():
                    result_path = date_dir / benchmark_id / "result.json"
                    if result_path.exists():
                        return self._load_result_from_file(result_path)

        return None

    def load_config(self, benchmark_id: str, date: Optional[str] = None) -> Optional[BenchmarkConfig]:
        """
        Load a benchmark configuration by ID.

        Args:
            benchmark_id: Benchmark ID
            date: Optional date string (YYYY-MM-DD)

        Returns:
            BenchmarkConfig if found, None otherwise
        """
        if date:
            config_path = self.base_dir / date / benchmark_id / "config.json"
            if config_path.exists():
                return self._load_config_from_file(config_path)
        else:
            # Search all dates
            for date_dir in sorted(self.base_dir.iterdir(), reverse=True):
                if date_dir.is_dir():
                    config_path = date_dir / benchmark_id / "config.json"
                    if config_path.exists():
                        return self._load_config_from_file(config_path)

        return None

    def list_results(
        self,
        date: Optional[str] = None,
        suite_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[dict]:
        """
        List available benchmark results.

        Args:
            date: Filter by date (YYYY-MM-DD)
            suite_type: Filter by suite type
            limit: Maximum number of results to return

        Returns:
            List of result metadata dictionaries
        """
        results = []

        # Determine which dates to search
        if date:
            date_dirs = [self.base_dir / date] if (self.base_dir / date).exists() else []
        else:
            date_dirs = sorted(self.base_dir.iterdir(), reverse=True)

        # Search for results
        for date_dir in date_dirs:
            if not date_dir.is_dir():
                continue

            for benchmark_dir in date_dir.iterdir():
                if not benchmark_dir.is_dir():
                    continue

                metadata_path = benchmark_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    # Filter by suite type if specified
                    if suite_type and metadata.get("suite_type") != suite_type:
                        continue

                    metadata["date"] = date_dir.name
                    results.append(metadata)

                    # Check limit
                    if limit and len(results) >= limit:
                        return results

        return results

    def delete_result(self, benchmark_id: str, date: Optional[str] = None) -> bool:
        """
        Delete a benchmark result.

        Args:
            benchmark_id: Benchmark ID to delete
            date: Optional date string (YYYY-MM-DD)

        Returns:
            True if deleted, False if not found
        """
        import shutil

        if date:
            result_dir = self.base_dir / date / benchmark_id
            if result_dir.exists():
                shutil.rmtree(result_dir)
                return True
        else:
            # Search all dates
            for date_dir in self.base_dir.iterdir():
                if date_dir.is_dir():
                    result_dir = date_dir / benchmark_id
                    if result_dir.exists():
                        shutil.rmtree(result_dir)
                        return True

        return False

    def get_result_path(self, benchmark_id: str, date: Optional[str] = None) -> Optional[Path]:
        """
        Get the filesystem path for a benchmark result.

        Args:
            benchmark_id: Benchmark ID
            date: Optional date string (YYYY-MM-DD)

        Returns:
            Path to result directory if it exists, None otherwise
        """
        if date:
            result_dir = self.base_dir / date / benchmark_id
            return result_dir if result_dir.exists() else None
        else:
            # Search all dates
            for date_dir in sorted(self.base_dir.iterdir(), reverse=True):
                if date_dir.is_dir():
                    result_dir = date_dir / benchmark_id
                    if result_dir.exists():
                        return result_dir

        return None

    def _load_result_from_file(self, result_path: Path) -> SuiteResult:
        """Load SuiteResult from JSON file."""
        with open(result_path, "r") as f:
            data = json.load(f)

        # Reconstruct SuiteResult (simplified - would need full deserialization)
        # For now, return None - full implementation would reconstruct from dict
        # This is a placeholder for the actual deserialization logic
        return None  # TODO: Implement full deserialization

    def _load_config_from_file(self, config_path: Path) -> BenchmarkConfig:
        """Load BenchmarkConfig from JSON file."""
        with open(config_path, "r") as f:
            data = json.load(f)

        return BenchmarkConfig.from_dict(data)

    def __repr__(self) -> str:
        """String representation."""
        return f"ResultsManager(base_dir={self.base_dir})"
