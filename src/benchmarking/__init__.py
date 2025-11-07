"""
Benchmarking module for LLMs and VLMs.

This module provides comprehensive benchmarking capabilities for evaluating
the performance, resource efficiency, quality, and stress characteristics of
language models and vision language models.

Key Components:
- Core: Metrics collection, system monitoring, orchestration
- Suites: Speed, Resources, Quality, Stress, Complete system analysis
- Handlers: LLM/VLM inference execution and endpoint management
- Reporters: Multi-format result export (JSON, CSV, HTML)
- CLI: User interface and menu flows (integrated with existing TUI)
- Models: Pydantic data structures for configuration and results

Quick Start:
    >>> from benchmarking import BenchmarkRunner, BenchmarkConfig, ModelType, SuiteType
    >>> config = BenchmarkConfig(
    ...     model_type=ModelType.LLM,
    ...     suite_type=SuiteType.SPEED,
    ...     models=["llama2:7b"],
    ...     endpoints={"llama2:7b": ["text/chat"]},
    ...     test_data={"prompts": ["Test prompt"]},
    ...     num_runs=5
    ... )
    >>> runner = BenchmarkRunner()
    >>> result = runner.run(config)
"""

# Core orchestration
from src.benchmarking.core.benchmark_runner import BenchmarkRunner
from src.benchmarking.core.results_manager import ResultsManager

# Models
from src.benchmarking.models.metric_types import (
    SuiteType,
    ModelType,
    ExecutionMode,
    ResultStatus,
    EndpointType,
    MetricUnit,
)
from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.suite_result import (
    SuiteResult,
    PerformanceMetric,
    AggregateStats,
)

# Suites
from src.benchmarking.suites.suite_speed import SpeedSuite
from src.benchmarking.suites.suite_resources import ResourcesSuite
from src.benchmarking.suites.suite_complete import CompleteSuite

# Reporters
from src.benchmarking.reporters.json_reporter import JSONReporter
from src.benchmarking.reporters.csv_reporter import CSVReporter
from src.benchmarking.reporters.console_formatter import ConsoleFormatter

__all__ = [
    # Core
    "BenchmarkRunner",
    "ResultsManager",
    # Enums
    "SuiteType",
    "ModelType",
    "ExecutionMode",
    "ResultStatus",
    "EndpointType",
    "MetricUnit",
    # Models
    "BenchmarkConfig",
    "SuiteResult",
    "PerformanceMetric",
    "AggregateStats",
    # Suites
    "SpeedSuite",
    "ResourcesSuite",
    "CompleteSuite",
    # Reporters
    "JSONReporter",
    "CSVReporter",
    "ConsoleFormatter",
]

__version__ = "0.1.0"
