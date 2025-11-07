"""
Pytest fixtures for benchmarking tests.

Provides reusable mock objects, sample configurations, and test data.
"""

import pytest
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any


@pytest.fixture
def sample_benchmark_config():
    """Return a valid BenchmarkConfig for testing."""
    from src.benchmarking.models.benchmark_config import BenchmarkConfig
    from src.benchmarking.models.metric_types import SuiteType, ModelType, ExecutionMode

    return BenchmarkConfig(
        benchmark_id=str(uuid4()),
        model_type=ModelType.LLM,
        suite_type=SuiteType.SPEED,
        models=["llama2:7b", "mistral:7b"],
        endpoints={"llama2:7b": ["text/chat"], "mistral:7b": ["text/chat"]},
        test_data={"prompts": ["Test prompt 1", "Test prompt 2"]},
        parameters={"temperature": 0.7, "max_tokens": 100},
        num_runs=5,
        num_warmup=2,
        execution_mode=ExecutionMode.FOREGROUND,
        export_formats=["json"],
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_llm_config():
    """Return a BenchmarkConfig specifically for LLM testing."""
    from src.benchmarking.models.benchmark_config import BenchmarkConfig
    from src.benchmarking.models.metric_types import SuiteType, ModelType, ExecutionMode

    return BenchmarkConfig(
        benchmark_id=str(uuid4()),
        model_type=ModelType.LLM,
        suite_type=SuiteType.COMPLETE,
        models=["llama2:7b"],
        endpoints={"llama2:7b": ["text/chat"]},
        test_data={"prompts": ["What is Python?", "Explain machine learning."]},
        parameters={"temperature": 0.7},
        num_runs=3,
        num_warmup=1,
        execution_mode=ExecutionMode.FOREGROUND,
        export_formats=["json", "csv"],
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_vlm_config():
    """Return a BenchmarkConfig specifically for VLM testing."""
    from src.benchmarking.models.benchmark_config import BenchmarkConfig
    from src.benchmarking.models.metric_types import SuiteType, ModelType, ExecutionMode

    return BenchmarkConfig(
        benchmark_id=str(uuid4()),
        model_type=ModelType.VLM,
        suite_type=SuiteType.RESOURCES,
        models=["llava:7b"],
        endpoints={"llava:7b": ["vision/analyze"]},
        test_data={
            "prompts": ["Describe this image"],
            "images": ["test_image_1.jpg", "test_image_2.jpg"]
        },
        parameters={"temperature": 0.5},
        num_runs=3,
        num_warmup=1,
        execution_mode=ExecutionMode.FOREGROUND,
        export_formats=["json"],
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_performance_metrics():
    """Return sample PerformanceMetric objects for testing."""
    from src.benchmarking.models.suite_result import PerformanceMetric
    from src.benchmarking.models.metric_types import MetricUnit

    return [
        PerformanceMetric(
            name="latency_ms",
            value=120.5,
            unit=MetricUnit.MILLISECONDS,
            run_number=1,
            model_id="llama2:7b",
            endpoint="text/chat",
            is_warmup=False
        ),
        PerformanceMetric(
            name="latency_ms",
            value=115.3,
            unit=MetricUnit.MILLISECONDS,
            run_number=2,
            model_id="llama2:7b",
            endpoint="text/chat",
            is_warmup=False
        ),
        PerformanceMetric(
            name="tokens_per_sec",
            value=45.2,
            unit=MetricUnit.TOKENS_PER_SECOND,
            run_number=1,
            model_id="llama2:7b",
            endpoint="text/chat",
            is_warmup=False
        ),
    ]


@pytest.fixture
def sample_aggregate_stats():
    """Return sample AggregateStats for testing."""
    from src.benchmarking.models.suite_result import AggregateStats
    from src.benchmarking.models.metric_types import MetricUnit

    return AggregateStats(
        metric_name="latency_ms",
        unit=MetricUnit.MILLISECONDS,
        count=5,
        mean=117.8,
        median=115.3,
        min=110.2,
        max=125.4,
        std_dev=5.6,
        variance=31.36,
        p25=113.0,
        p50=115.3,
        p75=120.5,
        p95=124.0,
        p99=125.2
    )


@pytest.fixture
def mock_llm_handler(mocker):
    """Return a mock LLMHandler for testing."""
    mock = mocker.Mock()
    mock.execute.return_value = "Mock LLM response text"
    mock.count_tokens.return_value = 25
    mock.validate_output.return_value = True
    return mock


@pytest.fixture
def mock_vlm_handler(mocker):
    """Return a mock VLMHandler for testing."""
    mock = mocker.Mock()
    mock.execute.return_value = "Mock VLM response text"
    mock.load_image.return_value = b"mock_image_data"
    mock.validate_output.return_value = True
    return mock


@pytest.fixture
def mock_system_monitor(mocker):
    """Return a mock system monitor for testing."""
    from src.benchmarking.models.suite_result import PerformanceMetric
    from src.benchmarking.models.metric_types import MetricUnit

    mock = mocker.Mock()
    mock.start.return_value = None
    mock.stop.return_value = {
        "cpu_percent": 45.2,
        "memory_mb": 1024.5,
        "gpu_percent": 80.3,
        "temperature_c": 65.0
    }
    return mock


@pytest.fixture
def sample_model_registry():
    """Return a sample model provider registry for testing."""
    return {
        "llama2:7b": {
            "provider": "ollama",
            "endpoints": ["text/chat"],
            "loaded": False
        },
        "mistral:7b": {
            "provider": "ollama",
            "endpoints": ["text/chat"],
            "loaded": False
        },
        "llava:7b": {
            "provider": "ollama",
            "endpoints": ["vision/analyze"],
            "loaded": False
        }
    }


@pytest.fixture
def sample_test_prompts():
    """Return sample LLM test prompts."""
    return [
        "What is artificial intelligence?",
        "Explain the concept of machine learning in simple terms.",
        "Write a haiku about coding.",
        "Translate 'Hello, world!' into French.",
        "Calculate 15 * 24."
    ]


@pytest.fixture
def sample_test_images(tmp_path):
    """Create sample test images and return their paths."""
    from PIL import Image
    import numpy as np

    images = []
    for i in range(3):
        # Create a simple colored image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / f"test_image_{i}.jpg"
        img.save(img_path)
        images.append(str(img_path))

    return images


@pytest.fixture
def sample_suite_result():
    """Return a complete SuiteResult for testing."""
    from src.benchmarking.models.suite_result import SuiteResult, AggregateStats
    from src.benchmarking.models.metric_types import SuiteType, ModelType, ResultStatus, MetricUnit

    return SuiteResult(
        suite_type=SuiteType.SPEED,
        benchmark_id=str(uuid4()),
        model_type=ModelType.LLM,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        status=ResultStatus.SUCCESS,
        models_tested=["llama2:7b"],
        endpoints_tested={"llama2:7b": ["text/chat"]},
        total_runs=5,
        successful_runs=5,
        failed_runs=0,
        aggregates_global={
            "latency_ms": AggregateStats(
                metric_name="latency_ms",
                unit=MetricUnit.MILLISECONDS,
                count=5,
                mean=120.0,
                median=118.0,
                min=115.0,
                max=125.0,
                std_dev=3.5
            )
        },
        config_snapshot={},
        duration_seconds=60.0
    )
