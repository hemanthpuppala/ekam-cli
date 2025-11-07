"""
Unit tests for ConfigValidator.

Tests configuration validation logic.
"""

import pytest

from src.benchmarking.core.config_validator import ConfigValidator, ConfigValidationError
from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.metric_types import ModelType, SuiteType, ExecutionMode


class TestConfigValidator:
    """Test suite for ConfigValidator."""

    def test_valid_config(self):
        """Test validation of a valid configuration."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=["llama2:7b"],
            endpoints={"llama2:7b": ["text/chat"]},
            test_data={"prompts": ["Test prompt"]},
            parameters={"temperature": 0.7},
            num_runs=5,
            num_warmup=2
        )

        is_valid, errors = ConfigValidator.validate_config(config)

        assert is_valid
        assert len(errors) == 0

    def test_empty_models_list(self):
        """Test validation fails with empty models list."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=[],  # Empty!
            endpoints={},
            test_data={"prompts": ["Test"]},
            parameters={}
        )

        is_valid, errors = ConfigValidator.validate_config(config)

        assert not is_valid
        assert any("models" in error.lower() for error in errors)

    def test_missing_endpoints(self):
        """Test validation fails when endpoint missing for model."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=["llama2:7b"],
            endpoints={},  # Missing endpoint!
            test_data={"prompts": ["Test"]},
            parameters={}
        )

        is_valid, errors = ConfigValidator.validate_config(config)

        assert not is_valid
        assert any("endpoint" in error.lower() for error in errors)

    def test_empty_endpoints_list(self):
        """Test validation fails with empty endpoints list."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=["llama2:7b"],
            endpoints={"llama2:7b": []},  # Empty list!
            test_data={"prompts": ["Test"]},
            parameters={}
        )

        is_valid, errors = ConfigValidator.validate_config(config)

        assert not is_valid
        assert any("endpoint" in error.lower() for error in errors)

    def test_missing_llm_prompts(self):
        """Test validation fails when LLM config missing prompts."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=["llama2:7b"],
            endpoints={"llama2:7b": ["text/chat"]},
            test_data={},  # Missing prompts!
            parameters={}
        )

        is_valid, errors = ConfigValidator.validate_config(config)

        assert not is_valid
        assert any("prompt" in error.lower() for error in errors)

    def test_missing_vlm_images(self):
        """Test validation fails when VLM config missing images."""
        config = BenchmarkConfig(
            model_type=ModelType.VLM,
            suite_type=SuiteType.SPEED,
            models=["llava:7b"],
            endpoints={"llava:7b": ["vision/analyze"]},
            test_data={"prompts": ["Describe this image"]},  # Missing images!
            parameters={}
        )

        is_valid, errors = ConfigValidator.validate_config(config)

        assert not is_valid
        assert any("image" in error.lower() for error in errors)

    def test_invalid_temperature(self):
        """Test validation fails with invalid temperature."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=["llama2:7b"],
            endpoints={"llama2:7b": ["text/chat"]},
            test_data={"prompts": ["Test"]},
            parameters={"temperature": 5.0}  # Too high!
        )

        is_valid, errors = ConfigValidator.validate_config(config)

        assert not is_valid
        assert any("temperature" in error.lower() for error in errors)

    def test_invalid_max_tokens(self):
        """Test validation fails with invalid max_tokens."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=["llama2:7b"],
            endpoints={"llama2:7b": ["text/chat"]},
            test_data={"prompts": ["Test"]},
            parameters={"max_tokens": -1}  # Negative!
        )

        is_valid, errors = ConfigValidator.validate_config(config)

        assert not is_valid
        assert any("max_tokens" in error.lower() for error in errors)

    def test_quality_suite_requires_multiple_runs(self):
        """Test quality suite needs at least 3 runs."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.QUALITY,
            models=["llama2:7b"],
            endpoints={"llama2:7b": ["text/chat"]},
            test_data={"prompts": ["Test"]},
            parameters={},
            num_runs=2  # Too few!
        )

        is_valid, errors = ConfigValidator.validate_config(config)

        assert not is_valid
        assert any("quality" in error.lower() and "runs" in error.lower() for error in errors)

    def test_stress_suite_requires_many_runs(self):
        """Test stress suite needs at least 10 runs."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.STRESS,
            models=["llama2:7b"],
            endpoints={"llama2:7b": ["text/chat"]},
            test_data={"prompts": ["Test"]},
            parameters={},
            num_runs=5  # Too few!
        )

        is_valid, errors = ConfigValidator.validate_config(config)

        assert not is_valid
        assert any("stress" in error.lower() for error in errors)

    def test_validate_or_raise_success(self):
        """Test validate_or_raise with valid config."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=["llama2:7b"],
            endpoints={"llama2:7b": ["text/chat"]},
            test_data={"prompts": ["Test"]},
            parameters={}
        )

        # Should not raise
        ConfigValidator.validate_or_raise(config)

    def test_validate_or_raise_failure(self):
        """Test validate_or_raise with invalid config."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=[],  # Invalid!
            endpoints={},
            test_data={"prompts": ["Test"]},
            parameters={}
        )

        with pytest.raises(ConfigValidationError):
            ConfigValidator.validate_or_raise(config)

    def test_get_warnings_excessive_runs(self):
        """Test warnings for excessive runs."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=["llama2:7b"],
            endpoints={"llama2:7b": ["text/chat"]},
            test_data={"prompts": ["Test"]},
            parameters={},
            num_runs=150  # Many runs
        )

        warnings = ConfigValidator.get_warnings(config)

        assert len(warnings) > 0
        assert any("runs" in warning.lower() for warning in warnings)

    def test_get_warnings_no_warmup(self):
        """Test warning for no warmup runs."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=["llama2:7b"],
            endpoints={"llama2:7b": ["text/chat"]},
            test_data={"prompts": ["Test"]},
            parameters={},
            num_warmup=0  # No warmup
        )

        warnings = ConfigValidator.get_warnings(config)

        assert any("warmup" in warning.lower() for warning in warnings)

    def test_get_warnings_many_models(self):
        """Test warning for many models."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.SPEED,
            models=["model1", "model2", "model3", "model4", "model5", "model6"],
            endpoints={
                f"model{i}": ["text/chat"] for i in range(1, 7)
            },
            test_data={"prompts": ["Test"]},
            parameters={}
        )

        warnings = ConfigValidator.get_warnings(config)

        assert any("models" in warning.lower() for warning in warnings)

    def test_get_warnings_complete_suite(self):
        """Test warning for complete suite."""
        config = BenchmarkConfig(
            model_type=ModelType.LLM,
            suite_type=SuiteType.COMPLETE,
            models=["llama2:7b"],
            endpoints={"llama2:7b": ["text/chat"]},
            test_data={"prompts": ["Test"]},
            parameters={}
        )

        warnings = ConfigValidator.get_warnings(config)

        assert any("complete" in warning.lower() for warning in warnings)
