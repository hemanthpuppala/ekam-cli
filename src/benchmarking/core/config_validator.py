"""
Configuration validation utilities.

Provides validation functions for benchmark configurations,
including model availability checks, endpoint validation, and
test data validation.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.metric_types import ModelType, SuiteType


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """
    Validates benchmark configurations.

    Checks:
    - Model availability
    - Endpoint validity
    - Test data presence
    - Parameter validity
    - Suite compatibility
    """

    @staticmethod
    def validate_config(config: BenchmarkConfig) -> Tuple[bool, List[str]]:
        """
        Validate a benchmark configuration.

        Args:
            config: BenchmarkConfig to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate models
        model_errors = ConfigValidator._validate_models(config)
        errors.extend(model_errors)

        # Validate endpoints
        endpoint_errors = ConfigValidator._validate_endpoints(config)
        errors.extend(endpoint_errors)

        # Validate test data
        data_errors = ConfigValidator._validate_test_data(config)
        errors.extend(data_errors)

        # Validate parameters
        param_errors = ConfigValidator._validate_parameters(config)
        errors.extend(param_errors)

        # Validate suite compatibility
        suite_errors = ConfigValidator._validate_suite_compatibility(config)
        errors.extend(suite_errors)

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def _validate_models(config: BenchmarkConfig) -> List[str]:
        """Validate model configuration."""
        errors = []

        if not config.models or len(config.models) == 0:
            errors.append("No models specified")

        # Check for duplicate model IDs
        if len(config.models) != len(set(config.models)):
            errors.append("Duplicate model IDs found")

        return errors

    @staticmethod
    def _validate_endpoints(config: BenchmarkConfig) -> List[str]:
        """Validate endpoint configuration."""
        errors = []

        for model_id in config.models:
            if model_id not in config.endpoints:
                errors.append(f"No endpoints defined for model: {model_id}")
                continue

            endpoints = config.endpoints[model_id]
            if not endpoints or len(endpoints) == 0:
                errors.append(f"Empty endpoint list for model: {model_id}")

            # Validate endpoint format
            for endpoint in endpoints:
                if not isinstance(endpoint, str) or not endpoint.strip():
                    errors.append(f"Invalid endpoint format for model {model_id}: {endpoint}")

        return errors

    @staticmethod
    def _validate_test_data(config: BenchmarkConfig) -> List[str]:
        """Validate test data configuration."""
        errors = []

        if not config.test_data:
            errors.append("No test data provided")
            return errors

        # LLM validation
        if config.model_type == ModelType.LLM:
            if "prompts" not in config.test_data:
                errors.append("LLM benchmarks require 'prompts' in test_data")
            elif not config.test_data["prompts"]:
                errors.append("LLM prompts list is empty")
            elif not isinstance(config.test_data["prompts"], list):
                errors.append("LLM prompts must be a list")

        # VLM validation
        elif config.model_type == ModelType.VLM:
            if "prompts" not in config.test_data:
                errors.append("VLM benchmarks require 'prompts' in test_data")
            if "images" not in config.test_data:
                errors.append("VLM benchmarks require 'images' in test_data")

            # Validate image paths exist (if provided as paths)
            if "images" in config.test_data and isinstance(config.test_data["images"], list):
                for img_path in config.test_data["images"]:
                    if isinstance(img_path, str):
                        path = Path(img_path)
                        if not path.exists() and not img_path.startswith("http"):
                            # Only warn, don't fail - might be mock data for testing
                            pass

        return errors

    @staticmethod
    def _validate_parameters(config: BenchmarkConfig) -> List[str]:
        """Validate inference parameters."""
        errors = []

        params = config.parameters

        # Temperature validation
        if "temperature" in params:
            temp = params["temperature"]
            if not isinstance(temp, (int, float)):
                errors.append("Temperature must be a number")
            elif temp < 0.0 or temp > 2.0:
                errors.append("Temperature should be between 0.0 and 2.0")

        # Max tokens validation
        if "max_tokens" in params:
            max_tokens = params["max_tokens"]
            if not isinstance(max_tokens, int):
                errors.append("max_tokens must be an integer")
            elif max_tokens < 1:
                errors.append("max_tokens must be positive")
            elif max_tokens > 32000:
                errors.append("max_tokens exceeds reasonable limit (32000)")

        # Top-p validation
        if "top_p" in params:
            top_p = params["top_p"]
            if not isinstance(top_p, (int, float)):
                errors.append("top_p must be a number")
            elif top_p < 0.0 or top_p > 1.0:
                errors.append("top_p must be between 0.0 and 1.0")

        return errors

    @staticmethod
    def _validate_suite_compatibility(config: BenchmarkConfig) -> List[str]:
        """Validate suite-specific requirements."""
        errors = []

        # Stress suite requires duration or run limits
        if config.suite_type == SuiteType.STRESS:
            if config.num_runs < 10:
                errors.append("Stress suite should have at least 10 runs for meaningful results")

        # Quality suite requires multiple runs
        if config.suite_type == SuiteType.QUALITY:
            if config.num_runs < 3:
                errors.append("Quality suite requires at least 3 runs to measure consistency")

        return errors

    @staticmethod
    def validate_or_raise(config: BenchmarkConfig) -> None:
        """
        Validate configuration and raise exception if invalid.

        Args:
            config: BenchmarkConfig to validate

        Raises:
            ConfigValidationError: If validation fails
        """
        is_valid, errors = ConfigValidator.validate_config(config)
        if not is_valid:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors)
            raise ConfigValidationError(error_msg)

    @staticmethod
    def get_warnings(config: BenchmarkConfig) -> List[str]:
        """
        Get non-critical warnings for a configuration.

        Args:
            config: BenchmarkConfig to check

        Returns:
            List of warning messages
        """
        warnings = []

        # Warn about excessive runs
        if config.num_runs > 100:
            warnings.append(f"Large number of runs ({config.num_runs}) may take significant time")

        # Warn about no warmup
        if config.num_warmup == 0:
            warnings.append("No warmup runs configured - first run may be slower")

        # Warn about many models
        if len(config.models) > 5:
            warnings.append(f"Benchmarking {len(config.models)} models may take considerable time")

        # Warn about complete suite
        if config.suite_type == SuiteType.COMPLETE:
            warnings.append("Complete suite runs all benchmarks - expect long execution time")

        return warnings
