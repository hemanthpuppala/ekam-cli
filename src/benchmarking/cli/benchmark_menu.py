"""
Benchmark menu flow controller.

Orchestrates the 7-step benchmark configuration pipeline:
1. Model type selection (LLM/VLM)
2. Suite selection (Complete, Speed, Resources, Quality, Stress)
3. Model selection (multi-select)
4. Endpoint selection (per model)
5. Test data configuration (default/custom)
6. Parameters configuration (runs, warmup, etc.)
7. Execution mode (foreground/background)
8. Configuration review and confirmation

Returns a validated BenchmarkConfig ready for execution.
"""

from typing import Optional, Dict, Any
from pathlib import Path

from rich.console import Console

from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.metric_types import ModelType, SuiteType, ExecutionMode
from src.benchmarking.cli.benchmark_screens import (
    show_model_type_selection,
    show_suite_selection,
    show_model_selection,
    show_endpoint_selection,
    show_test_data_selection,
    show_parameters_config,
    show_execution_mode,
    show_config_review,
    show_benchmark_error
)
from loguru import logger


console = Console()


class BenchmarkMenuFlow:
    """
    Manages the complete benchmark configuration flow.

    Handles navigation between configuration screens, state persistence,
    validation, and produces a validated BenchmarkConfig.
    """

    def __init__(self, session_manager):
        """
        Initialize menu flow.

        Args:
            session_manager: SessionManager instance for querying available models
        """
        self.session_manager = session_manager
        self.state: Dict[str, Any] = {}

    def run(self) -> Optional[BenchmarkConfig]:
        """
        Execute the complete benchmark configuration flow with backward navigation.

        Guides user through 7 configuration screens with ability to go back and edit,
        validates input at each step, and produces a BenchmarkConfig object.

        Returns:
            BenchmarkConfig if user completes flow, None if cancelled
        """
        try:
            logger.info("Starting benchmark configuration flow with backward navigation")

            # State tracking for backward navigation
            current_step = 0
            total_steps = 7

            while True:
                # Step 1: Model Type Selection
                if current_step == 0:
                    logger.debug("Step 1/7: Model Type Selection")
                    current_value = self.state.get("model_type")
                    result = show_model_type_selection(current_value)

                    if result is None:
                        logger.info("User cancelled at step 1")
                        return None

                    value, action = result

                    if action == "cancel":
                        logger.info("User cancelled at model type selection")
                        return None
                    elif action == "back":
                        # Can't go back from step 1
                        continue
                    else:  # next
                        self.state["model_type"] = value
                        logger.debug(f"Model type selected: {value.value}")
                        current_step = 1

                # Step 2: Suite Selection
                elif current_step == 1:
                    logger.debug("Step 2/7: Suite Selection")
                    current_value = self.state.get("suite")
                    result = show_suite_selection(current_value)

                    if result is None:
                        logger.info("User cancelled at step 2")
                        return None

                    value, action = result

                    if action == "cancel":
                        logger.info("User cancelled at suite selection")
                        return None
                    elif action == "back":
                        current_step = 0  # Go back to model type
                    else:  # next
                        self.state["suite"] = value
                        logger.debug(f"Suite selected: {value.value}")
                        current_step = 2

                # Step 3: Model Selection (multi-select - TODO: refactor later)
                elif current_step == 2:
                    logger.debug("Step 3/7: Model Selection")
                    model_type = self.state["model_type"]
                    models = show_model_selection(model_type, self.session_manager)

                    if models is None or not models:
                        # For now, treat cancel as going back
                        logger.debug("Going back from model selection")
                        current_step = 1
                        continue

                    self.state["models"] = models
                    logger.debug(f"Models selected: {len(models)} model(s)")
                    current_step = 3

                # Step 4: Endpoint Selection
                elif current_step == 3:
                    logger.debug("Step 4/7: Endpoint Selection")
                    models = self.state["models"]
                    model_type = self.state["model_type"]
                    current_value = self.state.get("endpoints")

                    result = show_endpoint_selection(models, model_type, current_value)

                    if result is None:
                        logger.info("User cancelled at step 4")
                        return None

                    value, action = result

                    if action == "cancel":
                        logger.info("User cancelled at endpoint selection")
                        return None
                    elif action == "back":
                        current_step = 2  # Go back to model selection
                    else:  # next
                        self.state["endpoints"] = value
                        logger.debug(f"Endpoints configured for {len(value)} model(s)")
                        current_step = 4

                # Step 5: Test Data Selection
                elif current_step == 4:
                    logger.debug("Step 5/7: Test Data Selection")
                    model_type = self.state["model_type"]
                    endpoints = self.state.get("endpoints")

                    # Extract endpoint for VLM
                    selected_endpoint = None
                    if model_type == ModelType.VLM and endpoints:
                        first_model = next(iter(endpoints.values()))
                        if first_model:
                            selected_endpoint = first_model[0]

                    current_value = self.state.get("test_data")
                    result = show_test_data_selection(model_type, selected_endpoint, current_value)

                    if result is None:
                        logger.info("User cancelled at step 5")
                        return None

                    value, action = result

                    if action == "cancel":
                        logger.info("User cancelled at test data selection")
                        return None
                    elif action == "back":
                        current_step = 3  # Go back to endpoint selection
                    else:  # next
                        self.state["test_data"] = value
                        logger.debug(f"Test data configured: {value.get('source', 'unknown')}")
                        current_step = 5

                # Step 6: Parameters Configuration
                elif current_step == 5:
                    logger.debug("Step 6/7: Parameters Configuration")
                    current_value = self.state.get("parameters")
                    result = show_parameters_config(current_value)

                    if result is None:
                        logger.info("User cancelled at step 6")
                        return None

                    value, action = result

                    if action == "cancel":
                        logger.info("User cancelled at parameters configuration")
                        return None
                    elif action == "back":
                        current_step = 4  # Go back to test data
                    else:  # next
                        self.state["parameters"] = value
                        logger.debug(f"Parameters configured: {value.get('num_runs', 0)} runs")
                        current_step = 6

                # Step 7: Execution Mode
                elif current_step == 6:
                    logger.debug("Step 7/7: Execution Mode")
                    current_value = self.state.get("execution_mode")
                    result = show_execution_mode(current_value)

                    if result is None:
                        logger.info("User cancelled at step 7")
                        return None

                    value, action = result

                    if action == "cancel":
                        logger.info("User cancelled at execution mode selection")
                        return None
                    elif action == "back":
                        current_step = 5  # Go back to parameters
                    else:  # next
                        self.state["execution_mode"] = value
                        logger.debug(f"Execution mode: {value.value}")
                        current_step = 7  # Proceed to review

                # Step 8: Configuration Review
                elif current_step == 7:
                    logger.debug("Step 8: Configuration Review")
                    config_summary = self._build_config_summary()
                    review_action = show_config_review(config_summary)

                    if review_action == "cancel":
                        logger.info("User cancelled at configuration review")
                        console.print("\n[yellow]Configuration cancelled.[/yellow]")
                        return None

                    elif review_action == "edit":
                        logger.info("User chose to edit configuration, going back to Step 7")
                        current_step = 6  # Go back to execution mode (Step 7)
                        continue

                    elif review_action == "confirm":
                        # Build and validate BenchmarkConfig
                        try:
                            config = self._build_config()
                            logger.info("BenchmarkConfig successfully created and validated")
                            return config

                        except Exception as e:
                            logger.error(f"Failed to build BenchmarkConfig: {str(e)}")
                            show_benchmark_error(f"Configuration validation failed: {str(e)}")
                            return None

                    else:
                        # Unknown action, cancel
                        logger.warning(f"Unknown review action: {review_action}")
                        return None

        except KeyboardInterrupt:
            logger.info("User interrupted benchmark configuration flow")
            console.print("\n[yellow]Configuration cancelled by user.[/yellow]")
            return None

        except Exception as e:
            logger.error(f"Unexpected error in benchmark flow: {str(e)}", exc_info=True)
            show_benchmark_error(f"Unexpected error: {str(e)}")
            return None

    def _build_config_summary(self) -> Dict[str, Any]:
        """
        Build configuration summary for review screen.

        Returns:
            Dictionary with human-readable configuration summary
        """
        summary = {
            "model_type": self.state["model_type"].value.upper(),
            "suite": self.state["suite"].display_name(),
            "models": self.state["models"],
            "num_runs": self.state["parameters"].get("num_runs", 0),
            "num_warmup": self.state["parameters"].get("num_warmup", 0),
            "execution_mode": self.state["execution_mode"].value.title(),
            "export_formats": self.state["parameters"].get("export_formats", []),
        }

        # Add estimated duration
        suite_durations = {
            SuiteType.COMPLETE: 60,
            SuiteType.SPEED: 15,
            SuiteType.RESOURCES: 15,
            SuiteType.QUALITY: 20,
            SuiteType.STRESS: 45
        }

        base_duration = suite_durations.get(self.state["suite"], 30)
        num_models = len(self.state["models"])
        estimated_duration = base_duration * num_models

        summary["estimated_duration"] = estimated_duration

        return summary

    def _build_config(self) -> BenchmarkConfig:
        """
        Build validated BenchmarkConfig from collected state.

        Returns:
            Validated BenchmarkConfig instance

        Raises:
            ValueError: If configuration is invalid
        """
        # Extract parameters
        params = self.state["parameters"]

        # Build generation parameters
        generation_params = {
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 512),
        }

        # Build config dictionary
        config_dict = {
            "suite_type": self.state["suite"],
            "model_type": self.state["model_type"],
            "models": self.state["models"],
            "endpoints": self.state["endpoints"],
            "test_data": self.state["test_data"],
            "parameters": generation_params,
            "num_runs": params.get("num_runs", 5),
            "num_warmup": params.get("num_warmup", 1),
            "execution_mode": self.state["execution_mode"],
            "export_formats": params.get("export_formats", ["json", "csv"]),
            "output_dir": Path("results"),
        }

        # Validate and create BenchmarkConfig
        try:
            config = BenchmarkConfig(**config_dict)
            logger.debug("BenchmarkConfig validation successful")
            return config

        except Exception as e:
            logger.error(f"BenchmarkConfig validation failed: {str(e)}")
            raise ValueError(f"Invalid configuration: {str(e)}")

    def reset(self):
        """Reset flow state (useful for testing or retry)."""
        self.state.clear()
        logger.debug("Menu flow state reset")

    def __repr__(self) -> str:
        """String representation."""
        return f"BenchmarkMenuFlow(state_keys={list(self.state.keys())})"


def run_benchmark_configuration(session_manager) -> Optional[BenchmarkConfig]:
    """
    Convenience function to run benchmark configuration flow.

    Args:
        session_manager: SessionManager instance

    Returns:
        BenchmarkConfig if completed, None if cancelled
    """
    flow = BenchmarkMenuFlow(session_manager)
    return flow.run()
