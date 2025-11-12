"""
Benchmark menu flow controller.

Orchestrates the 8-step benchmark configuration pipeline:
1. Model type selection (LLM/VLM)
2. Suite selection (Complete, Speed, Resources, Quality, Stress)
   2.5. If Complete suite → Sub-suite selection
3. Model selection (multi-select)
4. Endpoint selection (per model)
5. Number of prompts (1-50, suite-specific defaults)
6. Test data configuration (default/custom with manual entry)
7. Parameters configuration (warmup, temperature, max_tokens)
8. Execution mode (foreground/background)
9. Configuration review and confirmation (with save option)

Returns a validated BenchmarkConfig ready for execution.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from rich.console import Console

from src.benchmarking.models.benchmark_config import BenchmarkConfig
from src.benchmarking.models.metric_types import ModelType, SuiteType, ExecutionMode
from src.benchmarking.cli.benchmark_screens import (
    show_model_type_selection,
    show_suite_selection,
    show_complete_suite_selection,
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


def save_benchmark_config(state: Dict[str, Any], filepath: Optional[Path] = None) -> bool:
    """
    Save benchmark configuration to JSON file.

    Args:
        state: Configuration state dictionary
        filepath: Optional custom save path, defaults to .ekam/saved_configs/

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Default save location
        if filepath is None:
            save_dir = Path.cwd() / ".ekam" / "saved_configs"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suite_name = state.get("suite_type", "unknown").value if hasattr(state.get("suite_type"), "value") else "unknown"
            filepath = save_dir / f"benchmark_config_{suite_name}_{timestamp}.json"

        # Convert state to JSON-serializable format
        serializable_state = {}
        for key, value in state.items():
            if hasattr(value, "value"):  # Enum types
                serializable_state[key] = value.value
            elif isinstance(value, Path):
                serializable_state[key] = str(value)
            else:
                serializable_state[key] = value

        # Write to file
        with open(filepath, 'w') as f:
            json.dump(serializable_state, f, indent=2)

        console.print(f"\n[green]✓ Configuration saved to: {filepath}[/green]")
        return True

    except Exception as e:
        console.print(f"\n[red]✗ Failed to save configuration: {str(e)}[/red]")
        logger.error(f"Config save failed: {str(e)}")
        return False


def load_benchmark_config(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Load benchmark configuration from JSON file.

    Args:
        filepath: Path to saved config file

    Returns:
        State dictionary if loaded successfully, None otherwise
    """
    try:
        if not filepath.exists():
            console.print(f"\n[red]✗ Config file not found: {filepath}[/red]")
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert enum values back
        if "model_type" in data:
            data["model_type"] = ModelType(data["model_type"])
        if "suite_type" in data:
            data["suite_type"] = SuiteType(data["suite_type"])
        if "execution_mode" in data:
            data["execution_mode"] = ExecutionMode(data["execution_mode"])

        console.print(f"\n[green]✓ Configuration loaded from: {filepath}[/green]")
        return data

    except Exception as e:
        console.print(f"\n[red]✗ Failed to load configuration: {str(e)}[/red]")
        logger.error(f"Config load failed: {str(e)}")
        return None


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
            total_steps = 8

            while True:
                # Step 1: Model Type Selection
                if current_step == 0:
                    logger.info("Step 1/8: Model Type Selection")
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
                        logger.info(f"Step 1 completed: Model type = {value.value}")
                        current_step = 1

                # Step 2: Suite Selection
                elif current_step == 1:
                    logger.info("Step 2/8: Suite Selection")
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
                        logger.info(f"Step 2 completed: Suite = {value.value}")

                        # If Complete suite selected, show sub-suite selection
                        if value == SuiteType.COMPLETE:
                            logger.info("Complete suite selected, showing sub-suite selection")
                            current_value_suites = self.state.get("complete_suites")
                            suite_result = show_complete_suite_selection(current_value_suites)

                            if suite_result is None:
                                logger.info("User cancelled at Complete suite configuration")
                                return None

                            suites_value, suites_action = suite_result

                            if suites_action == "cancel":
                                logger.info("User cancelled at Complete suite configuration")
                                return None
                            elif suites_action == "back":
                                # Go back to suite selection
                                continue
                            else:  # next
                                self.state["complete_suites"] = suites_value
                                logger.info(f"Complete sub-suites: {', '.join(suites_value)}")

                        current_step = 2

                # Step 3: Model Selection (multi-select)
                elif current_step == 2:
                    logger.info("Step 3/8: Model Selection")
                    model_type = self.state["model_type"]
                    result = show_model_selection(model_type, self.session_manager)

                    if result is None:
                        logger.info("User cancelled at step 3")
                        return None

                    models, action = result

                    if action == "cancel":
                        logger.info("User cancelled at model selection")
                        return None
                    elif action == "back":
                        current_step = 1  # Go back to suite selection
                        continue
                    else:  # next
                        if not models:
                            logger.info("No models selected; staying on model selection")
                            current_step = 2
                            continue
                        self.state["models"] = models
                        logger.info(f"Step 3 completed: {len(models)} model(s) selected")
                        for model_id in models:
                            logger.info(f"  - {model_id}")
                        current_step = 3

                # Step 4: Endpoint Selection
                elif current_step == 3:
                    logger.info("Step 4/8: Endpoint Selection")
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
                        logger.info(f"Step 4 completed: Endpoints configured for {len(value)} model(s)")
                        for model_id, endpoints in value.items():
                            logger.info(f"  - {model_id}: {endpoints}")
                        current_step = 4

                # Step 5: Number of Prompts Configuration
                elif current_step == 4:
                    logger.info("Step 5/8: Number of Prompts Configuration")
                    suite_name = self.state["suite"].value
                    current_value = self.state.get("num_prompts")

                    from src.benchmarking.cli.benchmark_screens import show_num_prompts_config
                    result = show_num_prompts_config(suite_name, current_value)

                    if result is None:
                        logger.info("User cancelled at step 5")
                        return None

                    value, action = result

                    if action == "cancel":
                        logger.info("User cancelled at num prompts configuration")
                        return None
                    elif action == "back":
                        current_step = 3  # Go back to endpoint selection
                    else:  # next
                        self.state["num_prompts"] = value
                        logger.info(f"Step 5 completed: Number of prompts = {value}")
                        current_step = 5

                # Step 6: Test Data Selection
                elif current_step == 5:
                    logger.info("Step 6/8: Test Data Selection")
                    model_type = self.state["model_type"]
                    num_prompts = self.state["num_prompts"]
                    suite_name = self.state["suite"].value
                    endpoints = self.state.get("endpoints")

                    # Extract endpoint for VLM
                    selected_endpoint = None
                    if model_type == ModelType.VLM and endpoints:
                        first_model = next(iter(endpoints.values()))
                        if first_model:
                            selected_endpoint = first_model[0]

                    current_value = self.state.get("test_data")
                    result = show_test_data_selection(model_type, num_prompts, suite_name, selected_endpoint, current_value)

                    if result is None:
                        logger.info("User cancelled at step 6")
                        return None

                    value, action = result

                    if action == "cancel":
                        logger.info("User cancelled at test data selection")
                        return None
                    elif action == "back":
                        current_step = 4  # Go back to num_prompts
                    else:  # next
                        self.state["test_data"] = value
                        source = value.get('source', 'unknown')
                        logger.info(f"Step 6 completed: Test data source = {source}")
                        if 'pairs' in value:
                            logger.info(f"  - {len(value['pairs'])} image-prompt pairs configured")
                        elif 'prompts' in value:
                            logger.info(f"  - {len(value['prompts'])} prompts configured")
                        current_step = 6

                # Step 7: Parameters Configuration
                elif current_step == 6:
                    logger.info("Step 7/8: Parameters Configuration")
                    current_value = self.state.get("parameters")
                    result = show_parameters_config(current_value)

                    if result is None:
                        logger.info("User cancelled at step 7")
                        return None

                    value, action = result

                    if action == "cancel":
                        logger.info("User cancelled at parameters configuration")
                        return None
                    elif action == "back":
                        current_step = 5  # Go back to test data
                    else:  # next
                        self.state["parameters"] = value
                        logger.info(f"Step 7 completed: Parameters configured")
                        logger.info(f"  - Temperature: {value.get('temperature', 'default')}")
                        logger.info(f"  - Max tokens: {value.get('max_tokens', 'default')}")
                        logger.info(f"  - Num runs: {value.get('num_runs', 'default')}")
                        logger.info(f"  - Num warmup: {value.get('num_warmup', 'default')}")
                        current_step = 7

                # Step 8: Execution Mode (skipped, default to Foreground)
                elif current_step == 7:
                    from src.benchmarking.models.metric_types import ExecutionMode
                    logger.info("Step 8/8: Execution Mode (auto-configured to Foreground)")
                    self.state["execution_mode"] = ExecutionMode.FOREGROUND
                    current_step = 8  # Proceed to review

                # Final: Configuration Review
                elif current_step == 8:
                    logger.info("Final Step: Configuration Review")
                    config_summary = self._build_config_summary()
                    review_action = show_config_review(config_summary, self.state)

                    if review_action == "cancel":
                        logger.info("User cancelled at configuration review")
                        console.print("\n[yellow]Configuration cancelled.[/yellow]")
                        return None

                    elif review_action == "edit":
                        logger.info("User chose to edit configuration from review screen")
                        current_step = 7  # Go back to execution mode (Step 8)
                        continue

                    elif review_action == "save":
                        # Save was handled in show_config_review, just continue loop
                        logger.info("User saved configuration template")
                        continue

                    elif review_action == "confirm":
                        # Build and validate BenchmarkConfig
                        logger.info("User confirmed configuration - building BenchmarkConfig")
                        try:
                            config = self._build_config()
                            logger.info("=" * 80)
                            logger.info("BENCHMARK CONFIGURATION CONFIRMED")
                            logger.info("=" * 80)
                            logger.info(f"Suite: {self.state['suite'].value}")
                            logger.info(f"Models: {len(self.state['models'])}")
                            logger.info(f"Num prompts: {self.state['num_prompts']}")
                            logger.info(f"Num runs: {self.state['parameters'].get('num_runs', 'default')}")
                            logger.info(f"Execution mode: {self.state['execution_mode'].value}")
                            logger.info("=" * 80)
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
        # Ensure models is a list (defensive coding)
        models = self.state["models"]
        if not isinstance(models, list):
            logger.warning(f"models in summary was not a list (type={type(models)}), converting to list")
            models = [models] if models else []
            self.state["models"] = models  # Fix the state as well

        summary = {
            "model_type": self.state["model_type"].value.upper(),
            "suite": self.state["suite"].display_name(),
            "models": models,
            "num_runs": self.state["parameters"].get("num_runs", 0),
            "num_warmup": self.state["parameters"].get("num_warmup", 0),
            "execution_mode": self.state["execution_mode"].value.title(),
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
        num_models = len(models)
        estimated_duration = base_duration * num_models

        summary["estimated_duration"] = estimated_duration
        summary["export_formats"] = ["json", "csv"]

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

        # Build generation parameters (common across providers)
        generation_params = {
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "top_k": params.get("top_k", 50),
            "max_tokens": params.get("max_tokens", 512),
            "n_ctx": params.get("n_ctx", 4096),
            # Include run controls here so suite-specific builders can see them
            "num_runs": params.get("num_runs", 5),
            "num_warmup": params.get("num_warmup", 1),
        }

        # Add Complete suite configuration if applicable
        if self.state["suite"] == SuiteType.COMPLETE and "complete_suites" in self.state:
            generation_params["complete"] = {
                "suites_to_run": self.state["complete_suites"],
                "suite_execution_order": self.state["complete_suites"],  # Use same order
                "stop_on_suite_failure": False,
            }

        # Ensure models is a list (defensive coding)
        models = self.state["models"]
        if not isinstance(models, list):
            logger.warning(f"models was not a list (type={type(models)}), converting to list")
            models = [models] if models else []

        # Build config dictionary
        config_dict = {
            "suite_type": self.state["suite"],
            "model_type": self.state["model_type"],
            "models": models,
            "endpoints": self.state["endpoints"],
            "test_data": self.state["test_data"],
            "parameters": generation_params,
            "num_runs": params.get("num_runs", 5),
            "num_warmup": params.get("num_warmup", 1),
            "execution_mode": self.state["execution_mode"],
            "export_formats": ["json", "csv"],
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
