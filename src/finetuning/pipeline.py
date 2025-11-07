"""High-level finetuning pipeline for convenient workflow execution."""

import json
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from .config import FinetuneConfig, FinetuneMethod
from .orchestrator import FinetuneOrchestrator, PipelineStep


class FinetuningPipeline:
    """High-level API for finetuning pipeline execution.

    Provides convenience methods for:
    - Configuration management and validation
    - Complete training workflows
    - Progress tracking and monitoring
    - Checkpoint management
    - Background execution support
    """

    def __init__(self, config: FinetuneConfig):
        """Initialize finetuning pipeline.

        Args:
            config: Finetuning configuration
        """
        self.config = config
        self.orchestrator = FinetuneOrchestrator(config)
        self.execution_log = []

        logger.info(f"Pipeline initialized: {config.run_name}")

    def register_monitor(self, callback: Callable) -> None:
        """Register progress monitoring callback.

        Args:
            callback: Function called with status updates
        """
        self.orchestrator.register_callback("progress_update", callback)
        self.orchestrator.register_callback("step_complete", callback)
        self.orchestrator.register_callback("error", callback)

    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate configuration completeness.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            self.config.validate_method_config()
        except ValueError as e:
            errors.append(str(e))

        if not self.config.data.dataset_path and not self.config.data.dataset_name:
            errors.append("Dataset path or name required")

        if not self.config.model_id:
            errors.append("Model ID required")

        return len(errors) == 0, errors

    def prepare(self) -> bool:
        """Prepare for training without starting it.

        Returns:
            True if preparation successful
        """
        try:
            self.orchestrator.prepare_training()
            self._log_execution("prepare", success=True)
            return True
        except Exception as e:
            self._log_execution("prepare", success=False, error=str(e))
            raise

    def execute_full_workflow(self) -> dict[str, Any]:
        """Execute complete finetuning workflow.

        Workflow:
        1. Validate configuration
        2. Prepare model and dataset
        3. Setup training
        4. Execute training
        5. Save model/adapters

        Returns:
            Dictionary with workflow results

        Raises:
            Exception: If any workflow step fails
        """
        try:
            self._log_execution("workflow_start", metadata={"method": self.config.method})

            # Validate
            is_valid, errors = self.validate_config()
            if not is_valid:
                raise ValueError(f"Config validation failed: {errors}")

            # Prepare
            logger.info("Preparing training...")
            self.prepare()

            # Train
            logger.info("Starting training...")
            metrics = self.orchestrator.start_training()

            # Save
            logger.info("Saving model...")
            save_path = self.orchestrator.save_checkpoint()

            result = {
                "success": True,
                "run_name": self.config.run_name,
                "method": self.config.method,
                "metrics": metrics,
                "save_path": str(save_path),
            }

            self._log_execution("workflow_complete", success=True, result=result)
            return result

        except Exception as e:
            error_result = {
                "success": False,
                "run_name": self.config.run_name,
                "error": str(e),
            }
            self._log_execution("workflow_error", success=False, error=str(e))
            raise

    def get_status(self) -> dict[str, Any]:
        """Get current pipeline status.

        Returns:
            Status dictionary with execution state
        """
        return self.orchestrator.get_status()

    def pause(self) -> None:
        """Pause ongoing training."""
        self.orchestrator.pause_training()
        self._log_execution("pause")

    def resume(self) -> dict[str, float]:
        """Resume paused training.

        Returns:
            Updated metrics
        """
        try:
            metrics = self.orchestrator.resume_training()
            self._log_execution("resume", success=True)
            return metrics
        except Exception as e:
            self._log_execution("resume", success=False, error=str(e))
            raise

    def save_checkpoint(self, name: Optional[str] = None) -> Path:
        """Save model checkpoint.

        Args:
            name: Custom checkpoint name

        Returns:
            Path to saved checkpoint
        """
        path = self.orchestrator.save_checkpoint(name)
        self._log_execution("checkpoint", success=True, path=str(path))
        return path

    def get_config_summary(self) -> str:
        """Get formatted configuration summary.

        Returns:
            Human-readable configuration description
        """
        summary = f"""
Finetuning Configuration: {self.config.run_name}
{'=' * 60}

Method: {self.config.method.upper()}
Model: {self.config.model_id}
Device: {self.config.device}
Mixed Precision: {self.config.mixed_precision}

Training Parameters:
  Epochs: {self.config.trainer.num_train_epochs}
  Batch Size: {self.config.trainer.per_device_train_batch_size}
  Learning Rate: {self.config.trainer.learning_rate}
  Warmup Steps: {self.config.trainer.warmup_steps}

Dataset:
  Source: {self.config.data.dataset_path or self.config.data.dataset_name}
  Text Column: {self.config.data.text_column}
  Validation Split: {self.config.data.val_split}

Output Directory: {self.config.output_dir / self.config.run_name}
"""

        if self.config.method == FinetuneMethod.LORA and self.config.lora_config:
            lora = self.config.lora_config
            summary += f"""
LoRA Configuration:
  Rank (r): {lora.r}
  Alpha: {lora.lora_alpha}
  Dropout: {lora.lora_dropout}
  Target Modules: {", ".join(lora.target_modules)}
"""

        elif self.config.method == FinetuneMethod.QLORA and self.config.qlora_config:
            qlora = self.config.qlora_config
            summary += f"""
QLoRA Configuration:
  Bits: {qlora.quantization_bits}
  Quantization Type: {qlora.bnb_4bit_quant_type}
  Rank (r): {qlora.r}
  Alpha: {qlora.lora_alpha}
  Dropout: {qlora.lora_dropout}
  Target Modules: {", ".join(qlora.target_modules)}
"""

        return summary

    def export_config(self, output_path: Path) -> None:
        """Export configuration to JSON file.

        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.config.dict()
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Configuration exported to {output_path}")

    def _log_execution(
        self,
        action: str,
        success: bool = True,
        error: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log pipeline action for audit trail.

        Args:
            action: Action name
            success: Whether action succeeded
            error: Error message if failed
            **kwargs: Additional metadata
        """
        log_entry = {
            "action": action,
            "success": success,
            "error": error,
            "metadata": kwargs,
        }
        self.execution_log.append(log_entry)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.orchestrator.cleanup()
        logger.info("Pipeline cleanup complete")
