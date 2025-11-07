"""Intelligent orchestrator for stateful finetuning pipeline management."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from .config import FinetuneConfig, FinetuneMethod
from .methods import FullTrainer, LoRATrainer, QLoRATrainer


class PipelineStep(str, Enum):
    """Pipeline execution steps."""

    INIT = "init"
    CONFIG_REVIEW = "config_review"
    MODEL_LOADING = "model_loading"
    DATASET_PREP = "dataset_prep"
    TRAINING_SETUP = "training_setup"
    TRAINING = "training"
    SAVE = "save"
    COMPLETE = "complete"


class TrainerState(str, Enum):
    """Trainer execution states."""

    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineContext:
    """Context maintained throughout pipeline execution."""

    config: FinetuneConfig
    current_step: PipelineStep = PipelineStep.INIT
    trainer_state: TrainerState = TrainerState.IDLE
    trainer: Optional[Any] = None
    progress: float = 0.0
    messages: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    can_go_back: bool = True

    def add_message(self, message: str) -> None:
        """Add status message to context."""
        self.messages.append(message)
        logger.info(message)

    def update_progress(self, step: PipelineStep, percent: float) -> None:
        """Update pipeline progress."""
        self.current_step = step
        self.progress = min(100, max(0, percent))
        logger.info(f"Progress: {step.value} - {self.progress:.1f}%")

    def mark_failed(self, error: str) -> None:
        """Mark pipeline as failed with error."""
        self.trainer_state = TrainerState.FAILED
        self.error = error
        logger.error(f"Pipeline failed: {error}")


class FinetuneOrchestrator:
    """Intelligent orchestrator managing finetuning pipeline execution.

    Features:
    - Stateful execution with context awareness
    - Support for resuming interrupted training
    - Progress tracking and metrics collection
    - Device-aware training strategy selection
    - Cross-step navigation (back, skip, resume)
    """

    def __init__(self, config: FinetuneConfig):
        """Initialize orchestrator with configuration.

        Args:
            config: Finetuning configuration
        """
        self.config = config
        self.context = PipelineContext(config=config)
        self.callbacks: dict[str, list[Callable]] = {
            "step_start": [],
            "step_complete": [],
            "progress_update": [],
            "error": [],
        }

        logger.info(f"Orchestrator initialized: {config.method} finetuning of {config.model_id}")

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for pipeline events.

        Args:
            event: Event type (step_start, step_complete, progress_update, error)
            callback: Function to call when event occurs
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def _emit(self, event: str, data: Any = None) -> None:
        """Emit event to registered callbacks.

        Args:
            event: Event type
            data: Event data
        """
        for callback in self.callbacks.get(event, []):
            try:
                callback(data or self.context)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _create_trainer(self) -> Any:
        """Create appropriate trainer based on method.

        Returns:
            Trainer instance

        Raises:
            ValueError: If method is unsupported
        """
        self.context.update_progress(PipelineStep.MODEL_LOADING, 20)
        self.context.add_message(f"Creating {self.config.method} trainer...")

        if self.config.method == FinetuneMethod.LORA:
            self.config.validate_method_config()
            return LoRATrainer(self.config)
        elif self.config.method == FinetuneMethod.QLORA:
            self.config.validate_method_config()
            return QLoRATrainer(self.config)
        elif self.config.method == FinetuneMethod.FULL:
            return FullTrainer(self.config)
        else:
            raise ValueError(f"Unsupported method: {self.config.method}")

    def prepare_training(self) -> bool:
        """Prepare model and dataset for training.

        Returns:
            True if preparation successful

        Raises:
            Exception: If preparation fails
        """
        try:
            self.context.trainer_state = TrainerState.PREPARING
            self._emit("step_start", {"step": PipelineStep.MODEL_LOADING})

            # Create and setup trainer
            trainer = self._create_trainer()
            trainer.prepare_model()
            self.context.update_progress(PipelineStep.MODEL_LOADING, 50)
            self._emit("progress_update")

            self.context.add_message("Model prepared successfully")
            self.context.update_progress(PipelineStep.DATASET_PREP, 60)

            trainer.prepare_dataset()
            self.context.update_progress(PipelineStep.DATASET_PREP, 80)
            self.context.add_message("Dataset prepared successfully")

            self.context.update_progress(PipelineStep.TRAINING_SETUP, 90)
            trainer.setup_training()
            self.context.add_message("Training setup complete")

            self.context.trainer = trainer
            self.context.update_progress(PipelineStep.TRAINING_SETUP, 100)
            self._emit("step_complete", {"step": PipelineStep.TRAINING_SETUP})

            return True

        except Exception as e:
            error_msg = f"Preparation failed: {e}"
            self.context.mark_failed(error_msg)
            self._emit("error", {"error": error_msg})
            raise

    def start_training(self) -> dict[str, float]:
        """Start model finetuning.

        Returns:
            Training metrics

        Raises:
            RuntimeError: If trainer not prepared
        """
        try:
            if self.context.trainer is None:
                raise RuntimeError("Trainer not prepared. Call prepare_training() first")

            self.context.trainer_state = TrainerState.TRAINING
            self.context.current_step = PipelineStep.TRAINING
            self._emit("step_start", {"step": PipelineStep.TRAINING})

            # Execute training
            metrics = self.context.trainer.train()

            self.context.metrics = metrics
            self.context.trainer_state = TrainerState.COMPLETED
            self.context.update_progress(PipelineStep.TRAINING, 100)
            self.context.add_message(f"Training completed: {metrics}")

            self._emit("step_complete", {"step": PipelineStep.TRAINING})
            return metrics

        except Exception as e:
            error_msg = f"Training failed: {e}"
            self.context.mark_failed(error_msg)
            self._emit("error", {"error": error_msg})
            raise

    def save_checkpoint(self, name: Optional[str] = None) -> Path:
        """Save model checkpoint.

        Args:
            name: Custom checkpoint name

        Returns:
            Path to saved checkpoint

        Raises:
            RuntimeError: If trainer not initialized
        """
        try:
            if self.context.trainer is None:
                raise RuntimeError("No trainer to save from")

            if self.config.method in [FinetuneMethod.LORA, FinetuneMethod.QLORA]:
                path = self.context.trainer.save_adapter(
                    Path(self.config.output_dir) / (name or f"{self.config.run_name}_adapter")
                )
            else:
                path = self.context.trainer.save_model(
                    Path(self.config.output_dir) / (name or self.config.run_name)
                )

            self.context.add_message(f"Checkpoint saved: {path}")
            return path

        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            raise

    def get_status(self) -> dict[str, Any]:
        """Get current pipeline status.

        Returns:
            Status dictionary with current state and progress
        """
        return {
            "step": self.context.current_step.value,
            "trainer_state": self.context.trainer_state.value,
            "progress": self.context.progress,
            "messages": self.context.messages[-5:],  # Last 5 messages
            "metrics": self.context.metrics,
            "error": self.context.error,
            "can_go_back": self.context.can_go_back,
        }

    def pause_training(self) -> None:
        """Pause ongoing training."""
        if self.context.trainer_state == TrainerState.TRAINING:
            self.context.trainer_state = TrainerState.PAUSED
            self.context.add_message("Training paused")

    def resume_training(self) -> dict[str, float]:
        """Resume paused training."""
        if self.context.trainer_state != TrainerState.PAUSED:
            raise RuntimeError("No paused training to resume")

        self.context.trainer_state = TrainerState.TRAINING
        self.context.add_message("Resuming training...")
        return self.start_training()

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.context.trainer:
            self.context.trainer.cleanup()
            logger.info("Resources cleaned up")
