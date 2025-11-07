"""Base trainer abstraction for modular finetuning implementations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .config import FinetuneConfig


class BaseTrainer(ABC):
    """Abstract base class for all finetuning trainers.

    Provides interface for model loading, training, and checkpoint management
    across different finetuning methods (LoRA, QLoRA, Full).
    """

    def __init__(self, config: FinetuneConfig):
        """Initialize trainer with configuration.

        Args:
            config: Complete finetuning configuration
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_history = {}

        logger.info(f"Initializing {self.__class__.__name__} with method: {config.method}")

    @abstractmethod
    def prepare_model(self) -> tuple[Any, Any]:
        """Load and prepare model for training.

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def prepare_dataset(self) -> tuple[Any, Any, Any]:
        """Prepare dataset for training.

        Returns:
            Tuple of (train_dataset, eval_dataset, data_collator)

        Raises:
            ValueError: If dataset is invalid or missing
        """
        pass

    @abstractmethod
    def setup_training(self) -> Any:
        """Setup trainer instance with appropriate configuration.

        Returns:
            Trainer instance ready for training

        Raises:
            RuntimeError: If training setup fails
        """
        pass

    def train(self) -> dict[str, float]:
        """Execute training loop.

        Returns:
            Dictionary with final training metrics

        Raises:
            RuntimeError: If training fails
        """
        try:
            if self.trainer is None:
                raise RuntimeError("Training not set up. Call setup_training() first")

            logger.info(f"Starting training for {self.config.run_name}")
            train_result = self.trainer.train()

            # Save final metrics
            self.training_history = {
                "train_loss": train_result.training_loss,
                "epoch": train_result.epoch,
            }

            logger.info(f"✓ Training completed: {self.training_history}")
            return self.training_history

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

    def save_model(self, output_path: Optional[Path] = None) -> Path:
        """Save finetuned model or adapters.

        Args:
            output_path: Custom path to save model (uses config.output_dir if None)

        Returns:
            Path to saved model

        Raises:
            RuntimeError: If save fails
        """
        try:
            save_path = output_path or self.config.output_dir / self.config.run_name
            save_path.mkdir(parents=True, exist_ok=True)

            if self.trainer is None:
                raise RuntimeError("No trainer instance to save from")

            self.trainer.save_model(str(save_path))

            if self.tokenizer:
                self.tokenizer.save_pretrained(str(save_path))

            logger.info(f"✓ Model saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save model: {e}", exc_info=True)
            raise

    def save_adapter(self, output_path: Optional[Path] = None) -> Path:
        """Save only adapter weights (for PEFT methods).

        Args:
            output_path: Custom path to save adapters (uses config.output_dir if None)

        Returns:
            Path to saved adapters

        Raises:
            NotImplementedError: If method doesn't support adapter saving
        """
        try:
            if self.model is None:
                raise RuntimeError("No model loaded to save adapters from")

            save_path = output_path or self.config.output_dir / f"{self.config.run_name}_adapter"
            save_path.mkdir(parents=True, exist_ok=True)

            # Save only adapter weights if PEFT model
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(str(save_path))
                logger.info(f"✓ Adapter saved to {save_path}")
                return save_path
            else:
                logger.warning("Model doesn't support adapter saving")
                return self.save_model(output_path)

        except Exception as e:
            logger.error(f"Failed to save adapter: {e}", exc_info=True)
            raise

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint directory

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            if self.trainer is None:
                raise RuntimeError("Trainer not initialized")

            self.trainer.train(resume_from_checkpoint=str(checkpoint_path))
            logger.info(f"✓ Resumed training from {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            raise

    def get_training_metrics(self) -> dict[str, Any]:
        """Get current training metrics and history.

        Returns:
            Dictionary with training progress and metrics
        """
        metrics = {
            "method": self.config.method,
            "run_name": self.config.run_name,
            "model_id": self.config.model_id,
            "history": self.training_history,
        }

        if self.trainer and hasattr(self.trainer, "state"):
            metrics.update({
                "current_epoch": self.trainer.state.epoch,
                "global_step": self.trainer.state.global_step,
                "max_steps": self.trainer.state.max_steps,
            })

        return metrics

    def cleanup(self) -> None:
        """Clean up resources (GPU memory, etc)."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Cleaned up training resources")
        except Exception as e:
            logger.debug(f"Cleanup warning: {e}")
