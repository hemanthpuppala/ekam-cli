"""Full model finetuning trainer implementation."""

from pathlib import Path
from typing import Any

from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch

from ..base_trainer import BaseTrainer
from ..config import FinetuneConfig
from ..data_loader import load_dataset


class FullTrainer(BaseTrainer):
    """Full model finetuning trainer.

    Full finetuning updates all model parameters:
    - Updates 100% of model parameters
    - Highest quality improvements on custom domains
    - Requires significant VRAM/memory
    - ~24GB+ VRAM for 7B models
    - Slowest training but best results

    Memory requirements:
    - 7B model: ~24GB VRAM
    - 13B model: ~50GB VRAM
    - Recommended: A100, H100, or RTX 4090+

    Best for:
    - Significant domain shift from pretraining
    - Task-specific architectural needs
    - When maximum accuracy needed and hardware available
    - Transfer learning with large custom datasets
    """

    def prepare_model(self) -> tuple[Any, Any]:
        """Load full model for complete finetuning.

        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            logger.info(f"Loading full model: {self.config.model_id}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("✓ Tokenizer loaded")

            # Load full model without quantization
            device_map = self._get_device_map()
            dtype = torch.float16 if self.config.device == "cuda" else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
                device_map=device_map,
            )

            logger.info(f"✓ Full model loaded on {self.config.device} ({dtype})")

            # Count trainable parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            logger.info(f"Model size: {total_params / 1e9:.2f}B parameters")
            logger.info(f"Trainable: {trainable_params / 1e9:.2f}B ({100 * trainable_params / total_params:.2f}%)")

            self.model = model
            return model, self.tokenizer

        except Exception as e:
            logger.error(f"Failed to prepare model: {e}", exc_info=True)
            raise

    def prepare_dataset(self) -> tuple[Any, Any, Any]:
        """Load and prepare dataset for full finetuning.

        Returns:
            Tuple of (train_dataset, eval_dataset, data_collator)
        """
        try:
            logger.info(f"Loading dataset from {self.config.data.dataset_path}")

            train_dataset, eval_dataset = load_dataset(
                config=self.config.data,
                tokenizer=self.tokenizer,
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            logger.info(f"✓ Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
            return train_dataset, eval_dataset, data_collator

        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}", exc_info=True)
            raise

    def setup_training(self) -> Trainer:
        """Setup trainer for full finetuning with memory optimization.

        Returns:
            Configured Trainer instance
        """
        try:
            if self.model is None:
                self.prepare_model()

            train_dataset, eval_dataset, data_collator = self.prepare_dataset()

            # Full finetuning training arguments with memory optimization
            training_args = TrainingArguments(
                output_dir=str(self.config.output_dir / self.config.run_name),
                num_train_epochs=self.config.trainer.num_train_epochs,
                per_device_train_batch_size=self.config.trainer.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.trainer.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.trainer.gradient_accumulation_steps,
                learning_rate=self.config.trainer.learning_rate,
                weight_decay=self.config.trainer.weight_decay,
                warmup_steps=self.config.trainer.warmup_steps,
                max_grad_norm=self.config.trainer.max_grad_norm,
                lr_scheduler_type=self.config.trainer.lr_scheduler_type,
                save_strategy=self.config.trainer.save_strategy,
                evaluation_strategy=self.config.trainer.eval_strategy,
                save_total_limit=self.config.trainer.save_total_limit,
                load_best_model_at_end=self.config.trainer.load_best_model_at_end,
                metric_for_best_model=self.config.trainer.metric_for_best_model,
                greater_is_better=self.config.trainer.greater_is_better,
                logging_steps=self.config.trainer.logging_steps,
                report_to=self.config.trainer.report_to,
                fp16=self.config.mixed_precision == "fp16",
                bf16=self.config.mixed_precision == "bf16",
                seed=self.config.seed,
                dataloader_num_workers=self.config.num_workers,
                remove_unused_columns=False,
                gradient_checkpointing=True,  # Memory optimization
                optim="adamw_8bit",  # Memory-efficient optimizer if available
            )

            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )

            self.trainer = trainer
            logger.info("✓ Full finetuning trainer setup complete")
            return trainer

        except Exception as e:
            logger.error(f"Failed to setup full training: {e}", exc_info=True)
            raise

    def _get_device_map(self) -> "str | dict":
        """Determine device mapping for full model.

        Returns:
            Device string or mapping configuration
        """
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                logger.warning("Full finetuning requires CUDA. Falling back to CPU (very slow)")
                return "cpu"
        return self.config.device
