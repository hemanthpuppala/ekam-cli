"""LoRA (Low-Rank Adaptation) trainer implementation."""

from pathlib import Path
from typing import Any

from loguru import logger
from peft import LoraConfig, get_peft_model
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


class LoRATrainer(BaseTrainer):
    """LoRA (Low-Rank Adaptation) trainer.

    LoRA is a parameter-efficient finetuning method that:
    - Adds small trainable adapters to model layers
    - Keeps model weights frozen
    - Reduces trainable parameters to 0.062% of full finetuning
    - Uses ~57% memory vs full finetuning
    - Works on CPU, GPU (CUDA), and MPS (Apple Silicon)

    Best for:
    - Limited VRAM/memory scenarios
    - Domain adaptation with moderate-sized datasets
    - Multi-task finetuning (multiple adapters per model)
    """

    def prepare_model(self) -> tuple[Any, Any]:
        """Load base model and apply LoRA configuration.

        Returns:
            Tuple of (peft_model, tokenizer)
        """
        try:
            logger.info(f"Loading base model: {self.config.model_id}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("✓ Tokenizer loaded")

            # Load model on specified device
            device_map = self._get_device_map()
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map=device_map,
            )
            logger.info(f"✓ Base model loaded on {self.config.device}")

            # Apply LoRA configuration
            lora_config = self.config.lora_config
            logger.info(f"Applying LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")

            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                task_type="CAUSAL_LM",
            )

            peft_model = get_peft_model(model, peft_config)
            peft_model.print_trainable_parameters()

            self.model = peft_model
            return peft_model, self.tokenizer

        except Exception as e:
            logger.error(f"Failed to prepare model: {e}", exc_info=True)
            raise

    def prepare_dataset(self) -> tuple[Any, Any, Any]:
        """Load and prepare dataset for training.

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
        """Setup trainer with training arguments.

        Returns:
            Configured Trainer instance
        """
        try:
            if self.model is None:
                self.prepare_model()

            train_dataset, eval_dataset, data_collator = self.prepare_dataset()

            # Setup training arguments
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
            logger.info("✓ Trainer setup complete")
            return trainer

        except Exception as e:
            logger.error(f"Failed to setup training: {e}", exc_info=True)
            raise

    def _get_device_map(self) -> str | dict:
        """Determine device mapping for model loading.

        Returns:
            Device string or mapping configuration
        """
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
