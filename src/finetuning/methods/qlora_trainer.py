"""QLoRA (Quantized LoRA) trainer implementation."""

from pathlib import Path
from typing import Any

from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import torch

from ..base_trainer import BaseTrainer
from ..config import FinetuneConfig
from ..data_loader import load_dataset


class QLoRATrainer(BaseTrainer):
    """QLoRA (Quantized LoRA) trainer.

    QLoRA combines 4-bit quantization with LoRA adapters:
    - Quantizes model to 4-bit (or 8-bit) precision
    - Applies LoRA adapters on top
    - Uses only 8% memory vs full finetuning
    - Can finetune 7B models with 8GB VRAM
    - Slight quality trade-off for massive memory savings

    Memory breakdown for 7B model:
    - Full finetuning: ~24GB VRAM
    - LoRA finetuning: ~13.6GB VRAM
    - QLoRA finetuning: ~1.9GB VRAM (!)

    Best for:
    - Consumer hardware with limited VRAM (< 16GB)
    - Large model finetuning on budget
    - When every MB of VRAM matters
    - Works on CUDA, but has limitations on MPS and CPU
    """

    def prepare_model(self) -> tuple[Any, Any]:
        """Load model in 4-bit quantization and apply LoRA.

        Returns:
            Tuple of (peft_model, tokenizer)
        """
        try:
            logger.info(f"Loading model in {self.config.qlora_config.quantization_bits}-bit quantization")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("✓ Tokenizer loaded")

            # Setup 4-bit quantization configuration
            qlora_config = self.config.qlora_config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=qlora_config.quantization_bits == 4,
                load_in_8bit=qlora_config.quantization_bits == 8,
                bnb_4bit_compute_dtype=self._get_dtype(qlora_config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
            )

            logger.info(f"4-bit config: quant_type={qlora_config.bnb_4bit_quant_type}, "
                       f"double_quant={qlora_config.bnb_4bit_use_double_quant}")

            # Load quantized model
            device_map = self._get_device_map()
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                quantization_config=bnb_config,
                device_map=device_map,
            )
            logger.info(f"✓ Model loaded in {qlora_config.quantization_bits}-bit on {self.config.device}")

            # Apply LoRA configuration
            lora_config = LoraConfig(
                r=qlora_config.r,
                lora_alpha=qlora_config.lora_alpha,
                target_modules=qlora_config.target_modules,
                lora_dropout=qlora_config.lora_dropout,
                bias=qlora_config.bias,
                task_type="CAUSAL_LM",
            )

            peft_model = get_peft_model(model, lora_config)
            peft_model.print_trainable_parameters()

            self.model = peft_model
            return peft_model, self.tokenizer

        except Exception as e:
            logger.error(f"Failed to prepare QLoRA model: {e}", exc_info=True)
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
        """Setup trainer with QLoRA-optimized settings.

        Returns:
            Configured Trainer instance
        """
        try:
            if self.model is None:
                self.prepare_model()

            train_dataset, eval_dataset, data_collator = self.prepare_dataset()

            # QLoRA-optimized training arguments
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
                optim="paged_adamw_8bit",  # QLoRA recommendation: memory-efficient optimizer
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
            logger.info("✓ QLoRA trainer setup complete")
            return trainer

        except Exception as e:
            logger.error(f"Failed to setup QLoRA training: {e}", exc_info=True)
            raise

    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch dtype.

        Args:
            dtype_str: Dtype name (float16, float32, bfloat16)

        Returns:
            torch.dtype
        """
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float16)

    def _get_device_map(self) -> "str | dict":
        """Determine device mapping for model loading.

        Returns:
            Device string or mapping configuration
        """
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                logger.warning("QLoRA has limited support on MPS. Falling back to CPU (slower)")
                return "cpu"
            else:
                return "cpu"
        return self.config.device
