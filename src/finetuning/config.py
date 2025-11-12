"""Configuration models for finetuning pipeline."""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class FinetuneMethod(str, Enum):
    """Supported finetuning methods."""

    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"


class DataConfig(BaseModel):
    """Dataset configuration for finetuning."""

    dataset_path: Path = Field(description="Path to dataset file (JSON, CSV, or Hugging Face dataset)")
    dataset_name: Optional[str] = Field(
        default=None, description="HuggingFace dataset name (if using HF datasets)"
    )
    text_column: str = Field(default="text", description="Column name containing training text")
    split: str = Field(default="train", description="Dataset split to use")
    max_samples: Optional[int] = Field(default=None, description="Limit samples for debugging")
    val_split: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Validation split ratio (0.0-1.0)",
    )
    preprocessing: dict = Field(
        default_factory=dict,
        description="Text preprocessing options (lowercase, remove_special_chars, etc)",
    )


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) specific configuration."""

    r: int = Field(default=8, ge=1, le=128, description="LoRA rank (8-16 balanced, higher=more params)")
    lora_alpha: int = Field(default=16, ge=1, description="LoRA scaling factor (typically 2*r)")
    target_modules: list[str] = Field(
        default=["q_proj", "v_proj"],
        description="Modules to apply LoRA to (q_proj, v_proj, k_proj, etc)",
    )
    lora_dropout: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Dropout rate in LoRA layers"
    )
    bias: str = Field(
        default="none",
        pattern="^(none|all|lora_only)$",
        description="Bias tuning: none (no bias), all (all biases), lora_only (LoRA biases only)",
    )
    modules_to_save: list[str] = Field(
        default=[],
        description="Additional modules to save (e.g., embedding layers for domain adaptation)",
    )


class QLoRAConfig(LoRAConfig):
    """QLoRA (Quantized LoRA) specific configuration."""

    quantization_bits: int = Field(
        default=4,
        pattern="^(4|8)$",
        description="Quantization precision: 4-bit (8% memory) or 8-bit (40% memory)",
    )
    bnb_4bit_compute_dtype: str = Field(
        default="float16",
        pattern="^(float16|float32|bfloat16)$",
        description="Compute dtype (float16 for speed, bfloat16 for stability)",
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        pattern="^(nf4|fp4)$",
        description="Quantization type: nf4 (recommended) or fp4",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True, description="Use double quantization (additional 0.4 bits reduction)"
    )


class TrainerConfig(BaseModel):
    """Training hyperparameters."""

    num_train_epochs: int = Field(default=3, ge=1, le=100, description="Number of training epochs")
    per_device_train_batch_size: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Batch size (smaller for limited VRAM, e.g., 4 for 8GB)",
    )
    per_device_eval_batch_size: int = Field(
        default=16, ge=1, description="Evaluation batch size (can be larger)"
    )
    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
        description="Gradient accumulation (simulates larger batch size without memory overhead)",
    )
    learning_rate: float = Field(
        default=1e-4, ge=1e-6, le=1e-2, description="Initial learning rate (1e-4 for LoRA, 5e-5 for full)"
    )
    weight_decay: float = Field(default=0.01, ge=0.0, description="L2 regularization strength")
    warmup_steps: int = Field(default=100, ge=0, description="Linear warmup steps")
    max_grad_norm: float = Field(default=1.0, ge=0.1, description="Gradient clipping threshold")
    lr_scheduler_type: str = Field(
        default="cosine",
        pattern="^(linear|cosine|cosine_with_restarts|polynomial)$",
        description="Learning rate schedule strategy",
    )
    save_strategy: str = Field(
        default="epoch",
        pattern="^(steps|epoch|no)$",
        description="When to save checkpoints",
    )
    eval_strategy: str = Field(
        default="epoch",
        pattern="^(steps|epoch|no)$",
        description="When to evaluate on validation set",
    )
    save_total_limit: int = Field(
        default=3, ge=1, description="Keep only N best checkpoints"
    )
    load_best_model_at_end: bool = Field(
        default=True,
        description="Load best checkpoint after training",
    )
    metric_for_best_model: str = Field(
        default="eval_loss",
        description="Metric to track for model selection",
    )
    greater_is_better: bool = Field(
        default=False,
        description="Whether higher metric values are better (loss: False, accuracy: True)",
    )
    logging_steps: int = Field(default=10, ge=1, description="Log metrics every N steps")
    report_to: list[str] = Field(
        default=["tensorboard"],
        description="Logging backends (tensorboard, wandb, etc)",
    )
    push_to_hub: bool = Field(
        default=False,
        description="Upload final model to Hugging Face Hub",
    )


class FinetuneConfig(BaseModel):
    """Complete finetuning configuration combining all components."""

    # Identifiers
    run_name: str = Field(description="Unique name for this finetuning run")
    method: FinetuneMethod = Field(description="Finetuning method: LoRA, QLoRA, or Full")
    model_id: str = Field(description="Model identifier to finetune")
    output_dir: Path = Field(
        default=Path("results/finetuned_models"),
        description="Directory to save finetuned models and adapters",
    )

    # Data configuration
    data: DataConfig = Field(description="Dataset configuration")

    # Method-specific configurations
    lora_config: Optional[LoRAConfig] = Field(
        default=None, description="LoRA configuration (required if method=lora)"
    )
    qlora_config: Optional[QLoRAConfig] = Field(
        default=None, description="QLoRA configuration (required if method=qlora)"
    )

    # Training configuration
    trainer: TrainerConfig = Field(default_factory=TrainerConfig, description="Training hyperparameters")

    # Runtime settings
    device: str = Field(
        default="auto",
        pattern="^(auto|cuda|mps|cpu)$",
        description="Device to train on (auto for detection)",
    )
    mixed_precision: str = Field(
        default="fp16",
        pattern="^(no|fp16|bf16)$",
        description="Mixed precision training (fp16 for CUDA, bf16 for stability)",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    num_workers: int = Field(default=4, ge=0, description="DataLoader worker processes")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def validate_method_config(self) -> None:
        """Validate that method-specific config is present."""
        if self.method == FinetuneMethod.LORA and not self.lora_config:
            raise ValueError("lora_config required when method=lora")
        if self.method == FinetuneMethod.QLORA and not self.qlora_config:
            raise ValueError("qlora_config required when method=qlora")

    def get_method_config(self) -> "LoRAConfig | QLoRAConfig":
        """Get the configuration for the selected method."""
        if self.method == FinetuneMethod.LORA:
            return self.lora_config
        elif self.method == FinetuneMethod.QLORA:
            return self.qlora_config
        else:
            raise ValueError(f"Unknown method: {self.method}")
