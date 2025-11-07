"""Modular finetuning pipeline for LLMs and VLMs.

This module provides a comprehensive finetuning system supporting:
- Parameter-Efficient Fine-Tuning (PEFT) methods: LoRA, QLoRA
- Full model fine-tuning with gradient accumulation
- Cross-platform support: CUDA, MPS (Apple Silicon), CPU
- Stateful, modular pipeline with intelligent orchestration
- Live progress monitoring and background execution
"""

from .config import (
    DataConfig,
    FinetuneConfig,
    LoRAConfig,
    QLoRAConfig,
    TrainerConfig,
)
from .methods import LoRATrainer, QLoRATrainer, FullTrainer
from .orchestrator import FinetuneOrchestrator
from .pipeline import FinetuningPipeline
from .model_discovery import FinetuneModelDiscovery

__all__ = [
    "DataConfig",
    "FinetuneConfig",
    "LoRAConfig",
    "QLoRAConfig",
    "TrainerConfig",
    "LoRATrainer",
    "QLoRATrainer",
    "FullTrainer",
    "FinetuningPipeline",
    "FinetuneOrchestrator",
    "FinetuneModelDiscovery",
]
