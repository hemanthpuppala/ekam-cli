"""Finetuning integration for main app."""

from pathlib import Path
from typing import Optional

from loguru import logger

from ..cli.tui_manager import tui
from ..finetuning.config import FinetuneConfig, FinetuneMethod, LoRAConfig, QLoRAConfig
from ..finetuning.pipeline import FinetuningPipeline
from ..finetuning.ui.menus import (
    ConfigurationMenu,
    FinetuneHomeMenu,
    MethodSelectionMenu,
    ProgressMonitor,
)
from ..finetuning.ui.information import MethodInformation
from ..services.session import SessionManager


def show_finetuning_home_menu() -> str:
    """Display finetuning home menu with navigation.

    Returns:
        Next action: "new_run", "inference", "quantization", etc.
    """
    tui.clear_screen()

    home_menu = FinetuneHomeMenu()

    while True:
        choice = home_menu.display()
        target = home_menu.handle_selection(choice)

        if target == "new_run":
            return "start_finetuning"
        elif target == "load_config":
            return "load_finetuning_config"
        elif target == "view_info":
            _show_method_comparison()
        elif target == "requirements":
            _show_system_requirements()
        elif target == "manage_models":
            _manage_saved_models()
        elif target == "inference":
            return "inference"
        elif target == "quantization":
            return "quantization"
        elif target == "main_menu":
            return "main_menu"
        elif target == "show_info":
            _show_finetuning_info()
        elif target == "help":
            _show_finetuning_help()
        elif target == "quit":
            return "quit"


def start_finetuning_workflow(session_manager: SessionManager) -> Optional[FinetuningPipeline]:
    """Start new finetuning workflow with user prompts.

    Returns:
        Initialized FinetuningPipeline or None if cancelled
    """
    tui.clear_screen()

    # Step 0: Select model from all providers
    from ..finetuning.model_discovery import FinetuneModelDiscovery
    from ..finetuning.ui.menus import ModelSelectionMenu
    
    tui.show_panel(
        "[bold cyan]Step 1: Select Model for Finetuning[/bold cyan]",
        title="Finetuning Setup",
        border_style="cyan",
    )

    model_discovery = FinetuneModelDiscovery(session_manager)
    model_menu = ModelSelectionMenu(model_discovery)
    selected_model_id = model_menu.display()

    if selected_model_id is None:
        return None

    tui.clear_screen()

    # Step 1: Select method
    tui.show_panel(
        "[bold cyan]Step 2: Select Finetuning Method[/bold cyan]",
        title="Finetuning Setup",
        border_style="cyan",
    )

    method_menu = MethodSelectionMenu()
    method = method_menu.display()

    if method is None:
        return None

    tui.clear_screen()

    # Step 2: Configure settings
    tui.show_panel(
        "[bold cyan]Step 3: Configure Finetuning Parameters[/bold cyan]",
        title="Configuration",
        border_style="cyan",
    )

    config_menu = ConfigurationMenu(method)
    user_config = config_menu.configure_interactive()

    # Step 3: Build configuration object
    tui.clear_screen()
    tui.show_panel(
        "[bold yellow]Building finetuning configuration...[/bold yellow]",
        title="Initialization",
    )

    # Create method-specific config
    if method == FinetuneMethod.LORA:
        method_config = LoRAConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        )
        finetune_config = FinetuneConfig(
            run_name=user_config["run_name"],
            method=method,
            model_id=selected_model_id,
            lora_config=method_config,
            data=_create_data_config(user_config),
            trainer=_create_trainer_config(user_config),
        )

    elif method == FinetuneMethod.QLORA:
        method_config = QLoRAConfig(
            r=8,
            lora_alpha=16,
            quantization_bits=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        )
        finetune_config = FinetuneConfig(
            run_name=user_config["run_name"],
            method=method,
            model_id=selected_model_id,
            qlora_config=method_config,
            data=_create_data_config(user_config),
            trainer=_create_trainer_config(user_config),
        )

    else:  # FULL
        finetune_config = FinetuneConfig(
            run_name=user_config["run_name"],
            method=method,
            model_id=selected_model_id,
            data=_create_data_config(user_config),
            trainer=_create_trainer_config(user_config),
        )

    # Create pipeline
    pipeline = FinetuningPipeline(finetune_config)

    # Show configuration summary
    tui.clear_screen()
    tui.show_panel(
        pipeline.get_config_summary(),
        title="Configuration Summary",
        border_style="green",
    )

    # Confirm and proceed
    proceed = tui.prompt("Proceed with finetuning? [y/N]:", style="cyan").strip().lower() in ["y", "yes"]

    if proceed:
        return pipeline
    else:
        tui.show_message("Finetuning cancelled")
        return None


def execute_finetuning(pipeline: FinetuningPipeline) -> bool:
    """Execute finetuning workflow.

    Args:
        pipeline: Configured FinetuningPipeline

    Returns:
        True if successful
    """
    try:
        tui.clear_screen()
        progress_monitor = ProgressMonitor()

        # Register progress callback
        pipeline.register_monitor(lambda status: progress_monitor.update(status))

        # Execute workflow
        tui.show_panel(
            "[bold cyan]Starting finetuning...[/bold cyan]",
            title="Execution",
        )

        result = pipeline.execute_full_workflow()

        tui.show_panel(
            f"""[bold green]✓ Finetuning completed successfully![/bold green]

Run: {result['run_name']}
Method: {result['method'].upper()}
Loss: {result['metrics'].get('train_loss', 'N/A')}
Model saved to: {result['save_path']}""",
            title="Success",
            border_style="green",
        )

        return True

    except Exception as e:
        logger.error(f"Finetuning failed: {e}", exc_info=True)
        tui.show_error(f"Finetuning failed: {e}")
        return False

    finally:
        pipeline.cleanup()


def _show_method_comparison() -> None:
    """Display detailed comparison of finetuning methods."""
    tui.clear_screen()
    info = MethodInformation()
    tui.show_panel(
        info.get_comparison(),
        title="Finetuning Methods Comparison",
        border_style="cyan",
        expand=False,
    )


def _show_system_requirements() -> None:
    """Show system requirements for finetuning."""
    requirements_text = """
SYSTEM REQUIREMENTS FOR FINETUNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LoRA FINETUNING
───────────────────────────────────────────────────────────────────
✓ Minimum GPU VRAM: 12GB (RTX 3060, RTX 4060 Ti)
✓ RAM: 16GB+
✓ Disk: 100GB+ free (for models and checkpoints)
✓ PyTorch with CUDA support recommended for speed
✓ Supports: CUDA, MPS (Apple Silicon), CPU (slower)

QLoRA FINETUNING
───────────────────────────────────────────────────────────────────
✓ Minimum GPU VRAM: 8GB (RTX 3050, RTX 4050, M1/M2/M3 Mac)
✓ RAM: 16GB+
✓ Disk: 100GB+ free
✓ Requires: bitsandbytes (CUDA only) or CPU fallback
✓ Supports: CUDA (best), CPU (works but slower)
✓ MPS: Limited 4-bit support

FULL FINETUNING
───────────────────────────────────────────────────────────────────
✓ Minimum GPU VRAM: 24GB (RTX A5000, RTX 6000)
✓ Recommended: A100 (40GB), H100 (80GB), RTX 4090 (24GB) + 2x GPU
✓ RAM: 32GB+
✓ Disk: 200GB+ free
✓ CUDA required (not practical on CPU)
✓ Supports: CUDA only

DEPENDENCIES
───────────────────────────────────────────────────────────────────
Core:
  • PyTorch 2.0+
  • transformers 4.30+
  • peft (Parameter-Efficient Fine-Tuning)
  • datasets (for data loading)
  • accelerate (for distributed training)

Optional:
  • wandb (monitoring and visualization)
  • tensorboard (local monitoring)
  • bitsandbytes (4-bit quantization)

DISK SPACE ESTIMATION
───────────────────────────────────────────────────────────────────
Per Model & Dataset:
  • Model checkpoint: ~14GB (7B), ~25GB (13B), ~70GB (70B)
  • Training checkpoints: 3-5 × model size
  • Adapters (LoRA): 50-300MB
  • Optimizer states: ~50% model size

Total needed: ~100-200GB free space for typical setup

INTERNET
───────────────────────────────────────────────────────────────────
• Download models: 1st time only (14-70GB per model)
• HuggingFace Hub access required
• Recommended: Fast internet (500 Mbps+) for model downloads
"""

    tui.show_panel(
        requirements_text,
        title="System Requirements",
        border_style="yellow",
        expand=False,
    )


def _show_finetuning_info() -> None:
    """Show general finetuning information."""
    info_text = """
FINETUNING OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What is Finetuning?
───────────────────────────────────────────────────────────────────
Finetuning is the process of adapting a pre-trained language model to
your specific domain or task. Instead of training from scratch (expensive),
you leverage the knowledge already learned by the model and adapt it to
your specific use case.

Benefits of Finetuning
───────────────────────────────────────────────────────────────────
✓ 100-1000x faster than training from scratch
✓ Requires 100-1000x less data than training from scratch
✓ Significantly better results than prompting alone
✓ Can specialize models for your domain/task
✓ Fine-grained control over model behavior

When to Finetune
───────────────────────────────────────────────────────────────────
✓ Prompting alone doesn't give desired results
✓ Need consistent output format or style
✓ Domain-specific knowledge required
✓ Task-specific language patterns
✓ When you have 100+ examples of desired behavior

When NOT to Finetune
───────────────────────────────────────────────────────────────────
✗ Simple prompting works well
✗ Only 10-20 examples available
✗ Task is too simple (just needs summarization, etc.)
✗ Hardware constraints are severe and can't use LoRA/QLoRA

Typical Workflow
───────────────────────────────────────────────────────────────────
1. Prepare dataset (~1K-100K examples for LoRA)
2. Select finetuning method (LoRA/QLoRA/Full)
3. Configure hyperparameters
4. Run finetuning (minutes to hours)
5. Evaluate and iterate
6. Deploy finetuned model or adapters

Expected Improvements
───────────────────────────────────────────────────────────────────
After finetuning, you typically see:
  • 5-15% accuracy improvement (LoRA)
  • 10-30% improvement (QLoRA with good data)
  • 20-50%+ improvement (Full finetuning with large datasets)
  • Much better output consistency
  • Better handling of domain-specific terminology
  • More reliable following of instructions
"""

    tui.show_panel(
        info_text,
        title="About Finetuning",
        border_style="cyan",
        expand=False,
    )


def _show_finetuning_help() -> None:
    """Show finetuning help and common issues."""
    help_text = """
FINETUNING HELP & TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Common Issues
───────────────────────────────────────────────────────────────────

Issue: Out of Memory (OOM)
Solution:
  1. Reduce batch_size (try 4, then 2)
  2. Enable gradient_accumulation (4-8 steps)
  3. Use QLoRA instead of LoRA
  4. Reduce max sequence length
  5. Use smaller model (7B instead of 13B)

Issue: Training Loss Not Decreasing
Solution:
  1. Increase learning_rate (try 5e-4)
  2. Check dataset quality and format
  3. Ensure text_column matches your data
  4. Verify dataset isn't too small (<100 examples)
  5. Reduce learning_rate (try 5e-5) if overfitting

Issue: Poor Results After Training
Solution:
  1. Ensure sufficient training data (at least 500 examples)
  2. Verify data quality and relevance
  3. Train for more epochs (try 5-10)
  4. Evaluate on validation set
  5. Check if learning_rate too high/low

Issue: Very Slow Training
Solution:
  1. Enable GPU support (check PyTorch CUDA)
  2. Reduce dataset size for testing
  3. Use LoRA instead of full finetuning
  4. Enable mixed precision (fp16)

Tips for Success
───────────────────────────────────────────────────────────────────
✓ Start with LoRA for exploration
✓ Quality > Quantity for training data
✓ Remove duplicates and low-quality examples
✓ Include diverse examples
✓ Validate on held-out test set
✓ Start with default hyperparameters
✓ Gradually increase complexity
✓ Monitor validation loss

Dataset Preparation Checklist
───────────────────────────────────────────────────────────────────
□ Data in correct format (JSON/CSV)
□ 'text' column (or specified column) with examples
□ No missing values or corrupted entries
□ Duplicates removed
□ Appropriate length (not too short, not too long)
□ Domain-relevant and representative
□ At least 500 examples recommended
□ Clean formatting and grammar
□ Balanced distribution if multi-class
"""

    tui.show_panel(
        help_text,
        title="Help & Troubleshooting",
        border_style="yellow",
        expand=False,
    )


def _manage_saved_models() -> None:
    """Show and manage saved finetuned models and adapters."""
    finetuned_dir = Path("results/finetuned_models")

    if not finetuned_dir.exists():
        tui.show_message("No finetuned models found yet.")
        return

    models = list(finetuned_dir.iterdir())
    if not models:
        tui.show_message("No finetuned models found yet.")
        return

    models_text = "SAVED FINETUNED MODELS & ADAPTERS\n"
    models_text += "━" * 60 + "\n\n"

    for model_path in models:
        model_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        model_size_mb = model_size / (1024 * 1024)
        models_text += f"📦 {model_path.name}\n"
        models_text += f"   Size: {model_size_mb:.1f}MB\n"
        models_text += f"   Path: {model_path}\n\n"

    tui.show_panel(
        models_text,
        title="Saved Models",
        border_style="green",
        expand=False,
    )


def _create_data_config(user_config: dict):
    """Create DataConfig from user input."""
    from ..finetuning.config import DataConfig

    return DataConfig(
        dataset_path=Path(user_config["dataset_path"]),
        text_column=user_config.get("text_column", "text"),
        val_split=0.1,
    )


def _create_trainer_config(user_config: dict):
    """Create TrainerConfig from user input."""
    from ..finetuning.config import TrainerConfig

    return TrainerConfig(
        num_train_epochs=user_config.get("num_epochs", 3),
        per_device_train_batch_size=user_config.get("batch_size", 8),
        learning_rate=user_config.get("learning_rate", 1e-4),
    )
