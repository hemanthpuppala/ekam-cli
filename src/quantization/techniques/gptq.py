"""GPTQ quantization implementation (Phase 2).

GPTQ (GPT Quantization) provides GPU-optimized 3-bit and 4-bit quantization
with better performance than traditional methods.

References:
    - https://github.com/IST-DASLab/gptq
    - https://github.com/AutoGPTQ/AutoGPTQ
"""

from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ...models.model import ModelInfo
from ...models.provider import ProviderType
from ..models import QuantizationTask, QuantizationType
from .base import BaseQuantizer


class GPTQQuantizer(BaseQuantizer):
    """GPTQ quantization implementation.

    Phase 2 implementation will support:
    - GPTQ 4-bit (int4)
    - GPTQ 3-bit (int3)
    - GPU-optimized inference
    - Auto-GPTQ library integration
    """

    def check_availability(self) -> tuple[bool, str]:
        """Check if GPTQ tools are available.

        Returns:
            (available, message)
        """
        # Check for auto-gptq library
        try:
            import auto_gptq
        except ImportError:
            return False, "Auto-GPTQ not installed. Run: pip install auto-gptq"

        # Check if CUDA is available (required for GPTQ quantization)
        try:
            import torch
            if not torch.cuda.is_available():
                return False, (
                    "GPTQ requires CUDA GPU. "
                    "Your system doesn't have CUDA available. "
                    "For CPU/Mac, use Generic quantization (FP16/INT8/INT4) or GGUF instead."
                )
        except ImportError:
            return False, "PyTorch not installed"

        return True, "Auto-GPTQ available (CUDA detected)"

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported quantization types.

        Returns:
            List of supported types
        """
        return [
            QuantizationType.GPTQ_8BIT,
            QuantizationType.GPTQ_4BIT,
            QuantizationType.GPTQ_3BIT,
        ]

    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for quantization.

        GPTQ works with HuggingFace models.
        Note: Ollama/GGUF models are converted to HF format by orchestrator first.

        Args:
            model_info: Model to quantize

        Returns:
            Path to model directory, or None if not compatible
        """
        # Check if source_path is a converted HF directory (from Ollama/GGUF)
        # The orchestrator updates source_path after GGUF→HF conversion
        if model_info.source_path and model_info.source_path.exists():
            # If it's a directory with config.json, it's a HF model
            if model_info.source_path.is_dir():
                config_file = model_info.source_path / "config.json"
                if config_file.exists():
                    logger.info(f"Using converted HF model: {model_info.source_path}")
                    return model_info.source_path

        # GPTQ works with HuggingFace models
        if model_info.provider != ProviderType.HUGGINGFACE:
            return None

        # Check if model_id is a local path (from ./models directory or manual path)
        if model_info.model_id:
            model_path = Path(model_info.model_id)
            # If it's an absolute path and exists, it's a local model
            if model_path.exists() and model_path.is_absolute():
                return model_path

        # model_id is HuggingFace ID - return None (transformers will find it)
        return None

    def estimate_output_size(
        self, model_info: ModelInfo, quant_type: QuantizationType
    ) -> float:
        """Estimate output file size in GB.

        Args:
            model_info: Source model
            quant_type: Quantization type

        Returns:
            Estimated size in GB
        """
        original_size = model_info.size_gb

        size_factors = {
            QuantizationType.GPTQ_8BIT: 0.5,  # 50% of original
            QuantizationType.GPTQ_4BIT: 0.35,  # 35% of original
            QuantizationType.GPTQ_3BIT: 0.25,  # 25% of original
        }

        factor = size_factors.get(quant_type, 0.35)
        return original_size * factor

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Execute GPTQ quantization.

        Args:
            task: Quantization task
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful
        """
        try:
            from transformers import AutoTokenizer
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

            from ..models import TaskStatus

            task.status = TaskStatus.RUNNING
            logger.info(f"Starting GPTQ {task.quant_type.display_name} quantization")

            # Check if auto-gptq is available and auto-install if needed
            available, message = self.check_and_install_dependencies(
                auto_install=True,
                show_progress=True
            )
            if not available:
                task.status = TaskStatus.FAILED
                task.error = message
                logger.error(f"Auto-GPTQ not available: {message}")
                return False
            
            logger.info(f"✓ Auto-GPTQ available: {message}")

            # Update progress: Validating
            if progress_callback:
                progress_callback(5.0, None)
            task.progress = 5.0

            # Get model identifier (can be local path or HF model ID)
            # For HuggingFace models, use model_id directly - transformers will find cached version
            model_path = self.get_source_model_path(task.model_info)
            if model_path and model_path.exists():
                # Local path exists - use it
                model_identifier = str(model_path)
                logger.info(f"Quantizing model from local path: {model_path}")
            else:
                # Use model_id (HF will find cached model or download)
                model_identifier = task.model_info.model_id
                logger.info(f"Quantizing model using HF model ID: {model_identifier}")

            # Update progress: Configuring
            if progress_callback:
                progress_callback(10.0, None)
            task.progress = 10.0

            # Determine bit size
            if task.quant_type == QuantizationType.GPTQ_8BIT:
                bits = 8
            elif task.quant_type == QuantizationType.GPTQ_4BIT:
                bits = 4
            elif task.quant_type == QuantizationType.GPTQ_3BIT:
                bits = 3
            else:
                raise ValueError(f"Unsupported GPTQ quantization type: {task.quant_type}")

            logger.info(f"Configuring GPTQ {bits}-bit quantization")

            # Create quantization config
            quantize_config = BaseQuantizeConfig(
                bits=bits,
                group_size=128,
                desc_act=False,  # Disable for faster quantization
            )

            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_identifier,
                trust_remote_code=True,
            )

            # Update progress: Loading model
            if progress_callback:
                progress_callback(20.0, None)
            task.progress = 20.0

            # Load model for quantization
            logger.info("Loading model for GPTQ quantization...")

            # Check if model is a VLM
            from ...models.model import ModelType
            is_vlm = task.model_info.model_type == ModelType.VLM

            if is_vlm:
                # VLMs are not well-supported by AutoGPTQ
                # Provide helpful error message
                raise RuntimeError(
                    f"GPTQ quantization is not well-supported for VLMs.\n\n"
                    f"Model: {model_identifier}\n"
                    f"Type: Vision-Language Model (VLM)\n\n"
                    f"AutoGPTQ is designed for pure language models.\n"
                    f"For VLMs, please use:\n"
                    f"  • BitsAndBytes 4-bit (BnB 4-bit NF4)\n"
                    f"  • Generic quantization (FP16/INT8/INT4)\n\n"
                    f"Go back and select a different quantization method."
                )

            model = AutoGPTQForCausalLM.from_pretrained(
                model_identifier,
                quantize_config=quantize_config,
                trust_remote_code=True,
            )

            logger.info("Model loaded successfully")

            # Update progress: Preparing calibration data
            if progress_callback:
                progress_callback(30.0, None)
            task.progress = 30.0

            # Prepare simple calibration dataset
            logger.info("Preparing calibration dataset...")
            calibration_samples = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Python is a high-level programming language.",
                "The sky is blue and the grass is green.",
            ]

            # Tokenize calibration data
            examples = []
            for text in calibration_samples:
                examples.append(
                    tokenizer(
                        text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                    )
                )

            # Update progress: Quantizing
            if progress_callback:
                progress_callback(40.0, None)
            task.progress = 40.0

            # Run quantization
            logger.info("Running GPTQ quantization (this may take several minutes)...")
            model.quantize(examples)

            logger.info("Quantization completed")

            # Update progress: Saving
            if progress_callback:
                progress_callback(80.0, None)
            task.progress = 80.0

            # Ensure output directory exists
            task.output_path.parent.mkdir(parents=True, exist_ok=True)

            # For GPTQ, we save as a directory (HuggingFace format)
            output_dir = task.output_path.parent / task.output_path.stem
            logger.info(f"Saving quantized model to: {output_dir}")

            # Save model and tokenizer
            model.save_quantized(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Update task output_path to the directory
            task.output_path = output_dir

            # Update progress: Complete
            if progress_callback:
                progress_callback(100.0, 0)
            task.progress = 100.0
            task.status = TaskStatus.COMPLETED

            # Calculate actual size
            total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
            size_gb = total_size / (1024 ** 3)

            logger.info(f"GPTQ quantization completed: {output_dir} ({size_gb:.2f}GB)")
            return True

        except Exception as e:
            error_msg = f"GPTQ quantization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = error_msg
            return False
