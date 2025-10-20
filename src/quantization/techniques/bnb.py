"""BitsAndBytes quantization implementation (Phase 2).

BitsAndBytes provides HuggingFace-integrated 4-bit and 8-bit quantization
with easy integration into transformers models.

References:
    - https://github.com/TimDettmers/bitsandbytes
    - https://huggingface.co/docs/transformers/main_classes/quantization
"""

from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ...models.model import ModelInfo
from ...models.provider import ProviderType
from ..models import QuantizationTask, QuantizationType
from .base import BaseQuantizer


class BitsAndBytesQuantizer(BaseQuantizer):
    """BitsAndBytes quantization implementation.

    Phase 2 implementation will support:
    - BnB 4-bit (NF4, FP4)
    - BnB 8-bit
    - Double quantization
    - Direct HuggingFace integration
    """

    def check_availability(self) -> tuple[bool, str]:
        """Check if BitsAndBytes tools are available.

        Returns:
            (available, message)
        """
        # Check for bitsandbytes library
        try:
            import bitsandbytes as bnb
        except ImportError:
            return False, "BitsAndBytes not installed. Run: pip install bitsandbytes"

        # Check if CUDA is available (required for BitsAndBytes quantization)
        try:
            import torch
            if not torch.cuda.is_available():
                return False, (
                    "BitsAndBytes requires CUDA GPU. "
                    "Your system doesn't have CUDA available. "
                    "For CPU/Mac, use Generic quantization (FP16/INT8/INT4) instead."
                )
        except ImportError:
            return False, "PyTorch not installed"

        return True, "BitsAndBytes available (CUDA detected)"

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported quantization types.

        Returns:
            List of supported types
        """
        return [
            QuantizationType.BNB_8BIT,
            QuantizationType.BNB_4BIT_NF4,
            QuantizationType.BNB_4BIT_FP4,
        ]

    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for quantization.

        BitsAndBytes works with HuggingFace models.

        Args:
            model_info: Model to quantize

        Returns:
            Path to model directory, or None if not compatible
        """
        # BitsAndBytes works with HuggingFace models
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
            QuantizationType.BNB_8BIT: 0.5,  # 50% of original
            QuantizationType.BNB_4BIT_NF4: 0.35,  # 35% of original
            QuantizationType.BNB_4BIT_FP4: 0.35,  # 35% of original
        }

        factor = size_factors.get(quant_type, 0.35)
        return original_size * factor

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Execute BitsAndBytes quantization.

        Args:
            task: Quantization task
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            from ..models import TaskStatus

            task.status = TaskStatus.RUNNING
            logger.info(f"Starting BitsAndBytes {task.quant_type.display_name} quantization")

            # Check if bitsandbytes is available
            available, message = self.check_availability()
            if not available:
                task.status = TaskStatus.FAILED
                task.error = message
                logger.error(f"BitsAndBytes not available: {message}")
                return False

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

            # Update progress: Loading model
            if progress_callback:
                progress_callback(10.0, None)
            task.progress = 10.0

            # Configure BitsAndBytes quantization
            if task.quant_type == QuantizationType.BNB_8BIT:
                logger.info("Configuring BitsAndBytes 8-bit quantization")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
            elif task.quant_type == QuantizationType.BNB_4BIT_NF4:
                logger.info("Configuring BitsAndBytes 4-bit NF4 quantization")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif task.quant_type == QuantizationType.BNB_4BIT_FP4:
                logger.info("Configuring BitsAndBytes 4-bit FP4 quantization")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="fp4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            else:
                raise ValueError(f"Unsupported BitsAndBytes quantization type: {task.quant_type}")

            # Load model with quantization (suppress transformers output)
            logger.info("Loading model with BitsAndBytes quantization config...")
            if progress_callback:
                progress_callback(30.0, None)
            task.progress = 30.0

            # Import output suppressor for clean TUI
            from ...utils.output_suppressor import suppress_transformers_output

            # Try AutoModelForCausalLM first (works for LLMs)
            # Fallback to AutoModel for VLMs and other architectures
            try:
                from transformers import AutoModel

                logger.info("Attempting to load with AutoModelForCausalLM...")
                
                with suppress_transformers_output():
                    model = AutoModelForCausalLM.from_pretrained(
                        model_identifier,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                
                logger.info("Model loaded successfully with AutoModelForCausalLM")

            except Exception as e:
                logger.warning(f"AutoModelForCausalLM failed: {str(e)[:200]}")
                logger.info("Trying AutoModel as fallback (for VLMs and other architectures)...")

                try:
                    with suppress_transformers_output():
                        model = AutoModel.from_pretrained(
                            model_identifier,
                            quantization_config=bnb_config,
                            device_map="auto",
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                        )
                    
                    logger.info("Model loaded successfully with AutoModel")

                except Exception as e2:
                    logger.error(f"AutoModel also failed: {str(e2)[:200]}")
                    raise RuntimeError(
                        f"Failed to load model with BitsAndBytes quantization.\n\n"
                        f"Model: {model_identifier}\n"
                        f"Error: {str(e)}\n\n"
                        f"This model may not be compatible with BitsAndBytes quantization."
                    )

            logger.info("Model loaded and quantized in memory")

            # Update progress: Quantization done, now saving
            if progress_callback:
                progress_callback(70.0, None)
            task.progress = 70.0

            # Load tokenizer/processor (VLMs need processor)
            logger.info("Detecting model type and loading tokenizer/processor...")
            
            # Detect if VLM
            is_vlm = False
            processor_or_tokenizer = None
            
            try:
                config = model.config
                arch = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else ""
                
                vlm_patterns = [
                    "ForConditionalGeneration", "VisionTextDual", "Llava", "Blip",
                    "Qwen2VL", "Qwen3VL", "QwenVL", "InstructBlip", "Idefics"
                ]
                is_vlm = any(pattern in arch for pattern in vlm_patterns)
                
                if not is_vlm:
                    config_dict = config.to_dict()
                    is_vlm = any(key in config_dict for key in ["vision_config", "visual_config", "image_encoder"])
                
                logger.info(f"Model architecture: {arch}, VLM: {is_vlm}")
            except Exception as e:
                logger.debug(f"Could not detect VLM from config: {e}")
            
            # Load processor for VLMs, tokenizer for LLMs
            if is_vlm:
                try:
                    logger.info("Loading processor for VLM...")
                    from transformers import AutoProcessor
                    
                    processor_or_tokenizer = AutoProcessor.from_pretrained(
                        model_identifier,
                        trust_remote_code=True
                    )
                    logger.info("✓ Processor loaded successfully (VLM)")
                except Exception as e:
                    logger.warning(f"Failed to load processor for VLM, trying tokenizer: {e}")
                    is_vlm = False
            
            if not is_vlm:
                logger.info("Loading tokenizer...")
                processor_or_tokenizer = AutoTokenizer.from_pretrained(
                    model_identifier,
                    trust_remote_code=True,
                )

            # Update progress: Saving
            if progress_callback:
                progress_callback(80.0, None)
            task.progress = 80.0

            # Ensure output directory exists
            task.output_path.parent.mkdir(parents=True, exist_ok=True)

            # For BitsAndBytes, we save as a directory (HuggingFace format)
            output_dir = task.output_path.parent / task.output_path.stem
            logger.info(f"Saving quantized model to: {output_dir}")

            # Save model and tokenizer/processor
            model.save_pretrained(output_dir)
            processor_or_tokenizer.save_pretrained(output_dir)
            
            if is_vlm:
                logger.info("✓ Saved VLM processor (includes image_processor + tokenizer)")
            else:
                logger.info("✓ Saved tokenizer")

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

            logger.info(f"BitsAndBytes quantization completed: {output_dir} ({size_gb:.2f}GB)")
            return True

        except Exception as e:
            error_msg = f"BitsAndBytes quantization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = error_msg
            return False
