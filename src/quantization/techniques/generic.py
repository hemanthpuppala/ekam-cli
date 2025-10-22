"""Cross-platform generic quantization implementation (FP16, INT8, INT4).

Universal quantization using PyTorch and HuggingFace Transformers.
Works across ALL platforms and devices:
- Mac (M1/M2/Intel) with CPU or Metal GPU
- Windows with CPU or CUDA/ROCm GPU
- Linux (x86/ARM) with CPU or CUDA/ROCm GPU
- Raspberry Pi (ARM, CPU-only)
- Jetson devices (ARM + CUDA)
- Any edge device

Supports multiple quantization backends:
- FP16: Works on all platforms (CPU, CUDA, Metal, etc.)
- INT8: Platform-specific (bitsandbytes on CUDA, PyTorch on CPU/Metal)
- INT4: Platform-specific (bitsandbytes on CUDA, not available on CPU/Metal)

References:
    - https://huggingface.co/docs/transformers/main_classes/quantization
    - https://pytorch.org/docs/stable/quantization.html
    - https://pytorch.org/docs/stable/backends.html
"""

from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ...models.model import ModelInfo
from ...models.provider import ProviderType
from ..models import QuantizationTask, QuantizationType, TaskStatus
from .base import BaseQuantizer
from .model_validator import ModelValidator
from .platform_detector import PlatformDetector, DeviceType, DeviceCapabilities


class GenericQuantizer(BaseQuantizer):
    """Cross-platform generic quantization for FP16, INT8, and INT4.

    Adapts to available hardware and backends across all platforms.
    """

    def __init__(self):
        """Initialize quantizer and detect platform capabilities."""
        super().__init__()
        self.capabilities: Optional[DeviceCapabilities] = None

    def _get_capabilities(self) -> DeviceCapabilities:
        """Get cached device capabilities."""
        if self.capabilities is None:
            self.capabilities = PlatformDetector.detect_all()
        return self.capabilities

    def check_availability(self) -> tuple[bool, str]:
        """Check if PyTorch and Transformers are available.

        Returns:
            (available, message)
        """
        try:
            import torch
            import transformers

            # Detect platform capabilities
            caps = self._get_capabilities()

            # Build status message
            messages = [
                f"PyTorch {torch.__version__}",
                f"Platform: {caps.platform.value}",
                f"Device: {caps.device_type.value}",
                f"RAM: {caps.available_ram_gb:.1f}GB/{caps.total_ram_gb:.1f}GB",
            ]

            if caps.has_gpu:
                messages.append(f"GPU: {caps.gpu_name}")

            # Quantization support
            quant_support = []
            if caps.supports_fp16:
                quant_support.append("FP16")
            if caps.supports_int8:
                quant_support.append("INT8")
            if caps.supports_int4:
                quant_support.append("INT4")

            messages.append(f"Supports: {', '.join(quant_support)}")

            # Backend info
            if "bitsandbytes" in caps.backends:
                messages.append("bitsandbytes available")
            elif caps.device_type == DeviceType.CUDA:
                messages.append("bitsandbytes NOT available (install for better INT8/INT4)")

            return True, " | ".join(messages)
        except ImportError as e:
            missing = "torch" if "torch" in str(e) else "transformers"
            return False, f"{missing} not installed. Run: pip install {missing}"

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported quantization types for current platform.

        Returns:
            List of supported types based on hardware capabilities
        """
        caps = self._get_capabilities()

        supported = []

        # FP16 works everywhere
        if caps.supports_fp16:
            supported.append(QuantizationType.FP16)

        # INT8 support varies by platform
        if caps.supports_int8:
            supported.append(QuantizationType.INT8)

        # INT4 only available on certain platforms
        if caps.supports_int4:
            supported.append(QuantizationType.INT4)

        return supported

    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for quantization.

        Generic quantization works with HuggingFace models and Ollama models.
        For Ollama/GGUF models, they need to be converted to FP16 first.

        Args:
            model_info: Model to quantize

        Returns:
            Path to model directory, or None if not compatible
        """
        # HuggingFace models - direct support
        if model_info.provider == ProviderType.HUGGINGFACE:
            # Check if model_id is a local path (from ./models directory or manual path)
            if model_info.model_id:
                model_path = Path(model_info.model_id)
                # If it's an absolute path and exists, it's a local model
                if model_path.exists() and model_path.is_absolute():
                    return model_path

            # model_id is HuggingFace ID (e.g., "Qwen/Qwen3-VL-4B-Instruct")
            # Return None - transformers will find it in cache
            return None

        # Ollama/GGUF models - use source_path (will be converted by orchestrator)
        elif model_info.provider in [ProviderType.OLLAMA, ProviderType.GGUF]:
            # The orchestrator will convert GGUF → FP16 GGUF
            # Then we load the FP16 GGUF with transformers
            if model_info.source_path:
                return model_info.source_path
            return None

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

        # Size reduction factors
        size_factors = {
            QuantizationType.FP16: 0.5,  # 50% of original
            QuantizationType.INT8: 0.25,  # 25% of original
            QuantizationType.INT4: 0.125,  # 12.5% of original
        }

        factor = size_factors.get(quant_type, 0.5)
        return original_size * factor

    def _validate_and_repair_model(self, model_path: Path) -> bool:
        """Validate model directory and attempt repairs.

        Args:
            model_path: Path to model directory

        Returns:
            True if model is valid or was repaired

        Raises:
            ValueError: If model is invalid and cannot be repaired
        """
        # Validate model directory
        is_valid, error_msg, validation_info = ModelValidator.validate_model_directory(model_path)

        if is_valid:
            logger.info(f"Model directory validation passed: {model_path}")
            return True

        # Model is invalid - try to repair
        logger.warning(f"Model validation failed, attempting auto-repair...")

        if ModelValidator.attempt_config_repair(model_path):
            # Re-validate after repair
            is_valid, error_msg, validation_info = ModelValidator.validate_model_directory(model_path)

            if is_valid:
                logger.info("Model successfully repaired!")
                return True

        # Could not repair - generate helpful error
        helpful_msg = ModelValidator.get_helpful_error_message(model_path, validation_info)
        raise ValueError(helpful_msg)

    def _get_optimal_device_config(
        self,
        torch_module,
        quant_type: QuantizationType
    ) -> dict:
        """Get optimal device configuration for current platform.

        Args:
            torch_module: Imported torch module
            quant_type: Quantization type

        Returns:
            Dictionary of device configuration kwargs
        """
        caps = self._get_capabilities()
        config = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        # Check for accelerate (optional but helpful)
        accelerate_available = "accelerate" in caps.backends

        # Check if we have enough RAM for device_map="auto"
        # device_map="auto" can offload to disk which causes "meta device" warnings
        low_memory = caps.available_ram_gb < 3.0

        # Device placement strategy
        if accelerate_available and caps.has_gpu and not low_memory:
            # Best case: accelerate available with GPU and sufficient RAM
            config["device_map"] = "auto"
            logger.info(f"Using device_map='auto' with accelerate on {caps.device_type.value}")

        elif caps.device_type == DeviceType.CUDA:
            # CUDA without accelerate or with low memory
            device = "cuda:0"
            if quant_type in [QuantizationType.INT8, QuantizationType.INT4]:
                config["device_map"] = {"": device}
            else:
                config["_use_explicit_device"] = device
            logger.info(f"Using CUDA device: {device}")

            if low_memory:
                logger.warning(f"Low RAM detected ({caps.available_ram_gb:.1f}GB) - using explicit device placement")

        elif caps.device_type == DeviceType.METAL:
            # Apple Silicon with MPS (Metal Performance Shaders)
            # Don't use device_map="auto" on MPS - use explicit device
            device = "mps"
            config["_use_explicit_device"] = device
            logger.info("Using Apple Metal GPU (MPS)")

            if low_memory:
                logger.warning(f"Low RAM detected ({caps.available_ram_gb:.1f}GB) - quantization may be slow")

        else:
            # CPU fallback (works on ALL platforms)
            device = "cpu"
            config["_use_explicit_device"] = device
            logger.info(f"Using CPU on {caps.platform.value}")

            # Optimize for edge devices or low memory
            if caps.is_edge_device or low_memory:
                logger.info("Memory-constrained environment detected - using optimized settings")

        return config

    def _can_use_bitsandbytes(
        self,
        quant_type: QuantizationType
    ) -> bool:
        """Check if bitsandbytes can be used for quantization.

        Args:
            quant_type: Quantization type

        Returns:
            True if bitsandbytes should be used
        """
        if quant_type not in [QuantizationType.INT8, QuantizationType.INT4]:
            return False

        caps = self._get_capabilities()

        # bitsandbytes only works on CUDA currently
        if caps.device_type != DeviceType.CUDA:
            return False

        # Check if bitsandbytes is available
        return "bitsandbytes" in caps.backends

    def _load_model_with_fallbacks(
        self,
        model_path_str: str,
        quant_type: QuantizationType,
        torch_module
    ):
        """Load model with platform-specific strategies.

        Supports all platforms: Mac (M1/Intel), Windows, Linux, ARM devices.

        Args:
            model_path_str: Path to model directory
            quant_type: Quantization type
            torch_module: Imported torch module

        Returns:
            Loaded model

        Raises:
            Exception: If loading fails
        """
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoConfig,
        )

        caps = self._get_capabilities()

        # Prepare loading kwargs
        load_kwargs = self._get_optimal_device_config(torch_module, quant_type)

        # Extract explicit device if set
        explicit_device = load_kwargs.pop("_use_explicit_device", None)

        # Configure quantization based on backend
        if quant_type == QuantizationType.FP16:
            load_kwargs["torch_dtype"] = torch_module.float16
            logger.info("Quantizing to FP16 (works on all platforms)")

        elif quant_type == QuantizationType.INT8:
            if self._can_use_bitsandbytes(quant_type):
                # Use bitsandbytes on CUDA
                load_kwargs["load_in_8bit"] = True
                logger.info("Quantizing to INT8 using bitsandbytes (CUDA)")
            else:
                # INT8 without bitsandbytes: Not supported for persistent quantization
                # PyTorch dynamic quantization can't be saved to disk
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"INT8 QUANTIZATION REQUIRES CUDA + BITSANDBYTES\n"
                    f"{'='*60}\n\n"
                    f"Your System:\n"
                    f"  Platform: {caps.platform.value}\n"
                    f"  Device: {caps.device_type.value}\n"
                    f"  RAM: {caps.available_ram_gb:.1f}GB available\n"
                    f"  CUDA: Not available\n"
                    f"  bitsandbytes: {'installed' if 'bitsandbytes' in caps.backends else 'not installed'}\n\n"
                    f"INT8 quantization requires:\n"
                    f"  • CUDA GPU (NVIDIA) - Metal/CPU not supported\n"
                    f"  • bitsandbytes library\n\n"
                    f"Why CPU/Metal INT8 doesn't work:\n"
                    f"  • PyTorch's torch.quantization.quantize_dynamic() is runtime-only\n"
                    f"  • Cannot be saved with save_pretrained()\n"
                    f"  • Model reverts to FP32 when saving\n"
                    f"  • bitsandbytes INT8 requires CUDA (not available on Mac)\n\n"
                    f"✓ Available options on Mac/CPU:\n"
                    f"  • FP16 quantization (50% size reduction, works everywhere)\n"
                    f"  • GGUF quantization (for supported models, various bit depths)\n\n"
                    f"For INT8/INT4:\n"
                    f"  • Use GGUF quantization (Q8_0, Q4_K_M, etc.)\n"
                    f"  • Or get a CUDA GPU + install bitsandbytes\n"
                    f"{'='*60}\n"
                )

        elif quant_type == QuantizationType.INT4:
            if self._can_use_bitsandbytes(quant_type):
                # Use bitsandbytes on CUDA
                load_kwargs["load_in_4bit"] = True
                logger.info("Quantizing to INT4 using bitsandbytes (CUDA)")
            else:
                # INT4 not available without bitsandbytes
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"INT4 QUANTIZATION REQUIRES CUDA + BITSANDBYTES\n"
                    f"{'='*60}\n\n"
                    f"Your System:\n"
                    f"  Platform: {caps.platform.value}\n"
                    f"  Device: {caps.device_type.value}\n"
                    f"  RAM: {caps.available_ram_gb:.1f}GB available\n"
                    f"  CUDA: Not available\n"
                    f"  bitsandbytes: {'installed' if 'bitsandbytes' in caps.backends else 'not installed'}\n\n"
                    f"INT4 quantization requires:\n"
                    f"  • CUDA GPU (NVIDIA) - Metal/CPU not supported\n"
                    f"  • bitsandbytes library\n\n"
                    f"Why CPU/Metal INT4 doesn't work:\n"
                    f"  • PyTorch does not support persistent INT4 quantization on CPU\n"
                    f"  • bitsandbytes INT4 requires CUDA (not available on Mac)\n\n"
                    f"✓ Available options on Mac/CPU:\n"
                    f"  • FP16 quantization (50% size reduction, works everywhere)\n"
                    f"  • GGUF quantization (Q4_K_M, Q4_K_S, Q5_K_M, etc.)\n\n"
                    f"For INT4/INT8:\n"
                    f"  • Use GGUF quantization (supports Q4_K_M, Q8_0, etc.)\n"
                    f"  • Or get a CUDA GPU + install bitsandbytes\n"
                    f"{'='*60}\n"
                )

        # Try multiple model classes for maximum compatibility (LLMs and VLMs)
        model = None
        model_strategies = [
            ('AutoModelForCausalLM', 'Most LLMs and some VLMs'),
            ('AutoModelForVision2Seq', 'Modern VLMs (Qwen2-VL, Idefics, etc.)'),
            ('AutoModel', 'Generic fallback'),
        ]
        
        last_error = None
        for model_class_name, description in model_strategies:
            try:
                logger.info(f"Loading with {model_class_name} ({description})...")
                
                if model_class_name == 'AutoModelForVision2Seq':
                    try:
                        from transformers import AutoModelForVision2Seq
                        model = AutoModelForVision2Seq.from_pretrained(model_path_str, **load_kwargs)
                    except ImportError:
                        logger.debug("AutoModelForVision2Seq not available in this transformers version")
                        continue
                elif model_class_name == 'AutoModelForCausalLM':
                    model = AutoModelForCausalLM.from_pretrained(model_path_str, **load_kwargs)
                else:  # AutoModel
                    model = AutoModel.from_pretrained(model_path_str, **load_kwargs)

                # Move to explicit device if needed
                if explicit_device is not None:
                    logger.info(f"Moving model to {explicit_device}")
                    model = model.to(explicit_device)

                # Verify model has generate method (important for VLMs)
                if not hasattr(model, 'generate'):
                    logger.warning(f"{model_class_name} loaded but has no .generate() method")
                    logger.info("This model class may not support generation, trying next...")
                    model = None
                    continue

                logger.info(f"✓ Successfully loaded model with {model_class_name}")
                break

            except Exception as e:
                logger.debug(f"{model_class_name} failed: {str(e)[:200]}")
                last_error = e
                continue
        
        if model is None:
            raise RuntimeError(
                f"Failed to load model with any strategy.\n\n"
                f"Platform: {caps.platform.value}\n"
                f"Device: {caps.device_type.value}\n"
                f"Tried: {', '.join([s[0] for s in model_strategies])}\n\n"
                f"Last error: {str(last_error)[:200]}\n\n"
                f"Try:\n"
                f"  1. Check model files are complete\n"
                f"  2. Ensure enough RAM ({caps.available_ram_gb:.1f}GB available)\n"
                f"  3. Try FP16 instead of INT8/INT4 if memory limited\n"
            )

        return model

    def _detect_vlm_components(self, model):
        """Detect vision and language components in a VLM.

        Args:
            model: Loaded model

        Returns:
            tuple: (vision_module, language_module, has_projection)
        """
        vision = None
        language = None
        has_projection = False

        # Common VLM vision encoder attributes
        vision_attrs = ['visual', 'vision_model', 'vision_tower', 'image_encoder', 'vision_encoder']
        for attr in vision_attrs:
            if hasattr(model, attr):
                vision = getattr(model, attr)
                logger.info(f"Detected vision encoder: model.{attr}")
                break

        # Common VLM language decoder attributes
        language_attrs = ['language_model', 'text_model', 'lm_head', 'model', 'transformer']
        for attr in language_attrs:
            if hasattr(model, attr):
                language = getattr(model, attr)
                logger.info(f"Detected language decoder: model.{attr}")
                break

        # Check for projection layer
        projection_attrs = ['mm_projector', 'vision_projection', 'text_projection', 'projector']
        for attr in projection_attrs:
            if hasattr(model, attr):
                has_projection = True
                logger.info(f"Detected projection layer: model.{attr}")
                break

        return vision, language, has_projection

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Execute generic quantization.

        Args:
            task: Quantization task
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful
        """
        try:
            import torch
            from transformers import AutoTokenizer

            task.status = TaskStatus.RUNNING
            logger.info(f"Starting {task.quant_type.display_name} quantization")

            # Update progress: Validating model
            task.stage = "Initializing"
            task.substage = "Validating model files"
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

                # Validate and repair model directory for HF models only
                # Skip validation for GGUF files (from Ollama/GGUF providers)
                if model_path.suffix != '.gguf':
                    logger.info(f"Validating model directory: {model_path}")
                    self._validate_and_repair_model(model_path)
                else:
                    logger.info(f"GGUF file detected, skipping HF validation: {model_path}")
            else:
                # Use model_id (HF will find cached model or download)
                model_identifier = task.model_info.model_id
                logger.info(f"Quantizing model using HF model ID: {model_identifier}")
                # Skip validation for HF model IDs (transformers handles it)

            # Update progress: Loading model
            task.stage = "Loading"
            task.substage = "Loading model from disk"
            if progress_callback:
                progress_callback(10.0, None)
            task.progress = 10.0

            logger.info(f"Loading model from {model_identifier}")

            # CRITICAL: Validate quantization type is supported on this platform
            if task.quant_type in [QuantizationType.INT8, QuantizationType.INT4]:
                has_cuda = torch.cuda.is_available()
                if not has_cuda:
                    error_msg = (
                        f"\n{'='*60}\n"
                        f"{task.quant_type.display_name} QUANTIZATION REQUIRES CUDA + BITSANDBYTES\n"
                        f"{'='*60}\n\n"
                        f"Your System:\n"
                        f"  Platform: {device_caps.platform.value}\n"
                        f"  Device: {device_caps.device_type.value}\n"
                        f"  RAM: {device_caps.available_ram_gb:.1f}GB available\n"
                        f"  CUDA: Not available\n"
                        f"  bitsandbytes: {'installed' if device_caps.backends.get('bitsandbytes') else 'not installed'}\n\n"
                        f"{task.quant_type.display_name} quantization requires:\n"
                        f"  • CUDA GPU (NVIDIA) - Metal/CPU not supported\n"
                        f"  • bitsandbytes library\n\n"
                        f"Why CPU/Metal {task.quant_type.display_name} doesn't work:\n"
                        f"  • PyTorch does not support persistent {task.quant_type.display_name} quantization on CPU\n"
                        f"  • bitsandbytes {task.quant_type.display_name} requires CUDA (not available on Mac)\n\n"
                        f"✓ Available options on Mac/CPU:\n"
                        f"  • FP16 quantization (50% size reduction, works everywhere)\n"
                        f"  • GGUF quantization (Q4_K_M, Q4_K_S, Q5_K_M, etc.)\n"
                        f"  • MLX quantization (4-bit, Apple Silicon only, pip install mlx mlx-lm mlx-vlm)\n\n"
                        f"For {task.quant_type.display_name}:\n"
                        f"  • Use GGUF quantization (supports Q4_K_M, Q8_0, etc.)\n"
                        f"  • Use MLX quantization (4-bit for Mac)\n"
                        f"  • Or get a CUDA GPU + install bitsandbytes\n"
                        f"{'='*60}\n"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            # Import output suppressor for clean TUI
            from ...utils.output_suppressor import suppress_transformers_output

            # Load model with fallback strategies (suppress library output)
            logger.info(f"Loading model in {task.quant_type.display_name} format")
            with suppress_transformers_output():
                model = self._load_model_with_fallbacks(
                    model_identifier,
                    task.quant_type,
                    torch
                )

            # VLM Component-Level Quantization Support
            if task.vlm_components:
                logger.info(f"VLM component-level quantization requested: {task.vlm_components}")

                # Detect VLM components
                vision, language, has_projection = self._detect_vlm_components(model)

                if task.vlm_components == "language" and language is not None:
                    logger.info("✓ Component-level: Quantizing LANGUAGE DECODER only")
                    logger.info("Vision encoder will remain in FULL PRECISION (FP16/FP32)")
                    logger.info("This is OPTIMAL for VLMs: smaller size, preserved vision quality!")
                    
                    # Quantize only the language component
                    if task.quant_type == QuantizationType.FP16:
                        logger.info("Converting language decoder to FP16...")
                        language.to(torch.float16)
                        # Vision encoder stays in its original dtype
                        logger.info("✓ Language decoder: FP16")
                        logger.info("✓ Vision encoder: Full precision (unchanged)")
                    
                    elif task.quant_type in [QuantizationType.INT8, QuantizationType.INT4]:
                        # For INT8/INT4, we need to selectively quantize
                        logger.info(f"Quantizing language decoder to {task.quant_type.display_name}...")
                        logger.warning("INT8/INT4 component-level quantization requires BitsAndBytes on CUDA")
                        logger.warning("For now, quantizing entire model. Use FP16 for true component-level.")
                        # Fall through to normal quantization below
                
                elif task.vlm_components == "vision" and vision is not None:
                    logger.info("✓ Component-level: Quantizing VISION ENCODER only")
                    logger.info("Language decoder will remain in FULL PRECISION")
                    logger.info("NOTE: This is less common - usually language is quantized instead")
                    
                    # Quantize only the vision component
                    if task.quant_type == QuantizationType.FP16:
                        logger.info("Converting vision encoder to FP16...")
                        vision.to(torch.float16)
                        logger.info("✓ Vision encoder: FP16")
                        logger.info("✓ Language decoder: Full precision (unchanged)")
                    else:
                        logger.warning("INT8/INT4 vision-only quantization not recommended")
                        logger.info("Falling back to standard quantization")

                elif task.vlm_components == "both" or (vision and language):
                    logger.info("Component-level: Quantizing BOTH components (standard VLM quantization)")
                    # Quantize entire model (current behavior - falls through)

                else:
                    logger.warning(f"Could not detect VLM components for {task.vlm_components} quantization")
                    logger.info("Falling back to standard quantization (entire model)")

            # Update progress: Model loaded, now quantizing
            task.stage = "Quantizing"
            task.substage = f"Converting to {task.quant_type.display_name}"
            if progress_callback:
                progress_callback(50.0, None)
            task.progress = 50.0

            # Add intermediate progress updates during quantization
            if progress_callback:
                progress_callback(60.0, None)
            task.progress = 60.0

            # Load tokenizer/processor (suppress library output)
            # For VLMs, we need processor (tokenizer + image_processor)
            # For LLMs, we just need tokenizer
            task.substage = "Loading tokenizer/processor"
            logger.info("Detecting model type and loading appropriate tokenizer/processor...")
            
            # Check if model is a VLM by inspecting architecture
            is_vlm = False
            processor_or_tokenizer = None
            last_error = None
            
            # Try to detect VLM from model's config
            try:
                config = model.config
                arch = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else ""
                
                # VLM patterns
                vlm_patterns = [
                    "ForConditionalGeneration", "VisionTextDual", "Llava", "Blip",
                    "Qwen2VL", "Qwen3VL", "QwenVL", "InstructBlip", "Idefics"
                ]
                is_vlm = any(pattern in arch for pattern in vlm_patterns)
                
                # Also check for vision config
                if not is_vlm:
                    config_dict = config.to_dict()
                    is_vlm = any(key in config_dict for key in ["vision_config", "visual_config", "image_encoder"])
                
                logger.info(f"Model architecture: {arch}, VLM: {is_vlm}")
            except Exception as e:
                logger.debug(f"Could not detect VLM from config: {e}")
            
            # Load processor for VLMs, tokenizer for LLMs
            if is_vlm:
                # Try loading processor for VLM
                try:
                    logger.info("Loading processor for VLM...")
                    from transformers import AutoProcessor
                    
                    with suppress_transformers_output():
                        processor_or_tokenizer = AutoProcessor.from_pretrained(
                            model_identifier,
                            trust_remote_code=True
                        )
                    logger.info("✓ Processor loaded successfully (VLM)")
                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to load processor for VLM, trying tokenizer: {e}")
                    is_vlm = False  # Fall back to tokenizer
            
            if not is_vlm:
                # Load tokenizer for LLM (or VLM fallback)
                logger.info("Loading tokenizer...")
                try:
                    logger.debug(f"Trying to load tokenizer with trust_remote_code from: {model_identifier}")
                    with suppress_transformers_output():
                        processor_or_tokenizer = AutoTokenizer.from_pretrained(
                            model_identifier,
                            trust_remote_code=True
                        )
                    logger.info("Tokenizer loaded successfully with trust_remote_code")
                except Exception as e:
                    last_error = e
                    logger.warning(f"Could not load tokenizer with trust_remote_code: {e}")

                    # Fallback: try without trust_remote_code
                    try:
                        logger.debug("Attempting to load tokenizer without trust_remote_code...")
                        with suppress_transformers_output():
                            processor_or_tokenizer = AutoTokenizer.from_pretrained(model_identifier)
                        logger.info("Tokenizer loaded successfully without trust_remote_code")
                    except Exception as e2:
                        last_error = e2
                        logger.error(f"Tokenizer loading failed (second attempt): {e2}")

            if processor_or_tokenizer is None:
                # All attempts failed - provide detailed error
                error_msg = (
                    f"Could not load tokenizer/processor from {model_identifier}\n\n"
                    f"Last error: {str(last_error)}\n\n"
                    f"Try checking the model identifier is correct."
                )

                raise RuntimeError(error_msg)

            # Update progress: Saving
            task.stage = "Saving"
            task.substage = "Writing model to disk"
            if progress_callback:
                progress_callback(75.0, None)
            task.progress = 75.0

            # Create output directory
            task.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save quantized model
            logger.info(f"Saving quantized model to {task.output_path}")
            model.save_pretrained(task.output_path)

            # Update progress during save
            task.substage = f"Writing {'processor' if is_vlm else 'tokenizer'}"
            if progress_callback:
                progress_callback(90.0, None)
            task.progress = 90.0
            
            # Save processor (for VLMs) or tokenizer (for LLMs)
            processor_or_tokenizer.save_pretrained(task.output_path)
            
            if is_vlm:
                logger.info("✓ Saved VLM processor (includes image_processor + tokenizer)")
            else:
                logger.info("✓ Saved tokenizer")

            # Calculate estimated size for display
            if task.output_path.exists():
                if task.output_path.is_dir():
                    total_bytes = sum(f.stat().st_size for f in task.output_path.rglob("*") if f.is_file())
                    task.estimated_size_gb = total_bytes / (1024 ** 3)

            # Update progress: Complete
            task.stage = "Completed"
            task.substage = None
            if progress_callback:
                progress_callback(100.0, 0)
            task.progress = 100.0
            task.status = TaskStatus.COMPLETED

            logger.info(f"Quantization completed successfully: {task.output_path}")
            return True

        except ImportError as e:
            error_msg = f"Missing dependency: {str(e)}"
            logger.error(error_msg)
            task.status = TaskStatus.FAILED
            task.error = error_msg
            return False

        except Exception as e:
            error_msg = f"Quantization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = error_msg
            return False
