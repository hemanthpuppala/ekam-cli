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
from typing import Callable, Optional, Tuple
import re

from loguru import logger

from ...models.model import ModelInfo
from ...models.provider import ProviderType
from ...models.endpoints import ModelType
from ..models import QuantizationTask, QuantizationType, TaskStatus
from .base import BaseQuantizer
from .model_validator import ModelValidator
from .platform_detector import PlatformDetector, DeviceType, DeviceCapabilities


def _extract_model_metadata_early(
    task: QuantizationTask,
    model_identifier: str,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Extract model metadata EARLY without loading the full model.

    This uses information already available in task.model_info (set at discovery time)
    plus lightweight config loading to determine model requirements before expensive
    model loading.

    Args:
        task: Quantization task with model_info metadata
        model_identifier: Path to model or HF model ID

    Returns:
        (is_vlm, model_type_str, model_architecture)
        - is_vlm: True if model has vision capabilities, False otherwise
        - model_type_str: Model type from config (e.g., "gemma3", "llama2")
        - model_architecture: Architecture string from model_info.architecture or detected
    """
    is_vlm = False
    model_type_str = None
    model_architecture = None

    # PHASE 1: Use metadata already available from discovery time
    # This is extremely efficient - no model loading needed
    logger.debug(f"[Metadata] Using pre-discovered model type: {task.model_info.model_type}")

    # Know if VLM from task metadata (set during model discovery)
    is_vlm = task.model_info.model_type == ModelType.VLM
    logger.debug(f"[Metadata] Model is VLM: {is_vlm} (from discovery metadata)")

    # Get architecture from model_info if available
    if task.model_info.architecture:
        model_architecture = task.model_info.architecture
        logger.debug(f"[Metadata] Using pre-discovered architecture: {model_architecture}")
    else:
        logger.debug("[Metadata] Architecture not in model_info, will extract from config")

    # PHASE 2: Load lightweight config (JSON only, not tensors)
    # This is fast and gives us model_type for tokenizer discovery
    try:
        from transformers import AutoConfig

        logger.debug(f"[Metadata] Loading config from {model_identifier}...")
        config = AutoConfig.from_pretrained(
            model_identifier,
            trust_remote_code=True,
        )

        # Get model_type for tokenizer discovery
        if hasattr(config, "model_type"):
            model_type_str = config.model_type
            logger.debug(f"[Metadata] Detected model_type from config: {model_type_str}")

        # Verify/update architecture if not in model_info
        if not model_architecture and hasattr(config, "architectures"):
            architectures = config.architectures if config.architectures else []
            if architectures:
                model_architecture = architectures[0]
                logger.debug(f"[Metadata] Detected architecture from config: {model_architecture}")

        # Double-check VLM status from config (in case discovery missed it)
        if not is_vlm:
            config_dict = config.to_dict()
            vision_indicators = any(
                key in config_dict for key in ["vision_config", "visual_config", "image_encoder"]
            )
            if vision_indicators:
                is_vlm = True
                logger.debug("[Metadata] Detected VLM from vision config in model config")

    except Exception as e:
        logger.warning(f"[Metadata] Could not load config: {str(e)[:100]}")
        logger.debug(f"[Metadata] Will proceed with available metadata")

    return is_vlm, model_type_str, model_architecture


def _find_canonical_tokenizer_model(model_type: str) -> Optional[str]:
    """
    Dynamically find a canonical tokenizer model for a given model_type.

    This function implements a universal, future-proof strategy for finding
    tokenizer models without hardcoding every variant. Works across all model types.

    Strategy (in order of preference):
    1. Try model_type directly (works for: gpt2, phi, bloom, etc.)
    2. Try with common organization prefixes (google/, meta-llama/, Qwen/, etc.)
    3. Try with common size suffixes (-7b, -13b, -base, etc.)
    4. Search HuggingFace Hub for models with that model_type (by downloads)
    5. Try transformers library detection with offline fallback

    Works offline: If network is unavailable, uses local cached tokenizers from transformers.

    Args:
        model_type: The model type string (e.g., "gemma3", "llama", "qwen2")

    Returns:
        A HuggingFace model identifier string, or None if not found
    """
    from transformers import AutoTokenizer

    logger.debug(f"[Tokenizer Discovery] Starting dynamic search for model_type: {model_type}")

    # Strategy 1: Try model_type directly
    # Some models are already valid HF model IDs (gpt2, phi, bloom, etc.)
    try:
        logger.debug(f"[Strategy 1] Trying direct model_type: {model_type}")
        AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
        logger.info(f"[Tokenizer Discovery] ✓ Found tokenizer directly: {model_type}")
        return model_type
    except Exception as e:
        logger.debug(f"[Strategy 1] Failed for {model_type}: {str(e)[:100]}")

    # Strategy 2: Try with common organization prefixes
    # Order matters: try common ones first
    org_prefixes = [
        ("google/", "Google models"),
        ("meta-llama/", "Meta Llama models"),
        ("mistralai/", "Mistral models"),
        ("Qwen/", "Qwen models"),
        ("microsoft/", "Microsoft models"),
        ("facebook/", "Facebook models"),
        ("stabilityai/", "Stability AI models"),
        ("openai/", "OpenAI models"),
    ]

    for prefix, description in org_prefixes:
        # Capitalize first letter for consistency with HF naming
        capitalized = model_type[0].upper() + model_type[1:] if model_type else ""
        candidates = [
            f"{prefix}{capitalized}",
            f"{prefix}{model_type}",
            f"{prefix}{model_type.replace('_', '-')}",
        ]

        for candidate in candidates:
            try:
                logger.debug(f"[Strategy 2] Trying {description}: {candidate}")
                AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
                logger.info(f"[Tokenizer Discovery] ✓ Found tokenizer with org prefix: {candidate}")
                return candidate
            except Exception as e:
                logger.debug(f"[Strategy 2] Failed for {candidate}: {str(e)[:100]}")

    # Strategy 3: Try with common size suffixes (-7b, -13b, -base, etc.)
    size_suffixes = ["-7b", "-13b", "-base", "-small", "-medium", "-large", "-7b-hf", "-13b-hf"]

    for suffix in size_suffixes:
        for prefix, _ in org_prefixes:
            capitalized = model_type[0].upper() + model_type[1:] if model_type else ""
            candidates = [
                f"{prefix}{capitalized}{suffix}",
                f"{prefix}{model_type}{suffix}",
            ]

            for candidate in candidates:
                try:
                    logger.debug(f"[Strategy 3] Trying with size suffix: {candidate}")
                    AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
                    logger.info(f"[Tokenizer Discovery] ✓ Found tokenizer with suffix: {candidate}")
                    return candidate
                except Exception as e:
                    logger.debug(f"[Strategy 3] Failed for {candidate}: {str(e)[:100]}")

    # Strategy 4: Search HuggingFace Hub API (requires internet)
    try:
        from huggingface_hub import list_models

        logger.debug(f"[Strategy 4] Searching HuggingFace Hub for model_type '{model_type}'...")

        # Search for models matching the model_type
        models = list_models(
            search=model_type,
            library_name="transformers",
            sort="downloads",
            direction=-1,
            limit=10  # Get top 10
        )

        if models:
            for model_info in models:
                try:
                    candidate = model_info.id
                    logger.debug(
                        f"[Strategy 4] Trying HF Hub result: {candidate} (downloads: {model_info.downloads})"
                    )
                    AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
                    logger.info(f"[Tokenizer Discovery] ✓ Found via HF Hub search: {candidate}")
                    return candidate
                except Exception as e:
                    logger.debug(f"[Strategy 4] Failed for {candidate}: {str(e)[:100]}")
    except Exception as e:
        logger.debug(f"[Strategy 4] HF Hub search failed (offline or API error): {str(e)[:60]}")

    # Strategy 5: Use well-known canonical models from transformers library
    # These are cached locally and work offline
    logger.debug("[Strategy 5] Using canonical models from transformers library (works offline)...")

    # Map model types to well-known canonical HF models that have cached tokenizers
    canonical_by_type = {
        "bert": "google-bert/bert-base-uncased",
        "roberta": "FacebookAI/roberta-base",
        "gpt2": "gpt2",
        "t5": "google-t5/t5-base",
        "llama": "meta-llama/Llama-2-7b",
        "llama2": "meta-llama/Llama-2-7b",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "qwen": "Qwen/Qwen2-7B",
        "qwen2": "Qwen/Qwen2-7B",
        "gemma": "google/gemma-7b",
        "gemma2": "google/gemma2-9b",
        "gemma3": "google/gemma-7b",  # Gemma3 uses same tokenizer as Gemma
        "phi": "microsoft/phi-2",
        "phi2": "microsoft/phi-2",
        "bloom": "bigscience/bloom",
        "falcon": "tiiuae/falcon-7b",
        "mpt": "mosaicml/mpt-7b",
    }

    # Try exact match first
    if model_type.lower() in canonical_by_type:
        canonical_model = canonical_by_type[model_type.lower()]
        try:
            logger.debug(f"[Strategy 5] Trying canonical model for {model_type}: {canonical_model}")
            AutoTokenizer.from_pretrained(canonical_model, trust_remote_code=True)
            logger.info(f"[Tokenizer Discovery] ✓ Using canonical model: {canonical_model}")
            return canonical_model
        except Exception as e:
            logger.debug(f"[Strategy 5] Failed for {canonical_model}: {str(e)[:100]}")

    # Try partial match (e.g., "gemma3" matches "gemma" key)
    for base_type, canonical_model in canonical_by_type.items():
        if base_type in model_type.lower() or model_type.lower() in base_type:
            try:
                logger.debug(
                    f"[Strategy 5] Trying canonical model (partial match {base_type}): {canonical_model}"
                )
                AutoTokenizer.from_pretrained(canonical_model, trust_remote_code=True)
                logger.info(f"[Tokenizer Discovery] ✓ Using canonical model: {canonical_model}")
                return canonical_model
            except Exception as e:
                logger.debug(f"[Strategy 5] Failed for {canonical_model}: {str(e)[:100]}")

    logger.warning(f"[Tokenizer Discovery] Could not find tokenizer for model_type '{model_type}' using any strategy")
    return None


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
            # Check if this is a GGUF file (from Ollama conversion)
            is_gguf = model_path_str.endswith('.gguf')

            if is_gguf:
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"GENERIC QUANTIZATION FROM OLLAMA NOT YET SUPPORTED\n"
                    f"{'='*60}\n\n"
                    f"Good news: Your Ollama model was successfully converted to FP16!\n"
                    f"📁 Intermediate file saved: {model_path_str}\n\n"
                    f"However: transformers cannot load GGUF files for further quantization.\n"
                    f"GGUF is designed for inference (llama.cpp), not quantization pipelines.\n\n"
                    f"✅ What you can do:\n"
                    f"  1. Use the FP16 GGUF file with llama.cpp or Ollama\n"
                    f"  2. Try GGUF requantization (option 3) instead\n"
                    f"  3. Download the original HuggingFace model for Generic quantization\n\n"
                    f"💡 Recommended: Use GGUF requantization for Ollama models\n"
                    f"   Example: Q8_0 → Q5_K_S, Q4_K_M → Q3_K_S\n"
                    f"{'='*60}\n"
                )
            else:
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
            # For Ollama models, orchestrator will provide converted HF directory path
            model_path = self.get_source_model_path(task.model_info)
            if model_path and model_path.exists():
                # Local path exists - use it
                model_identifier = str(model_path)
                logger.info(f"Quantizing model from local path: {model_path}")

                # Validate and repair model directory
                # Note: Ollama models are converted to HF format by orchestrator before reaching here
                if model_path.is_dir():
                    logger.info(f"Validating model directory: {model_path}")
                    self._validate_and_repair_model(model_path)
                else:
                    logger.info(f"Model file: {model_path}")
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

                # For converted Ollama models (from GGUF→HF), the tokenizer won't be in the directory
                # So we try multiple strategies:
                tokenizer_sources = []

                # Strategy 1: Direct path (HF model with tokenizer files)
                tokenizer_sources.append((model_identifier, "Local model directory"))

                # Strategy 2: Dynamic discovery - use the new universal tokenizer finder
                # This works for ANY model type (gemma, llama, qwen, etc.)
                try:
                    if Path(model_identifier).is_dir():
                        # This is a local directory - check if it has tokenizer files
                        dir_path = Path(model_identifier)
                        has_tokenizer_files = any(
                            (dir_path / f).exists()
                            for f in ['tokenizer.json', 'tokenizer.model', 'vocab.json', 'tokenizer_config.json']
                        )

                        if not has_tokenizer_files:
                            # Use dynamic tokenizer discovery
                            try:
                                config = model.config
                                model_type = config.model_type if hasattr(config, 'model_type') else None

                                if model_type:
                                    logger.debug(f"No tokenizer files found, using dynamic discovery for model_type: {model_type}")
                                    canonical_model = _find_canonical_tokenizer_model(model_type)

                                    if canonical_model:
                                        tokenizer_sources.append((canonical_model, f"Dynamically discovered tokenizer for {model_type}"))
                                        logger.debug(f"Will try discovered tokenizer: {canonical_model}")
                            except Exception as e:
                                logger.debug(f"Could not detect architecture for tokenizer discovery: {e}")
                except Exception as e:
                    logger.debug(f"Error checking for tokenizer files: {e}")

                # Try each tokenizer source
                for tokenizer_source, description in tokenizer_sources:
                    try:
                        logger.debug(f"Attempting to load tokenizer from: {description} ({tokenizer_source})")
                        with suppress_transformers_output():
                            processor_or_tokenizer = AutoTokenizer.from_pretrained(
                                tokenizer_source,
                                trust_remote_code=True
                            )
                        logger.info(f"✓ Tokenizer loaded successfully from: {description}")
                        break
                    except Exception as e:
                        last_error = e
                        logger.debug(f"Failed to load from {tokenizer_source}: {e}")
                        continue

                # If all attempts failed
                if processor_or_tokenizer is None:
                    logger.error(f"Could not load tokenizer from any source")

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
            # For GGUF-related errors, preserve the full helpful message
            error_str = str(e)
            if "GENERIC QUANTIZATION FROM OLLAMA NOT YET SUPPORTED" in error_str:
                # Don't wrap GGUF error messages - they're already formatted
                error_msg = error_str
            else:
                # Wrap other errors
                error_msg = f"Quantization failed: {error_str}"

            logger.error(error_msg, exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = error_msg
            return False
