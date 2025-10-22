"""Quantized models provider - discovers models from results/quantizations/."""

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from ..models.endpoints import CompatibilityStatus, ModelType, ProviderType
from ..models.model import ModelInfo
from ..models.model_cache import ModelMetadataCache
from ..models.provider import ProviderConfig
from ..models.system import SystemSpecs
from .base import BaseProvider


class QuantizedProvider(BaseProvider):
    """Provider for quantized models.

    Discovers and manages models quantized by the built-in quantization feature.
    All quantized models are stored in results/quantizations/ with metadata.
    """

    def __init__(self, config: ProviderConfig, system_specs: Optional[SystemSpecs] = None):
        """Initialize QuantizedProvider.

        Args:
            config: Provider configuration
            system_specs: System specifications for compatibility checking
        """
        self.config = config
        self.system_specs = system_specs or SystemSpecs.detect()
        self.quantized_dir = Path(config.models_dir or "results/quantizations")
        self.quantized_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata cache for efficient model inspection
        self.metadata_cache = ModelMetadataCache()
        logger.debug("Initialized model metadata cache")

        logger.info(f"QuantizedProvider initialized: {self.quantized_dir}")

    def discover_models(self) -> list[ModelInfo]:
        """Discover all quantized models.

        Scans results/quantizations/ for:
        - .gguf files (GGUF quantized models)
        - .safetensors directories (HF format quantized models)
        - Reads metadata files for quantization info

        Returns:
            List of discovered quantized models
        """
        models = []

        if not self.quantized_dir.exists():
            logger.info("Quantized models directory does not exist yet")
            return models

        # Discover GGUF files (exclude only macOS resource forks)
        for gguf_file in self.quantized_dir.glob("*.gguf"):
            # Skip macOS resource forks (._* files)
            if gguf_file.name.startswith("._"):
                continue

            try:
                model_info = self._create_model_info_from_gguf(gguf_file)
                if model_info:
                    models.append(model_info)
            except Exception as e:
                logger.error(f"Failed to process GGUF file {gguf_file}: {e}")

        # Discover HF format directories (contain .safetensors files)
        for item in self.quantized_dir.iterdir():
            if item.is_dir() and list(item.glob("*.safetensors")):
                try:
                    model_info = self._create_model_info_from_hf(item)
                    if model_info:
                        models.append(model_info)
                except Exception as e:
                    logger.error(f"Failed to process HF model {item}: {e}")

        logger.info(f"Discovered {len(models)} quantized models")
        return models

    def _create_model_info_from_gguf(self, gguf_path: Path) -> Optional[ModelInfo]:
        """Create ModelInfo from GGUF file using metadata cache.

        Args:
            gguf_path: Path to .gguf file

        Returns:
            ModelInfo or None if invalid
        """
        # Get metadata from cache (reads GGUF header and metadata file)
        cached_metadata = self.metadata_cache.get_metadata(str(gguf_path), provider="quantized")

        size_gb = gguf_path.stat().st_size / (1024 ** 3)

        # Check if this is an FP16 intermediate file
        is_fp16 = gguf_path.stem.endswith("_f16")

        # Try to read metadata file for original model info
        metadata_file = gguf_path.with_suffix(".json")
        quant_type = "fp16" if is_fp16 else "unknown"
        original_model = "unknown"
        params_billions = None
        ram_gb = None

        # Use cached metadata if available
        if cached_metadata:
            quant_type = cached_metadata.quantization
            params_billions = cached_metadata.params_billions
            ram_gb = cached_metadata.ram_gb
            # Architecture might contain original model info
            if cached_metadata.architecture and cached_metadata.architecture != "unknown":
                original_model = cached_metadata.architecture

        # Also read JSON metadata file for additional info
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    # Only override quant_type if not FP16 and not from cache
                    if not is_fp16 and not cached_metadata:
                        quant_type = metadata.get("quant_type", "unknown")
                    # Prefer JSON metadata for original model name
                    if "original_model" in metadata:
                        original_model = metadata.get("original_model", "unknown")
            except Exception as e:
                logger.warning(f"Could not read metadata for {gguf_path}: {e}")

        # Model ID: Use full absolute path so we can load it later
        # Format: "quantized:gguf:/absolute/path/to/file.gguf"
        model_id = f"quantized:gguf:{gguf_path.absolute()}"

        # Display name: Extract model name from filename and show quantization type
        # New format (no timestamp): Provider_Model-Name_quanttype[_counter]
        # Examples: 
        #   - "Qwen_Qwen3-0.6B_q4_k_m.gguf" → "Qwen/Qwen3-0.6B"
        #   - "allenai_OLMo-1B_q4_k_m_2.gguf" → "allenai/OLMo-1B" (with counter)
        # Old format (backward compat): Provider_Model-Name_quanttype_YYYYMMDD_HHMMSS
        filename_stem = gguf_path.stem

        # Try to extract meaningful model name from filename
        # Pattern: Provider_Model-Name_quanttype[_timestamp][_counter](_f16)
        name = None

        if "_" in filename_stem:
            parts = filename_stem.split("_")

            # Check if ends with "_f16" (FP16 GGUF intermediate file)
            is_fp16 = len(parts) >= 1 and parts[-1] == "f16"
            if is_fp16 and parts[-1] == "f16":
                parts = parts[:-1]

            # Remove trailing counter if present (e.g., "_2", "_3")
            if len(parts) >= 1 and parts[-1].isdigit() and len(parts[-1]) <= 2:
                parts = parts[:-1]

            # Remove timestamp parts (YYYYMMDD_HHMMSS) for backward compatibility
            # Only if they look like timestamps (8 digits + 6 digits)
            if len(parts) >= 2:
                if (parts[-1].isdigit() and len(parts[-1]) == 6 and
                    parts[-2].isdigit() and len(parts[-2]) == 8):
                    parts = parts[:-2]  # Remove timestamp

            # Find the quantization type in the filename
            # Some quant types are 3 parts (q4_k_m), others are 2 parts (q8_0)
            quant_patterns_3 = ["q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k"]
            quant_patterns_2 = ["q8_0", "q4_0", "q5_0"]
            quant_idx = -1

            # Try 3-part patterns first
            for i, part in enumerate(parts):
                if i + 2 < len(parts):
                    potential_quant = "_".join(parts[i:i+3])
                    if potential_quant in quant_patterns_3:
                        quant_idx = i
                        break

            # Try 2-part patterns if not found
            if quant_idx == -1:
                for i, part in enumerate(parts):
                    if i + 1 < len(parts):
                        potential_quant = "_".join(parts[i:i+2])
                        if potential_quant in quant_patterns_2:
                            quant_idx = i
                            break

            # Extract model name (everything before quant type)
            if quant_idx > 0:
                # Join parts before quantization type
                model_name_parts = parts[:quant_idx]

                # Convert underscores to slashes for provider/model format
                # Example: ['Qwen', 'Qwen3-0', '6B'] → "Qwen/Qwen3-0.6B"
                if len(model_name_parts) >= 2:
                    # First part is usually provider (Qwen, mistralai, etc.)
                    provider = model_name_parts[0]

                    # Rest is model name - rejoin with dashes
                    model_parts = model_name_parts[1:]

                    # Handle version numbers split by underscore (e.g., '0' '6B' → '0.6B')
                    reconstructed = []
                    for i, p in enumerate(model_parts):
                        if i > 0 and model_parts[i-1].isdigit() and p and p[0].isdigit():
                            # This looks like a version split: "3-0" + "6B" → "3-0.6B"
                            reconstructed[-1] = reconstructed[-1] + "." + p
                        else:
                            reconstructed.append(p)

                    model_name = "-".join(reconstructed)
                    name = f"{provider}/{model_name}"
                else:
                    # Fallback: just join with dashes
                    name = "-".join(model_name_parts)

        # Final fallback: use the original filename stem
        if not name:
            name = filename_stem.replace("_", "-")

        # Add quantization type tag - make it very clear with descriptions
        if quant_type != "unknown":
            if is_fp16:
                name = f"{name} [FP16 - Half Precision]"
            elif quant_type == "q4_k_m":
                name = f"{name} [Q4_K_M - 4-bit Medium Quality]"
            elif quant_type == "q4_k_s":
                name = f"{name} [Q4_K_S - 4-bit Small Size]"
            elif quant_type == "q5_k_m":
                name = f"{name} [Q5_K_M - 5-bit Medium Quality]"
            elif quant_type == "q5_k_s":
                name = f"{name} [Q5_K_S - 5-bit Small Size]"
            elif quant_type == "q6_k":
                name = f"{name} [Q6_K - 6-bit High Quality]"
            elif quant_type == "q8_0":
                name = f"{name} [Q8_0 - 8-bit Highest Quality]"
            elif quant_type == "int8":
                name = f"{name} [INT8 - 8-bit Integer]"
            elif quant_type == "int4":
                name = f"{name} [INT4 - 4-bit Integer]"
            else:
                name = f"{name} [{quant_type.upper()}]"

        # Assess compatibility
        compatibility, compatibility_message = self._assess_compatibility_detailed(size_gb)

        # Import required types
        from ..models.endpoints import EndpointType

        return ModelInfo(
            model_id=model_id,
            name=name,
            provider=ProviderType.QUANTIZED,
            model_type=ModelType.LLM,  # Assume LLM for now
            size_gb=size_gb,
            params_billions=params_billions,  # From metadata cache
            ram_gb=ram_gb,  # From metadata cache
            capabilities=[EndpointType.TEXT],  # Default to text-only
            compatibility=compatibility,
            compatibility_message=compatibility_message,
            is_installed=True,
            quantization=quant_type if quant_type != "unknown" else None,
            architecture=original_model if original_model != "unknown" else None,
        )

    def _create_model_info_from_hf(self, model_dir: Path) -> Optional[ModelInfo]:
        """Create ModelInfo from HuggingFace format directory using metadata cache.

        Args:
            model_dir: Path to model directory

        Returns:
            ModelInfo or None if invalid
        """
        # Get metadata from cache (reads config.json)
        cached_metadata = self.metadata_cache.get_metadata(str(model_dir), provider="huggingface")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        size_gb = total_size / (1024 ** 3)

        # Try to read metadata
        metadata_file = model_dir / "quantization_metadata.json"
        quant_type = "unknown"
        original_model = "unknown"
        params_billions = None
        ram_gb = None
        model_type = ModelType.LLM  # Default
        is_vlm = False

        # Use cached metadata if available
        if cached_metadata:
            quant_type = cached_metadata.quantization
            params_billions = cached_metadata.params_billions
            ram_gb = cached_metadata.ram_gb
            if cached_metadata.architecture and cached_metadata.architecture != "unknown":
                original_model = cached_metadata.architecture
            # Check if VLM from metadata
            if cached_metadata.model_type == "vlm":
                model_type = ModelType.VLM
                is_vlm = True
        
        # Also check config.json directly for VLM detection
        config_path = model_dir / "config.json"
        if not is_vlm and config_path.exists():
            try:
                import json
                with open(config_path) as f:
                    config = json.load(f)
                
                # Check architecture for VLM patterns
                arch = config.get("architectures", [""])[0]
                vlm_patterns = [
                    "ForConditionalGeneration", "VisionTextDual", "VisionEncoder",
                    "Llava", "Blip", "Qwen2VL", "Qwen3VL", "InstructBlip"
                ]
                is_vlm = any(pattern in arch for pattern in vlm_patterns)
                
                # Also check for vision config keys
                if not is_vlm:
                    is_vlm = any(key in config for key in ["vision_config", "visual_config", "image_encoder"])
                
                if is_vlm:
                    model_type = ModelType.VLM
                    logger.info(f"Detected VLM quantized model: {arch}")
            except Exception as e:
                logger.debug(f"Could not check config for VLM detection: {e}")

        # Read JSON metadata file - this MUST override cached metadata for quantized models
        # because cached metadata reads raw dtype (bf16) while JSON has actual quant type (int4)
        is_mlx = False
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)

                    # ALWAYS prefer JSON metadata for quantized models
                    # Cached metadata reads dtype from safetensors (bf16) which is wrong for MLX
                    quant_type = metadata.get("quant_type", quant_type)  # Override even if empty
                    original_model = metadata.get("original_model", original_model)

                    # Check if this is an MLX model
                    if metadata.get("module") == "mlx":
                        is_mlx = True
                        logger.info(f"Detected MLX quantized model: {model_dir.name}")
            except Exception as e:
                logger.warning(f"Could not read metadata for {model_dir}: {e}")
        else:
            # Metadata file doesn't exist - this is a problem for MLX detection
            logger.warning(f"No quantization_metadata.json found for {model_dir}")
            logger.warning("MLX models require metadata file for proper detection")

        # Model ID: Use full absolute path with format prefix
        # Format: "quantized:hf:/absolute/path/to/model_dir"
        model_id = f"quantized:hf:{model_dir.absolute()}"

        # Display name: Extract clean name from directory, removing module prefixes and quant types
        # Format: Provider_Model-Name_{module}-{quant_type}[_{counter}]
        # Examples:
        #   - "Qwen_Qwen3-0.6B_mlx-int4" → "Qwen/Qwen3-0.6B"
        #   - "allenai_OLMo-1B_fp16" → "allenai/OLMo-1B"
        #   - "Qwen_Qwen3-0.6B_mlx-int4_2" → "Qwen/Qwen3-0.6B" (with counter)
        dir_name = model_dir.name
        parts = dir_name.split("_")

        # Remove trailing counter if present (e.g., "_2", "_3")
        # This handles conflicts when the same model is quantized multiple times
        if len(parts) >= 1 and parts[-1].isdigit() and len(parts[-1]) <= 2:
            parts = parts[:-1]

        # Remove timestamp parts (YYYYMMDD_HHMMSS) for backwards compatibility with old naming
        # This handles models quantized before the timestamp removal fix
        if len(parts) >= 2:
            if (parts[-1].isdigit() and len(parts[-1]) == 6 and
                parts[-2].isdigit() and len(parts[-2]) == 8):
                parts = parts[:-2]  # Remove timestamp

        # Remove module-quant_type (e.g., "mlx-int4", "openvino-int8", "fp16", "q4_k_m")
        # Look for module prefixes or standalone quant types
        if len(parts) >= 1:
            last_part = parts[-1].lower()
            # Check if last part is a quant type (with or without module prefix)
            quant_indicators = [
                "int4", "int8", "int2", "fp16", "bf16", "fp32",
                "mlx-int4", "mlx-int8", "mlx-int2", "mlx-fp16",
                "openvino-int4", "openvino-int8", "openvino-fp16",
                # GGUF types
                "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0",
            ]
            if last_part in quant_indicators:
                parts = parts[:-1]  # Remove quant type

        # Reconstruct clean model name
        if len(parts) >= 2:
            # First part is provider, rest is model name
            provider = parts[0]
            model_parts = parts[1:]

            # Handle version numbers split by underscore (e.g., '0' '6B' → '0.6B')
            reconstructed = []
            for i, p in enumerate(model_parts):
                if i > 0 and model_parts[i-1].isdigit() and p and p[0].isdigit():
                    reconstructed[-1] = reconstructed[-1] + "." + p
                else:
                    reconstructed.append(p)

            model_name = "-".join(reconstructed)
            clean_name = f"{provider}/{model_name}"
        else:
            # Fallback: use what we have
            clean_name = "-".join(parts) if parts else dir_name

        # Add quantization type label
        if quant_type != "unknown":
            if quant_type == "fp16":
                name = f"{clean_name} [FP16 - Half Precision]"
            elif quant_type == "bf16":
                name = f"{clean_name} [BF16 - Brain Float16]"
            elif quant_type == "int8":
                name = f"{clean_name} [INT8 - 8-bit Integer]"
            elif quant_type == "int4":
                if is_mlx:
                    name = f"{clean_name} [INT4 MLX] ⚠️ Use MLX tools"
                else:
                    name = f"{clean_name} [INT4 - 4-bit Integer]"
            elif quant_type == "int2":
                if is_mlx:
                    name = f"{clean_name} [INT2 MLX] ⚠️ Use MLX tools"
                else:
                    name = f"{clean_name} [INT2 - 2-bit Integer]"
            else:
                name = f"{clean_name} [{quant_type.upper()}]"
        else:
            name = clean_name

        # Assess compatibility
        if is_mlx:
            # MLX models CAN be loaded - we have native MLX inference support!
            # Check if MLX is available on this system
            mlx_available, mlx_message = self._check_mlx_availability()
            if mlx_available:
                # MLX is available - assess normally based on size
                compatibility, compatibility_message = self._assess_compatibility_detailed(size_gb)
                compatibility_message = f"MLX format (native Apple Silicon) - {compatibility_message}"
            else:
                # MLX not available - mark incompatible
                compatibility = CompatibilityStatus.INCOMPATIBLE
                compatibility_message = mlx_message
        else:
            compatibility, compatibility_message = self._assess_compatibility_detailed(size_gb)

        # Import required types
        from ..models.endpoints import EndpointType
        
        # Set capabilities based on model type
        if is_vlm:
            capabilities = [
                EndpointType.QA,
                EndpointType.CAPTION,
                EndpointType.DETECT,
                EndpointType.POINT,
                EndpointType.TEXT,
            ]
        else:
            capabilities = [EndpointType.TEXT]

        return ModelInfo(
            model_id=model_id,
            name=name,
            provider=ProviderType.QUANTIZED,
            model_type=model_type,  # VLM or LLM based on detection
            size_gb=size_gb,
            params_billions=params_billions,  # From metadata cache
            ram_gb=ram_gb,  # From metadata cache
            capabilities=capabilities,  # VLM gets vision capabilities
            compatibility=compatibility,
            compatibility_message=compatibility_message,
            is_installed=True,
            quantization=quant_type if quant_type != "unknown" else None,
            architecture=original_model if original_model != "unknown" else None,
        )

    def _check_mlx_availability(self) -> tuple[bool, str]:
        """Check if MLX framework is available for inference.

        Returns:
            Tuple of (available, message)
        """
        import platform

        # Check if running on macOS
        if platform.system() != "Darwin":
            return False, "MLX only available on macOS (requires Apple Silicon)"

        # Check if Apple Silicon
        machine = platform.machine()
        if machine != "arm64":
            return False, f"MLX requires Apple Silicon, found: {machine}"

        # Check if mlx packages are installed
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
            import mlx_vlm  # noqa: F401
            return True, "MLX framework available"
        except ImportError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else "mlx"
            return False, f"MLX not installed: {missing}. Run: pip install mlx mlx-lm mlx-vlm"

    def _assess_compatibility_detailed(self, model_size_gb: float) -> tuple[CompatibilityStatus, str]:
        """Assess model compatibility with system.

        Args:
            model_size_gb: Model size in GB

        Returns:
            Tuple of (compatibility status, message)
        """
        recommended_size = self.system_specs.recommended_model_size_gb
        ratio = model_size_gb / recommended_size

        if model_size_gb < recommended_size * 0.7:
            return (
                CompatibilityStatus.PERFECT_FIT,
                f"Model uses {ratio*100:.0f}% of recommended size ({model_size_gb:.1f}GB / {recommended_size:.1f}GB)"
            )
        elif model_size_gb <= recommended_size:
            return (
                CompatibilityStatus.TIGHT_FIT,
                f"Model uses {ratio*100:.0f}% of recommended size ({model_size_gb:.1f}GB / {recommended_size:.1f}GB). May impact performance."
            )
        else:
            return (
                CompatibilityStatus.TOO_LARGE,
                f"Model exceeds recommended size by {(ratio-1)*100:.0f}% ({model_size_gb:.1f}GB vs {recommended_size:.1f}GB). May cause OOM errors."
            )

    def load_model(self, model_id: str, device: str) -> Any:
        """Load a quantized model.

        Delegates to appropriate backend based on format:
        - GGUF files → llama-cpp-python
        - HF format → transformers

        Args:
            model_id: Model identifier (format: "quantized:gguf:/path" or "quantized:hf:/path")
            device: Target device ("cuda", "mps", or "cpu")

        Returns:
            Loaded model handle
        """
        # Parse model_id to extract format and path
        # Format: "quantized:gguf:/path/to/file" or "quantized:hf:/path/to/dir"
        parts = model_id.split(":", 2)
        if len(parts) != 3 or parts[0] != "quantized":
            raise ValueError(f"Invalid quantized model_id format: {model_id}")

        model_format = parts[1]  # "gguf" or "hf"
        model_path = Path(parts[2])  # Path to model

        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        if model_format == "gguf":
            return self._load_gguf_model(model_path, device)
        elif model_format == "hf":
            return self._load_hf_model(model_path, device)
        else:
            raise ValueError(f"Unknown quantized model format: {model_format}")

    def _load_gguf_model(self, model_path: Path, device: str) -> Any:
        """Load GGUF quantized model using llama-cpp-python.

        Args:
            model_path: Path to .gguf file
            device: Target device ("cuda", "mps", or "cpu")

        Returns:
            Llama model instance
        """
        try:
            from llama_cpp import Llama

            logger.info(f"Loading GGUF model: {model_path}")

            # Determine GPU layers based on device
            n_gpu_layers = -1 if device in ["cuda", "mps"] else 0

            model = Llama(
                model_path=str(model_path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=2048,
                verbose=False,
            )

            logger.info(f"GGUF model loaded successfully on {device}")
            return model

        except ImportError:
            raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

    def _load_hf_model(self, model_path: Path, device: str) -> Any:
        """Load HuggingFace format quantized model (LLM or VLM).

        Automatically detects MLX models and uses MLX inference instead of transformers.

        Args:
            model_path: Path to model directory
            device: Target device ("cuda", "mps", or "cpu")

        Returns:
            Tuple of (model, processor/tokenizer, metadata) where metadata contains model type info
        """
        # Check if this is an MLX-quantized model
        metadata_file = model_path / "quantization_metadata.json"
        is_mlx = False
        is_vlm = False

        if metadata_file.exists():
            try:
                import json
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    module = metadata.get("module", "")

                    if module == "mlx":
                        is_mlx = True
                        original_model = metadata.get("original_model", "unknown")
                        # Detect VLM from original model name or provider info
                        is_vlm = "VL" in original_model or "vision" in original_model.lower()

                        logger.info(f"Detected MLX {'VLM' if is_vlm else 'LLM'} model: {model_path.name}")

            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse quantization metadata: {e}")
            except Exception as e:
                logger.debug(f"Error checking quantization metadata: {e}")

        # Route to appropriate loader
        if is_mlx:
            return self._load_mlx_model(model_path, is_vlm, device)
        else:
            return self._load_transformers_model(model_path, device)

    def _load_mlx_model(self, model_path: Path, is_vlm: bool, device: str) -> Any:
        """Load MLX quantized model using mlx-lm or mlx-vlm.

        Args:
            model_path: Path to MLX model directory
            is_vlm: Whether this is a VLM or LLM
            device: Target device (ignored for MLX - always uses Metal)

        Returns:
            Tuple of (model, processor/tokenizer, {"is_mlx": True, "is_vlm": bool})
        """
        try:
            if is_vlm:
                # Use mlx-vlm for Vision-Language Models
                from mlx_vlm import load

                logger.info(f"Loading MLX VLM from: {model_path}")
                model, processor = load(str(model_path))
                logger.info("✓ MLX VLM loaded successfully on Metal GPU")

                # Return with metadata marker (including model_path for inference)
                return (model, processor, {"is_mlx": True, "is_vlm": True, "model_path": str(model_path)})
            else:
                # Use mlx-lm for pure LLMs
                from mlx_lm import load

                logger.info(f"Loading MLX LLM from: {model_path}")
                model, tokenizer = load(str(model_path))
                logger.info("✓ MLX LLM loaded successfully on Metal GPU")

                # Return with metadata marker
                return (model, tokenizer, {"is_mlx": True, "is_vlm": False})

        except ImportError as e:
            missing = "mlx-vlm" if is_vlm else "mlx-lm"
            error_msg = (
                f"❌ Cannot load MLX model: {missing} not installed\n\n"
                f"Install with: pip install {missing}\n"
                f"Or install both: pip install mlx mlx-lm mlx-vlm"
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from e

        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise

    def _load_transformers_model(self, model_path: Path, device: str) -> Any:
        """Load model using HuggingFace transformers (non-MLX models).

        Args:
            model_path: Path to model directory
            device: Target device

        Returns:
            Tuple of (model, processor/tokenizer, {"is_mlx": False, "is_vlm": bool})
        """

        try:
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

            # Suppress library output during loading
            from ..utils.output_suppressor import suppress_transformers_output

            logger.info(f"Loading HF quantized model: {model_path}")

            # Check if this is a VLM by inspecting config.json
            config_path = model_path / "config.json"
            is_vlm = False
            architecture_name = "unknown"
            
            if config_path.exists():
                try:
                    import json
                    with open(config_path) as f:
                        config = json.load(f)
                    
                    # Check architecture for VLM patterns
                    architectures = config.get("architectures", [])
                    arch = architectures[0] if architectures else ""
                    architecture_name = arch
                    
                    # Comprehensive VLM pattern matching
                    vlm_patterns = [
                        "ForConditionalGeneration",  # Qwen2VL, LLaVA, etc.
                        "VisionTextDual",            # CLIP-based
                        "VisionEncoder",             # Vision encoders
                        "Llava", "LLaVA",            # LLaVA variants
                        "Blip", "BLIP",              # BLIP variants
                        "Qwen2VL", "Qwen3VL",        # Qwen VLMs
                        "QwenVL",                    # Original QwenVL
                        "InstructBlip",              # InstructBLIP
                        "Idefics",                   # Idefics
                        "Kosmos",                    # Kosmos
                        "Pix2Struct",                # Pix2Struct
                        "Git",                       # GIT model
                    ]
                    
                    # Case-insensitive pattern matching
                    arch_lower = arch.lower()
                    is_vlm = any(pattern.lower() in arch_lower for pattern in vlm_patterns)
                    
                    # Also check for vision config keys in config
                    if not is_vlm:
                        vision_keys = ["vision_config", "visual_config", "image_encoder", "vision_tower", "mm_vision_tower"]
                        is_vlm = any(key in config for key in vision_keys)
                        if is_vlm:
                            logger.info(f"Detected VLM from vision config keys: {[k for k in vision_keys if k in config]}")
                    
                    # Check model_type field
                    if not is_vlm:
                        model_type_str = config.get("model_type", "").lower()
                        is_vlm = any(vtype in model_type_str for vtype in ["vision", "vlm", "multimodal"])
                        if is_vlm:
                            logger.info(f"Detected VLM from model_type: {model_type_str}")
                    
                    logger.info(f"Model architecture: {arch}, VLM: {is_vlm}")
                except Exception as e:
                    logger.warning(f"Could not check model config for VLM detection: {e}")

            # Load with appropriate strategy
            with suppress_transformers_output():
                if is_vlm:
                    # Try to load as VLM with processor
                    try:
                        from transformers import AutoProcessor
                        
                        logger.info(f"Attempting to load VLM ({architecture_name}) with AutoProcessor...")
                        
                        processor = AutoProcessor.from_pretrained(
                            str(model_path),
                            trust_remote_code=True
                        )
                        
                        # Verify processor has image_processor (confirming it's a VLM)
                        if not hasattr(processor, 'image_processor'):
                            logger.warning(f"Processor loaded but has no image_processor - not a VLM!")
                            raise ValueError("Not a true VLM processor")
                        
                        # Check for Qwen3-VL + MPS incompatibility
                        # Qwen3-VL has known MPS bugs (matrix dimension errors in mps_matmul)
                        target_device = device
                        if 'qwen3' in architecture_name.lower() and device == 'mps':
                            logger.warning("⚠ Qwen3-VL has MPS compatibility issues")
                            logger.warning("  MPS causes 'incompatible dimensions' errors during inference")
                            logger.warning("  Forcing CPU mode for stability")
                            target_device = 'cpu'
                        
                        # Load VLM with appropriate model class
                        # Try multiple VLM model classes for maximum compatibility
                        model = None
                        model_classes = [
                            'AutoModelForVision2Seq',  # Modern VLMs (Qwen2-VL, etc.)
                            'AutoModelForCausalLM',    # Some VLMs use this (LLaVA)
                            'AutoModel',                # Generic fallback
                        ]
                        
                        for model_class_name in model_classes:
                            try:
                                logger.info(f"Trying to load VLM with {model_class_name}...")
                                
                                # Prepare loading kwargs
                                load_kwargs = {
                                    "low_cpu_mem_usage": True,
                                    "trust_remote_code": True
                                }
                                
                                # Use explicit device for Qwen3+MPS workaround, otherwise use device_map
                                if target_device == 'cpu':
                                    load_kwargs["device_map"] = {"": "cpu"}  # Force CPU
                                    logger.debug("Loading with forced CPU device map")
                                else:
                                    load_kwargs["device_map"] = "auto"
                                
                                if model_class_name == 'AutoModelForVision2Seq':
                                    from transformers import AutoModelForVision2Seq
                                    model = AutoModelForVision2Seq.from_pretrained(
                                        str(model_path),
                                        **load_kwargs
                                    )
                                elif model_class_name == 'AutoModelForCausalLM':
                                    from transformers import AutoModelForCausalLM
                                    model = AutoModelForCausalLM.from_pretrained(
                                        str(model_path),
                                        **load_kwargs
                                    )
                                else:  # AutoModel
                                    model = AutoModel.from_pretrained(
                                        str(model_path),
                                        **load_kwargs
                                    )
                                
                                # Verify the model has generate method
                                if not hasattr(model, 'generate'):
                                    logger.warning(f"{model_class_name} loaded but has no .generate() method")
                                    model = None
                                    continue
                                
                                logger.info(f"✓ Successfully loaded VLM with {model_class_name}")
                                break
                                
                            except Exception as e:
                                logger.debug(f"{model_class_name} failed: {e}")
                                continue
                        
                        if model is None:
                            raise RuntimeError(
                                f"Could not load VLM model from {model_path}.\n"
                                f"Tried: {', '.join(model_classes)}\n"
                                f"None of these model classes worked or had .generate() method."
                            )
                        
                        logger.info(f"✓ Successfully loaded quantized VLM with processor on {target_device}")
                        if target_device != device:
                            logger.info(f"  Originally requested: {device}, using {target_device} for compatibility")
                        logger.info(f"  Architecture: {architecture_name}")
                        logger.info(f"  Processor type: {type(processor).__name__}")
                        return (model, processor, {"is_mlx": False, "is_vlm": True})
                    except Exception as e:
                        logger.warning(f"Failed to load as VLM: {e}")
                        logger.warning(f"Falling back to LLM loading...")
                        is_vlm = False
                
                # Load as LLM (fallback or non-VLM)
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=True
                )

                # PRODUCTION FIX: Load models with custom configurations (like OLMo)
                # When models are quantized and saved, custom config classes get copied to the output dir
                # However, sometimes the modeling_*.py file is missing, which breaks loading
                #
                # Multi-strategy loading approach:
                # 1. Try loading directly with trust_remote_code=True
                # 2. If modeling file is missing, copy it from original model's HF cache
                # 3. If that fails, try loading weights with original model architecture
                # 4. Final fallback: Load from original model_id if available in metadata

                model = None
                error_messages = []

                # Strategy 1: Try AutoModelForCausalLM (most common for LLMs)
                try:
                    logger.debug("Attempting to load with AutoModelForCausalLM...")
                    model = AutoModelForCausalLM.from_pretrained(
                        str(model_path),
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    logger.info("✓ Loaded with AutoModelForCausalLM")
                except Exception as e:
                    error_msg = str(e)
                    error_messages.append(f"AutoModelForCausalLM: {error_msg[:200]}")
                    logger.debug(f"AutoModelForCausalLM failed: {error_msg}")

                # Strategy 2: If modeling file is missing, try to fix it
                if model is None and "does not appear to have a file named modeling_" in str(error_messages[-1]):
                    try:
                        import json
                        import shutil

                        logger.info("Detected missing modeling file - attempting to fix...")

                        # Read quantization metadata to get original model
                        metadata_file = model_path / "quantization_metadata.json"
                        original_model_id = None

                        if metadata_file.exists():
                            with open(metadata_file) as f:
                                metadata = json.load(f)
                                original_model_id = metadata.get("original_model", None)
                                logger.info(f"Original model from metadata: {original_model_id}")

                        # Also try to infer from directory name if metadata doesn't have it
                        if not original_model_id:
                            # Directory name format: Provider_Model-Name_quant_type
                            # Example: allenai_OLMo-1B_fp16_1 → allenai/OLMo-1B
                            dir_name = model_path.name
                            parts = dir_name.split("_")

                            # Remove quant type and counter at the end
                            if len(parts) >= 2:
                                # Try to reconstruct original model ID
                                # Heuristic: Provider/ModelName format
                                original_model_id = f"{parts[0]}/{'-'.join(parts[1:-1])}"
                                # Clean up common quant type suffixes
                                for suffix in ["fp16", "int8", "int4", "bf16", "mlx-int4", "openvino-int8"]:
                                    original_model_id = original_model_id.replace(f"-{suffix}", "")
                                logger.info(f"Inferred original model: {original_model_id}")

                        if original_model_id:
                            # Strategy 2a: Copy missing modeling file from original model's cache
                            try:
                                from transformers.utils import cached_file
                                from pathlib import Path as P

                                # Get model architecture to find the modeling file name
                                config_path = model_path / "config.json"
                                with open(config_path) as f:
                                    config_json = json.load(f)

                                arch = config_json.get("architectures", [""])[0]
                                if arch:
                                    # Infer modeling file name from architecture
                                    # E.g., OLMoForCausalLM → modeling_olmo.py
                                    model_type = arch.replace("ForCausalLM", "").replace("Model", "").lower()
                                    modeling_file = f"modeling_{model_type}.py"

                                    logger.info(f"Looking for {modeling_file} in original model cache...")

                                    # Try to get file from HF cache
                                    try:
                                        cached_modeling_file = cached_file(
                                            original_model_id,
                                            modeling_file,
                                            _raise_exceptions_for_missing_entries=False
                                        )

                                        if cached_modeling_file and P(cached_modeling_file).exists():
                                            # Copy to quantized model directory
                                            dest = model_path / modeling_file
                                            shutil.copy2(cached_modeling_file, dest)
                                            logger.info(f"✓ Copied {modeling_file} from original model cache")

                                            # Retry loading
                                            model = AutoModelForCausalLM.from_pretrained(
                                                str(model_path),
                                                device_map="auto",
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=True
                                            )
                                            logger.info("✓ Successfully loaded after copying modeling file!")
                                    except Exception as copy_error:
                                        logger.debug(f"Could not copy modeling file: {copy_error}")

                            except Exception as fix_error:
                                logger.debug(f"Strategy 2a failed: {fix_error}")

                            # Strategy 2b: Load architecture from original model, then load weights
                            if model is None:
                                try:
                                    logger.info(f"Attempting to load architecture from {original_model_id}...")

                                    # Load original model architecture (without weights)
                                    from transformers import AutoConfig
                                    config = AutoConfig.from_pretrained(
                                        original_model_id,
                                        trust_remote_code=True
                                    )

                                    # Load model architecture
                                    base_model = AutoModelForCausalLM.from_config(
                                        config,
                                        trust_remote_code=True
                                    )

                                    # Load quantized weights into the model
                                    logger.info("Loading quantized weights into model architecture...")

                                    # Auto-install safetensors if needed
                                    try:
                                        from safetensors.torch import load_file
                                    except ImportError:
                                        logger.info("safetensors not found, installing...")
                                        from ..utils.dependency_installer import DependencyInstaller
                                        if DependencyInstaller.install_package("safetensors", auto_install=True, quiet=True):
                                            from safetensors.torch import load_file
                                        else:
                                            raise ImportError("Failed to install safetensors. Please run: pip install safetensors")

                                    import torch

                                    # Load the safetensors file
                                    weights_file = model_path / "model.safetensors"
                                    if weights_file.exists():
                                        state_dict = load_file(str(weights_file))
                                        base_model.load_state_dict(state_dict)

                                        # Move to device
                                        if device == "mps":
                                            base_model = base_model.to("mps")
                                        elif device == "cuda":
                                            base_model = base_model.to("cuda")
                                        else:
                                            base_model = base_model.to("cpu")

                                        model = base_model
                                        logger.info(f"✓ Loaded with weights from {original_model_id} architecture")
                                except Exception as arch_error:
                                    logger.debug(f"Strategy 2b failed: {arch_error}")
                                    error_messages.append(f"Architecture loading: {str(arch_error)[:100]}")

                    except Exception as e:
                        logger.debug(f"Missing modeling file fix failed: {e}")

                # Strategy 3: Try with AutoConfig
                if model is None:
                    try:
                        from transformers import AutoConfig

                        logger.debug("Attempting to load with AutoConfig...")
                        config = AutoConfig.from_pretrained(
                            str(model_path),
                            trust_remote_code=True
                        )

                        model = AutoModelForCausalLM.from_pretrained(
                            str(model_path),
                            config=config,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                        logger.info("✓ Loaded with AutoConfig approach")
                    except Exception as e:
                        error_msg = str(e)
                        error_messages.append(f"AutoConfig: {error_msg[:100]}")
                        logger.debug(f"AutoConfig approach failed: {error_msg}")

                # Strategy 4: Last resort - try AutoModel (generic fallback)
                if model is None:
                    try:
                        logger.debug("Attempting final fallback with AutoModel...")
                        model = AutoModel.from_pretrained(
                            str(model_path),
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                        logger.info("✓ Loaded with AutoModel (generic fallback)")
                    except Exception as e:
                        error_msg = str(e)
                        error_messages.append(f"AutoModel: {error_msg[:100]}")
                        logger.error(f"AutoModel failed: {error_msg}")

                # If all strategies failed, raise comprehensive error
                if model is None:
                    error_summary = "\n  - ".join(error_messages)
                    raise RuntimeError(
                        f"Failed to load quantized model from {model_path}\n\n"
                        f"Tried multiple loading strategies:\n  - {error_summary}\n\n"
                        f"This may be due to:\n"
                        f"  • Missing modeling files (modeling_*.py)\n"
                        f"  • Incompatible model architecture\n"
                        f"  • Corrupted model files\n"
                        f"  • Missing dependencies for custom model code\n\n"
                        f"Recommendation: Re-quantize the model to ensure all files are copied."
                    )

                logger.info(f"Loaded quantized LLM with tokenizer on {device}")
                return (model, tokenizer, {"is_mlx": False, "is_vlm": False})

        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")

    def unload_model(self, model_handle: Any) -> None:
        """Unload model and free resources.

        Args:
            model_handle: Model instance (can be tuple of (model, processor) or single model)
        """
        # GGUF models (llama-cpp-python) handle cleanup automatically
        # HF models need manual cleanup
        try:
            # Handle tuple (model, processor/tokenizer)
            if isinstance(model_handle, tuple):
                model, _ = model_handle
                if hasattr(model, "cpu"):
                    model.cpu()
                del model_handle
            else:
                # Single model
                if hasattr(model_handle, "cpu"):
                    model_handle.cpu()
                del model_handle
            
            # Clear GPU cache if available
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            logger.info("Quantized model unloaded")
        except Exception as e:
            logger.warning(f"Error during model unload: {e}")

    def run_qa(
        self,
        model_handle: Any,
        image: Any,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None
    ) -> str:
        """Run question answering on quantized VLM.

        Args:
            model_handle: Loaded model (tuple of (model, processor, metadata) for HF/MLX VLMs)
            image: PIL Image
            question: Question text
            conversation_history: Optional conversation history

        Returns:
            Answer text
        """
        # Check if this is a 3-tuple (new format with metadata)
        if isinstance(model_handle, tuple) and len(model_handle) == 3:
            model, processor_or_tokenizer, metadata = model_handle
            is_mlx = metadata.get("is_mlx", False)
            is_vlm = metadata.get("is_vlm", False)

            if not is_vlm:
                raise NotImplementedError(
                    "This quantized model is a text-only LLM, not a VLM.\n"
                    "QA endpoint requires a vision-language model.\n\n"
                    "The model was quantized as an LLM without vision capabilities.\n"
                    "To use vision tasks, quantize a VLM (e.g., LLaVA, Qwen2-VL, BLIP)."
                )

            # Route to appropriate inference method
            if is_mlx:
                return self._run_qa_mlx(model, processor_or_tokenizer, image, question, conversation_history, metadata)
            else:
                return self._run_qa_transformers(model, processor_or_tokenizer, image, question, conversation_history)

        # Backward compatibility: 2-tuple format (legacy)
        elif isinstance(model_handle, tuple) and len(model_handle) == 2:
            model, processor_or_tokenizer = model_handle
            
            # Check if it's a processor (VLM) - processors have image_processor attribute
            # Tokenizers do NOT have this, so this is the reliable way to detect VLMs
            if hasattr(processor_or_tokenizer, 'image_processor'):
                # This is a VLM with processor
                import torch
                from ..utils.history_formatter import format_qa_history
                
                # Get device
                model_device = next(model.parameters()).device
                
                # Format question with history
                full_question = format_qa_history(
                    conversation_history=conversation_history,
                    current_question=question,
                    max_turns=5
                )
                
                # Prepare inputs with automatic format detection for all VLMs
                # Try messages format first (modern VLMs: Qwen2-VL, Qwen3-VL, Idefics, etc.)
                # Fall back to standard format (LLaVA, BLIP, etc.)
                inputs = None
                
                # Strategy 1: Try messages format with chat template (if available)
                if hasattr(processor_or_tokenizer, 'apply_chat_template'):
                    try:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": full_question}
                                ]
                            }
                        ]
                        
                        # Apply chat template to get properly formatted text with image tokens
                        text_prompt = processor_or_tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        # Process with the formatted text
                        inputs = processor_or_tokenizer(
                            text=[text_prompt],
                            images=[image],
                            padding=True,
                            return_tensors="pt"
                        )
                        
                        logger.debug("✓ Using messages format with chat template")
                        
                    except Exception as e:
                        logger.debug(f"Messages format failed: {e}, trying standard format")
                        inputs = None
                
                # Strategy 2: Standard text + images format (fallback)
                if inputs is None:
                    try:
                        inputs = processor_or_tokenizer(
                            text=full_question,
                            images=image,
                            return_tensors="pt"
                        )
                        logger.debug("✓ Using standard text + images format")
                    except Exception as e:
                        logger.error(f"Both input formats failed: {e}")
                        raise RuntimeError(
                            f"Failed to prepare inputs for VLM inference.\n"
                            f"Processor: {type(processor_or_tokenizer).__name__}\n"
                            f"Error: {e}"
                        )
                
                # Move inputs to model device (handle each tensor individually for MPS compatibility)
                inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                # Warn if on CPU (very slow)
                if str(model_device) == 'cpu':
                    logger.warning("⏳ Running inference on CPU - this will be SLOW (30s-2min)")
                    logger.warning("   Consider using a smaller model or CUDA/MPS-compatible device")
                
                # Generate (reduced tokens for faster response on CPU)
                logger.info("Generating response...")
                max_tokens = 512 if str(model_device) == 'cpu' else 1024
                
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=max_tokens)
                
                logger.info("✓ Response generated")
                
                # Decode
                response = processor_or_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                
                # Extract answer
                if conversation_history and "A:" in response:
                    parts = response.split("A:")
                    if len(parts) > 1:
                        response = parts[-1].strip()
                
                return response.strip()
            else:
                # This is a tokenizer (LLM), not a processor - cannot do vision tasks
                raise NotImplementedError(
                    "This quantized model is a text-only LLM, not a VLM.\n"
                    "QA endpoint requires a vision-language model.\n\n"
                    "The model was quantized as an LLM without vision capabilities.\n"
                    "To use vision tasks, quantize a VLM (e.g., LLaVA, Qwen2-VL, BLIP)."
                )
        
        # GGUF models don't support VLM yet
        raise NotImplementedError(
            "QA endpoint requires HuggingFace format quantized VLM.\n"
            "GGUF quantized models don't support vision inference yet."
        )

    def _run_qa_mlx(
        self,
        model: Any,
        processor: Any,
        image: Any,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """Run QA inference using MLX framework.

        Args:
            model: MLX model
            processor: MLX processor
            image: PIL Image
            question: Question text
            conversation_history: Optional conversation history
            metadata: Optional metadata dict with model_path

        Returns:
            Answer text
        """
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        logger.info("Running MLX VLM inference...")
        
        # Get model path from metadata (needed for config loading)
        model_path = metadata.get("model_path") if metadata else None

        try:
            # MLX VLM expects image path or list of image paths
            # If it's a PIL Image, we need to save it temporarily
            import tempfile
            from pathlib import Path

            # Save PIL image to temp file if needed
            if hasattr(image, 'save'):
                # It's a PIL Image
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    image.save(tmp.name, format='JPEG')
                    image_path = tmp.name
                    logger.debug(f"Saved image to temp file: {image_path}")
            else:
                # Assume it's already a path
                image_path = str(image)

            logger.debug(f"Image path: {image_path}")
            
            # PRODUCTION FIX: Use MLX's apply_chat_template to format prompt correctly
            # This adds required image tokens and applies model-specific template
            try:
                # Load model config to get chat template (requires model_path)
                if not model_path:
                    raise ValueError("model_path not available in metadata")
                    
                config = load_config(model_path)
                
                # Format question with history using MLX's template system
                # For VLMs, we need to include image placeholder in the content
                if conversation_history:
                    # Build message list with history
                    messages = []
                    for user_msg, bot_msg in conversation_history[-3:]:  # Last 3 turns
                        messages.append({"role": "user", "content": user_msg})
                        messages.append({"role": "assistant", "content": bot_msg})
                    # For current question, include image reference
                    messages.append({"role": "user", "content": question})
                else:
                    # Single question with image
                    messages = [{"role": "user", "content": question}]
                
                # Apply chat template
                prompt = apply_chat_template(processor, config, messages, num_images=1)
                logger.debug(f"Applied chat template. Prompt: {prompt[:200]}...")
                
                # CRITICAL: Check if image tokens were added
                # Some models need explicit image tokens like <image>, <|vision_start|>, etc.
                has_image_token = any(token in prompt for token in ['<image>', '<|image|>', '<|vision_start|>', '<|im_start|>user<image>'])
                
                if not has_image_token:
                    logger.warning("Chat template didn't add image tokens, adding manually")
                    # Prepend image token to the prompt
                    # Different models use different tokens - try common ones
                    if '<|im_start|>' in prompt:
                        # Qwen format - add after user start
                        prompt = prompt.replace('<|im_start|>user\n', '<|im_start|>user\n<image>\n', 1)
                    else:
                        # Generic - prepend to start
                        prompt = '<image>\n' + prompt
                    logger.debug(f"Modified prompt with image token: {prompt[:200]}...")
                
            except Exception as template_error:
                # Fallback: Use raw question if template fails
                logger.warning(f"Chat template failed: {template_error}, using raw prompt")
                if conversation_history:
                    # Build simple conversation format
                    conv_text = ""
                    for user_msg, bot_msg in conversation_history[-3:]:
                        conv_text += f"Q: {user_msg}\nA: {bot_msg}\n\n"
                    conv_text += f"Q: {question}\nA:"
                    prompt = conv_text
                else:
                    prompt = question

            # Generate response using MLX VLM
            # CORRECT Signature: generate(model, processor, prompt, image, **kwargs)
            # Note: 'image' can be a single path or list of paths
            # MEMORY FIX: Reduced max_tokens from 1024 to 256 to prevent OOM crashes on 8GB systems
            # MLX VLM accepts temperature and top_p as kwargs (passed to generate_step)
            response = generate(
                model,
                processor,
                prompt,  # Prompt comes BEFORE image in MLX VLM
                image_path,  # Image path comes AFTER prompt
                max_tokens=256,  # Reduced from 1024 to fit in 8GB RAM
                temperature=0.7,
                top_p=0.9,
                verbose=False
            )

            # Clean up temp file if we created one
            if hasattr(image, 'save'):
                try:
                    Path(image_path).unlink()
                except:
                    pass

            logger.info("✓ MLX response generated")

            # PRODUCTION FIX: Extract text from MLX GenerationResult
            # MLX VLM returns a GenerationResult object with .text attribute
            if hasattr(response, 'text'):
                # It's a GenerationResult object - extract the text
                response_text = response.text
                logger.debug(f"Extracted text from GenerationResult (tokens: {getattr(response, 'generation_tokens', 'N/A')})")
            elif isinstance(response, str):
                # Already a string
                response_text = response
            else:
                # Fallback: Convert to string
                response_text = str(response)
                logger.warning(f"Unexpected response type: {type(response)}, converting to string")
            
            # Clean response
            response_text = response_text.strip()

            # Clean up any prompt echo if present
            if prompt in response_text:
                response_text = response_text.replace(prompt, "").strip()

            return response_text if response_text else "I don't have a response."

        except Exception as e:
            logger.error(f"MLX VLM inference failed: {e}", exc_info=True)
            raise

    def _run_qa_transformers(
        self,
        model: Any,
        processor: Any,
        image: Any,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None
    ) -> str:
        """Run QA inference using HuggingFace transformers.

        Args:
            model: Transformers model
            processor: Transformers processor
            image: PIL Image
            question: Question text
            conversation_history: Optional conversation history

        Returns:
            Answer text
        """
        import torch
        from ..utils.history_formatter import format_qa_history

        # Get device
        model_device = next(model.parameters()).device

        # Format question with history
        full_question = format_qa_history(
            conversation_history=conversation_history,
            current_question=question,
            max_turns=5
        )

        # Prepare inputs with automatic format detection for all VLMs
        inputs = None

        # Strategy 1: Try messages format with chat template (if available)
        if hasattr(processor, 'apply_chat_template'):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": full_question}
                        ]
                    }
                ]

                text_prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = processor(
                    text=[text_prompt],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                )

                logger.debug("✓ Using messages format with chat template")

            except Exception as e:
                logger.debug(f"Messages format failed: {e}, trying standard format")
                inputs = None

        # Strategy 2: Standard text + images format (fallback)
        if inputs is None:
            try:
                inputs = processor(
                    text=full_question,
                    images=image,
                    return_tensors="pt"
                )
                logger.debug("✓ Using standard text + images format")
            except Exception as e:
                logger.error(f"Both input formats failed: {e}")
                raise RuntimeError(
                    f"Failed to prepare inputs for VLM inference.\n"
                    f"Processor: {type(processor).__name__}\n"
                    f"Error: {e}"
                )

        # Move inputs to model device
        inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # Warn if on CPU
        if str(model_device) == 'cpu':
            logger.warning("⏳ Running inference on CPU - this will be SLOW (30s-2min)")

        # Generate
        logger.info("Generating response...")
        max_tokens = 512 if str(model_device) == 'cpu' else 1024

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_tokens)

        logger.info("✓ Response generated")

        # Decode
        response = processor.batch_decode(output, skip_special_tokens=True)[0]

        # Extract answer
        if conversation_history and "A:" in response:
            parts = response.split("A:")
            if len(parts) > 1:
                response = parts[-1].strip()

        return response.strip()

    def _run_text_mlx(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        conversation_history: Optional[list[tuple[str, str]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float
    ) -> str:
        """Run text generation using MLX framework.

        Args:
            model: MLX model
            tokenizer: MLX tokenizer
            prompt: Input text
            conversation_history: Optional conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Repetition penalty (not used by MLX)

        Returns:
            Generated text
        """
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        from ..utils.history_formatter import format_conversation_history

        # Format prompt with history
        if conversation_history:
            formatted_prompt = format_conversation_history(
                conversation_history,
                prompt,
                max_turns=5
            )
        else:
            formatted_prompt = prompt

        logger.info("Running MLX LLM inference...")

        # Create sampler with MLX's make_sampler (uses 'temp' not 'temperature')
        # Note: repetition_penalty is not supported by MLX's make_sampler
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            min_p=0.0,
            min_tokens_to_keep=1
        )

        # MLX generate function - pass sampler instead of individual parameters
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False
        )

        logger.info("✓ MLX text response generated")

        # Clean response
        from ..utils.response_cleaner import clean_model_response
        if response:
            response = clean_model_response(response, aggressive=True)

        return response if response else "I don't have a response."

    def _run_text_transformers(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        conversation_history: Optional[list[tuple[str, str]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float
    ) -> str:
        """Run text generation using HuggingFace transformers.

        Args:
            model: Transformers model
            tokenizer: Transformers tokenizer
            prompt: Input text
            conversation_history: Optional conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Repetition penalty

        Returns:
            Generated text
        """
        import torch
        from ..utils.history_formatter import format_conversation_history

        # Get device
        model_device = next(model.parameters()).device

        # Build messages
        messages = []
        if conversation_history:
            for user_msg, bot_msg in conversation_history[-5:]:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": prompt})

        # Format with tokenizer's chat template if available
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # Fallback to simple format
            formatted_prompt = format_conversation_history(
                conversation_history,
                prompt,
                max_turns=5
            )

        # Tokenize and generate
        logger.debug(f"Tokenizing prompt (length: {len(formatted_prompt)})")
        inputs = tokenizer(formatted_prompt, return_tensors="pt")

        # Validate tokenization output
        if inputs is None or "input_ids" not in inputs:
            raise RuntimeError(f"Tokenization failed - tokenizer returned: {inputs}")

        if inputs["input_ids"] is None:
            raise RuntimeError("Tokenization failed - input_ids is None")

        logger.debug(f"Tokenized successfully: {inputs['input_ids'].shape}")

        # Move tensors to device individually and validate
        for key, value in list(inputs.items()):
            if value is None:
                logger.warning(f"Tokenizer returned None for '{key}', removing from inputs")
                inputs.pop(key)
            elif hasattr(value, 'to'):
                inputs[key] = value.to(model_device)

        # Ensure attention_mask exists (required for generation)
        if "attention_mask" not in inputs or inputs.get("attention_mask") is None:
            logger.warning("No attention_mask in inputs, creating one")
            import torch
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        # Remove token_type_ids if model doesn't use them (e.g., OLMo, Llama)
        # This prevents "model_kwargs not used" errors
        if "token_type_ids" in inputs and not getattr(model.config, "type_vocab_size", 0):
            inputs.pop("token_type_ids")

        logger.debug(f"Final inputs keys: {list(inputs.keys())}")

        # Prepare generation kwargs with proper token IDs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }

        # Only add pad_token_id if it's set
        if tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            # Fallback: use eos_token as pad_token
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

        # Only add eos_token_id if it's set
        if tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

        logger.debug(f"Generation kwargs: {gen_kwargs}")

        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            logger.debug(f"Generation complete. Output shape: {outputs.shape if outputs is not None else 'None'}")

            if outputs is None:
                raise RuntimeError("model.generate() returned None")

        except Exception as gen_error:
            logger.error(f"Generation failed: {gen_error}", exc_info=True)
            raise RuntimeError(f"Model generation failed: {gen_error}")

        # Decode only new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        if generated_tokens is None or len(generated_tokens) == 0:
            logger.warning("No tokens were generated")
            return "I don't have a response."

        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean response
        from ..utils.response_cleaner import clean_model_response
        if response:
            response = clean_model_response(response, aggressive=True)

        return response if response else "I don't have a response."

    def run_caption(
        self,
        model_handle: Any,
        image: Any,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        detail_level: str = "detailed"
    ) -> str:
        """Generate image caption with quantized VLM.
        
        Args:
            model_handle: Loaded model
            image: PIL Image
            conversation_history: Optional conversation history
            detail_level: "detailed" or "short"
            
        Returns:
            Caption text
            
        Raises:
            NotImplementedError: If model is not a VLM
        """
        prompt = (
            "Describe this image in detail."
            if detail_level == "detailed"
            else "Describe this image briefly."
        )
        return self.run_qa(model_handle, image, prompt, conversation_history)

    def run_detect(self, model_handle: Any, image: Any, object_name: str) -> list[dict]:
        """Detect objects with quantized VLM.
        
        Args:
            model_handle: Loaded model
            image: PIL Image
            object_name: Object to detect
            
        Returns:
            List of detections
            
        Raises:
            NotImplementedError: If model is not a VLM
        """
        from ..utils.vlm_response_parser import parse_detection_response
        
        prompt = (
            f"Detect all instances of '{object_name}' in this image. "
            f"Provide bounding box coordinates in JSON format as: "
            f'[{{"bbox": [x1, y1, x2, y2], "label": "{object_name}"}}]'
        )
        response = self.run_qa(model_handle, image, prompt)
        
        # Parse response
        detections = parse_detection_response(response, object_name)
        for detection in detections:
            detection["raw"] = response
        
        return detections

    def run_point(self, model_handle: Any, image: Any, object_name: str) -> dict:
        """Point to object location with quantized VLM.
        
        Args:
            model_handle: Loaded model
            image: PIL Image
            object_name: Object to locate
            
        Returns:
            Coordinates dict
            
        Raises:
            NotImplementedError: If model is not a VLM
        """
        from ..utils.vlm_response_parser import parse_point_response
        
        prompt = (
            f"Where is the '{object_name}' in this image? "
            f"Provide the center coordinates in JSON format as: "
            f'{{"x": <number>, "y": <number>}}'
        )
        response = self.run_qa(model_handle, image, prompt)
        
        # Parse response
        coordinates = parse_point_response(response, object_name)
        coordinates["raw"] = response
        
        return coordinates

    def run_text(
        self,
        model_handle: Any,
        prompt: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        custom_parameters: Optional[dict] = None
    ) -> str:
        """Run text generation on quantized model with conversation history support.

        Args:
            model_handle: Loaded model
            prompt: Input text
            conversation_history: Optional list of (user_msg, bot_response) tuples
            custom_parameters: Optional custom generation parameters (temperature, top_p, max_tokens, etc.)

        Returns:
            Generated text
        """
        # Extract parameters from custom_parameters
        # Increased default max_tokens for more detailed responses
        max_tokens = 1024  # Allow longer responses by default
        temperature = 0.7
        top_p = 0.9
        repetition_penalty = 1.1  # Prevent infinite loops
        frequency_penalty = 0.0
        presence_penalty = 0.0

        if custom_parameters:
            max_tokens = custom_parameters.get("max_tokens", max_tokens)
            temperature = custom_parameters.get("temperature", temperature)
            top_p = custom_parameters.get("top_p", top_p)
            repetition_penalty = custom_parameters.get("repetition_penalty", repetition_penalty)
            frequency_penalty = custom_parameters.get("frequency_penalty", frequency_penalty)
            presence_penalty = custom_parameters.get("presence_penalty", presence_penalty)

        # Check if this is a 3-tuple (new format with metadata)
        if isinstance(model_handle, tuple) and len(model_handle) == 3:
            model, tokenizer, metadata = model_handle
            is_mlx = metadata.get("is_mlx", False)

            # Route to appropriate inference method
            if is_mlx:
                return self._run_text_mlx(
                    model, tokenizer, prompt, conversation_history,
                    max_tokens, temperature, top_p, repetition_penalty
                )
            else:
                return self._run_text_transformers(
                    model, tokenizer, prompt, conversation_history,
                    max_tokens, temperature, top_p, repetition_penalty
                )

        # Backward compatibility: 2-tuple format (legacy)
        elif isinstance(model_handle, tuple) and len(model_handle) == 2:
            model, tokenizer = model_handle

            # Use HuggingFace transformers for text generation
            import torch
            from ..utils.history_formatter import format_conversation_history
            
            # Get device
            model_device = next(model.parameters()).device
            
            # Build messages
            messages = []
            if conversation_history:
                for user_msg, bot_msg in conversation_history[-5:]:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": bot_msg})
            messages.append({"role": "user", "content": prompt})
            
            # Format with tokenizer's chat template if available
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback to simple format
                formatted_prompt = format_conversation_history(
                    conversation_history,
                    prompt,
                    max_turns=5
                )
            
            # Tokenize and generate
            logger.debug(f"Tokenizing prompt (length: {len(formatted_prompt)})")
            inputs = tokenizer(formatted_prompt, return_tensors="pt")

            # Validate tokenization output
            if inputs is None or "input_ids" not in inputs:
                raise RuntimeError(f"Tokenization failed - tokenizer returned: {inputs}")

            if inputs["input_ids"] is None:
                raise RuntimeError("Tokenization failed - input_ids is None")

            logger.debug(f"Tokenized successfully: {inputs['input_ids'].shape}")

            # Move tensors to device individually and validate
            for key, value in list(inputs.items()):
                if value is None:
                    logger.warning(f"Tokenizer returned None for '{key}', removing from inputs")
                    inputs.pop(key)
                elif hasattr(value, 'to'):
                    inputs[key] = value.to(model_device)

            # Ensure attention_mask exists (required for generation)
            if "attention_mask" not in inputs or inputs.get("attention_mask") is None:
                logger.warning("No attention_mask in inputs, creating one")
                import torch
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            # Remove token_type_ids if model doesn't use them (e.g., OLMo, Llama)
            if "token_type_ids" in inputs and not getattr(model.config, "type_vocab_size", 0):
                inputs.pop("token_type_ids")

            logger.debug(f"Final inputs keys: {list(inputs.keys())}")

            # Prepare generation kwargs with proper token IDs
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
            }

            # Only add pad_token_id if it's set
            if tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
            elif tokenizer.eos_token_id is not None:
                # Fallback: use eos_token as pad_token
                gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

            # Only add eos_token_id if it's set
            if tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

            logger.debug(f"Generation kwargs: {gen_kwargs}")

            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)

                logger.debug(f"Generation complete. Output shape: {outputs.shape if outputs is not None else 'None'}")

                if outputs is None:
                    raise RuntimeError("model.generate() returned None")

            except Exception as gen_error:
                logger.error(f"Generation failed: {gen_error}", exc_info=True)
                raise RuntimeError(f"Model generation failed: {gen_error}")

            # Decode only new tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]

            if generated_tokens is None or len(generated_tokens) == 0:
                logger.warning("No tokens were generated")
                return "I don't have a response."

            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Clean response
            from ..utils.response_cleaner import clean_model_response
            if response:
                response = clean_model_response(response, aggressive=True)
            
            return response if response else "I don't have a response."
        
        elif hasattr(model_handle, "__call__"):
            # llama-cpp-python Llama model (GGUF)
            # Format conversation history if provided
            if conversation_history:
                # Use SIMPLE template with explicit system prompt for conversation context
                from ..utils.history_formatter import format_conversation_history, ChatTemplate

                # Add system prompt - clear and balanced for all model types
                system_prompt = (
                    "You are a helpful AI assistant. Provide clear, direct answers to the user's questions. "
                    "Keep responses focused and relevant to what was asked."
                )

                formatted_prompt = format_conversation_history(
                    conversation_history,
                    prompt,
                    max_turns=5,
                    system_prompt=system_prompt,
                    template=ChatTemplate.SIMPLE
                )
            else:
                # For simple prompts without history, use clear instruction format
                formatted_prompt = (
                    "You are a helpful AI assistant. Provide clear, direct answers.\n\n"
                    f"User: {prompt}\nAssistant:"
                )

            # Define stop tokens to prevent template leakage and repetition
            # Be careful not to interfere with ongoing generation
            stop_tokens = [
                "\n\nUser:",          # Stop before next user turn (with double newline)
                "\n\nHuman:",         # Alternative role marker
                "\n### Instruction:", # Stop at new instruction blocks
                "<|im_end|>",        # ChatML end token
                "<|endoftext|>",     # Common end token
            ]

            # Allow custom stop tokens from parameters
            if custom_parameters and "stop" in custom_parameters:
                custom_stop = custom_parameters.get("stop")
                if isinstance(custom_stop, list):
                    stop_tokens.extend(custom_stop)
                elif isinstance(custom_stop, str):
                    stop_tokens.append(custom_stop)

            response = model_handle(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_tokens,
                echo=False,
            )

            # Extract and clean the response text using shared utility
            from ..utils.response_cleaner import clean_model_response

            response_text = response["choices"][0]["text"]
            cleaned_response = clean_model_response(response_text, aggressive=True)

            return cleaned_response
        else:
            # HuggingFace model
            from transformers import AutoTokenizer

            # For HF models, we need the model path to load tokenizer
            # This should be passed in custom_parameters or we need to store it
            model_path = custom_parameters.get("model_path") if custom_parameters else None
            if not model_path:
                raise ValueError("model_path required for HF model inference")

            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Format with conversation history if provided
            if conversation_history:
                # Build conversation for HF models
                messages = []
                for user_msg, bot_msg in conversation_history:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": bot_msg})
                messages.append({"role": "user", "content": prompt})

                # Use apply_chat_template if available
                if hasattr(tokenizer, "apply_chat_template"):
                    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    # Fallback: simple concatenation
                    formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            else:
                formatted_prompt = prompt

            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            outputs = model_handle.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_model_info(self, model_id: str) -> dict:
        """Get detailed model information for quantized models.

        Args:
            model_id: Model identifier

        Returns:
            Dict with model metadata including default_parameters
        """
        # Find the model in discovered models
        models = self.discover_models()
        model = next((m for m in models if m.model_id == model_id), None)

        if not model:
            # Return minimal info if not found
            return {
                "model_id": model_id,
                "name": model_id,
                "provider": "quantized",
                "default_parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": 1024,  # Updated default for quantized models
                    "repeat_penalty": 1.1,
                }
            }

        # Return full model info
        return {
            "model_id": model.model_id,
            "name": model.name,
            "provider": "quantized",
            "architecture": model.architecture or "Unknown",
            "quantization": model.quantization or "Unknown",
            "size_gb": model.size_gb,
            "model_type": str(model.model_type).upper() if hasattr(model.model_type, 'value') else str(model.model_type).upper(),
            "capabilities": [str(cap) for cap in model.capabilities] if model.capabilities else ["text"],
            "default_parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 1024,  # Updated default for quantized models
                "repeat_penalty": 1.1,
            }
        }

    def estimate_model_size(self, model_name: str) -> float:
        """Estimate model size (already quantized, so return actual size).

        Args:
            model_name: Model name

        Returns:
            Model size in GB
        """
        models = self.discover_models()
        model = next((m for m in models if model_name in m.model_id), None)
        return model.size_gb if model else 0.0

    def install_model(self, model_name: str, **kwargs) -> bool:
        """Cannot install models - they are created via quantization."""
        logger.warning("Cannot install quantized models - use quantization feature")
        return False

    def delete_model(self, model_id: str) -> bool:
        """Delete a quantized model.

        Args:
            model_id: Model identifier (format: "quantized:gguf:/path" or "quantized:hf:/path")

        Returns:
            True if deleted successfully
        """
        try:
            # Parse model_id to extract path
            parts = model_id.split(":", 2)
            if len(parts) != 3 or parts[0] != "quantized":
                logger.error(f"Invalid model_id format: {model_id}")
                return False

            model_path = Path(parts[2])

            # Handle both relative and absolute paths
            # If path is relative, make it absolute (for backward compatibility)
            if not model_path.is_absolute():
                model_path = model_path.absolute()

            if not model_path.exists():
                logger.error(f"Model path does not exist: {model_path}")
                return False

            if model_path.is_file():
                # Delete GGUF file
                model_path.unlink()
                logger.info(f"Deleted GGUF file: {model_path}")

                # Delete metadata if exists
                metadata_file = model_path.with_suffix(".json")
                if metadata_file.exists():
                    metadata_file.unlink()
                    logger.info(f"Deleted metadata file: {metadata_file}")

                # Delete F16 intermediate file if exists (GGUF quantization creates these)
                f16_file = model_path.parent / f"{model_path.stem}_f16.gguf"
                if f16_file.exists():
                    f16_file.unlink()
                    logger.info(f"Deleted F16 intermediate file: {f16_file}")

            elif model_path.is_dir():
                # Delete HF/MLX directory
                import shutil
                try:
                    shutil.rmtree(model_path)
                    logger.info(f"Deleted model directory: {model_path}")
                except PermissionError as pe:
                    logger.error(f"Permission denied deleting {model_path}: {pe}")
                    logger.info("Trying to delete files individually...")
                    # Try to delete files individually
                    for item in model_path.rglob("*"):
                        if item.is_file():
                            try:
                                item.unlink()
                            except Exception:
                                pass
                    # Try to remove directory again
                    try:
                        shutil.rmtree(model_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Could not fully clean up {model_path}: {cleanup_error}")

            # Update model cache to remove this model
            try:
                self.cache.invalidate_model(str(model_path))
            except Exception as cache_error:
                logger.debug(f"Could not invalidate cache for {model_path}: {cache_error}")

            logger.info(f"Successfully deleted quantized model: {model_id}")
            return True

        except FileNotFoundError as fnf:
            # Model already deleted or never existed
            logger.warning(f"Model path not found (already deleted?): {fnf}")
            # Still return True since the goal (model not existing) is achieved
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}", exc_info=True)
            return False
