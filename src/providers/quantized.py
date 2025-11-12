"""Quantized models provider - discovers models from results/quantizations/."""

import gc
import json
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from ..models.endpoints import CompatibilityStatus, ModelType, ProviderType
from ..models.model import ModelInfo
from ..models.model_cache import ModelMetadataCache
from ..models.provider import ProviderConfig
from ..models.system import SystemSpecs
from ..services.llama_server_manager import LlamaServerManager
from ..services.llama_cli_server_manager import LlamaCLIServerManager
from ..config.server_config import get_llama_server_type
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

        # llama-server instances for GGUF models (enables GPU acceleration + KV cache reuse)
        self.llama_servers: Dict[str, Any] = {}  # model_path -> LlamaServerManager
        self.use_llama_server = True  # Always use server for GGUF models (GPU + cache)
        # Session-scoped user selection of mmproj per loaded language model
        self._session_mmproj_choice: Dict[str, Path] = {}
        # Cache which vision API format works for each model (to avoid repeated 400/500 errors)
        self._vision_api_format_cache: Dict[str, str] = {}  # model_path -> "openai" or "llamacpp"
        # VLM image rescaling preferences (per-model, session-scoped)
        # Maps model_id -> (width, height) or None to disable
        self._rescale_resolution: Dict[str, Optional[tuple[int, int]]] = {}

        logger.info(f"QuantizedProvider initialized: {self.quantized_dir} (llama-server mode: ON)")

    def _cleanup_memory(self, device: str):
        """Clean up memory before CPU fallback to maximize available RAM.

        Args:
            device: Current device ("mps", "cuda", or "cpu")
        """
        # Force garbage collection
        gc.collect()
        logger.debug("Garbage collection triggered")

        # Clear GPU cache if on GPU
        if device in ["cuda", "mps"]:
            try:
                import torch
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared")
                elif device == "mps":
                    # MPS doesn't have explicit cache clearing, but we can still GC
                    logger.debug("MPS memory freed via garbage collection")
            except Exception as e:
                logger.debug(f"Could not clear GPU cache: {e}")

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

        # Discover GGUF files (exclude resource forks and mmproj component files)
        for gguf_file in self.quantized_dir.glob("*.gguf"):
            # Skip macOS resource forks (._* files)
            if gguf_file.name.startswith("._"):
                continue

            # Skip mmproj files (VLM vision encoder components - auto-detected when loading)
            if gguf_file.name.startswith("mmproj-"):
                logger.debug(f"Skipping mmproj component file: {gguf_file.name}")
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
        # Import required types at the top
        from ..models.endpoints import EndpointType

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

        # Display name: Use raw filename stem (no complex parsing)
        # Example: "Qwen_Qwen3-VL-4B-Instruct_q4_k_m_f16.gguf" → "Qwen_Qwen3-VL-4B-Instruct_q4_k_m_f16"
        name = gguf_path.stem

        # Detect if this is a VLM using robust 2-step verification
        # Step 1: Check actual files/metadata (most reliable)
        # Step 2: Fallback to name patterns (if step 1 fails)
        # If either passes → VLM, if both fail → LLM
        model_type = ModelType.LLM  # Default
        capabilities = [EndpointType.TEXT]  # Default
        detection_method = None

        # ============ STEP 1: ROBUST FILE-BASED DETECTION ============

        # 1A. Check for mmproj file (MOST RELIABLE - VLMs always need vision encoder)
        if model_type == ModelType.LLM:
            mmproj_patterns = [
                f"mmproj-{gguf_path.stem}.gguf",
                f"mmproj-{gguf_path.stem}_f16.gguf",
                f"mmproj-{gguf_path.stem.replace('_f16', '')}_f16.gguf",
                # Check variations without language suffix
                f"mmproj-{gguf_path.stem.replace('_language', '')}.gguf",
                f"mmproj-{gguf_path.stem.replace('_language', '')}_f16.gguf",
            ]
            for pattern in mmproj_patterns:
                mmproj_path = gguf_path.parent / pattern
                if mmproj_path.exists():
                    model_type = ModelType.VLM
                    capabilities = [
                        EndpointType.QA,
                        EndpointType.CAPTION,
                        EndpointType.DETECT,
                        EndpointType.POINT,
                        EndpointType.TEXT,
                    ]
                    detection_method = f"mmproj file present: {pattern}"
                    logger.debug(f"✓ VLM detected - {detection_method}")
                    break

        # 1B. Check cached metadata model_type (from GGUF header or config.json)
        if model_type == ModelType.LLM and cached_metadata:
            if cached_metadata.model_type == "vlm":
                model_type = ModelType.VLM
                capabilities = [
                    EndpointType.QA,
                    EndpointType.CAPTION,
                    EndpointType.DETECT,
                    EndpointType.POINT,
                    EndpointType.TEXT,
                ]
                detection_method = "cached metadata model_type=vlm"
                logger.debug(f"✓ VLM detected - {detection_method}")

        # 1C. Check GGUF architecture field (from header)
        if model_type == ModelType.LLM and cached_metadata and cached_metadata.architecture:
            arch_lower = cached_metadata.architecture.lower()
            vlm_architectures = [
                'qwen3vl', 'qwen2vl', 'qwen2_5vl', 'qwenvl',
                'llava', 'minicpmv', 'minicpmo',
                'pixtral', 'smolvlm', 'smolvlm2',
                'internvl', 'moondream', 'cogvlm',
                'clip',  # Vision encoder architecture
            ]
            if any(vlm_arch in arch_lower for vlm_arch in vlm_architectures):
                model_type = ModelType.VLM
                capabilities = [
                    EndpointType.QA,
                    EndpointType.CAPTION,
                    EndpointType.DETECT,
                    EndpointType.POINT,
                    EndpointType.TEXT,
                ]
                detection_method = f"GGUF architecture: {cached_metadata.architecture}"
                logger.debug(f"✓ VLM detected - {detection_method}")

        # 1D. Check source model config (if quantized from HF model)
        if model_type == ModelType.LLM and metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    source_model_path = metadata.get("source_model_path")

                    if source_model_path:
                        from ..models.vlm_detector import VLMDetector
                        from pathlib import Path as P

                        source_path = P(source_model_path)
                        if source_path.exists():
                            vlm_info = VLMDetector.detect(source_path)
                            if vlm_info.is_vlm:
                                model_type = ModelType.VLM
                                capabilities = [
                                    EndpointType.QA,
                                    EndpointType.CAPTION,
                                    EndpointType.DETECT,
                                    EndpointType.POINT,
                                    EndpointType.TEXT,
                                ]
                                detection_method = f"source model VLM detector: {vlm_info.architecture.display_name}"
                                logger.debug(f"✓ VLM detected - {detection_method}")
            except Exception as e:
                logger.debug(f"Could not check source model: {e}")

        # ============ STEP 2: FALLBACK TO NAME PATTERNS ============
        if model_type == ModelType.LLM:
            name_lower = gguf_path.stem.lower()

            # VLM keyword patterns (comprehensive list)
            vlm_name_patterns = [
                '-vl-', '_vl_', '.vl.',  # Generic VL patterns
                'qwen2-vl', 'qwen2.5-vl', 'qwen3-vl', 'qwenvl',
                'llava', 'minicpm-v', 'minicpmv', 'moondream',
                'internvl', 'cogvlm', 'paligemma', 'idefics',
                'vision', 'multimodal', 'vlm',
            ]

            if any(pattern in name_lower for pattern in vlm_name_patterns):
                model_type = ModelType.VLM
                capabilities = [
                    EndpointType.QA,
                    EndpointType.CAPTION,
                    EndpointType.DETECT,
                    EndpointType.POINT,
                    EndpointType.TEXT,
                ]
                detection_method = f"filename pattern match"
                logger.debug(f"✓ VLM detected (fallback) - {detection_method}: {gguf_path.name}")

        # Final fallback: If still LLM, log it
        if model_type == ModelType.LLM:
            logger.debug(f"✗ Classified as LLM (no VLM indicators found): {gguf_path.name}")

        # Assess compatibility
        compatibility, compatibility_message = self._assess_compatibility_detailed(size_gb)

        return ModelInfo(
            model_id=model_id,
            name=name,
            provider=ProviderType.QUANTIZED,
            model_type=model_type,
            size_gb=size_gb,
            params_billions=params_billions,  # From metadata cache
            ram_gb=ram_gb,  # From metadata cache
            capabilities=capabilities,
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

        # ============ ROBUST 2-STEP VLM DETECTION ============
        # Step 1: Check actual files/metadata (most reliable)
        # Step 2: Fallback to name patterns (if step 1 fails)
        detection_method = None

        # STEP 1: File-based detection

        # 1A. Check cached metadata model_type
        if cached_metadata and cached_metadata.model_type == "vlm":
            model_type = ModelType.VLM
            is_vlm = True
            detection_method = "cached metadata model_type=vlm"
            logger.debug(f"✓ VLM detected - {detection_method}")

        # 1B. Check config.json for VLM indicators
        if not is_vlm:
            config_path = model_dir / "config.json"
            if config_path.exists():
                try:
                    import json
                    with open(config_path) as f:
                        config = json.load(f)

                    # Check architecture class names
                    arch = config.get("architectures", [""])[0] if config.get("architectures") else ""
                    vlm_arch_patterns = [
                        "ForConditionalGeneration", "VisionTextDual", "VisionEncoder",
                        "Llava", "LLaVA", "Blip", "BLIP", "Qwen2VL", "Qwen3VL",
                        "QwenVL", "InstructBlip", "MiniCPM", "Moondream",
                        "InternVL", "CogVLM", "PaliGemma", "Idefics"
                    ]
                    if any(pattern in arch for pattern in vlm_arch_patterns):
                        model_type = ModelType.VLM
                        is_vlm = True
                        detection_method = f"config.json architecture: {arch}"
                        logger.debug(f"✓ VLM detected - {detection_method}")

                    # Check for vision config keys in config
                    if not is_vlm:
                        vision_keys = ["vision_config", "visual_config", "image_encoder", "vision_tower", "mm_vision_tower"]
                        if any(key in config for key in vision_keys):
                            model_type = ModelType.VLM
                            is_vlm = True
                            detection_method = f"config.json has vision keys"
                            logger.debug(f"✓ VLM detected - {detection_method}")

                    # Check model_type field
                    if not is_vlm and config.get("model_type"):
                        model_type_str = config["model_type"].lower()
                        if "vision" in model_type_str or "vlm" in model_type_str or "multimodal" in model_type_str:
                            model_type = ModelType.VLM
                            is_vlm = True
                            detection_method = f"config.json model_type: {config['model_type']}"
                            logger.debug(f"✓ VLM detected - {detection_method}")

                except Exception as e:
                    logger.debug(f"Could not check config.json for VLM detection: {e}")

        # 1C. Check for vision-related files in directory
        if not is_vlm:
            vision_files = ["preprocessor_config.json", "processor_config.json"]
            for vision_file in vision_files:
                if (model_dir / vision_file).exists():
                    model_type = ModelType.VLM
                    is_vlm = True
                    detection_method = f"vision file present: {vision_file}"
                    logger.debug(f"✓ VLM detected - {detection_method}")
                    break

        # STEP 2: Fallback to name patterns
        if not is_vlm:
            name_lower = model_dir.name.lower()
            vlm_name_patterns = [
                '-vl-', '_vl_', '.vl.',
                'qwen2-vl', 'qwen2.5-vl', 'qwen3-vl', 'qwenvl',
                'llava', 'minicpm-v', 'minicpmv', 'moondream',
                'internvl', 'cogvlm', 'paligemma', 'idefics',
                'vision', 'multimodal', 'vlm',
            ]

            if any(pattern in name_lower for pattern in vlm_name_patterns):
                model_type = ModelType.VLM
                is_vlm = True
                detection_method = "filename pattern match"
                logger.debug(f"✓ VLM detected (fallback) - {detection_method}: {model_dir.name}")

        # Final: Log if still LLM
        if not is_vlm:
            logger.debug(f"✗ Classified as LLM (no VLM indicators found): {model_dir.name}")

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

        # Display name: Use raw directory name (no complex parsing)
        # Example: "Qwen_Qwen3-VL-4B-Instruct_mlx-int4" → "Qwen_Qwen3-VL-4B-Instruct_mlx-int4"
        name = model_dir.name

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
        # O(1) time and space complexity - simple division and comparison
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
                f"Model is {ratio:.1f}x larger than recommended ({model_size_gb:.1f}GB vs {recommended_size:.1f}GB). May cause OOM errors."
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

    def _normalize_vlm_base(self, name: str) -> str:
        """Normalize a VLM family base name for robust mmproj↔language matching.

        Strategy:
        - Lowercase and unify separators to underscores
        - Trim leading "mmproj-" if present
        - Iteratively strip trailing role/quantization suffixes (supports multi-token like q4_k_m)
        - Return the stable family base (e.g., "qwen_qwen3-vl-2b-instruct")
        """
        import re

        n = name.lower()
        if n.startswith("mmproj-"):
            n = n[len("mmproj-"):]

        # Unify separators
        n = re.sub(r"[\s\-/]+", "_", n)
        tokens = [t for t in n.split("_") if t]

        # Known suffix tokens
        role_suffixes = {
            "language", "vision", "text", "model", "encoder", "projector", "mmproj",
        }
        quant_suffixes = {
            # 3-token quant patterns
            "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q3_k_m", "q3_k_s", "q3_k_l",
            # 2-token
            "q5_0", "q5_1", "q4_0", "q4_1", "q3_k", "q2_k", "q6_k", "q8_0",
            # IQ/TQ variants
            "iq4_xs", "iq4_nl", "iq3_xs", "iq3_xxs", "iq3_s", "iq3_m",
            "iq2_xs", "iq2_xxs", "iq2_s", "iq2_m", "iq1_m", "iq1_s",
            "tq1_0", "tq2_0",
            # Precision
            "bf16", "fp16", "f16", "f32",
        }

        def strip_suffixes(parts: list[str]) -> list[str]:
            changed = True
            while parts and changed:
                changed = False
                # Try multi-token quant suffixes first (3, then 2)
                if len(parts) >= 3 and "_".join(parts[-3:]) in quant_suffixes:
                    parts = parts[:-3]
                    changed = True
                    continue
                if len(parts) >= 2 and "_".join(parts[-2:]) in quant_suffixes:
                    parts = parts[:-2]
                    changed = True
                    continue
                # Single-token suffix (quant or role)
                if parts[-1] in role_suffixes or parts[-1] in quant_suffixes:
                    parts = parts[:-1]
                    changed = True
            return parts

        base_tokens = strip_suffixes(tokens)
        return "_".join(base_tokens)

    def _find_mmproj_candidates(self, model_path: Path) -> list[Path]:
        """Find all mmproj candidates across results/quantizations for this language model.

        Scans the entire quantization library for files matching the normalized base name.
        """
        if not model_path.exists() or not model_path.is_file():
            return []

        lang_base = self._normalize_vlm_base(model_path.stem)
        candidates: list[Path] = []

        # Search entire quantized_dir recursively for mmproj-*.gguf (skip macOS resource forks)
        for mmproj in self.quantized_dir.rglob("mmproj-*.gguf"):
            if mmproj.name.startswith("._"):
                continue
            # Normalize candidate base: remove prefix and normalize
            cand_base = self._normalize_vlm_base(mmproj.stem[len("mmproj-"):])
            # Strict match or robust containment in either direction
            if cand_base == lang_base or cand_base in lang_base or lang_base in cand_base:
                candidates.append(mmproj)

        # Also check local directory patterns (backward compatibility)
        local_single = self._find_mmproj_file_legacy(model_path)
        if local_single and local_single not in candidates:
            candidates.append(local_single)

        # De-duplicate while preserving order
        uniq = []
        seen = set()
        for p in candidates:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq

    def _find_mmproj_file_legacy(self, model_path: Path) -> Optional[Path]:
        """Legacy local-directory mmproj finder for backward compatibility."""
        try:
            model_dir = model_path.parent
            model_name = model_path.stem
            mmproj_path = model_dir / f"mmproj-{model_name}.gguf"
            if mmproj_path.exists():
                return mmproj_path
            base_name = model_name.replace("_language", "")
            for mmproj_candidate in model_dir.glob(f"mmproj-{base_name}*.gguf"):
                return mmproj_candidate
            metadata_file = model_path.with_suffix('.json')
            if metadata_file.exists():
                import json
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    if "vlm_vision_file" in metadata:
                        path = Path(metadata["vlm_vision_file"])
                        if path.exists():
                            return path
        except Exception:
            pass
        return None

    def _find_language_candidates(self, model_path: Path) -> list[Path]:
        """Find candidate language-decoder GGUF files that match the selected model family.

        Excludes mmproj files and excludes CLIP/vision-only architectures.
        """
        lang_base = self._normalize_vlm_base(model_path.stem)
        candidates: list[Path] = []

        for gguf in self.quantized_dir.rglob("*.gguf"):
            if gguf.name.startswith("._"):
                continue
            if gguf.name.startswith("mmproj-"):
                continue
            # Normalize base and filter by family
            cand_base = self._normalize_vlm_base(gguf.stem)
            if not (cand_base == lang_base or cand_base in lang_base or lang_base in cand_base):
                continue
            # Inspect architecture and exclude CLIP/projector
            meta = self.metadata_cache.get_metadata(str(gguf), provider="gguf")
            arch = (meta.architecture or "unknown").lower() if meta else "unknown"
            if "clip" in arch:
                continue
            candidates.append(gguf)

        # Ensure current path is included if it is NOT a CLIP/projector
        if model_path.exists() and model_path.is_file():
            try:
                meta = self.metadata_cache.get_metadata(str(model_path), provider="gguf")
                arch = (meta.architecture or "unknown").lower() if meta else "unknown"
                if "clip" not in arch and model_path not in candidates:
                    candidates.insert(0, model_path)
            except Exception:
                pass
        # De-duplicate
        seen = set()
        uniq: list[Path] = []
        for p in candidates:
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq

    def _load_gguf_model(self, model_path: Path, device: str) -> Any:
        """Load GGUF quantized model using llama-server for GPU acceleration.

        Uses llama-server instead of direct llama-cpp-python loading to enable:
        - Full GPU utilization (matches llama.cpp WebUI performance)
        - Persistent KV cache across conversation turns
        - 40+ tokens/s performance on follow-up messages

        Args:
            model_path: Path to .gguf file
            device: Target device ("cuda", "mps", or "cpu")

        Returns:
            LlamaServerManager instance (acts as model handle)
        """
        try:
            logger.info(f"Loading GGUF model with llama-server: {model_path}")

            # Check if server already running for this model
            model_key = str(model_path)
            if model_key in self.llama_servers:
                server = self.llama_servers[model_key]
                if server.is_running():
                    logger.info(f"Reusing existing llama-server for {model_path.name}")
                    return server

            # Determine GPU layers based on device
            n_gpu_layers = -1 if device in ["cuda", "mps"] else 0

            # Context length (n_ctx): prompt user in interactive mode, use default in benchmark mode
            import os
            min_ctx, max_ctx = 256, 32768
            default_ctx = 4096
            n_ctx = default_ctx

            try:
                if not os.getenv("EKAM_BENCHMARKING"):
                    # Interactive mode: always prompt user
                    from ..cli.text_input import professional_prompt
                    from ..cli.tui_manager import tui
                    tui.console.print("\n[cyan]Context Window (n_ctx)[/cyan] — tokens kept in memory per request")
                    tui.console.print(
                        f"[dim]Min:[/dim] {min_ctx}   [dim]Max:[/dim] {max_ctx}   "
                        f"[dim]Default:[/dim] {default_ctx}   [dim]Recommended:[/dim] {default_ctx}"
                    )
                    tui.console.print("[dim]Higher values increase RAM/VRAM usage and may reduce throughput.[/dim]")
                    n_ctx = professional_prompt.get_numeric(
                        "Context length (tokens)",
                        min_val=min_ctx,
                        max_val=max_ctx,
                        default=default_ctx,
                        style="cyan",
                    )
                else:
                    # Benchmark mode: use env var if set, otherwise default
                    env_ctx = os.getenv("EKAM_N_CTX")
                    if env_ctx:
                        n_ctx = max(min_ctx, min(max_ctx, int(env_ctx)))

                logger.info(
                    f"Using context length (n_ctx): {n_ctx} tokens (min {min_ctx}, max {max_ctx}; recommended {default_ctx})"
                )
            except Exception as e:
                logger.warning(f"Error setting context length: {e}. Using default {default_ctx}")
                n_ctx = default_ctx

            # PERFORMANCE: Use optimized thread count (physical cores, capped at 8)
            physical_cores = self.system_specs.cpu_cores_physical
            optimal_threads = min(physical_cores, 8)
            optimal_threads_batch = self.system_specs.cpu_cores_physical

            # Find available port
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                port = s.getsockname()[1]

            # Sanity check: ensure main model is not a CLIP/projector file
            try:
                lang_meta = self.metadata_cache.get_metadata(str(model_path), provider="gguf")
            except Exception:
                lang_meta = None

            lang_arch = (lang_meta.architecture or "unknown").lower() if lang_meta else "unknown"
            if "clip" in lang_arch:
                from ..cli.text_input import professional_prompt
                from ..cli.tui_manager import tui
                tui.console.print("[yellow]\nWarning: The selected GGUF appears to be a vision projector (CLIP).\nIt cannot be used as the main language model.[/yellow]")

                lang_candidates = self._find_language_candidates(model_path)
                if not lang_candidates:
                    raise RuntimeError(
                        "Selected file is a CLIP projector. No matching language decoder GGUF found.\n"
                        "Please re-quantize with the latest EKAM (fixes component roles), then try again."
                    )

                options = []
                for p in lang_candidates:
                    meta = self.metadata_cache.get_metadata(str(p), provider="gguf")
                    size_gb = p.stat().st_size / (1024 ** 3)
                    arch = (meta.architecture or "unknown").upper() if meta else "UNKNOWN"
                    label = f"[green]{p.name}[/green]"
                    desc = f"{size_gb:.2f} GB • {arch}"
                    options.append((str(p), label, desc))

                choice = professional_prompt.get_arrow_selection(
                    options=options,
                    title="Select Language Decoder GGUF",
                    instructions="Use ↑/↓ arrows, Enter to select, q to cancel",
                )
                if choice is None:
                    raise RuntimeError("Model load cancelled: language decoder selection required")
                model_path = Path(choice)
                logger.info(f"User selected language decoder: {model_path.name}")

            # Check for mmproj file(s) (VLM vision encoder)
            mmproj_path: Optional[Path] = None

            # Reuse session selection if present for this language model
            model_key = str(model_path)
            if model_key in self._session_mmproj_choice:
                sel = self._session_mmproj_choice[model_key]
                if sel.exists():
                    mmproj_path = sel
                    logger.info(f"Using previously selected mmproj: {mmproj_path.name}")
                else:
                    # Clear stale selection
                    del self._session_mmproj_choice[model_key]

            if mmproj_path is None:
                candidates = self._find_mmproj_candidates(model_path)
                if len(candidates) == 1:
                    mmproj_path = candidates[0]
                    logger.info(f"Detected VLM - using mmproj file: {mmproj_path.name}")
                elif len(candidates) > 1:
                    # Interactive selection using arrow keys
                    from ..cli.text_input import professional_prompt
                    from ..cli.tui_manager import tui
                    import sys
                    import os

                    # Build option tuples (value, label, description)
                    options = []
                    for p in candidates:
                        meta = self.metadata_cache.get_metadata(str(p), provider="gguf")
                        size_gb = p.stat().st_size / (1024 ** 3)
                        quant = (meta.quantization.upper() if meta else "UNKNOWN")
                        label = f"[green]{p.name}[/green]"
                        desc = f"{size_gb:.2f} GB • {quant}"
                        options.append((str(p), label, desc))

                    # Non-interactive (CI/benchmarking): auto-select highest precision
                    if (not sys.stdin.isatty()) or os.getenv("EKAM_BENCHMARKING") == "1":
                        tui.console.print("[yellow]Non-interactive: selecting highest-precision mmproj automatically[/yellow]")
                        def rank(q: str) -> int:
                            order = [
                                "f32", "f16", "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q5_1", "q5_0",
                                "q4_k_m", "q4_k_s", "q4_1", "q4_0", "q3_k_m", "q3_k_s", "q2_k",
                            ]
                            ql = q.lower()
                            return order.index(ql) if ql in order else len(order)
                        # Pick best by quant rank, then by largest size
                        def key_for(pth: Path):
                            m = self.metadata_cache.get_metadata(str(pth), provider="gguf")
                            q = m.quantization if m else "unknown"
                            return (rank(q), -pth.stat().st_size)
                        mmproj_path = sorted(candidates, key=key_for)[0]
                        logger.info(f"Auto-selected mmproj: {mmproj_path.name}")
                    else:
                        import os as _os
                        # In benchmarking mode, never prompt — auto-select best to avoid UI interruption
                        if _os.getenv("EKAM_BENCHMARK_MODE"):
                            def _rank(q: str) -> int:
                                order = [
                                    "f32", "f16", "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q5_1", "q5_0",
                                    "q4_k_m", "q4_k_s", "q4_1", "q4_0", "q3_k_m", "q3_k_s", "q2_k",
                                ]
                            
                                ql = q.lower()
                                return order.index(ql) if ql in order else len(order)
                            def _key_for(pth: Path):
                                m = self.metadata_cache.get_metadata(str(pth), provider="gguf")
                                q = getattr(m, 'quantization', 'unknown') if m else 'unknown'
                                return (_rank(q), -pth.stat().st_size)
                            mmproj_path = sorted(candidates, key=_key_for)[0]
                            logger.info(f"Auto-selected mmproj (benchmark mode): {mmproj_path.name}")
                        else:
                            tui.console.print("\n[cyan]Multiple vision encoders found for this model[/cyan]")
                            tui.console.print("Select which mmproj (vision encoder) to use:")
                            choice = professional_prompt.get_arrow_selection(
                                options=options,
                                title="Select Vision Encoder (mmproj)",
                                instructions="Use ↑/↓ arrows, Enter to select, q to cancel",
                            )
                            if choice is None:
                                # User cancelled: abort load
                                raise RuntimeError("Model load cancelled: mmproj selection required")
                            mmproj_path = Path(choice)
                            logger.info(f"User selected mmproj: {mmproj_path.name}")

                    # Remember for this session until unload
                    if mmproj_path is not None:
                        self._session_mmproj_choice[model_key] = mmproj_path

            # VLM Image Rescaling Configuration (only for VLM models)
            # Ask user for target resolution to normalize input images
            is_vlm_model = mmproj_path is not None
            rescale_width, rescale_height = None, None
            model_id_for_rescale = str(model_path)  # Use full path as key
            if is_vlm_model and model_id_for_rescale not in self._rescale_resolution:
                # Default rescaling resolution
                default_resolution = "1280x1024"
                try:
                    # In benchmark mode, resolution is set via benchmark config (not here)
                    # In regular inference mode, prompt user
                    if not os.getenv("EKAM_BENCHMARK_MODE"):
                        from ..cli.text_input import professional_prompt
                        from ..cli.tui_manager import tui
                        tui.clear_screen()
                        tui.console.print("\n[cyan]VLM Image Rescaling[/cyan] — normalize input image resolution")
                        tui.console.print(
                            f"[dim]Images will be rescaled to consistent resolution (maintains aspect ratio with padding)[/dim]"
                        )
                        tui.console.print(
                            f"[dim]Default:[/dim] {default_resolution}   [dim]Range:[/dim] 256x256 to 4096x4096"
                        )
                        tui.console.print(
                            "[dim]Enter resolution as WIDTHxHEIGHT (e.g., 1280x1024) or press Enter for default[/dim]"
                        )

                        resolution_input = professional_prompt.get_input(
                            f"Image resolution (default: {default_resolution})",
                            style="cyan",
                            allow_multiline=False,
                            show_instructions=False
                        )

                        # Use default if user pressed Enter without input
                        if not resolution_input.strip():
                            resolution_input = default_resolution

                        # Parse resolution (WIDTHxHEIGHT)
                        import re
                        match = re.match(r'(\d+)x(\d+)', resolution_input.strip())
                        if match:
                            rescale_width, rescale_height = int(match.group(1)), int(match.group(2))
                            # Validate range (256-4096 per dimension)
                            if not (256 <= rescale_width <= 4096 and 256 <= rescale_height <= 4096):
                                logger.warning(
                                    f"Resolution {rescale_width}x{rescale_height} out of range (256-4096). "
                                    f"Using default {default_resolution}"
                                )
                                rescale_width, rescale_height = 1280, 1024
                        else:
                            logger.warning(f"Invalid resolution format '{resolution_input}'. Using default {default_resolution}")
                            rescale_width, rescale_height = 1280, 1024
                    else:
                        # Benchmark mode - use default (will be overridden by benchmark config)
                        rescale_width, rescale_height = 1280, 1024

                    # Store rescaling preference for this model
                    if rescale_width and rescale_height:
                        self._rescale_resolution[model_id_for_rescale] = (rescale_width, rescale_height)
                        logger.info(f"VLM image rescaling enabled: {rescale_width}x{rescale_height}")
                except Exception as e:
                    logger.warning(f"Error configuring image rescaling: {e}. Disabling rescaling.")
                    self._rescale_resolution[model_id_for_rescale] = None

            # Show loading box only if not in benchmark mode
            import os
            if not os.getenv("EKAM_BENCHMARKING"):
                from ..cli.tui_manager import tui as _tui
                _tui.clear_screen()
                _tui.show_message(
                    f"Loading model: {model_path.name}\n\n[dim]Please wait...[/dim]",
                    title="Loading",
                    style="cyan",
                )

            # Determine which server implementation to use
            # In interactive mode (non-VLM), ask user; otherwise use config
            server_type = get_llama_server_type()

            # For LLMs (non-VLM): always ask user to choose server type
            # VLMs automatically use subprocess mode (no choice needed)
            if not mmproj_path:
                from ..cli.text_input import professional_prompt
                from ..cli.tui_manager import tui
                tui.console.print("\n[cyan]Server Type Selection[/cyan] — choose llama-server implementation")
                tui.console.print("[dim]Two modes available for text models:[/dim]")

                choice = professional_prompt.get_arrow_selection(
                    options=[
                        ("subprocess", "Subprocess (llama-server binary)", "Uses patched llama-server binary - no Flask dependency, same as VLMs"),
                        ("python", "Python (llama-cpp-python + Flask)", "Uses Python bindings with Flask HTTP wrapper - better integration")
                    ],
                    title="Select Server Type",
                    instructions="Use ↑/↓ arrows, Enter to select",
                )

                if choice == "subprocess":
                    server_type = "llama-server"
                    logger.info("User selected: subprocess llama-server binary")
                elif choice == "python":
                    server_type = "llama-cli-server"
                    logger.info("User selected: Python bindings + Flask server")
                # If user cancels (None), use default from config

            if server_type == "llama-cli-server":
                # Use custom CLI server with isolated mtmd_context (supports both LLM and VLM)
                # Server type will be logged once at startup by the server itself
                server = LlamaCLIServerManager(
                    model_path=str(model_path),
                    host="127.0.0.1",
                    port=port,
                )
            else:
                # Use legacy llama-server (has vision model bugs)
                # Server type will be logged once at startup by the server itself
                server = LlamaServerManager(
                    model_path=str(model_path),
                    host="127.0.0.1",
                    port=port,
                )

            # Build start parameters (device only for LlamaCLIServerManager)
            start_params = {
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx,
                "n_batch": 2048,
                "n_ubatch": 512,
                "n_threads": optimal_threads,
                "n_threads_batch": optimal_threads_batch,
                "mmproj_path": str(mmproj_path.resolve()) if mmproj_path else None,
            }

            # Only pass device to LlamaCLIServerManager (legacy LlamaServerManager doesn't support it)
            if isinstance(server, LlamaCLIServerManager):
                start_params["device"] = device

            success = server.start(**start_params)

            if not success:
                raise RuntimeError(f"Failed to start {server_type} for {model_path}")

            # Store server instance
            self.llama_servers[model_key] = server

            logger.info(
                f"✓ llama-server started on port {port} "
                f"(GPU layers={n_gpu_layers}, threads={optimal_threads}, "
                f"ctx={n_ctx}, batch=2048, KV cache=GPU VRAM)"
            )
            return server

        except Exception as e:
            import traceback
            logger.error(f"Failed to start llama-server: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to load GGUF model via server: {e}")

    def _load_hf_model(self, model_path: Path, device: str) -> Any:
        """Load HuggingFace format quantized model (LLM or VLM).

        Automatically detects MLX models and uses MLX inference instead of transformers.

        Args:
            model_path: Path to model directory
            device: Target device ("cuda", "mps", or "cpu")

        Returns:
            Tuple of (model, processor/tokenizer, metadata) where metadata contains model type info
        """
        # DYNAMIC ROUTING: Read quantization metadata to determine inference method
        metadata_file = model_path / "quantization_metadata.json"
        module = None
        quant_type = None
        is_vlm = False

        if metadata_file.exists():
            try:
                import json
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    module = metadata.get("module", "").lower()
                    quant_type = metadata.get("quant_type", "unknown")
                    original_model = metadata.get("original_model", "unknown")

                    # Detect VLM from original model name or provider info
                    is_vlm = "VL" in original_model or "vision" in original_model.lower()

                    logger.info(
                        f"Quantization metadata: module={module}, "
                        f"quant_type={quant_type}, is_vlm={is_vlm}"
                    )

            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse quantization metadata: {e}")
            except Exception as e:
                logger.debug(f"Error reading quantization metadata: {e}")
        else:
            logger.warning(f"No quantization_metadata.json found for {model_path.name}")
            logger.info("Will attempt to detect inference method from config.json")

        # Route based on module (dynamic dispatch)
        if module == "mlx":
            logger.info(f"Routing to MLX inference for {quant_type} model")
            return self._load_mlx_model(model_path, is_vlm, device)
        elif module in ["bnb", "bitsandbytes"]:
            logger.info(f"Routing to HuggingFace transformers (BitsAndBytes {quant_type})")
            return self._load_transformers_model(model_path, device)
        elif module == "gptq":
            logger.info(f"Routing to HuggingFace transformers (GPTQ {quant_type})")
            return self._load_transformers_model(model_path, device)
        elif module == "awq":
            logger.info(f"Routing to HuggingFace transformers (AWQ {quant_type})")
            return self._load_transformers_model(model_path, device)
        elif module in ["generic", "transformers"]:
            logger.info(f"Routing to HuggingFace transformers (Generic {quant_type})")
            return self._load_transformers_model(model_path, device)
        else:
            # Fallback: No metadata or unknown module
            logger.info(f"Unknown/missing module ({module}), defaulting to HuggingFace transformers")
            logger.info("Transformers will auto-detect quantization from config.json")
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

                                # Try loading with initial kwargs
                                model = None
                                try:
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
                                except Exception as e:
                                    # Handle disk offloading error: "trying to offload the whole model to the disk"
                                    # This happens on CPU/Metal with limited memory when device_map="auto"
                                    error_str = str(e).lower()
                                    if "offload" in error_str or "disk" in error_str:
                                        logger.debug(f"device_map='auto' caused error: {str(e)[:100]}")
                                        logger.debug(f"Retrying with explicit device {target_device}")
                                        # Retry with explicit device placement
                                        try:
                                            load_kwargs["device_map"] = {"": target_device}

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
                                        except Exception as fallback_e:
                                            fallback_error_str = str(fallback_e).lower()
                                            # Check if this is a memory/buffer error
                                            if "buffer" in fallback_error_str or "memory" in fallback_error_str or "cuda" in fallback_error_str or "out of" in fallback_error_str:
                                                if target_device != "cpu":
                                                    logger.debug(f"Device {target_device} has insufficient memory for VLM: {str(fallback_e)[:100]}")
                                                    logger.info(f"Falling back to CPU loading for VLM (model will run slower)")
                                                    # Clean up memory before CPU fallback
                                                    self._cleanup_memory(target_device)
                                                    # Final fallback: load on CPU
                                                    try:
                                                        load_kwargs["device_map"] = {"": "cpu"}

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
                                                    except Exception as cpu_e:
                                                        logger.debug(f"CPU loading for VLM also failed: {str(cpu_e)[:100]}")
                                                        raise
                                                else:
                                                    logger.debug(f"Already on CPU, cannot fall back further: {str(fallback_e)[:100]}")
                                                    raise
                                            else:
                                                logger.debug(f"VLM fallback failed with different error: {str(fallback_e)[:100]}")
                                                raise
                                    else:
                                        raise

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
                    model = None
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            str(model_path),
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                        logger.info("✓ Loaded with AutoModelForCausalLM")
                    except Exception as e:
                        # Handle disk offloading error: "trying to offload the whole model to the disk"
                        # This happens on CPU/Metal with limited memory
                        error_str = str(e).lower()
                        if "offload" in error_str or "disk" in error_str:
                            logger.debug(f"device_map='auto' caused error (likely disk offloading): {str(e)[:100]}")
                            logger.debug(f"Falling back to explicit device placement: {device}")
                            # Retry with explicit device instead of device_map
                            try:
                                model = AutoModelForCausalLM.from_pretrained(
                                    str(model_path),
                                    device_map={"": device},  # Explicit device placement
                                    low_cpu_mem_usage=True,
                                    trust_remote_code=True
                                )
                                logger.info(f"✓ Loaded with AutoModelForCausalLM (explicit device: {device})")
                            except Exception as fallback_e:
                                fallback_error_str = str(fallback_e).lower()
                                # Check if this is a memory/buffer error
                                if "buffer" in fallback_error_str or "memory" in fallback_error_str or "cuda" in fallback_error_str or "out of" in fallback_error_str:
                                    if device != "cpu":
                                        logger.debug(f"Device {device} has insufficient memory: {str(fallback_e)[:100]}")
                                        logger.info(f"Falling back to CPU loading (model will run slower)")
                                        # Clean up memory before CPU fallback
                                        self._cleanup_memory(device)
                                        # Final fallback: load on CPU
                                        try:
                                            model = AutoModelForCausalLM.from_pretrained(
                                                str(model_path),
                                                device_map={"": "cpu"},
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=True
                                            )
                                            logger.info(f"✓ Loaded with AutoModelForCausalLM on CPU (trading performance for compatibility)")
                                        except Exception as cpu_e:
                                            logger.debug(f"CPU loading also failed: {str(cpu_e)[:100]}")
                                            raise
                                    else:
                                        logger.debug(f"Already on CPU, cannot fall back further: {str(fallback_e)[:100]}")
                                        raise
                                else:
                                    logger.debug(f"Fallback failed with different error: {str(fallback_e)[:100]}")
                                    raise
                        else:
                            raise
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
                                            try:
                                                model = AutoModelForCausalLM.from_pretrained(
                                                    str(model_path),
                                                    device_map="auto",
                                                    low_cpu_mem_usage=True,
                                                    trust_remote_code=True
                                                )
                                                logger.info("✓ Successfully loaded after copying modeling file!")
                                            except Exception as e:
                                                # Fallback if device_map="auto" tries disk offloading
                                                error_str = str(e).lower()
                                                if "offload" in error_str or "disk" in error_str:
                                                    logger.debug(f"device_map='auto' caused error: {str(e)[:100]}")
                                                    logger.debug(f"Retrying with explicit device: {device}")
                                                    try:
                                                        model = AutoModelForCausalLM.from_pretrained(
                                                            str(model_path),
                                                            device_map={"": device},
                                                            low_cpu_mem_usage=True,
                                                            trust_remote_code=True
                                                        )
                                                        logger.info("✓ Successfully loaded after copying modeling file!")
                                                    except Exception as fallback_e:
                                                        fallback_error_str = str(fallback_e).lower()
                                                        # Check if this is a memory/buffer error
                                                        if "buffer" in fallback_error_str or "memory" in fallback_error_str or "cuda" in fallback_error_str or "out of" in fallback_error_str:
                                                            if device != "cpu":
                                                                logger.debug(f"Device {device} has insufficient memory: {str(fallback_e)[:100]}")
                                                                logger.info(f"Falling back to CPU loading (model will run slower)")
                                                                # Clean up memory before CPU fallback
                                                                self._cleanup_memory(device)
                                                                # Final fallback: load on CPU
                                                                try:
                                                                    model = AutoModelForCausalLM.from_pretrained(
                                                                        str(model_path),
                                                                        device_map={"": "cpu"},
                                                                        low_cpu_mem_usage=True,
                                                                        trust_remote_code=True
                                                                    )
                                                                    logger.info("✓ Successfully loaded on CPU after copying modeling file!")
                                                                except Exception as cpu_e:
                                                                    logger.debug(f"CPU loading also failed: {str(cpu_e)[:100]}")
                                                                    raise
                                                            else:
                                                                logger.debug(f"Already on CPU, cannot fall back further: {str(fallback_e)[:100]}")
                                                                raise
                                                        else:
                                                            logger.debug(f"Fallback failed with different error: {str(fallback_e)[:100]}")
                                                            raise
                                                else:
                                                    raise
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

                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                str(model_path),
                                config=config,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                trust_remote_code=True
                            )
                            logger.info("✓ Loaded with AutoConfig approach")
                        except Exception as e:
                            # Fallback for disk offloading
                            error_str = str(e).lower()
                            if "offload" in error_str or "disk" in error_str:
                                logger.debug(f"device_map='auto' caused error (likely disk offloading): {str(e)[:100]}")
                                logger.debug(f"Retrying with explicit device: {device}")
                                try:
                                    model = AutoModelForCausalLM.from_pretrained(
                                        str(model_path),
                                        config=config,
                                        device_map={"": device},
                                        low_cpu_mem_usage=True,
                                        trust_remote_code=True
                                    )
                                    logger.info("✓ Loaded with AutoConfig approach (explicit device)")
                                except Exception as fallback_e:
                                    fallback_error_str = str(fallback_e).lower()
                                    # Check if this is a memory/buffer error
                                    if "buffer" in fallback_error_str or "memory" in fallback_error_str or "cuda" in fallback_error_str or "out of" in fallback_error_str:
                                        if device != "cpu":
                                            logger.debug(f"Device {device} has insufficient memory: {str(fallback_e)[:100]}")
                                            logger.info(f"Falling back to CPU loading (model will run slower)")
                                            # Clean up memory before CPU fallback
                                            self._cleanup_memory(device)
                                            # Final fallback: load on CPU
                                            try:
                                                model = AutoModelForCausalLM.from_pretrained(
                                                    str(model_path),
                                                    config=config,
                                                    device_map={"": "cpu"},
                                                    low_cpu_mem_usage=True,
                                                    trust_remote_code=True
                                                )
                                                logger.info("✓ Loaded with AutoConfig approach on CPU")
                                            except Exception as cpu_e:
                                                logger.debug(f"CPU loading also failed: {str(cpu_e)[:100]}")
                                                raise
                                        else:
                                            logger.debug(f"Already on CPU, cannot fall back further: {str(fallback_e)[:100]}")
                                            raise
                                    else:
                                        logger.debug(f"Fallback failed with different error: {str(fallback_e)[:100]}")
                                        raise
                            else:
                                raise
                    except Exception as e:
                        error_msg = str(e)
                        error_messages.append(f"AutoConfig: {error_msg[:100]}")
                        logger.debug(f"AutoConfig approach failed: {error_msg}")

                # Strategy 4: Last resort - try AutoModel (generic fallback)
                if model is None:
                    try:
                        logger.debug("Attempting final fallback with AutoModel...")
                        try:
                            model = AutoModel.from_pretrained(
                                str(model_path),
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                trust_remote_code=True
                            )
                            logger.info("✓ Loaded with AutoModel (generic fallback)")
                        except Exception as e:
                            # Fallback for disk offloading
                            error_str = str(e).lower()
                            if "offload" in error_str or "disk" in error_str:
                                logger.debug(f"device_map='auto' caused error (likely disk offloading): {str(e)[:100]}")
                                logger.debug(f"Retrying with explicit device: {device}")
                                try:
                                    model = AutoModel.from_pretrained(
                                        str(model_path),
                                        device_map={"": device},
                                        low_cpu_mem_usage=True,
                                        trust_remote_code=True
                                    )
                                    logger.info(f"✓ Loaded with AutoModel on {device} (explicit device)")
                                except Exception as fallback_e:
                                    fallback_error_str = str(fallback_e).lower()
                                    # Check if this is a memory/buffer error
                                    if "buffer" in fallback_error_str or "memory" in fallback_error_str or "cuda" in fallback_error_str or "out of" in fallback_error_str:
                                        if device != "cpu":
                                            logger.debug(f"Device {device} has insufficient memory: {str(fallback_e)[:100]}")
                                            logger.info(f"Falling back to CPU loading (model will run slower)")
                                            # Clean up memory before CPU fallback
                                            self._cleanup_memory(device)
                                            # Final fallback: load on CPU
                                            try:
                                                model = AutoModel.from_pretrained(
                                                    str(model_path),
                                                    device_map={"": "cpu"},
                                                    low_cpu_mem_usage=True,
                                                    trust_remote_code=True
                                                )
                                                logger.info(f"✓ Loaded with AutoModel on CPU (generic fallback)")
                                            except Exception as cpu_e:
                                                logger.debug(f"CPU loading also failed: {str(cpu_e)[:100]}")
                                                raise
                                        else:
                                            logger.debug(f"Already on CPU, cannot fall back further: {str(fallback_e)[:100]}")
                                            raise
                                    else:
                                        logger.debug(f"Fallback failed with different error: {str(fallback_e)[:100]}")
                                        raise
                            else:
                                raise
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
            model_handle: Model instance (LlamaServerManager, tuple of (model, processor), or single model)
        """
        try:
            # Stop server if this is a server instance (works for both types)
            if isinstance(model_handle, (LlamaServerManager, LlamaCLIServerManager)):
                model_key = str(model_handle.model_path)
                if model_key in self.llama_servers:
                    model_handle.stop()
                    del self.llama_servers[model_key]
                    # Clear any session mmproj selection for this model
                    if model_key in self._session_mmproj_choice:
                        try:
                            del self._session_mmproj_choice[model_key]
                        except Exception:
                            pass
                    server_type = "custom CLI server" if isinstance(model_handle, LlamaCLIServerManager) else "llama-server"
                    logger.info(f"{server_type} stopped for {model_handle.model_path.name}")
                return

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
        conversation_history: Optional[list[tuple[str, str]]] = None,
        custom_parameters: Optional[dict] = None,
        timeout: Optional[float] = None,
        stream_callback: Optional[callable] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Run question answering on quantized VLM.

        Args:
            model_handle: Loaded model (tuple of (model, processor, metadata) for HF/MLX VLMs)
            image: PIL Image
            question: Question text
            conversation_history: Optional conversation history
            custom_parameters: Custom inference parameters
            timeout: Optional timeout in seconds (not used, for API compatibility)
            stream_callback: Optional callback for token streaming (delta: str, is_first: bool)
            session_id: Optional session ID for KV cache (None = stateless)

        Returns:
            Answer text
        """
        # Extract streaming callback from both direct parameter and custom_parameters (for backward compatibility)
        if stream_callback is None and custom_parameters:
            stream_callback = custom_parameters.get("_stream_callback")

        # Check if we're in benchmark mode - disable streaming if so
        import os
        is_benchmark = os.getenv("EKAM_BENCHMARK_MODE") == "1"
        effective_callback = None if is_benchmark else stream_callback

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
                return self._run_qa_mlx(model, processor_or_tokenizer, image, question, conversation_history, metadata, effective_callback)
            else:
                return self._run_qa_transformers(model, processor_or_tokenizer, image, question, conversation_history, effective_callback)

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

        # GGUF models (LlamaServerManager) - use llama-server API with mmproj for vision
        # model_handle is the LlamaServerManager instance directly (not a tuple)
        return self._run_qa_gguf_server(model_handle, None, image, question, conversation_history, custom_parameters, session_id)

    def _run_qa_gguf_server(
        self,
        model: Any,
        processor: Any,
        image: Any,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        custom_parameters: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Run QA inference using GGUF model via llama-server with --mmproj.

        EXPERIMENTAL: Supports GGUF VLMs (Qwen3-VL, LLaVA, etc.) with mmproj files.

        Args:
            model: LlamaServerManager instance
            processor: Not used (llama-server handles tokenization)
            image: PIL Image
            question: Question text
            conversation_history: Optional conversation history

        Returns:
            Answer text
        """
        from ..services.llama_server_manager import LlamaServerManager
        from ..services.llama_cli_server_manager import LlamaCLIServerManager
        from ..benchmarking.utils.llama_delay import apply_llama_server_delay
        import base64
        from io import BytesIO
        import time

        logger.info("Running GGUF VLM inference via llama-server...")

        # WORKAROUND: Add throttling between vision requests to reduce race conditions
        # NOTE: This is only needed for legacy llama-server. Custom CLI server has isolated contexts.
        if not hasattr(self, '_last_vision_request_time'):
            self._last_vision_request_time = {}

        model_key = str(model.model_path) if hasattr(model, 'model_path') else 'default'
        last_request_time = self._last_vision_request_time.get(model_key, 0)
        time_since_last = time.time() - last_request_time

        # Only throttle for legacy llama-server (custom server doesn't need this)
        if isinstance(model, LlamaServerManager):
            min_interval = 0.05  # 50ms minimum between requests for legacy server
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                logger.debug(f"Throttling (legacy server): sleeping {sleep_time:.3f}s to avoid race condition")
                time.sleep(sleep_time)

        self._last_vision_request_time[model_key] = time.time()

        # Verify model is a server instance (either type)
        if not isinstance(model, (LlamaServerManager, LlamaCLIServerManager)):
            raise TypeError(f"Expected LlamaServerManager or LlamaCLIServerManager, got {type(model).__name__}")

        # Apply VLM image rescaling if configured for this model
        model_path = str(model.model_path) if hasattr(model, 'model_path') else None
        if model_path and model_path in self._rescale_resolution:
            rescale_config = self._rescale_resolution[model_path]
            if rescale_config:  # Not None (None = disabled)
                target_width, target_height = rescale_config
                original_size = image.size
                # Only rescale if image is different size
                if image.size != (target_width, target_height):
                    from ..utils.image import rescale_image_with_padding
                    image = rescale_image_with_padding(image, target_width, target_height)
                    logger.info(
                        f"Rescaled image from {original_size[0]}x{original_size[1]} to "
                        f"{target_width}x{target_height} (with aspect ratio preservation)"
                    )

        # Prepare image in multiple formats for llama-server API
        # 1. Save to temporary file for file:// URL (native llama.cpp format)
        import tempfile
        import os
        temp_image_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name, format="JPEG")
                temp_image_path = tmp.name
        except Exception as e:
            logger.warning(f"Could not save temp image file: {e}")

        # 2. Convert to base64 data URL for OpenAI-style format
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"

        # Debug logging
        logger.debug(f"Image encoding: {len(img_bytes)} bytes → {len(img_base64)} base64 chars")
        logger.debug(f"Image size: {image.size}, mode: {image.mode}")
        if temp_image_path:
            logger.debug(f"Temp image saved to: {temp_image_path}")

        # Build messages array with vision content
        # llama-server with --mmproj supports OpenAI-style vision messages
        messages = []

        # System prompt (from /config)
        sys_prompt = None
        if custom_parameters and isinstance(custom_parameters.get("system_prompt"), str):
            sys_prompt = custom_parameters.get("system_prompt")
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
            logger.debug(f"System prompt (raw): {sys_prompt}")

        # Add conversation history (text-only, no images in history)
        if conversation_history:
            for user_msg, bot_msg in conversation_history[-3:]:  # Last 3 turns
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": bot_msg})

        # Add current question with image
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": img_data_url}}
            ]
        })

        logger.debug(f"Sending vision request to llama-server with {len(messages)} messages")

        # Gather advanced parameters from custom_parameters
        extra = {}
        if custom_parameters:
            for key in [
                "top_k","min_p","typical_p","tfs_z","repeat_penalty",
                "presence_penalty","frequency_penalty","penalty_last_n",
                "mirostat","mirostat_tau","mirostat_eta","seed","n_keep",
                "ignore_eos","grammar","logit_bias","n_probs",
            ]:
                if key in custom_parameters:
                    extra[key] = custom_parameters[key]

        # Check if we've cached which format works for this model
        model_key = str(model.model_path) if hasattr(model, 'model_path') else None
        cached_format = self._vision_api_format_cache.get(model_key) if model_key else None

        # IMPORTANT: Always include image in message content to ensure llama-server processes it
        # The top-level `images` parameter is unreliable and sometimes ignored

        # Extract just base64 data for proprietary format (remove data URL prefix)
        img_base64_only = img_base64  # Already have this from line 2039

        # ============================================================================
        # 2025 llama.cpp STANDARD FORMAT: OpenAI-compatible image_url content parts
        # ============================================================================
        # Reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/utils.hpp:616-660
        # The llama.cpp server supports OpenAI-compatible format with image_url in content array:
        # - data:image/*;base64,<base64> (standard, portable, proven to work)
        # - http:// or https:// URLs (downloads remote images automatically)
        # - file:// URLs (experimental, may not be supported in all versions)
        #
        # Priority order (as requested by user):
        # 1. PRIMARY: OpenAI base64 data URL format (most compatible with llama.cpp web UI)
        # 2. FALLBACK: Proprietary [img-1] format (for compatibility with older code)
        # ============================================================================

        # Format 1: OpenAI-compatible format with base64 data URL (PRIMARY)
        # This is what the llama.cpp web UI uses internally
        openai_messages = []
        if sys_prompt:
            openai_messages.append({"role": "system", "content": sys_prompt})
        if conversation_history:
            for user_msg, bot_msg in conversation_history[-3:]:
                openai_messages.append({"role": "user", "content": user_msg})
                openai_messages.append({"role": "assistant", "content": bot_msg})
        openai_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": img_data_url}}
            ]
        })

        # Format 2: Proprietary llama.cpp format with [img-1] placeholder (FALLBACK for compatibility)
        # This is a custom format not part of standard llama.cpp
        # Keeping as fallback for any edge cases where OpenAI format might not work
        proprietary_messages = []
        if sys_prompt:
            proprietary_messages.append({"role": "system", "content": sys_prompt})
        if conversation_history:
            for user_msg, bot_msg in conversation_history[-3:]:
                proprietary_messages.append({"role": "user", "content": user_msg})
                proprietary_messages.append({"role": "assistant", "content": bot_msg})
        proprietary_messages.append({
            "role": "user",
            "content": f"[img-1]\n{question}"
        })
        proprietary_image_data = [{"id": 1, "data": img_base64}]

        # ============================================================================
        # CACHE MIGRATION: Clear old "proprietary" cache to force OpenAI format retry
        # ============================================================================
        # Since we've changed the priority order to prefer OpenAI format first,
        # we need to clear any cached "proprietary" formats and force re-detection.
        # This ensures the new OpenAI-first priority takes effect.
        # ============================================================================
        if cached_format == "proprietary":
            logger.info(
                "⚠️  Migrating from cached 'proprietary' format → forcing re-detection with OpenAI-first priority"
            )
            if model_key and model_key in self._vision_api_format_cache:
                del self._vision_api_format_cache[model_key]
            cached_format = None  # Force format detection below

        # Try cached format first (only for openai_base64, which is our PRIMARY format)
        if cached_format == "proprietary_old_disabled":
            # This block is disabled - proprietary format is now FALLBACK only
            # Use cached proprietary format (DEPRECATED - now using OpenAI first)
            try:
                # Build and log exact HTTP request
                try:
                    url, payload = model.build_chat_payload(
                        messages=proprietary_messages,
                        image_data=proprietary_image_data,
                        max_tokens=1024,
                        temperature=0.7,
                        top_p=0.9,
                        stream=False,
                        **extra,
                    )
                    # Log request (proprietary format)
                    img_src_path = getattr(image, "_source_path", None)
                    logger.debug(f"Request: POST {url} | format=proprietary | image={Path(img_src_path).name if img_src_path else 'unknown'}")
                except Exception:
                    pass

                response = model.chat_completion(
                    messages=proprietary_messages,
                    image_data=proprietary_image_data,
                    max_tokens=1024,
                    temperature=0.7,
                    stream=False,
                    session_id=session_id,
                    **extra,
                )

                if isinstance(response, dict) and "choices" in response:
                    answer = response["choices"][0]["message"]["content"]
                    error_keywords = [
                        "not a valid image", "invalid image", "unable to process",
                        "cannot", "no image", "do not see an image"
                    ]
                    if any(k in answer.lower() for k in error_keywords):
                        logger.warning("Cached proprietary format produced suspected error response; clearing cache and retrying detection")
                        if model_key and model_key in self._vision_api_format_cache:
                            del self._vision_api_format_cache[model_key]
                    else:
                        logger.info("✓ GGUF VLM inference successful (cached proprietary format)")
                        # Clean up temp file before returning
                        if temp_image_path:
                            try:
                                os.unlink(temp_image_path)
                            except Exception:
                                pass
                        return answer.strip()
                else:
                    # Truncate response to prevent logging huge base64 payloads
                    response_str = str(response)
                    truncated_response = response_str[:200] + "..." if len(response_str) > 200 else response_str
                    logger.error(f"Unexpected response format: {truncated_response}")
                    raise RuntimeError("Failed to get response from llama-server")
            except Exception as e:
                logger.warning(f"Cached format failed: {e}. Re-detecting format...")
                # Clear cache and fall through to format detection
                if model_key and model_key in self._vision_api_format_cache:
                    del self._vision_api_format_cache[model_key]

        # Handle renamed cached formats for backward compatibility
        if cached_format == "openai":
            cached_format = "openai_base64"
            logger.info("Migrating cached format 'openai' → 'openai_base64'")

        if cached_format == "openai_base64":
            # Use cached OpenAI base64 format (PRIMARY format - standard llama.cpp)
            # WORKAROUND: Add retry logic for intermittent mtmd_encode_chunk() failures
            max_retries = 3
            retry_count = 0
            last_error = None

            while retry_count < max_retries:
                try:
                    # Build and log exact HTTP request
                    try:
                        url, payload = model.build_chat_payload(
                            messages=openai_messages,
                            max_tokens=1024,
                            temperature=0.7,
                            top_p=0.9,
                            stream=False,
                            **extra,
                        )
                        # Log request (OpenAI format)
                        img_src_path = getattr(image, "_source_path", None)
                        logger.debug(f"Request: POST {url} | format=openai_base64 | image={Path(img_src_path).name if img_src_path else 'unknown'} | attempt={retry_count+1}/{max_retries}")
                    except Exception:
                        pass

                    response = model.chat_completion(
                        messages=openai_messages,
                        max_tokens=1024,
                        temperature=0.7,
                        stream=False,
                        session_id=session_id,
                        **extra,
                    )
                    # Success - record it and break out of retry loop
                    if hasattr(model, 'record_inference_success'):
                        model.record_inference_success()
                    break

                except Exception as e:
                    last_error = e
                    retry_count += 1

                    # Check if this is the intermittent image encoding error (HTTP 500)
                    is_encoding_error = False
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = getattr(e.response, 'status_code', 0)
                        if status_code == 500:
                            try:
                                error_body = getattr(e.response, 'text', '')
                                if 'failed to process image' in error_body.lower():
                                    is_encoding_error = True
                            except:
                                pass

                    if is_encoding_error and retry_count < max_retries:
                        import time
                        backoff_delay = 0.1 * (2 ** (retry_count - 1))  # Exponential backoff: 0.1s, 0.2s, 0.4s
                        logger.warning(
                            f"⚠️  mtmd_encode_chunk() failure detected (HTTP 500) - "
                            f"attempt {retry_count}/{max_retries}. "
                            f"Retrying after {backoff_delay:.2f}s delay..."
                        )
                        time.sleep(backoff_delay)

                        # Health check before retry
                        try:
                            if not model.is_running():
                                logger.error("llama-server died, cannot retry")
                                raise last_error
                        except:
                            pass
                        continue
                    else:
                        # Not an encoding error, or max retries reached
                        raise

            # If we exhausted retries, record failure and potentially restart server
            if retry_count >= max_retries:
                logger.error(f"Failed after {max_retries} retry attempts")

                # Track consecutive failures and check if restart is needed
                if hasattr(model, 'record_inference_failure'):
                    should_restart = model.record_inference_failure()

                    if should_restart and hasattr(model, 'restart'):
                        logger.warning("Attempting automatic server restart...")
                        restart_success = model.restart()

                        if restart_success:
                            logger.info("Server restarted successfully. Retrying inference once...")
                            # Try one more time with fresh server
                            try:
                                response = model.chat_completion(
                                    messages=openai_messages,
                                    max_tokens=1024,
                                    temperature=0.7,
                                    stream=False,
                                    session_id=session_id,
                                    **extra,
                                )
                                # Success after restart
                                if hasattr(model, 'record_inference_success'):
                                    model.record_inference_success()
                                logger.info("✓ Inference successful after server restart")
                                # Continue to response processing below
                            except Exception as retry_error:
                                logger.error(f"Inference failed even after restart: {retry_error}")
                                raise last_error
                        else:
                            logger.error("Server restart failed")
                            raise last_error
                    else:
                        raise last_error
                else:
                    raise last_error

            # Response validation
            try:
                if isinstance(response, dict) and "choices" in response:
                    answer = response["choices"][0]["message"]["content"]
                    error_keywords = [
                        "not a valid image", "invalid image", "unable to process",
                        "cannot", "no image", "do not see an image"
                    ]
                    if any(k in answer.lower() for k in error_keywords):
                        logger.warning("Cached OpenAI base64 format produced suspected error response; clearing cache and retrying detection")
                        if model_key and model_key in self._vision_api_format_cache:
                            del self._vision_api_format_cache[model_key]
                    else:
                        logger.info("✓ GGUF VLM inference successful (cached openai_base64 format)")
                        # Clean up temp file before returning
                        if temp_image_path:
                            try:
                                os.unlink(temp_image_path)
                            except Exception:
                                pass
                        # Force Metal backend synchronization to free fragmented memory buffers
                        try:
                            import requests
                            health_url = f"{model.base_url}/health"
                            requests.get(health_url, timeout=0.5)
                            logger.debug("Metal backend sync triggered via /health endpoint")
                        except Exception as e:
                            logger.debug(f"Metal sync call failed (non-critical): {e}")
                        # Apply post-inference delay for llama-server state cleanup
                        apply_llama_server_delay("quantized")
                        return answer.strip()
                else:
                    # Truncate response to prevent logging huge base64 payloads
                    response_str = str(response)
                    truncated_response = response_str[:200] + "..." if len(response_str) > 200 else response_str
                    logger.error(f"Unexpected response format: {truncated_response}")
                    raise RuntimeError("Failed to get response from llama-server")
            except Exception as e:
                logger.warning(f"Cached format failed: {e}. Re-detecting format...")
                # Clear cache and fall through to format detection
                if model_key and model_key in self._vision_api_format_cache:
                    del self._vision_api_format_cache[model_key]

        # ============================================================================
        # BENCHMARK MODE: Strict format locking
        # ============================================================================
        # Check if we're in benchmark mode (set by benchmark_runner.py)
        import os
        is_benchmark_mode = os.getenv("EKAM_BENCHMARK_MODE") == "1"

        # In benchmark mode with cached format: skip format detection, use cached only
        # This ensures consistent methodology across all counted runs (no fallback attempts)
        if is_benchmark_mode and cached_format:
            logger.info(f"🔒 Benchmark strict mode: locked to '{cached_format}' format (no fallbacks)")
            # Skip format detection loop - will only run cached format block above
            # If cached format already succeeded above, we returned
            # If we're here, cached format failed - let it raise exception for retry logic
            raise RuntimeError(
                f"Benchmark strict mode: cached format '{cached_format}' failed. "
                "This run will be retried once, then skipped if fails again."
            )

        # Format detection: Try formats in priority order (as requested by user)
        # Priority: OpenAI base64 (PRIMARY - standard llama.cpp web UI format) → Proprietary [img-1] (FALLBACK)
        # Note: This loop is SKIPPED in benchmark mode when we have a cached format
        formats_to_try = []
        formats_to_try.append(("openai_base64", openai_messages, None))  # PRIMARY: Standard OpenAI format with base64
        formats_to_try.append(("proprietary", proprietary_messages, proprietary_image_data))  # FALLBACK: Custom [img-1] format

        last_error = None
        for format_tuple in formats_to_try:
            format_name = format_tuple[0]
            test_messages = format_tuple[1]
            test_image_data = format_tuple[2] if len(format_tuple) > 2 else None

            try:
                logger.info(f"Trying vision format: {format_name}")
                logger.debug(f"Request messages: {test_messages}")
                if test_image_data:
                    logger.debug(f"Image data parameter: {len(test_image_data)} items")

                # Build kwargs for chat_completion
                chat_kwargs = {
                    "messages": test_messages,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "stream": False,
                    "session_id": session_id,
                }
                chat_kwargs.update(extra)
                if test_image_data:
                    chat_kwargs["image_data"] = test_image_data

                # WORKAROUND: Add retry logic for format detection (same as cached format)
                max_retries = 2  # Fewer retries for format detection
                retry_count = 0
                format_last_error = None

                while retry_count < max_retries:
                    try:
                        # Log exact request built for llama.cpp
                        try:
                            url, payload = model.build_chat_payload(**chat_kwargs)
                            # Log request (format detection)
                            img_src_path = getattr(image, "_source_path", None)
                            logger.debug(f"Request: POST {url} | format={format_name} | image={Path(img_src_path).name if img_src_path else 'unknown'} | attempt={retry_count+1}/{max_retries}")
                        except Exception:
                            pass

                        response = model.chat_completion(**chat_kwargs)
                        # Success - record it and break out of retry loop
                        if hasattr(model, 'record_inference_success'):
                            model.record_inference_success()
                        break

                    except Exception as retry_e:
                        format_last_error = retry_e
                        retry_count += 1

                        # Check for encoding error
                        is_encoding_error = False
                        if hasattr(retry_e, 'response') and retry_e.response is not None:
                            status_code = getattr(retry_e.response, 'status_code', 0)
                            if status_code == 500:
                                try:
                                    error_body = getattr(retry_e.response, 'text', '')
                                    if 'failed to process image' in error_body.lower():
                                        is_encoding_error = True
                                except:
                                    pass

                        if is_encoding_error and retry_count < max_retries:
                            backoff_delay = 0.1 * (2 ** (retry_count - 1))
                            logger.debug(f"Format detection: retry after {backoff_delay:.2f}s")
                            time.sleep(backoff_delay)
                            continue
                        else:
                            # Not encoding error or max retries - raise to try next format
                            raise

                # Check if we exhausted retries
                if retry_count >= max_retries:
                    raise format_last_error

                if isinstance(response, dict) and "choices" in response:
                    answer = response["choices"][0]["message"]["content"]
                    # Validate content before caching to avoid poisoning cache
                    error_keywords = [
                        "not a valid image", "invalid image", "unable to process",
                        "cannot", "no image", "do not see an image"
                    ]
                    if any(k in answer.lower() for k in error_keywords):
                        logger.warning(f"Format {format_name} produced suspected error response; trying next format")
                        last_error = RuntimeError(f"{format_name} returned suspected error response")
                        continue
                    # Cache this format as working
                    if model_key:
                        self._vision_api_format_cache[model_key] = format_name
                    logger.info(f"✓ GGUF VLM inference successful ({format_name} format)")
                    logger.debug(f"Cached format '{format_name}' for future requests")
                    # Clean up temp file before returning
                    if temp_image_path:
                        try:
                            os.unlink(temp_image_path)
                        except Exception:
                            pass
                    # Force Metal backend synchronization to free fragmented memory buffers
                    try:
                        import requests
                        health_url = f"{model.base_url}/health"
                        requests.get(health_url, timeout=0.5)
                        logger.debug("Metal backend sync triggered via /health endpoint")
                    except Exception as e:
                        logger.debug(f"Metal sync call failed (non-critical): {e}")
                    # Apply post-inference delay for llama-server state cleanup
                    apply_llama_server_delay("quantized")
                    return answer.strip()
                else:
                    logger.warning(f"Format {format_name}: Unexpected response format")
                    last_error = RuntimeError(f"Unexpected response format from {format_name}")
                    continue

            except Exception as e:
                logger.error(f"Format {format_name} failed with error: {type(e).__name__}: {e}")
                # Try to extract more details from HTTP errors
                error_details = None
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        status_code = getattr(e.response, 'status_code', 'unknown')
                        error_body = getattr(e.response, 'text', None) or str(getattr(e.response, 'content', ''))
                        # Truncate error body to prevent huge base64 dumps in logs
                        truncated_body = error_body[:500] + "..." if len(error_body) > 500 else error_body
                        logger.error(f"HTTP {status_code} Error Response Body: {truncated_body}")
                        error_details = error_body
                    except Exception as parse_err:
                        logger.debug(f"Could not parse error response: {parse_err}")
                # Try to get llama-server's output for debugging
                try:
                    if hasattr(model, 'get_recent_output'):
                        server_output = model.get_recent_output(max_lines=20)
                        if server_output:
                            logger.error(f"llama-server output:\n{server_output}")
                except:
                    pass
                last_error = (e, error_details)
                continue

        # All formats failed - track failure and potentially restart server
        # Clean up temp file before raising
        if temp_image_path:
            try:
                os.unlink(temp_image_path)
                logger.debug(f"Cleaned up temp image: {temp_image_path}")
            except Exception as cleanup_err:
                logger.debug(f"Could not clean up temp image: {cleanup_err}")

        # Track consecutive failures and check if restart is needed
        if hasattr(model, 'record_inference_failure'):
            should_restart = model.record_inference_failure()

            if should_restart and hasattr(model, 'restart'):
                logger.warning("All formats failed. Attempting automatic server restart...")
                restart_success = model.restart()

                if restart_success:
                    logger.info("Server restarted. Retrying with OpenAI base64 format...")
                    # Try one more time with fresh server (use primary format only)
                    try:
                        response = model.chat_completion(
                            messages=openai_messages,
                            max_tokens=1024,
                            temperature=0.7,
                            stream=False,
                            session_id=session_id,
                            **extra,
                        )
                        # Success after restart
                        if hasattr(model, 'record_inference_success'):
                            model.record_inference_success()

                        if isinstance(response, dict) and "choices" in response:
                            answer = response["choices"][0]["message"]["content"]
                            logger.info("✓ GGUF VLM inference successful after server restart")
                            # Apply post-inference delay
                            apply_llama_server_delay("quantized")
                            return answer.strip()
                    except Exception as retry_error:
                        logger.error(f"Inference failed even after restart: {retry_error}")

        if last_error:
            logger.error(f"All vision formats failed. Last error: {last_error}")
            logger.warning(
                "Your llama-server might not support vision inference with --mmproj yet.\n"
                "This is an experimental feature. Try:\n"
                "1. Update llama.cpp to the latest version\n"
                "2. Rebuild llama-server with vision support\n"
                "3. Or use HuggingFace format VLMs instead"
            )
            raise last_error
        else:
            raise RuntimeError("All vision formats failed with no specific error")

    def _run_qa_mlx(
        self,
        model: Any,
        processor: Any,
        image: Any,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        metadata: Optional[dict] = None,
        stream_callback: Optional[callable] = None
    ) -> str:
        """Run QA inference using MLX framework.

        Args:
            model: MLX model
            processor: MLX processor
            image: PIL Image
            question: Question text
            conversation_history: Optional conversation history
            metadata: Optional metadata dict with model_path
            stream_callback: Optional callback for token streaming (delta: str, is_first: bool)

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

            # Generate response using MLX VLM with optional streaming
            # CORRECT Signature: generate(model, processor, prompt, image, **kwargs)
            # Note: 'image' can be a single path or list of paths
            # MEMORY FIX: Reduced max_tokens from 1024 to 256 to prevent OOM crashes on 8GB systems
            # MLX VLM accepts temperature and top_p as kwargs (passed to generate_step)
            # verbose=True enables token-by-token streaming
            if stream_callback:
                # Enable streaming mode
                response_text = ""
                is_first_token = True
                for chunk in generate(
                    model,
                    processor,
                    prompt,
                    image_path,
                    max_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    verbose=True  # Returns generator for streaming
                ):
                    # Extract text from chunk
                    if hasattr(chunk, 'text'):
                        token_text = chunk.text
                    elif isinstance(chunk, str):
                        token_text = chunk
                    else:
                        token_text = str(chunk)

                    response_text += token_text
                    stream_callback(token_text, is_first=is_first_token)
                    is_first_token = False
            else:
                # Non-streaming mode
                response = generate(
                    model,
                    processor,
                    prompt,
                    image_path,
                    max_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    verbose=False
                )

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

            # Clean up temp file if we created one
            if hasattr(image, 'save'):
                try:
                    Path(image_path).unlink()
                except:
                    pass

            logger.info("✓ MLX response generated")
            
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
        conversation_history: Optional[list[tuple[str, str]]] = None,
        stream_callback: Optional[callable] = None
    ) -> str:
        """Run QA inference using HuggingFace transformers.

        Args:
            model: Transformers model
            processor: Transformers processor
            image: PIL Image
            question: Question text
            conversation_history: Optional conversation history
            stream_callback: Optional callback for token streaming (delta: str, is_first: bool)

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

        # Generate with optional streaming
        logger.info("Generating response...")
        max_tokens = 512 if str(model_device) == 'cpu' else 1024

        if stream_callback:
            # Use streaming with TextIteratorStreamer (llama.cpp-style token-by-token)
            from transformers import TextIteratorStreamer
            from threading import Thread

            # llama.cpp-style streaming configuration
            streamer = TextIteratorStreamer(
                processor.tokenizer,
                skip_prompt=True,  # Don't yield input prompt tokens
                skip_special_tokens=True,
                timeout=None  # Wait for each token immediately (no buffering)
            )

            # Run generation in background thread (required for streaming)
            generation_kwargs = {**inputs, "max_new_tokens": max_tokens, "streamer": streamer}
            generation_thread = Thread(
                target=model.generate,
                kwargs=generation_kwargs,
                daemon=True
            )
            generation_thread.start()

            # Stream tokens and invoke callback (immediate token-by-token like llama.cpp)
            response = ""
            is_first_token = True
            try:
                for new_text in streamer:
                    if new_text:  # Only send non-empty tokens
                        response += new_text
                        stream_callback(new_text, is_first=is_first_token)
                        is_first_token = False
            except Exception as stream_error:
                logger.error(f"VLM streaming error: {stream_error}", exc_info=True)
                # Continue with partial response

            # Wait for generation to complete
            generation_thread.join(timeout=600)  # 10 minute timeout for VLMs
            if generation_thread.is_alive():
                logger.warning("VLM generation thread still alive after timeout")
        else:
            # Non-streaming generation
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
        repetition_penalty: float,
        stream_callback: Optional[callable] = None
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
            stream_callback: Optional callback for token streaming (delta: str, is_first: bool)

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

        # MLX generate function with optional streaming
        # MLX's generate returns a generator when verbose=True, enabling token-by-token streaming
        if stream_callback:
            # Enable streaming by setting verbose=True to get token-by-token generation
            response = ""
            is_first_token = True
            for token_text in generate(
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=True  # Returns generator for streaming
            ):
                response += token_text
                stream_callback(token_text, is_first=is_first_token)
                is_first_token = False
        else:
            # Non-streaming generation
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
        repetition_penalty: float,
        stream_callback: Optional[callable] = None
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
            stream_callback: Optional callback for streaming tokens (for timing metrics)

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
            # Use streaming if callback is provided (for timing metrics)
            if stream_callback:
                from transformers import TextIteratorStreamer
                from threading import Thread

                # llama.cpp-style streaming: token-by-token with immediate display
                # Key settings:
                # - skip_prompt=True: Don't yield the input prompt tokens
                # - skip_special_tokens=True: Clean output
                # - timeout=None: Wait indefinitely for each token (no buffering)
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,  # Critical: Don't stream the input prompt
                    skip_special_tokens=True,
                    timeout=None  # Wait for each token immediately
                )
                gen_kwargs["streamer"] = streamer

                # Run generation in background thread (required for streaming)
                generation_kwargs = {**inputs, **gen_kwargs}
                generation_thread = Thread(
                    target=model.generate,
                    kwargs=generation_kwargs,
                    daemon=True  # Don't block program exit
                )
                generation_thread.start()

                # Stream tokens and invoke callback (llama.cpp-style: immediate token-by-token)
                response = ""
                is_first_token = True
                try:
                    for new_text in streamer:
                        if new_text:  # Only send non-empty tokens
                            response += new_text
                            stream_callback(new_text, is_first=is_first_token)
                            is_first_token = False
                except Exception as stream_error:
                    logger.error(f"Streaming error: {stream_error}", exc_info=True)
                    # Continue with partial response

                # Wait for generation to complete
                generation_thread.join(timeout=300)  # 5 minute timeout
                if generation_thread.is_alive():
                    logger.warning("Generation thread still alive after timeout")
                    # Thread will be terminated when daemon thread exits
            else:
                # Non-streaming generation
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)

                logger.debug(f"Generation complete. Output shape: {outputs.shape if outputs is not None else 'None'}")

                if outputs is None:
                    raise RuntimeError("model.generate() returned None")

                # Decode only new tokens
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]

                if generated_tokens is None or len(generated_tokens) == 0:
                    logger.warning("No tokens were generated")
                    return "I don't have a response."

                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        except Exception as gen_error:
            logger.error(f"Generation failed: {gen_error}", exc_info=True)
            raise RuntimeError(f"Model generation failed: {gen_error}")

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
        detail_level: str = "detailed",
        stream_callback: Optional[callable] = None,
    ) -> str:
        """Generate image caption with quantized VLM.

        Args:
            model_handle: Loaded model
            image: PIL Image
            conversation_history: Optional conversation history
            detail_level: "detailed" or "short"
            stream_callback: Optional callback for token streaming (delta: str, is_first: bool)

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
        return self.run_qa(model_handle, image, prompt, conversation_history, stream_callback=stream_callback)

    def run_detect(
        self,
        model_handle: Any,
        image: Any,
        object_name: str,
        timeout: Optional[float] = None
    ) -> list[dict]:
        """Detect objects with quantized VLM.

        Args:
            model_handle: Loaded model
            image: PIL Image
            object_name: Object to detect
            timeout: Optional timeout in seconds (not used, for API compatibility)

        Returns:
            List of detections

        Raises:
            NotImplementedError: If model is not a VLM
        """
        from ..utils.vlm_response_parser import parse_detection_response
        
        prompt = (
            f"Detect all instances of '{object_name}' in this image.\n\n"
            f"Return ONLY valid JSON in this exact structure:\n"
            f"{{\n"
            f"  \"{object_name}_1\": [x1, y1, x2, y2],\n"
            f"  \"{object_name}_2\": [x1, y1, x2, y2]\n"
            f"}}\n\n"
            f"Rules:\n"
            f"- Coordinates must be normalized (0.0 to 1.0)\n"
            f"- 0.0 is left/top edge, 1.0 is right/bottom edge\n"
            f"- Do not include any text before or after the JSON\n"
            f"- Number each instance sequentially (_1, _2, _3, etc.)"
        )
        response = self.run_qa(model_handle, image, prompt)
        
        # Parse response
        detections = parse_detection_response(response, object_name)
        for detection in detections:
            detection["raw"] = response
        
        return detections

    def run_point(
        self,
        model_handle: Any,
        image: Any,
        object_name: str,
        timeout: Optional[float] = None
    ) -> dict:
        """Point to object location with quantized VLM.

        Args:
            model_handle: Loaded model
            image: PIL Image
            object_name: Object to locate
            timeout: Optional timeout in seconds (not used, for API compatibility)

        Returns:
            Coordinates dict

        Raises:
            NotImplementedError: If model is not a VLM
        """
        from ..utils.vlm_response_parser import parse_point_response
        
        prompt = (
            f"Locate the '{object_name}' in this image.\n\n"
            f"Return ONLY valid JSON in this exact structure:\n"
            f"{{\n"
            f"  \"{object_name}\": [x, y]\n"
            f"}}\n\n"
            f"Rules:\n"
            f"- Coordinates must be normalized (0.0 to 1.0)\n"
            f"- 0.0 is left/top edge, 1.0 is right/bottom edge\n"
            f"- Do not include any text before or after the JSON"
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
        custom_parameters: Optional[dict] = None,
        stream_callback: Optional[callable] = None,
    ) -> str:
        """Run text generation on quantized model with conversation history support.

        Args:
            model_handle: Loaded model
            prompt: Input text
            conversation_history: Optional list of (user_msg, bot_response) tuples
            custom_parameters: Optional custom generation parameters (temperature, top_p, max_tokens, etc.)
            stream_callback: Optional callback for token streaming (delta: str, is_first: bool)

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

        # Extract streaming callback from both direct parameter and custom_parameters (for backward compatibility)
        if stream_callback is None and custom_parameters:
            stream_callback = custom_parameters.get("_stream_callback")

        # Check if we're in benchmark mode - disable streaming if so
        import os
        is_benchmark = os.getenv("EKAM_BENCHMARK_MODE") == "1"
        effective_callback = None if is_benchmark else stream_callback

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
                    max_tokens, temperature, top_p, repetition_penalty,
                    effective_callback
                )
            else:
                return self._run_text_transformers(
                    model, tokenizer, prompt, conversation_history,
                    max_tokens, temperature, top_p, repetition_penalty,
                    effective_callback
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
        
        elif isinstance(model_handle, (LlamaServerManager, LlamaCLIServerManager)):
            # llama-server or custom CLI server (GGUF via HTTP API for GPU acceleration + KV cache reuse)
            # Extract model name from server's model path
            model_path_str = str(model_handle.model_path).lower()

            # Extract model name from path for template detection
            from pathlib import Path
            filename = Path(model_path_str).stem
            # Remove quantization suffixes
            quant_patterns = [
                "_q2_k", "_q3_k_s", "_q3_k_m", "_q3_k_l",
                "_q4_0", "_q4_1", "_q4_k_s", "_q4_k_m",
                "_q5_0", "_q5_1", "_q5_k_s", "_q5_k_m",
                "_q6_k", "_q8_0", "_f16", "_f32"
            ]
            model_name = filename.lower()
            for pattern in quant_patterns:
                model_name = model_name.replace(pattern, "")
            logger.debug(f"Extracted model name for template detection: {model_name}")

            # Convert conversation history to messages format
            messages = []

            # System message
            sys_prompt = None
            if custom_parameters and isinstance(custom_parameters.get("system_prompt"), str):
                sys_prompt = custom_parameters.get("system_prompt")
            messages.append({
                "role": "system",
                "content": sys_prompt or "You are a helpful assistant."
            })

            # Add conversation history
            if conversation_history:
                recent_history = conversation_history[-5:]  # Last 5 turns
                for user_msg, ai_response in recent_history:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": ai_response})

            # Add current user message
            messages.append({"role": "user", "content": prompt})

            # Use template-specific stop tokens
            from ..utils.history_formatter import get_stop_tokens
            stop_tokens = get_stop_tokens(model_name=model_name)
            logger.debug(f"Using template-specific stop tokens for GGUF server: {stop_tokens}")

            # Allow custom stop tokens from parameters
            if custom_parameters and "stop" in custom_parameters:
                custom_stop = custom_parameters.get("stop")
                if isinstance(custom_stop, list):
                    stop_tokens.extend(custom_stop)
                elif isinstance(custom_stop, str):
                    stop_tokens.append(custom_stop)

            try:
                # Use llama-server's chat completion API (GPU + KV cache reuse!)
                # Gather advanced parameters from custom_parameters
                extra = {}
                if custom_parameters:
                    for key in [
                        "top_k","min_p","typical_p","tfs_z","repeat_penalty",
                        "presence_penalty","frequency_penalty","penalty_last_n",
                        "mirostat","mirostat_tau","mirostat_eta","seed","n_keep",
                        "ignore_eos","grammar","logit_bias","n_probs",
                    ]:
                        if key in custom_parameters:
                            extra[key] = custom_parameters[key]

                response_stream = model_handle.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_tokens,
                    stream=True,
                    **extra,
                )

                # Accumulate streaming chunks into full response
                full_text = ""
                is_first_token = True
                for chunk in response_stream:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta_obj = chunk['choices'][0].get('delta', {})
                        delta = delta_obj.get('content', '')
                        if delta:
                            full_text += delta
                            # Invoke streaming callback for timing metrics (TTFT, ITL)
                            if stream_callback:
                                stream_callback(delta, is_first=is_first_token)
                                is_first_token = False

                response = {"choices": [{"text": full_text}]}

            except Exception as gen_error:
                logger.error(f"llama-server streaming failed: {gen_error}")
                # Fallback to non-streaming
                try:
                    response_data = model_handle.chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop_tokens,
                        stream=False,
                    )
                    message = response_data['choices'][0].get('message', {})
                    response = {"choices": [{"text": message.get('content', '')}]}
                except Exception as e:
                    logger.error(f"llama-server inference failed: {e}")
                    raise

            # Extract and clean the response text using shared utility
            from ..utils.response_cleaner import clean_model_response

            response_text = response["choices"][0]["text"]

            # LOG RAW RESPONSE FROM LLAMA.CPP (BEFORE ANY PROCESSING)
            logger.info("="*80)
            logger.info(f"[LLAMA.CPP RAW RESPONSE] ({len(response_text)} chars)")
            logger.info(response_text)
            logger.info("="*80)
            logger.info(f"[LLAMA.CPP] Has <think> tags: {'<think>' in response_text.lower()}")
            logger.info(f"[LLAMA.CPP] Has </think> tags: {'</think>' in response_text.lower()}")

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
