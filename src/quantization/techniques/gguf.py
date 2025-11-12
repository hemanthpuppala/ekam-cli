"""GGUF quantization technique using llama.cpp."""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

# Local GGUF inspector utilities
from ...models.gguf_inspect import detect_mmproj_input_dim

from ...models.model import ModelInfo
from ...models.provider import ProviderType
from ...models.vlm_detector import VLMDetector, VLMArchitecture
from ..models import QuantizationTask, QuantizationType, TaskStatus
from .base import BaseQuantizer


class GGUFQuantizer(BaseQuantizer):
    """GGUF quantization using llama.cpp tools."""

    def __init__(self):
        """Initialize GGUF quantizer."""
        self.quantize_binary: Optional[Path] = None
        self.convert_script: Optional[Path] = None

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported quantization types (all 40 GGUF formats)."""
        return [
            # Full precision formats
            QuantizationType.GGUF_F32,
            QuantizationType.GGUF_F16,
            QuantizationType.GGUF_BF16,

            # 8-bit quantization
            QuantizationType.GGUF_Q8_0,

            # 6-bit quantization
            QuantizationType.GGUF_Q6_K,

            # 5-bit quantization
            QuantizationType.GGUF_Q5_K,
            QuantizationType.GGUF_Q5_K_M,
            QuantizationType.GGUF_Q5_K_S,
            QuantizationType.GGUF_Q5_0,
            QuantizationType.GGUF_Q5_1,

            # 4-bit quantization
            QuantizationType.GGUF_IQ4_XS,
            QuantizationType.GGUF_IQ4_NL,
            QuantizationType.GGUF_Q4_K,
            QuantizationType.GGUF_Q4_K_M,
            QuantizationType.GGUF_Q4_K_S,
            QuantizationType.GGUF_Q4_0,
            QuantizationType.GGUF_Q4_1,

            # 3-bit quantization
            QuantizationType.GGUF_IQ3_M,
            QuantizationType.GGUF_IQ3_S,
            QuantizationType.GGUF_IQ3_XS,
            QuantizationType.GGUF_IQ3_XXS,
            QuantizationType.GGUF_Q3_K,
            QuantizationType.GGUF_Q3_K_L,
            QuantizationType.GGUF_Q3_K_M,
            QuantizationType.GGUF_Q3_K_S,

            # 2-bit quantization
            QuantizationType.GGUF_IQ2_M,
            QuantizationType.GGUF_IQ2_S,
            QuantizationType.GGUF_IQ2_XS,
            QuantizationType.GGUF_IQ2_XXS,
            QuantizationType.GGUF_Q2_K,
            QuantizationType.GGUF_Q2_K_S,

            # 1-bit quantization
            QuantizationType.GGUF_IQ1_M,
            QuantizationType.GGUF_IQ1_S,

            # Ternary quantization (experimental)
            QuantizationType.GGUF_TQ1_0,
            QuantizationType.GGUF_TQ2_0,
        ]

    def check_availability(self) -> tuple[bool, str]:
        """Check if GGUF quantization tools are available.

        Returns:
            (available, message) tuple
        """
        # For GGUF quantization of HuggingFace models, we need:
        # 1. llama-cpp-python with quantization support
        # 2. llama.cpp tools (convert_hf_to_gguf.py + llama-quantize)

        # Check for llama-cpp-python (for quantization API)
        has_llama_cpp = False
        try:
            import llama_cpp
            if hasattr(llama_cpp, 'llama_model_quantize'):
                has_llama_cpp = True
                logger.info("Found llama-cpp-python with quantization support")
        except ImportError:
            logger.warning("llama-cpp-python not found")

        # Check for llama.cpp CLI tools (required for HF→GGUF conversion)
        # Look for llama-quantize and convert script
        possible_quantize_paths = [
            # macOS paths
            Path("/opt/homebrew/bin/llama-quantize"),
            Path("/opt/homebrew/Cellar/llama.cpp/6730/bin/llama-quantize"),
            Path("/usr/local/bin/llama-quantize"),
            # Linux paths
            Path("/usr/bin/llama-quantize"),
            Path("/usr/local/bin/llama-quantize"),
            # Build output directories (common for source builds)
            Path("./llama.cpp/llama-quantize"),
            Path("./llama.cpp/build/bin/llama-quantize"),
            Path("./llama.cpp/build/llama-quantize"),
            Path("./llama.cpp/bin/llama-quantize"),
            # User home directory
            Path.home() / ".llama.cpp" / "quantize",
            Path.home() / ".llama.cpp" / "llama-quantize",
            Path.home() / "llama.cpp" / "llama-quantize",
            Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
            # Alternative names
            Path("./llama.cpp/quantize"),
        ]

        possible_convert_paths = [
            # macOS paths
            Path("/opt/homebrew/bin/convert_hf_to_gguf.py"),
            Path("/opt/homebrew/Cellar/llama.cpp/6730/bin/convert_hf_to_gguf.py"),
            Path("/usr/local/bin/convert_hf_to_gguf.py"),
            # Linux paths
            Path("/usr/bin/convert_hf_to_gguf.py"),
            Path("/usr/local/bin/convert_hf_to_gguf.py"),
            # Build/source directories
            Path("./llama.cpp/convert_hf_to_gguf.py"),
            Path("./llama.cpp/convert-hf-to-gguf.py"),
            Path("./llama.cpp/convert.py"),
            # User home directory
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            Path.home() / "llama.cpp" / "convert-hf-to-gguf.py",
        ]

        # Find llama-quantize
        for path in possible_quantize_paths:
            if path.exists():
                self.quantize_binary = path
                logger.info(f"Found llama-quantize at {path}")
                break

        # Try to find in PATH using shutil.which (cross-platform)
        if not self.quantize_binary:
            import shutil
            quantize_path = shutil.which("llama-quantize")
            if quantize_path:
                self.quantize_binary = Path(quantize_path)
                logger.info(f"Found llama-quantize in PATH: {self.quantize_binary}")
            else:
                # Fallback to 'which' command for Unix systems
                try:
                    result = subprocess.run(
                        ["which", "llama-quantize"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        self.quantize_binary = Path(result.stdout.strip())
                        logger.info(f"Found llama-quantize in PATH: {self.quantize_binary}")
                except Exception as e:
                    logger.debug(f"Error checking for llama-quantize: {e}")

        # Find convert_hf_to_gguf.py
        for path in possible_convert_paths:
            if path.exists():
                self.convert_script = path
                logger.info(f"Found convert_hf_to_gguf.py at {path}")
                break

        # Log what we found for debugging
        logger.debug(f"llama-quantize binary: {self.quantize_binary if self.quantize_binary else 'NOT FOUND'}")
        logger.debug(f"convert_hf_to_gguf.py script: {self.convert_script if self.convert_script else 'NOT FOUND'}")

        # If we have llama-quantize and convert script, we can do full workflow
        if self.quantize_binary and self.convert_script:
            logger.info(f"✓ Full GGUF workflow available (quantize: {self.quantize_binary}, convert: {self.convert_script})")
            return True, f"Using llama.cpp tools (full HF→GGUF workflow)"
        elif self.quantize_binary:
            logger.warning(f"⚠ Only llama-quantize found (no convert script) - can only quantize existing GGUF files")
            return True, f"Using llama-quantize (GGUF→GGUF quantization only)"

        # Tools not found - provide helpful error message with specific diagnostics
        missing_tools = []
        if not self.quantize_binary:
            missing_tools.append("llama-quantize binary")
        if not self.convert_script:
            missing_tools.append("convert_hf_to_gguf.py script")

        logger.error(f"Missing required tools: {', '.join(missing_tools)}")

        error_msg = (
            f"GGUF quantization requires llama.cpp tools.\n\n"
            f"❌ Missing: {', '.join(missing_tools)}\n\n"
            "📦 Install llama.cpp:\n"
            "  macOS:  brew install llama.cpp\n"
            "  Linux:  Build from source (https://github.com/ggml-org/llama.cpp)\n"
            "         Make sure to run 'make llama-quantize' after building\n\n"
            "🔍 Or add llama.cpp to your PATH:\n"
            "  export PATH=$PATH:/path/to/llama.cpp/build/bin\n\n"
            "💡 Alternative: Use Generic quantization (FP16/INT8/INT4)\n"
            "   - Works without additional tools\n"
            "   - Produces .safetensors output\n\n"
            "💡 Or: Download pre-quantized GGUF models from HuggingFace\n"
            "   - Search for models with 'GGUF' in the name\n"
            "   - Example: TheBloke's quantized models"
        )
        return False, error_msg

    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for quantization.

        Args:
            model_info: Model information

        Returns:
            Path to model, or None if not found
        """
        if model_info.provider == ProviderType.GGUF:
            # GGUF provider stores local path
            if hasattr(model_info, "local_path") and model_info.local_path:
                return Path(model_info.local_path)
            # Try source_path (new unified field)
            elif model_info.source_path:
                return model_info.source_path

        elif model_info.provider == ProviderType.OLLAMA:
            # Ollama models: use source_path which points to blob file
            if model_info.source_path:
                logger.info(f"Using Ollama model blob: {model_info.source_path}")
                return model_info.source_path
            else:
                logger.warning(f"Ollama model {model_info.name} has no source_path")
                return None

        elif model_info.provider == ProviderType.HUGGINGFACE:
            # HuggingFace models: use model_id as path or download location
            # The model_id might be a local path or HF repo ID
            model_id = model_info.model_id

            # Check if it's a local path
            local_path = Path(model_id)
            if local_path.exists():
                logger.info(f"Using local HuggingFace model: {local_path}")
                return local_path

            # Check in HuggingFace cache
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if hf_cache.exists():
                # HF cache uses format: models--<org>--<model>
                cache_name = f"models--{model_id.replace('/', '--')}"
                cache_path = hf_cache / cache_name / "snapshots"

                if cache_path.exists():
                    # Get the latest snapshot
                    snapshots = list(cache_path.iterdir())
                    if snapshots:
                        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
                        logger.info(f"Found HuggingFace model in cache: {latest}")
                        return latest

            # Model not found locally, will need to download
            logger.warning(f"HuggingFace model {model_id} not found locally")
            return Path(model_id)  # Return as-is, conversion will handle download

        return None

    def _convert_and_quantize_vlm_components(
        self,
        hf_model_path: Path,
        task: QuantizationTask,
        vlm_info,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Convert and quantize VLM components separately.

        This handles component-level VLM quantization:
        - Vision encoder selected: quantized vision + F16 language
        - Language decoder selected: F16 vision + quantized language
        - Both selected: quantized vision + quantized language

        Args:
            hf_model_path: Path to HF model directory
            task: Quantization task
            vlm_info: VLM information
            progress_callback: Optional progress callback

        Returns:
            True if successful
        """
        logger.info(f"🔧 VLM Component-Level Quantization: {task.vlm_components}")
        logger.info("   Generating both language decoder and vision encoder files")

        output_dir = task.output_path.parent
        base_name = task.output_path.stem

        # Step 1: Convert language model (WITHOUT --mmproj)
        # Note: We still pass vlm_info so that --trust-remote-code is added for custom architectures
        logger.info("📝 Step 1/4: Converting language decoder to F16...")
        language_f16_path = output_dir / f"{base_name}_language_f16.gguf"

        task.progress = 0.0
        # For language-only: pass vlm_info but with is_vlm=False
        # This ensures --trust-remote-code is added but NOT --mmproj
        class LanguageOnlyVLMInfo:
            is_vlm = False  # Don't add --mmproj
            architecture = vlm_info.architecture if vlm_info else None

        if not self._convert_hf_to_gguf(
            hf_model_path, language_f16_path, task, vlm_info=LanguageOnlyVLMInfo(), progress_callback=progress_callback
        ):
            return False

        logger.info(f"✓ Language decoder F16: {language_f16_path.name}")

        # Step 2: Convert vision encoder (WITH --mmproj)
        logger.info("📝 Step 2/4: Converting vision encoder to F16...")
        vision_f16_path = output_dir / f"mmproj-{base_name}_f16.gguf"

        task.progress = 25.0
        if not self._convert_hf_to_gguf(
            hf_model_path, vision_f16_path, task, vlm_info=vlm_info, progress_callback=progress_callback
        ):
            return False
        logger.info(f"✓ Vision encoder F16: {vision_f16_path.name}")

        # Step 3 & 4: Quantize based on component selection
        language_final_path = language_f16_path
        vision_final_path = vision_f16_path

        if task.vlm_components in ["language", "both"]:
            quant_type = task.language_decoder_type or task.quantization_type
            logger.info(f"📝 Step 3/4: Quantizing language decoder to {quant_type.value}...")
            # Name language file with the selected quant type for clarity
            language_final_path = output_dir / f"{base_name}_{quant_type.value}_language.gguf"

            task.progress = 50.0
            if not self._quantize_gguf_file(
                language_f16_path, language_final_path, quant_type, progress_callback
            ):
                return False

            logger.info(f"✓ Language decoder quantized: {language_final_path.name}")

        if task.vlm_components in ["vision", "both"]:
            quant_type = task.vision_encoder_type or task.quantization_type
            logger.info(f"📝 Step 4/4: Quantizing vision encoder to {quant_type.value}...")
            # Name projector file with the selected quant type for clarity
            vision_final_path = output_dir / f"mmproj-{base_name}_{quant_type.value}.gguf"

            task.progress = 75.0

            # Try to quantize vision encoder
            # Note: Vision encoders often have small tensors (patch embeddings) that can't be quantized
            quantization_success = self._quantize_gguf_file(
                vision_f16_path, vision_final_path, quant_type, progress_callback
            )

            if not quantization_success:
                # First attempt failed; try a safe fallback ladder before F16
                logger.warning(f"⚠️  Vision encoder quantization to {quant_type.value} failed; trying fallbacks")
                fallback_chain = []
                try:
                    from ..models import QuantizationType as QT
                    # Pragmatic ladder: Q6_K → Q5_K_M → IQ4_NL → Q4_K_M
                    fallback_chain = [QT.GGUF_Q6_K, QT.GGUF_Q5_K_M, QT.GGUF_IQ4_NL, QT.GGUF_Q4_K_M]
                except Exception:
                    fallback_chain = []

                selected_fallback = None
                for fb in fallback_chain:
                    try:
                        trial_path = output_dir / f"mmproj-{base_name}_{fb.value}.gguf"
                        logger.info(f"Attempting fallback vision quantization: {fb.value}")
                        if self._quantize_gguf_file(vision_f16_path, trial_path, fb, progress_callback):
                            selected_fallback = fb
                            vision_final_path = trial_path
                            break
                    except Exception:
                        continue

                if selected_fallback is None:
                    # All fallbacks failed — use F16
                    logger.info("Using F16 precision for vision encoder instead")
                    vision_final_path = vision_f16_path
                    try:
                        from ..models import QuantizationType as QT
                        task.attempted_vision_quant_type = quant_type
                        task.warning_message = (
                            f"Vision encoder could not be quantized to {quant_type.value.upper()}; using F16 instead. "
                            f"You can quantize the vision encoder separately with a different type from the pipeline."
                        )
                        task.vision_encoder_type = QT.GGUF_F16
                    except Exception:
                        pass
                else:
                    # We selected a fallback type successfully
                    try:
                        from ..models import QuantizationType as QT
                        task.attempted_vision_quant_type = quant_type
                        task.vision_encoder_type = selected_fallback
                        task.warning_message = (
                            f"Vision encoder could not be quantized to {quant_type.value.upper()}; used {selected_fallback.value.upper()} instead."
                        )
                    except Exception:
                        pass
                    logger.info(f"✓ Vision encoder quantized with fallback: {selected_fallback.value}")
            else:
                logger.info(f"✓ Vision encoder quantized: {vision_final_path.name}")

        # Store file paths and sizes in task metadata
        task.output_path = language_final_path
        task.vlm_language_file = str(language_final_path)
        task.vlm_vision_file = str(vision_final_path)

        # Calculate combined file size
        language_size_gb = language_final_path.stat().st_size / (1024 ** 3)
        vision_size_gb = vision_final_path.stat().st_size / (1024 ** 3)
        task.vlm_language_size_gb = language_size_gb
        task.vlm_vision_size_gb = vision_size_gb

        task.status = TaskStatus.COMPLETED
        task.progress = 100.0
        task.eta_seconds = 0.0

        logger.info(f"✅ VLM Quantization Complete!")
        logger.info(f"   Language: {language_final_path.name} ({language_size_gb:.2f} GB)")
        logger.info(f"   Vision:   {vision_final_path.name} ({vision_size_gb:.2f} GB)")
        logger.info(f"   Total:    {language_size_gb + vision_size_gb:.2f} GB")
        logger.info(f"⚠️  Both files are required together for inference")

        return True

    def _quantize_gguf_file(
        self,
        source_path: Path,
        output_path: Path,
        quant_type: QuantizationType,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Quantize a single GGUF file.

        Args:
            source_path: Input GGUF file (F16)
            output_path: Output quantized GGUF file
            quant_type: Quantization type to apply
            progress_callback: Optional progress callback

        Returns:
            True if successful
        """
        if not self.quantize_binary:
            logger.error("llama-quantize binary not found")
            return False

        # Build quantization command
        quant_type_map = {
            # Full precision formats
            QuantizationType.GGUF_F32: "F32",
            QuantizationType.GGUF_F16: "F16",
            QuantizationType.GGUF_BF16: "BF16",
            # 8-bit
            QuantizationType.GGUF_Q8_0: "Q8_0",
            # 6-bit
            QuantizationType.GGUF_Q6_K: "Q6_K",
            # 5-bit
            QuantizationType.GGUF_Q5_K: "Q5_K",
            QuantizationType.GGUF_Q5_K_S: "Q5_K_S",
            QuantizationType.GGUF_Q5_K_M: "Q5_K_M",
            QuantizationType.GGUF_Q5_0: "Q5_0",
            QuantizationType.GGUF_Q5_1: "Q5_1",
            # 4-bit
            QuantizationType.GGUF_Q4_K: "Q4_K",
            QuantizationType.GGUF_Q4_K_S: "Q4_K_S",
            QuantizationType.GGUF_Q4_K_M: "Q4_K_M",
            QuantizationType.GGUF_Q4_0: "Q4_0",
            QuantizationType.GGUF_Q4_1: "Q4_1",
            # 3-bit
            QuantizationType.GGUF_Q3_K: "Q3_K",
            QuantizationType.GGUF_Q3_K_S: "Q3_K_S",
            QuantizationType.GGUF_Q3_K_M: "Q3_K_M",
            QuantizationType.GGUF_Q3_K_L: "Q3_K_L",
            # 2-bit
            QuantizationType.GGUF_Q2_K: "Q2_K",
            QuantizationType.GGUF_Q2_K_S: "Q2_K_S",
            # IQ formats (Importance-weighted Quantization)
            QuantizationType.GGUF_IQ4_XS: "IQ4_XS",
            QuantizationType.GGUF_IQ4_NL: "IQ4_NL",
            QuantizationType.GGUF_IQ3_S: "IQ3_S",
            QuantizationType.GGUF_IQ3_M: "IQ3_M",
            QuantizationType.GGUF_IQ3_XS: "IQ3_XS",
            QuantizationType.GGUF_IQ3_XXS: "IQ3_XXS",
            QuantizationType.GGUF_IQ2_S: "IQ2_S",
            QuantizationType.GGUF_IQ2_M: "IQ2_M",
            QuantizationType.GGUF_IQ2_XS: "IQ2_XS",
            QuantizationType.GGUF_IQ2_XXS: "IQ2_XXS",
            QuantizationType.GGUF_IQ1_S: "IQ1_S",
            QuantizationType.GGUF_IQ1_M: "IQ1_M",
        }

        quant_type_arg = quant_type_map.get(quant_type)
        if not quant_type_arg:
            logger.error(f"Unknown quantization type: {quant_type}")
            return False

        cmd = [
            str(self.quantize_binary),
            "--allow-requantize",
            str(source_path),
            str(output_path),
            quant_type_arg,
        ]

        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                logger.debug(f"Quantize: {line.strip()}")

            return_code = process.wait()

            if return_code == 0:
                logger.info(f"✓ Quantization successful: {output_path.name}")
                return True
            else:
                logger.error(f"Quantization failed with code {return_code}")
                # Remove partial/corrupted output if created to avoid confusion
                try:
                    if output_path.exists():
                        output_path.unlink()
                        logger.warning(f"Removed partial output: {output_path.name}")
                except Exception as _e:
                    logger.debug(f"Could not remove partial output {output_path}: {_e}")
                return False

        except Exception as e:
            logger.error(f"Quantization error: {e}", exc_info=True)
            return False

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Perform GGUF quantization."""
        # Check if quantize tool is available
        available, message = self.check_availability()
        if not available:
            task.status = TaskStatus.FAILED
            task.error = message
            logger.error(f"Quantization failed: {message}")
            return False

        # Get source model path or HF model ID
        source_path = self.get_source_model_path(task.model_info)

        # For HuggingFace models, we need a local path for conversion
        # If source_path doesn't exist, try to get cached path from HF
        if task.model_info.provider == ProviderType.HUGGINGFACE:
            if not source_path or not source_path.exists():
                # Try to get cached model path from HuggingFace
                try:
                    from huggingface_hub import snapshot_download
                    logger.info(f"Model path not found, checking HuggingFace cache for: {task.model_info.model_id}")
                    # This will find the cached model or download if needed
                    source_path = Path(snapshot_download(
                        repo_id=task.model_info.model_id,
                        local_files_only=True,  # Only use cache, don't download
                    ))
                    logger.info(f"Found model in HuggingFace cache: {source_path}")
                except Exception as e:
                    logger.error(f"Could not find model in cache: {e}")
                    task.status = TaskStatus.FAILED
                    task.error = f"Model not found in cache: {task.model_info.model_id}. Please ensure the model is downloaded."
                    return False

            if source_path and source_path.exists():
                # Check if it's a GGUF file or HF model directory
                if source_path.is_file() and source_path.suffix == '.gguf':
                    logger.info(f"Found GGUF file: {source_path}")
                elif source_path.is_dir():
                    # It's a HF model directory (safetensors format) - needs conversion

                    # PRODUCTION CHECK: Verify architecture compatibility with llama.cpp BEFORE conversion
                    # This prevents wasted time and provides clear feedback for unsupported models
                    logger.info("Checking architecture compatibility with llama.cpp/GGUF...")
                    
                    try:
                        import json
                        config_path = source_path / "config.json"
                        if config_path.exists():
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            
                            arch_type = config.get('model_type', 'unknown')
                            logger.debug(f"Model architecture: {arch_type}")
                            
                            # Check if architecture is supported by llama.cpp
                            is_supported, support_msg = self._check_llamacpp_architecture_support(arch_type)
                            
                            if not is_supported:
                                logger.error(f"Architecture '{arch_type}' not supported by llama.cpp")
                                task.status = TaskStatus.FAILED
                                task.error = (
                                    f"❌ Model architecture '{arch_type}' not supported by llama.cpp\n\n"
                                    f"Model: {task.model_info.model_id}\n"
                                    f"Architecture: {arch_type}\n\n"
                                    f"{support_msg}\n\n"
                                    f"🔧 Solutions:\n\n"
                                    f"1. **Use Generic FP16** (Recommended - works with ALL models):\n"
                                    f"   → Works with LLMs and VLMs\n"
                                    f"   → 50% size reduction\n"
                                    f"   → No architecture restrictions\n\n"
                                    f"2. **Use OpenVINO** (Intel CPU optimized):\n"
                                    f"   → INT4/INT8 quantization\n"
                                    f"   → Works with most architectures\n\n"
                                    f"3. **Use MLX** (Mac only - if architecture is supported):\n"
                                    f"   → Check MLX compatibility separately\n\n"
                                    f"💡 This model will NOT work with GGUF quantization. Please choose an alternative method."
                                )
                                return False
                            
                            logger.info(f"✓ Architecture '{arch_type}' supported by llama.cpp")
                    
                    except Exception as e:
                        logger.warning(f"Could not pre-check architecture compatibility: {e}")
                        logger.info("Continuing with conversion - llama.cpp will report if unsupported")

                    # Check if it's a VLM (Vision-Language Model) using VLMDetector
                    vlm_info = VLMDetector.detect(source_path)

                    if vlm_info.is_vlm:
                        logger.info(f"Detected VLM: {vlm_info.architecture.display_name} (confidence: {vlm_info.confidence})")

                        # Check if architecture is actually supported by llama.cpp GGUF converter
                        if not vlm_info.architecture.supports_gguf_conversion:
                            task.status = TaskStatus.FAILED
                            task.error = (
                                f"❌ VLM Architecture Not Yet Supported by llama.cpp\n\n"
                                f"Model: {task.model_info.name}\n"
                                f"Architecture: {vlm_info.architecture.display_name}\n"
                                f"Confidence: {vlm_info.confidence}\n\n"
                                f"This VLM architecture is not yet supported by llama.cpp's GGUF converter.\n"
                                f"While the architecture is modern, llama.cpp support is still in development.\n\n"
                                f"✅ CURRENTLY SUPPORTED VLMs IN GGUF:\n"
                                f"  • Qwen3-VL (latest)\n"
                                f"  • Qwen2-VL\n"
                                f"  • Qwen2.5-VL\n"
                                f"  • SmolVLM\n"
                                f"  • More being added regularly\n\n"
                                f"✅ SOLUTIONS:\n"
                                f"  1. Use Generic FP16/BF16 quantization (works on ALL models)\n"
                                f"     → 50% size reduction\n"
                                f"     → No architecture restrictions\n"
                                f"     → Full VLM functionality preserved\n\n"
                                f"  2. Use MLX quantization (if on Apple Silicon)\n"
                                f"     → Optimized for Mac M-series chips\n"
                                f"     → Works with all VLMs\n\n"
                                f"  3. Wait for llama.cpp support\n"
                                f"     → Check llama.cpp GitHub for updates\n"
                                f"     → Monitor: https://github.com/ggml-org/llama.cpp\n\n"
                                f"💡 Recommendation: Select 'Generic quantization' → 'FP16' for immediate use."
                            )
                            return False

                        # Check if architecture supports modern conversion (--mmproj flag)
                        if vlm_info.architecture.supports_modern_conversion:
                            logger.info(f"VLM architecture {vlm_info.architecture.display_name} supports modern GGUF conversion with --mmproj")
                            # Modern VLM: will use --mmproj flag in conversion
                            # Store VLM info in task for conversion step
                            task.vlm_info = vlm_info

                        elif vlm_info.architecture.needs_legacy_conversion:
                            # Legacy VLM: needs special conversion scripts
                            task.status = TaskStatus.FAILED
                            task.error = (
                                f"❌ Legacy VLM Architecture Detected\n\n"
                                f"Model: {task.model_info.name}\n"
                                f"Architecture: {vlm_info.architecture.display_name}\n"
                                f"Language Model: {vlm_info.language_model_type or 'Unknown'}\n"
                                f"Vision Encoder: {vlm_info.vision_encoder_type or 'Unknown'}\n\n"
                                f"This VLM requires legacy conversion scripts not yet supported.\n\n"
                                f"Legacy architectures: {', '.join([a.display_name for a in [VLMArchitecture.LLAVA_1_5, VLMArchitecture.LLAVA_1_6, VLMArchitecture.MINICPM_V_2_5, VLMArchitecture.MINICPM_V_2_6, VLMArchitecture.GLM_EDGE, VLMArchitecture.GRANITE_VISION]])}\n\n"
                                f"✅ SOLUTIONS:\n"
                                f"  1. Use Generic FP16 quantization (works on all VLMs)\n"
                                f"     → 50% size reduction\n"
                                f"     → No architecture restrictions\n\n"
                                f"  2. Find a pre-quantized GGUF version on HuggingFace\n"
                                f"     → Search for '{task.model_info.name} GGUF'\n\n"
                                f"💡 Modern VLMs (Qwen2-VL, Gemma 3, SmolVLM, etc.) are fully supported."
                            )
                            logger.error(f"Legacy VLM architecture not supported: {vlm_info.architecture.display_name}")
                            return False

                        else:
                            # Unknown VLM architecture
                            logger.warning(f"Unknown VLM architecture: {vlm_info.architecture.value}")
                            logger.info("Attempting conversion anyway - may fail if unsupported")
                            task.vlm_info = vlm_info
                    else:
                        # Not a VLM - standard language model
                        logger.info("Detected language-only model (not a VLM)")
                        task.vlm_info = None

                    if not self.convert_script:
                        task.status = TaskStatus.FAILED
                        task.error = (
                            f"HuggingFace model conversion requires convert_hf_to_gguf.py.\n\n"
                            f"The convert script was not found. Please ensure llama.cpp is properly installed:\n"
                            f"  brew install llama.cpp  # macOS\n\n"
                            f"💡 Alternative: Use Generic quantization instead"
                        )
                        logger.error(task.error)
                        return False

                    # Convert HF model to GGUF F16 first
                    logger.info(f"Converting HF model to GGUF format: {source_path}")

                    try:
                        vlm_info = getattr(task, 'vlm_info', None)

                        # For VLMs with component selection, use dual conversion workflow
                        if vlm_info and vlm_info.is_vlm and task.vlm_components:
                            logger.info(f"VLM component-level quantization: {task.vlm_components}")
                            logger.info("⚠️  Both language and vision files will be generated for inference")

                            if not self._convert_and_quantize_vlm_components(
                                source_path, task, vlm_info, progress_callback
                            ):
                                return False

                            # Task is complete - skip normal quantization workflow
                            return True

                        # Standard single-file conversion for non-VLMs or full VLM quantization
                        f16_path = task.output_path.parent / f"{task.output_path.stem}_f16.gguf"

                        # Step 1: Convert safetensors → GGUF F16 (with optional --mmproj for VLMs)
                        if not self._convert_hf_to_gguf(source_path, f16_path, task, vlm_info, progress_callback):
                            return False

                        # Store intermediate file path for user info
                        task.intermediate_file = f16_path

                        # Check if target format is F16/F32/BF16 (no quantization needed)
                        if task.quant_type in [QuantizationType.GGUF_F16, QuantizationType.GGUF_F32, QuantizationType.GGUF_BF16]:
                            # Just rename the converted file - no quantization step needed
                            import shutil
                            if task.quant_type == QuantizationType.GGUF_F16:
                                # Already F16, just rename
                                logger.info(f"Target format is F16 - renaming converted file")
                                shutil.move(str(f16_path), str(task.output_path))
                            else:
                                # Need F32 or BF16 - reconvert with correct outtype
                                logger.info(f"Target format is {task.quant_type.value.upper()} - converting with --outtype {task.quant_type.value}")
                                if not self._convert_hf_to_gguf(
                                    Path(task.model_info.source_path),
                                    task.output_path,
                                    task,
                                    vlm_info,
                                    progress_callback,
                                    outtype=task.quant_type.value
                                ):
                                    return False
                                # Clean up intermediate F16 file
                                if f16_path.exists():
                                    f16_path.unlink()

                            task.progress = 100.0
                            task.status = TaskStatus.COMPLETED
                            logger.info(f"✓ Conversion complete: {task.output_path}")
                            if progress_callback:
                                progress_callback(100.0, 0.0)
                            return True

                        # Update source_path to the F16 GGUF file for quantization
                        source_path = f16_path
                        logger.info(f"Conversion complete, now quantizing: {source_path}")
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = f"Conversion error: {str(e)}"
                        logger.error(f"Conversion failed: {e}", exc_info=True)
                        return False
        elif not source_path or not source_path.exists():
            task.status = TaskStatus.FAILED
            task.error = f"Source model not found: {task.model_info.name}"
            logger.error(task.error)
            return False

        # Ensure output directory exists
        task.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build quantization command - map all 40 GGUF types
        quant_type_map = {
            # Full precision formats
            QuantizationType.GGUF_F32: "F32",
            QuantizationType.GGUF_F16: "F16",
            QuantizationType.GGUF_BF16: "BF16",

            # 8-bit quantization
            QuantizationType.GGUF_Q8_0: "Q8_0",

            # 6-bit quantization
            QuantizationType.GGUF_Q6_K: "Q6_K",

            # 5-bit quantization
            QuantizationType.GGUF_Q5_K: "Q5_K",
            QuantizationType.GGUF_Q5_K_M: "Q5_K_M",
            QuantizationType.GGUF_Q5_K_S: "Q5_K_S",
            QuantizationType.GGUF_Q5_0: "Q5_0",
            QuantizationType.GGUF_Q5_1: "Q5_1",

            # 4-bit quantization
            QuantizationType.GGUF_IQ4_XS: "IQ4_XS",
            QuantizationType.GGUF_IQ4_NL: "IQ4_NL",
            QuantizationType.GGUF_Q4_K: "Q4_K",
            QuantizationType.GGUF_Q4_K_M: "Q4_K_M",
            QuantizationType.GGUF_Q4_K_S: "Q4_K_S",
            QuantizationType.GGUF_Q4_0: "Q4_0",
            QuantizationType.GGUF_Q4_1: "Q4_1",

            # 3-bit quantization
            QuantizationType.GGUF_IQ3_M: "IQ3_M",
            QuantizationType.GGUF_IQ3_S: "IQ3_S",
            QuantizationType.GGUF_IQ3_XS: "IQ3_XS",
            QuantizationType.GGUF_IQ3_XXS: "IQ3_XXS",
            QuantizationType.GGUF_Q3_K: "Q3_K",
            QuantizationType.GGUF_Q3_K_L: "Q3_K_L",
            QuantizationType.GGUF_Q3_K_M: "Q3_K_M",
            QuantizationType.GGUF_Q3_K_S: "Q3_K_S",

            # 2-bit quantization
            QuantizationType.GGUF_IQ2_M: "IQ2_M",
            QuantizationType.GGUF_IQ2_S: "IQ2_S",
            QuantizationType.GGUF_IQ2_XS: "IQ2_XS",
            QuantizationType.GGUF_IQ2_XXS: "IQ2_XXS",
            QuantizationType.GGUF_Q2_K: "Q2_K",
            QuantizationType.GGUF_Q2_K_S: "Q2_K_S",

            # 1-bit quantization
            QuantizationType.GGUF_IQ1_M: "IQ1_M",
            QuantizationType.GGUF_IQ1_S: "IQ1_S",

            # Ternary quantization
            QuantizationType.GGUF_TQ1_0: "TQ1_0",
            QuantizationType.GGUF_TQ2_0: "TQ2_0",
        }

        quant_type_arg = quant_type_map.get(task.quant_type)
        if not quant_type_arg:
            task.status = TaskStatus.FAILED
            task.error = f"Unsupported quantization type: {task.quant_type}"
            logger.error(task.error)
            return False

        cmd = [
            str(self.quantize_binary),
            "--allow-requantize",  # Allow requantizing already quantized models
            str(source_path),
            str(task.output_path),
            quant_type_arg,
        ]

        logger.info(f"Running quantization command: {' '.join(cmd)}")

        # Check if source is already quantized (requantization)
        if source_path.suffix == ".gguf":
            try:
                # Quick check - if file exists and is GGUF, it might be quantized
                logger.warning(
                    f"Requantizing from already quantized GGUF model. "
                    f"This may reduce quality compared to quantizing from FP16/FP32. "
                    f"Source: {source_path.name}"
                )
            except Exception:
                pass

        # Update task status
        task.status = TaskStatus.RUNNING

        # If we did a conversion, progress starts at 50%, otherwise 0%
        base_progress = task.progress if task.progress > 0 else 0.0
        progress_range = 100.0 - base_progress

        try:
            # Run quantization process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            start_time = time.time()
            last_progress = base_progress

            # Monitor output for progress
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break

                logger.debug(f"Quantize output: {line.strip()}")

                # Try to parse progress from output
                match = re.search(r"\[\s*(\d+)/\s*(\d+)\]", line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    quant_progress = (current / total) * 100.0

                    # Scale to remaining progress range
                    task.progress = base_progress + (quant_progress / 100.0) * progress_range

                    # Estimate ETA
                    elapsed = time.time() - start_time
                    if quant_progress > 0:
                        total_estimated = (elapsed / quant_progress) * 100
                        eta = total_estimated - elapsed
                        task.eta_seconds = eta

                        if progress_callback and abs(task.progress - last_progress) >= 1.0:
                            progress_callback(task.progress, eta)
                            last_progress = task.progress

            # Wait for process to complete
            return_code = process.wait()

            if return_code == 0:
                task.status = TaskStatus.COMPLETED
                task.progress = 100.0
                task.eta_seconds = 0.0
                logger.info(f"Quantization completed: {task.output_path}")
                return True
            else:
                task.status = TaskStatus.FAILED
                task.error = f"Quantization process failed with code {return_code}"
                logger.error(task.error)
                return False

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"Quantization error: {str(e)}"
            logger.error(f"Quantization failed: {e}", exc_info=True)
            return False

    def estimate_output_size(
        self, model_info: ModelInfo, quant_type: QuantizationType
    ) -> float:
        """Estimate output file size in GB for all 40 GGUF quantization types."""
        # Size factors relative to original model size
        # Based on bits-per-weight and overhead
        size_factors = {
            # Full precision formats
            QuantizationType.GGUF_F32: 1.0,      # 32-bit = 100% of original
            QuantizationType.GGUF_F16: 0.5,      # 16-bit = 50% of original
            QuantizationType.GGUF_BF16: 0.5,     # 16-bit = 50% of original

            # 8-bit quantization (~8.5 bpw)
            QuantizationType.GGUF_Q8_0: 0.27,    # 8.5 bpw

            # 6-bit quantization (~6.5 bpw)
            QuantizationType.GGUF_Q6_K: 0.21,    # 6.56 bpw

            # 5-bit quantization (~5-6 bpw)
            QuantizationType.GGUF_Q5_K: 0.18,    # 5.54 bpw
            QuantizationType.GGUF_Q5_K_M: 0.18,  # 5.54 bpw
            QuantizationType.GGUF_Q5_K_S: 0.17,  # 5.5 bpw
            QuantizationType.GGUF_Q5_0: 0.18,    # 5.5 bpw
            QuantizationType.GGUF_Q5_1: 0.19,    # 5.75 bpw

            # 4-bit quantization (~4-5 bpw)
            QuantizationType.GGUF_IQ4_XS: 0.13,  # 4.25 bpw - best quality 4-bit
            QuantizationType.GGUF_IQ4_NL: 0.14,  # 4.5 bpw
            QuantizationType.GGUF_Q4_K: 0.15,    # 4.5 bpw
            QuantizationType.GGUF_Q4_K_M: 0.15,  # 4.58 bpw
            QuantizationType.GGUF_Q4_K_S: 0.14,  # 4.55 bpw
            QuantizationType.GGUF_Q4_0: 0.14,    # 4.5 bpw
            QuantizationType.GGUF_Q4_1: 0.15,    # 4.75 bpw

            # 3-bit quantization (~3-4 bpw)
            QuantizationType.GGUF_IQ3_M: 0.11,   # 3.7 bpw
            QuantizationType.GGUF_IQ3_S: 0.10,   # 3.5 bpw
            QuantizationType.GGUF_IQ3_XS: 0.10,  # 3.3 bpw
            QuantizationType.GGUF_IQ3_XXS: 0.09, # 3.06 bpw - smallest 3-bit
            QuantizationType.GGUF_Q3_K: 0.11,    # 3.9 bpw
            QuantizationType.GGUF_Q3_K_L: 0.12,  # 4.0 bpw
            QuantizationType.GGUF_Q3_K_M: 0.11,  # 3.91 bpw
            QuantizationType.GGUF_Q3_K_S: 0.10,  # 3.5 bpw

            # 2-bit quantization (~2-3 bpw)
            QuantizationType.GGUF_IQ2_M: 0.08,   # 2.7 bpw
            QuantizationType.GGUF_IQ2_S: 0.07,   # 2.5 bpw
            QuantizationType.GGUF_IQ2_XS: 0.07,  # 2.31 bpw
            QuantizationType.GGUF_IQ2_XXS: 0.06, # 2.06 bpw - smallest 2-bit
            QuantizationType.GGUF_Q2_K: 0.08,    # 2.8 bpw
            QuantizationType.GGUF_Q2_K_S: 0.08,  # 2.67 bpw

            # 1-bit quantization (~1.5 bpw)
            QuantizationType.GGUF_IQ1_M: 0.05,   # 1.75 bpw
            QuantizationType.GGUF_IQ1_S: 0.04,   # 1.56 bpw

            # Ternary quantization (experimental, ~1-2 bpw)
            QuantizationType.GGUF_TQ1_0: 0.04,   # ~1.69 bpw
            QuantizationType.GGUF_TQ2_0: 0.06,   # ~2.06 bpw
        }

        factor = size_factors.get(quant_type, 0.5)  # Default to F16 size if unknown
        return model_info.size_gb * factor

    def _check_llamacpp_architecture_support(self, arch_type: str) -> tuple[bool, str]:
        """Check if architecture is supported by llama.cpp dynamically.
        
        llama.cpp supports a specific set of transformer architectures.
        This method checks against known supported architectures.
        
        Args:
            arch_type: Model architecture type (e.g., 'llama', 'hf_olmo', 'mistral')
            
        Returns:
            (is_supported, message) tuple
        """
        # Known supported architectures in llama.cpp
        # Based on convert_hf_to_gguf.py's Model classes
        supported_architectures = {
            # Core Llama family
            'llama', 'llama2', 'llama3',
            
            # Mistral family
            'mistral', 'mixtral',
            
            # Phi family
            'phi', 'phi2', 'phi3', 'phi3small',
            
            # Qwen family
            'qwen', 'qwen2', 'qwen2moe',
            
            # Google
            'gemma', 'gemma2',
            
            # GPT family
            'gpt2', 'gpt_neox', 'gptj', 'gpt_bigcode',
            
            # Other supported
            'falcon', 'baichuan', 'starcoder', 'starcoder2',
            'mpt', 'bloom', 'stablelm', 'refact', 'persimmon',
            'olmo',  # Standard OLMo (not hf_olmo)
            'openelm', 'arctic', 'deepseek', 'deepseek2',
            'command-r', 'dbrx', 'megrez', 'exaone',
            'orion', 'internlm2', 'granite', 'chameleon',
            'cohere', 'mamba'
        }
        
        # Normalize architecture name (lowercase, remove special chars)
        arch_normalized = arch_type.lower().replace('-', '').replace('_', '')
        
        # Check exact match
        if arch_type.lower() in supported_architectures:
            return True, f"Architecture '{arch_type}' is supported by llama.cpp"
        
        # Check normalized match (handles variants like gpt-2 vs gpt2)
        for supported in supported_architectures:
            supported_norm = supported.replace('-', '').replace('_', '')
            if arch_normalized == supported_norm:
                return True, f"Architecture '{arch_type}' is supported by llama.cpp"
        
        # Check if it's a known unsupported architecture
        known_unsupported = {
            'hf_olmo': 'OLMo with HuggingFace wrapper is not supported. Standard OLMo models may work.',
            'olmo_hf': 'OLMo with HuggingFace wrapper is not supported. Standard OLMo models may work.',
        }
        
        if arch_type.lower() in known_unsupported:
            reason = known_unsupported[arch_type.lower()]
            return False, (
                f"llama.cpp does not support this architecture variant.\n"
                f"{reason}\n\n"
                f"Supported architectures include: llama, mistral, phi, qwen, gemma, gpt2, falcon, etc.\n"
                f"For a complete list, visit: https://github.com/ggerganov/llama.cpp"
            )
        
        # Unknown architecture - be optimistic but warn
        logger.warning(f"Architecture '{arch_type}' not in known list, conversion may fail")
        return True, (
            f"Architecture '{arch_type}' is not in the known supported list.\n"
            f"llama.cpp will determine compatibility during conversion.\n"
            f"If conversion fails, try Generic FP16 or OpenVINO quantization instead."
        )

    def _convert_hf_to_gguf(
        self,
        hf_model_path: Path,
        output_path: Path,
        task: QuantizationTask,
        vlm_info = None,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
        outtype: str = "f16",
    ) -> bool:
        """Convert HuggingFace model to GGUF format.

        Args:
            hf_model_path: Path to HF model directory
            output_path: Output GGUF file path
            task: Quantization task
            vlm_info: Optional VLMInfo for VLM models (enables --mmproj)
            progress_callback: Optional progress callback
            outtype: Output precision type (f16, f32, bf16, etc.)

        Returns:
            True if conversion successful
        """
        if not self.convert_script:
            logger.error("convert_hf_to_gguf.py not found")
            return False

        logger.info(f"Converting {hf_model_path} → {output_path}")
        task.progress = 0.0

        # Determine which convert script to use
        # Priority: 1. Local llama.cpp, 2. Homebrew installation
        local_llamacpp = Path("./llama.cpp")
        if local_llamacpp.exists() and (local_llamacpp / "convert_hf_to_gguf.py").exists():
            convert_script = local_llamacpp / "convert_hf_to_gguf.py"
            # Add llama.cpp/gguf-py to PYTHONPATH for bundled gguf library
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{local_llamacpp / 'gguf-py'}:{env.get('PYTHONPATH', '')}"
            logger.info(f"Using local llama.cpp with bundled gguf library")
        else:
            convert_script = self.convert_script
            env = None
            logger.info(f"Using system convert script: {convert_script}")

        # Build conversion command
        cmd = [
            "python3",
            str(convert_script),
            str(hf_model_path),
            "--outfile", str(output_path),
            "--outtype", outtype,  # Convert to specified format (f16, f32, bf16, etc.)
        ]

        # Do NOT add --trust-remote-code here: llama.cpp's converter does not accept it.
        # Any trust_remote_code handling applies to Transformers loading, not this CLI.

        # Add --mmproj flag for VLM models to extract vision encoder separately
        if vlm_info and getattr(vlm_info, "is_vlm", False):
            # The --mmproj flag is boolean only (no filename argument)
            # llama.cpp automatically creates: mmproj-<model_name>.gguf
            cmd.append("--mmproj")

            # Calculate expected mmproj filename based on llama.cpp's naming convention
            mmproj_path = output_path.parent / f"mmproj-{output_path.name}"
            logger.info(f"VLM detected: Will extract vision encoder to {mmproj_path.name}")
            # Store mmproj path for later reference
            task.mmproj_file = mmproj_path

        logger.info(f"Running conversion: {' '.join(cmd)}")

        try:
            # Run conversion process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,  # Use custom env with PYTHONPATH if using local llama.cpp
            )

            start_time = time.time()
            output_lines = []  # Collect output for error reporting

            # Monitor output
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break

                output_lines.append(line.strip())
                logger.debug(f"Convert output: {line.strip()}")

                # Report progress (conversion doesn't have progress bars, just show activity)
                elapsed = time.time() - start_time
                if elapsed > 1 and task.progress < 40:  # First 40% is conversion
                    task.progress = min(40, elapsed / 2)  # Rough estimate
                    if progress_callback:
                        progress_callback(task.progress, None)

            # Wait for completion
            return_code = process.wait()

            if return_code == 0:
                task.progress = 50.0  # Conversion done, quantization next
                logger.info(f"Conversion successful: {output_path}")
                if progress_callback:
                    progress_callback(50.0, None)

                # Post-conversion validation for VLMs: detect mmproj input dimension
                try:
                    if vlm_info and getattr(task, "mmproj_file", None):
                        mmproj_path = getattr(task, "mmproj_file", None)
                        if mmproj_path and Path(mmproj_path).exists():
                            in_dim = detect_mmproj_input_dim(Path(mmproj_path))
                            if in_dim is not None:
                                # Common cases: 1152 (single-crop), 2304 (multi-crop: global+regional)
                                if in_dim == 2304:
                                    logger.warning(
                                        "⚠️  Detected mmproj expecting 2304-dim input (likely multi-crop).\n"
                                        "   llama.cpp single-crop features are 1152-dim. This repo includes a safe\n"
                                        "   concat-zero fallback in tools/mtmd/clip.cpp to avoid crashes. Output quality\n"
                                        "   may be degraded vs. the original multi-crop training."
                                    )
                                elif in_dim == 1152:
                                    logger.info("✓ mmproj expects 1152-dim input (single-crop) — fully compatible")
                                else:
                                    logger.info(f"ℹ️  mmproj expects {in_dim}-dim input (unusual); continuing")
                            # Optional smoke test on demand (uses local llama.cpp binaries)
                            self._maybe_run_vlm_smoke_test(language_path=output_path, mmproj_path=Path(mmproj_path))
                        else:
                            logger.debug("No mmproj file found to validate post-conversion")
                except Exception as e:
                    logger.warning(f"mmproj validation skipped due to error: {e}")
                return True
            else:
                # Conversion failed - provide helpful error message
                full_output = "\n".join(output_lines)

                # Check for specific error types
                is_unicode_error = "UnicodeDecodeError" in full_output
                is_unknown_arch = "does not recognize this architecture" in full_output or "KeyError" in full_output
                is_not_supported = "not supported" in full_output.lower()

                # Handle unsupported/unknown architecture errors
                if is_unicode_error or is_unknown_arch:
                    # Extract model type from config if possible
                    config_path = hf_model_path / "config.json"
                    arch_name = "Unknown"
                    if config_path.exists():
                        try:
                            with open(config_path) as f:
                                config = json.load(f)
                            arch_name = config.get("model_type", "unknown")
                        except:
                            pass

                    error_msg = (
                        f"\n{'='*60}\n"
                        f"GGUF CONVERSION NOT SUPPORTED\n"
                        f"{'='*60}\n\n"
                        f"Model architecture: {arch_name}\n"
                        f"Source: {hf_model_path}\n\n"
                        f"This model architecture is not yet supported by:\n"
                        f"  • Your version of transformers library\n"
                        f"  • llama.cpp's GGUF converter\n\n"
                        f"This is common for:\n"
                        f"  • Vision-Language Models (VLMs) - especially new ones\n"
                        f"  • Newer/experimental architectures\n"
                        f"  • Multimodal models\n\n"
                        f"✓ Available alternatives:\n"
                        f"  • Generic FP16 quantization (works on all models)\n"
                        f"  • BitsAndBytes 4-bit (if CUDA GPU available)\n"
                        f"  • MLX quantization (if Apple Silicon)\n"
                        f"  • Update transformers: pip install --upgrade transformers\n\n"
                        f"💡 Recommendation:\n"
                        f"  Use 'Generic quantization (PyTorch-based)' → 'FP16'\n"
                        f"  for 50% size reduction (works on all models)\n"
                        f"{'='*60}\n"
                    )
                elif is_not_supported:
                    # Extract architecture name if present
                    arch_name = "Unknown"
                    for line in output_lines:
                        if "architecture:" in line.lower():
                            parts = line.split(":")
                            if len(parts) > 1:
                                arch_name = parts[1].strip()
                                break

                    error_msg = (
                        f"\n{'='*60}\n"
                        f"GGUF CONVERSION NOT SUPPORTED\n"
                        f"{'='*60}\n\n"
                        f"Model architecture: {arch_name}\n"
                        f"Source: {hf_model_path}\n\n"
                        f"This model architecture is not supported by llama.cpp's\n"
                        f"GGUF converter. This is common for:\n"
                        f"  • Vision-Language Models (VLMs)\n"
                        f"  • Newer/experimental architectures\n"
                        f"  • Multimodal models\n\n"
                        f"✓ Available alternatives:\n"
                        f"  • Generic FP16 quantization (works on all models)\n"
                        f"  • BitsAndBytes 4-bit (if CUDA GPU available)\n"
                        f"  • GPTQ/AWQ (for pure LLMs, CUDA GPU required)\n\n"
                        f"💡 Recommendation:\n"
                        f"  Go back and select 'Generic quantization (PyTorch-based)'\n"
                        f"  and choose FP16 for 50% size reduction.\n"
                        f"{'='*60}\n"
                    )
                else:
                    # Generic conversion error
                    error_msg = (
                        f"\n{'='*60}\n"
                        f"GGUF CONVERSION FAILED\n"
                        f"{'='*60}\n\n"
                        f"Return code: {return_code}\n"
                        f"Source: {hf_model_path}\n\n"
                        f"Conversion output (last 500 chars):\n"
                        f"{full_output[-500:]}\n\n"
                        f"Try:\n"
                        f"  • Generic FP16 quantization instead\n"
                        f"  • Check llama.cpp compatibility\n"
                        f"  • Update transformers: pip install --upgrade transformers\n"
                        f"{'='*60}\n"
                    )

                task.status = TaskStatus.FAILED
                task.error = error_msg
                logger.error(task.error)
                return False

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"Conversion error: {str(e)}"
            logger.error(f"Conversion failed: {e}", exc_info=True)
            return False

    def _maybe_run_vlm_smoke_test(self, language_path: Path, mmproj_path: Path) -> None:
        """Optionally run a tiny end-to-end VLM smoke test.

        Controlled by environment variable EKAM_VLM_SMOKETEST=1.
        Requires a local llama.cpp binary (llama-mtmd-cli or llava-cli) and a test image.
        """
        if os.environ.get("EKAM_VLM_SMOKETEST", "0") != "1":
            return

        # Find a test image
        candidate_images = [
            Path("assets/image_3.jpg"),
            Path("assets/test.jpg"),
            Path("examples/image.jpg"),
        ]
        image_path = next((p for p in candidate_images if p.exists()), None)
        if not image_path:
            logger.warning("VLM smoke test skipped: no test image found in assets/examples")
            return

        # Find a suitable binary
        candidate_bins = [
            Path("./llama.cpp/build/bin/llama-mtmd-cli"),
            Path("./llama.cpp/build/bin/llava-cli"),
            Path("./llama.cpp/llava-cli"),
        ]
        bin_path = next((p for p in candidate_bins if p.exists()), None)
        if not bin_path:
            logger.warning("VLM smoke test skipped: llama.cpp CLI binary not found (build it first)")
            return

        cmd = [
            str(bin_path),
            "-m", str(language_path),
            "--mmproj", str(mmproj_path),
            "--image", str(image_path),
            "-p", "Describe the image in one sentence.",
            "-n", "16",
        ]

        logger.info(f"Running VLM smoke test: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode != 0:
                logger.warning(
                    "VLM smoke test failed (non-zero exit). This does not block quantization.\n"
                    f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
                )
            else:
                logger.info("✓ VLM smoke test completed (see stdout for caption)")
        except Exception as e:
            logger.warning(f"VLM smoke test skipped due to error: {e}")
