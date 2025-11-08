"""GGUF quantization technique using llama.cpp."""

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ...models.model import ModelInfo
from ...models.provider import ProviderType
from ..models import QuantizationTask, QuantizationType, TaskStatus
from .base import BaseQuantizer


class GGUFQuantizer(BaseQuantizer):
    """GGUF quantization using llama.cpp tools."""

    def __init__(self):
        """Initialize GGUF quantizer."""
        self.quantize_binary: Optional[Path] = None
        self.convert_script: Optional[Path] = None

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported quantization types."""
        return [
            QuantizationType.GGUF_Q4_K_M,
            QuantizationType.GGUF_Q4_K_S,
            QuantizationType.GGUF_Q5_K_M,
            QuantizationType.GGUF_Q5_K_S,
            QuantizationType.GGUF_Q6_K,
            QuantizationType.GGUF_Q8_0,
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

                    # Check if it's a VLM (Vision-Language Model)
                    is_vlm, architecture_name = self._check_if_vlm(source_path)
                    if is_vlm:
                        # VLM detected - check if language-only quantization is requested
                        if task.vlm_components == "language":
                            logger.info(f"VLM detected ({architecture_name}), but language-only quantization requested")
                            logger.info("Proceeding with GGUF conversion of language decoder component only")
                            logger.warning("NOTE: Language-only component extraction is experimental")
                            # TODO: Implement actual language decoder extraction
                            # For now, attempt full model conversion (may fail for some VLMs)
                        else:
                            # Full VLM or vision component quantization - NOT supported
                            task.status = TaskStatus.FAILED
                            task.error = (
                                f"❌ VLM GGUF Conversion Not Supported\n\n"
                                f"Model: {task.model_info.name}\n"
                                f"Architecture: {architecture_name}\n"
                                f"Type: Vision-Language Model (VLM)\n\n"
                                f"Why this fails:\n"
                                f"  • llama.cpp only supports pure language models\n"
                                f"  • VLMs have vision encoders (CLIP, ViT) that GGUF doesn't handle\n"
                                f"  • Multimodal projection layers are not supported\n\n"
                                f"✅ SOLUTIONS:\n"
                                f"  1. Use 'Advanced' quantization for component-level control\n"
                                f"     → Select 'Language Only' to enable GGUF conversion\n"
                                f"  2. Use 'Generic' quantization (GPTQ/AWQ/BnB)\n"
                                f"     → Quantizes entire VLM (vision + language)\n"
                                f"     → Supports FP16/INT8/INT4\n\n"
                                f"Go back and select a different quantization method."
                            )
                            logger.error(f"VLM architecture detected: {architecture_name}. GGUF conversion not supported for full VLMs.")
                            return False

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
                    f16_path = task.output_path.parent / f"{task.output_path.stem}_f16.gguf"

                    try:
                        # Step 1: Convert safetensors → GGUF F16
                        if not self._convert_hf_to_gguf(source_path, f16_path, task, progress_callback):
                            return False

                        # Store intermediate file path for user info
                        task.intermediate_file = f16_path

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

        # Build quantization command
        quant_type_map = {
            QuantizationType.GGUF_Q4_K_M: "Q4_K_M",
            QuantizationType.GGUF_Q4_K_S: "Q4_K_S",
            QuantizationType.GGUF_Q5_K_M: "Q5_K_M",
            QuantizationType.GGUF_Q5_K_S: "Q5_K_S",
            QuantizationType.GGUF_Q6_K: "Q6_K",
            QuantizationType.GGUF_Q8_0: "Q8_0",
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
        """Estimate output file size in GB."""
        size_factors = {
            QuantizationType.GGUF_Q4_K_M: 0.5,
            QuantizationType.GGUF_Q4_K_S: 0.45,
            QuantizationType.GGUF_Q5_K_M: 0.6,
            QuantizationType.GGUF_Q5_K_S: 0.55,
            QuantizationType.GGUF_Q6_K: 0.7,
            QuantizationType.GGUF_Q8_0: 0.9,
        }

        factor = size_factors.get(quant_type, 0.5)
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

    def _check_if_vlm(self, model_path: Path) -> tuple[bool, str]:
        """Check if a HuggingFace model is a Vision-Language Model.

        VLMs cannot be converted to GGUF because llama.cpp doesn't support
        vision encoders or multimodal architectures.

        Args:
            model_path: Path to HuggingFace model directory

        Returns:
            (is_vlm, architecture_name)
        """
        config_path = model_path / "config.json"
        if not config_path.exists():
            logger.warning(f"config.json not found in {model_path}")
            return (False, "Unknown")

        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Get architecture name
            architecture = config.get("architectures", ["Unknown"])[0]

            # Known VLM architecture patterns
            # These architectures have vision encoders and are NOT supported by llama.cpp
            vlm_patterns = [
                "ForConditionalGeneration",  # Qwen3VL, LLaVA, etc.
                "VisionTextDualEncoder",     # CLIP-based models
                "VisionEncoder",             # Vision-only encoders
                "BlipForConditionalGeneration",
                "Blip2ForConditionalGeneration",
                "LlavaForConditionalGeneration",
                "InstructBlipForConditionalGeneration",
                "Pix2StructForConditionalGeneration",
                "VipLlavaForConditionalGeneration",
                "Qwen2VLForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "QwenVLForConditionalGeneration",
                "CogVLMForCausalLM",
                "Idefics",
                "Kosmos",
                "Flamingo",
                "GitForCausalLM",
            ]

            # Check if architecture matches any VLM pattern
            for pattern in vlm_patterns:
                if pattern in architecture:
                    logger.info(f"Detected VLM architecture: {architecture}")
                    return (True, architecture)

            # Check config for vision-related keys
            has_vision_config = any(key in config for key in [
                "vision_config",
                "visual_config",
                "image_encoder",
                "vision_tower",
                "mm_vision_tower",
            ])

            if has_vision_config:
                logger.info(f"Detected VLM config keys in: {architecture}")
                return (True, architecture)

            # Not a VLM
            logger.info(f"Detected language-only architecture: {architecture}")
            return (False, architecture)

        except Exception as e:
            logger.error(f"Error checking VLM status: {e}")
            return (False, "Unknown")

    def _convert_hf_to_gguf(
        self,
        hf_model_path: Path,
        output_path: Path,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Convert HuggingFace model to GGUF format.

        Args:
            hf_model_path: Path to HF model directory
            output_path: Output GGUF file path
            task: Quantization task
            progress_callback: Optional progress callback

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
            "--outtype", "f16",  # Convert to F16 GGUF
        ]

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
                return True
            else:
                # Conversion failed - provide helpful error message
                full_output = "\n".join(output_lines)

                # Check if it's an unsupported architecture error
                if "not supported" in full_output.lower():
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
                        f"Conversion output:\n"
                        f"{full_output[-500:]}\n\n"  # Last 500 chars
                        f"Try:\n"
                        f"  • Generic FP16 quantization instead\n"
                        f"  • Check llama.cpp compatibility\n"
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
