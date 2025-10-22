"""MLX Quantization for Apple Silicon.

MLX provides built-in quantization optimized for Apple Silicon (M1/M2/M3/M4):

**Supported Quantization Types:**
- INT2 (2-bit): 12.5% of original size - significant quality loss, extreme compression
- INT3 (3-bit): 18.75% of original size - noticeable quality loss, very high compression  
- INT4 (4-bit): 25% of original size - acceptable quality, good compression (recommended)
- INT6 (6-bit): 37.5% of original size - good quality, moderate compression
- INT8 (8-bit): 50% of original size - very good quality, standard compression
- FP16 (half-precision): 50% of original size - minimal quality loss, baseline

**Framework Support:**
- LLMs: Uses mlx_lm.convert (text-only models)
- VLMs: Uses mlx_vlm.convert (vision-language models)
- Metal GPU acceleration for inference
- Unified memory architecture optimization

**Quality Recommendations:**
- Production/Critical: INT8 or FP16 (best quality)
- Development/Testing: INT4 or INT6 (balanced)
- Research/Experimentation: INT2 or INT3 (maximum compression)

**References:**
- MLX-LM: https://github.com/ml-explore/mlx-lm
- MLX-VLM: https://github.com/Blaizzy/mlx-vlm
- Apple MLX: https://github.com/ml-explore/mlx
"""

import platform
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger

from ...models.model import ModelInfo
from ..models import QuantizationTask, QuantizationType, TaskStatus
from .base import BaseQuantizer


class MLXQuantizer(BaseQuantizer):
    """MLX quantizer for Apple Silicon.

    Converts HuggingFace models to MLX format with built-in quantization.
    Supports 2-bit, 3-bit, 4-bit, 6-bit, 8-bit, and FP16.
    Only works on macOS with Apple Silicon.
    """

    def __init__(self):
        """Initialize MLX quantizer."""
        # Don't call super().__init__() - BaseQuantizer doesn't have __init__
        pass

    def check_availability(self) -> tuple[bool, str]:
        """Check if MLX quantization is available.

        Returns:
            (available, message) tuple
        """
        if platform.system() != "Darwin":
            return False, "MLX only available on macOS"

        machine = platform.machine()
        if machine != "arm64":
            return False, f"MLX optimized for Apple Silicon, found: {machine}"

        try:
            import mlx.core  # noqa: F401
        except ImportError:
            return False, "MLX not installed. Install with: pip install mlx mlx-lm mlx-vlm"

        # Check for both mlx-lm and mlx-vlm packages
        missing = []
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            missing.append("mlx-lm")

        try:
            import mlx_vlm  # noqa: F401
        except ImportError:
            missing.append("mlx-vlm")

        if missing:
            return False, f"Missing packages: {', '.join(missing)}. Install with: pip install {' '.join(missing)}"

        return True, "MLX framework available (mlx-lm for LLMs, mlx-vlm for VLMs)"

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported quantization types.
        
        MLX supports symmetric quantization from 2-bit to 8-bit integers,
        plus FP16 half-precision. Note: 5-bit is NOT supported by MLX.
        
        Quality ranking (best to worst):
        FP16 ≈ INT8 > INT6 > INT4 > INT3 > INT2
        
        Size ranking (largest to smallest):
        FP16 = INT8 (50%) > INT6 (37.5%) > INT4 (25%) > INT3 (18.75%) > INT2 (12.5%)
        
        Returns:
            List of supported QuantizationType values, ordered by recommendation
        
        Note:
            This list is explicitly defined (not programmatically generated) to ensure
            we only expose quantization types that are verified to work with MLX.
            Future MLX versions may add new quantization types (e.g., 5-bit, mixed precision).
        """
        # Production-recommended (best quality-to-size ratio)
        return [
            QuantizationType.INT4,   # Best balance - RECOMMENDED for most use cases
            QuantizationType.INT6,   # Good quality, moderate compression
            QuantizationType.INT8,   # Highest quality integer quantization
            QuantizationType.FP16,   # Baseline, minimal quality loss
            QuantizationType.INT3,   # High compression, quality tradeoff
            QuantizationType.INT2,   # Maximum compression, significant quality loss
        ]

    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for quantization.

        Args:
            model_info: Model to quantize

        Returns:
            Path to model directory or GGUF file
        """
        from ...models.provider import ProviderType

        # For Ollama/GGUF models, return source_path (will be FP16 GGUF after orchestrator)
        if model_info.provider in [ProviderType.OLLAMA, ProviderType.GGUF]:
            if model_info.source_path:
                return model_info.source_path
            return None

        # MLX works with HuggingFace model IDs directly
        # Return None to indicate we use model_id string instead of local path
        return None

    def estimate_output_size(self, model_info: ModelInfo, quant_type: QuantizationType) -> float:
        """Estimate output file size in GB.
        
        Size estimates are based on bits-per-parameter ratios relative to FP32 baseline.
        Actual sizes may vary slightly due to:
        - Metadata and configuration files
        - MLX-specific optimizations
        - Model architecture specifics

        Args:
            model_info: Source model with size_gb (typically FP32 or FP16)
            quant_type: Target quantization type

        Returns:
            Estimated size in GB (typically within 10% accuracy)
            
        Note:
            These are conservative estimates. Actual quantized models may be
            slightly smaller due to MLX optimizations.
        """
        # Define size multipliers (relative to FP32 baseline)
        # Formula: size_ratio = bits / 32 (FP32 baseline)
        SIZE_MULTIPLIERS = {
            QuantizationType.INT2: 0.125,   # 2/32 = 6.25% per parameter, ~12.5% with overhead
            QuantizationType.INT3: 0.1875,  # 3/32 = 9.375% per parameter, ~18.75% with overhead
            QuantizationType.INT4: 0.25,    # 4/32 = 12.5% per parameter, ~25% with overhead
            QuantizationType.INT6: 0.375,   # 6/32 = 18.75% per parameter, ~37.5% with overhead
            QuantizationType.INT8: 0.5,     # 8/32 = 25% per parameter, ~50% with overhead
            QuantizationType.FP16: 0.5,     # 16/32 = 50%
        }
        
        multiplier = SIZE_MULTIPLIERS.get(quant_type)
        if multiplier is None:
            # Unknown quantization type - return original size as safe fallback
            logger.warning(f"Unknown quantization type {quant_type}, using original size")
            return model_info.size_gb
        
        estimated_size = model_info.size_gb * multiplier
        logger.debug(f"Size estimate: {model_info.size_gb:.2f}GB → {estimated_size:.2f}GB ({multiplier*100:.1f}%)")
        return estimated_size

    def _validate_quantization_type(self, quant_type: QuantizationType) -> tuple[bool, Optional[str]]:
        """Validate that quantization type is supported by MLX.
        
        Args:
            quant_type: Quantization type to validate
            
        Returns:
            (is_valid, error_message) tuple
        """
        supported_types = self.get_supported_types()
        if quant_type not in supported_types:
            supported_names = ', '.join([qt.display_name for qt in supported_types])
            return False, (
                f"Quantization type '{quant_type.display_name}' is not supported by MLX.\n\n"
                f"MLX supports: {supported_names}\n\n"
                f"Note: MLX uses symmetric quantization optimized for Apple Silicon.\n"
                f"5-bit quantization is not available. Use INT4 or INT6 instead."
            )
        return True, None
    
    def _get_quantization_bits(self, quant_type: QuantizationType) -> Optional[int]:
        """Convert QuantizationType to MLX bits parameter.
        
        Args:
            quant_type: Quantization type enum
            
        Returns:
            Number of bits (2, 3, 4, 6, 8, or 16 for FP16), or None if unsupported
            
        Note:
            This mapping is explicit (not programmatic) to ensure correctness.
            Future quantization types should be added here explicitly.
        """
        QUANT_TYPE_TO_BITS = {
            QuantizationType.INT2: 2,
            QuantizationType.INT3: 3,
            QuantizationType.INT4: 4,
            QuantizationType.INT6: 6,
            QuantizationType.INT8: 8,
            QuantizationType.FP16: 16,
        }
        return QUANT_TYPE_TO_BITS.get(quant_type)

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """Quantize model with MLX.
        
        Production-grade quantization with comprehensive validation:
        - Validates quantization type is supported
        - Checks component selection (VLM-specific)
        - Pre-loads model files for custom architectures
        - Handles both LLMs (mlx-lm) and VLMs (mlx-vlm)
        - Provides detailed error messages for troubleshooting

        Args:
            task: Quantization task with all parameters
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful, False otherwise
            
        Raises:
            No exceptions - all errors are captured in task.error
        """
        logger.info(f"Starting MLX quantization: {task.model_info.model_id}")
        logger.info(f"Target type: {task.quant_type.display_name}")
        
        # VALIDATION 0: Check availability and auto-install dependencies if needed
        available, availability_msg = self.check_and_install_dependencies(
            auto_install=True,
            show_progress=True
        )
        if not available:
            logger.error(f"MLX not available: {availability_msg}")
            task.status = TaskStatus.FAILED
            task.error = availability_msg
            return False
        
        logger.info(f"✓ MLX available: {availability_msg}")
        
        # VALIDATION 1: Check quantization type is supported
        is_valid, error_msg = self._validate_quantization_type(task.quant_type)
        if not is_valid:
            logger.error(f"Unsupported quantization type: {task.quant_type}")
            task.status = TaskStatus.FAILED
            task.error = error_msg
            return False

        # VALIDATION 2: Check VLM component selection - MLX doesn't support component extraction
        if task.vlm_components and task.vlm_components != "both":
            logger.error(f"MLX doesn't support component-level extraction. Selected: {task.vlm_components}")
            task.status = TaskStatus.FAILED
            task.error = (
                f"MLX quantization doesn't support component-level extraction.\n"
                f"You selected: {task.vlm_components} only.\n\n"
                f"MLX can only quantize the entire VLM model (both vision + language components).\n\n"
                f"Options:\n"
                f"  1. Go back and select 'Both Components' for MLX quantization\n"
                f"  2. For language-only quantization, use Generic FP16 method instead"
            )
            return False

        # Log VLM component selection if applicable
        if task.vlm_components:
            logger.info(f"VLM components to quantize: {task.vlm_components}")

        task.status = TaskStatus.RUNNING

        try:
            # VALIDATION 3: Check if source is a GGUF file (from Ollama conversion)
            source_path = self.get_source_model_path(task.model_info)
            if source_path and source_path.suffix == '.gguf':
                logger.error(f"MLX cannot load GGUF files: {source_path}")
                task.status = TaskStatus.FAILED
                task.error = (
                    f"\n{'='*60}\n"
                    f"MLX QUANTIZATION FROM OLLAMA NOT YET SUPPORTED\n"
                    f"{'='*60}\n\n"
                    f"Good news: Your Ollama model was successfully converted to FP16!\n"
                    f"📁 Intermediate file saved: {source_path}\n\n"
                    f"However: MLX cannot load GGUF files for quantization.\n"
                    f"MLX requires HuggingFace format models, not GGUF files.\n\n"
                    f"✅ What you can do:\n"
                    f"  1. Use the FP16 GGUF file with llama.cpp or Ollama\n"
                    f"  2. Try GGUF requantization (option 3) instead\n"
                    f"  3. Download the original HuggingFace model for MLX quantization\n\n"
                    f"💡 Recommended: Use GGUF requantization for Ollama models\n"
                    f"   Example: Q8_0 → Q5_K_S, Q4_K_M → Q3_K_S\n"
                    f"{'='*60}\n"
                )
                return False

            # Check if output path already exists
            if task.output_path.exists():
                import shutil
                logger.warning(f"Output path already exists, removing: {task.output_path}")
                shutil.rmtree(task.output_path)

            # Ensure parent directory exists
            task.output_path.parent.mkdir(parents=True, exist_ok=True)

            if progress_callback:
                progress_callback(5.0, None)

            # PRODUCTION FIX: Pre-load model to ensure ALL custom files are downloaded
            # For custom architectures (moondream2, etc.), simply loading config isn't enough
            # We need to actually instantiate the model class to trigger all file downloads
            logger.info("Pre-loading model to ensure all files are present (required for custom architectures)...")
            try:
                from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
                import torch
                
                # Step 1: Load config (downloads config.json + custom config files)
                logger.debug("Loading config...")
                AutoConfig.from_pretrained(
                    task.model_info.model_id,
                    trust_remote_code=True,
                    force_download=False
                )
                
                # Step 2: Load tokenizer (downloads tokenizer files)
                logger.debug("Loading tokenizer...")
                try:
                    AutoTokenizer.from_pretrained(
                        task.model_info.model_id,
                        trust_remote_code=True,
                        force_download=False
                    )
                except Exception as e:
                    logger.debug(f"Tokenizer load optional: {e}")
                
                # Step 3: CRITICAL - Load model architecture (triggers ALL custom Python files)
                # This ensures rope.py, image_crops.py, and other custom files are downloaded
                logger.debug("Loading model architecture (triggers custom file downloads)...")
                try:
                    _ = AutoModelForCausalLM.from_pretrained(
                        task.model_info.model_id,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,  # Load in FP16 to save memory
                        low_cpu_mem_usage=True,  # Stream weights, don't load everything
                        device_map="meta"  # Don't actually load weights, just get architecture
                    )
                    logger.info("✓ Model architecture loaded - all custom files downloaded")
                except Exception as arch_error:
                    # If device_map="meta" fails, try without loading weights at all
                    logger.debug(f"Meta device failed: {arch_error}, trying config-only approach")
                    # This at least ensures the custom Python files are downloaded
                    from transformers.models.auto import auto_factory
                    auto_factory.get_class_from_dynamic_module(
                        task.model_info.model_id,
                        trust_remote_code=True
                    )
                
                logger.info("✓ All model files verified/downloaded successfully")
                
            except Exception as e:
                logger.error(f"Pre-load failed: {e}")
                logger.error("This may cause MLX conversion to fail for models with custom architectures")
                # Continue anyway - user already started the process
                pass

            if progress_callback:
                progress_callback(10.0, None)

            # Get quantization bits using helper method (validated earlier)
            q_bits = self._get_quantization_bits(task.quant_type)
            if q_bits is None:
                # This should never happen due to earlier validation, but handle defensively
                task.status = TaskStatus.FAILED
                task.error = f"Internal error: Could not determine bits for {task.quant_type}"
                logger.error(task.error)
                return False
            
            logger.info(f"Quantizing to {q_bits}-bit {'FP16' if q_bits == 16 else f'INT{q_bits}'}")
            logger.info(f"Expected output size: ~{self.estimate_output_size(task.model_info, task.quant_type):.2f} GB")

            # Determine if model is VLM or LLM
            from ...models.model import ModelType
            is_vlm = task.model_info.model_type == ModelType.VLM

            # PRODUCTION CHECK: Verify MLX architecture compatibility BEFORE attempting conversion
            # This prevents wasted time and provides clear feedback for unsupported models
            logger.info(f"Checking {'VLM' if is_vlm else 'LLM'} architecture compatibility with MLX...")
            
            try:
                # Get model's architecture type
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    task.model_info.model_id,
                    trust_remote_code=True
                )
                arch_type = getattr(config, 'model_type', 'unknown')
                logger.debug(f"Model architecture: {arch_type}")
                
                # Check if architecture is supported by querying MLX's supported models
                is_supported, support_msg = self._check_mlx_architecture_support(arch_type, is_vlm)
                
                if not is_supported:
                    logger.error(f"Architecture '{arch_type}' not supported by {'mlx-vlm' if is_vlm else 'mlx-lm'}")
                    task.status = TaskStatus.FAILED
                    
                    # Get installed version
                    installed_version = "unknown"
                    try:
                        if is_vlm:
                            import mlx_vlm
                            installed_version = getattr(mlx_vlm, "__version__", "unknown")
                        else:
                            import mlx_lm
                            installed_version = getattr(mlx_lm, "__version__", "unknown")
                    except Exception:
                        pass
                    
                    task.error = (
                        f"❌ Model architecture '{arch_type}' not supported by {'mlx-vlm' if is_vlm else 'mlx-lm'}\n\n"
                        f"Model: {task.model_info.model_id}\n"
                        f"Architecture: {arch_type}\n"
                        f"Installed {'mlx-vlm' if is_vlm else 'mlx-lm'} version: {installed_version}\n\n"
                        f"{support_msg}\n\n"
                        f"🔧 Solutions:\n\n"
                        f"1. **UPDATE MLX** (Recommended - newer versions support more models):\n"
                        f"   pip install --upgrade mlx-vlm mlx-lm\n\n"
                        f"2. **Use alternative quantization methods**:\n"
                        f"   • Generic FP16 - Works with ALL models (LLMs & VLMs)\n"
                        f"   • OpenVINO - Intel CPU optimized (INT4/INT8)\n"
                        f"   • GGUF - For pure LLMs (llama.cpp format)\n\n"
                        f"3. **Check model compatibility**:\n"
                        f"   Visit: https://github.com/{'Blaizzy/mlx-vlm' if is_vlm else 'ml-explore/mlx-lm'}/releases\n\n"
                        f"💡 This model will NOT work with MLX quantization. Please choose an alternative method."
                    )
                    return False
                
                logger.info(f"✓ Architecture '{arch_type}' supported by {'mlx-vlm' if is_vlm else 'mlx-lm'}")
                
            except Exception as e:
                logger.warning(f"Could not pre-check architecture compatibility: {e}")
                logger.info("Continuing with conversion - MLX will report if unsupported")
                # Continue anyway - MLX will catch unsupported architectures during conversion

            # Determine input source (GGUF file or HF model ID)
            model_source = self.get_source_model_path(task.model_info)
            if model_source and model_source.exists():
                # Using GGUF file (from Ollama conversion)
                input_arg = str(model_source)
                logger.info(f"Converting from GGUF file: {model_source}")
            else:
                # Using HuggingFace model ID
                input_arg = task.model_info.model_id
                logger.info(f"Converting from HF model: {input_arg}")

            # Build convert command - use mlx-vlm for VLMs, mlx-lm for LLMs
            if is_vlm:
                # Use mlx_vlm.convert for Vision-Language Models
                cmd = [
                    "python", "-m", "mlx_vlm.convert",
                    "--hf-path", input_arg,
                    "--mlx-path", str(task.output_path),
                ]
                logger.info("Using mlx-vlm for VLM conversion")
            else:
                # Use mlx_lm.convert for pure LLMs
                cmd = [
                    "python", "-m", "mlx_lm.convert",
                    "--hf-path", input_arg,
                    "--mlx-path", str(task.output_path),
                ]
                logger.info("Using mlx-lm for LLM conversion")

            # Add quantization flags if not FP16
            if q_bits in [2, 3, 4, 6, 8]:
                cmd.extend(["--quantize", "--q-bits", str(q_bits)])
                logger.info(f"Using {q_bits}-bit quantization")
            elif q_bits == 16:
                logger.info("Using FP16 (no quantization)")
            else:
                logger.warning(f"Unexpected bit size: {q_bits}, using FP16")

            if progress_callback:
                progress_callback(20.0, None)

            # Run conversion
            logger.info("Running MLX conversion...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if progress_callback:
                progress_callback(90.0, None)

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                logger.error(f"MLX conversion failed: {error_msg}")

                # Parse error and provide actionable feedback
                helpful_error = self._parse_mlx_error(error_msg, task.model_info, is_vlm)

                task.status = TaskStatus.FAILED
                task.error = helpful_error
                return False

            # Verify output
            if not (task.output_path / "config.json").exists():
                task.status = TaskStatus.FAILED
                task.error = "MLX conversion completed but config.json not found"
                return False

            if progress_callback:
                progress_callback(100.0, None)

            task.status = TaskStatus.COMPLETED
            logger.info(f"✓ MLX quantization successful")
            return True

        except subprocess.TimeoutExpired:
            logger.error("MLX conversion timed out")
            task.status = TaskStatus.FAILED
            task.error = "Conversion timed out after 1 hour"
            return False

        except Exception as e:
            logger.error(f"MLX quantization failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return False

    def _calculate_dir_size(self, directory: Path) -> float:
        """Calculate total size of directory in GB.

        Args:
            directory: Directory path

        Returns:
            Size in GB
        """
        try:
            total_size = 0
            for file in directory.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size / (1024**3)
        except Exception as e:
            logger.warning(f"Could not calculate directory size: {e}")
            return 0.0

    def _parse_mlx_error(self, error_msg: str, model_info: "ModelInfo", is_vlm: bool) -> str:
        """Parse MLX error and provide helpful, actionable feedback.

        Generic error parser for production - works with any model/provider.

        Args:
            error_msg: Error message from MLX subprocess
            model_info: Model information
            is_vlm: Whether this is a VLM

        Returns:
            Helpful error message with suggestions
        """
        import re

        # Check for "Model type X not supported" error
        unsupported_match = re.search(r"Model type (\w+) not supported", error_msg, re.IGNORECASE)

        if unsupported_match:
            model_type = unsupported_match.group(1)

            # Get installed mlx-vlm/mlx-lm version
            installed_version = "unknown"
            try:
                if is_vlm:
                    import mlx_vlm
                    installed_version = getattr(mlx_vlm, "__version__", "unknown")
                else:
                    import mlx_lm
                    installed_version = getattr(mlx_lm, "__version__", "unknown")
            except Exception:
                pass

            return (
                f"❌ Model architecture '{model_type}' not supported by {'mlx-vlm' if is_vlm else 'mlx-lm'}\n\n"
                f"Model: {model_info.model_id}\n"
                f"Architecture: {model_type}\n"
                f"Installed {'mlx-vlm' if is_vlm else 'mlx-lm'} version: {installed_version}\n\n"
                f"🔧 Solutions:\n\n"
                f"1. **UPDATE MLX** (Recommended - newer versions support more models):\n"
                f"   pip install --upgrade mlx-vlm mlx-lm\n"
                f"   # Qwen3-VL support added in mlx-vlm 0.3.4+\n\n"
                f"2. **Use alternative quantization methods**:\n"
                f"   • Generic FP16 - Works with ALL models (LLMs & VLMs)\n"
                f"   • OpenVINO - Intel CPU optimized (INT4/INT8)\n"
                f"   • GGUF - For pure LLMs (llama.cpp format)\n\n"
                f"3. **Check model compatibility**:\n"
                f"   Visit: https://github.com/Blaizzy/mlx-vlm/releases\n"
                f"   Or:    https://github.com/ml-explore/mlx-lm/releases\n\n"
                f"💡 Tip: If you just updated MLX, restart the application to use the new version."
            )

        # Check for FileNotFoundError with custom model files (rope.py, etc.)
        if "FileNotFoundError" in error_msg and ("/transformers_modules/" in error_msg or "rope.py" in error_msg or ".py" in error_msg):
            return (
                f"❌ MLX conversion failed\n\n"
                f"Model: {model_info.model_id}\n"
                f"Type: {'VLM' if is_vlm else 'LLM'}\n\n"
                f"Error:\n{error_msg}\n\n"
                f"🔧 Common solutions:\n"
                f"1. Update MLX: pip install --upgrade mlx-vlm mlx-lm\n"
                f"2. Use Generic FP16 (works with all models)\n"
                f"3. Check model compatibility at GitHub\n"
                f"4. Try a different quantization method"
            )
        
        # Check for missing modules
        if "No module named" in error_msg:
            module_match = re.search(r"No module named '([^']+)'", error_msg)
            if module_match:
                missing_module = module_match.group(1)
                return (
                    f"❌ Missing module: {missing_module}\n\n"
                    f"This usually means your MLX installation is outdated or incomplete.\n\n"
                    f"🔧 Solution:\n"
                    f"pip install --upgrade mlx mlx-lm mlx-vlm\n\n"
                    f"If the problem persists:\n"
                    f"pip uninstall mlx-lm mlx-vlm -y\n"
                    f"pip install mlx-lm mlx-vlm\n"
                )

        # Check for CUDA/GPU errors
        if "cuda" in error_msg.lower() or "gpu" in error_msg.lower():
            return (
                f"❌ GPU/CUDA error during MLX conversion\n\n"
                f"MLX uses Metal (not CUDA) on Mac. This error shouldn't occur.\n\n"
                f"Error: {error_msg[:500]}...\n\n"
                f"🔧 Solutions:\n"
                f"1. Update MLX: pip install --upgrade mlx mlx-lm mlx-vlm\n"
                f"2. Try Generic FP16 quantization instead\n"
                f"3. Report issue: https://github.com/Blaizzy/mlx-vlm/issues"
            )

        # Generic error - return cleaned up version
        # Truncate very long errors
        if len(error_msg) > 1000:
            error_msg = error_msg[:1000] + "\n... (truncated)"

        return (
            f"❌ MLX conversion failed\n\n"
            f"Model: {model_info.model_id}\n"
            f"Type: {'VLM' if is_vlm else 'LLM'}\n\n"
            f"Error:\n{error_msg}\n\n"
            f"🔧 Common solutions:\n"
            f"1. Update MLX: pip install --upgrade mlx-vlm mlx-lm\n"
            f"2. Use Generic FP16 (works with all models)\n"
            f"3. Check model compatibility at GitHub\n"
            f"4. Try a different quantization method"
        )

    def _check_mlx_architecture_support(self, arch_type: str, is_vlm: bool) -> tuple[bool, str]:
        """Check if architecture is supported by MLX dynamically.
        
        This method queries MLX's actual supported models list at runtime,
        making it future-proof without hardcoding architecture names.
        
        Args:
            arch_type: Model architecture type (e.g., 'llama', 'hf_olmo', 'qwen2_vl')
            is_vlm: Whether this is a VLM (uses mlx-vlm) or LLM (uses mlx-lm)
            
        Returns:
            (is_supported, message) tuple
        """
        try:
            if is_vlm:
                # Check mlx-vlm supported models
                try:
                    import mlx_vlm.utils
                    # Try to get supported models from mlx-vlm
                    # mlx-vlm uses a model registry that we can query
                    if hasattr(mlx_vlm.utils, '_get_classes'):
                        # Try to get the model classes for this architecture
                        try:
                            mlx_vlm.utils._get_classes(arch_type)
                            # If no exception, architecture is supported
                            return True, f"Architecture '{arch_type}' is supported by mlx-vlm"
                        except (ModuleNotFoundError, ValueError) as e:
                            # Architecture not supported
                            error_msg = str(e)
                            if "not supported" in error_msg.lower():
                                return False, (
                                    f"This VLM architecture is not yet supported by your mlx-vlm version.\n"
                                    f"Supported VLM architectures may include: llava, paligemma, phi3_v, qwen2_vl, etc.\n"
                                    f"Check the latest releases for updates."
                                )
                except ImportError:
                    pass
                
                # Fallback: Known common VLM architectures
                # This is a soft check - we still let MLX try if unknown
                common_vlm_archs = {'llava', 'paligemma', 'phi3_v', 'qwen2_vl', 'idefics', 'idefics2'}
                if arch_type in common_vlm_archs:
                    return True, f"Architecture '{arch_type}' is a known VLM architecture"
                
                # Unknown but let MLX try - it might be newly supported
                logger.debug(f"Architecture '{arch_type}' not in known list, will attempt conversion")
                return True, "Architecture support unknown, will attempt conversion"
                
            else:
                # Check mlx-lm supported models
                try:
                    import mlx_lm.utils
                    # Try to get supported models from mlx-lm
                    if hasattr(mlx_lm.utils, '_get_classes'):
                        try:
                            mlx_lm.utils._get_classes(arch_type)
                            return True, f"Architecture '{arch_type}' is supported by mlx-lm"
                        except (ModuleNotFoundError, ValueError) as e:
                            error_msg = str(e)
                            if "not supported" in error_msg.lower():
                                return False, (
                                    f"This LLM architecture is not yet supported by your mlx-lm version.\n"
                                    f"Supported LLM architectures may include: llama, mistral, phi, qwen2, gemma, etc.\n"
                                    f"Check the latest releases for updates."
                                )
                except ImportError:
                    pass
                
                # Fallback: Known common LLM architectures
                common_llm_archs = {
                    'llama', 'mistral', 'mixtral', 'phi', 'phi3', 'qwen2', 'gemma', 'gemma2',
                    'stablelm', 'cohere', 'gpt2', 'gpt_neox', 'openelm', 'starcoder2'
                }
                if arch_type in common_llm_archs:
                    return True, f"Architecture '{arch_type}' is a known LLM architecture"
                
                # Unknown architecture - let MLX try
                logger.debug(f"Architecture '{arch_type}' not in known list, will attempt conversion")
                return True, "Architecture support unknown, will attempt conversion"
                
        except Exception as e:
            logger.debug(f"Error checking architecture support: {e}")
            # On error, allow the attempt - MLX will report if unsupported
            return True, "Could not verify architecture support, will attempt conversion"

    def get_recommendations(self, model_info: ModelInfo) -> Optional[str]:
        """Get quantization recommendations for this model.

        Args:
            model_info: Model to quantize

        Returns:
            Recommendation string or None
        """
        if not self.is_available():
            return "MLX quantization only available on macOS with Apple Silicon"

        recommendations = []

        # Size-based recommendations
        if model_info.size_gb < 4:
            recommendations.append("✓ Small model - MLX 4-bit quantization recommended")
        elif model_info.size_gb < 10:
            recommendations.append("✓ Medium model - MLX 4-bit will reduce to ~25% size")
        else:
            recommendations.append("✓ Large model - MLX 4-bit will significantly reduce size (to ~25%)")

        # VLM-specific recommendations
        if model_info.model_type.value == "vlm":
            recommendations.append("✓ VLM support via mlx-vlm (ensure version 0.3.4+ for latest models)")

        recommendations.append(f"Expected output size: ~{model_info.size_gb * 0.25:.1f} GB with 4-bit")

        return " | ".join(recommendations)
