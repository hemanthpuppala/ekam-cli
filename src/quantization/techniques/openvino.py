"""OpenVINO Quantization for Intel CPUs and iGPUs.

OpenVINO provides production-grade quantization optimized for Intel hardware using
NCCF (Neural Network Compression Framework):

**Supported Quantization Types:**
- INT8 (8-bit): 50% size reduction - RECOMMENDED for production
  • Symmetric & asymmetric modes
  • Per-channel & per-tensor quantization
  • Best quality/performance balance
  • Group sizes: 32, 64, 128, 256

- INT4 (4-bit): 75% size reduction - Maximum compression
  • Symmetric mode only (group-wise)
  • Per-group quantization
  • Group sizes: 32, 64, 128 (128 recommended)
  • Acceptable quality loss for most use cases

- FP16 (half-precision): 50% size reduction - Baseline
  • No compression artifacts
  • Direct FP32→FP16 conversion
  • Ideal for GPU inference

**NOT Supported:**
- INT2, INT3, INT5, INT6 (not available in NNCF)
- FP8 (experimental, not production-ready)

**Hardware Optimization:**
- Intel CPUs: AVX-512, VNNI (Vector Neural Network Instructions), AMX (Advanced Matrix Extensions)
- Intel iGPUs: XMX (Xe Matrix Extensions) for matrix operations
- Intel Arc GPUs: Full acceleration support

**VLM Support (Vision-Language Models):**
- Qwen2-VL, Qwen-VL: Full support ✅
- Phi-3-Vision, Phi-3.5-Vision: Full support ✅
- InternVL2: Full support ✅
- LLaVA: Experimental support ⚠️

**Quality Recommendations:**
- Production/Critical: INT8 (best quality-to-size ratio)
- Development/Testing: INT4 or INT8
- Research: INT4 (maximum compression)

**References:**
- OpenVINO Docs: https://docs.openvino.ai/2025/
- NNCF Quantization: https://docs.openvino.ai/2025/openvino-workflow/model-optimization.html
- Optimum Intel: https://huggingface.co/docs/optimum/intel/index
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from ...models.model import ModelInfo
from ..models import QuantizationTask, QuantizationType, TaskStatus
from .base import BaseQuantizer


class OpenVINOQuantizer(BaseQuantizer):
    """OpenVINO quantizer for Intel-optimized inference.

    Converts HuggingFace models to OpenVINO IR format with INT4/INT8/FP16 quantization.
    Uses NNCF (Neural Network Compression Framework) for advanced quantization.
    
    **Supported Platforms:**
    - Linux (x86_64, ARM64)
    - Windows (x86_64)
    - macOS (x86_64, ARM64 via Rosetta)
    
    **Optimized For:**
    - Intel CPUs (12th gen+ for best performance)
    - Intel iGPUs (Iris Xe+)
    - Intel Arc GPUs
    
    **Not Optimized For:**
    - AMD CPUs/GPUs (will work, but not optimized)
    - NVIDIA GPUs (use GPTQ/AWQ instead)
    - Apple Silicon (use MLX instead)
    """

    def __init__(self):
        """Initialize OpenVINO quantizer."""
        # Don't call super().__init__() - BaseQuantizer doesn't have __init__
        pass

    def check_availability(self) -> tuple[bool, str]:
        """Check if OpenVINO quantization is available.

        Returns:
            (available, message) tuple
        """
        try:
            import openvino as ov  # noqa: F401
            from optimum.intel import OVQuantizer  # noqa: F401
            return True, "OpenVINO and Optimum-Intel available"
        except ImportError as e:
            return False, f"OpenVINO not available: {e}. Install with: pip install openvino optimum[openvino]"

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported quantization types.
        
        OpenVINO NNCF supports only INT4, INT8, and FP16.
        INT2, INT3, INT5, INT6 are NOT supported by NNCF.
        
        Quality ranking (best to worst):
        FP16 > INT8 > INT4
        
        Size ranking (largest to smallest):
        FP16 = INT8 (50%) > INT4 (25%)
        
        Returns:
            List of supported QuantizationType values, ordered by recommendation
            
        Note:
            This list is explicitly defined to match NNCF capabilities.
            OpenVINO does not support 2/3/5/6-bit quantization.
        """
        # Production-recommended ordering
        return [
            QuantizationType.INT8,   # RECOMMENDED - best quality/size balance
            QuantizationType.INT4,   # Maximum compression
            QuantizationType.FP16,   # Baseline, no compression
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

        # OpenVINO works with HuggingFace model IDs directly
        # Return None to indicate we use model_id string instead of local path
        return None

    def estimate_output_size(self, model_info: ModelInfo, quant_type: QuantizationType) -> float:
        """Estimate output file size in GB.
        
        OpenVINO IR format includes:
        - .xml file (model graph)
        - .bin file (weights)
        - config.json and other metadata
        
        Size estimates include ~5-10% overhead for IR format metadata.

        Args:
            model_info: Source model with size_gb (typically FP32 or FP16)
            quant_type: Target quantization type

        Returns:
            Estimated size in GB (typically within 15% accuracy)
            
        Note:
            OpenVINO IR format may have slight overhead compared to
            raw safetensors due to graph optimization metadata.
        """
        # Size multipliers (relative to FP32 baseline)
        SIZE_MULTIPLIERS = {
            QuantizationType.INT4: 0.25,   # 4-bit = 25% + overhead
            QuantizationType.INT8: 0.5,    # 8-bit = 50% + overhead
            QuantizationType.FP16: 0.5,    # 16-bit = 50%
        }
        
        multiplier = SIZE_MULTIPLIERS.get(quant_type)
        if multiplier is None:
            # Unknown quantization type - return original size as safe fallback
            logger.warning(f"Unknown quantization type {quant_type}, using original size")
            return model_info.size_gb
        
        # Add 10% overhead for OpenVINO IR format
        estimated_size = (model_info.size_gb * multiplier) * 1.1
        logger.debug(f"Size estimate (OpenVINO IR): {model_info.size_gb:.2f}GB → {estimated_size:.2f}GB ({multiplier*100:.1f}% + IR overhead)")
        return estimated_size

    def _validate_quantization_type(self, quant_type: QuantizationType) -> tuple[bool, Optional[str]]:
        """Validate that quantization type is supported by OpenVINO.
        
        Args:
            quant_type: Quantization type to validate
            
        Returns:
            (is_valid, error_message) tuple
        """
        supported_types = self.get_supported_types()
        if quant_type not in supported_types:
            supported_names = ', '.join([qt.display_name for qt in supported_types])
            return False, (
                f"Quantization type '{quant_type.display_name}' is not supported by OpenVINO NNCF.\n\n"
                f"OpenVINO supports: {supported_names}\n\n"
                f"Note: OpenVINO NNCF uses power-of-2 bit widths only.\n"
                f"INT2, INT3, INT5, INT6 are not available.\n\n"
                f"Recommendations:\n"
                f"  • For best quality: Use INT8\n"
                f"  • For maximum compression: Use INT4\n"
                f"  • For baseline: Use FP16"
            )
        return True, None
    
    def _get_compression_config(self, quant_type: QuantizationType, group_size: int = 128) -> Optional[dict]:
        """Get OpenVINO compression configuration for quantization type.
        
        Args:
            quant_type: Quantization type enum
            group_size: Group size for quantization (32, 64, 128, or 256)
                       Smaller = better quality, larger model
                       Larger = more compression, some quality loss
            
        Returns:
            Compression config dictionary or None for FP16
            
        Note:
            - INT8: Supports both symmetric and asymmetric (using symmetric for speed)
            - INT4: Symmetric only (NNCF limitation)
            - Group size only applies to INT4 (INT8 uses per-channel)
        """
        if quant_type == QuantizationType.INT8:
            return {
                "bits": 8,
                "sym": True,  # Symmetric quantization (faster, slightly lower quality)
                "group_size": 128,  # Per-channel for INT8
            }
        elif quant_type == QuantizationType.INT4:
            return {
                "bits": 4,
                "sym": True,  # INT4 only supports symmetric in NNCF
                "group_size": group_size,  # Critical for INT4 quality
            }
        elif quant_type == QuantizationType.FP16:
            return None  # No compression for FP16
        
        return None

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """Quantize model with OpenVINO NNCF.
        
        Production-grade quantization with comprehensive validation:
        - Validates quantization type is supported
        - Uses optimal compression configurations
        - Exports to OpenVINO IR format (.xml + .bin)
        - Includes tokenizer and configuration files
        - Provides detailed error messages

        Args:
            task: Quantization task with all parameters
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful, False otherwise
            
        Raises:
            No exceptions - all errors are captured in task.error
        """
        logger.info(f"Starting OpenVINO quantization: {task.model_info.model_id}")
        logger.info(f"Target type: {task.quant_type.display_name}")
        
        # VALIDATION 1: Check quantization type is supported
        is_valid, error_msg = self._validate_quantization_type(task.quant_type)
        if not is_valid:
            logger.error(f"Unsupported quantization type: {task.quant_type}")
            task.status = TaskStatus.FAILED
            task.error = error_msg
            return False

        # Log VLM component selection if applicable
        if task.vlm_components:
            logger.info(f"VLM components to quantize: {task.vlm_components}")

        # Check availability and auto-install dependencies if needed
        available, msg = self.check_and_install_dependencies(
            auto_install=True,
            show_progress=True
        )
        if not available:
            task.status = TaskStatus.FAILED
            task.error = msg
            logger.error(msg)
            return False
        
        logger.info(f"✓ OpenVINO available: {msg}")
        task.status = TaskStatus.RUNNING

        try:
            from optimum.intel import OVModelForCausalLM, OVConfig
            from transformers import AutoTokenizer

            # Check if output path already exists
            if task.output_path.exists():
                import shutil
                logger.warning(f"Output path already exists, removing: {task.output_path}")
                shutil.rmtree(task.output_path)

            # Ensure parent directory exists
            task.output_path.parent.mkdir(parents=True, exist_ok=True)

            if progress_callback:
                progress_callback(10.0, None)

            # Get compression configuration using helper method (validated earlier)
            compression_config = self._get_compression_config(task.quant_type, group_size=128)
            
            if compression_config:
                bits = compression_config["bits"]
                group_size = compression_config.get("group_size", "N/A")
                sym = "symmetric" if compression_config["sym"] else "asymmetric"
                logger.info(f"Using INT{bits} {sym} quantization (group_size={group_size})")
            else:
                logger.info("Using FP16 (no compression)")
            
            logger.info(f"Expected output size: ~{self.estimate_output_size(task.model_info, task.quant_type):.2f} GB")

            if progress_callback:
                progress_callback(20.0, None)

            # Determine input source (GGUF file or HF model ID)
            model_source = self.get_source_model_path(task.model_info)
            if model_source and model_source.exists():
                # Using GGUF file (from Ollama conversion)
                model_input = str(model_source)
                logger.info(f"Loading from GGUF file: {model_source}")
            else:
                # Using HuggingFace model ID
                model_input = task.model_info.model_id
                logger.info(f"Loading from HF model: {model_input}")

            # Load and export model to OpenVINO IR format
            logger.info("Converting to OpenVINO IR format...")

            # Create OVConfig with quantization settings
            if compression_config:
                ov_config = OVConfig(quantization_config=compression_config)
            else:
                ov_config = OVConfig()

            # Load model with quantization
            model = OVModelForCausalLM.from_pretrained(
                model_input,
                export=True,
                config=ov_config,
                trust_remote_code=True,
            )

            if progress_callback:
                progress_callback(70.0, None)

            # Load tokenizer (use model_input which could be GGUF or HF ID)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_input,
                    trust_remote_code=True
                )
            except Exception as e:
                logger.warning(f"Could not load tokenizer from {model_input}: {e}")
                # Try with original model_id as fallback
                logger.info(f"Falling back to model_id for tokenizer: {task.model_info.model_id}")
                tokenizer = AutoTokenizer.from_pretrained(
                    task.model_info.model_id,
                    trust_remote_code=True
                )

            # Save quantized model
            logger.info(f"Saving quantized model to {task.output_path}")
            model.save_pretrained(task.output_path)
            tokenizer.save_pretrained(task.output_path)

            if progress_callback:
                progress_callback(90.0, None)

            # Verify output
            if not (task.output_path / "openvino_model.xml").exists():
                task.status = TaskStatus.FAILED
                task.error = "OpenVINO conversion completed but model files not found"
                return False

            if progress_callback:
                progress_callback(100.0, None)

            task.status = TaskStatus.COMPLETED
            logger.info(f"✓ OpenVINO quantization successful")
            return True

        except Exception as e:
            logger.error(f"OpenVINO quantization failed: {e}")
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

    def get_recommendations(self, model_info: ModelInfo) -> Optional[str]:
        """Get quantization recommendations for this model.

        Args:
            model_info: Model to quantize

        Returns:
            Recommendation string or None
        """
        if not self.is_available():
            return "OpenVINO quantization requires: pip install openvino optimum[openvino]"

        recommendations = []

        # Size-based recommendations
        if model_info.size_gb < 4:
            recommendations.append("✓ Small model - INT8 recommended for best quality/speed")
        elif model_info.size_gb < 10:
            recommendations.append("✓ Medium model - INT8 or INT4 for faster inference")
        else:
            recommendations.append("✓ Large model - INT4 recommended for memory savings")

        # Hardware-specific recommendations
        recommendations.append("✓ Optimized for Intel CPUs (AVX-512, VNNI, AMX)")
        recommendations.append("✓ Supports Intel iGPUs with XMX acceleration")

        # VLM-specific recommendations
        if model_info.model_type.value == "vlm":
            vlm_supported = ["qwen2-vl", "qwen-vl", "phi-3-vision", "phi-3.5-vision", "internvl"]
            model_lower = model_info.model_id.lower()
            is_supported = any(kw in model_lower for kw in vlm_supported)

            if is_supported:
                recommendations.append("✓ VLM fully supported by OpenVINO 2025.2")
            else:
                recommendations.append("⚠ VLM support experimental (may require updates)")

        # Size estimates
        expected_size_int8 = model_info.size_gb * 0.5  # ~50% of original
        expected_size_int4 = model_info.size_gb * 0.25  # ~25% of original
        recommendations.append(f"Expected size: INT8={expected_size_int8:.1f}GB, INT4={expected_size_int4:.1f}GB")

        return " | ".join(recommendations)
