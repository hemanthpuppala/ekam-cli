"""Quantization Orchestrator - Coordinates multi-step quantization workflows.

This orchestrator handles complex quantization pipelines that require format
conversion before quantization. It automatically determines the optimal path
from source format to target quantization.

Examples:
    Ollama Q4_K_M → Generic INT4:
        1. Dequantize Q4_K_M → FP16 (GGUFDequantizer)
        2. Convert GGUF FP16 → HF Safetensors (llama.cpp convert)
        3. Quantize HF → Generic INT4 (GenericQuantizer)

    Ollama Q8_0 → MLX INT4:
        1. Dequantize Q8_0 → FP16 (GGUFDequantizer)
        2. Convert GGUF FP16 → HF Safetensors
        3. Quantize HF → MLX INT4 (MLXQuantizer)
"""

from pathlib import Path
from typing import Optional, Callable, Tuple
from enum import Enum

from loguru import logger

from ..models.model import ModelInfo
from ..models.provider import ProviderType
from ..models.endpoints import ModelType
from .models import QuantizationTask, QuantizationType, TaskStatus
from .converters.gguf_dequantizer import GGUFDequantizer
from .converters.gguf_to_hf import GGUFToHFConverter


class ConversionPath(Enum):
    """Conversion paths for different quantization scenarios."""
    DIRECT = "direct"  # No conversion needed
    DEQUANT_ONLY = "dequant_only"  # Just dequantize GGUF
    GGUF_TO_GENERIC = "gguf_to_generic"  # GGUF → FP16 → Generic
    GGUF_TO_MLX = "gguf_to_mlx"  # GGUF → FP16 → MLX
    GGUF_TO_OPENVINO = "gguf_to_openvino"  # GGUF → FP16 → OpenVINO
    GGUF_TO_ADVANCED = "gguf_to_advanced"  # GGUF → FP16 → HF → GPTQ/AWQ/BnB


class QuantizationDirective(Enum):
    """Instructions from orchestrator to quantizer about what to do.

    Used to tell quantizers whether to:
    - SKIP: Stop after conversion (FP16 is already final output)
    - QUANTIZE_INT8: Convert to INT8 after conversion to FP16
    - QUANTIZE_INT4: Convert to INT4 after conversion to FP16
    - QUANTIZE: Generic quantization instruction (use method_family to determine exact type)
    """
    SKIP = "skip"  # FP16 generic: Final output ready, no quantization needed
    QUANTIZE_INT8 = "quantize_int8"  # INT8 generic: Need to apply INT8 quantization
    QUANTIZE_INT4 = "quantize_int4"  # INT4 generic: Need to apply INT4 quantization
    QUANTIZE = "quantize"  # Generic quantization: Let quantizer decide based on type
    CONTINUE_GGUF = "continue_gguf"  # GGUF re-quantization: Proceed with quantizer
    CONTINUE_MLX = "continue_mlx"  # MLX quantization: Let MLX quantizer handle
    CONTINUE_OPENVINO = "continue_openvino"  # OpenVINO: Let OpenVINO quantizer handle
    CONTINUE_ADVANCED = "continue_advanced"  # Advanced (GPTQ/AWQ/BnB): Let quantizer handle


class QuantizationOrchestrator:
    """Orchestrates multi-step quantization workflows.

    Determines the optimal conversion path and coordinates converters
    and quantizers to achieve the desired output format.
    """

    def __init__(self, intermediate_dir: Optional[Path] = None):
        """Initialize orchestrator.

        Args:
            intermediate_dir: Directory for intermediate files (saved for user)
        """
        # Save intermediate files in main quantizations folder, not .temp
        # Users can use these FP16 GGUF files with llama.cpp or Ollama
        self.intermediate_dir = intermediate_dir or Path("results/quantizations")
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Initialize converters
        self.gguf_dequantizer = GGUFDequantizer()
        self.gguf_to_hf_converter = GGUFToHFConverter()

        logger.debug(f"Initialized QuantizationOrchestrator (intermediate: {self.intermediate_dir})")

    def determine_conversion_path(
        self, model_info: ModelInfo, target_quant_type: QuantizationType
    ) -> Tuple[ConversionPath, QuantizationDirective]:
        """Determine the conversion path and quantization directive.

        Orchestrator is intelligent about what needs to happen:
        - FP16 is NOT a quantization, it's just dtype conversion (final output after conversion)
        - INT8/INT4/etc are real quantizations (require quantizer after conversion)
        - VLMs (Vision-Language Models) have special restrictions

        Args:
            model_info: Source model information
            target_quant_type: Target quantization type

        Returns:
            Tuple of (ConversionPath, QuantizationDirective) indicating workflow and what quantizer should do

        Raises:
            ValueError: If VLM is selected with unsupported quantization method
        """
        # Check if this is a VLM (Vision-Language Model)
        is_vlm = model_info.model_type == ModelType.VLM

        # Check source provider
        is_ollama = model_info.provider == ProviderType.OLLAMA
        is_gguf = model_info.provider == ProviderType.GGUF
        is_hf = model_info.provider == ProviderType.HUGGINGFACE

        # Check target quantization type and family
        target_family = target_quant_type.method_family

        # ========== VLM INCOMPATIBILITY CHECKS ==========
        # GGUF: VLMs cannot be converted to GGUF (llama.cpp doesn't support vision encoders)
        if is_vlm and target_family == "GGUF":
            raise ValueError(
                f"❌ GGUF quantization is NOT supported for Vision-Language Models (VLMs)\n\n"
                f"Why: llama.cpp doesn't support vision encoders (CLIP, ViT) or multimodal architectures.\n\n"
                f"Available alternatives for VLMs:\n"
                f"  • Generic FP16 (size: 50%, quality: perfect) - RECOMMENDED\n"
                f"  • Generic INT8 (size: 25%, CUDA required, component-level with language-only)\n"
                f"  • Generic INT4 (size: 12.5%, CUDA required, component-level with language-only)\n"
                f"  • BitsAndBytes (HuggingFace integration, CUDA required)\n"
                f"  • MLX INT4 (Apple Silicon, component-level support)\n"
            )

        # GPTQ: Not designed for VLMs
        if is_vlm and target_family == "GPTQ":
            raise ValueError(
                f"❌ GPTQ quantization is NOT supported for Vision-Language Models (VLMs)\n\n"
                f"Why: GPTQ is specifically designed for language-only models.\n\n"
                f"Available alternatives for VLMs:\n"
                f"  • Generic FP16 (RECOMMENDED) or INT8/INT4\n"
                f"  • BitsAndBytes (CUDA required)\n"
                f"  • MLX INT4 (Apple Silicon)\n"
            )

        # AWQ: Not designed for VLMs
        if is_vlm and target_family == "AWQ":
            raise ValueError(
                f"❌ AWQ quantization is NOT supported for Vision-Language Models (VLMs)\n\n"
                f"Why: AWQ is specifically designed for language-only models.\n\n"
                f"Available alternatives for VLMs:\n"
                f"  • Generic FP16 (RECOMMENDED) or INT8/INT4\n"
                f"  • BitsAndBytes (CUDA required)\n"
                f"  • MLX INT4 (Apple Silicon)\n"
            )

        # ========== STANDARD CONVERSION LOGIC ==========

        # CASE 1: GGUF/Ollama → GGUF quantization (direct requantization)
        # No conversion needed, quantizer handles it directly
        if (is_ollama or is_gguf) and target_family == "GGUF":
            return ConversionPath.DIRECT, QuantizationDirective.CONTINUE_GGUF

        # CASE 2: Ollama/GGUF → Generic quantization
        # Special handling: FP16 is NOT a quantization, INT8/INT4/etc are
        if is_ollama or is_gguf:
            if target_family == "Generic":
                # IMPORTANT: FP16 is just dtype conversion, not quantization
                # After GGUF → FP16 GGUF → FP16 Safetensors, we're DONE
                # For INT8/INT4/etc, we need to quantize after conversion
                if target_quant_type == QuantizationType.FP16:
                    return ConversionPath.GGUF_TO_GENERIC, QuantizationDirective.SKIP
                elif target_quant_type == QuantizationType.INT8:
                    return ConversionPath.GGUF_TO_GENERIC, QuantizationDirective.QUANTIZE_INT8
                elif target_quant_type == QuantizationType.INT4:
                    return ConversionPath.GGUF_TO_GENERIC, QuantizationDirective.QUANTIZE_INT4
                elif target_quant_type == QuantizationType.INT6:
                    return ConversionPath.GGUF_TO_GENERIC, QuantizationDirective.QUANTIZE
                elif target_quant_type == QuantizationType.INT3:
                    return ConversionPath.GGUF_TO_GENERIC, QuantizationDirective.QUANTIZE
                elif target_quant_type == QuantizationType.INT2:
                    return ConversionPath.GGUF_TO_GENERIC, QuantizationDirective.QUANTIZE
                else:
                    return ConversionPath.GGUF_TO_GENERIC, QuantizationDirective.QUANTIZE

            elif target_family == "MLX":
                # MLX supports VLMs (experimental)
                return ConversionPath.GGUF_TO_MLX, QuantizationDirective.CONTINUE_MLX

            elif target_family == "OpenVINO":
                # OpenVINO supports VLMs (experimental)
                return ConversionPath.GGUF_TO_OPENVINO, QuantizationDirective.CONTINUE_OPENVINO

            elif target_family in ["GPTQ", "AWQ", "BitsAndBytes"]:
                return ConversionPath.GGUF_TO_ADVANCED, QuantizationDirective.CONTINUE_ADVANCED

            # Fallback for other families
            return ConversionPath.GGUF_TO_GENERIC, QuantizationDirective.QUANTIZE

        # CASE 3: HuggingFace → Any (already supported, direct)
        # No conversion needed, quantizer handles it directly
        if is_hf:
            return ConversionPath.DIRECT, QuantizationDirective.QUANTIZE

        # Default: direct quantization
        return ConversionPath.DIRECT, QuantizationDirective.QUANTIZE

    def needs_conversion(self, conversion_path: ConversionPath) -> bool:
        """Check if conversion is needed before quantization.

        Args:
            conversion_path: Conversion path

        Returns:
            True if intermediate conversion steps are required
        """
        return conversion_path != ConversionPath.DIRECT

    def create_intermediate_path(self, task: QuantizationTask, suffix: str) -> Path:
        """Create path for intermediate conversion file.

        Args:
            task: Quantization task
            suffix: File suffix (e.g., "_fp16", "_hf")

        Returns:
            Path for intermediate file in results/quantizations/
        """
        base_name = task.output_path.stem

        # For GGUF intermediate files, always add .gguf extension
        # This is needed so the Generic/MLX/OpenVINO quantizers can detect GGUF files
        # Save in main quantizations folder so users can easily find and use them
        if suffix == "_fp16":  # FP16 GGUF intermediate
            intermediate = self.intermediate_dir / f"{base_name}{suffix}.gguf"
        elif task.output_path.suffix == ".gguf":
            # It's a GGUF file output, create intermediate GGUF file
            intermediate = self.intermediate_dir / f"{base_name}{suffix}.gguf"
        else:
            # It's a directory output (HF/MLX/OpenVINO), intermediate is still GGUF
            intermediate = self.intermediate_dir / f"{base_name}{suffix}.gguf"

        return intermediate

    def execute_conversion_pipeline(
        self,
        task: QuantizationTask,
        conversion_path: ConversionPath,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> Optional[Path]:
        """Execute the conversion pipeline to prepare model for quantization.

        Args:
            task: Quantization task
            conversion_path: Conversion path to execute
            progress_callback: Progress callback

        Returns:
            Path to converted model ready for quantization, or None if failed
        """
        source_path = task.model_info.source_path
        if not source_path:
            logger.error(f"No source path for model {task.model_info.name}")
            return None

        logger.info(
            f"Executing conversion pipeline: {conversion_path.value} "
            f"for {task.model_info.name}"
        )

        try:
            # DIRECT: No conversion needed
            if conversion_path == ConversionPath.DIRECT:
                logger.debug("Direct quantization, no conversion needed")
                return source_path

            # GGUF_TO_GENERIC: Dequantize + Convert to HF format
            if conversion_path == ConversionPath.GGUF_TO_GENERIC:
                logger.info("Step 1: Dequantizing GGUF to FP16")

                # Create intermediate FP16 GGUF file
                fp16_path = self.create_intermediate_path(task, "_fp16")

                # Dequantize
                success = self.gguf_dequantizer.convert(
                    source_path, fp16_path, progress_callback
                )

                if not success:
                    logger.error("Dequantization failed")
                    return None

                logger.info(f"Dequantization completed: {fp16_path}")

                # Step 2: Convert GGUF FP16 → HuggingFace format
                logger.info("Step 2: Converting FP16 GGUF to HuggingFace format")

                # Create HF output directory
                base_name = task.output_path.stem
                hf_path = self.intermediate_dir / f"{base_name}_hf"

                # Convert to HF
                success = self.gguf_to_hf_converter.convert(
                    fp16_path, hf_path, progress_callback
                )

                if not success:
                    logger.error("GGUF→HF conversion failed")
                    logger.info(f"FP16 GGUF file saved at: {fp16_path}")
                    logger.info("You can use this FP16 GGUF with llama.cpp or Ollama")
                    return None

                logger.info(f"HuggingFace conversion completed: {hf_path}")
                return hf_path

            # GGUF_TO_MLX/OPENVINO: Dequantize GGUF to FP16 (direct GGUF loading)
            if conversion_path in [
                ConversionPath.GGUF_TO_MLX,
                ConversionPath.GGUF_TO_OPENVINO,
            ]:
                logger.info("Step 1: Dequantizing GGUF to FP16")
                logger.info(
                    "Note: GGUF FP16 will be loaded by mlx-lm/openvino for quantization"
                )

                # Create intermediate FP16 GGUF file
                fp16_path = self.create_intermediate_path(task, "_fp16")

                # Dequantize
                success = self.gguf_dequantizer.convert(
                    source_path, fp16_path, progress_callback
                )

                if not success:
                    logger.error("Dequantization failed")
                    return None

                logger.info(f"Dequantization completed: {fp16_path}")

                # For MLX/OpenVINO, they will handle loading GGUF FP16 directly
                # - MLX: Uses mlx-lm which can load GGUF
                # - OpenVINO: Uses optimum-intel with transformers backend
                return fp16_path

            # GGUF_TO_ADVANCED: Dequantize + Convert to HF for GPTQ/AWQ/BnB
            if conversion_path == ConversionPath.GGUF_TO_ADVANCED:
                logger.warning(
                    "Advanced quantization (GPTQ/AWQ/BnB) from GGUF involves "
                    "quality loss. Consider using source HuggingFace model if available."
                )

                # Step 1: Dequantize to FP16
                logger.info("Step 1: Dequantizing GGUF to FP16")
                fp16_path = self.create_intermediate_path(task, "_fp16")
                success = self.gguf_dequantizer.convert(
                    source_path, fp16_path, progress_callback
                )

                if not success:
                    logger.error("Dequantization failed")
                    return None

                logger.info(f"Dequantization completed: {fp16_path}")

                # Step 2: Convert GGUF FP16 → HF format
                logger.info("Step 2: Converting FP16 GGUF to HuggingFace format")

                # Create HF output directory
                base_name = task.output_path.stem
                hf_path = self.intermediate_dir / f"{base_name}_hf"

                # Convert to HF
                success = self.gguf_to_hf_converter.convert(
                    fp16_path, hf_path, progress_callback
                )

                if not success:
                    logger.error("GGUF→HF conversion failed")
                    logger.info(f"FP16 GGUF file saved at: {fp16_path}")
                    logger.info("You can use this FP16 GGUF with llama.cpp or Ollama")
                    return None

                logger.info(f"HuggingFace conversion completed: {hf_path}")
                return hf_path

            logger.error(f"Unknown conversion path: {conversion_path}")
            return None

        except Exception as e:
            logger.error(f"Conversion pipeline failed: {e}", exc_info=True)
            return None

    def cleanup_intermediate_files(self, task: QuantizationTask) -> None:
        """Clean up intermediate conversion files.

        Note: This method is no longer used. Intermediate files are preserved
        for users to use with llama.cpp or Ollama.

        Args:
            task: Quantization task
        """
        try:
            # Clean up intermediate directory for this task
            task_temp_pattern = task.output_path.stem + "*"
            for temp_file in self.intermediate_dir.glob(task_temp_pattern):
                if temp_file.is_file():
                    temp_file.unlink()
                    logger.debug(f"Removed intermediate file: {temp_file}")
                elif temp_file.is_dir():
                    import shutil
                    shutil.rmtree(temp_file)
                    logger.debug(f"Removed intermediate directory: {temp_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup intermediate files: {e}")
