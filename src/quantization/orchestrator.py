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
from typing import Optional, Callable
from enum import Enum

from loguru import logger

from ..models.model import ModelInfo
from ..models.provider import ProviderType
from .models import QuantizationTask, QuantizationType, TaskStatus
from .converters.gguf_dequantizer import GGUFDequantizer


class ConversionPath(Enum):
    """Conversion paths for different quantization scenarios."""
    DIRECT = "direct"  # No conversion needed
    DEQUANT_ONLY = "dequant_only"  # Just dequantize GGUF
    GGUF_TO_GENERIC = "gguf_to_generic"  # GGUF → FP16 → Generic
    GGUF_TO_MLX = "gguf_to_mlx"  # GGUF → FP16 → MLX
    GGUF_TO_OPENVINO = "gguf_to_openvino"  # GGUF → FP16 → OpenVINO
    GGUF_TO_ADVANCED = "gguf_to_advanced"  # GGUF → FP16 → HF → GPTQ/AWQ/BnB


class QuantizationOrchestrator:
    """Orchestrates multi-step quantization workflows.

    Determines the optimal conversion path and coordinates converters
    and quantizers to achieve the desired output format.
    """

    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize orchestrator.

        Args:
            temp_dir: Directory for intermediate files
        """
        self.temp_dir = temp_dir or Path("results/quantizations/.temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize converters
        self.gguf_dequantizer = GGUFDequantizer()

        logger.debug(f"Initialized QuantizationOrchestrator (temp: {self.temp_dir})")

    def determine_conversion_path(
        self, model_info: ModelInfo, target_quant_type: QuantizationType
    ) -> ConversionPath:
        """Determine the conversion path needed for quantization.

        Args:
            model_info: Source model information
            target_quant_type: Target quantization type

        Returns:
            ConversionPath enum indicating the required workflow
        """
        # Check source provider
        is_ollama = model_info.provider == ProviderType.OLLAMA
        is_gguf = model_info.provider == ProviderType.GGUF
        is_hf = model_info.provider == ProviderType.HUGGINGFACE

        # Check target quantization family
        target_family = target_quant_type.method_family

        # CASE 1: GGUF/Ollama → GGUF quantization (direct requantization)
        if (is_ollama or is_gguf) and target_family == "GGUF":
            return ConversionPath.DIRECT

        # CASE 2: Ollama/GGUF → Generic/MLX/OpenVINO/Advanced
        # Needs dequantization + format conversion
        if is_ollama or is_gguf:
            if target_family == "Generic":
                return ConversionPath.GGUF_TO_GENERIC
            elif target_family == "MLX":
                return ConversionPath.GGUF_TO_MLX
            elif target_family == "OpenVINO":
                return ConversionPath.GGUF_TO_OPENVINO
            elif target_family in ["GPTQ", "AWQ", "BitsAndBytes"]:
                return ConversionPath.GGUF_TO_ADVANCED

        # CASE 3: HuggingFace → Any (already supported, direct)
        if is_hf:
            return ConversionPath.DIRECT

        # Default: direct quantization
        return ConversionPath.DIRECT

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
            suffix: File suffix (e.g., "_fp16.gguf", "_hf")

        Returns:
            Path for intermediate file
        """
        base_name = task.output_path.stem
        if task.output_path.suffix == ".gguf":
            # It's a file output, create intermediate file
            intermediate = self.temp_dir / f"{base_name}{suffix}.gguf"
        else:
            # It's a directory output, create intermediate directory
            intermediate = self.temp_dir / f"{base_name}{suffix}"

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

            # GGUF_TO_GENERIC/MLX/OPENVINO: Dequantize GGUF first
            if conversion_path in [
                ConversionPath.GGUF_TO_GENERIC,
                ConversionPath.GGUF_TO_MLX,
                ConversionPath.GGUF_TO_OPENVINO,
            ]:
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

                # For Generic/MLX/OpenVINO, the respective quantizers will handle
                # loading the GGUF FP16 file with transformers/mlx-lm
                return fp16_path

            # GGUF_TO_ADVANCED: Need additional HF conversion
            if conversion_path == ConversionPath.GGUF_TO_ADVANCED:
                logger.warning(
                    "Advanced quantization (GPTQ/AWQ/BnB) from GGUF involves "
                    "quality loss. Consider using source HuggingFace model if available."
                )

                # Step 1: Dequantize to FP16
                fp16_path = self.create_intermediate_path(task, "_fp16")
                success = self.gguf_dequantizer.convert(
                    source_path, fp16_path, progress_callback
                )

                if not success:
                    return None

                # Step 2: Convert GGUF FP16 → HF format
                # This would need a GGUFToHFConverter (not implemented yet)
                # For now, return fp16_path and let quantizer handle it
                logger.warning(
                    "GGUF→HF conversion not yet implemented. "
                    "Advanced quantization may not work."
                )
                return fp16_path

            logger.error(f"Unknown conversion path: {conversion_path}")
            return None

        except Exception as e:
            logger.error(f"Conversion pipeline failed: {e}", exc_info=True)
            return None

    def cleanup_intermediate_files(self, task: QuantizationTask) -> None:
        """Clean up intermediate conversion files.

        Args:
            task: Quantization task
        """
        try:
            # Clean up temp directory for this task
            task_temp_pattern = task.output_path.stem + "*"
            for temp_file in self.temp_dir.glob(task_temp_pattern):
                if temp_file.is_file():
                    temp_file.unlink()
                    logger.debug(f"Removed intermediate file: {temp_file}")
                elif temp_file.is_dir():
                    import shutil
                    shutil.rmtree(temp_file)
                    logger.debug(f"Removed intermediate directory: {temp_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup intermediate files: {e}")
