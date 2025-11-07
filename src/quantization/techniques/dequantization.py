"""Dequantization quantizer - Convert quantized models to full precision.

This "quantizer" actually does the opposite: it takes quantized GGUF models
(Q4_K_M, Q8_0, Q5_K, etc.) and converts them to full precision formats:
- FP16 GGUF (for llama.cpp/Ollama)
- FP16 HuggingFace safetensors
- FP32 GGUF
- FP32 HuggingFace safetensors

This is useful for:
1. Converting Ollama models to other formats
2. Recovering full quality from quantized models
3. Preparing models for advanced quantization (GPTQ, AWQ, etc.)
"""

from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ...models.model import ModelInfo
from ...models.provider import ProviderType
from ..models import QuantizationTask, QuantizationType, TaskStatus
from .base import BaseQuantizer
from ..converters.gguf_dequantizer import GGUFDequantizer
from ..converters.gguf_to_hf import GGUFToHFConverter


class DequantizationQuantizer(BaseQuantizer):
    """Dequantization quantizer - converts quantized models to full precision.

    Supports any GGUF quantization format and outputs to:
    - FP16/FP32 GGUF (for llama.cpp/Ollama)
    - FP16/FP32 HuggingFace safetensors
    """

    def __init__(self):
        """Initialize dequantizer."""
        self.gguf_dequantizer = GGUFDequantizer()
        self.gguf_to_hf_converter = GGUFToHFConverter()

    def check_availability(self) -> tuple[bool, str]:
        """Check if dequantization tools are available.

        Returns:
            (available, message)
        """
        try:
            import torch
            import numpy as np

            return True, "Dequantization tools available (torch, numpy)"
        except ImportError as e:
            return False, f"Missing dependency: {e}"

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported dequantization types.

        Returns:
            List of supported types
        """
        return [
            QuantizationType.DEQUANT_FP16_GGUF,
            QuantizationType.DEQUANT_FP16_HF,
            QuantizationType.DEQUANT_FP32_GGUF,
            QuantizationType.DEQUANT_FP32_HF,
        ]

    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for dequantization.

        Supports:
        - GGUF files (local .gguf files)
        - Ollama models (registered with Ollama, managed by Ollama library)
        - HuggingFace models (local or remote, identified by model_id)

        Args:
            model_info: Model to dequantize

        Returns:
            Path for GGUF files, model_id string for Ollama/HF, or None if not compatible
        """
        from ...models.provider import ProviderType

        # Support GGUF, Ollama, and HuggingFace models
        if model_info.provider == ProviderType.GGUF:
            # GGUF file: check if local file exists
            if model_info.source_path and model_info.source_path.exists():
                if model_info.source_path.suffix.lower() == ".gguf":
                    return model_info.source_path
            return None

        elif model_info.provider == ProviderType.OLLAMA:
            # Ollama model: check if model_id is set (managed by Ollama)
            if model_info.model_id:
                # Return model_id as identifier (Ollama will manage the actual model)
                return Path(model_info.model_id)
            return None

        elif model_info.provider == ProviderType.HUGGINGFACE:
            # HuggingFace model: check if model_id is set
            if model_info.model_id:
                # Return model_id as identifier (transformers will handle loading)
                return Path(model_info.model_id)
            return None

        return None

    def estimate_output_size(
        self, model_info: ModelInfo, quant_type: QuantizationType
    ) -> float:
        """Estimate output file size in GB.

        Args:
            model_info: Source model
            quant_type: Dequantization type

        Returns:
            Estimated size in GB
        """
        # Dequantized models are larger than quantized originals
        original_size = model_info.size_gb

        size_factors = {
            # FP16 is roughly 2x the quantized size (depends on original quant)
            QuantizationType.DEQUANT_FP16_GGUF: 2.0,
            QuantizationType.DEQUANT_FP16_HF: 2.0,
            # FP32 is roughly 4x (depends on original quant)
            QuantizationType.DEQUANT_FP32_GGUF: 4.0,
            QuantizationType.DEQUANT_FP32_HF: 4.0,
        }

        factor = size_factors.get(quant_type, 2.0)
        return original_size * factor

    def dequantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Execute dequantization/conversion to full precision.

        Supports converting:
        - Quantized GGUF → Full precision GGUF/HF
        - Ollama models → Full precision GGUF/HF
        - HuggingFace models → Full precision GGUF/HF

        Args:
            task: Dequantization task
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful
        """
        try:
            task.status = TaskStatus.RUNNING
            logger.info(f"Starting conversion: {task.model_info.name} → {task.quant_type.display_name}")

            if progress_callback:
                progress_callback(5.0, None)
            task.progress = 5.0

            # Get source model path/identifier
            source_path = self.get_source_model_path(task.model_info)
            if not source_path:
                raise ValueError(f"Source model not found for {task.model_info.name}")

            logger.info(f"Source model: {source_path}")
            logger.info(f"Output format: {task.quant_type.display_name}")
            logger.info(f"Model provider: {task.model_info.provider.value}")

            # Check model provider type
            from ...models.provider import ProviderType
            is_gguf_file = task.model_info.provider == ProviderType.GGUF
            is_ollama = task.model_info.provider == ProviderType.OLLAMA
            is_hf = task.model_info.provider == ProviderType.HUGGINGFACE

            # Determine output format
            is_hf_output = "_hf" in task.quant_type.value or "hf" in task.quant_type.value
            is_fp32 = "fp32" in task.quant_type.value
            is_gguf_output = "gguf" in task.quant_type.value

            # For HF outputs, need two steps: dequant to FP16/FP32 GGUF, then convert to HF
            # For GGUF outputs, just dequantize to target precision
            precision = "f32" if is_fp32 else "f16"

            # Special handling for HuggingFace models
            if is_hf:
                logger.info(f"Handling HuggingFace model: {task.model_info.model_id}")
                # For HF models, use Generic quantizer to preserve tokenizer/config
                # HF models are already in FP32, just need format conversion
                from .generic import GenericQuantizer
                generic_quantizer = GenericQuantizer()

                # Create a modified task that uses Generic FP16/FP32
                from ..models import QuantizationType as QT
                if is_hf_output:
                    # HF→HF: Use generic quantizer to save in target precision
                    generic_quant_type = QT.FP32 if is_fp32 else QT.FP16
                    logger.info(f"Converting HF model to {generic_quant_type.display_name}")
                    success = generic_quantizer.quantize(task, progress_callback)
                    if success:
                        task.status = TaskStatus.COMPLETED
                    return success
                else:
                    # HF→GGUF: Need to convert to GGUF format
                    # This requires HF→GGUF conversion (not currently in dequantization)
                    logger.warning("HF→GGUF conversion not yet supported in dequantization. Use GGUF conversion method instead.")
                    raise ValueError("HF→GGUF conversion requires separate GGUF conversion workflow")

            # For GGUF files and Ollama models, use dequantization pipeline
            if is_gguf_output:
                # Simple case: GGUF → FP16/FP32 GGUF
                logger.info(f"Step 1: Dequantizing GGUF to {precision.upper()}")

                if progress_callback:
                    progress_callback(30.0, None)
                task.progress = 30.0

                # Create output directory
                task.output_path.parent.mkdir(parents=True, exist_ok=True)

                # Dequantize to target precision
                success = self.gguf_dequantizer.convert(
                    source_path,
                    task.output_path,
                    progress_callback,
                    target_precision=precision
                )

                if not success:
                    logger.error("Dequantization failed")
                    return False

                if progress_callback:
                    progress_callback(100.0, None)
                task.progress = 100.0

                logger.info(f"Dequantization completed: {task.output_path}")

            else:
                # Complex case: GGUF → FP16/FP32 GGUF → HuggingFace
                logger.info(f"Step 1: Dequantizing GGUF to {precision.upper()}")

                if progress_callback:
                    progress_callback(20.0, None)
                task.progress = 20.0

                # Create intermediate FP16/FP32 GGUF
                intermediate_gguf = task.output_path.parent / f"{task.output_path.stem}_fp_intermediate.gguf"
                intermediate_gguf.parent.mkdir(parents=True, exist_ok=True)

                # Dequantize to intermediate GGUF
                success = self.gguf_dequantizer.convert(
                    source_path,
                    intermediate_gguf,
                    None,  # No callback for intermediate step
                    target_precision=precision
                )

                if not success:
                    logger.error("Dequantization to intermediate GGUF failed")
                    return False

                logger.info(f"Intermediate GGUF created: {intermediate_gguf}")

                if progress_callback:
                    progress_callback(50.0, None)
                task.progress = 50.0

                # Step 2: Convert to HuggingFace format
                logger.info(f"Step 2: Converting {precision.upper()} GGUF to HuggingFace")

                success = self.gguf_to_hf_converter.convert(
                    intermediate_gguf,
                    task.output_path,
                    progress_callback
                )

                if not success:
                    logger.error("Conversion to HuggingFace format failed")
                    logger.info(f"Intermediate GGUF saved at: {intermediate_gguf}")
                    return False

                # Clean up intermediate file
                try:
                    intermediate_gguf.unlink()
                    logger.debug(f"Removed intermediate GGUF: {intermediate_gguf}")
                except Exception as e:
                    logger.warning(f"Could not remove intermediate GGUF: {e}")

                if progress_callback:
                    progress_callback(100.0, None)
                task.progress = 100.0

                logger.info(f"Dequantization completed: {task.output_path}")

            task.status = TaskStatus.COMPLETED
            return True

        except Exception as e:
            error_msg = f"Dequantization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = error_msg
            return False

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Execute dequantization (quantize method interface).

        Args:
            task: Dequantization task
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful
        """
        return self.dequantize(task, progress_callback)
