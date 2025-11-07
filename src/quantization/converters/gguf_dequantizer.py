"""GGUF Dequantizer - Convert quantized GGUF to FP16 GGUF.

This converter takes a quantized GGUF file (Q4_K_M, Q8_0, etc.) and converts
it back to FP16 GGUF format. This is useful as an intermediate step when
converting Ollama models to other formats like Generic, MLX, or OpenVINO.

Workflow:
    Ollama GGUF Q4_K_M → GGUF FP16 → Load with transformers → Generic INT4

Note: This involves quality loss as quantization is lossy. The FP16 version
      won't be as good as the original FP16 model.
"""

import subprocess
from pathlib import Path
from typing import Optional, Callable

from loguru import logger

from .base import BaseConverter


class GGUFDequantizer(BaseConverter):
    """Convert quantized GGUF files to FP16 GGUF.

    Uses llama-quantize with special parameters to dequantize to FP16.
    """

    def __init__(self):
        """Initialize GGUF dequantizer."""
        self.quantize_binary = self._find_llama_quantize()

    def _find_llama_quantize(self) -> Optional[Path]:
        """Find llama-quantize binary.

        Returns:
            Path to llama-quantize or None
        """
        possible_paths = [
            Path("/opt/homebrew/bin/llama-quantize"),
            Path("/usr/local/bin/llama-quantize"),
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Found llama-quantize at {path}")
                return path

        # Try which command
        try:
            result = subprocess.run(
                ["which", "llama-quantize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                path = Path(result.stdout.strip())
                logger.info(f"Found llama-quantize in PATH: {path}")
                return path
        except Exception:
            pass

        return None

    def can_convert(self, source_format: str, target_format: str) -> bool:
        """Check if this converter handles the conversion.

        Args:
            source_format: Should be "gguf_quantized"
            target_format: Should be "gguf_fp16"

        Returns:
            True if supported
        """
        return (
            source_format.lower() in ["gguf_quantized", "gguf"]
            and target_format.lower() in ["gguf_fp16", "fp16"]
        )

    def check_availability(self) -> tuple[bool, str]:
        """Check if llama-quantize is available.

        Returns:
            (available, message)
        """
        if self.quantize_binary and self.quantize_binary.exists():
            return True, f"Using llama-quantize at {self.quantize_binary}"
        return False, "llama-quantize not found. Install llama.cpp"

    def convert(
        self,
        source_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
        target_precision: str = "f16",
    ) -> bool:
        """Dequantize GGUF to full precision (FP16 or FP32).

        Note: llama.cpp doesn't have a direct dequantize command.
        We use F16/F32 quantization types which effectively means "no quantization".

        Args:
            source_path: Path to quantized GGUF file
            output_path: Path for dequantized GGUF output
            progress_callback: Optional progress callback
            target_precision: Target precision ("f16" or "f32"). Defaults to "f16".

        Returns:
            True if successful
        """
        # Validate paths
        valid, message = self.validate_paths(source_path, output_path)
        if not valid:
            logger.error(message)
            return False

        # Validate target precision
        target_precision = target_precision.lower()
        if target_precision not in ["f16", "f32"]:
            logger.error(f"Invalid target precision: {target_precision}. Use 'f16' or 'f32'")
            return False

        precision_name = "FP32" if target_precision == "f32" else "FP16"
        precision_type = "F32" if target_precision == "f32" else "F16"

        logger.info(f"Dequantizing {source_path} to {precision_name}: {output_path}")

        try:
            # Use F16/F32 type for dequantization
            # Need --allow-requantize because source is already quantized
            cmd = [
                str(self.quantize_binary),
                "--allow-requantize",  # Required for requantizing from quantized models
                str(source_path),
                str(output_path),
                precision_type,  # FP16 or FP32 quantization (effectively dequantization)
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Monitor progress
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                logger.debug(f"Dequantize: {line.strip()}")

                # Call progress callback if provided
                if progress_callback:
                    # Simple progress estimation
                    progress_callback(50.0, None)

            return_code = process.wait()

            if return_code == 0:
                logger.info(f"Dequantization completed: {output_path}")
                if progress_callback:
                    progress_callback(100.0, 0)
                return True
            else:
                logger.error(f"Dequantization failed with code {return_code}")
                return False

        except Exception as e:
            logger.error(f"Dequantization error: {e}", exc_info=True)
            return False

    def estimate_output_size(self, source_path: Path, target_precision: str = "f16") -> float:
        """Estimate dequantized output size.

        FP16 is approximately 2x the size of Q4 and 1.8x the size of Q8.
        FP32 is approximately 4x the size of Q4.

        Args:
            source_path: Source GGUF file
            target_precision: Target precision ("f16" or "f32")

        Returns:
            Estimated size in GB
        """
        try:
            source_size_gb = source_path.stat().st_size / (1024**3)
            # Rough estimate based on target precision
            factor = 4.0 if target_precision.lower() == "f32" else 2.0
            return source_size_gb * factor
        except Exception:
            return 0.0

    def get_conversion_description(self) -> str:
        """Get description.

        Returns:
            Description string
        """
        return "GGUF Dequantizer (Quantized → FP16/FP32)"
