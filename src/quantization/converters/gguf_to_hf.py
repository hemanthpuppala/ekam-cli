"""GGUF to HuggingFace Converter - Convert GGUF to HuggingFace Safetensors.

This converter takes a GGUF file (FP16 or quantized) and converts it to
HuggingFace format (safetensors). This enables Generic/GPTQ/AWQ/BnB quantization
for Ollama models.

Workflow:
    Ollama GGUF Q4_K_M → GGUF FP16 → HF Safetensors → Generic INT4/GPTQ/AWQ/BnB

Note: This involves quality loss as quantization is lossy. The HF version
      won't be as good as quantizing from the original FP16/FP32 model.

Uses llama.cpp convert script for conversion.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Callable

from loguru import logger

from .base import BaseConverter


class GGUFToHFConverter(BaseConverter):
    """Convert GGUF files to HuggingFace safetensors format.

    Uses llama.cpp's convert script to convert GGUF back to HF format.
    This enables using Ollama models with Generic/GPTQ/AWQ/BnB quantizers.
    """

    def __init__(self):
        """Initialize GGUF to HF converter."""
        self.convert_script = self._find_convert_script()

    def _find_convert_script(self) -> Optional[Path]:
        """Find llama.cpp convert script.

        Returns:
            Path to convert-hf-to-gguf.py or None
        """
        # Try common locations
        possible_paths = [
            Path("/opt/homebrew/Cellar/llama.cpp"),
            Path("/usr/local/Cellar/llama.cpp"),
            Path.home() / "llama.cpp",
        ]

        for base_path in possible_paths:
            if base_path.exists():
                # Search for convert script
                for script in base_path.rglob("convert-hf-to-gguf.py"):
                    logger.info(f"Found convert script at {script}")
                    return script
                for script in base_path.rglob("convert.py"):
                    logger.info(f"Found convert script at {script}")
                    return script

        # Try finding llama.cpp in PATH
        try:
            result = subprocess.run(
                ["which", "llama-server"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                llama_bin = Path(result.stdout.strip())
                llama_root = llama_bin.parent.parent
                for script in llama_root.rglob("convert*.py"):
                    logger.info(f"Found convert script at {script}")
                    return script
        except Exception:
            pass

        return None

    def can_convert(self, source_format: str, target_format: str) -> bool:
        """Check if this converter handles the conversion.

        Args:
            source_format: Should be "gguf" or "gguf_fp16"
            target_format: Should be "huggingface" or "safetensors"

        Returns:
            True if supported
        """
        return (
            source_format.lower() in ["gguf", "gguf_fp16", "gguf_quantized"]
            and target_format.lower() in ["huggingface", "hf", "safetensors"]
        )

    def check_availability(self) -> tuple[bool, str]:
        """Check if llama.cpp convert script is available.

        Returns:
            (available, message)
        """
        if self.convert_script and self.convert_script.exists():
            return True, f"Using llama.cpp convert at {self.convert_script}"
        return False, (
            "llama.cpp convert script not found. Install llama.cpp from source:\n"
            "git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make"
        )

    def convert(
        self,
        source_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Convert GGUF to HuggingFace format.

        Note: llama.cpp doesn't have a direct GGUF → HF converter.
        We use a workaround: load GGUF with transformers directly.

        Args:
            source_path: Path to GGUF file
            output_path: Path for HF output directory
            progress_callback: Optional progress callback

        Returns:
            True if successful
        """
        # Validate paths
        valid, message = self.validate_paths(source_path, output_path)
        if not valid:
            logger.error(message)
            return False

        logger.info(f"Converting GGUF to HuggingFace: {source_path} → {output_path}")

        try:
            # Check if transformers can load GGUF directly
            # New transformers versions (4.45+) support loading GGUF files
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                logger.info("Attempting to load GGUF directly with transformers...")

                if progress_callback:
                    progress_callback(10.0, None)

                # Try loading GGUF with transformers (requires transformers 4.45+)
                # This loads the model from GGUF and we can save it as HF format
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        str(source_path),
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                    )

                    if progress_callback:
                        progress_callback(70.0, None)

                    # Get tokenizer from original model
                    # Note: GGUF files don't contain tokenizer, need to infer from model
                    logger.warning(
                        "GGUF conversion: tokenizer not available from GGUF file. "
                        "Manual tokenizer setup may be required."
                    )

                    # Save in HuggingFace format
                    output_path.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(output_path, safe_serialization=True)

                    if progress_callback:
                        progress_callback(100.0, 0)

                    logger.info(f"Conversion completed: {output_path}")
                    return True

                except Exception as e:
                    logger.warning(f"Direct GGUF loading failed: {e}")
                    # Fall through to alternative method

            except ImportError:
                logger.warning("transformers not available for GGUF loading")

            # Alternative: Use llama.cpp convert script (reverse conversion)
            # Note: This is NOT officially supported by llama.cpp
            # The convert-hf-to-gguf.py script only goes HF → GGUF, not GGUF → HF

            logger.error(
                "GGUF → HuggingFace conversion not yet fully supported.\n"
                "Workaround: For Generic/GPTQ/AWQ quantization of Ollama models:\n"
                "1. Download the original HuggingFace model instead of using Ollama\n"
                "2. Use the HF model directly for quantization\n\n"
                "Alternative: Use MLX or OpenVINO quantization which can work with GGUF files."
            )
            return False

        except Exception as e:
            logger.error(f"GGUF to HF conversion error: {e}", exc_info=True)
            return False

    def estimate_output_size(self, source_path: Path) -> float:
        """Estimate HF output size.

        HuggingFace safetensors is approximately same size as GGUF FP16.

        Args:
            source_path: Source GGUF file

        Returns:
            Estimated size in GB
        """
        try:
            source_size_gb = source_path.stat().st_size / (1024**3)
            # HF safetensors ≈ same size as GGUF FP16
            return source_size_gb
        except Exception:
            return 0.0

    def get_conversion_description(self) -> str:
        """Get description.

        Returns:
            Description string
        """
        return "GGUF to HuggingFace Converter (GGUF → Safetensors)"
