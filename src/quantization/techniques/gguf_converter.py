"""GGUF Converter - Convert any model to GGUF format.

Converts HuggingFace models to GGUF format using llama.cpp's convert.py.
This enables quantization of any model to GGUF Q4/Q5/Q6/Q8 formats.

References:
    - https://github.com/ggerganov/llama.cpp/blob/master/convert.py
"""

import os
import subprocess
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ...models.model import ModelInfo
from ...models.provider import ProviderType
from ..models import QuantizationTask, QuantizationType, TaskStatus
from .base import BaseQuantizer


class GGUFConverter(BaseQuantizer):
    """Convert models to GGUF format.

    Converts HuggingFace models to GGUF FP16 format,
    which can then be quantized to Q4/Q5/Q6/Q8.
    """

    def __init__(self):
        """Initialize GGUF converter."""
        super().__init__()
        self.convert_script = self._find_convert_script()

    def _find_convert_script(self) -> Optional[Path]:
        """Find llama.cpp convert.py script.

        Returns:
            Path to convert.py or None if not found
        """
        # Common locations for llama.cpp convert.py
        possible_locations = [
            # If llama-cpp-python is installed
            Path.home() / ".cache" / "llama.cpp" / "convert.py",
            # If llama.cpp repo is cloned
            Path("llama.cpp") / "convert.py",
            Path.home() / "llama.cpp" / "convert.py",
            # System-wide
            Path("/usr/local/share/llama.cpp/convert.py"),
        ]

        for location in possible_locations:
            if location.exists():
                logger.info(f"Found convert.py at: {location}")
                return location

        logger.warning("convert.py not found. Will attempt to download from llama.cpp repo")
        return None

    def check_availability(self) -> tuple[bool, str]:
        """Check if GGUF conversion is available.

        Returns:
            (available, message)
        """
        # Check for llama-cpp-python
        try:
            import llama_cpp
            llama_available = True
        except ImportError:
            return False, "llama-cpp-python not installed. Run: pip install llama-cpp-python"

        # Check for transformers
        try:
            import transformers
        except ImportError:
            return False, "transformers not installed. Run: pip install transformers"

        # Check for convert script
        if self.convert_script is None:
            return False, "llama.cpp convert.py not found. Clone llama.cpp repo or use --download option"

        return True, "GGUF conversion available (llama.cpp)"

    def get_supported_types(self) -> list[QuantizationType]:
        """Get supported quantization types.

        GGUF converter produces FP16 GGUF files.

        Returns:
            List of supported types
        """
        # FP16 GGUF (no additional quantization)
        return [QuantizationType.FP16]

    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for conversion.

        Works with HuggingFace models.

        Args:
            model_info: Model to convert

        Returns:
            Path to model directory, or None if not compatible
        """
        # GGUF converter works with HuggingFace models
        if model_info.provider != ProviderType.HUGGINGFACE:
            return None

        if model_info.file_path:
            return Path(model_info.file_path)

        return None

    def estimate_output_size(
        self, model_info: ModelInfo, quant_type: QuantizationType
    ) -> float:
        """Estimate output file size in GB.

        FP16 GGUF is approximately same size as original FP16 model.

        Args:
            model_info: Source model
            quant_type: Quantization type

        Returns:
            Estimated size in GB
        """
        # FP16 GGUF is ~50% of original FP32
        return model_info.size_gb * 0.5

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Convert model to GGUF format.

        Args:
            task: Conversion task
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful
        """
        try:
            task.status = TaskStatus.RUNNING
            logger.info(f"Starting GGUF conversion for {task.model_info.name}")

            # Get source model path
            source_path = task.model_info.file_path
            if not source_path or not Path(source_path).exists():
                raise ValueError(f"Source model not found: {source_path}")

            # Update progress: Preparing
            if progress_callback:
                progress_callback(10.0, None)
            task.progress = 10.0

            # Ensure convert script exists
            if not self.convert_script or not self.convert_script.exists():
                # Try to download convert.py
                self.convert_script = self._download_convert_script()
                if not self.convert_script:
                    raise RuntimeError("Cannot find or download convert.py from llama.cpp")

            # Ensure output directory exists
            task.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Update progress: Converting
            if progress_callback:
                progress_callback(30.0, None)
            task.progress = 30.0

            # Run conversion
            logger.info(f"Converting {source_path} to {task.output_path}")

            cmd = [
                "python3",
                str(self.convert_script),
                str(source_path),
                "--outfile", str(task.output_path),
                "--outtype", "f16",  # FP16 output
            ]

            logger.debug(f"Running command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=os.environ.copy(),
            )

            # Monitor progress
            for line in process.stdout:
                logger.debug(line.strip())

                # Update progress (estimated)
                if "Writing" in line:
                    if progress_callback:
                        progress_callback(80.0, None)
                    task.progress = 80.0

            process.wait()

            if process.returncode != 0:
                task.status = TaskStatus.FAILED
                task.error = f"Conversion failed with exit code {process.returncode}"
                logger.error(task.error)
                return False

            # Verify output exists
            if not task.output_path.exists():
                task.status = TaskStatus.FAILED
                task.error = "Output file not created"
                logger.error(task.error)
                return False

            # Update progress: Complete
            if progress_callback:
                progress_callback(100.0, 0)
            task.progress = 100.0
            task.status = TaskStatus.COMPLETED

            output_size = task.output_path.stat().st_size / (1024 ** 3)
            logger.info(f"GGUF conversion completed: {task.output_path} ({output_size:.2f}GB)")
            return True

        except Exception as e:
            error_msg = f"GGUF conversion failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = error_msg
            return False

    def _download_convert_script(self) -> Optional[Path]:
        """Download convert.py from llama.cpp repository.

        Returns:
            Path to downloaded script or None if failed
        """
        try:
            import urllib.request

            url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert.py"
            download_path = Path.home() / ".cache" / "llama.cpp" / "convert.py"

            download_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading convert.py from {url}")
            urllib.request.urlretrieve(url, download_path)
            logger.info(f"Downloaded convert.py to {download_path}")

            return download_path

        except Exception as e:
            logger.error(f"Failed to download convert.py: {e}")
            return None
