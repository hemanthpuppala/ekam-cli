"""GGUF quantization using llama.cpp."""

import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ..models.model import ModelInfo
from ..models.provider import ProviderType
from .models import QuantizationTask, QuantizationType, TaskStatus


class GGUFQuantizer:
    """Handle GGUF quantization using llama.cpp tools."""

    def __init__(self):
        """Initialize GGUF quantizer."""
        self.quantize_binary: Optional[Path] = None

    def check_availability(self) -> tuple[bool, str]:
        """Check if ll

ama.cpp quantization tools are available.

        Returns:
            (available, message) tuple
        """
        # Check for llama-quantize in common locations
        possible_paths = [
            Path.home() / ".llama.cpp" / "quantize",
            Path("/usr/local/bin/llama-quantize"),
            Path("/opt/homebrew/bin/llama-quantize"),
            Path("./llama.cpp/quantize"),
        ]

        for path in possible_paths:
            if path.exists():
                self.quantize_binary = path
                return True, f"Found llama.cpp quantize tool at {path}"

        # Try to find it in PATH
        try:
            result = subprocess.run(
                ["which", "llama-quantize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                self.quantize_binary = Path(result.stdout.strip())
                return True, f"Found llama-quantize in PATH: {self.quantize_binary}"
        except Exception:
            pass

        return False, "llama.cpp quantize tool not found. Please install llama.cpp tools."

    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for quantization.

        Args:
            model_info: Model to quantize

        Returns:
            Path to model file, or None if not available
        """
        if model_info.provider == ProviderType.GGUF:
            # GGUF provider stores local path
            if hasattr(model_info, "local_path") and model_info.local_path:
                return Path(model_info.local_path)

        elif model_info.provider == ProviderType.OLLAMA:
            # Ollama models are stored in ~/.ollama/models
            ollama_models = Path.home() / ".ollama" / "models"
            # Try to find the model manifest
            manifest_path = ollama_models / "manifests" / "registry.ollama.ai" / "library" / model_info.model_id
            if manifest_path.exists():
                logger.info(f"Found Ollama model manifest: {manifest_path}")
                # Ollama models need to be exported first
                # This is complex - for MVP, we'll skip Ollama models
                return None

        elif model_info.provider == ProviderType.HUGGINGFACE:
            # HuggingFace models need to be converted to GGUF first
            # Check if already converted
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            # This is complex - for MVP, we'll skip HF models
            return None

        return None

    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Perform GGUF quantization.

        Args:
            task: Quantization task
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful
        """
        # Check if quantize tool is available
        available, message = self.check_availability()
        if not available:
            task.status = TaskStatus.FAILED
            task.error = message
            logger.error(f"Quantization failed: {message}")
            return False

        # Get source model path
        source_path = self.get_source_model_path(task.model_info)
        if not source_path or not source_path.exists():
            task.status = TaskStatus.FAILED
            task.error = f"Source model not found for {task.model_info.name}. Only local GGUF models can be quantized currently."
            logger.error(task.error)
            return False

        # Ensure output directory exists
        task.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build quantization command
        # llama-quantize usage: llama-quantize [options] model-f32.gguf model-quant.gguf type
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
            str(source_path),
            str(task.output_path),
            quant_type_arg,
        ]

        logger.info(f"Running quantization command: {' '.join(cmd)}")

        # Update task status
        task.status = TaskStatus.RUNNING
        task.progress = 0.0

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
            last_progress = 0.0

            # Monitor output for progress
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break

                logger.debug(f"Quantize output: {line.strip()}")

                # Try to parse progress from output
                # llama.cpp outputs lines like: "[ 123/ 456] ..."
                match = re.search(r"\[\s*(\d+)/\s*(\d+)\]", line)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    progress = (current / total) * 100.0
                    task.progress = progress

                    # Estimate ETA
                    elapsed = time.time() - start_time
                    if progress > 0:
                        total_estimated = (elapsed / progress) * 100
                        eta = total_estimated - elapsed
                        task.eta_seconds = eta

                        if progress_callback and abs(progress - last_progress) >= 1.0:
                            progress_callback(progress, eta)
                            last_progress = progress

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
        """Estimate output file size in GB.

        Args:
            model_info: Source model
            quant_type: Quantization type

        Returns:
            Estimated size in GB
        """
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
