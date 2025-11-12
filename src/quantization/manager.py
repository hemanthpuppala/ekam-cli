"""Main quantization manager that orchestrates all quantization operations."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from ..models.model import ModelInfo
from ..models.system import SystemSpecs
from .core.background import BackgroundJobManager
from .core.recommendations import get_quantization_recommendations
from .models import QuantizationModule, QuantizationTask, QuantizationType


class QuantizationManager:
    """Manage model quantization operations."""

    def __init__(self, system_specs: SystemSpecs, output_dir: Path):
        """Initialize quantization manager.

        Args:
            system_specs: System specifications
            output_dir: Directory for quantized models (e.g., results/quantizations/)
        """
        self.system_specs = system_specs
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize background job manager
        state_file = output_dir / ".quantization_state.json"
        self.job_manager = BackgroundJobManager(state_file)

        logger.info(f"Quantization manager initialized. Output dir: {output_dir}")

    def get_quantizable_models(self, all_models: list[ModelInfo]) -> list[ModelInfo]:
        """Filter models that can be quantized.

        Supports:
        - GGUF models: Direct quantization (Q4/Q5/Q6/Q8)
        - HuggingFace models: Via GGUF conversion or generic quantization (FP16/INT8/INT4)
        - Ollama models: GGUF requantization + format conversion (FP16/INT8/INT4/MLX/OpenVINO)

        Args:
            all_models: List of all available models

        Returns:
            List of quantizable models
        """
        from ..models.provider import ProviderType

        quantizable = []
        for model in all_models:
            # GGUF models: Direct quantization
            if model.provider == ProviderType.GGUF:
                quantizable.append(model)
            # HuggingFace models: GGUF conversion + quantization OR generic quantization
            elif model.provider == ProviderType.HUGGINGFACE:
                quantizable.append(model)
            # Ollama models: GGUF requantization + multi-format conversion
            elif model.provider == ProviderType.OLLAMA:
                # Only include if we have source_path (blob location)
                if model.source_path:
                    quantizable.append(model)
                else:
                    logger.debug(f"Skipping Ollama model {model.name} - no source_path")

        logger.info(
            f"Found {len(quantizable)} quantizable models "
            f"(GGUF: {sum(1 for m in quantizable if m.provider == ProviderType.GGUF)}, "
            f"HF: {sum(1 for m in quantizable if m.provider == ProviderType.HUGGINGFACE)}, "
            f"Ollama: {sum(1 for m in quantizable if m.provider == ProviderType.OLLAMA)})"
        )
        return quantizable

    def get_recommendations(
        self,
        model_info: ModelInfo,
        use_gpu: bool = False,
        method: str = "generic",
    ) -> list:
        """Get quantization recommendations for a model.

        Args:
            model_info: Model to quantize
            use_gpu: Whether GPU will be used
            method: Quantization method ("generic", "advanced", "gguf", "gguf_conversion", "dequantization", "mlx", "openvino")

        Returns:
            List of recommendations
        """
        from ..models.provider import ProviderType

        # GGUF models: Direct quantization or dequantization
        if model_info.provider == ProviderType.GGUF:
            if method == "dequantization":
                # Dequantization: Convert quantized GGUF to full precision
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="Dequantization",
                    use_gpu=use_gpu,
                )
            else:
                # Default: GGUF requantization (Q4→Q3, Q4→Q5, etc.)
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="GGUF",
                    use_gpu=use_gpu,
                )
        # HuggingFace models: Support Generic, Advanced, GGUF conversion, Dequantization, MLX, OpenVINO
        elif model_info.provider == ProviderType.HUGGINGFACE:
            if method == "dequantization":
                # Dequantization: Convert to full precision GGUF or HF format
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="Dequantization",
                    use_gpu=use_gpu,
                )
            elif method == "gguf_conversion":
                # GGUF conversion + quantization
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="GGUF",  # Will convert to GGUF first
                    use_gpu=use_gpu,
                )
            elif method == "advanced":
                # Advanced 4-bit quantization (GPTQ/AWQ/BnB)
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="Advanced",
                    use_gpu=use_gpu,
                )
            elif method == "mlx":
                # MLX quantization (Apple Silicon)
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="MLX",
                    use_gpu=False,  # MLX uses Metal
                )
            elif method == "openvino":
                # OpenVINO quantization (Intel CPU/iGPU)
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="OpenVINO",
                    use_gpu=False,  # OpenVINO CPU-based
                )
            else:
                # Generic quantization (FP16/INT8/INT4)
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="Generic",
                    use_gpu=use_gpu,
                )
        # Ollama models: GGUF requantization, dequantization, or conversions
        elif model_info.provider == ProviderType.OLLAMA:
            # Ollama models are GGUF-based, route based on method
            if method == "dequantization":
                # Dequantization: Convert quantized GGUF to full precision
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="Dequantization",
                    use_gpu=use_gpu,
                )
            elif method == "gguf" or method == "gguf_conversion":
                # GGUF requantization (Q4→Q3, Q4→Q5, etc.)
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="GGUF",
                    use_gpu=use_gpu,
                )
            elif method == "mlx":
                # Convert GGUF → MLX format
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="MLX",
                    use_gpu=False,
                )
            elif method == "openvino":
                # Convert GGUF → OpenVINO format
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="OpenVINO",
                    use_gpu=False,
                )
            elif method == "advanced":
                # Convert GGUF → HF → GPTQ/AWQ/BnB (with warnings)
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="Advanced",
                    use_gpu=use_gpu,
                )
            else:
                # Generic: Convert GGUF → FP16/INT8/INT4
                return get_quantization_recommendations(
                    model_info=model_info,
                    system_specs=self.system_specs,
                    method_family="Generic",
                    use_gpu=use_gpu,
                )

        return []

    def create_task(
        self,
        model_info: ModelInfo,
        quant_type: QuantizationType,
        module: QuantizationModule,
        use_gpu: bool = False,
        background: bool = False,
        vlm_components: Optional[str] = None,
        vision_encoder_type: Optional[QuantizationType] = None,
        language_decoder_type: Optional[QuantizationType] = None,
    ) -> QuantizationTask:
        """Create a quantization task.

        Args:
            model_info: Model to quantize
            quant_type: Quantization type
            module: Quantization module to use
            use_gpu: Whether to use GPU
            background: Whether to run in background
            vlm_components: For VLMs, which components to quantize ("vision", "language", "both", or None)
            vision_encoder_type: For VLM component-level, quantization type for vision encoder
            language_decoder_type: For VLM component-level, quantization type for language decoder

        Returns:
            Created task
        """
        # Generate task ID
        task_id = f"quant_{uuid.uuid4().hex[:8]}"

        # Extract clean model name from model_id
        # Format: "provider/model-name" or "model-name"
        clean_model_id = model_info.model_id.replace('/', '_').replace(':', '_')

        # Build descriptive name with module prefix for special modules
        # Format: {model_name}_{module}-{quant_type}
        # Examples:
        #   - MLX INT4: Qwen_Qwen3-0.6B_mlx-int4
        #   - OpenVINO INT8: Qwen_Qwen3-0.6B_openvino-int8
        #   - GGUF Q4_K_M: Qwen_Qwen3-0.6B_q4_k_m (no module prefix for GGUF)
        #   - Generic INT4: Qwen_Qwen3-0.6B_int4 (no module prefix for generic)
        #
        # NO TIMESTAMPS in the path to avoid module import issues with transformers

        if module == QuantizationModule.MLX:
            # MLX quantization - add "mlx-" prefix to quant type
            module_prefix = "mlx-"
        elif module == QuantizationModule.OPENVINO:
            # OpenVINO quantization - add "openvino-" prefix to quant type
            module_prefix = "openvino-"
        else:
            # GGUF, Generic, Advanced (GPTQ/AWQ/BnB) - no module prefix
            module_prefix = ""

        base_name = f"{clean_model_id}_{module_prefix}{quant_type.value}"

        # Determine output type: directory or file
        # GGUF outputs (.gguf files):
        #   - GGUF quantization (Q4/Q5/Q6/Q8)
        #   - Dequantization to GGUF (DEQUANT_FP16_GGUF, DEQUANT_FP32_GGUF)
        # Directory outputs (save_pretrained):
        #   - MLX, OpenVINO, Generic, GPTQ, AWQ, BnB
        #   - Dequantization to HF (DEQUANT_FP16_HF, DEQUANT_FP32_HF)

        is_gguf_output = (
            module == QuantizationModule.LLAMA_CPP and
            "_hf" not in quant_type.value  # HF formats contain "_hf" in the enum value
        )

        if is_gguf_output:
            # GGUF outputs to files with .gguf extension
            output_filename = f"{base_name}.gguf"
            output_path = self.output_dir / output_filename

            # Handle conflicts: if file exists, append number before extension
            if output_path.exists():
                counter = 1
                while (self.output_dir / f"{base_name}_{counter}.gguf").exists():
                    counter += 1
                output_path = self.output_dir / f"{base_name}_{counter}.gguf"
                logger.info(f"Output file exists, using: {output_path.name}")
        else:
            # Directory output for MLX, OpenVINO, Generic, GPTQ, AWQ, BnB
            # Also for HF dequantization outputs
            # All these use save_pretrained() which creates a directory with model files
            output_path = self.output_dir / base_name

            # Handle conflicts: if directory exists, append number
            if output_path.exists():
                counter = 1
                while (self.output_dir / f"{base_name}_{counter}").exists():
                    counter += 1
                output_path = self.output_dir / f"{base_name}_{counter}"
                logger.info(f"Output directory exists, using: {output_path.name}")

        # Create task
        task = QuantizationTask(
            task_id=task_id,
            model_info=model_info,
            quant_type=quant_type,
            module=module,
            output_path=output_path,
            use_gpu=use_gpu,
            background=background,
            vlm_components=vlm_components,
            vision_encoder_type=vision_encoder_type,
            language_decoder_type=language_decoder_type,
        )

        logger.info(f"Created quantization task {task_id}: {model_info.name} → {quant_type.display_name}")
        return task

    def submit_task(self, task: QuantizationTask) -> str:
        """Submit a task for execution.

        Args:
            task: Task to execute

        Returns:
            Task ID
        """
        return self.job_manager.submit_task(task)

    def get_active_tasks(self) -> list[QuantizationTask]:
        """Get all active tasks.

        Returns:
            List of active tasks
        """
        return self.job_manager.get_active_tasks()

    def get_all_tasks(self) -> list[QuantizationTask]:
        """Get all tasks (active and completed).

        Returns:
            List of all tasks
        """
        return self.job_manager.get_all_tasks()

    def get_task(self, task_id: str) -> Optional[QuantizationTask]:
        """Get task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task if found
        """
        return self.job_manager.get_task(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task ID

        Returns:
            True if cancelled
        """
        return self.job_manager.cancel_task(task_id)

    def remove_task(self, task_id: str) -> bool:
        """Remove a completed or failed task from the job list.

        Args:
            task_id: Task ID

        Returns:
            True if removed
        """
        return self.job_manager.remove_task(task_id)

    def has_active_jobs(self) -> bool:
        """Check if there are active jobs.

        Returns:
            True if active jobs exist
        """
        return self.job_manager.has_active_jobs()

    def get_status_summary(self) -> str:
        """Get status summary for display.

        Returns:
            Status string
        """
        return self.job_manager.get_status_summary()

    def check_can_run_multiple(self) -> bool:
        """Check if system can handle multiple concurrent jobs.

        Returns:
            True if multiple jobs allowed
        """
        from .recommendations import check_can_quantize_multiple

        return check_can_quantize_multiple(self.system_specs)

    def get_available_quantization_types(self, model_info: ModelInfo) -> list[QuantizationType]:
        """Get available quantization types for a model.

        Args:
            model_info: Model to check

        Returns:
            List of available quantization types
        """
        # Phase 1: Only GGUF types
        return [
            QuantizationType.GGUF_Q4_K_M,
            QuantizationType.GGUF_Q4_K_S,
            QuantizationType.GGUF_Q5_K_M,
            QuantizationType.GGUF_Q5_K_S,
            QuantizationType.GGUF_Q6_K,
            QuantizationType.GGUF_Q8_0,
        ]

    def get_available_modules(self, quant_type: QuantizationType) -> list[QuantizationModule]:
        """Get available modules for a quantization type.

        Args:
            quant_type: Quantization type

        Returns:
            List of available modules
        """
        # Phase 1: Only llama.cpp for GGUF
        if quant_type.method_family == "GGUF":
            return [QuantizationModule.LLAMA_CPP]

        return []

    def check_model_path_compatibility(self, model_path: Path) -> dict:
        """Check if a model path is compatible with quantization.

        Args:
            model_path: Path to model directory or file

        Returns:
            Dictionary with compatibility info:
            {
                "compatible": bool,
                "reason": str,
                "model_type": str,  # "gguf", "huggingface", "unknown"
                "can_convert_to_gguf": bool,
                "can_quantize_generic": bool,
                "available_methods": list[str]
            }
        """
        result = {
            "compatible": False,
            "reason": "",
            "model_type": "unknown",
            "can_convert_to_gguf": False,
            "can_quantize_generic": False,
            "available_methods": []
        }

        # Check if path exists
        if not model_path.exists():
            result["reason"] = f"Path does not exist: {model_path}"
            return result

        # Check if it's a GGUF file
        if model_path.is_file() and model_path.suffix == ".gguf":
            result["compatible"] = True
            result["model_type"] = "gguf"
            result["can_quantize_generic"] = False
            result["can_convert_to_gguf"] = False
            result["available_methods"] = ["GGUF Quantization (Q4/Q5/Q6/Q8)"]
            result["reason"] = "GGUF model - can be quantized directly"
            return result

        # Check if it's a HuggingFace model directory
        if model_path.is_dir():
            # Look for HF model files
            has_safetensors = list(model_path.glob("*.safetensors")) or list(model_path.glob("model.safetensors*"))
            has_bin = list(model_path.glob("*.bin")) or list(model_path.glob("pytorch_model*.bin"))
            has_config = (model_path / "config.json").exists()

            # HuggingFace models typically have model files (.safetensors or .bin)
            # config.json is nice to have but not strictly required for quantization
            if has_safetensors or has_bin:
                result["compatible"] = True
                result["model_type"] = "huggingface"
                result["can_convert_to_gguf"] = True
                result["can_quantize_generic"] = True
                result["available_methods"] = [
                    "Generic Quantization (FP16/INT8/INT4)",
                    "GGUF Conversion → GGUF Quantization (Q4/Q5/Q6/Q8)"
                ]

                if has_config:
                    result["reason"] = "HuggingFace model - can be quantized generically or converted to GGUF"
                else:
                    result["reason"] = "HuggingFace model (no config.json found, but model files detected) - can be quantized generically or converted to GGUF"

                return result

        result["reason"] = "Not a recognized model format (expected GGUF file or HuggingFace directory with safetensors/bin files)"
        return result
