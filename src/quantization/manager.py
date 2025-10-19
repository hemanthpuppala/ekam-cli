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

        logger.info(f"Found {len(quantizable)} quantizable models (GGUF: {sum(1 for m in quantizable if m.provider == ProviderType.GGUF)}, HF: {sum(1 for m in quantizable if m.provider == ProviderType.HUGGINGFACE)})")
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
            method: Quantization method ("generic", "advanced", "gguf", "gguf_conversion")

        Returns:
            List of recommendations
        """
        from ..models.provider import ProviderType

        # GGUF models: Direct quantization
        if model_info.provider == ProviderType.GGUF:
            return get_quantization_recommendations(
                model_info=model_info,
                system_specs=self.system_specs,
                method_family="GGUF",
                use_gpu=use_gpu,
            )
        # HuggingFace models: Support Generic, Advanced, and GGUF conversion
        elif model_info.provider == ProviderType.HUGGINGFACE:
            if method == "gguf_conversion":
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
            else:
                # Generic quantization (FP16/INT8/INT4)
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
    ) -> QuantizationTask:
        """Create a quantization task.

        Args:
            model_info: Model to quantize
            quant_type: Quantization type
            module: Quantization module to use
            use_gpu: Whether to use GPU
            background: Whether to run in background
            vlm_components: For VLMs, which components to quantize ("vision", "language", "both", or None)

        Returns:
            Created task
        """
        # Generate task ID
        task_id = f"quant_{uuid.uuid4().hex[:8]}"

        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{model_info.model_id.replace('/', '_')}_{quant_type.value}_{timestamp}{quant_type.file_extension}"
        output_path = self.output_dir / output_filename

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
