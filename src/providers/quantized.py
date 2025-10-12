"""Quantized models provider - discovers models from results/quantizations/."""

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from ..models.endpoints import CompatibilityStatus, ModelType, ProviderType
from ..models.model import ModelInfo
from ..models.provider import ProviderConfig
from ..models.system import SystemSpecs
from .base import BaseProvider


class QuantizedProvider(BaseProvider):
    """Provider for quantized models.

    Discovers and manages models quantized by the built-in quantization feature.
    All quantized models are stored in results/quantizations/ with metadata.
    """

    def __init__(self, config: ProviderConfig, system_specs: Optional[SystemSpecs] = None):
        """Initialize QuantizedProvider.

        Args:
            config: Provider configuration
            system_specs: System specifications for compatibility checking
        """
        super().__init__(config)
        self.system_specs = system_specs or SystemSpecs.detect()
        self.quantized_dir = Path(config.models_dir or "results/quantizations")
        self.quantized_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"QuantizedProvider initialized: {self.quantized_dir}")

    def discover_models(self) -> list[ModelInfo]:
        """Discover all quantized models.

        Scans results/quantizations/ for:
        - .gguf files (GGUF quantized models)
        - .safetensors directories (HF format quantized models)
        - Reads metadata files for quantization info

        Returns:
            List of discovered quantized models
        """
        models = []

        if not self.quantized_dir.exists():
            logger.info("Quantized models directory does not exist yet")
            return models

        # Discover GGUF files
        for gguf_file in self.quantized_dir.glob("*.gguf"):
            try:
                model_info = self._create_model_info_from_gguf(gguf_file)
                if model_info:
                    models.append(model_info)
            except Exception as e:
                logger.error(f"Failed to process GGUF file {gguf_file}: {e}")

        # Discover HF format directories (contain .safetensors files)
        for item in self.quantized_dir.iterdir():
            if item.is_dir() and list(item.glob("*.safetensors")):
                try:
                    model_info = self._create_model_info_from_hf(item)
                    if model_info:
                        models.append(model_info)
                except Exception as e:
                    logger.error(f"Failed to process HF model {item}: {e}")

        logger.info(f"Discovered {len(models)} quantized models")
        return models

    def _create_model_info_from_gguf(self, gguf_path: Path) -> Optional[ModelInfo]:
        """Create ModelInfo from GGUF file.

        Args:
            gguf_path: Path to .gguf file

        Returns:
            ModelInfo or None if invalid
        """
        size_gb = gguf_path.stat().st_size / (1024 ** 3)

        # Try to read metadata file
        metadata_file = gguf_path.with_suffix(".json")
        quant_type = "unknown"
        original_model = "unknown"

        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    quant_type = metadata.get("quant_type", "unknown")
                    original_model = metadata.get("original_model", "unknown")
            except Exception as e:
                logger.warning(f"Could not read metadata for {gguf_path}: {e}")

        # Model ID: filename without extension
        model_id = f"quantized/{gguf_path.stem}"

        # Display name: show quantization type
        name = f"{gguf_path.stem}"
        if quant_type != "unknown":
            name = f"{gguf_path.stem} [{quant_type.upper()}]"

        # Assess compatibility
        compatibility = self._assess_compatibility(size_gb)

        return ModelInfo(
            model_id=model_id,
            name=name,
            provider=ProviderType.QUANTIZED,
            model_type=ModelType.LLM,  # Assume LLM for now
            size_gb=size_gb,
            file_path=str(gguf_path),
            compatibility=compatibility,
            is_installed=True,
            metadata={
                "format": "gguf",
                "quant_type": quant_type,
                "original_model": original_model,
                "quantized_date": gguf_path.stat().st_mtime,
            },
        )

    def _create_model_info_from_hf(self, model_dir: Path) -> Optional[ModelInfo]:
        """Create ModelInfo from HuggingFace format directory.

        Args:
            model_dir: Path to model directory

        Returns:
            ModelInfo or None if invalid
        """
        # Calculate total size
        total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        size_gb = total_size / (1024 ** 3)

        # Try to read metadata
        metadata_file = model_dir / "quantization_metadata.json"
        quant_type = "unknown"
        original_model = "unknown"

        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    quant_type = metadata.get("quant_type", "unknown")
                    original_model = metadata.get("original_model", "unknown")
            except Exception as e:
                logger.warning(f"Could not read metadata for {model_dir}: {e}")

        # Model ID
        model_id = f"quantized/{model_dir.name}"

        # Display name
        name = f"{model_dir.name}"
        if quant_type != "unknown":
            name = f"{model_dir.name} [{quant_type.upper()}]"

        # Assess compatibility
        compatibility = self._assess_compatibility(size_gb)

        return ModelInfo(
            model_id=model_id,
            name=name,
            provider=ProviderType.QUANTIZED,
            model_type=ModelType.LLM,
            size_gb=size_gb,
            file_path=str(model_dir),
            compatibility=compatibility,
            is_installed=True,
            metadata={
                "format": "huggingface",
                "quant_type": quant_type,
                "original_model": original_model,
                "quantized_date": model_dir.stat().st_mtime,
            },
        )

    def _assess_compatibility(self, model_size_gb: float) -> CompatibilityStatus:
        """Assess model compatibility with system.

        Args:
            model_size_gb: Model size in GB

        Returns:
            Compatibility status
        """
        recommended_size = self.system_specs.recommended_model_size_gb

        if model_size_gb < recommended_size * 0.7:
            return CompatibilityStatus.PERFECT_FIT
        elif model_size_gb <= recommended_size:
            return CompatibilityStatus.TIGHT_FIT
        else:
            return CompatibilityStatus.TOO_LARGE

    def load_model(self, model_id: str, **kwargs) -> Any:
        """Load a quantized model.

        Delegates to appropriate backend based on format:
        - GGUF files → llama-cpp-python
        - HF format → transformers

        Args:
            model_id: Model identifier
            **kwargs: Additional loading parameters

        Returns:
            Loaded model handle
        """
        # Find model
        models = self.discover_models()
        model_info = next((m for m in models if m.model_id == model_id), None)

        if not model_info:
            raise ValueError(f"Quantized model not found: {model_id}")

        model_format = model_info.metadata.get("format")
        model_path = Path(model_info.file_path)

        if model_format == "gguf":
            return self._load_gguf_model(model_path, **kwargs)
        elif model_format == "huggingface":
            return self._load_hf_model(model_path, **kwargs)
        else:
            raise ValueError(f"Unknown quantized model format: {model_format}")

    def _load_gguf_model(self, model_path: Path, **kwargs) -> Any:
        """Load GGUF quantized model using llama-cpp-python.

        Args:
            model_path: Path to .gguf file
            **kwargs: Additional parameters

        Returns:
            Llama model instance
        """
        try:
            from llama_cpp import Llama

            logger.info(f"Loading GGUF model: {model_path}")

            # Determine device
            device = self.config.get_primary_device()
            n_gpu_layers = -1 if device in ["cuda", "mps"] else 0

            model = Llama(
                model_path=str(model_path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=kwargs.get("n_ctx", 2048),
                verbose=False,
            )

            logger.info(f"GGUF model loaded successfully on {device}")
            return model

        except ImportError:
            raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

    def _load_hf_model(self, model_path: Path, **kwargs) -> Any:
        """Load HuggingFace format quantized model.

        Args:
            model_path: Path to model directory
            **kwargs: Additional parameters

        Returns:
            Transformers model instance
        """
        try:
            from transformers import AutoModelForCausalLM

            logger.info(f"Loading HF quantized model: {model_path}")

            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            logger.info("HF quantized model loaded successfully")
            return model

        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")

    def unload_model(self, model_handle: Any) -> None:
        """Unload model and free resources.

        Args:
            model_handle: Model instance to unload
        """
        # GGUF models (llama-cpp-python) handle cleanup automatically
        # HF models need manual cleanup
        try:
            if hasattr(model_handle, "cpu"):
                model_handle.cpu()
            del model_handle
            logger.info("Quantized model unloaded")
        except Exception as e:
            logger.warning(f"Error during model unload: {e}")

    def run_qa(self, model_handle: Any, image: Any, question: str) -> str:
        """Not implemented for quantized provider."""
        raise NotImplementedError("QA endpoint not yet implemented for quantized models")

    def run_caption(self, model_handle: Any, image: Any) -> str:
        """Not implemented for quantized provider."""
        raise NotImplementedError("Caption endpoint not yet implemented for quantized models")

    def run_detect(self, model_handle: Any, image: Any, object_name: str) -> list[dict]:
        """Not implemented for quantized provider."""
        raise NotImplementedError("Detect endpoint not yet implemented for quantized models")

    def run_point(self, model_handle: Any, image: Any, object_name: str) -> dict:
        """Not implemented for quantized provider."""
        raise NotImplementedError("Point endpoint not yet implemented for quantized models")

    def run_text(
        self, model_handle: Any, prompt: str, max_tokens: int = 512, **kwargs
    ) -> str:
        """Run text generation on quantized model.

        Args:
            model_handle: Loaded model
            prompt: Input text
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # Detect model type and use appropriate method
        if hasattr(model_handle, "__call__"):
            # llama-cpp-python Llama model
            response = model_handle(
                prompt,
                max_tokens=max_tokens,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                echo=False,
            )
            return response["choices"][0]["text"]
        else:
            # HuggingFace model
            from transformers import AutoTokenizer

            model_path = kwargs.get("model_path")
            if not model_path:
                raise ValueError("model_path required for HF model inference")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model_handle.generate(**inputs, max_new_tokens=max_tokens)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def estimate_model_size(self, model_name: str) -> float:
        """Estimate model size (already quantized, so return actual size).

        Args:
            model_name: Model name

        Returns:
            Model size in GB
        """
        models = self.discover_models()
        model = next((m for m in models if model_name in m.model_id), None)
        return model.size_gb if model else 0.0

    def install_model(self, model_name: str, **kwargs) -> bool:
        """Cannot install models - they are created via quantization."""
        logger.warning("Cannot install quantized models - use quantization feature")
        return False

    def delete_model(self, model_id: str) -> bool:
        """Delete a quantized model.

        Args:
            model_id: Model identifier

        Returns:
            True if deleted successfully
        """
        models = self.discover_models()
        model_info = next((m for m in models if m.model_id == model_id), None)

        if not model_info:
            logger.error(f"Model not found: {model_id}")
            return False

        try:
            model_path = Path(model_info.file_path)

            if model_path.is_file():
                # Delete GGUF file
                model_path.unlink()
                # Delete metadata if exists
                metadata_file = model_path.with_suffix(".json")
                if metadata_file.exists():
                    metadata_file.unlink()
            elif model_path.is_dir():
                # Delete HF directory
                import shutil
                shutil.rmtree(model_path)

            logger.info(f"Deleted quantized model: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
