"""Quantized models provider - discovers models from results/quantizations/."""

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from ..models.endpoints import CompatibilityStatus, ModelType, ProviderType
from ..models.model import ModelInfo
from ..models.model_cache import ModelMetadataCache
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
        self.config = config
        self.system_specs = system_specs or SystemSpecs.detect()
        self.quantized_dir = Path(config.models_dir or "results/quantizations")
        self.quantized_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata cache for efficient model inspection
        self.metadata_cache = ModelMetadataCache()
        logger.debug("Initialized model metadata cache")

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

        # Discover GGUF files (exclude only macOS resource forks)
        for gguf_file in self.quantized_dir.glob("*.gguf"):
            # Skip macOS resource forks (._* files)
            if gguf_file.name.startswith("._"):
                continue

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
        """Create ModelInfo from GGUF file using metadata cache.

        Args:
            gguf_path: Path to .gguf file

        Returns:
            ModelInfo or None if invalid
        """
        # Get metadata from cache (reads GGUF header and metadata file)
        cached_metadata = self.metadata_cache.get_metadata(str(gguf_path), provider="quantized")

        size_gb = gguf_path.stat().st_size / (1024 ** 3)

        # Check if this is an FP16 intermediate file
        is_fp16 = gguf_path.stem.endswith("_f16")

        # Try to read metadata file for original model info
        metadata_file = gguf_path.with_suffix(".json")
        quant_type = "fp16" if is_fp16 else "unknown"
        original_model = "unknown"
        params_billions = None
        ram_gb = None

        # Use cached metadata if available
        if cached_metadata:
            quant_type = cached_metadata.quantization
            params_billions = cached_metadata.params_billions
            ram_gb = cached_metadata.ram_gb
            # Architecture might contain original model info
            if cached_metadata.architecture and cached_metadata.architecture != "unknown":
                original_model = cached_metadata.architecture

        # Also read JSON metadata file for additional info
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    # Only override quant_type if not FP16 and not from cache
                    if not is_fp16 and not cached_metadata:
                        quant_type = metadata.get("quant_type", "unknown")
                    # Prefer JSON metadata for original model name
                    if "original_model" in metadata:
                        original_model = metadata.get("original_model", "unknown")
            except Exception as e:
                logger.warning(f"Could not read metadata for {gguf_path}: {e}")

        # Model ID: Use full path so we can load it later
        # Format: "quantized:gguf:/path/to/file.gguf"
        model_id = f"quantized:gguf:{gguf_path}"

        # Display name: Extract model name from filename and show quantization type
        # Example: "_Volumes_..._salamandra-2b_q4_k_m_20251012_221056" → "salamandra-2b [Q4_K_M]"
        # Example: "_Volumes_..._salamandra-2b_q4_k_m_20251012_221056_f16" → "salamandra-2b [FP16]"
        filename_stem = gguf_path.stem

        # Try to extract meaningful model name from filename
        # Pattern: ...model-name_quanttype_timestamp or ...model-name_quanttype_timestamp_f16
        if "_" in filename_stem:
            # Remove path components (anything before last occurrence of model identifier)
            parts = filename_stem.split("_")

            # Handle FP16 files: remove the trailing "f16" part
            if is_fp16 and parts[-1] == "f16":
                parts = parts[:-1]

            # Find the quantization type in the filename (q4_k_m, q5_k_m, etc.)
            quant_idx = -1
            for i, part in enumerate(parts):
                if part.startswith("q") and i + 2 < len(parts):
                    # Check if this looks like a quant type (e.g., q4_k_m)
                    potential_quant = "_".join(parts[i:i+3])
                    if potential_quant in ["q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0"]:
                        quant_idx = i
                        break

            # Extract model name (everything before quant type or timestamp)
            if quant_idx > 0:
                # Join everything before quant_idx, but try to extract just the model name
                model_part = "_".join(parts[:quant_idx])
                # Take the last meaningful part (usually the actual model name)
                model_name_parts = model_part.split("_")
                # Find the last part that looks like a model name (not a path component)
                for i in range(len(model_name_parts) - 1, -1, -1):
                    if model_name_parts[i] and not model_name_parts[i].startswith("Volumes"):
                        # Use the last 2 parts as the model name, skipping "models" if present
                        start_idx = max(0, i - 1)
                        name_parts = model_name_parts[start_idx:]

                        # Remove "models" prefix if present
                        if name_parts and name_parts[0] == "models":
                            name_parts = name_parts[1:]

                        name = "-".join(name_parts) if name_parts else filename_stem
                        break
            else:
                # No quant type found, might be FP16 without quant prefix
                # Find the last meaningful parts before timestamp
                model_name_parts = []
                for i in range(len(parts) - 1, -1, -1):
                    # Skip timestamp-like parts (8 digits)
                    if parts[i].isdigit() and len(parts[i]) == 8:
                        continue
                    # Skip time-like parts (6 digits)
                    if parts[i].isdigit() and len(parts[i]) == 6:
                        continue
                    # Skip path components
                    if parts[i].startswith("Volumes"):
                        continue
                    # Found meaningful part
                    if model_name_parts or parts[i]:
                        model_name_parts.insert(0, parts[i])
                        # Get at most 2-3 meaningful parts
                        if len(model_name_parts) >= 2:
                            break

                if model_name_parts and model_name_parts[0] == "models":
                    model_name_parts = model_name_parts[1:]

                name = "-".join(model_name_parts) if model_name_parts else filename_stem
        else:
            name = filename_stem

        # Add quantization type tag - make it very clear with descriptions
        if quant_type != "unknown":
            if is_fp16:
                name = f"{name} [FP16 - Half Precision]"
            elif quant_type == "q4_k_m":
                name = f"{name} [Q4_K_M - 4-bit Medium Quality]"
            elif quant_type == "q4_k_s":
                name = f"{name} [Q4_K_S - 4-bit Small Size]"
            elif quant_type == "q5_k_m":
                name = f"{name} [Q5_K_M - 5-bit Medium Quality]"
            elif quant_type == "q5_k_s":
                name = f"{name} [Q5_K_S - 5-bit Small Size]"
            elif quant_type == "q6_k":
                name = f"{name} [Q6_K - 6-bit High Quality]"
            elif quant_type == "q8_0":
                name = f"{name} [Q8_0 - 8-bit Highest Quality]"
            elif quant_type == "int8":
                name = f"{name} [INT8 - 8-bit Integer]"
            elif quant_type == "int4":
                name = f"{name} [INT4 - 4-bit Integer]"
            else:
                name = f"{name} [{quant_type.upper()}]"

        # Assess compatibility
        compatibility, compatibility_message = self._assess_compatibility_detailed(size_gb)

        # Import required types
        from ..models.endpoints import EndpointType

        return ModelInfo(
            model_id=model_id,
            name=name,
            provider=ProviderType.QUANTIZED,
            model_type=ModelType.LLM,  # Assume LLM for now
            size_gb=size_gb,
            params_billions=params_billions,  # From metadata cache
            ram_gb=ram_gb,  # From metadata cache
            capabilities=[EndpointType.TEXT],  # Default to text-only
            compatibility=compatibility,
            compatibility_message=compatibility_message,
            is_installed=True,
            quantization=quant_type if quant_type != "unknown" else None,
            architecture=original_model if original_model != "unknown" else None,
        )

    def _create_model_info_from_hf(self, model_dir: Path) -> Optional[ModelInfo]:
        """Create ModelInfo from HuggingFace format directory using metadata cache.

        Args:
            model_dir: Path to model directory

        Returns:
            ModelInfo or None if invalid
        """
        # Get metadata from cache (reads config.json)
        cached_metadata = self.metadata_cache.get_metadata(str(model_dir), provider="huggingface")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        size_gb = total_size / (1024 ** 3)

        # Try to read metadata
        metadata_file = model_dir / "quantization_metadata.json"
        quant_type = "unknown"
        original_model = "unknown"
        params_billions = None
        ram_gb = None

        # Use cached metadata if available
        if cached_metadata:
            quant_type = cached_metadata.quantization
            params_billions = cached_metadata.params_billions
            ram_gb = cached_metadata.ram_gb
            if cached_metadata.architecture and cached_metadata.architecture != "unknown":
                original_model = cached_metadata.architecture

        # Also read JSON metadata file for additional info
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    # Prefer JSON metadata for quantization type and original model
                    if "quant_type" in metadata:
                        quant_type = metadata.get("quant_type", "unknown")
                    if "original_model" in metadata:
                        original_model = metadata.get("original_model", "unknown")
            except Exception as e:
                logger.warning(f"Could not read metadata for {model_dir}: {e}")

        # Model ID: Use full path with format prefix
        # Format: "quantized:hf:/path/to/model_dir"
        model_id = f"quantized:hf:{model_dir}"

        # Display name - make quantization type very clear
        name = f"{model_dir.name}"
        if quant_type != "unknown":
            # Add descriptive quantization type label
            if quant_type == "fp16":
                name = f"{model_dir.name} [FP16 - Half Precision]"
            elif quant_type == "int8":
                name = f"{model_dir.name} [INT8 - 8-bit Integer]"
            elif quant_type == "int4":
                name = f"{model_dir.name} [INT4 - 4-bit Integer]"
            else:
                name = f"{model_dir.name} [{quant_type.upper()}]"

        # Assess compatibility
        compatibility, compatibility_message = self._assess_compatibility_detailed(size_gb)

        # Import required types
        from ..models.endpoints import EndpointType

        return ModelInfo(
            model_id=model_id,
            name=name,
            provider=ProviderType.QUANTIZED,
            model_type=ModelType.LLM,
            size_gb=size_gb,
            params_billions=params_billions,  # From metadata cache
            ram_gb=ram_gb,  # From metadata cache
            capabilities=[EndpointType.TEXT],  # Default to text-only
            compatibility=compatibility,
            compatibility_message=compatibility_message,
            is_installed=True,
            quantization=quant_type if quant_type != "unknown" else None,
            architecture=original_model if original_model != "unknown" else None,
        )

    def _assess_compatibility_detailed(self, model_size_gb: float) -> tuple[CompatibilityStatus, str]:
        """Assess model compatibility with system.

        Args:
            model_size_gb: Model size in GB

        Returns:
            Tuple of (compatibility status, message)
        """
        recommended_size = self.system_specs.recommended_model_size_gb
        ratio = model_size_gb / recommended_size

        if model_size_gb < recommended_size * 0.7:
            return (
                CompatibilityStatus.PERFECT_FIT,
                f"Model uses {ratio*100:.0f}% of recommended size ({model_size_gb:.1f}GB / {recommended_size:.1f}GB)"
            )
        elif model_size_gb <= recommended_size:
            return (
                CompatibilityStatus.TIGHT_FIT,
                f"Model uses {ratio*100:.0f}% of recommended size ({model_size_gb:.1f}GB / {recommended_size:.1f}GB). May impact performance."
            )
        else:
            return (
                CompatibilityStatus.TOO_LARGE,
                f"Model exceeds recommended size by {(ratio-1)*100:.0f}% ({model_size_gb:.1f}GB vs {recommended_size:.1f}GB). May cause OOM errors."
            )

    def load_model(self, model_id: str, device: str) -> Any:
        """Load a quantized model.

        Delegates to appropriate backend based on format:
        - GGUF files → llama-cpp-python
        - HF format → transformers

        Args:
            model_id: Model identifier (format: "quantized:gguf:/path" or "quantized:hf:/path")
            device: Target device ("cuda", "mps", or "cpu")

        Returns:
            Loaded model handle
        """
        # Parse model_id to extract format and path
        # Format: "quantized:gguf:/path/to/file" or "quantized:hf:/path/to/dir"
        parts = model_id.split(":", 2)
        if len(parts) != 3 or parts[0] != "quantized":
            raise ValueError(f"Invalid quantized model_id format: {model_id}")

        model_format = parts[1]  # "gguf" or "hf"
        model_path = Path(parts[2])  # Path to model

        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        if model_format == "gguf":
            return self._load_gguf_model(model_path, device)
        elif model_format == "hf":
            return self._load_hf_model(model_path, device)
        else:
            raise ValueError(f"Unknown quantized model format: {model_format}")

    def _load_gguf_model(self, model_path: Path, device: str) -> Any:
        """Load GGUF quantized model using llama-cpp-python.

        Args:
            model_path: Path to .gguf file
            device: Target device ("cuda", "mps", or "cpu")

        Returns:
            Llama model instance
        """
        try:
            from llama_cpp import Llama

            logger.info(f"Loading GGUF model: {model_path}")

            # Determine GPU layers based on device
            n_gpu_layers = -1 if device in ["cuda", "mps"] else 0

            model = Llama(
                model_path=str(model_path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=2048,
                verbose=False,
            )

            logger.info(f"GGUF model loaded successfully on {device}")
            return model

        except ImportError:
            raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

    def _load_hf_model(self, model_path: Path, device: str) -> Any:
        """Load HuggingFace format quantized model.

        Args:
            model_path: Path to model directory
            device: Target device ("cuda", "mps", or "cpu")

        Returns:
            Transformers model instance
        """
        try:
            from transformers import AutoModelForCausalLM

            logger.info(f"Loading HF quantized model: {model_path}")

            # Use device_map="auto" for optimal placement
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            logger.info(f"HF quantized model loaded successfully on {device}")
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
        self,
        model_handle: Any,
        prompt: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        custom_parameters: Optional[dict] = None
    ) -> str:
        """Run text generation on quantized model with conversation history support.

        Args:
            model_handle: Loaded model
            prompt: Input text
            conversation_history: Optional list of (user_msg, bot_response) tuples
            custom_parameters: Optional custom generation parameters (temperature, top_p, max_tokens, etc.)

        Returns:
            Generated text
        """
        # Extract parameters from custom_parameters
        # Increased default max_tokens for more detailed responses
        max_tokens = 1024  # Allow longer responses by default
        temperature = 0.7
        top_p = 0.9
        repetition_penalty = 1.1  # Prevent infinite loops
        frequency_penalty = 0.0
        presence_penalty = 0.0

        if custom_parameters:
            max_tokens = custom_parameters.get("max_tokens", max_tokens)
            temperature = custom_parameters.get("temperature", temperature)
            top_p = custom_parameters.get("top_p", top_p)
            repetition_penalty = custom_parameters.get("repetition_penalty", repetition_penalty)
            frequency_penalty = custom_parameters.get("frequency_penalty", frequency_penalty)
            presence_penalty = custom_parameters.get("presence_penalty", presence_penalty)

        # Detect model type and use appropriate method
        if hasattr(model_handle, "__call__"):
            # llama-cpp-python Llama model
            # Format conversation history if provided
            if conversation_history:
                # Use SIMPLE template with explicit system prompt for conversation context
                from ..utils.history_formatter import format_conversation_history, ChatTemplate

                # Add system prompt - clear and balanced for all model types
                system_prompt = (
                    "You are a helpful AI assistant. Provide clear, direct answers to the user's questions. "
                    "Keep responses focused and relevant to what was asked."
                )

                formatted_prompt = format_conversation_history(
                    conversation_history,
                    prompt,
                    max_turns=5,
                    system_prompt=system_prompt,
                    template=ChatTemplate.SIMPLE
                )
            else:
                # For simple prompts without history, use clear instruction format
                formatted_prompt = (
                    "You are a helpful AI assistant. Provide clear, direct answers.\n\n"
                    f"User: {prompt}\nAssistant:"
                )

            # Define stop tokens to prevent template leakage and repetition
            # Be careful not to interfere with ongoing generation
            stop_tokens = [
                "\n\nUser:",          # Stop before next user turn (with double newline)
                "\n\nHuman:",         # Alternative role marker
                "\n### Instruction:", # Stop at new instruction blocks
                "<|im_end|>",        # ChatML end token
                "<|endoftext|>",     # Common end token
            ]

            # Allow custom stop tokens from parameters
            if custom_parameters and "stop" in custom_parameters:
                custom_stop = custom_parameters.get("stop")
                if isinstance(custom_stop, list):
                    stop_tokens.extend(custom_stop)
                elif isinstance(custom_stop, str):
                    stop_tokens.append(custom_stop)

            response = model_handle(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop_tokens,
                echo=False,
            )

            # Extract and clean the response text using shared utility
            from ..utils.response_cleaner import clean_model_response

            response_text = response["choices"][0]["text"]
            cleaned_response = clean_model_response(response_text, aggressive=True)

            return cleaned_response
        else:
            # HuggingFace model
            from transformers import AutoTokenizer

            # For HF models, we need the model path to load tokenizer
            # This should be passed in custom_parameters or we need to store it
            model_path = custom_parameters.get("model_path") if custom_parameters else None
            if not model_path:
                raise ValueError("model_path required for HF model inference")

            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Format with conversation history if provided
            if conversation_history:
                # Build conversation for HF models
                messages = []
                for user_msg, bot_msg in conversation_history:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": bot_msg})
                messages.append({"role": "user", "content": prompt})

                # Use apply_chat_template if available
                if hasattr(tokenizer, "apply_chat_template"):
                    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    # Fallback: simple concatenation
                    formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            else:
                formatted_prompt = prompt

            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            outputs = model_handle.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True
            )
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_model_info(self, model_id: str) -> dict:
        """Get detailed model information for quantized models.

        Args:
            model_id: Model identifier

        Returns:
            Dict with model metadata including default_parameters
        """
        # Find the model in discovered models
        models = self.discover_models()
        model = next((m for m in models if m.model_id == model_id), None)

        if not model:
            # Return minimal info if not found
            return {
                "model_id": model_id,
                "name": model_id,
                "provider": "quantized",
                "default_parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": 1024,  # Updated default for quantized models
                    "repeat_penalty": 1.1,
                }
            }

        # Return full model info
        return {
            "model_id": model.model_id,
            "name": model.name,
            "provider": "quantized",
            "architecture": model.architecture or "Unknown",
            "quantization": model.quantization or "Unknown",
            "size_gb": model.size_gb,
            "model_type": str(model.model_type).upper() if hasattr(model.model_type, 'value') else str(model.model_type).upper(),
            "capabilities": [str(cap) for cap in model.capabilities] if model.capabilities else ["text"],
            "default_parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 1024,  # Updated default for quantized models
                "repeat_penalty": 1.1,
            }
        }

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
            model_id: Model identifier (format: "quantized:gguf:/path" or "quantized:hf:/path")

        Returns:
            True if deleted successfully
        """
        try:
            # Parse model_id to extract path
            parts = model_id.split(":", 2)
            if len(parts) != 3 or parts[0] != "quantized":
                logger.error(f"Invalid model_id format: {model_id}")
                return False

            model_path = Path(parts[2])

            if not model_path.exists():
                logger.error(f"Model path does not exist: {model_path}")
                return False

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
