"""Dynamic llama.cpp VLM support detection.

This module queries llama.cpp's convert_hf_to_gguf.py to determine which VLM
architectures are actually supported, avoiding hardcoded model lists.
"""

import sys
from pathlib import Path
from typing import Set, Optional
from loguru import logger


class LlamaCppVLMSupport:
    """Detects VLM support by querying llama.cpp's converter."""

    _supported_vlm_architectures: Optional[Set[str]] = None
    _llama_cpp_path: Optional[Path] = None

    @classmethod
    def get_llama_cpp_path(cls) -> Optional[Path]:
        """Find llama.cpp directory in the project.

        Returns:
            Path to llama.cpp directory, or None if not found
        """
        if cls._llama_cpp_path is not None:
            return cls._llama_cpp_path

        # Try common locations
        candidates = [
            Path.cwd() / "llama.cpp",
            Path.cwd().parent / "llama.cpp",
            Path(__file__).parent.parent.parent / "llama.cpp",
        ]

        for candidate in candidates:
            converter_path = candidate / "convert_hf_to_gguf.py"
            if converter_path.exists():
                cls._llama_cpp_path = candidate
                logger.debug(f"Found llama.cpp at: {candidate}")
                return candidate

        logger.warning("llama.cpp not found in project - VLM support detection will use fallback")
        return None

    @classmethod
    def get_supported_vlm_architectures(cls) -> Set[str]:
        """Get list of VLM architectures supported by llama.cpp.

        This dynamically imports llama.cpp's converter and queries its
        registered MMPROJ models, avoiding hardcoded lists.

        Returns:
            Set of HuggingFace architecture names (e.g., "Qwen2VLForConditionalGeneration")
        """
        # Return cached result if available
        if cls._supported_vlm_architectures is not None:
            return cls._supported_vlm_architectures

        llama_cpp_dir = cls.get_llama_cpp_path()

        if llama_cpp_dir is None:
            # Fallback: return known working models as of Nov 2025
            logger.warning("Using fallback VLM support list (llama.cpp not found)")
            return cls._get_fallback_vlm_architectures()

        try:
            # Dynamically import llama.cpp's converter
            converter_path = llama_cpp_dir / "convert_hf_to_gguf.py"

            # Add llama.cpp to Python path temporarily
            sys.path.insert(0, str(llama_cpp_dir))

            try:
                # Import the converter module
                import convert_hf_to_gguf
                from convert_hf_to_gguf import ModelBase, ModelType

                # Get all registered MMPROJ (vision) models
                mmproj_models = ModelBase._model_classes[ModelType.MMPROJ]

                # Extract architecture names
                supported = set(mmproj_models.keys())

                logger.info(f"✓ Detected {len(supported)} VLM architectures from llama.cpp")
                logger.debug(f"Supported VLM architectures: {sorted(supported)}")

                cls._supported_vlm_architectures = supported
                return supported

            finally:
                # Clean up Python path
                sys.path.remove(str(llama_cpp_dir))

                # Remove imported module to avoid conflicts
                if 'convert_hf_to_gguf' in sys.modules:
                    del sys.modules['convert_hf_to_gguf']

        except Exception as e:
            logger.warning(f"Failed to query llama.cpp for VLM support: {e}")
            logger.debug("Falling back to known VLM architectures", exc_info=True)
            return cls._get_fallback_vlm_architectures()

    @classmethod
    def _get_fallback_vlm_architectures(cls) -> Set[str]:
        """Fallback list of known VLM architectures (as of Nov 2025).

        This is used when llama.cpp cannot be queried directly.
        Based on llama.cpp commit as of November 2025.

        Returns:
            Set of HuggingFace architecture names
        """
        fallback = {
            # Qwen family
            "Qwen2VLForConditionalGeneration",
            "Qwen2VLModel",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2_5OmniModel",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",

            # SmolVLM
            "SmolVLMForConditionalGeneration",
            "Idefics3ForConditionalGeneration",  # SmolVLM uses this

            # Gemma
            "Gemma3ForConditionalGeneration",

            # InternVL
            "InternVisionModel",

            # LLaVA family
            "LlavaForConditionalGeneration",

            # Llama 4 Vision
            "Llama4ForConditionalGeneration",

            # Mistral
            "Mistral3ForConditionalGeneration",

            # Others
            "CogVLMForCausalLM",
            "KimiVLForConditionalGeneration",
            "JanusForConditionalGeneration",
            "Lfm2VlForConditionalGeneration",
            "LightOnOCRForConditionalGeneration",

            # Audio models (also use mmproj)
            "Qwen2AudioForConditionalGeneration",
            "UltravoxModel",
            "VoxtralForConditionalGeneration",
        }

        cls._supported_vlm_architectures = fallback
        return fallback

    @classmethod
    def is_architecture_supported(cls, architecture_name: str) -> bool:
        """Check if a specific HuggingFace architecture is supported by llama.cpp.

        Args:
            architecture_name: HuggingFace architecture (e.g., "Qwen2VLForConditionalGeneration")

        Returns:
            True if llama.cpp supports this architecture for GGUF conversion
        """
        supported = cls.get_supported_vlm_architectures()
        return architecture_name in supported

    @classmethod
    def clear_cache(cls):
        """Clear cached llama.cpp support data.

        Useful for testing or when llama.cpp is updated.
        """
        cls._supported_vlm_architectures = None
        cls._llama_cpp_path = None
        logger.debug("Cleared llama.cpp VLM support cache")
