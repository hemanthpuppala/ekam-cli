"""VLM (Vision Language Model) architecture detection.

This module detects if a model is a VLM and identifies its specific architecture
by inspecting config.json and model files. Architecture mappings are loaded
dynamically from vlm_registry.json for extensibility.

Supported VLM architectures:
- Qwen3-VL, Qwen2-VL, Qwen2.5-VL
- Gemma 3 (vision variants)
- LLaVA 1.5, LLaVA 1.6
- Pixtral 12B
- SmolVLM, SmolVLM2
- InternVL 2.5, InternVL 3
- Moondream2
- MiniCPM-V 2.5, 2.6, MiniCPM-O 2.6
- GLM-Edge
- IBM Granite Vision
- Mistral Small Vision
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

from loguru import logger


class VLMArchitecture(Enum):
    """Supported VLM architectures."""

    # Modern architectures (2025)
    QWEN_VL = "qwen-vl"  # Qwen3-VL and future versions
    QWEN2_VL = "qwen2-vl"
    QWEN2_5_VL = "qwen2.5-vl"
    GEMMA3_VISION = "gemma3-vision"
    PIXTRAL = "pixtral"
    SMOLVLM = "smolvlm"
    SMOLVLM2 = "smolvlm2"
    INTERNVL_2_5 = "internvl-2.5"
    INTERNVL_3 = "internvl-3"
    MOONDREAM2 = "moondream2"
    MISTRAL_SMALL_VISION = "mistral-small-vision"

    # Legacy architectures
    LLAVA_1_5 = "llava-1.5"
    LLAVA_1_6 = "llava-1.6"
    MINICPM_V_2_5 = "minicpm-v-2.5"
    MINICPM_V_2_6 = "minicpm-v-2.6"
    MINICPM_O_2_6 = "minicpm-o-2.6"
    GLM_EDGE = "glm-edge"
    GRANITE_VISION = "granite-vision"

    # Unknown/unsupported
    UNKNOWN = "unknown"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
            self.QWEN_VL: "Qwen3-VL",
            self.QWEN2_VL: "Qwen2-VL",
            self.QWEN2_5_VL: "Qwen2.5-VL",
            self.GEMMA3_VISION: "Gemma 3 Vision",
            self.PIXTRAL: "Pixtral 12B",
            self.SMOLVLM: "SmolVLM",
            self.SMOLVLM2: "SmolVLM2",
            self.INTERNVL_2_5: "InternVL 2.5",
            self.INTERNVL_3: "InternVL 3",
            self.MOONDREAM2: "Moondream2",
            self.MISTRAL_SMALL_VISION: "Mistral Small Vision",
            self.LLAVA_1_5: "LLaVA 1.5",
            self.LLAVA_1_6: "LLaVA 1.6",
            self.MINICPM_V_2_5: "MiniCPM-V 2.5",
            self.MINICPM_V_2_6: "MiniCPM-V 2.6",
            self.MINICPM_O_2_6: "MiniCPM-O 2.6",
            self.GLM_EDGE: "GLM-Edge",
            self.GRANITE_VISION: "Granite Vision",
            self.UNKNOWN: "Unknown VLM",
        }
        return names.get(self, self.value)

    @property
    def supports_modern_conversion(self) -> bool:
        """Check if architecture supports modern convert_hf_to_gguf.py --mmproj."""
        modern_archs = {
            self.QWEN_VL,
            self.QWEN2_VL,
            self.QWEN2_5_VL,
            self.GEMMA3_VISION,
            self.PIXTRAL,
            self.SMOLVLM,
            self.SMOLVLM2,
            self.INTERNVL_2_5,
            self.INTERNVL_3,
            self.MOONDREAM2,
            self.MISTRAL_SMALL_VISION,
        }
        return self in modern_archs

    @property
    def supports_gguf_conversion(self) -> bool:
        """Check if architecture is supported by llama.cpp GGUF converter.

        This dynamically queries llama.cpp's convert_hf_to_gguf.py to determine
        support, avoiding hardcoded lists that become outdated.

        Note: This checks actual llama.cpp support by reading the converter's
        registered model architectures.
        """
        # Import here to avoid circular dependency
        from .llama_cpp_support import LlamaCppVLMSupport

        # Get the HuggingFace architecture pattern for this VLM
        # (defined in vlm_registry.json)
        registry = VLMDetector._load_registry()
        vlm_archs = registry.get("vlm_architectures", {})

        # Map our enum to registry key
        enum_to_key = {
            VLMArchitecture.QWEN_VL: "qwen_vl",
            VLMArchitecture.QWEN2_VL: "qwen2_vl",
            VLMArchitecture.QWEN2_5_VL: "qwen2_5_vl",
            VLMArchitecture.GEMMA3_VISION: "gemma3_vision",
            VLMArchitecture.PIXTRAL: "pixtral",
            VLMArchitecture.SMOLVLM: "smolvlm",
            VLMArchitecture.SMOLVLM2: "smolvlm2",
            VLMArchitecture.INTERNVL_2_5: "internvl_2_5",
            VLMArchitecture.INTERNVL_3: "internvl_3",
            VLMArchitecture.MOONDREAM2: "moondream2",
            VLMArchitecture.MISTRAL_SMALL_VISION: "mistral_small_vision",
            VLMArchitecture.LLAVA_1_5: "llava_1_5",
            VLMArchitecture.LLAVA_1_6: "llava_1_6",
            VLMArchitecture.MINICPM_V_2_5: "minicpm_v_2_5",
            VLMArchitecture.MINICPM_V_2_6: "minicpm_v_2_6",
            VLMArchitecture.MINICPM_O_2_6: "minicpm_o_2_6",
            VLMArchitecture.GLM_EDGE: "glm_edge",
            VLMArchitecture.GRANITE_VISION: "granite_vision",
        }

        arch_key = enum_to_key.get(self)
        if not arch_key:
            return False

        arch_def = vlm_archs.get(arch_key, {})
        architecture_patterns = arch_def.get("architecture_patterns", [])

        if not architecture_patterns:
            return False

        # Get supported architectures from llama.cpp
        supported_archs = LlamaCppVLMSupport.get_supported_vlm_architectures()

        # Check if any of our architecture patterns match llama.cpp's supported list
        for pattern in architecture_patterns:
            if pattern in supported_archs:
                logger.debug(f"✓ {self.display_name} supported: {pattern} in llama.cpp")
                return True

        logger.debug(f"✗ {self.display_name} not supported by llama.cpp")
        return False

    @property
    def needs_legacy_conversion(self) -> bool:
        """Check if architecture needs legacy conversion scripts."""
        legacy_archs = {
            self.LLAVA_1_5,
            self.LLAVA_1_6,
            self.MINICPM_V_2_5,
            self.MINICPM_V_2_6,
            self.MINICPM_O_2_6,
            self.GLM_EDGE,
            self.GRANITE_VISION,
        }
        return self in legacy_archs


@dataclass
class VLMInfo:
    """VLM detection result."""

    is_vlm: bool
    architecture: VLMArchitecture
    language_model_type: Optional[str] = None  # e.g., "qwen2", "llama", "gemma"
    vision_encoder_type: Optional[str] = None  # e.g., "clip", "siglip"
    has_vision_config: bool = False
    has_mmproj: bool = False  # Whether mmproj file exists
    confidence: str = "unknown"  # "high", "medium", "low"


class VLMDetector:
    """Detect VLM architectures from HuggingFace models using dynamic registry."""

    _registry_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def _load_registry(cls) -> Dict[str, Any]:
        """Load VLM registry from JSON file (with caching).

        Returns:
            Dictionary with VLM architecture definitions
        """
        if cls._registry_cache is not None:
            return cls._registry_cache

        # Path to registry file
        registry_path = Path(__file__).parent / "vlm_registry.json"

        try:
            if registry_path.exists():
                with open(registry_path, 'r', encoding='utf-8') as f:
                    cls._registry_cache = json.load(f)
                logger.debug(f"Loaded VLM registry from {registry_path}")
            else:
                logger.warning(f"VLM registry not found at {registry_path}, using fallback")
                cls._registry_cache = {"vlm_architectures": {}, "vision_indicators": {}}

            return cls._registry_cache

        except Exception as e:
            logger.error(f"Error loading VLM registry: {e}")
            return {"vlm_architectures": {}, "vision_indicators": {}}

    @classmethod
    def detect(cls, model_path: Path) -> VLMInfo:
        """Detect if model is a VLM and identify its architecture.

        Args:
            model_path: Path to model directory or config.json

        Returns:
            VLMInfo with detection results
        """
        # Ensure we have a directory path
        if model_path.is_file():
            model_dir = model_path.parent
        else:
            model_dir = model_path

        config_path = model_dir / "config.json"

        if not config_path.exists():
            logger.warning(f"config.json not found in {model_dir}")
            return VLMInfo(
                is_vlm=False,
                architecture=VLMArchitecture.UNKNOWN,
                confidence="low"
            )

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Load registry
            registry = cls._load_registry()

            # Check for vision indicators
            has_vision = cls._has_vision_capability(config, registry)

            if not has_vision:
                return VLMInfo(
                    is_vlm=False,
                    architecture=VLMArchitecture.UNKNOWN,
                    confidence="high"
                )

            # Identify specific architecture using registry
            architecture, confidence = cls._identify_architecture(config, model_dir, registry)

            # Extract additional info
            language_type = cls._extract_language_model_type(config)
            vision_type = cls._extract_vision_encoder_type(config)
            has_vision_config = "vision_config" in config or "vision_tower" in config

            # Check if mmproj file exists
            mmproj_exists = cls._check_mmproj_exists(model_dir)

            # Adjust confidence if no specific architecture found
            if architecture == VLMArchitecture.UNKNOWN:
                confidence = "medium"

            logger.info(f"Detected VLM: {architecture.display_name} (confidence: {confidence})")

            return VLMInfo(
                is_vlm=True,
                architecture=architecture,
                language_model_type=language_type,
                vision_encoder_type=vision_type,
                has_vision_config=has_vision_config,
                has_mmproj=mmproj_exists,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error detecting VLM: {e}")
            return VLMInfo(
                is_vlm=False,
                architecture=VLMArchitecture.UNKNOWN,
                confidence="low"
            )

    @classmethod
    def _has_vision_capability(cls, config: dict, registry: Dict[str, Any]) -> bool:
        """Check if config indicates vision capability using registry.

        Args:
            config: Model config.json
            registry: VLM registry with vision indicators

        Returns:
            True if model has vision capability
        """
        vision_config = registry.get("vision_indicators", {})
        config_keys = vision_config.get("config_keys", [
            "vision_config",
            "vision_tower",
            "mm_vision_tower",
            "image_size",
            "vision_model",
            "visual",
        ])

        # Check config keys
        for indicator in config_keys:
            if indicator in config:
                logger.debug(f"Found vision indicator: {indicator}")
                return True

        # Check model_type for known VLM keywords (including moondream variants)
        model_type = config.get("model_type", "").lower()
        vlm_keywords = vision_config.get("model_type_keywords", ["vl", "vision", "multimodal", "mm", "moondream"])

        for keyword in vlm_keywords:
            if keyword in model_type:
                logger.debug(f"Found VLM keyword '{keyword}' in model_type: {model_type}")
                return True

        # Check architectures for vision patterns AND specific known VLM models
        architectures = config.get("architectures", [])
        for arch in architectures:
            arch_lower = arch.lower()
            # Standard vision patterns
            if any(keyword in arch_lower for keyword in ["vision", "vl", "multimodal", "conditional"]):
                logger.debug(f"Found vision pattern in architecture: {arch}")
                return True
            # Specific VLM architectures that don't follow standard naming
            if any(keyword in arch_lower for keyword in ["moondream", "hfmoondream"]):
                logger.debug(f"Found known VLM architecture: {arch}")
                return True

        return False

    @classmethod
    def _identify_architecture(
        cls, config: dict, model_dir: Path, registry: Dict[str, Any]
    ) -> tuple[VLMArchitecture, str]:
        """Identify specific VLM architecture from config.

        Uses a scoring system to match against known architectures from registry,
        then falls back to extracting architecture name from actual config if no
        match is found. This makes detection dynamic and not reliant on hardcoding.

        Args:
            config: Model config.json
            model_dir: Model directory path
            registry: VLM registry with architecture definitions (optional)

        Returns:
            Tuple of (VLMArchitecture, confidence_level)
        """
        model_type = config.get("model_type", "").lower()
        architectures = [arch.lower() for arch in config.get("architectures", [])]
        model_name = str(model_dir.name).lower()

        vlm_archs = registry.get("vlm_architectures", {})

        # First, try to match against known architectures in registry
        matches = []

        for arch_key, arch_def in vlm_archs.items():
            score = 0
            confidence = "low"

            # Check model_type patterns (highest priority)
            model_type_patterns = arch_def.get("model_type_patterns", [])
            for pattern in model_type_patterns:
                if cls._match_pattern(pattern, model_type):
                    score += 10
                    confidence = "high"
                    logger.debug(f"Matched model_type pattern '{pattern}' for {arch_key}")

            # Check architecture patterns (high priority)
            arch_patterns = arch_def.get("architecture_patterns", [])
            for pattern in arch_patterns:
                for arch in architectures:
                    if cls._match_pattern(pattern, arch):
                        score += 8
                        confidence = "high" if score >= 15 else "medium"
                        logger.debug(f"Matched architecture pattern '{pattern}' for {arch_key}")

            # Check version indicators (medium priority)
            if score > 0:
                version_indicators = arch_def.get("version_indicators", [])
                for indicator in version_indicators:
                    if indicator in model_type or indicator in model_name:
                        score += 3
                        logger.debug(f"Matched version indicator '{indicator}' for {arch_key}")

            # Check aliases in model name (lower priority)
            aliases = arch_def.get("aliases", [])
            for alias in aliases:
                if alias in model_name:
                    score += 5
                    confidence = "high" if score >= 10 else confidence
                    logger.debug(f"Matched alias '{alias}' for {arch_key}")

            if score > 0:
                matches.append((arch_key, score, confidence))

        # If we found a match in registry, use it
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            best_arch_key, _, confidence = matches[0]

            # Convert key to VLMArchitecture enum
            enum_map = {
                "qwen_vl": VLMArchitecture.QWEN_VL,
                "qwen2_vl": VLMArchitecture.QWEN2_VL,
                "qwen2_5_vl": VLMArchitecture.QWEN2_5_VL,
                "gemma3_vision": VLMArchitecture.GEMMA3_VISION,
                "pixtral": VLMArchitecture.PIXTRAL,
                "smolvlm": VLMArchitecture.SMOLVLM,
                "smolvlm2": VLMArchitecture.SMOLVLM2,
                "internvl_2_5": VLMArchitecture.INTERNVL_2_5,
                "internvl_3": VLMArchitecture.INTERNVL_3,
                "moondream2": VLMArchitecture.MOONDREAM2,
                "mistral_small_vision": VLMArchitecture.MISTRAL_SMALL_VISION,
                "llava_1_5": VLMArchitecture.LLAVA_1_5,
                "llava_1_6": VLMArchitecture.LLAVA_1_6,
                "minicpm_v_2_5": VLMArchitecture.MINICPM_V_2_5,
                "minicpm_v_2_6": VLMArchitecture.MINICPM_V_2_6,
                "minicpm_o_2_6": VLMArchitecture.MINICPM_O_2_6,
                "glm_edge": VLMArchitecture.GLM_EDGE,
                "granite_vision": VLMArchitecture.GRANITE_VISION,
            }

            architecture = enum_map.get(best_arch_key, VLMArchitecture.UNKNOWN)
            logger.debug(f"Architecture detection: {best_arch_key} (score from registry)")
            return architecture, confidence

        # Fallback: Extract architecture from the config dynamically
        # This allows detection of unknown VLM architectures based on config structure
        return cls._extract_vlm_architecture_from_config(config, model_type, architectures)

    @staticmethod
    def _extract_vlm_architecture_from_config(
        config: dict, model_type: str, architectures: List[str]
    ) -> tuple[VLMArchitecture, str]:
        """Extract VLM architecture dynamically from config structure.

        When no known architecture matches, derive the VLM type from the config
        structure and naming patterns without relying on hardcoded names.

        Args:
            config: Model config.json
            model_type: Normalized model_type field
            architectures: List of architecture class names

        Returns:
            Tuple of (VLMArchitecture, confidence_level)
        """
        # Extract architecture name from architectures array
        # e.g., "Lfm2VlForConditionalGeneration" → "LFM2-VL"
        if architectures:
            arch_name = architectures[0] if isinstance(architectures, list) else str(architectures)
            arch_name = arch_name.lower()

            # Log the detected architecture name
            logger.info(f"Detected unknown VLM with architecture: {arch_name}")
            logger.debug(f"Model has vision_config and text_config, indicating it's a VLM")
            logger.debug(f"Architecture: {arch_name}")

            # Return UNKNOWN but with medium confidence since we confirmed it's a VLM
            # The actual architecture name is logged for reference
            return VLMArchitecture.UNKNOWN, "medium"

        # If we can't extract architecture name, still return UNKNOWN
        # but the _has_vision_capability check already confirmed it's a VLM
        return VLMArchitecture.UNKNOWN, "low"

    @staticmethod
    def _match_pattern(pattern: str, text: str) -> bool:
        """Match a pattern against text with normalization.

        Args:
            pattern: Pattern to match (can use underscores or dashes)
            text: Text to match against

        Returns:
            True if pattern matches
        """
        # Normalize both pattern and text (replace dashes with underscores)
        norm_pattern = pattern.lower().replace("-", "_").replace(".", "_")
        norm_text = text.lower().replace("-", "_").replace(".", "_")

        return norm_pattern in norm_text

    @staticmethod
    def _extract_language_model_type(config: dict) -> Optional[str]:
        """Extract the language model type (e.g., qwen2, llama, gemma)."""
        model_type = config.get("model_type", "").lower()

        # Extract base language model
        if "qwen" in model_type:
            return "qwen2"
        elif "llama" in model_type or "vicuna" in model_type:
            return "llama"
        elif "gemma" in model_type:
            return "gemma"
        elif "mistral" in model_type:
            return "mistral"
        elif "phi" in model_type:
            return "phi"

        # Check architectures for language model
        architectures = config.get("architectures", [])
        for arch in architectures:
            arch_lower = arch.lower()
            if "qwen" in arch_lower:
                return "qwen2"
            elif "llama" in arch_lower:
                return "llama"
            elif "gemma" in arch_lower:
                return "gemma"

        return None

    @staticmethod
    def _extract_vision_encoder_type(config: dict) -> Optional[str]:
        """Extract the vision encoder type (e.g., clip, siglip)."""
        # Check vision_config for encoder type
        vision_config = config.get("vision_config", {})
        if vision_config:
            model_type = vision_config.get("model_type", "").lower()
            if "clip" in model_type:
                return "clip"
            elif "siglip" in model_type:
                return "siglip"

        # Check for vision tower configuration
        vision_tower = config.get("mm_vision_tower", "") or config.get("vision_tower", "")
        if isinstance(vision_tower, str):
            vision_tower = vision_tower.lower()
            if "clip" in vision_tower:
                return "clip"
            elif "siglip" in vision_tower:
                return "siglip"

        return None

    @staticmethod
    def _check_mmproj_exists(model_dir: Path) -> bool:
        """Check if mmproj file already exists in model directory."""
        # Check for common mmproj filenames
        mmproj_patterns = [
            "mmproj-*.gguf",
            "*-mmproj.gguf",
            "*mmproj*.gguf",
            "*.mmproj",
        ]

        try:
            for pattern in mmproj_patterns:
                if list(model_dir.glob(pattern)):
                    logger.debug(f"Found mmproj file matching pattern: {pattern}")
                    return True
        except Exception as e:
            logger.debug(f"Error checking for mmproj files: {e}")

        return False

    @staticmethod
    def clear_registry_cache() -> None:
        """Clear the cached registry (useful for testing or reloading).

        Use this if you modify the vlm_registry.json file and want to reload it.
        """
        VLMDetector._registry_cache = None
        logger.debug("Cleared VLM registry cache")
