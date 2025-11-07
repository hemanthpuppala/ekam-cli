"""Model-specific method registry for future-proof VLM endpoint support.

This module automatically discovers and registers model-specific methods (e.g., moondream.caption(),
moondream.detect()) and provides a unified interface with intelligent fallbacks.

Production-ready features:
- Auto-discovery via introspection
- Config-based registry for known models
- HuggingFace metadata parsing
- Intelligent fallback chains
- Caching for performance
- Support for new architectures without code changes
"""

import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from loguru import logger


class ModelMethodSignature:
    """Describes a model-specific method's signature and capabilities."""

    def __init__(
        self,
        method_name: str,
        endpoint_type: str,
        required_args: List[str],
        optional_args: Dict[str, Any] = None,
        return_format: str = "text",
        description: str = ""
    ):
        """Initialize method signature.

        Args:
            method_name: Name of the model method (e.g., 'caption', 'detect')
            endpoint_type: Endpoint type (qa, caption, detect, point)
            required_args: List of required argument names
            optional_args: Dict of optional argument names to default values
            return_format: Return format ('text', 'dict', 'list_dict')
            description: Human-readable description
        """
        self.method_name = method_name
        self.endpoint_type = endpoint_type
        self.required_args = required_args
        self.optional_args = optional_args or {}
        self.return_format = return_format
        self.description = description

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "method_name": self.method_name,
            "endpoint_type": self.endpoint_type,
            "required_args": self.required_args,
            "optional_args": self.optional_args,
            "return_format": self.return_format,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMethodSignature":
        """Create from dictionary."""
        return cls(**data)


class ModelMethodRegistry:
    """Registry for discovering and caching model-specific methods.

    Uses intelligent multi-level fallback strategy:
    1. Introspection - Auto-discover methods at runtime
    2. Config registry - Known architecture mappings
    3. HuggingFace metadata - Parse model config/README
    4. Generic fallback - Use standard prompting
    """

    # Registry path for known architecture methods
    REGISTRY_PATH = Path.home() / ".cache" / "vlm-tester" / "model_methods_registry.json"

    # Known architecture method mappings (fallback when introspection unavailable)
    DEFAULT_ARCHITECTURES = {
        "moondream": {
            "caption": ModelMethodSignature(
                method_name="caption",
                endpoint_type="caption",
                required_args=["image"],
                optional_args={"length": "normal", "stream": False},
                return_format="dict",  # Returns {'caption': str}
                description="Moondream optimized caption method"
            ),
            "detect": ModelMethodSignature(
                method_name="detect",
                endpoint_type="detect",
                required_args=["image", "object"],
                optional_args={},
                return_format="dict",  # Returns {'objects': [bbox dicts]}
                description="Moondream bounding box detection"
            ),
            "point": ModelMethodSignature(
                method_name="point",
                endpoint_type="point",
                required_args=["image", "object"],
                optional_args={},
                return_format="dict",  # Returns {'points': [(x, y)]}
                description="Moondream object pointing"
            ),
            "query": ModelMethodSignature(
                method_name="query",
                endpoint_type="qa",
                required_args=["image", "question"],
                optional_args={},
                return_format="dict",  # Returns {'answer': str}
                description="Moondream QA method"
            ),
            # Also has encode_image + answer_question (legacy)
            "answer_question": ModelMethodSignature(
                method_name="answer_question",
                endpoint_type="qa",
                required_args=["image_embeds", "question", "tokenizer"],
                optional_args={},
                return_format="text",
                description="Moondream legacy QA (requires encode_image first)"
            )
        },
        # Qwen/LLaVA/LFM2 use standard transformers - no special methods
        "qwen2_vl": {},
        "qwen2_5_vl": {},
        "llava": {},
        "llava_next": {},
        "lfm2_vl": {},
        "internvl": {},
        "minicpm_v": {},
        "paligemma": {},
    }

    def __init__(self):
        """Initialize registry with persistent cache."""
        self.registry_path = self.REGISTRY_PATH
        self.architectures = self._load_registry()
        self.method_cache = {}  # (model_id, architecture) -> discovered methods

        logger.debug(f"Initialized ModelMethodRegistry with {len(self.architectures)} architectures")

    def _load_registry(self) -> Dict[str, Dict[str, ModelMethodSignature]]:
        """Load registry from disk or create default.

        Returns:
            Dict mapping architecture -> {method_name -> ModelMethodSignature}
        """
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)

                # Deserialize ModelMethodSignature objects
                registry = {}
                for arch, methods_dict in data.items():
                    registry[arch] = {
                        method_name: ModelMethodSignature.from_dict(sig_dict)
                        for method_name, sig_dict in methods_dict.items()
                    }

                logger.debug(f"Loaded method registry from {self.registry_path}")
                return registry

            except Exception as e:
                logger.warning(f"Failed to load method registry: {e}")

        # Create default
        logger.info(f"Creating new method registry at {self.registry_path}")
        default_registry = {}
        for arch, methods in self.DEFAULT_ARCHITECTURES.items():
            if methods:  # Only add if has methods
                default_registry[arch] = methods

        self._save_registry(default_registry)
        return default_registry

    def _save_registry(self, registry: Optional[Dict] = None) -> None:
        """Save registry to disk.

        Args:
            registry: Registry to save (uses self.architectures if None)
        """
        if registry is None:
            registry = self.architectures

        try:
            # Serialize ModelMethodSignature objects
            data = {}
            for arch, methods_dict in registry.items():
                data[arch] = {
                    method_name: sig.to_dict()
                    for method_name, sig in methods_dict.items()
                }

            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)

            logger.debug(f"Saved method registry to {self.registry_path}")

        except Exception as e:
            logger.error(f"Failed to save method registry: {e}")

    def discover_methods(
        self,
        model: Any,
        model_id: str,
        architecture: Optional[str] = None
    ) -> Dict[str, ModelMethodSignature]:
        """Discover model-specific methods using multi-level strategy.

        Strategy order:
        1. Check cache
        2. Introspection (check if model has methods)
        3. Registry lookup by architecture
        4. HuggingFace metadata parsing
        5. Return empty dict (will use generic methods)

        Args:
            model: Loaded model instance
            model_id: Model identifier
            architecture: Optional architecture hint (e.g., 'moondream', 'qwen2_vl')

        Returns:
            Dict mapping endpoint_type -> ModelMethodSignature
        """
        # Check cache first
        cache_key = (model_id, architecture or "unknown")
        if cache_key in self.method_cache:
            logger.debug(f"Using cached methods for {model_id}")
            return self.method_cache[cache_key]

        discovered_methods = {}

        # Strategy 1: Introspection (highest priority)
        logger.debug(f"Strategy 1: Introspecting model {model_id}")
        introspected = self._introspect_model_methods(model)
        if introspected:
            logger.info(f"✓ Discovered {len(introspected)} methods via introspection for {model_id}")
            discovered_methods.update(introspected)

        # Strategy 2: Registry lookup (if architecture provided)
        if architecture and architecture in self.architectures:
            logger.debug(f"Strategy 2: Using registry for architecture '{architecture}'")
            registry_methods = self.architectures[architecture]

            # Merge with introspected (introspection takes priority)
            for method_name, sig in registry_methods.items():
                if sig.endpoint_type not in discovered_methods:
                    discovered_methods[sig.endpoint_type] = sig

            if registry_methods:
                logger.info(f"✓ Found {len(registry_methods)} methods in registry for {architecture}")

        # Strategy 3: HuggingFace metadata (if model has config)
        if not discovered_methods and hasattr(model, 'config'):
            logger.debug(f"Strategy 3: Parsing HuggingFace metadata")
            hf_methods = self._parse_hf_metadata(model, model_id)
            if hf_methods:
                logger.info(f"✓ Discovered {len(hf_methods)} methods from HF metadata")
                discovered_methods.update(hf_methods)

        # Cache results (even if empty)
        self.method_cache[cache_key] = discovered_methods

        if not discovered_methods:
            logger.debug(f"No model-specific methods found for {model_id}, will use generic fallbacks")

        return discovered_methods

    def _introspect_model_methods(self, model: Any) -> Dict[str, ModelMethodSignature]:
        """Introspect model to discover methods.

        Checks for known method names: caption, detect, point, query, etc.

        Args:
            model: Model instance

        Returns:
            Dict mapping endpoint_type -> ModelMethodSignature
        """
        methods = {}

        # Known method patterns to look for
        # expected_args should list all REQUIRED positional arguments
        method_patterns = [
            ("caption", "caption", ["image"]),
            ("detect", "detect", ["image", "object"]),
            ("point", "point", ["image", "object"]),
            ("query", "qa", ["image", "question"]),
            ("answer_question", "qa", ["image_embeds", "question"]),
            ("encode_image", None, ["image"]),  # Helper, not an endpoint
        ]

        for method_name, endpoint_type, expected_args in method_patterns:
            if hasattr(model, method_name):
                method = getattr(model, method_name)

                if callable(method):
                    # Inspect method signature
                    try:
                        sig = inspect.signature(method)
                        params = list(sig.parameters.keys())

                        # Remove 'self' if present
                        if params and params[0] == 'self':
                            params = params[1:]

                        # Create signature (skip if it's just a helper)
                        if endpoint_type:
                            method_sig = ModelMethodSignature(
                                method_name=method_name,
                                endpoint_type=endpoint_type,
                                required_args=params[:len(expected_args)] if params else expected_args,
                                optional_args={p: None for p in params[len(expected_args):]} if len(params) > len(expected_args) else {},
                                return_format="auto",  # Will infer from return value
                                description=f"Auto-discovered {method_name} method"
                            )
                            methods[endpoint_type] = method_sig
                            logger.debug(f"  Found method: {method_name}({', '.join(params)})")

                    except Exception as e:
                        logger.debug(f"Could not inspect {method_name}: {e}")

        return methods

    def _parse_hf_metadata(self, model: Any, model_id: str) -> Dict[str, ModelMethodSignature]:
        """Parse HuggingFace config/README for method information.

        Args:
            model: Model instance with config
            model_id: Model ID

        Returns:
            Dict mapping endpoint_type -> ModelMethodSignature
        """
        methods = {}

        # Try to get architecture from config
        if hasattr(model, 'config'):
            config = model.config

            # Check config.architectures
            if hasattr(config, 'architectures') and config.architectures:
                arch_name = config.architectures[0].lower()

                # Map architecture names to our registry
                arch_mapping = {
                    "moondreammodel": "moondream",
                    "qwen2vlmodel": "qwen2_vl",
                    "qwen2_5vlmodel": "qwen2_5_vl",
                    "llavaforconditionalgeneration": "llava",
                    "llavanextforconditionalgeneration": "llava_next",
                    "lfm2vlforconditionalgeneration": "lfm2_vl",
                }

                for pattern, registry_arch in arch_mapping.items():
                    if pattern in arch_name:
                        if registry_arch in self.architectures:
                            logger.debug(f"Matched architecture {arch_name} -> {registry_arch}")
                            return self.architectures[registry_arch]

        return methods

    def get_method_for_endpoint(
        self,
        endpoint_type: str,
        model: Any,
        model_id: str,
        architecture: Optional[str] = None
    ) -> Optional[Tuple[str, ModelMethodSignature]]:
        """Get model-specific method for an endpoint type.

        Args:
            endpoint_type: Endpoint type (qa, caption, detect, point)
            model: Model instance
            model_id: Model identifier
            architecture: Optional architecture hint

        Returns:
            Tuple of (method_name, signature) if found, None otherwise
        """
        # Discover all methods for this model
        methods = self.discover_methods(model, model_id, architecture)

        if endpoint_type in methods:
            sig = methods[endpoint_type]
            return (sig.method_name, sig)

        return None

    def register_architecture_methods(
        self,
        architecture: str,
        methods: Dict[str, ModelMethodSignature]
    ) -> None:
        """Register methods for a new architecture.

        Args:
            architecture: Architecture name (e.g., 'new_vlm')
            methods: Dict mapping method_name -> ModelMethodSignature
        """
        self.architectures[architecture] = methods
        self._save_registry()
        logger.info(f"Registered {len(methods)} methods for architecture '{architecture}'")


# Global registry instance
_registry_instance: Optional[ModelMethodRegistry] = None


def get_method_registry() -> ModelMethodRegistry:
    """Get global method registry instance.

    Returns:
        ModelMethodRegistry singleton
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelMethodRegistry()
    return _registry_instance
