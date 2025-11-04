"""Model method adapter for unified interface with intelligent fallbacks.

Provides a common interface for calling model-specific methods with automatic
fallback to generic prompting if model-specific methods fail.
"""

from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from loguru import logger

from .model_method_registry import ModelMethodSignature, get_method_registry


class ModelMethodAdapter:
    """Adapter for calling model-specific methods with fallbacks.

    Provides unified interface:
    - adapter.caption(model, image, **kwargs) -> str
    - adapter.detect(model, image, object_name, **kwargs) -> List[dict]
    - adapter.point(model, image, object_name, **kwargs) -> dict
    - adapter.qa(model, image, question, **kwargs) -> str

    With intelligent fallback chain:
    1. Try model-specific method (e.g., model.caption())
    2. Fall back to generic prompting (via callback)
    3. Log warnings and errors appropriately
    """

    def __init__(self, model: Any, model_id: str, architecture: Optional[str] = None):
        """Initialize adapter.

        Args:
            model: Loaded model instance
            model_id: Model identifier
            architecture: Optional architecture hint
        """
        self.model = model
        self.model_id = model_id
        self.architecture = architecture
        self.registry = get_method_registry()

        # Discover methods once at initialization
        self.available_methods = self.registry.discover_methods(model, model_id, architecture)

        if self.available_methods:
            logger.info(
                f"ModelMethodAdapter initialized for {model_id}: "
                f"{len(self.available_methods)} native methods available "
                f"({', '.join(self.available_methods.keys())})"
            )
        else:
            logger.debug(f"ModelMethodAdapter initialized for {model_id}: using generic methods only")

    def _has_native_method(self, endpoint_type: str) -> bool:
        """Check if model has native method for endpoint."""
        return endpoint_type in self.available_methods

    def _call_native_method(
        self,
        endpoint_type: str,
        **kwargs
    ) -> Optional[Any]:
        """Call model's native method with proper argument mapping.

        Args:
            endpoint_type: Endpoint type (qa, caption, detect, point)
            **kwargs: Arguments to pass

        Returns:
            Method result or None if method doesn't exist or fails
        """
        if endpoint_type not in self.available_methods:
            return None

        signature = self.available_methods[endpoint_type]
        method_name = signature.method_name

        if not hasattr(self.model, method_name):
            logger.warning(f"Method {method_name} not found on model (cache stale?)")
            return None

        method = getattr(self.model, method_name)

        try:
            # Map kwargs to method's expected arguments
            method_args = self._map_args_to_method(signature, kwargs)

            # Moondream requires positional arguments, not keyword arguments
            # Check if this is a moondream model
            is_moondream = "moondream" in self.model_id.lower()

            if is_moondream and signature.required_args:
                # Pass arguments positionally in the order of required_args
                positional_args = []
                for arg_name in signature.required_args:
                    if arg_name in method_args:
                        positional_args.append(method_args[arg_name])

                logger.debug(f"Calling native method (positional): {method_name}(<{len(positional_args)} args>)")
                result = method(*positional_args)
            else:
                # Standard keyword argument call
                logger.debug(f"Calling native method: {method_name}({list(method_args.keys())})")
                result = method(**method_args)

            # Log raw result for debugging (without truncation)
            logger.debug(f"Native method {method_name} raw result type: {type(result)}")
            if isinstance(result, dict):
                logger.debug(f"Native method {method_name} raw result keys: {list(result.keys())}")
                logger.debug(f"Native method {method_name} raw result (FULL): {result}")
            elif isinstance(result, list):
                logger.debug(f"Native method {method_name} raw result length: {len(result)}")
                logger.debug(f"Native method {method_name} raw result (FULL): {result}")
            else:
                logger.debug(f"Native method {method_name} raw result (FULL): {result}")

            # Extract result based on return format
            if signature.return_format == "dict":
                # Methods like moondream.caption() return {'caption': str}
                extracted = self._extract_from_dict_result(result, endpoint_type)
                logger.debug(f"Extracted value type: {type(extracted)}, value: {extracted}")
                return extracted
            else:
                # Direct return
                return result

        except Exception as e:
            logger.warning(
                f"Native method {method_name} failed: {e}. "
                f"Will fall back to generic method."
            )
            logger.debug(f"Native method error details:", exc_info=True)
            return None

    def _map_args_to_method(
        self,
        signature: ModelMethodSignature,
        provided_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map provided arguments to method's expected arguments.

        Args:
            signature: Method signature
            provided_args: Arguments provided by caller

        Returns:
            Dict of arguments to pass to method
        """
        method_args = {}

        # Map required args
        for req_arg in signature.required_args:
            # Handle common argument name variations
            arg_value = None

            if req_arg in provided_args:
                arg_value = provided_args[req_arg]
            elif req_arg == "object" and "object_name" in provided_args:
                arg_value = provided_args["object_name"]
            elif req_arg == "object_name" and "object" in provided_args:
                arg_value = provided_args["object"]
            elif req_arg == "question" and "prompt" in provided_args:
                arg_value = provided_args["prompt"]
            elif req_arg == "prompt" and "question" in provided_args:
                arg_value = provided_args["question"]

            if arg_value is not None:
                method_args[req_arg] = arg_value
            else:
                logger.debug(f"Required argument '{req_arg}' not provided, method may fail")

        # Add optional args if provided
        for opt_arg, default_value in signature.optional_args.items():
            if opt_arg in provided_args:
                method_args[opt_arg] = provided_args[opt_arg]
            # Note: We don't add defaults here, let the method handle it

        # Pass through any additional kwargs not in signature
        # (hybrid approach - allow model-specific extras)
        for key, value in provided_args.items():
            if key not in method_args and key not in ["fallback_callback"]:
                method_args[key] = value

        return method_args

    def _convert_moondream_detections(self, detections: List[dict], object_name: str) -> List[dict]:
        """Convert moondream detection format to standard format.

        Moondream returns: {"x_min": 0.2, "y_min": 0.3, "x_max": 0.4, "y_max": 0.8}
        Standard format: {"label": "person", "bbox": [0.2, 0.3, 0.4, 0.8], "confidence": 0.9}

        Args:
            detections: List of moondream detection dicts
            object_name: Object being detected (for label)

        Returns:
            List of detections in standard format
        """
        converted = []

        for idx, det in enumerate(detections):
            if not isinstance(det, dict):
                logger.warning(f"Detection {idx} is not a dict: {det}")
                continue

            # Check if this is moondream format
            if 'x_min' in det and 'y_min' in det and 'x_max' in det and 'y_max' in det:
                # Moondream format - convert to standard
                bbox = [det['x_min'], det['y_min'], det['x_max'], det['y_max']]
                label = det.get('label', object_name)  # Moondream doesn't provide labels
                confidence = det.get('confidence', 0.9)

                logger.debug(f"Converting moondream detection {idx + 1}: bbox={bbox}, label={label}")

                converted.append({
                    'label': label,
                    'bbox': bbox,
                    'confidence': confidence
                })
            elif 'bbox' in det:
                # Already in standard format
                logger.debug(f"Detection {idx + 1} already in standard format")
                converted.append(det)
            else:
                logger.warning(f"Unknown detection format for detection {idx + 1}: {det}")

        logger.info(f"Converted {len(converted)}/{len(detections)} moondream detections to standard format")
        return converted

    def _extract_from_dict_result(self, result: Any, endpoint_type: str) -> Any:
        """Extract value from dict result.

        Args:
            result: Result from method call
            endpoint_type: Endpoint type

        Returns:
            Extracted value
        """
        if not isinstance(result, dict):
            return result

        # Common dict key mappings
        key_mappings = {
            "caption": ["caption", "text", "output"],
            "qa": ["answer", "response", "output", "text"],
            "detect": ["objects", "detections", "bboxes"],
            "point": ["points", "coordinates", "point"]
        }

        if endpoint_type in key_mappings:
            for key in key_mappings[endpoint_type]:
                if key in result:
                    extracted = result[key]

                    # For point endpoint, if we get an array, take the first point
                    if endpoint_type == "point" and isinstance(extracted, list) and len(extracted) > 0:
                        logger.debug(f"Extracted first point from array of {len(extracted)} points")
                        return extracted[0]

                    return extracted

        # If no known key, return whole dict
        logger.debug(f"Could not extract value from dict result, returning whole dict")
        return result

    def caption(
        self,
        image: Image.Image,
        fallback_callback: Optional[callable] = None,
        **kwargs
    ) -> str:
        """Generate image caption using native method or fallback.

        Args:
            image: PIL Image
            fallback_callback: Function to call if native method fails
            **kwargs: Additional arguments (e.g., length='short' for moondream)

        Returns:
            Caption text
        """
        # Try native method first
        if self._has_native_method("caption"):
            logger.debug("Using native caption method")
            result = self._call_native_method("caption", image=image, **kwargs)

            if result is not None:
                logger.debug("✓ Native caption method succeeded")
                return str(result)

        # Fallback to generic
        if fallback_callback:
            logger.debug("Using fallback callback for caption")
            return fallback_callback(image=image, **kwargs)

        raise RuntimeError("No caption method available and no fallback provided")

    def detect(
        self,
        image: Image.Image,
        object_name: str,
        fallback_callback: Optional[callable] = None,
        **kwargs
    ) -> List[dict]:
        """Detect objects using native method or fallback.

        Args:
            image: PIL Image
            object_name: Object to detect
            fallback_callback: Function to call if native method fails
            **kwargs: Additional arguments

        Returns:
            List of detection dicts with bboxes
        """
        # Try native method
        if self._has_native_method("detect"):
            logger.debug("Using native detect method")
            result = self._call_native_method(
                "detect",
                image=image,
                object=object_name,
                object_name=object_name,
                **kwargs
            )

            if result is not None:
                logger.debug("✓ Native detect method succeeded")
                logger.debug(f"Detect result type after extraction: {type(result)}")
                logger.debug(f"Detect result value (FULL): {result}")

                # Ensure it's a list and convert moondream format if needed
                detections = []
                if isinstance(result, list):
                    detections = result
                elif isinstance(result, dict) and "objects" in result:
                    detections = result["objects"]
                else:
                    logger.warning(f"Unexpected detect result format: {type(result)}, wrapping in list")
                    detections = [result] if result else []

                # Convert moondream format (x_min, y_min, x_max, y_max) to standard format
                converted_detections = self._convert_moondream_detections(detections, object_name)
                logger.info(f"Returning {len(converted_detections)} detections from native method")
                logger.debug(f"Final converted detections: {converted_detections}")
                return converted_detections

        # Fallback
        if fallback_callback:
            logger.debug("Using fallback callback for detect")
            return fallback_callback(image=image, object_name=object_name, **kwargs)

        raise RuntimeError("No detect method available and no fallback provided")

    def point(
        self,
        image: Image.Image,
        object_name: str,
        fallback_callback: Optional[callable] = None,
        **kwargs
    ) -> dict:
        """Point to object using native method or fallback.

        Args:
            image: PIL Image
            object_name: Object to locate
            fallback_callback: Function to call if native method fails
            **kwargs: Additional arguments

        Returns:
            Dict with coordinates (x, y) or bbox
        """
        # Try native method
        if self._has_native_method("point"):
            logger.debug("Using native point method")
            result = self._call_native_method(
                "point",
                image=image,
                object=object_name,
                object_name=object_name,
                **kwargs
            )

            if result is not None:
                logger.debug("✓ Native point method succeeded")
                return result if isinstance(result, dict) else {"result": result}

        # Fallback
        if fallback_callback:
            logger.debug("Using fallback callback for point")
            return fallback_callback(image=image, object_name=object_name, **kwargs)

        raise RuntimeError("No point method available and no fallback provided")

    def qa(
        self,
        image: Image.Image,
        question: str,
        fallback_callback: Optional[callable] = None,
        **kwargs
    ) -> str:
        """Answer question using native method or fallback.

        Args:
            image: PIL Image
            question: Question text
            fallback_callback: Function to call if native method fails
            **kwargs: Additional arguments

        Returns:
            Answer text
        """
        # Try native method
        if self._has_native_method("qa"):
            logger.debug("Using native QA method")
            result = self._call_native_method(
                "qa",
                image=image,
                question=question,
                prompt=question,
                **kwargs
            )

            if result is not None:
                logger.debug("✓ Native QA method succeeded")
                return str(result)

        # Fallback
        if fallback_callback:
            logger.debug("Using fallback callback for QA")
            return fallback_callback(image=image, question=question, **kwargs)

        raise RuntimeError("No QA method available and no fallback provided")

    def has_native_method_for(self, endpoint_type: str) -> bool:
        """Check if model has native method for endpoint.

        Args:
            endpoint_type: Endpoint type (qa, caption, detect, point)

        Returns:
            True if native method available
        """
        return self._has_native_method(endpoint_type)

    def get_available_endpoints(self) -> List[str]:
        """Get list of available endpoint types.

        Returns:
            List of endpoint type strings
        """
        return list(self.available_methods.keys())
