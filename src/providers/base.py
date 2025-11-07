"""Base provider interface for all model providers."""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from ..models.model import ModelInfo


class BaseProvider(ABC):
    """Abstract base class for all model providers.

    All providers must implement this interface to ensure consistent
    behavior across Ollama, HuggingFace, LM Studio, and GGUF providers.
    """

    @abstractmethod
    def discover_models(self) -> list[ModelInfo]:
        """Discover available models from this provider.

        Returns:
            List of ModelInfo objects with capabilities and metadata
        """
        pass

    @abstractmethod
    def load_model(self, model_id: str, device: str) -> Any:
        """Load model into memory on specified device.

        Args:
            model_id: Unique model identifier
            device: Target device ("cuda", "mps", or "cpu")

        Returns:
            Provider-specific model handle

        Raises:
            RuntimeError: If model cannot be loaded
        """
        pass

    @abstractmethod
    def unload_model(self, handle: Any) -> None:
        """Unload model from memory and free resources.

        Args:
            handle: Provider-specific model handle from load_model()
        """
        pass

    # VLM Endpoints (vision + language)

    @abstractmethod
    def run_qa(self, handle: Any, image: Image.Image, question: str) -> str:
        """Run question answering on image.

        Args:
            handle: Loaded model handle
            image: PIL Image to analyze
            question: Question to answer about image

        Returns:
            Text answer to question

        Raises:
            NotImplementedError: If model doesn't support QA endpoint
        """
        pass

    @abstractmethod
    def run_caption(self, handle: Any, image: Image.Image, detail_level: str = "detailed") -> str:
        """Generate image caption.

        Args:
            handle: Loaded model handle
            image: PIL Image to caption
            detail_level: "detailed" or "short"

        Returns:
            Text caption describing image

        Raises:
            NotImplementedError: If model doesn't support caption endpoint
        """
        pass

    @abstractmethod
    def run_detect(self, handle: Any, image: Image.Image, object_name: str) -> list[dict]:
        """Detect objects in image.

        Args:
            handle: Loaded model handle
            image: PIL Image to analyze
            object_name: Object to detect (e.g., "car", "person")

        Returns:
            List of detections: [{"label": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]

        Raises:
            NotImplementedError: If model doesn't support detect endpoint
        """
        pass

    @abstractmethod
    def run_point(self, handle: Any, image: Image.Image, object_name: str) -> dict:
        """Point to object center in image.

        Args:
            handle: Loaded model handle
            image: PIL Image to analyze
            object_name: Object to locate (e.g., "sign", "door")

        Returns:
            Coordinates dict: {"x": int, "y": int} or {"bbox": [x1,y1,x2,y2]}

        Raises:
            NotImplementedError: If model doesn't support point endpoint
        """
        pass

    # LLM Endpoint (text-only)

    @abstractmethod
    def run_text(self, handle: Any, prompt: str) -> str:
        """Generate text response to prompt.

        Args:
            handle: Loaded model handle
            prompt: Text prompt for generation

        Returns:
            Generated text response

        Raises:
            NotImplementedError: If model doesn't support text endpoint
        """
        pass

    # Model Management

    @abstractmethod
    def install_model(self, model_name: str) -> bool:
        """Install/download model from provider.

        Args:
            model_name: Model name to install

        Returns:
            True if installation succeeded

        Raises:
            NotImplementedError: If provider doesn't support installation
        """
        pass

    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Delete model from provider storage.

        Args:
            model_id: Model identifier to delete

        Returns:
            True if deletion succeeded

        Raises:
            NotImplementedError: If provider doesn't support deletion
        """
        pass

    @abstractmethod
    def estimate_model_size(self, model_name: str) -> float:
        """Estimate model size before download.

        Args:
            model_name: Model name to check

        Returns:
            Estimated size in GB

        Raises:
            NotImplementedError: If provider doesn't support size estimation
        """
        pass
