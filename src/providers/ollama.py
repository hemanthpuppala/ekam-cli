"""Ollama provider implementation using REST API."""

import base64
import io
import json
from typing import Any, Callable, Optional

import httpx
from loguru import logger
from PIL import Image

from ..models.endpoints import CompatibilityStatus, EndpointType, ModelType, ProviderType
from ..models.model import ModelInfo
from ..models.model_cache import ModelMetadataCache
from ..models.provider import ProviderConfig
from ..models.system import SystemSpecs
from ..utils.history_formatter import format_qa_history
from ..utils.ollama_file_locator import OllamaFileLocator
from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """Ollama provider using HTTP API."""

    def __init__(self, config: ProviderConfig, system_specs: Optional['SystemSpecs'] = None):
        """Initialize Ollama provider.

        Args:
            config: Provider configuration with host and timeout settings
            system_specs: System specifications for dynamic GPU/CPU detection
        """
        self.config = config
        self.base_url = str(config.host).rstrip("/")
        self.timeout = config.timeout_seconds
        self.client = httpx.Client(timeout=self.timeout)

        # Store or detect system specs
        self.system_specs = system_specs or SystemSpecs.detect()

        # Initialize metadata cache for efficient model inspection
        self.metadata_cache = ModelMetadataCache()
        logger.debug("Initialized model metadata cache")

        # Initialize file locator for quantization support
        self.file_locator = OllamaFileLocator()
        logger.debug("Initialized Ollama file locator for quantization")

        # Per-model context length overrides (num_ctx)
        self._ctx_overrides: dict[str, int] = {}

    def discover_models(self) -> list[ModelInfo]:
        """Discover available models from Ollama server.

        Returns:
            List of ModelInfo objects with capabilities

        Raises:
            RuntimeError: If Ollama server is not accessible
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            models = []
            for model_data in data.get("models", []):
                model_info = self._parse_model_info(model_data)
                models.append(model_info)

            logger.info(f"Discovered {len(models)} models from Ollama")
            return models

        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise RuntimeError(
                f"Ollama not accessible at {self.base_url}. "
                "Run `ollama serve` or visit https://ollama.ai/docs"
            )
        except Exception as e:
            logger.error(f"Error discovering Ollama models: {e}")
            raise RuntimeError(f"Failed to discover Ollama models: {e}")

    def _parse_model_info(self, model_data: dict) -> ModelInfo:
        """Parse Ollama model data into ModelInfo using metadata cache.

        Args:
            model_data: Raw model data from Ollama API

        Returns:
            ModelInfo instance with capabilities
        """
        name = model_data.get("name", "unknown")

        # Get metadata from cache (uses ollama show --json command)
        metadata = self.metadata_cache.get_metadata(name, provider="ollama")

        if metadata:
            # Map metadata model_type to ModelType enum
            if metadata.model_type == "vlm":
                model_type = ModelType.VLM
                capabilities = [
                    EndpointType.QA,
                    EndpointType.CAPTION,
                    EndpointType.DETECT,
                    EndpointType.POINT,
                    EndpointType.TEXT,
                ]
            elif metadata.model_type == "embedding":
                model_type = ModelType.EMBEDDING
                capabilities = []
            else:  # llm or unknown
                model_type = ModelType.LLM
                capabilities = [EndpointType.TEXT]

            # Get source path for quantization support
            source_path = self.file_locator.get_model_path(name)
            if source_path:
                logger.debug(f"Located source file for {name}: {source_path}")
            else:
                logger.debug(f"Could not locate source file for {name}")

            # Create ModelInfo with metadata
            return ModelInfo(
                model_id=name,
                name=name,
                provider=ProviderType.OLLAMA,
                size_gb=metadata.file_size_gb,
                architecture=metadata.architecture,
                quantization=metadata.quantization,
                params_billions=metadata.params_billions,
                ram_gb=metadata.ram_gb,
                vram_gb=metadata.vram_gb,
                model_type=model_type,
                capabilities=capabilities,
                compatibility=CompatibilityStatus.PERFECT_FIT,
                compatibility_message="Compatibility not yet assessed",
                is_installed=True,
                source_path=source_path,
            )
        else:
            # Fallback: metadata inspection failed, use API introspection
            logger.warning(f"Could not get metadata for {name}, using API fallback")
            size_bytes = model_data.get("size", 0)
            size_gb = size_bytes / (1024**3)
            model_type, capabilities = self._classify_model_dynamic(name)

            # Get source path for quantization support (even in fallback)
            source_path = self.file_locator.get_model_path(name)

            return ModelInfo(
                model_id=name,
                name=name,
                provider=ProviderType.OLLAMA,
                size_gb=size_gb,
                model_type=model_type,
                capabilities=capabilities,
                compatibility=CompatibilityStatus.PERFECT_FIT,
                compatibility_message="Compatibility not yet assessed",
                is_installed=True,
                source_path=source_path,
            )

    def _classify_model_dynamic(self, name: str) -> tuple[ModelType, list[EndpointType]]:
        """Dynamically classify model type by inspecting model details via API.

        This method queries the Ollama API to get model metadata and detect
        vision capabilities automatically, rather than relying on manual keywords.

        Args:
            name: Model name (e.g., "llava:7b", "llama3.2:3b", "moondream:latest")

        Returns:
            (ModelType, list of EndpointType)
        """
        try:
            # Query Ollama API for detailed model information
            response = self.client.post(
                f"{self.base_url}/api/show",
                json={"name": name},
                timeout=5.0
            )
            response.raise_for_status()
            model_details = response.json()

            # Check for embedding models FIRST (they may contain vision-like keywords)
            if self._is_embedding_model(model_details, name):
                logger.info(f"Detected Embedding model: {name}")
                return (ModelType.EMBEDDING, [])

            # Check for vision capabilities with strict validation
            has_vision = self._detect_vision_capability(model_details, name)

            if has_vision:
                logger.info(f"Detected VLM: {name}")
                return (
                    ModelType.VLM,
                    [
                        EndpointType.QA,
                        EndpointType.CAPTION,
                        EndpointType.DETECT,
                        EndpointType.POINT,
                        EndpointType.TEXT,
                    ],
                )

            # Default to text-only LLM
            logger.info(f"Detected LLM: {name}")
            return (ModelType.LLM, [EndpointType.TEXT])

        except Exception as e:
            # If API call fails, fall back to keyword-based classification
            logger.warning(f"API introspection failed for {name}: {e}, falling back to keywords")
            return self._classify_model_fallback(name)

    def _detect_vision_capability(self, model_details: dict, name: str) -> bool:
        """Detect if a model has vision capabilities from its metadata.

        Args:
            model_details: Model details from /api/show endpoint
            name: Model name for exclusion checks

        Returns:
            True if model supports vision inputs
        """
        name_lower = name.lower()

        # STEP 1: Check if name explicitly indicates VLM (highest priority)
        vlm_name_patterns = [
            "llava", "moondream", "bakllava", "qwen2.5-vl", "qwen-vl", "qwenvl",
            "qwen2.5vl", "minicpm-v", "cogvlm", "internvl", "fuyu", "kosmos",
            "paligemma", "idefics", "instructblip", "llava-next", "phi-3-vision",
            "llama3.2-vision"
        ]

        for pattern in vlm_name_patterns:
            if pattern in name_lower:
                logger.debug(f"VLM detected via name pattern: {pattern}")
                return True

        # STEP 2: EXCLUSION - Known non-VLM patterns (check AFTER positive patterns)
        non_vlm_patterns = [
            "coder",      # qwen2.5-coder, codellama, etc.
            "embed",      # mxbai-embed, nomic-embed, etc.
            "qwen2.5:",   # qwen2.5:3b is LLM, qwen2.5-vl is VLM
            "gemma",      # Pure LLM (unless explicitly vision)
            "deepseek",   # Pure LLM
            "mistral:",   # Pure LLM (mistral-vision would match above)
        ]

        for pattern in non_vlm_patterns:
            if pattern in name_lower:
                logger.debug(f"Excluded from VLM: {name} (pattern: {pattern})")
                return False

        # Method 1: Check template for EXPLICIT image placeholders
        template = model_details.get("template", "")
        vision_template_indicators = [
            "[img]",           # LLaVA style
            "<image>",         # XML-style image tags
            "{{.Image}}",      # Ollama template variable for images
            "<|image|>",       # Special image tokens
            "<|vision|>",      # Special vision tokens
            "[IMAGE]",         # Image markers
            "{{image}}",       # Image variables
        ]

        if any(indicator in template for indicator in vision_template_indicators):
            logger.debug(f"Vision detected via template indicators")
            return True

        # Method 2: Check model family/architecture (most reliable)
        model_info = model_details.get("model_info", {})
        family = model_info.get("general.architecture", "").lower()
        vision_families = [
            "llava",
            "moondream",
            "bakllava",
            "cogvlm",
            "internvl",
            "qwen2vl",        # Note: qwen2vl, not qwen2
            "minicpm-v",
            "paligemma",
        ]

        if any(fam in family for fam in vision_families):
            logger.debug(f"Vision detected via family: {family}")
            return True

        # Method 3: Check for vision projection layers (hardware evidence)
        details_str = str(model_details).lower()
        projection_keywords = ["vision_proj", "mm_projector", "visual_proj", "image_encoder"]

        if any(keyword in details_str for keyword in projection_keywords):
            logger.debug(f"Vision detected via projection layers")
            return True

        # Method 4: Check modelfile ONLY for specific multimodal indicators
        # (disabled by default as it's too aggressive)
        # modelfile = model_details.get("modelfile", "")
        # if "multimodal" in modelfile.lower():  # Very specific keyword
        #     return True

        return False

    def _is_embedding_model(self, model_details: dict, name: str) -> bool:
        """Check if model is an embedding model.

        Args:
            model_details: Model details from API
            name: Model name

        Returns:
            True if model is for embeddings
        """
        # Check model info
        model_info = model_details.get("model_info", {})
        family = model_info.get("general.architecture", "").lower()

        if "embed" in family or "embed" in name.lower():
            return True

        # Check modelfile for embedding-specific parameters
        modelfile = model_details.get("modelfile", "").lower()
        if "embedding" in modelfile and "text generation" not in modelfile:
            return True

        return False

    def _classify_model_fallback(self, name: str) -> tuple[ModelType, list[EndpointType]]:
        """Fallback keyword-based classification when API introspection fails.

        Args:
            name: Model name

        Returns:
            (ModelType, list of EndpointType)
        """
        name_lower = name.lower()

        # Vision-language models - comprehensive keyword list
        vlm_keywords = [
            "llava", "vision", "clip", "moondream", "bakllava",
            "qwen2.5-vl", "qwen-vl", "qwenvl", "minicpm-v",
            "cogvlm", "internvl", "fuyu", "kosmos", "blip",
            "paligemma", "idefics", "instructblip", "llava-next"
        ]

        if any(keyword in name_lower for keyword in vlm_keywords):
            return (
                ModelType.VLM,
                [
                    EndpointType.QA,
                    EndpointType.CAPTION,
                    EndpointType.DETECT,
                    EndpointType.POINT,
                    EndpointType.TEXT,
                ],
            )

        # Embedding models
        if "embed" in name_lower:
            return (ModelType.EMBEDDING, [])

        # Default to text-only LLM
        return (ModelType.LLM, [EndpointType.TEXT])

    def load_model(self, model_id: str, device: str) -> Any:
        """Load model (Ollama manages model loading internally).

        Args:
            model_id: Model identifier
            device: Ignored (Ollama manages devices)

        Returns:
            Model identifier as handle
        """
        logger.info(f"Loading Ollama model: {model_id}")
        # Prompt for context length (num_ctx) before first use for this model
        if model_id not in self._ctx_overrides:
            from ..cli.text_input import professional_prompt
            from ..cli.tui_manager import tui
            min_ctx, max_ctx = 256, 32768
            default_ctx = 4096
            recommended_ctx = 4096

            # Clear any prior panels (e.g., global loading) before prompting
            tui.clear_screen()

            import os
            min_ctx, max_ctx = 256, 32768
            default_ctx = 4096
            n_ctx = default_ctx
            try:
                env_ctx = os.getenv("EKAM_N_CTX")
                if env_ctx:
                    n_ctx = max(min_ctx, min(max_ctx, int(env_ctx)))
                if not os.getenv("EKAM_BENCHMARK_MODE"):
                    tui.console.print("\n[cyan]Context Window (n_ctx)[/cyan] — tokens kept in memory per request")
                    tui.console.print(
                        f"[dim]Min:[/dim] {min_ctx}   [dim]Max:[/dim] {max_ctx}   "
                        f"[dim]Default:[/dim] {default_ctx}   [dim]Recommended:[/dim] {default_ctx}"
                    )
                    tui.console.print("[dim]Higher values increase RAM/VRAM usage and may reduce throughput.[/dim]")
                    n_ctx = professional_prompt.get_numeric(
                        "Context length (tokens)",
                        min_val=min_ctx,
                        max_val=max_ctx,
                        default=default_ctx,
                        style="cyan",
                    )
            except Exception:
                n_ctx = default_ctx
            logger.info(f"Using Ollama num_ctx: {n_ctx}")
            self._ctx_overrides[model_id] = n_ctx
            # Clear and show a compact loading box after prompt
            tui.clear_screen()
            if not os.getenv("EKAM_BENCHMARK_MODE"):
                tui.show_message(
                    f"Loading model: {model_id}\n\n[dim]Please wait...[/dim]",
                    title="Loading",
                    style="cyan",
                )

        # Ollama loads models on-demand during inference
        # Return model_id as handle for tracking
        return model_id

    def unload_model(self, handle: Any) -> None:
        """Unload model (Ollama manages unloading automatically).

        Args:
            handle: Model identifier from load_model()
        """
        logger.info(f"Unloading Ollama model: {handle}")
        # Ollama unloads models automatically after idle timeout
        pass

    def get_model_info(self, model_id: str) -> dict:
        """Get detailed model information from Ollama.

        Args:
            model_id: Model identifier

        Returns:
            Dict with comprehensive model metadata including:
            - model_id, name, provider
            - architecture, family, quantization
            - parameter_count, size_gb
            - license
            - default_parameters (temperature, etc.)
            - chat_template
            - capabilities
        """
        try:
            # Query Ollama API for detailed model information
            response = self.client.post(
                f"{self.base_url}/api/show",
                json={"name": model_id},
                timeout=10.0
            )
            response.raise_for_status()
            details = response.json()

            # Extract model info
            model_info_data = details.get("model_info", {})

            # Parse quantization from name (e.g., "llama3.2:3b-q4" -> "Q4")
            quantization = "Unknown"
            name_lower = model_id.lower()
            if "q4" in name_lower:
                quantization = "Q4_K_M" if "q4_k_m" in name_lower else "Q4_0"
            elif "q5" in name_lower:
                quantization = "Q5_K_M"
            elif "q8" in name_lower:
                quantization = "Q8_0"
            elif "fp16" in name_lower:
                quantization = "FP16"

            # Extract parameter count from name (e.g., "3b", "7b")
            param_count = "Unknown"
            for size in ["0.5b", "1b", "1.5b", "3b", "7b", "11b", "13b", "30b", "70b"]:
                if size in name_lower:
                    param_count = size.upper()
                    break

            # Build comprehensive info dict
            info = {
                "model_id": model_id,
                "name": model_id,
                "provider": "ollama",
                "architecture": model_info_data.get("general.architecture", "Unknown"),
                "family": details.get("details", {}).get("family", "Unknown"),
                "format": details.get("details", {}).get("format", "Unknown"),
                "parameter_count": param_count,
                "quantization": quantization,
                "size_gb": details.get("size", 0) / (1024**3),
                "license": model_info_data.get("general.license", "Unknown"),
                "model_type": details.get("details", {}).get("family", "llm").upper(),
                "chat_template": details.get("template", "N/A"),
                "capabilities": self._get_capabilities_from_details(details, model_id),
                "default_parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": 1024,  # Increased for more detailed responses
                    "repeat_penalty": 1.1,
                }
            }

            return info

        except Exception as e:
            logger.error(f"Failed to get model info for {model_id}: {e}")
            # Return minimal info on error
            return {
                "model_id": model_id,
                "name": model_id,
                "provider": "ollama",
                "architecture": "Unknown",
                "error": str(e)
            }

    def _get_capabilities_from_details(self, details: dict, model_id: str) -> list[str]:
        """Extract capabilities from model details.

        Args:
            details: Model details from /api/show
            model_id: Model identifier

        Returns:
            List of capability strings
        """
        capabilities = []

        # Check for vision
        if self._detect_vision_capability(details, model_id):
            capabilities.extend(["vision", "image-qa", "image-caption"])
        else:
            capabilities.append("text-only")

        # Check for embedding
        if self._is_embedding_model(details, model_id):
            capabilities.append("embeddings")
        else:
            capabilities.append("text-generation")

        # Check for chat support (most models)
        if "chat" in model_id.lower() or details.get("template"):
            capabilities.append("chat")

        # Check for instruct/completion
        if "instruct" in model_id.lower():
            capabilities.append("instruction-following")

        return capabilities

    def run_qa(
        self,
        handle: Any,
        image: Image.Image,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        stream_callback: Optional[callable] = None,
    ) -> str:
        """Run question answering on image using Ollama with unified history.

        Args:
            handle: Model identifier
            image: PIL Image to analyze
            question: Question to answer
            conversation_history: Optional list of (question, answer) tuples
            stream_callback: Optional callback for token streaming (delta: str, is_first: bool)

        Returns:
            Text answer
        """
        model_id = handle
        image_b64 = self._encode_image(image)

        # Use unified Q&A history formatter (shared across all providers)
        formatted_question = format_qa_history(
            conversation_history=conversation_history,
            current_question=question,
            max_turns=5
        )

        payload = {
            "model": model_id,
            "prompt": formatted_question,
            "images": [image_b64],
            "stream": bool(stream_callback),  # Enable streaming if callback provided
        }

        if stream_callback:
            # Streaming mode
            full_response = ""
            with self.client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                response.raise_for_status()

                is_first_token = True
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            delta = chunk.get("response", "")
                            if delta:
                                full_response += delta
                                stream_callback(delta, is_first=is_first_token)
                                is_first_token = False

                            # Check if done
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

            return full_response.strip()
        else:
            # Non-streaming mode
            response = self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()

            return result.get("response", "").strip()

    def run_caption(
        self,
        handle: Any,
        image: Image.Image,
        detail_level: str = "detailed",
        conversation_history: Optional[list[tuple[str, str]]] = None,
        stream_callback: Optional[callable] = None,
    ) -> str:
        """Generate image caption using Ollama with unified history.

        Args:
            handle: Model identifier
            image: PIL Image to caption
            detail_level: "detailed" or "short"
            conversation_history: Optional list of (prompt, response) tuples
            stream_callback: Optional callback for token streaming (delta: str, is_first: bool)

        Returns:
            Text caption
        """
        model_id = handle
        image_b64 = self._encode_image(image)

        prompt = (
            "Describe this image in detail."
            if detail_level == "detailed"
            else "Describe this image briefly in one sentence."
        )

        # Use unified Q&A history formatter (shared across all providers)
        formatted_prompt = format_qa_history(
            conversation_history=conversation_history,
            current_question=prompt,
            max_turns=5
        )

        payload = {
            "model": model_id,
            "prompt": formatted_prompt,
            "images": [image_b64],
            "stream": bool(stream_callback),  # Enable streaming if callback provided
        }

        if stream_callback:
            # Streaming mode
            full_response = ""
            with self.client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                response.raise_for_status()

                is_first_token = True
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            delta = chunk.get("response", "")
                            if delta:
                                full_response += delta
                                stream_callback(delta, is_first=is_first_token)
                                is_first_token = False

                            # Check if done
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

            return full_response.strip()
        else:
            # Non-streaming mode
            response = self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()

            return result.get("response", "").strip()

    def run_detect(self, handle: Any, image: Image.Image, object_name: str) -> list[dict]:
        """Detect objects in image using Ollama.

        Args:
            handle: Model identifier
            image: PIL Image to analyze
            object_name: Object to detect

        Returns:
            List of detection dicts with parsed bounding boxes
        """
        from ..utils.vlm_response_parser import parse_detection_response

        model_id = handle
        image_b64 = self._encode_image(image)

        prompt = (
            f"Detect all instances of '{object_name}' in this image.\n\n"
            f"Process:\n"
            f"1. Examine the image carefully to identify each '{object_name}'\n"
            f"2. For each instance, determine the bounding box coordinates\n"
            f"3. Number instances sequentially (_1, _2, _3...)\n\n"
            f"Return ONLY this JSON format (no explanations):\n"
            f"{{\n"
            f"  \"{object_name}_1\": [x1, y1, x2, y2],\n"
            f"  \"{object_name}_2\": [x1, y1, x2, y2]\n"
            f"}}"
        )

        payload = {
            "model": model_id,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }

        response = self.client.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()

        # Parse response to extract actual bounding boxes
        text_response = result.get("response", "")
        detections = parse_detection_response(text_response, object_name)

        # Add raw response to each detection for debugging
        for detection in detections:
            detection["raw"] = text_response

        return detections

    def run_point(self, handle: Any, image: Image.Image, object_name: str) -> dict:
        """Point to object center in image using Ollama.

        Args:
            handle: Model identifier
            image: PIL Image to analyze
            object_name: Object to locate

        Returns:
            Coordinates dict with parsed x, y values
        """
        from ..utils.vlm_response_parser import parse_point_response

        model_id = handle
        image_b64 = self._encode_image(image)

        prompt = (
            f"Locate the '{object_name}' in this image.\n\n"
            f"Process:\n"
            f"1. Identify the '{object_name}' in the image\n"
            f"2. Determine its center point\n"
            f"3. Convert to normalized coordinates (0.0 = left/top, 1.0 = right/bottom)\n\n"
            f"Return ONLY this JSON format (no explanations):\n"
            f"{{\n"
            f"  \"{object_name}\": [x, y]\n"
            f"}}"
        )

        payload = {
            "model": model_id,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }

        response = self.client.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()

        # Parse response to extract actual coordinates
        text_response = result.get("response", "")
        coordinates = parse_point_response(text_response, object_name)

        # Add raw response for debugging
        coordinates["raw"] = text_response

        return coordinates

    def run_text(
        self,
        handle: Any,
        prompt: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        system_prompt: Optional[str] = None,
        custom_parameters: Optional[dict] = None,
        stream_callback: Optional[callable] = None,
    ) -> str:
        """Generate text response using Ollama's /api/chat endpoint.

        Uses Ollama's native chat API which automatically handles chat templates
        for each model internally. This is more reliable than manual template
        formatting and prevents context deviation.

        Args:
            handle: Model identifier
            prompt: Current text prompt
            conversation_history: Optional list of (user_msg, bot_response) tuples
            system_prompt: Optional system prompt for context control
            custom_parameters: Optional dict of custom generation parameters
                              (max_tokens, temperature, top_p, top_k, repeat_penalty, seed)

        Returns:
            Generated text
        """
        model_id = handle

        # Build messages array for /api/chat endpoint
        # Ollama handles template formatting automatically per model
        messages = []

        # Add system message (allow /config system_prompt to override)
        if custom_parameters and not system_prompt and isinstance(custom_parameters.get("system_prompt"), str):
            system_prompt = custom_parameters.get("system_prompt")

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant. Provide clear, accurate, and concise responses."
            })

        # Add conversation history (limit to last 5 turns)
        if conversation_history:
            recent_history = conversation_history[-5:]
            logger.debug(f"Including {len(recent_history)} previous conversation turns")

            for user_msg, assistant_msg in recent_history:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})

        # Add current user message
        messages.append({"role": "user", "content": prompt})

        # Log the conversation structure for debugging
        logger.debug(f"Chat with {model_id}: {len(messages)} messages ({len([m for m in messages if m['role'] == 'user'])} user turns)")

        # Build options with defaults, override with custom parameters if provided
        options = {
            "num_predict": 512,    # Max tokens to generate
            "temperature": 0.7,    # Creativity level
            "top_p": 0.9,          # Nucleus sampling
            "top_k": 40,           # Top-K sampling
            "repeat_penalty": 1.1, # Repetition penalty
            "stop": [],            # Let model finish naturally
        }

        # Apply per-model context length if configured
        if model_id in self._ctx_overrides:
            options["num_ctx"] = self._ctx_overrides[model_id]

        # Override with custom parameters from session
        if custom_parameters:
            if "max_tokens" in custom_parameters:
                options["num_predict"] = custom_parameters["max_tokens"]
            if "temperature" in custom_parameters:
                options["temperature"] = custom_parameters["temperature"]
            if "top_p" in custom_parameters:
                options["top_p"] = custom_parameters["top_p"]
            if "top_k" in custom_parameters:
                options["top_k"] = custom_parameters["top_k"]
            if "repeat_penalty" in custom_parameters:
                options["repeat_penalty"] = custom_parameters["repeat_penalty"]
            if "seed" in custom_parameters:
                options["seed"] = custom_parameters["seed"]
            if "stop" in custom_parameters:
                options["stop"] = custom_parameters["stop"]

            logger.debug(f"Using custom parameters: {custom_parameters}")

        # Use /api/chat endpoint (Ollama handles templates automatically)
        # PERFORMANCE: Enable streaming for lower latency and progressive generation
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": True,  # Enable streaming
            "options": options
        }

        # Handle streaming response (accumulate for return; optionally stream via callback)
        full_response = ""

        with self.client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        # Accumulate message content from each chunk
                        if "message" in chunk and "content" in chunk["message"]:
                            delta = chunk["message"]["content"]
                            if delta:
                                full_response += delta
                                if stream_callback:
                                    try:
                                        stream_callback(delta, is_first=(len(full_response) == len(delta)))
                                    except Exception:
                                        pass

                        # Check if done
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming chunk: {line}")
                        continue

        message_content = full_response.strip()

        if not message_content:
            logger.warning(f"Empty response from {model_id}")
            return "No response generated."

        # LOG RAW RESPONSE FROM OLLAMA (BEFORE ANY PROCESSING)
        logger.info("="*80)
        logger.info(f"[OLLAMA RAW RESPONSE] ({len(message_content)} chars)")
        logger.info(message_content)
        logger.info("="*80)
        logger.info(f"[OLLAMA] Has <think> tags: {'<think>' in message_content.lower()}")
        logger.info(f"[OLLAMA] Has </think> tags: {'</think>' in message_content.lower()}")

        # Apply response cleaning to remove artifacts and meta-commentary
        from ..utils.response_cleaner import clean_model_response
        cleaned_response = clean_model_response(message_content, aggressive=True)

        return cleaned_response

    def install_model(
        self, model_name: str, progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> bool:
        """Install model from Ollama library with progress tracking.

        Args:
            model_name: Model name (e.g., "llama3.2:3b")
            progress_callback: Optional callback function(status, completed, total)

        Returns:
            True if installation succeeded
        """
        logger.info(f"Installing Ollama model: {model_name}")

        payload = {"name": model_name, "stream": True}

        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=600  # 10 minutes for large downloads
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")

                            # Extract progress information
                            completed = data.get("completed", 0)
                            total = data.get("total", 0)

                            # Call progress callback if provided
                            if progress_callback:
                                progress_callback(status, completed, total)

                            # Check for completion
                            if status == "success":
                                logger.info(f"Successfully installed {model_name}")
                                return True

                        except Exception as parse_error:
                            logger.debug(f"Error parsing progress line: {parse_error}")
                            continue

            logger.info(f"Successfully installed {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to install {model_name}: {e}")
            return False

    def delete_model(self, model_id: str) -> bool:
        """Delete model from Ollama.

        Args:
            model_id: Model identifier

        Returns:
            True if deletion succeeded
        """
        logger.info(f"Deleting Ollama model: {model_id}")

        payload = {"name": model_id}

        try:
            response = self.client.request(
                "DELETE",
                f"{self.base_url}/api/delete",
                json=payload
            )
            response.raise_for_status()
            logger.info(f"Successfully deleted {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {model_id}: {e}")
            return False

    def estimate_model_size(self, model_name: str) -> float:
        """Estimate model size by querying Ollama API for actual model info.

        Args:
            model_name: Model name (e.g., "llama3.2:3b")

        Returns:
            Estimated size in GB
        """
        try:
            # First check if model is already installed locally
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()

                # Get size from model info (in bytes)
                size_bytes = data.get("size", 0)

                if size_bytes > 0:
                    size_gb = size_bytes / (1024**3)
                    logger.info(f"Estimated size for {model_name}: {size_gb:.2f} GB (from Ollama API)")
                    return size_gb
                else:
                    # Try to get from details/parameters
                    details = data.get("details", {})
                    param_size = details.get("parameter_size", "")

                    # Parse parameter size (e.g., "3B" -> 3GB estimate)
                    if param_size:
                        return self._estimate_from_params(param_size)

            # Model not installed locally, try to estimate from registry
            logger.debug(f"Model {model_name} not found locally, checking Ollama library")

            # Query Ollama library API for model info
            library_response = requests.get(
                f"https://registry.ollama.ai/v2/library/{model_name.split(':')[0]}/manifests/{model_name.split(':')[1] if ':' in model_name else 'latest'}",
                timeout=5
            )

            if library_response.status_code == 200:
                manifest = library_response.json()

                # Sum layer sizes from manifest
                total_size = 0
                for layer in manifest.get("layers", []):
                    total_size += layer.get("size", 0)

                if total_size > 0:
                    size_gb = total_size / (1024**3)
                    logger.info(f"Estimated size for {model_name}: {size_gb:.2f} GB (from Ollama registry)")
                    return size_gb

        except requests.exceptions.Timeout:
            logger.debug(f"Timeout querying Ollama API for {model_name}, using fallback")
        except requests.exceptions.ConnectionError:
            logger.debug(f"Could not connect to Ollama for {model_name}, using fallback")
        except Exception as e:
            logger.debug(f"Error estimating Ollama model size for {model_name}: {e}")

        # Fallback to name-based estimation
        return self._estimate_size_from_name(model_name)

    def _estimate_from_params(self, param_size: str) -> float:
        """Estimate size from parameter count string.

        Args:
            param_size: Parameter size string (e.g., "3B", "7B")

        Returns:
            Estimated size in GB
        """
        param_lower = param_size.lower().replace("b", "")
        try:
            params = float(param_lower)
            # Rough estimate: 2 bytes per parameter (fp16) with overhead
            return params * 2.0
        except:
            return 4.0

    def _estimate_size_from_name(self, model_name: str) -> float:
        """Fallback: Estimate size based on model name patterns.

        Args:
            model_name: Model name

        Returns:
            Estimated size in GB
        """
        name_lower = model_name.lower()

        size_patterns = {
            "0.5b": 1.0,
            "1b": 2.0,
            "3b": 2.0,
            "7b": 4.0,
            "8b": 4.5,
            "11b": 6.5,
            "13b": 7.5,
            "30b": 18.0,
            "70b": 40.0,
        }

        for pattern, size in size_patterns.items():
            if pattern in name_lower:
                return size

        # Default estimate
        return 4.0

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string for Ollama API.

        Args:
            image: PIL Image

        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")

    # Convenience wrapper methods for app.py that accept file paths
    def qa(self, model_id: str, image_path: str, question: str) -> str:
        """Question answering convenience method.

        Args:
            model_id: Model identifier
            image_path: Path to image file
            question: Question to answer

        Returns:
            Text answer
        """
        image = Image.open(image_path)
        return self.run_qa(model_id, image, question)

    def caption(self, model_id: str, image_path: str, detail_level: str = "detailed") -> str:
        """Image captioning convenience method.

        Args:
            model_id: Model identifier
            image_path: Path to image file
            detail_level: "detailed" or "short"

        Returns:
            Text caption
        """
        image = Image.open(image_path)
        return self.run_caption(model_id, image, detail_level)

    def detect(self, model_id: str, image_path: str, prompt: str) -> str:
        """Object detection convenience method.

        Args:
            model_id: Model identifier
            image_path: Path to image file
            prompt: Detection prompt

        Returns:
            Detection results as text
        """
        image = Image.open(image_path)
        # Extract object name from prompt
        object_name = prompt.replace("Detect ", "").replace("detect ", "")
        detections = self.run_detect(model_id, image, object_name)

        # Format results as readable text
        if detections:
            result_text = f"Detected {len(detections)} instance(s):\n\n"
            for i, det in enumerate(detections, 1):
                result_text += f"{i}. {det.get('label', 'unknown')}\n"
                result_text += f"   Raw response: {det.get('raw', 'N/A')}\n"
            return result_text
        return "No objects detected."

    def point(self, model_id: str, image_path: str, object_name: str) -> str:
        """Object pointing convenience method.

        Args:
            model_id: Model identifier
            image_path: Path to image file
            object_name: Object to locate

        Returns:
            Location results as text
        """
        image = Image.open(image_path)
        result = self.run_point(model_id, image, object_name)

        # Format results as readable text
        x = result.get("x", 0)
        y = result.get("y", 0)
        raw = result.get("raw", "")

        return f"Object location: ({x}, {y})\n\nModel response:\n{raw}"

    def chat(
        self,
        model_id: str,
        message: str,
        conversation_history: Optional[list[tuple[str, str]]] = None
    ) -> str:
        """Text chat convenience method.

        Args:
            model_id: Model identifier
            message: User message
            conversation_history: Optional list of (user_msg, bot_response) tuples

        Returns:
            Model response
        """
        return self.run_text(model_id, message, conversation_history)
