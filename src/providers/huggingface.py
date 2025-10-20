"""HuggingFace provider implementation using transformers library."""

import os
from pathlib import Path
from typing import Any, Optional

import torch
from loguru import logger
from PIL import Image

from ..models.endpoints import CompatibilityStatus, EndpointType, ModelType
from ..models.model import ModelInfo
from ..models.model_cache import ModelMetadataCache
from ..models.provider import ProviderConfig
from ..utils.history_formatter import format_conversation_history, format_qa_history
from .base import BaseProvider


class HuggingFaceProvider(BaseProvider):
    """HuggingFace provider using transformers library."""

    def __init__(self, config: ProviderConfig):
        """Initialize HuggingFace provider.

        Args:
            config: Provider configuration with cache_dir and device settings
        """
        self.config = config
        self.cache_dir = Path(str(config.cache_dir)).expanduser()
        self.device_preference = config.device_preference or ["cuda", "mps", "cpu"]

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata cache for efficient model inspection
        self.metadata_cache = ModelMetadataCache()
        logger.debug("Initialized model metadata cache")

        # Determine best available device
        self.device = self._get_best_device()
        logger.info(f"HuggingFace provider initialized on device: {self.device}")

    def _get_best_device(self) -> str:
        """Determine best available device from preferences.

        Returns:
            Device string ("cuda", "mps", or "cpu")
        """
        for device in self.device_preference:
            if device == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif device == "mps" and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def discover_models(self) -> list[ModelInfo]:
        """Discover cached HuggingFace models using metadata cache.

        Scans two locations:
        1. HuggingFace cache directory (~/.cache/huggingface/hub/)
        2. Local models directory (./models/)

        Returns:
            List of ModelInfo objects for cached models
        """
        models = []

        # Location 1: Scan HuggingFace cache directory for model directories
        if self.cache_dir.exists():
            try:
                # Look for model directories (models--<org>--<name> format)
                for model_dir in self.cache_dir.glob("models--*"):
                    if not model_dir.is_dir():
                        continue

                    # Parse model name from directory (models--org--name -> org/name)
                    dir_name = model_dir.name
                    if dir_name.startswith("models--"):
                        parts = dir_name[8:].split("--")
                        if len(parts) >= 2:
                            model_id = "/".join(parts)

                            # Get metadata from cache (reads config.json only, no weight loading)
                            metadata = self.metadata_cache.get_metadata(str(model_dir), provider="huggingface")

                            # Convert metadata to ModelInfo
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

                                model_info = ModelInfo(
                                    model_id=model_id,
                                    name=model_id,
                                    provider="huggingface",
                                    size_gb=metadata.file_size_gb,
                                    architecture=metadata.architecture,
                                    quantization=metadata.quantization,
                                    params_billions=metadata.params_billions,
                                    ram_gb=metadata.ram_gb,
                                    vram_gb=metadata.vram_gb,
                                    params_exact=metadata.params_exact,
                                    ram_exact=metadata.ram_exact,
                                    model_type=model_type,
                                    capabilities=capabilities,
                                    compatibility=CompatibilityStatus.PERFECT_FIT,  # Will be assessed later
                                    compatibility_message="Compatibility not yet assessed",
                                    is_installed=True,
                                )
                            else:
                                # Fallback: metadata inspection failed, use basic info
                                logger.warning(f"Could not get metadata for {model_id}, using fallback")
                                size_gb = self._estimate_cached_model_size(model_dir)
                                model_info = ModelInfo(
                                    model_id=model_id,
                                    name=model_id,
                                    provider="huggingface",
                                    size_gb=size_gb,
                                    model_type=ModelType.LLM,
                                    capabilities=[EndpointType.TEXT],
                                    compatibility=CompatibilityStatus.PERFECT_FIT,
                                    compatibility_message="Compatibility not yet assessed",
                                    is_installed=True,
                                )

                            models.append(model_info)
                            logger.debug(f"Discovered HF model: {model_id} ({metadata.model_type if metadata else 'unknown'}, {metadata.quantization if metadata else 'unknown'})")

                logger.info(f"Discovered {len(models)} HuggingFace models from HF cache")
            except Exception as e:
                logger.error(f"Error discovering HuggingFace cache models: {e}")
        else:
            logger.warning(f"HuggingFace cache directory not found: {self.cache_dir}")

        # Location 2: Scan local ./models directory for HuggingFace format models
        local_models_dir = Path.cwd() / "models"
        if local_models_dir.exists():
            try:
                logger.info(f"Scanning local models directory: {local_models_dir}")
                for model_dir in local_models_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    # Check if it's a valid HuggingFace model (has config.json or safetensors)
                    has_config = (model_dir / "config.json").exists()
                    has_weights = list(model_dir.glob("*.safetensors")) or list(model_dir.glob("*.bin"))

                    if has_config or has_weights:
                        # Use directory name as model_id
                        model_id = model_dir.name
                        logger.info(f"Found local model: {model_id}")

                        # Get metadata
                        metadata = self.metadata_cache.get_metadata(str(model_dir), provider="huggingface")

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

                            model_info = ModelInfo(
                                model_id=str(model_dir),  # Use full path as model_id for local models
                                name=f"{model_id} (local)",
                                provider="huggingface",
                                size_gb=metadata.file_size_gb,
                                architecture=metadata.architecture,
                                quantization=metadata.quantization,
                                params_billions=metadata.params_billions,
                                ram_gb=metadata.ram_gb,
                                vram_gb=metadata.vram_gb,
                                params_exact=metadata.params_exact,
                                ram_exact=metadata.ram_exact,
                                model_type=model_type,
                                capabilities=capabilities,
                                compatibility=CompatibilityStatus.PERFECT_FIT,
                                compatibility_message="Compatibility not yet assessed",
                                is_installed=True,
                            )
                        else:
                            # Fallback: metadata inspection failed, use basic info
                            logger.warning(f"Could not get metadata for local model {model_id}, using fallback")
                            size_gb = self._estimate_cached_model_size(model_dir)
                            model_info = ModelInfo(
                                model_id=str(model_dir),  # Use full path as model_id
                                name=f"{model_id} (local)",
                                provider="huggingface",
                                size_gb=size_gb,
                                model_type=ModelType.LLM,
                                capabilities=[EndpointType.TEXT],
                                compatibility=CompatibilityStatus.PERFECT_FIT,
                                compatibility_message="Compatibility not yet assessed",
                                is_installed=True,
                            )

                        models.append(model_info)
                        logger.debug(f"Discovered local model: {model_id}")

                logger.info(f"Discovered {len([m for m in models if '(local)' in m.name])} models from ./models directory")
            except Exception as e:
                logger.error(f"Error discovering local models: {e}")

        logger.info(f"Total HuggingFace models discovered: {len(models)}")
        return models

    def _estimate_cached_model_size(self, model_dir: Path) -> float:
        """Estimate size of cached model in GB.

        Args:
            model_dir: Path to model cache directory

        Returns:
            Size in GB
        """
        try:
            total_size = 0
            for file in model_dir.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size / (1024**3)  # Convert to GB
        except Exception as e:
            logger.warning(f"Could not estimate size for {model_dir}: {e}")
            return 4.0  # Default estimate

    # DEPRECATED: This method is no longer used - metadata cache provides dynamic detection
    # Keeping for backward compatibility only (used in fallback scenarios)
    def _classify_hf_model(self, model_id: str) -> tuple[ModelType, list[EndpointType]]:
        """[DEPRECATED] Classify HuggingFace model by name patterns.

        This method is deprecated in favor of metadata cache which reads actual model config.
        Only used as a fallback when metadata inspection fails.

        Args:
            model_id: Model identifier (e.g., "llava-hf/llava-1.5-7b-hf")

        Returns:
            (ModelType, list of EndpointType)
        """
        model_lower = model_id.lower()

        # VLM keywords (fallback only)
        vlm_keywords = [
            "llava", "blip", "instructblip", "clip", "vision", "vit",
            "paligemma", "idefics", "fuyu", "kosmos", "qwen-vl",
            "cogvlm", "internvl", "minicpm-v", "phi-3-vision", "moondream"
        ]

        if any(keyword in model_lower for keyword in vlm_keywords):
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
        if "embed" in model_lower or "sentence-transformer" in model_lower:
            return (ModelType.EMBEDDING, [])

        # Default to LLM
        return (ModelType.LLM, [EndpointType.TEXT])

    def load_model(self, model_id: str, device: str) -> Any:
        """Load HuggingFace model with transformers.

        Args:
            model_id: Model identifier (e.g., "llava-hf/llava-1.5-7b-hf")
            device: Target device (used self.device instead)

        Returns:
            Tuple of (model, processor/tokenizer)
        """
        logger.info(f"Loading HuggingFace model: {model_id}")
        
        # Suppress transformers output during loading (keeps TUI clean)
        from ..utils.output_suppressor import suppress_transformers_output

        try:
            # Import base classes (always available)
            from transformers import AutoModel, AutoTokenizer

            # Determine if this is a VLM using metadata cache
            # This checks the actual model config, not just keywords
            model_dir = self.cache_dir / ("models--" + model_id.replace("/", "--"))
            metadata = self.metadata_cache.get_metadata(str(model_dir), provider="huggingface")

            is_vlm = False
            if metadata:
                is_vlm = metadata.model_type == "vlm"
                logger.debug(f"Metadata indicates model type: {metadata.model_type}")
            else:
                # Fallback to keyword detection if metadata not available
                logger.debug("Metadata not available, using keyword fallback")
                model_lower = model_id.lower()
                vlm_keywords = [
                    "llava", "blip", "instructblip", "clip", "vision", "vl",
                    "paligemma", "idefics", "fuyu", "kosmos", "qwen-vl", "qwen3-vl",
                    "cogvlm", "internvl", "minicpm-v", "phi-3-vision", "moondream"
                ]
                is_vlm = any(keyword in model_lower for keyword in vlm_keywords)

            if is_vlm:
                # Try to load as VLM with processor (only import if needed)
                try:
                    # Import VLM-specific classes (may not be available in all versions)
                    from transformers import AutoProcessor

                    # Suppress output during loading
                    with suppress_transformers_output():
                        processor = AutoProcessor.from_pretrained(
                            model_id,
                            cache_dir=str(self.cache_dir),
                            trust_remote_code=True
                        )

                        # Prepare loading kwargs with proper dtype parameter
                        vlm_load_kwargs = {
                            "cache_dir": str(self.cache_dir),
                            "trust_remote_code": True,
                            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                            "low_cpu_mem_usage": True,  # Stream weights during loading
                        }

                        # 2025 OPTIMIZATION: Flash Attention 2/3 support for faster inference
                        # Reduces memory usage and improves speed (requires flash-attn package)
                        try:
                            import flash_attn  # noqa: F401
                            vlm_load_kwargs["attn_implementation"] = "flash_attention_2"
                            logger.info("✓ Flash Attention 2 enabled for faster VLM inference")
                        except ImportError:
                            logger.debug("Flash Attention not available (install flash-attn for 2-4x speedup)")
                            # Fallback to SDPA (Scaled Dot-Product Attention) - PyTorch 2.0+ native
                            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                                vlm_load_kwargs["attn_implementation"] = "sdpa"
                                logger.info("✓ Using PyTorch SDPA for optimized attention")

                        # Load VLM with appropriate model class
                        # Try multiple classes in order of compatibility for custom/standard architectures
                        model = None
                        last_error = None
                        
                        # PRODUCTION FIX: Try AutoModelForCausalLM FIRST
                        # This works with most VLMs including custom architectures (moondream2, etc.)
                        # when trust_remote_code=True is enabled
                        model_classes_to_try = [
                            ('AutoModelForCausalLM', 'VLMs with custom configs (moondream, LLaVA, etc.)'),
                            ('AutoModelForVision2Seq', 'Modern VLMs (Qwen2-VL, Idefics, etc.)'),
                        ]
                        
                        for model_class_name, desc in model_classes_to_try:
                            try:
                                logger.debug(f"Trying {model_class_name} for VLM: {desc}")
                                
                                if model_class_name == 'AutoModelForCausalLM':
                                    from transformers import AutoModelForCausalLM
                                    
                                    # Try with full kwargs first
                                    try:
                                        model = AutoModelForCausalLM.from_pretrained(model_id, **vlm_load_kwargs)
                                    except (TypeError, ValueError) as param_error:
                                        # Some models don't support all parameters (e.g., attn_implementation)
                                        # Retry with minimal kwargs
                                        logger.debug(f"Full kwargs failed, retrying with minimal kwargs: {param_error}")
                                        minimal_kwargs = {
                                            "cache_dir": str(self.cache_dir),
                                            "trust_remote_code": True,
                                            "torch_dtype": vlm_load_kwargs.get("torch_dtype", torch.float32),
                                        }
                                        model = AutoModelForCausalLM.from_pretrained(model_id, **minimal_kwargs)
                                        
                                elif model_class_name == 'AutoModelForVision2Seq':
                                    from transformers import AutoModelForVision2Seq
                                    model = AutoModelForVision2Seq.from_pretrained(model_id, **vlm_load_kwargs)
                                
                                # Verify model has .generate() method
                                if not hasattr(model, 'generate'):
                                    logger.debug(f"{model_class_name} loaded but has no .generate() method, trying next...")
                                    model = None
                                    continue
                                
                                logger.info(f"✓ Loaded VLM with {model_class_name}")
                                break
                                
                            except Exception as e:
                                last_error = e
                                logger.debug(f"{model_class_name} failed: {str(e)[:200]}")
                                continue
                        
                        if model is None:
                            error_msg = f"Could not load VLM {model_id} with any compatible model class.\n"
                            error_msg += f"Tried: {', '.join([c[0] for c in model_classes_to_try])}\n"
                            if last_error:
                                error_msg += f"Last error: {str(last_error)[:300]}"
                            raise RuntimeError(error_msg)

                    # Determine target device - some models have MPS compatibility issues
                    target_device = self.device
                    if "qwen3" in model_id.lower() and self.device == "mps":
                        target_device = "cpu"
                        logger.warning(
                            f"Qwen3 has MPS compatibility issues. Using CPU instead. "
                            f"This may be slower but will work correctly."
                        )

                    model.to(target_device)
                    logger.info(f"Loaded VLM model {model_id} on {target_device}")
                    return (model, processor)
                except Exception as e:
                    logger.warning(f"Failed to load as VLM: {e}, trying as LLM...")

            # Load as LLM (suppress output)
            with suppress_transformers_output():
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True
                )

                # Prepare loading kwargs with proper dtype parameter
                load_kwargs = {
                    "cache_dir": str(self.cache_dir),
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,  # Stream weights during loading
                }

                # Use 'dtype' instead of deprecated 'torch_dtype'
                if self.device == "cuda":
                    load_kwargs["torch_dtype"] = torch.float16
                else:
                    load_kwargs["torch_dtype"] = torch.float32

                # 2025 OPTIMIZATION: Flash Attention for LLMs
                try:
                    import flash_attn  # noqa: F401
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("✓ Flash Attention 2 enabled for faster inference")
                except ImportError:
                    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                        load_kwargs["attn_implementation"] = "sdpa"
                        logger.info("✓ Using PyTorch SDPA for optimized attention")

                # PRODUCTION FIX: Use AutoModelForCausalLM for LLMs
                # This works with both standard and custom architectures when trust_remote_code=True
                # AutoModel doesn't support custom config classes, so we avoid it completely
                from transformers import AutoModelForCausalLM
                
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
                except (TypeError, ValueError) as param_error:
                    # Some models don't support all parameters
                    logger.debug(f"Full kwargs failed, retrying with minimal kwargs: {param_error}")
                    minimal_kwargs = {
                        "cache_dir": str(self.cache_dir),
                        "trust_remote_code": True,
                        "torch_dtype": load_kwargs.get("torch_dtype", torch.float32),
                    }
                    model = AutoModelForCausalLM.from_pretrained(model_id, **minimal_kwargs)

            # Determine target device - some models have MPS compatibility issues
            target_device = self.device

            # Qwen3 has known MPS compatibility issues with torch 2.5.1
            # Force CPU for Qwen3 on MPS devices to avoid matrix dimension errors
            if "qwen3" in model_id.lower() and self.device == "mps":
                target_device = "cpu"
                logger.warning(
                    f"Qwen3 has MPS compatibility issues. Using CPU instead. "
                    f"This may be slower but will work correctly."
                )

            model.to(target_device)
            logger.info(f"Loaded LLM model {model_id} on {target_device}")
            return (model, tokenizer)

        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {model_id}: {e}")
            raise RuntimeError(f"Failed to load model {model_id}: {e}")

    def unload_model(self, handle: Any) -> None:
        """Unload model and free GPU memory universally across all platforms.

        Args:
            handle: Tuple of (model, processor/tokenizer)
        """
        if handle is None:
            return

        try:
            model, _ = handle
            del model

            # Universal GPU memory cleanup - works on all platforms
            if torch.cuda.is_available():
                # NVIDIA CUDA (Windows, Linux, Jetson)
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
            elif torch.backends.mps.is_available():
                # Apple Metal (macOS with M-series)
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache")
            # Note: ROCm (AMD) uses same API as CUDA
            # CPU doesn't need explicit cache clearing

            logger.info("HuggingFace model unloaded")
        except Exception as e:
            logger.warning(f"Error unloading HuggingFace model: {e}")

    def run_qa(
        self,
        handle: Any,
        image: Image.Image,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None
    ) -> str:
        """Run question answering with VLM using unified history formatter.

        Args:
            handle: Tuple of (model, processor)
            image: PIL Image
            question: Question text
            conversation_history: Optional list of (user_msg, bot_response) tuples

        Returns:
            Answer text
        """
        model, processor = handle

        try:
            # UNIVERSAL DEVICE DETECTION - get actual device model is on
            # Works on all platforms: CUDA, MPS, CPU, ROCm, Jetson, etc.
            model_device = next(model.parameters()).device
            logger.debug(f"Model is on device: {model_device}")

            # Use unified Q&A history formatter (shared across all providers)
            full_question = format_qa_history(
                conversation_history=conversation_history,
                current_question=question,
                max_turns=5
            )

            # SPECIAL HANDLING: Moondream2 has a custom inference API
            # It uses model.encode_image() + model.answer_question() instead of standard processor
            if hasattr(model, 'encode_image') and hasattr(model, 'answer_question'):
                logger.debug("Detected moondream2 custom API - using encode_image + answer_question")
                try:
                    # Encode the image using moondream's custom encoder
                    image_embeds = model.encode_image(image)
                    
                    # Generate answer using moondream's custom method
                    answer = model.answer_question(
                        image_embeds=image_embeds,
                        question=full_question,
                        tokenizer=processor  # moondream uses tokenizer, not processor
                    )
                    
                    logger.info("✓ Response generated with moondream2 API")
                    return answer.strip()
                    
                except Exception as e:
                    logger.error(f"Moondream2 custom API failed: {e}")
                    # Fall through to try standard approach
                    pass

            # Prepare inputs with automatic format detection for all VLMs
            # Try messages format first (modern VLMs: Qwen2-VL, Qwen3-VL, Idefics, etc.)
            # Fall back to standard format (LLaVA, BLIP, etc.)
            inputs = None
            
            # Strategy 1: Try messages format with chat template (if available)
            if hasattr(processor, 'apply_chat_template'):
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": full_question}
                            ]
                        }
                    ]
                    
                    # Apply chat template to get properly formatted text with image tokens
                    text_prompt = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # Process with the formatted text
                    inputs = processor(
                        text=[text_prompt],
                        images=[image],
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    logger.debug("✓ Using messages format with chat template")
                    
                except Exception as e:
                    logger.debug(f"Messages format failed: {e}, trying standard format")
                    inputs = None
            
            # Strategy 2: Standard text + images format (fallback)
            if inputs is None:
                try:
                    inputs = processor(
                        text=full_question,
                        images=image,
                        return_tensors="pt"
                    )
                    logger.debug("✓ Using standard text + images format")
                except Exception as e:
                    logger.error(f"Both input formats failed: {e}")
                    raise RuntimeError(
                        f"Failed to prepare inputs for VLM inference.\n"
                        f"Processor: {type(processor).__name__}\n"
                        f"Error: {e}"
                    )
            
            # Move inputs to model device (handle each tensor individually for MPS compatibility)
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Warn if on CPU (very slow)
            if str(model_device) == 'cpu':
                logger.warning("⏳ Running inference on CPU - this will be SLOW (30s-2min)")
                logger.warning("   Consider using a smaller model or CUDA/MPS-compatible device")
            
            # Generate response (reduced tokens for faster response on CPU)
            logger.info("Generating response...")
            max_tokens = 512 if str(model_device) == 'cpu' else 1024

            # 2025 OPTIMIZATION: KV cache configuration for faster generation
            generation_config = {
                "max_new_tokens": max_tokens,
                "use_cache": True,  # Enable KV cache (default but explicit)
                "do_sample": False,  # Greedy decoding for consistency
            }

            with torch.no_grad():
                output = model.generate(**inputs, **generation_config)
            
            logger.info("✓ Response generated")

            # Decode response
            response = processor.batch_decode(output, skip_special_tokens=True)[0]

            # If we used history format, try to extract just the answer
            if conversation_history and "A:" in response:
                # Try to extract the last answer after "A:"
                parts = response.split("A:")
                if len(parts) > 1:
                    response = parts[-1].strip()

            return response.strip()

        except Exception as e:
            logger.error(f"QA inference failed: {e}")
            raise NotImplementedError(f"QA not supported for this model: {e}")

    def run_caption(
        self,
        handle: Any,
        image: Image.Image,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        detail_level: str = "detailed"
    ) -> str:
        """Generate image caption.

        Args:
            handle: Tuple of (model, processor)
            image: PIL Image
            conversation_history: Optional list of (user_msg, bot_response) tuples
            detail_level: "detailed" or "short"

        Returns:
            Caption text
        """
        prompt = (
            "Describe this image in detail."
            if detail_level == "detailed"
            else "Describe this image briefly."
        )
        return self.run_qa(handle, image, prompt, conversation_history)

    def run_detect(self, handle: Any, image: Image.Image, object_name: str) -> list[dict]:
        """Detect objects with VLM and parse bounding boxes.

        Args:
            handle: Tuple of (model, processor)
            image: PIL Image
            object_name: Object to detect

        Returns:
            List of detection dicts with parsed bounding boxes
        """
        from ..utils.vlm_response_parser import parse_detection_response

        prompt = (
            f"Detect all instances of '{object_name}' in this image. "
            f"Provide bounding box coordinates in JSON format as: "
            f'[{{"bbox": [x1, y1, x2, y2], "label": "{object_name}"}}]'
        )
        response = self.run_qa(handle, image, prompt)

        # Parse response to extract actual bounding boxes
        detections = parse_detection_response(response, object_name)

        # Add raw response to each detection for debugging
        for detection in detections:
            detection["raw"] = response

        return detections

    def run_point(self, handle: Any, image: Image.Image, object_name: str) -> dict:
        """Point to object location with VLM and parse coordinates.

        Args:
            handle: Tuple of (model, processor)
            image: PIL Image
            object_name: Object to locate

        Returns:
            Coordinates dict with parsed x, y values
        """
        from ..utils.vlm_response_parser import parse_point_response

        prompt = (
            f"Where is the '{object_name}' in this image? "
            f"Provide the center coordinates in JSON format as: "
            f'{{"x": <number>, "y": <number>}}'
        )
        response = self.run_qa(handle, image, prompt)

        # Parse response to extract actual coordinates
        coordinates = parse_point_response(response, object_name)

        # Add raw response for debugging
        coordinates["raw"] = response

        return coordinates

    def run_text(
        self,
        handle: Any,
        prompt: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        custom_parameters: Optional[dict] = None
    ) -> str:
        """Generate text response for LLM using tokenizer's built-in chat template.

        Uses the tokenizer's apply_chat_template() method which automatically
        formats conversations based on the model's native template. This is more
        reliable than manual template detection.

        Args:
            handle: Tuple of (model, tokenizer)
            prompt: Current text prompt
            conversation_history: Optional list of (user_msg, bot_response) tuples
            custom_parameters: Optional dict of custom generation parameters
                              (max_tokens, temperature, top_p, top_k, repeat_penalty, seed)

        Returns:
            Generated text
        """
        model, tokenizer = handle

        try:
            # UNIVERSAL DEVICE DETECTION - works on all platforms
            # Get the actual device the model is on (not self.device which may differ)
            # This handles: CUDA (Windows/Linux), MPS (Mac), CPU (all platforms)
            model_device = next(model.parameters()).device
            logger.debug(f"Model is on device: {model_device}")

            # Set padding token if not present (needed for some models like DialoGPT)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Build messages array (same format as OpenAI/Ollama)
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Provide clear, accurate, and concise responses."}
            ]

            # Add conversation history (limit to last 5 turns)
            if conversation_history:
                recent_history = conversation_history[-5:]
                logger.debug(f"Including {len(recent_history)} previous conversation turns")

                for user_msg, assistant_msg in recent_history:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})

            # Add current user message
            messages.append({"role": "user", "content": prompt})

            # Use tokenizer's built-in chat template (knows model's native format)
            try:
                # Try to use apply_chat_template (available in modern transformers)
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # Adds assistant prefix for response
                )
                logger.debug(f"Using tokenizer's native chat template")
            except Exception as e:
                # Fallback to manual formatting if tokenizer doesn't have chat template
                logger.warning(f"Tokenizer has no chat template, using fallback: {e}")

                # Extract model name for fallback template detection
                model_name = getattr(model.config, "_name_or_path", "")
                formatted_prompt = format_conversation_history(
                    conversation_history=conversation_history,
                    current_prompt=prompt,
                    max_turns=5,
                    system_prompt=None,
                    model_name=model_name
                )

            # Tokenize input and move to the SAME device as the model
            # This works universally: CUDA (NVIDIA), MPS (Apple), CPU (all), ROCm (AMD), etc.
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(model_device)

            # Build generation parameters with defaults
            gen_params = {
                "max_new_tokens": 1024,  # Increased for more detailed responses
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "use_cache": True,  # 2025 OPTIMIZATION: Enable KV cache for faster generation
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            # Override with custom parameters from session
            if custom_parameters:
                if "max_tokens" in custom_parameters:
                    gen_params["max_new_tokens"] = custom_parameters["max_tokens"]
                if "temperature" in custom_parameters:
                    gen_params["temperature"] = custom_parameters["temperature"]
                if "top_p" in custom_parameters:
                    gen_params["top_p"] = custom_parameters["top_p"]
                if "top_k" in custom_parameters:
                    gen_params["top_k"] = custom_parameters["top_k"]
                if "repeat_penalty" in custom_parameters:
                    gen_params["repetition_penalty"] = custom_parameters["repeat_penalty"]
                if "seed" in custom_parameters:
                    # Set seed for reproducibility
                    import random
                    import numpy as np
                    seed = custom_parameters["seed"]
                    torch.manual_seed(seed)
                    random.seed(seed)
                    np.random.seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                logger.debug(f"Using custom parameters: {custom_parameters}")

            # Generate with parameters
            with torch.no_grad():
                output = model.generate(**inputs, **gen_params)

            # Decode only the new tokens (exclude input prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_length:]

            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Clean up response - remove any remaining EOS tokens or extra whitespace
            if tokenizer.eos_token:
                response = response.replace(tokenizer.eos_token, "").strip()

            # Fallback: if empty, decode full output and try to remove prompt
            if not response:
                full_response = tokenizer.decode(output[0], skip_special_tokens=True)
                if full_response.startswith(formatted_prompt):
                    response = full_response[len(formatted_prompt):].strip()
                else:
                    response = full_response.strip()

            # Apply response cleaning to remove artifacts and meta-commentary
            from ..utils.response_cleaner import clean_model_response
            if response:
                response = clean_model_response(response, aggressive=True)

            return response if response else "I don't have a response."

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")

    def get_model_info(self, model_id: str) -> dict:
        """Get detailed model information for HuggingFace models.

        Args:
            model_id: Model identifier (e.g., "llava-hf/llava-1.5-7b-hf")

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
                "provider": "huggingface",
                "default_parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "max_tokens": 1024,  # Increased for more detailed responses
                    "repeat_penalty": 1.0,
                }
            }

        # Return full model info
        return {
            "model_id": model.model_id,
            "name": model.name,
            "provider": "huggingface",
            "architecture": model.architecture or "Unknown",
            "quantization": model.quantization or "None",
            "size_gb": model.size_gb,
            "model_type": str(model.model_type).upper() if hasattr(model.model_type, 'value') else str(model.model_type).upper(),
            "capabilities": [str(cap) for cap in model.capabilities] if model.capabilities else ["text"],
            "default_parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "max_tokens": 1024,  # Increased for more detailed responses
                "repeat_penalty": 1.0,
            }
        }

    def install_model(self, model_name: str, progress_callback=None) -> bool:
        """Download model from HuggingFace Hub with visible progress bars.

        Handles both regular transformers models and GGUF models:
        - GGUF repos: Downloads .gguf files directly (no transformers validation)
        - Regular models: Downloads via transformers (requires valid config.json)
        
        Progress bars are shown during download for transparency.
        No output suppression is used during installation.

        Args:
            model_name: Model identifier (e.g., "llava-hf/llava-1.5-7b-hf" or "TheBloke/Llama-2-7B-GGUF")
            progress_callback: Optional progress callback (not implemented)

        Returns:
            True if successful
        """
        logger.info(f"Downloading HuggingFace model: {model_name}")
        
        # NOTE: Do NOT use suppress_transformers_output() here!
        # We want users to see download progress bars during installation.

        try:
            from huggingface_hub import list_repo_files, hf_hub_download

            # STEP 1: Check if this is a GGUF repository
            # GGUF repos contain .gguf files and should be downloaded directly
            try:
                repo_files = list_repo_files(model_name)
                gguf_files = [f for f in repo_files if f.endswith('.gguf')]

                if gguf_files:
                    logger.info(f"Detected GGUF repository with {len(gguf_files)} .gguf file(s)")
                    logger.info(f"Downloading GGUF files directly (skipping transformers validation)")

                    # Get file sizes for better UX
                    from huggingface_hub import HfFileSystem
                    fs = HfFileSystem()
                    file_sizes = {}
                    try:
                        files_info = fs.ls(model_name, detail=True)
                        for finfo in files_info:
                            fname = finfo['name'].split('/')[-1]
                            if fname.endswith('.gguf'):
                                file_sizes[fname] = finfo['size'] / (1024**3)  # GB
                    except:
                        pass

                    # Smart file selection: prioritize smaller quantized versions
                    # Download order: Q4_K_M > Q5_K_M > Q8_0 > F16 (smallest to largest)
                    quantization_priority = {
                        'q2_k': 1, 'q3_k_m': 2, 'q4_k_m': 3, 'q4_k_s': 4,
                        'q5_k_m': 5, 'q5_k_s': 6, 'q6_k': 7, 'q8_0': 8,
                        'f16': 99, 'f32': 100  # Full precision last
                    }

                    def get_priority(filename):
                        fname_lower = filename.lower()
                        for quant_type, priority in quantization_priority.items():
                            if quant_type in fname_lower:
                                return priority
                        return 50  # Unknown quantization

                    # Sort files by priority (download best quantized version first)
                    sorted_files = sorted(gguf_files, key=get_priority)

                    # Determine download directory: use GGUF models directory from config
                    # Read from config.yaml to get the correct GGUF models directory
                    from pathlib import Path
                    import yaml

                    gguf_dir = Path.home() / "models" / "gguf"  # Default
                    try:
                        config_path = Path("config.yaml")
                        if config_path.exists():
                            with open(config_path) as f:
                                cfg = yaml.safe_load(f)
                                if cfg and 'providers' in cfg and 'gguf' in cfg['providers']:
                                    gguf_models_dir = cfg['providers']['gguf'].get('models_dir', '~/models/gguf/')
                                    gguf_dir = Path(gguf_models_dir).expanduser()
                    except Exception as e:
                        logger.debug(f"Could not read GGUF models_dir from config, using default: {e}")

                    # Download to GGUF provider's directory so it can be discovered
                    download_dir = gguf_dir / model_name.replace("/", "--")
                    download_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Downloading to GGUF models directory: {download_dir}")

                    # Download the recommended file (smallest good quantization)
                    best_file = sorted_files[0]
                    file_size_str = f" ({file_sizes.get(best_file, 0):.2f} GB)" if best_file in file_sizes else ""
                    logger.info(f"Downloading recommended quantization: {best_file}{file_size_str}")

                    if len(sorted_files) > 1:
                        logger.info(f"Note: {len(sorted_files)-1} other quantization(s) available but not downloaded to save space")
                        other_files = [f"{f} ({file_sizes.get(f, 0):.1f}GB)" if f in file_sizes else f
                                     for f in sorted_files[1:]]
                        logger.info(f"Other versions: {', '.join(other_files)}")

                    # Download with visible progress bar
                    # hf_hub_download shows tqdm progress by default (not suppressed)
                    downloaded_path = hf_hub_download(
                        repo_id=model_name,
                        filename=best_file,
                        local_dir=str(download_dir),
                        local_dir_use_symlinks=False  # Direct copy for GGUF compatibility
                    )
                    logger.info(f"Downloaded {best_file} successfully")
                    logger.info(f"Location: {downloaded_path}")
                    logger.info(f"Use the GGUF or Quantized provider to load this model")

                    # Invalidate metadata cache for GGUF provider (will be re-inspected on discovery)
                    # This ensures the new model shows up with correct metadata
                    self.metadata_cache.invalidate_model(str(download_dir))
                    logger.debug(f"Invalidated GGUF metadata cache for {model_name}")

                    return True

            except Exception as e:
                # If we can't check repo files, assume it's a regular model
                logger.debug(f"Could not check for GGUF files: {e}")

            # STEP 2: Regular transformers model download
            # Import base classes (always available)
            from transformers import AutoModel, AutoTokenizer

            model_lower = model_name.lower()

            # Check if VLM (vision-language model)
            vlm_keywords = [
                "llava", "blip", "instructblip", "vision", "vl", "clip",
                "paligemma", "idefics", "fuyu", "kosmos", "qwen-vl", "qwen3-vl", "moondream"
            ]
            is_vlm = any(keyword in model_lower for keyword in vlm_keywords)

            if is_vlm:
                # Download processor for VLM (with visible progress)
                try:
                    from transformers import AutoProcessor
                    # Progress bars shown during download
                    AutoProcessor.from_pretrained(
                        model_name,
                        cache_dir=str(self.cache_dir),
                        trust_remote_code=True
                    )
                    logger.info("Downloaded VLM processor")
                except Exception as e:
                    logger.debug(f"Could not download processor: {e}")

            # Download tokenizer (with visible progress)
            try:
                # Progress bars shown during download
                AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True
                )
                logger.info("Downloaded tokenizer")
            except Exception as e:
                logger.debug(f"Could not download tokenizer: {e}")

            # Download model weights (with visible progress)
            # Use snapshot_download for robust handling of all model types:
            # - Models with custom configurations (e.g., moondream2)
            # - Standard transformers models
            # - VLMs and LLMs
            # This avoids configuration class validation errors while still downloading all files
            from huggingface_hub import snapshot_download
            
            # Progress bars shown during download
            snapshot_download(
                repo_id=model_name,
                cache_dir=str(self.cache_dir),
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.py"],
                ignore_patterns=["*.gguf", "*.md", "*.git*"]
            )

            logger.info(f"Successfully downloaded {model_name}")

            # Invalidate cache for this model so it gets re-inspected on next discovery
            model_dir = self.cache_dir / ("models--" + model_name.replace("/", "--"))
            if model_dir.exists():
                self.metadata_cache.invalidate_model(str(model_dir))
                logger.debug(f"Invalidated metadata cache for {model_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False

    def delete_model(self, model_id: str) -> bool:
        """Delete cached model from disk.

        Args:
            model_id: Model identifier

        Returns:
            True if successful
        """
        logger.info(f"Deleting HuggingFace model: {model_id}")

        try:
            # Convert model_id to directory name (org/name -> models--org--name)
            dir_name = "models--" + model_id.replace("/", "--")
            model_dir = self.cache_dir / dir_name

            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                logger.info(f"Deleted {model_id}")
                return True
            else:
                logger.warning(f"Model directory not found: {model_dir}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete {model_id}: {e}")
            return False

    def estimate_model_size(self, model_name: str) -> float:
        """Estimate model size (rough estimate based on name).

        Args:
            model_name: Model identifier

        Returns:
            Estimated size in GB
        """
        # Rough estimates based on common model sizes
        name_lower = model_name.lower()

        if "7b" in name_lower:
            return 14.0  # 7B models ~14GB fp16
        elif "13b" in name_lower:
            return 26.0
        elif "3b" in name_lower:
            return 6.0
        elif "1b" in name_lower:
            return 2.0
        else:
            return 8.0  # Default estimate

    # Convenience methods for app.py
    def qa(self, model_id: str, image_path: str, question: str) -> str:
        """QA convenience wrapper (requires manual handle management)."""
        raise NotImplementedError("Use load_model() then run_qa() directly")

    def caption(self, model_id: str, image_path: str, detail_level: str = "detailed") -> str:
        """Caption convenience wrapper (requires manual handle management)."""
        raise NotImplementedError("Use load_model() then run_caption() directly")

    def detect(self, model_id: str, image_path: str, prompt: str) -> str:
        """Detect convenience wrapper (requires manual handle management)."""
        raise NotImplementedError("Use load_model() then run_detect() directly")

    def point(self, model_id: str, image_path: str, object_name: str) -> str:
        """Point convenience wrapper (requires manual handle management)."""
        raise NotImplementedError("Use load_model() then run_point() directly")

    def chat(self, model_id: str, message: str) -> str:
        """Chat convenience wrapper (requires manual handle management)."""
        raise NotImplementedError("Use load_model() then run_text() directly")
