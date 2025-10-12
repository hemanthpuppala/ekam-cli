"""
VLM Client - Unified interface for Vision Language Models
Supports both Ollama and HuggingFace models with specialized endpoints
"""

import os
import sys
import time
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import yaml

# Model-specific imports
try:
    import ollama
    from ollama import AsyncClient as OllamaAsyncClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama not installed. Install with: pip install ollama")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers torch")


class VLMClient:
    """Unified VLM client supporting multiple providers and models"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize VLM client with configuration"""
        self.config = self._load_config(config_path)
        self.current_model = None
        self.current_provider = None
        self.model_instance = None
        self.tokenizer = None

        # Stats
        self.total_inferences = 0
        self.total_time = 0.0

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from config"""
        models = []
        for model_id, model_info in self.config.get('models', {}).items():
            models.append({
                'id': model_id,
                'name': model_info.get('name'),
                'provider': model_info.get('provider'),
                'size_gb': model_info.get('size_gb'),
                'capabilities': model_info.get('capabilities', []),
                'description': model_info.get('description', '')
            })
        return models

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        return self.config.get('models', {}).get(model_id)

    async def load_model(self, model_id: str) -> bool:
        """Load a specific model"""
        try:
            model_info = self.get_model_info(model_id)
            if not model_info:
                print(f"Error: Model '{model_id}' not found in config")
                return False

            provider = model_info.get('provider')
            model_name = model_info.get('name')

            print(f"Loading {model_name} ({provider})...")

            # Check if same model already loaded
            if self.current_model == model_id:
                print("Model already loaded")
                return True

            # Unload current model if any
            if self.model_instance:
                await self.unload_model()

            # Load based on provider
            if provider == 'ollama':
                success = await self._load_ollama_model(model_name)
            elif provider == 'huggingface':
                success = await self._load_huggingface_model(model_name, model_info)
            else:
                print(f"Error: Unknown provider '{provider}'")
                return False

            if success:
                self.current_model = model_id
                self.current_provider = provider
                print(f"✓ Model loaded successfully")
                return True
            else:
                print(f"✗ Failed to load model")
                return False

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    async def _load_ollama_model(self, model_name: str) -> bool:
        """Load an Ollama model"""
        if not OLLAMA_AVAILABLE:
            print("Error: ollama library not installed")
            return False

        try:
            host = self.config.get('providers', {}).get('ollama', {}).get('host', 'http://localhost:11434')
            self.model_instance = OllamaAsyncClient(host=host)

            # Test if model exists by checking the model list
            try:
                response = await self.model_instance.list()
                models = response.models

                # Check if the model name exists in the list
                model_exists = any(m.model == model_name for m in models)

                if not model_exists:
                    print(f"Model '{model_name}' not found. Run: ollama pull {model_name}")
                    return False

                # Model exists, no need for test inference
                return True

            except Exception as e:
                print(f"Error checking Ollama model: {e}")
                return False

        except Exception as e:
            print(f"Error loading Ollama model: {e}")
            return False

    async def _load_huggingface_model(self, model_name: str, model_info: Dict[str, Any]) -> bool:
        """Load a HuggingFace model"""
        if not TRANSFORMERS_AVAILABLE:
            print("Error: transformers library not installed")
            return False

        try:
            # Special handling for Moondream2
            if 'moondream' in model_name.lower():
                return await self._load_moondream_model(model_name)

            # Generic HuggingFace loading
            print("Loading HuggingFace model (this may take a while)...")
            cache_dir = os.path.expanduser(
                self.config.get('providers', {}).get('huggingface', {}).get('cache_dir', '~/.cache/huggingface')
            )

            # Load in separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model_instance = await loop.run_in_executor(
                None,
                self._load_hf_sync,
                model_name,
                cache_dir
            )

            return self.model_instance is not None

        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            return False

    def _load_hf_sync(self, model_name: str, cache_dir: str):
        """Synchronous HuggingFace model loading"""
        try:
            from transformers import AutoProcessor, AutoModel

            processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )

            device = self.config.get('providers', {}).get('huggingface', {}).get('device', 'cuda')
            use_fp16 = self.config.get('providers', {}).get('huggingface', {}).get('use_fp16', True)

            # Use AutoModel with trust_remote_code for modern VLMs (Florence-2, BLIP, etc.)
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                torch_dtype=torch.float16 if use_fp16 and device == 'cuda' else torch.float32
            )

            # Handle device placement
            if torch.cuda.is_available() and device == 'cuda':
                model = model.to('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and device == 'mps':
                model = model.to('mps')

            model.eval()

            return {'model': model, 'processor': processor}

        except Exception as e:
            print(f"Error in sync HF loading: {e}")
            return None

    async def _load_moondream_model(self, model_name: str) -> bool:
        """Load Moondream2 model with special methods"""
        try:
            print("Loading Moondream2 with special methods support...")
            cache_dir = os.path.expanduser(
                self.config.get('providers', {}).get('huggingface', {}).get('cache_dir', '~/.cache/huggingface')
            )

            loop = asyncio.get_event_loop()
            self.model_instance = await loop.run_in_executor(
                None,
                self._load_moondream_sync,
                model_name,
                cache_dir
            )

            return self.model_instance is not None

        except Exception as e:
            print(f"Error loading Moondream: {e}")
            return False

    def _load_moondream_sync(self, model_name: str, cache_dir: str):
        """Synchronous Moondream loading"""
        try:
            device = self.config.get('providers', {}).get('huggingface', {}).get('device', 'cuda')

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            if torch.cuda.is_available() and device == 'cuda':
                model = model.to('cuda')

            model.eval()

            return {'model': model, 'tokenizer': tokenizer}

        except Exception as e:
            print(f"Error in sync Moondream loading: {e}")
            return None

    async def unload_model(self):
        """Unload current model and free memory"""
        if self.model_instance:
            if self.current_provider == 'huggingface' and TRANSFORMERS_AVAILABLE:
                if isinstance(self.model_instance, dict):
                    # Delete model components
                    if 'model' in self.model_instance:
                        del self.model_instance['model']
                    if 'processor' in self.model_instance:
                        del self.model_instance['processor']
                    if 'tokenizer' in self.model_instance:
                        del self.model_instance['tokenizer']

                # Clear GPU/CPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # Clear MPS cache (Apple Silicon)
                    try:
                        torch.mps.empty_cache()
                    except:
                        pass

            self.model_instance = None
            self.current_model = None
            self.current_provider = None

    async def cleanup_all_models(self):
        """Cleanup all loaded models and free memory (call at startup)"""
        # Unload any currently loaded model
        await self.unload_model()

        # Force garbage collection and memory cleanup
        if TRANSFORMERS_AVAILABLE:
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                    torch.mps.synchronize()
                except:
                    pass

    # ========== Endpoint Methods ==========

    async def qa(self, image_path: str, question: str, **params) -> Dict[str, Any]:
        """Question Answering endpoint"""
        if not self.model_instance:
            return {"error": "No model loaded"}

        try:
            # Load and validate image
            image = self._load_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}

            # Check if it's moondream (needs special prompt format)
            is_moondream = 'moondream' in self.current_model.lower()

            if is_moondream:
                # Moondream works with open-ended descriptive prompts
                # Convert question to descriptive format
                prompt = f"Describe what you see in this image. {question}"
            else:
                # Build prompt from template for other models
                template = self.config.get('endpoints', {}).get('qa', {}).get('prompt_template',
                                                                              "Question: {question}\nAnswer:")
                prompt = template.format(question=question)

            # Run inference
            start_time = time.time()
            result = await self._infer(image, prompt, **params)
            inference_time = time.time() - start_time

            # Update stats
            self.total_inferences += 1
            self.total_time += inference_time

            return {
                "endpoint": "qa",
                "question": question,
                "answer": result.get('response', ''),
                "inference_time_ms": round(inference_time * 1000, 2),
                "model": self.current_model,
                "provider": self.current_provider,
                "metadata": result.get('metadata', {})
            }

        except Exception as e:
            return {"error": str(e)}

    async def caption(self, image_path: str, detail_level: str = "detailed", **params) -> Dict[str, Any]:
        """Image Captioning endpoint"""
        if not self.model_instance:
            return {"error": "No model loaded"}

        try:
            image = self._load_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}

            # Build prompt based on detail level
            endpoints_config = self.config.get('endpoints', {}).get('caption', {})
            if detail_level == "short":
                prompt = endpoints_config.get('short_caption_template',
                                             "Describe this image briefly in one sentence.")
            else:
                prompt = endpoints_config.get('prompt_template',
                                             "Describe this image in detail.")

            start_time = time.time()
            result = await self._infer(image, prompt, **params)
            inference_time = time.time() - start_time

            self.total_inferences += 1
            self.total_time += inference_time

            return {
                "endpoint": "caption",
                "caption": result.get('response', ''),
                "detail_level": detail_level,
                "inference_time_ms": round(inference_time * 1000, 2),
                "model": self.current_model,
                "provider": self.current_provider,
                "metadata": result.get('metadata', {})
            }

        except Exception as e:
            return {"error": str(e)}

    async def detect(self, image_path: str, object_name: str, **params) -> Dict[str, Any]:
        """Object Detection endpoint"""
        if not self.model_instance:
            return {"error": "No model loaded"}

        try:
            image = self._load_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}

            # Check if model supports special detect method (e.g., Moondream)
            model_info = self.get_model_info(self.current_model)
            if model_info and model_info.get('special_methods') and 'moondream' in model_info.get('name', '').lower():
                # Use Moondream's detect method
                result = await self._moondream_detect(image, object_name)
            else:
                # Check if it's moondream (even if loaded dynamically)
                is_moondream = 'moondream' in self.current_model.lower()

                if is_moondream:
                    # Moondream works best with "Describe what you see..." format
                    prompt = f"Describe what you see in this image, focusing on {object_name}."
                else:
                    # Use generic prompt-based detection for other models
                    template = self.config.get('endpoints', {}).get('detect', {}).get('prompt_template',
                        "Detect and locate all instances of '{object}' in this image. Provide their locations.")
                    prompt = template.format(object=object_name)

                start_time = time.time()
                result = await self._infer(image, prompt, **params)
                inference_time = time.time() - start_time

                self.total_inferences += 1
                self.total_time += inference_time

                return {
                    "endpoint": "detect",
                    "object": object_name,
                    "detections": result.get('response', ''),
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "model": self.current_model,
                    "provider": self.current_provider,
                    "metadata": result.get('metadata', {})
                }

            return result

        except Exception as e:
            return {"error": str(e)}

    async def point(self, image_path: str, object_name: str, **params) -> Dict[str, Any]:
        """Object Pointing/Localization endpoint"""
        if not self.model_instance:
            return {"error": "No model loaded"}

        try:
            image = self._load_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}

            # Check for Moondream special method
            model_info = self.get_model_info(self.current_model)
            if model_info and model_info.get('special_methods') and 'moondream' in model_info.get('name', '').lower():
                result = await self._moondream_point(image, object_name)
            else:
                # Check if it's moondream (even if loaded dynamically)
                is_moondream = 'moondream' in self.current_model.lower()

                if is_moondream:
                    # Moondream works best with "Describe what you see..." format
                    prompt = f"Describe what you see in this image, focusing on the location and position of {object_name}."
                else:
                    # Generic prompt-based pointing for other models
                    template = self.config.get('endpoints', {}).get('point', {}).get('prompt_template',
                        "Where is the '{object}' in this image? Provide the coordinates.")
                    prompt = template.format(object=object_name)

                start_time = time.time()
                result = await self._infer(image, prompt, **params)
                inference_time = time.time() - start_time

                self.total_inferences += 1
                self.total_time += inference_time

                return {
                    "endpoint": "point",
                    "object": object_name,
                    "location": result.get('response', ''),
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "model": self.current_model,
                    "provider": self.current_provider,
                    "metadata": result.get('metadata', {})
                }

            return result

        except Exception as e:
            return {"error": str(e)}

    # ========== Helper Methods ==========

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and validate image"""
        try:
            path = Path(image_path)
            if not path.exists():
                print(f"Error: Image not found: {image_path}")
                return None

            image = Image.open(path)

            # Auto-resize if configured
            if self.config.get('image', {}).get('auto_resize', True):
                max_size = self.config.get('image', {}).get('max_size', 1920)
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image

        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    async def _infer(self, image: Image.Image, prompt: str, **params) -> Dict[str, Any]:
        """Run inference based on provider"""
        if self.current_provider == 'ollama':
            return await self._infer_ollama(image, prompt, **params)
        elif self.current_provider == 'huggingface':
            return await self._infer_huggingface(image, prompt, **params)
        else:
            return {"error": "Unknown provider"}

    async def _infer_ollama(self, image: Image.Image, prompt: str, **params) -> Dict[str, Any]:
        """Run Ollama inference"""
        try:
            # Convert image to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode()

            # Get model name
            model_info = self.get_model_info(self.current_model)
            if model_info:
                # Model loaded from config
                model_name = model_info.get('name')
            else:
                # Model loaded dynamically (not in config)
                model_name = self.current_model

            # Merge parameters
            inference_params = self.config.get('inference', {}).copy()
            inference_params.update(params)

            options = {
                "temperature": inference_params.get('temperature', 0.3),
                "top_p": inference_params.get('top_p', 0.9),
                "top_k": inference_params.get('top_k', 40),
                "seed": inference_params.get('seed', 42),
                "num_predict": inference_params.get('max_tokens', 500),
            }

            # Run inference
            response = await self.model_instance.generate(
                model=model_name,
                prompt=prompt,
                images=[image_b64],
                options=options,
                stream=False
            )

            # Response is a Pydantic model, access attributes directly
            response_text = response['response'] if isinstance(response, dict) else getattr(response, 'response', '')
            total_duration = response['total_duration'] if isinstance(response, dict) else getattr(response, 'total_duration', 0)
            eval_count = response['eval_count'] if isinstance(response, dict) else getattr(response, 'eval_count', 0)

            return {
                "response": response_text,
                "metadata": {
                    "total_duration_ms": total_duration / 1e6 if total_duration else 0,
                    "eval_count": eval_count
                }
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Ollama inference error: {error_details}")
            return {"error": str(e)}

    async def _infer_huggingface(self, image: Image.Image, prompt: str, **params) -> Dict[str, Any]:
        """Run HuggingFace inference"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._infer_hf_sync,
                image,
                prompt,
                params
            )
            return result

        except Exception as e:
            return {"error": str(e)}

    def _infer_hf_sync(self, image: Image.Image, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous HuggingFace inference"""
        try:
            model = self.model_instance['model']

            # Check if it's Moondream
            if 'tokenizer' in self.model_instance:
                # Moondream inference
                tokenizer = self.model_instance['tokenizer']
                enc_image = model.encode_image(image)

                response = model.answer_question(
                    image_embeds=enc_image,
                    question=prompt,
                    tokenizer=tokenizer
                )
            else:
                # Generic HF inference
                processor = self.model_instance['processor']

                inference_params = self.config.get('inference', {}).copy()
                inference_params.update(params)

                inputs = processor(images=image, text=prompt, return_tensors="pt")

                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=inference_params.get('max_tokens', 500),
                        temperature=inference_params.get('temperature', 0.3),
                        top_p=inference_params.get('top_p', 0.9),
                        top_k=inference_params.get('top_k', 40),
                        do_sample=inference_params.get('temperature', 0.3) > 0
                    )

                response = processor.decode(outputs[0], skip_special_tokens=True)

            return {
                "response": response,
                "metadata": {}
            }

        except Exception as e:
            return {"error": str(e)}

    async def _moondream_detect(self, image: Image.Image, object_name: str) -> Dict[str, Any]:
        """Use Moondream's detect method"""
        try:
            loop = asyncio.get_event_loop()
            start_time = time.time()

            result = await loop.run_in_executor(
                None,
                self._moondream_detect_sync,
                image,
                object_name
            )

            inference_time = time.time() - start_time
            self.total_inferences += 1
            self.total_time += inference_time

            result['inference_time_ms'] = round(inference_time * 1000, 2)
            result['model'] = self.current_model
            result['provider'] = self.current_provider

            return result

        except Exception as e:
            return {"error": str(e)}

    def _moondream_detect_sync(self, image: Image.Image, object_name: str) -> Dict[str, Any]:
        """Synchronous Moondream detect"""
        try:
            model = self.model_instance['model']
            enc_image = model.encode_image(image)

            # Moondream detect returns bounding boxes
            detections = model.detect(enc_image, object_name)

            return {
                "endpoint": "detect",
                "object": object_name,
                "detections": detections,
                "metadata": {}
            }

        except Exception as e:
            return {"error": str(e)}

    async def _moondream_point(self, image: Image.Image, object_name: str) -> Dict[str, Any]:
        """Use Moondream's point method"""
        try:
            loop = asyncio.get_event_loop()
            start_time = time.time()

            result = await loop.run_in_executor(
                None,
                self._moondream_point_sync,
                image,
                object_name
            )

            inference_time = time.time() - start_time
            self.total_inferences += 1
            self.total_time += inference_time

            result['inference_time_ms'] = round(inference_time * 1000, 2)
            result['model'] = self.current_model
            result['provider'] = self.current_provider

            return result

        except Exception as e:
            return {"error": str(e)}

    def _moondream_point_sync(self, image: Image.Image, object_name: str) -> Dict[str, Any]:
        """Synchronous Moondream point"""
        try:
            model = self.model_instance['model']
            enc_image = model.encode_image(image)

            # Moondream point returns coordinates
            coordinates = model.point(enc_image, object_name)

            return {
                "endpoint": "point",
                "object": object_name,
                "coordinates": coordinates,
                "metadata": {}
            }

        except Exception as e:
            return {"error": str(e)}

    # ========== System Detection & Compatibility ==========

    def get_system_specs(self) -> Dict[str, Any]:
        """Get system specifications"""
        try:
            import psutil
            import platform

            specs = {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
            }

            # CPU Info
            specs['cpu_count'] = psutil.cpu_count(logical=False)
            specs['cpu_count_logical'] = psutil.cpu_count(logical=True)
            specs['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else 0

            # Memory Info
            mem = psutil.virtual_memory()
            specs['ram_total_gb'] = mem.total / (1024**3)
            specs['ram_available_gb'] = mem.available / (1024**3)
            specs['ram_used_gb'] = mem.used / (1024**3)
            specs['ram_percent'] = mem.percent

            # GPU Info (PyTorch)
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                specs['gpu_available'] = True
                specs['gpu_name'] = torch.cuda.get_device_name(0)
                specs['gpu_count'] = torch.cuda.device_count()
                specs['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Try to get available memory
                try:
                    specs['gpu_memory_available_gb'] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
                except:
                    specs['gpu_memory_available_gb'] = specs['gpu_memory_total_gb']
            else:
                # Check for Apple Silicon (MPS)
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    specs['gpu_available'] = True
                    specs['gpu_name'] = 'Apple Silicon (MPS)'
                    specs['gpu_type'] = 'unified_memory'
                    # On Apple Silicon, GPU shares system RAM
                    specs['gpu_memory_total_gb'] = specs['ram_total_gb']
                    specs['gpu_memory_available_gb'] = specs['ram_available_gb']
                else:
                    specs['gpu_available'] = False
                    specs['gpu_name'] = 'CPU Only'

            # Recommended model size (realistic based on total RAM)
            # Use percentage-based approach for better scaling
            total_ram = specs['ram_total_gb']
            available_ram = specs['ram_available_gb']

            if total_ram <= 4:
                # Small systems: leave 1GB for system
                safety_margin = 1.0
            elif total_ram <= 8:
                # Medium systems (like yours): leave 0.5GB for system
                safety_margin = 0.5
            elif total_ram <= 16:
                # Larger systems: leave 1GB for system
                safety_margin = 1.0
            else:
                # Very large systems: leave 2GB for system
                safety_margin = 2.0

            # Calculate recommended size
            recommended = available_ram - safety_margin

            # Ensure it's at least 0.5GB
            specs['recommended_model_size_gb'] = max(0.5, recommended)

            return specs

        except Exception as e:
            print(f"Error getting system specs: {e}")
            return {
                'platform': 'Unknown',
                'cpu_count': 0,
                'ram_total_gb': 0,
                'ram_available_gb': 0,
                'gpu_available': False,
                'recommended_model_size_gb': 4
            }

    def check_model_compatibility(self, model_size_gb: float) -> Dict[str, Any]:
        """Check if a model is compatible with the system"""
        specs = self.get_system_specs()
        available_ram = specs['ram_available_gb']
        recommended = specs['recommended_model_size_gb']

        result = {
            'compatible': True,
            'status': 'perfect',
            'icon': '✅',
            'message': 'Perfect fit',
            'warnings': []
        }

        if model_size_gb <= recommended * 0.7:
            # Plenty of room
            result['status'] = 'perfect'
            result['icon'] = '✅'
            result['message'] = 'Perfect fit'

        elif model_size_gb <= recommended:
            # Good fit
            result['status'] = 'good'
            result['icon'] = '✅'
            result['message'] = 'Good fit'

        elif model_size_gb <= recommended * 1.2:
            # Tight fit
            result['status'] = 'tight'
            result['icon'] = '⚠️'
            result['message'] = 'Tight fit (90%+ RAM)'
            result['warnings'].append('May use significant RAM')

        elif model_size_gb <= available_ram:
            # Will work but might use swap
            result['status'] = 'swap'
            result['icon'] = '💾'
            result['message'] = 'May use swap memory'
            result['warnings'].append('Performance may be affected')
            result['warnings'].append('Consider smaller model')

        else:
            # Too large
            result['compatible'] = False
            result['status'] = 'too_large'
            result['icon'] = '❌'
            result['message'] = 'Too large for system'
            result['warnings'].append(f'Needs {model_size_gb:.1f}GB, only {available_ram:.1f}GB available')
            result['warnings'].append('Consider quantized version or smaller model')

        return result

    # ========== Model Discovery & Installation ==========

    async def discover_installed_ollama_models(self) -> List[Dict[str, Any]]:
        """Discover ALL installed Ollama models (VLM, LLM, Embedding)"""
        if not OLLAMA_AVAILABLE:
            print("Error: ollama library not installed")
            return []

        try:
            host = self.config.get('providers', {}).get('ollama', {}).get('host', 'http://localhost:11434')
            client = OllamaAsyncClient(host=host)

            # Test connection first with a simple request
            try:
                response = await client.list()
            except Exception as conn_err:
                # Check if it's a connection error
                error_msg = str(conn_err).lower()
                if 'connection' in error_msg or 'refused' in error_msg or 'failed to connect' in error_msg:
                    raise ConnectionError(
                        "Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. "
                        "https://ollama.com/download"
                    )
                else:
                    raise

            models = response.models  # Pydantic model, not dict

            discovered = []
            for model in models:
                # Access attributes directly (Pydantic model)
                name = model.model  # The 'model' attribute contains the name
                size_gb = model.size / 1e9
                modified = model.modified_at

                # Detect model type
                model_type = self.detect_model_type(name, 'ollama')

                discovered.append({
                    'name': name,
                    'provider': 'ollama',
                    'size_gb': size_gb,
                    'type': model_type,
                    'modified': str(modified),
                    'installed': True
                })

            return discovered

        except ConnectionError as e:
            # Re-raise connection errors with clear message
            print(f"Error discovering Ollama models: {e}")
            raise
        except Exception as e:
            print(f"Error discovering Ollama models: {e}")
            return []

    async def discover_installed_hf_models(self) -> List[Dict[str, Any]]:
        """Discover ALL installed HuggingFace models (VLM, LLM, Embedding)"""
        if not TRANSFORMERS_AVAILABLE:
            return []

        try:
            cache_dir = os.path.expanduser(
                self.config.get('providers', {}).get('huggingface', {}).get('cache_dir', '~/.cache/huggingface')
            )

            cache_path = Path(cache_dir) / "hub"
            if not cache_path.exists():
                return []

            discovered = []
            for model_dir in cache_path.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith('models--'):
                    # Extract model name from directory
                    model_name = model_dir.name.replace('models--', '').replace('--', '/')

                    # Check if model has snapshots (indicates it's properly downloaded)
                    snapshots_dir = model_dir / "snapshots"
                    if snapshots_dir.exists():
                        snapshots = list(snapshots_dir.iterdir())
                        if snapshots:
                            snapshot_path = snapshots[0]

                            # Verify it has config or processor (real model, not just metadata)
                            has_processor = (snapshot_path / "preprocessor_config.json").exists()
                            has_config = (snapshot_path / "config.json").exists()
                            has_model_files = any(
                                f.suffix in ['.bin', '.safetensors', '.pt', '.pth']
                                for f in snapshot_path.rglob('*') if f.is_file()
                            )

                            if (has_processor or has_config) and has_model_files:
                                # Get size
                                size_bytes = sum(f.stat().st_size for f in snapshot_path.rglob('*') if f.is_file())

                                # Detect model type
                                model_type = self.detect_model_type(model_name, 'huggingface')

                                discovered.append({
                                    'name': model_name,
                                    'provider': 'huggingface',
                                    'size_gb': size_bytes / 1e9,
                                    'type': model_type,
                                    'installed': True
                                })

            return discovered

        except Exception as e:
            print(f"Error discovering HuggingFace models: {e}")
            return []

    def _is_valid_vision_model_config(self, config_path: Path) -> bool:
        """Check if model config indicates it's a vision model"""
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Check model_type or architectures for vision models
            model_type = config.get('model_type', '').lower()
            architectures = config.get('architectures', [])

            # Known vision model types
            vision_types = ['florence', 'blip', 'llava', 'qwen', 'paligemma', 'idefics', 'kosmos', 'git', 'vit']

            # Check model type
            if any(vtype in model_type for vtype in vision_types):
                return True

            # Check architectures
            for arch in architectures:
                arch_lower = arch.lower()
                if any(vtype in arch_lower for vtype in vision_types):
                    return True
                if 'vision' in arch_lower or 'image' in arch_lower:
                    return True

            return False

        except Exception as e:
            # If we can't read config, allow it (benefit of doubt)
            return True

    def detect_model_type(self, model_name: str, provider: str) -> str:
        """Detect if a model is VLM, LLM, or Embedding model

        Args:
            model_name: Name of the model
            provider: Provider (ollama or huggingface)

        Returns:
            "VLM", "LLM", or "Embedding"
        """
        try:
            model_name_lower = model_name.lower()

            # Check for embedding models first (highest priority)
            embedding_keywords = ['embed', 'embedding', 'sentence-transformer']
            if any(keyword in model_name_lower for keyword in embedding_keywords):
                return "Embedding"

            if provider == "ollama":
                # Ollama: Check name for vision keywords
                vision_keywords = ['vision', '-vl', 'vl:', 'llava', 'moondream']
                if any(keyword in model_name_lower for keyword in vision_keywords):
                    return "VLM"
                else:
                    return "LLM"

            elif provider == "huggingface":
                # HuggingFace: Check config file
                cache_dir = os.path.expanduser(
                    self.config.get('providers', {}).get('huggingface', {}).get('cache_dir', '~/.cache/huggingface')
                )

                dir_name = "models--" + model_name.replace('/', '--')
                model_path = Path(cache_dir) / "hub" / dir_name

                if model_path.exists():
                    snapshots_dir = model_path / "snapshots"
                    if snapshots_dir.exists():
                        snapshots = list(snapshots_dir.iterdir())
                        if snapshots:
                            config_path = snapshots[0] / "config.json"
                            if config_path.exists():
                                if self._is_valid_vision_model_config(config_path):
                                    return "VLM"
                                else:
                                    return "LLM"

                # Fallback: Check name for vision keywords
                vision_keywords = ['florence', 'blip', 'llava', 'qwen', 'vision', 'vl', 'paligemma', 'idefics', 'kosmos']
                if any(keyword in model_name_lower for keyword in vision_keywords):
                    return "VLM"
                else:
                    return "LLM"

            return "LLM"  # Default to LLM

        except Exception as e:
            print(f"Error detecting model type: {e}")
            return "LLM"  # Safe default

    async def install_ollama_model(self, model_name: str, progress_callback=None) -> bool:
        """Install an Ollama model with progress tracking"""
        if not OLLAMA_AVAILABLE:
            print("Error: ollama library not installed")
            return False

        try:
            # Use synchronous client for streaming progress
            sync_client = ollama.Client(
                host=self.config.get('providers', {}).get('ollama', {}).get('host', 'http://localhost:11434')
            )

            print(f"Pulling model: {model_name}")

            for progress in sync_client.pull(model_name, stream=True):
                if progress_callback:
                    progress_callback(progress)

                # Display progress
                status = progress.get('status', '')
                if 'total' in progress and 'completed' in progress:
                    total = progress['total']
                    completed = progress['completed']
                    percent = (completed / total) * 100 if total > 0 else 0
                    print(f"\r{status}: {percent:.1f}% ({completed/1e6:.1f}MB / {total/1e6:.1f}MB)", end='', flush=True)
                else:
                    print(f"\r{status}", end='', flush=True)

            print("\n✓ Model downloaded successfully")
            return True

        except Exception as e:
            print(f"\n✗ Error installing model: {e}")
            return False

    async def install_hf_model(self, model_name: str, progress_callback=None) -> bool:
        """Install a HuggingFace model with progress tracking"""
        if not TRANSFORMERS_AVAILABLE:
            print("Error: transformers library not installed")
            return False

        try:
            from transformers import AutoModel, AutoProcessor

            cache_dir = os.path.expanduser(
                self.config.get('providers', {}).get('huggingface', {}).get('cache_dir', '~/.cache/huggingface')
            )

            print(f"Downloading model: {model_name}")
            print("(This may take a while...)")

            # Download in executor to avoid blocking
            loop = asyncio.get_event_loop()

            def download():
                try:
                    # Download processor/tokenizer first
                    AutoProcessor.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        trust_remote_code=True
                    )

                    # Download model
                    AutoModel.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        trust_remote_code=True
                    )

                    return True
                except Exception as e:
                    print(f"Error downloading: {e}")
                    return False

            success = await loop.run_in_executor(None, download)

            if success:
                print("✓ Model downloaded successfully")

            return success

        except Exception as e:
            print(f"✗ Error installing model: {e}")
            return False

    # ========== Model Deletion ==========

    async def delete_ollama_model(self, model_name: str) -> bool:
        """Delete an Ollama model"""
        if not OLLAMA_AVAILABLE:
            print("Error: ollama library not installed")
            return False

        try:
            # Use synchronous client
            sync_client = ollama.Client(
                host=self.config.get('providers', {}).get('ollama', {}).get('host', 'http://localhost:11434')
            )

            # Delete the model
            sync_client.delete(model_name)
            print(f"✓ Deleted Ollama model: {model_name}")
            return True

        except Exception as e:
            print(f"✗ Error deleting Ollama model: {e}")
            return False

    async def delete_hf_model(self, model_name: str) -> bool:
        """Delete a HuggingFace model from cache"""
        try:
            cache_dir = os.path.expanduser(
                self.config.get('providers', {}).get('huggingface', {}).get('cache_dir', '~/.cache/huggingface')
            )

            # Convert model name to directory format
            dir_name = "models--" + model_name.replace('/', '--')
            model_path = Path(cache_dir) / "hub" / dir_name

            if not model_path.exists():
                print(f"Model not found: {model_name}")
                return False

            # Delete the directory
            import shutil
            shutil.rmtree(model_path)
            print(f"✓ Deleted HuggingFace model: {model_name}")
            return True

        except Exception as e:
            print(f"✗ Error deleting HuggingFace model: {e}")
            return False

    def get_model_disk_usage(self, model_name: str, provider: str) -> float:
        """Get disk usage of a model in GB"""
        try:
            if provider == 'ollama':
                # Ollama models stored in ~/.ollama/models
                # Size already provided by ollama list command
                return 0.0  # Return 0 as placeholder, actual size from discovery

            elif provider == 'huggingface':
                cache_dir = os.path.expanduser(
                    self.config.get('providers', {}).get('huggingface', {}).get('cache_dir', '~/.cache/huggingface')
                )

                dir_name = "models--" + model_name.replace('/', '--')
                model_path = Path(cache_dir) / "hub" / dir_name

                if model_path.exists():
                    size_bytes = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                    return size_bytes / (1024**3)

            return 0.0

        except Exception as e:
            print(f"Error getting disk usage: {e}")
            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "current_model": self.current_model,
            "current_provider": self.current_provider,
            "total_inferences": self.total_inferences,
            "total_time_seconds": round(self.total_time, 2),
            "average_time_ms": round((self.total_time / max(1, self.total_inferences)) * 1000, 2)
        }
