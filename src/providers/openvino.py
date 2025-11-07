"""OpenVINO provider for Intel CPU/iGPU optimized VLM inference.

OpenVINO 2025.2 features:
- VLM support: Qwen2-VL, Phi-3.5-Vision, InternVL2
- NPU acceleration for AI PCs
- INT4/INT8 quantization with minimal quality loss
- Optimized for Intel CPUs (AVX-512 VNNI, Intel AMX)
- Integrated GPU (iGPU) support with XMX

Research: https://docs.openvino.ai/2025/
VLMs: https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching_vlm.html
"""

import platform
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from PIL import Image

from ..models.endpoints import CompatibilityStatus, EndpointType, ModelType
from ..models.model import ModelInfo
from ..models.provider import ProviderConfig
from ..utils.history_formatter import format_qa_history
from .base import BaseProvider


class OpenVINOProvider(BaseProvider):
    """OpenVINO provider for Intel-optimized inference.

    Supports both CPUs (with AVX-512, Intel AMX) and Intel iGPUs.
    Includes VLM support for Qwen2-VL, Phi-3.5-Vision, and InternVL2.
    """

    def __init__(self, config: ProviderConfig):
        """Initialize OpenVINO provider.

        Args:
            config: Provider configuration

        Raises:
            RuntimeError: If OpenVINO not available
        """
        self.config = config
        self.models_dir = Path(str(config.models_dir or "~/.cache/openvino/models")).expanduser()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Check platform
        self.platform_info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

        # Verify OpenVINO is available
        try:
            import openvino as ov  # noqa: F401
            from optimum.intel import OVModelForCausalLM  # noqa: F401
            logger.info("✓ OpenVINO and Optimum-Intel available")
            self.openvino_available = True
        except ImportError as e:
            logger.error(f"OpenVINO not installed: {e}")
            raise RuntimeError(
                "OpenVINO not installed. Install with:\n"
                "pip install openvino openvino-dev optimum[openvino]"
            )

        # Check for Intel hardware optimization
        self._check_intel_optimizations()

        # Initialize OpenVINO core
        self.ov_core = None
        try:
            import openvino as ov
            self.ov_core = ov.Core()
            available_devices = self.ov_core.available_devices
            logger.info(f"✓ OpenVINO devices available: {available_devices}")

            # Set preferred device (CPU, GPU, NPU)
            self.device = self._select_best_device(available_devices)
            logger.info(f"✓ Selected device: {self.device}")

        except Exception as e:
            logger.warning(f"Could not initialize OpenVINO core: {e}")
            self.device = "CPU"

        logger.info("OpenVINO provider initialized for Intel-optimized inference")

    def _check_intel_optimizations(self):
        """Check for Intel-specific CPU optimizations."""
        import cpuinfo

        try:
            cpu_info = cpuinfo.get_cpu_info()
            cpu_flags = cpu_info.get("flags", [])

            # Check for performance-enhancing instruction sets
            optimizations = []
            if "avx512" in " ".join(cpu_flags).lower():
                optimizations.append("AVX-512")
            if "vnni" in " ".join(cpu_flags).lower():
                optimizations.append("VNNI")
            if "amx" in " ".join(cpu_flags).lower():
                optimizations.append("Intel AMX")

            if optimizations:
                logger.info(f"✓ Intel CPU optimizations detected: {', '.join(optimizations)}")
            else:
                logger.debug("No Intel-specific optimizations detected (will use standard CPU)")

        except Exception as e:
            logger.debug(f"Could not detect CPU features: {e}")

    def _select_best_device(self, available_devices: list) -> str:
        """Select best OpenVINO device.

        Priority: NPU > GPU > CPU

        Args:
            available_devices: List of available devices

        Returns:
            Device string
        """
        # Priority order for 2025
        device_priority = ["NPU", "GPU", "CPU"]

        for device_type in device_priority:
            for device in available_devices:
                if device_type in device:
                    return device

        return "CPU"  # Default fallback

    def discover_models(self) -> list[ModelInfo]:
        """Discover cached OpenVINO models.

        Returns:
            List of ModelInfo objects for available models
        """
        models = []

        if not self.models_dir.exists():
            logger.debug(f"OpenVINO models directory not found: {self.models_dir}")
            return models

        try:
            for model_dir in self.models_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                # Check for OpenVINO IR format files
                has_xml = (model_dir / "openvino_model.xml").exists()
                has_bin = (model_dir / "openvino_model.bin").exists()

                if has_xml and has_bin:
                    model_id = model_dir.name
                    size_gb = self._estimate_model_size(model_dir)

                    # Detect if VLM
                    is_vlm = self._is_vlm_model(model_id)

                    if is_vlm:
                        model_type = ModelType.VLM
                        capabilities = [
                            EndpointType.QA,
                            EndpointType.CAPTION,
                            EndpointType.TEXT,
                        ]
                    else:
                        model_type = ModelType.LLM
                        capabilities = [EndpointType.TEXT]

                    models.append(
                        ModelInfo(
                            model_id=f"openvino/{model_id}",
                            name=f"{model_id} (OpenVINO)",
                            provider="openvino",
                            size_gb=size_gb,
                            model_type=model_type,
                            capabilities=capabilities,
                            compatibility=CompatibilityStatus.PERFECT_FIT,
                            compatibility_message="Optimized for Intel CPUs",
                            is_installed=True,
                        )
                    )
                    logger.debug(f"Discovered OpenVINO model: {model_id}")

        except Exception as e:
            logger.error(f"Error discovering OpenVINO models: {e}")

        logger.info(f"Total OpenVINO models discovered: {len(models)}")
        return models

    def _estimate_model_size(self, model_dir: Path) -> float:
        """Estimate model size in GB."""
        try:
            total_size = 0
            for file in model_dir.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size / (1024**3)
        except Exception:
            return 2.0

    def _is_vlm_model(self, model_id: str) -> bool:
        """Check if model is a VLM."""
        model_lower = model_id.lower()
        vlm_keywords = [
            "qwen2-vl", "qwen-vl", "phi-3-vision", "phi-3.5-vision",
            "internvl", "llava", "cogvlm", "minicpm-v"
        ]
        return any(keyword in model_lower for keyword in vlm_keywords)

    def load_model(self, model_id: str, device: str) -> Any:
        """Load OpenVINO model.

        Args:
            model_id: Model identifier or path
            device: Ignored (uses self.device)

        Returns:
            Tuple of (model, tokenizer/processor)
        """
        logger.info(f"Loading OpenVINO model: {model_id}")

        try:
            from optimum.intel import OVModelForCausalLM
            from transformers import AutoTokenizer, AutoProcessor

            # Determine if VLM
            is_vlm = self._is_vlm_model(model_id)

            # Load processor/tokenizer
            if is_vlm:
                try:
                    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                    logger.debug("Loaded VLM processor")
                except Exception:
                    processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                    logger.debug("Loaded tokenizer (processor failed)")
            else:
                processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

            # Load OpenVINO model
            model = OVModelForCausalLM.from_pretrained(
                model_id,
                device=self.device,
                ov_config={"PERFORMANCE_HINT": "LATENCY"},  # Optimize for low latency
                trust_remote_code=True,
            )

            logger.info(f"✓ Loaded OpenVINO model on {self.device}")
            return (model, processor)

        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            raise RuntimeError(f"Could not load OpenVINO model {model_id}: {e}")

    def unload_model(self, handle: Any) -> None:
        """Unload OpenVINO model."""
        logger.debug("OpenVINO model will be garbage collected")

    def run_qa(
        self,
        handle: Any,
        image: Image.Image,
        question: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Run QA with OpenVINO VLM.

        Args:
            handle: Model handle (model, processor)
            image: PIL Image
            question: Question text
            conversation_history: Optional conversation history

        Returns:
            Answer text
        """
        model, processor = handle

        try:
            # Format question with history
            if conversation_history:
                formatted_question = format_qa_history(
                    conversation_history=conversation_history,
                    current_question=question,
                    max_turns=5
                )
            else:
                formatted_question = question

            # For VLMs with vision support
            if hasattr(processor, "apply_chat_template"):
                # Modern VLMs with chat template
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": formatted_question}
                    ]
                }]

                text_prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
            else:
                # Standard text + image format
                inputs = processor(text=formatted_question, images=image, return_tensors="pt")

            # Generate
            logger.info("Generating response with OpenVINO VLM...")
            import torch
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=1024)

            # Decode
            response = processor.batch_decode(output, skip_special_tokens=True)[0]
            logger.info("✓ Response generated")

            return response.strip()

        except Exception as e:
            logger.error(f"OpenVINO VLM QA failed: {e}")
            raise NotImplementedError(f"QA not supported: {e}")

    def run_caption(
        self,
        handle: Any,
        image: Image.Image,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        detail_level: str = "detailed",
    ) -> str:
        """Generate image caption."""
        if detail_level == "detailed":
            prompt = "Provide a detailed description of this image."
        else:
            prompt = "Provide a brief caption for this image."

        return self.run_qa(handle, image, prompt, conversation_history)

    def run_detect(
        self,
        handle: Any,
        image: Image.Image,
        object_name: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Run object detection."""
        prompt = (
            f"Detect all instances of '{object_name}' in this image.\n\n"
            f"Return ONLY valid JSON in this exact structure:\n"
            f"{{\n"
            f"  \"{object_name}_1\": [x1, y1, x2, y2],\n"
            f"  \"{object_name}_2\": [x1, y1, x2, y2]\n"
            f"}}\n\n"
            f"Rules:\n"
            f"- Coordinates must be normalized (0.0 to 1.0)\n"
            f"- 0.0 is left/top edge, 1.0 is right/bottom edge\n"
            f"- Do not include any text before or after the JSON\n"
            f"- Number each instance sequentially (_1, _2, _3, etc.)"
        )
        return self.run_qa(handle, image, prompt, conversation_history)

    def run_point(
        self,
        handle: Any,
        image: Image.Image,
        object_name: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """Run object pointing."""
        prompt = (
            f"Locate the '{object_name}' in this image.\n\n"
            f"Return ONLY valid JSON in this exact structure:\n"
            f"{{\n"
            f"  \"{object_name}\": [x, y]\n"
            f"}}\n\n"
            f"Rules:\n"
            f"- Coordinates must be normalized (0.0 to 1.0)\n"
            f"- 0.0 is left/top edge, 1.0 is right/bottom edge\n"
            f"- Do not include any text before or after the JSON"
        )
        return self.run_qa(handle, image, prompt, conversation_history)

    def run_text(
        self,
        handle: Any,
        prompt: str,
        conversation_history: Optional[list[tuple[str, str]]] = None,
        custom_parameters: Optional[dict] = None,
    ) -> str:
        """Run text generation with OpenVINO.

        Args:
            handle: Model handle (model, tokenizer)
            prompt: Input prompt
            conversation_history: Optional conversation history
            custom_parameters: Optional generation parameters

        Returns:
            Generated text
        """
        model, tokenizer = handle

        try:
            # Format prompt with history
            if conversation_history:
                from ..utils.history_formatter import format_conversation_history
                formatted_prompt = format_conversation_history(
                    conversation_history=conversation_history,
                    current_prompt=prompt,
                    max_turns=5
                )
            else:
                formatted_prompt = prompt

            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt")

            # Build generation config
            max_tokens = custom_parameters.get("max_tokens", 1024) if custom_parameters else 1024
            temperature = custom_parameters.get("temperature", 0.7) if custom_parameters else 0.7

            # Generate
            logger.info("Generating text with OpenVINO...")
            import torch
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )

            # Decode (skip input tokens)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            logger.info("✓ Text generated")
            return response.strip()

        except Exception as e:
            logger.error(f"OpenVINO text generation failed: {e}")
            raise NotImplementedError(f"Text generation not supported: {e}")

    def run_chat(
        self,
        handle: Any,
        messages: list[dict],
        custom_parameters: Optional[dict] = None,
    ) -> str:
        """Run chat completion."""
        # Convert messages to conversation history
        conversation_history = []
        current_prompt = ""

        for msg in messages:
            if msg["role"] == "user":
                current_prompt = msg["content"]
            elif msg["role"] == "assistant":
                if current_prompt:
                    conversation_history.append((current_prompt, msg["content"]))
                    current_prompt = ""

        return self.run_text(handle, current_prompt, conversation_history, custom_parameters)
