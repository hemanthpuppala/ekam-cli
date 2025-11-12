"""Data models for quantization tasks and configurations."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from ..models.model import ModelInfo


class QuantizationType(Enum):
    """Available quantization types.
    
    Quantization reduces model size and memory usage by using lower-precision
    numeric representations. Supported across multiple frameworks:
    
    - **Generic (INT2-INT8, FP16)**: Universal format, works with MLX, PyTorch, HF
    - **GGUF**: llama.cpp format for CPU inference
    - **GPTQ/AWQ**: GPU-optimized quantization
    - **BitsAndBytes**: HuggingFace native quantization
    
    Quality vs Size tradeoff (larger = better quality, bigger size):
    FP16 > INT8 > INT6 > INT4 > INT3 > INT2
    """

    # Generic/Standard quantizations (HuggingFace/PyTorch/MLX)
    # These work across multiple frameworks and are framework-agnostic
    FP16 = "fp16"  # Half precision (float16) - 50% size, minimal quality loss
    INT8 = "int8"  # 8-bit integer - 50% size, very good quality
    INT6 = "int6"  # 6-bit integer - 37.5% size, good quality (MLX)
    INT4 = "int4"  # 4-bit integer - 25% size, acceptable quality
    INT3 = "int3"  # 3-bit integer - 18.75% size, noticeable quality loss (MLX)
    INT2 = "int2"  # 2-bit integer - 12.5% size, significant quality loss (MLX)

    # GGUF quantizations (llama.cpp)
    # Full precision formats
    GGUF_F32 = "f32"        # Float32 - full precision (26GB @ 7B)
    GGUF_F16 = "f16"        # Float16 - half precision (14GB @ 7B)
    GGUF_BF16 = "bf16"      # BFloat16 - brain float16 (14GB @ 7B)

    # 8-bit quantization
    GGUF_Q8_0 = "q8_0"      # 8-bit quantization (7.96GB @ 7B, +0.0026 ppl)

    # 6-bit quantization
    GGUF_Q6_K = "q6_k"      # 6-bit K-quant (6.14GB @ 7B, +0.0217 ppl)

    # 5-bit quantization
    GGUF_Q5_K = "q5_k"      # Alias for Q5_K_M
    GGUF_Q5_K_M = "q5_k_m"  # 5-bit K-quant Medium (5.33GB @ 7B, +0.0569 ppl)
    GGUF_Q5_K_S = "q5_k_s"  # 5-bit K-quant Small (5.21GB @ 7B, +0.1049 ppl)
    GGUF_Q5_0 = "q5_0"      # 5-bit original (5.21GB @ 7B, +0.1316 ppl)
    GGUF_Q5_1 = "q5_1"      # 5-bit improved (5.65GB @ 7B, +0.1062 ppl)

    # 4-bit quantization
    GGUF_IQ4_XS = "iq4_xs"  # 4.25 bpw importance quantization
    GGUF_IQ4_NL = "iq4_nl"  # 4.50 bpw non-linear quantization
    GGUF_Q4_K = "q4_k"      # Alias for Q4_K_M
    GGUF_Q4_K_M = "q4_k_m"  # 4-bit K-quant Medium (4.58GB @ 7B, +0.1754 ppl)
    GGUF_Q4_K_S = "q4_k_s"  # 4-bit K-quant Small (4.37GB @ 7B, +0.2689 ppl)
    GGUF_Q4_0 = "q4_0"      # 4-bit original (4.34GB @ 7B, +0.4685 ppl)
    GGUF_Q4_1 = "q4_1"      # 4-bit improved (4.78GB @ 7B, +0.4511 ppl)

    # 3-bit quantization
    GGUF_IQ3_M = "iq3_m"    # 3.66 bpw importance quantization mix
    GGUF_IQ3_S = "iq3_s"    # 3.44 bpw importance quantization
    GGUF_IQ3_XS = "iq3_xs"  # 3.3 bpw importance quantization
    GGUF_IQ3_XXS = "iq3_xxs" # 3.06 bpw importance quantization
    GGUF_Q3_K = "q3_k"      # Alias for Q3_K_M
    GGUF_Q3_K_L = "q3_k_l"  # 3-bit K-quant Large (4.03GB @ 7B, +0.5562 ppl)
    GGUF_Q3_K_M = "q3_k_m"  # 3-bit K-quant Medium (3.74GB @ 7B, +0.6569 ppl)
    GGUF_Q3_K_S = "q3_k_s"  # 3-bit K-quant Small (3.41GB @ 7B, +1.6321 ppl)

    # 2-bit quantization
    GGUF_IQ2_M = "iq2_m"    # 2.7 bpw importance quantization
    GGUF_IQ2_S = "iq2_s"    # 2.5 bpw importance quantization
    GGUF_IQ2_XS = "iq2_xs"  # 2.31 bpw importance quantization
    GGUF_IQ2_XXS = "iq2_xxs" # 2.06 bpw importance quantization
    GGUF_Q2_K = "q2_k"      # 2-bit K-quant (2.96GB @ 7B, +3.5199 ppl)
    GGUF_Q2_K_S = "q2_k_s"  # 2-bit K-quant Small (2.96GB @ 7B, +3.1836 ppl)

    # 1-bit quantization
    GGUF_IQ1_M = "iq1_m"    # 1.75 bpw importance quantization
    GGUF_IQ1_S = "iq1_s"    # 1.56 bpw importance quantization

    # Ternary quantization (experimental)
    GGUF_TQ1_0 = "tq1_0"    # 1.69 bpw ternary quantization
    GGUF_TQ2_0 = "tq2_0"    # 2.06 bpw ternary quantization

    # GPTQ quantizations (GPU-optimized)
    GPTQ_4BIT = "gptq_4bit"
    GPTQ_3BIT = "gptq_3bit"
    GPTQ_8BIT = "gptq_8bit"

    # AWQ quantizations (better quality than GPTQ)
    AWQ_4BIT = "awq_4bit"

    # BitsAndBytes quantizations (HuggingFace integrated)
    BNB_8BIT = "bnb_8bit"
    BNB_4BIT_NF4 = "bnb_4bit_nf4"
    BNB_4BIT_FP4 = "bnb_4bit_fp4"

    # Dequantization options (convert quantized models to full precision)
    DEQUANT_FP16_GGUF = "dequant_fp16_gguf"  # Any quant → FP16 GGUF (for llama.cpp/Ollama)
    DEQUANT_FP16_HF = "dequant_fp16_hf"      # Any quant → FP16 HuggingFace safetensors
    DEQUANT_FP32_GGUF = "dequant_fp32_gguf"  # Any quant → FP32 GGUF
    DEQUANT_FP32_HF = "dequant_fp32_hf"      # Any quant → FP32 HuggingFace safetensors

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
            # Generic
            self.FP16: "FP16 - Half Precision",
            self.INT8: "INT8 - 8-bit Integer",
            self.INT6: "INT6 - 6-bit Integer",
            self.INT4: "INT4 - 4-bit Integer",
            self.INT3: "INT3 - 3-bit Integer",
            self.INT2: "INT2 - 2-bit Integer",
            # GGUF - Full precision
            self.GGUF_F32: "F32 - Full Precision (26GB @ 7B)",
            self.GGUF_F16: "F16 - Half Precision (14GB @ 7B)",
            self.GGUF_BF16: "BF16 - Brain Float16 (14GB @ 7B)",

            # GGUF - 8-bit
            self.GGUF_Q8_0: "Q8_0 - 8-bit (7.96GB @ 7B)",

            # GGUF - 6-bit
            self.GGUF_Q6_K: "Q6_K - 6-bit (6.14GB @ 7B)",

            # GGUF - 5-bit
            self.GGUF_Q5_K: "Q5_K - 5-bit Medium (alias)",
            self.GGUF_Q5_K_M: "Q5_K_M - 5-bit Medium (5.33GB @ 7B)",
            self.GGUF_Q5_K_S: "Q5_K_S - 5-bit Small (5.21GB @ 7B)",
            self.GGUF_Q5_0: "Q5_0 - 5-bit Original (5.21GB @ 7B)",
            self.GGUF_Q5_1: "Q5_1 - 5-bit Improved (5.65GB @ 7B)",

            # GGUF - 4-bit
            self.GGUF_IQ4_XS: "IQ4_XS - 4-bit Importance (4.25 bpw)",
            self.GGUF_IQ4_NL: "IQ4_NL - 4-bit Non-Linear (4.50 bpw)",
            self.GGUF_Q4_K: "Q4_K - 4-bit Medium (alias)",
            self.GGUF_Q4_K_M: "Q4_K_M - 4-bit Medium (4.58GB @ 7B)",
            self.GGUF_Q4_K_S: "Q4_K_S - 4-bit Small (4.37GB @ 7B)",
            self.GGUF_Q4_0: "Q4_0 - 4-bit Original (4.34GB @ 7B)",
            self.GGUF_Q4_1: "Q4_1 - 4-bit Improved (4.78GB @ 7B)",

            # GGUF - 3-bit
            self.GGUF_IQ3_M: "IQ3_M - 3-bit Importance Mix (3.66 bpw)",
            self.GGUF_IQ3_S: "IQ3_S - 3-bit Importance (3.44 bpw)",
            self.GGUF_IQ3_XS: "IQ3_XS - 3-bit Importance XS (3.3 bpw)",
            self.GGUF_IQ3_XXS: "IQ3_XXS - 3-bit Importance XXS (3.06 bpw)",
            self.GGUF_Q3_K: "Q3_K - 3-bit Medium (alias)",
            self.GGUF_Q3_K_L: "Q3_K_L - 3-bit Large (4.03GB @ 7B)",
            self.GGUF_Q3_K_M: "Q3_K_M - 3-bit Medium (3.74GB @ 7B)",
            self.GGUF_Q3_K_S: "Q3_K_S - 3-bit Small (3.41GB @ 7B)",

            # GGUF - 2-bit
            self.GGUF_IQ2_M: "IQ2_M - 2-bit Importance (2.7 bpw)",
            self.GGUF_IQ2_S: "IQ2_S - 2-bit Importance (2.5 bpw)",
            self.GGUF_IQ2_XS: "IQ2_XS - 2-bit Importance XS (2.31 bpw)",
            self.GGUF_IQ2_XXS: "IQ2_XXS - 2-bit Importance XXS (2.06 bpw)",
            self.GGUF_Q2_K: "Q2_K - 2-bit (2.96GB @ 7B)",
            self.GGUF_Q2_K_S: "Q2_K_S - 2-bit Small (2.96GB @ 7B)",

            # GGUF - 1-bit
            self.GGUF_IQ1_M: "IQ1_M - 1-bit Importance (1.75 bpw)",
            self.GGUF_IQ1_S: "IQ1_S - 1-bit Importance (1.56 bpw)",

            # GGUF - Ternary (experimental)
            self.GGUF_TQ1_0: "TQ1_0 - Ternary 1-bit (1.69 bpw)",
            self.GGUF_TQ2_0: "TQ2_0 - Ternary 2-bit (2.06 bpw)",
            # GPTQ
            self.GPTQ_8BIT: "GPTQ 8-bit",
            self.GPTQ_4BIT: "GPTQ 4-bit",
            self.GPTQ_3BIT: "GPTQ 3-bit",
            # AWQ
            self.AWQ_4BIT: "AWQ 4-bit",
            # BitsAndBytes
            self.BNB_8BIT: "BitsAndBytes 8-bit",
            self.BNB_4BIT_NF4: "BitsAndBytes 4-bit NF4",
            self.BNB_4BIT_FP4: "BitsAndBytes 4-bit FP4",
            # Dequantization
            self.DEQUANT_FP16_GGUF: "Dequant → FP16 GGUF",
            self.DEQUANT_FP16_HF: "Dequant → FP16 HuggingFace",
            self.DEQUANT_FP32_GGUF: "Dequant → FP32 GGUF",
            self.DEQUANT_FP32_HF: "Dequant → FP32 HuggingFace",
        }
        return names.get(self, self.value)

    @property
    def file_extension(self) -> str:
        """File extension for this quantization type."""
        # GGUF formats (quantized + full precision)
        if self.value.startswith("q") or self.value.startswith("iq") or self.value.startswith("tq"):
            return ".gguf"
        elif self.value in ["f32", "f16", "bf16"]:  # GGUF full precision
            return ".gguf"
        elif self.value in ["fp16", "int8", "int6", "int4", "int3", "int2"]:  # Generic PyTorch/HF
            return ".safetensors"
        elif "gptq" in self.value:
            return ".safetensors"
        elif "awq" in self.value:
            return ".safetensors"
        elif "dequant" in self.value:
            # Dequantization: GGUF or HF based on type
            if "gguf" in self.value:
                return ".gguf"
            else:
                return ".safetensors"
        else:  # BitsAndBytes
            return ".safetensors"

    @property
    def method_family(self) -> str:
        """Quantization method family (Generic, GGUF, GPTQ, AWQ, BNB, Dequantization).

        Returns:
            Method family string for categorization and UI display
        """
        # Generic integer quantization (2/3/4/6/8-bit) - works across multiple frameworks
        if self.value in ["fp16", "int8", "int6", "int4", "int3", "int2"]:
            return "Generic"
        # GGUF quantization (llama.cpp format) - includes quantized and full precision
        elif self.value.startswith("q") or self.value.startswith("iq") or self.value.startswith("tq"):
            return "GGUF"
        elif self.value in ["f32", "f16", "bf16"]:  # GGUF full precision
            return "GGUF"
        # GPTQ quantization (GPU-optimized)
        elif "gptq" in self.value:
            return "GPTQ"
        # AWQ quantization (Activation-aware Weight Quantization)
        elif "awq" in self.value:
            return "AWQ"
        # Dequantization (convert quantized models to full precision)
        elif "dequant" in self.value:
            return "Dequantization"
        # BitsAndBytes quantization (HuggingFace native)
        else:
            return "BitsAndBytes"


class QuantizationModule(Enum):
    """Quantization modules/tools."""

    PYTORCH = "pytorch"  # Generic PyTorch/HuggingFace quantization
    LLAMA_CPP = "llama_cpp"
    AUTO_GPTQ = "auto_gptq"
    AUTO_AWQ = "auto_awq"
    OPTIMUM = "optimum"
    MLX = "mlx"  # Apple Silicon MLX framework
    OPENVINO = "openvino"  # Intel OpenVINO optimization

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
            self.PYTORCH: "PyTorch/Transformers",
            self.LLAMA_CPP: "llama.cpp",
            self.AUTO_GPTQ: "AutoGPTQ",
            self.AUTO_AWQ: "AutoAWQ",
            self.OPTIMUM: "Optimum (HuggingFace)",
            self.MLX: "MLX (Apple Silicon)",
            self.OPENVINO: "OpenVINO (Intel)",
        }
        return names.get(self, self.value)

    @property
    def pip_package(self) -> str:
        """PyPI package name."""
        packages = {
            self.PYTORCH: "torch",
            self.LLAMA_CPP: "llama-cpp-python",
            self.AUTO_GPTQ: "auto-gptq",
            self.AUTO_AWQ: "autoawq",
            self.OPTIMUM: "optimum",
            self.MLX: "mlx",
            self.OPENVINO: "openvino",
        }
        return packages.get(self, self.value)


class TaskStatus(Enum):
    """Quantization task status."""

    PENDING = "pending"
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def emoji(self) -> str:
        """Status emoji."""
        emojis = {
            self.PENDING: "⏳",
            self.PREPARING: "🔧",
            self.RUNNING: "⚙️",
            self.COMPLETED: "✅",
            self.FAILED: "❌",
            self.CANCELLED: "🚫",
        }
        return emojis.get(self, "")


@dataclass
class QuantizationTask:
    """Represents a quantization task."""

    task_id: str
    model_info: ModelInfo
    quant_type: QuantizationType
    module: QuantizationModule
    output_path: Path
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0-100
    eta_seconds: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    use_gpu: bool = False
    background: bool = False
    intermediate_file: Optional[Path] = None  # For GGUF: FP16 intermediate file
    vlm_components: Optional[str] = None  # For VLMs: "vision", "language", "both", or None
    vision_encoder_type: Optional[QuantizationType] = None  # For VLM component-level: vision encoder quant type
    language_decoder_type: Optional[QuantizationType] = None  # For VLM component-level: language decoder quant type
    quantizer_directive: Optional[str] = None  # From orchestrator: "skip", "quantize", "quantize_int8", etc.
    # UX: record if a component fell back to a safer precision
    attempted_vision_quant_type: Optional[QuantizationType] = None
    warning_message: Optional[str] = None

    @property
    def elapsed_seconds(self) -> Optional[float]:
        """Calculate elapsed time if task is running."""
        if self.started_at:
            end = self.completed_at or datetime.now()
            return (end - self.started_at).total_seconds()
        return None

    @property
    def is_active(self) -> bool:
        """Check if task is currently active."""
        return self.status in [TaskStatus.PENDING, TaskStatus.PREPARING, TaskStatus.RUNNING]

    @property
    def is_finished(self) -> bool:
        """Check if task is finished (completed, failed, or cancelled)."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "model_id": self.model_info.model_id,
            "model_name": self.model_info.name,
            "model_type": str(self.model_info.model_type) if hasattr(self.model_info, 'model_type') else "unknown",
            "model_size_gb": self.model_info.size_gb if hasattr(self.model_info, 'size_gb') else 0.0,
            "provider": str(self.model_info.provider),
            # Save ModelInfo required fields for Pydantic validation
            "capabilities": [str(cap) for cap in self.model_info.capabilities] if hasattr(self.model_info, 'capabilities') else [],
            "compatibility": str(self.model_info.compatibility) if hasattr(self.model_info, 'compatibility') else "perfect_fit",
            "compatibility_message": self.model_info.compatibility_message if hasattr(self.model_info, 'compatibility_message') else "",
            "quant_type": self.quant_type.value,
            "module": self.module.value,
            "output_path": str(self.output_path),
            "status": self.status.value,
            "progress": self.progress,
            "eta_seconds": self.eta_seconds,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "use_gpu": self.use_gpu,
            "background": self.background,
            "intermediate_file": str(self.intermediate_file) if self.intermediate_file else None,
            "vlm_components": self.vlm_components,
            "vision_encoder_type": self.vision_encoder_type.value if self.vision_encoder_type else None,
            "language_decoder_type": self.language_decoder_type.value if self.language_decoder_type else None,
            "quantizer_directive": self.quantizer_directive,
            "attempted_vision_quant_type": self.attempted_vision_quant_type.value if self.attempted_vision_quant_type else None,
            "warning_message": self.warning_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuantizationTask":
        """Reconstruct QuantizationTask from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            QuantizationTask instance
        """
        from ..models.model import ModelInfo
        from ..models.endpoints import ModelType, ProviderType, EndpointType, CompatibilityStatus

        # Parse model_type safely (could be string or enum value)
        model_type_str = data.get("model_type", "unknown")
        try:
            # Try to extract just the value if it's in format "ModelType.VLM"
            if "." in str(model_type_str):
                model_type_str = str(model_type_str).split(".")[-1].lower()

            # Attempt to create enum
            model_type = ModelType(model_type_str.lower())
        except (ValueError, AttributeError):
            # Fallback: keep as string or use "unknown"
            model_type = model_type_str if model_type_str else "unknown"

        # Parse provider safely
        provider_str = data.get("provider", "unknown")
        try:
            if "." in str(provider_str):
                provider_str = str(provider_str).split(".")[-1].lower()
            provider = ProviderType(provider_str.lower())
        except (ValueError, AttributeError):
            provider = provider_str if provider_str else "unknown"

        # Parse capabilities (required for ModelInfo)
        capabilities_data = data.get("capabilities", [])
        capabilities = []
        for cap_str in capabilities_data:
            try:
                # Remove enum prefix if present
                if "." in str(cap_str):
                    cap_str = str(cap_str).split(".")[-1].lower()
                capabilities.append(EndpointType(cap_str.lower()))
            except (ValueError, AttributeError):
                # Skip invalid capabilities
                pass

        # If no capabilities saved, default to text generation
        if not capabilities:
            capabilities = [EndpointType.TEXT]

        # Parse compatibility (required for ModelInfo)
        compatibility_str = data.get("compatibility", "perfect_fit")
        try:
            if "." in str(compatibility_str):
                compatibility_str = str(compatibility_str).split(".")[-1].lower()
            compatibility = CompatibilityStatus(compatibility_str.lower())
        except (ValueError, AttributeError):
            compatibility = CompatibilityStatus.PERFECT_FIT

        # Get compatibility message
        compatibility_message = data.get("compatibility_message", "Quantized model")

        # Reconstruct ModelInfo with all required fields
        model_info = ModelInfo(
            model_id=data["model_id"],
            name=data["model_name"],
            model_type=model_type,
            provider=provider,
            size_gb=data.get("model_size_gb", 0.0),
            capabilities=capabilities,
            compatibility=compatibility,
            compatibility_message=compatibility_message,
        )

        # Create task
        task = cls(
            task_id=data["task_id"],
            model_info=model_info,
            quant_type=QuantizationType(data["quant_type"]),
            module=QuantizationModule(data["module"]),
            output_path=Path(data["output_path"]),
            status=TaskStatus(data["status"]),
            progress=data["progress"],
            eta_seconds=data.get("eta_seconds"),
            error=data.get("error"),
            use_gpu=data.get("use_gpu", False),
            background=data.get("background", False),
            intermediate_file=Path(data["intermediate_file"]) if data.get("intermediate_file") else None,
            vlm_components=data.get("vlm_components"),
            vision_encoder_type=QuantizationType(data["vision_encoder_type"]) if data.get("vision_encoder_type") else None,
            language_decoder_type=QuantizationType(data["language_decoder_type"]) if data.get("language_decoder_type") else None,
            quantizer_directive=data.get("quantizer_directive"),
            attempted_vision_quant_type=QuantizationType(data["attempted_vision_quant_type"]) if data.get("attempted_vision_quant_type") else None,
            warning_message=data.get("warning_message"),
        )

        # Restore timestamps
        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])

        return task


@dataclass
class QuantizationRecommendation:
    """Recommendation for a quantization type."""

    quant_type: QuantizationType
    module: QuantizationModule
    reason: str
    estimated_size_gb: float
    estimated_time_minutes: float
    quality_score: int  # 1-10
    speed_score: int  # 1-10
    best_for: str  # "Edge", "Desktop", "Server", etc.
    is_recommended: bool = False
    requires_gpu: bool = False
    is_available: bool = True  # New: whether this option can be selected on current system
    unavailable_reason: str = ""  # New: why this option is not available

    @property
    def overall_score(self) -> float:
        """Combined score for sorting."""
        return (self.quality_score + self.speed_score) / 2
