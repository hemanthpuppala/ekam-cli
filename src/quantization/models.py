"""Data models for quantization tasks and configurations."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from ..models.model import ModelInfo


class QuantizationType(Enum):
    """Available quantization types."""

    # Generic/Standard quantizations (HuggingFace/PyTorch)
    FP16 = "fp16"  # Half precision (float16)
    INT8 = "int8"  # 8-bit integer quantization
    INT4 = "int4"  # 4-bit integer quantization

    # GGUF quantizations (llama.cpp)
    GGUF_Q4_K_M = "q4_k_m"
    GGUF_Q4_K_S = "q4_k_s"
    GGUF_Q5_K_M = "q5_k_m"
    GGUF_Q5_K_S = "q5_k_s"
    GGUF_Q6_K = "q6_k"
    GGUF_Q8_0 = "q8_0"

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

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
            # Generic
            self.FP16: "FP16 - Half Precision",
            self.INT8: "INT8 - 8-bit Integer",
            self.INT4: "INT4 - 4-bit Integer",
            # GGUF
            self.GGUF_Q4_K_M: "Q4_K_M - 4-bit Medium",
            self.GGUF_Q4_K_S: "Q4_K_S - 4-bit Small",
            self.GGUF_Q5_K_M: "Q5_K_M - 5-bit Medium",
            self.GGUF_Q5_K_S: "Q5_K_S - 5-bit Small",
            self.GGUF_Q6_K: "Q6_K - 6-bit",
            self.GGUF_Q8_0: "Q8_0 - 8-bit",
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
        }
        return names.get(self, self.value)

    @property
    def file_extension(self) -> str:
        """File extension for this quantization type."""
        if self.value.startswith("q"):  # GGUF
            return ".gguf"
        elif self.value in ["fp16", "int8", "int4"]:  # Generic PyTorch/HF
            return ".safetensors"
        elif "gptq" in self.value:
            return ".safetensors"
        elif "awq" in self.value:
            return ".safetensors"
        else:  # BitsAndBytes
            return ".safetensors"

    @property
    def method_family(self) -> str:
        """Quantization method family (Generic, GGUF, GPTQ, AWQ, BNB)."""
        if self.value in ["fp16", "int8", "int4"]:
            return "Generic"
        elif self.value.startswith("q"):
            return "GGUF"
        elif "gptq" in self.value:
            return "GPTQ"
        elif "awq" in self.value:
            return "AWQ"
        else:
            return "BitsAndBytes"


class QuantizationModule(Enum):
    """Quantization modules/tools."""

    PYTORCH = "pytorch"  # Generic PyTorch/HuggingFace quantization
    LLAMA_CPP = "llama_cpp"
    AUTO_GPTQ = "auto_gptq"
    AUTO_AWQ = "auto_awq"
    OPTIMUM = "optimum"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
            self.PYTORCH: "PyTorch/Transformers",
            self.LLAMA_CPP: "llama.cpp",
            self.AUTO_GPTQ: "AutoGPTQ",
            self.AUTO_AWQ: "AutoAWQ",
            self.OPTIMUM: "Optimum (HuggingFace)",
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
