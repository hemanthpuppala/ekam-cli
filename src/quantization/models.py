"""Data models for quantization tasks and configurations."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from ..models.model import ModelInfo


class QuantizationType(Enum):
    """Available quantization types."""

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

    # AWQ quantizations (better quality than GPTQ)
    AWQ_4BIT = "awq_4bit"

    # BitsAndBytes quantizations (HuggingFace integrated)
    BNB_8BIT = "bnb_8bit"
    BNB_4BIT_NF4 = "bnb_4bit_nf4"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
            self.GGUF_Q4_K_M: "Q4_K_M - 4-bit Medium",
            self.GGUF_Q4_K_S: "Q4_K_S - 4-bit Small",
            self.GGUF_Q5_K_M: "Q5_K_M - 5-bit Medium",
            self.GGUF_Q5_K_S: "Q5_K_S - 5-bit Small",
            self.GGUF_Q6_K: "Q6_K - 6-bit",
            self.GGUF_Q8_0: "Q8_0 - 8-bit",
            self.GPTQ_4BIT: "GPTQ 4-bit",
            self.GPTQ_3BIT: "GPTQ 3-bit",
            self.AWQ_4BIT: "AWQ 4-bit",
            self.BNB_8BIT: "BitsAndBytes 8-bit",
            self.BNB_4BIT_NF4: "BitsAndBytes 4-bit NF4",
        }
        return names.get(self, self.value)

    @property
    def file_extension(self) -> str:
        """File extension for this quantization type."""
        if self.value.startswith("q"):  # GGUF
            return ".gguf"
        elif "gptq" in self.value:
            return ".safetensors"
        elif "awq" in self.value:
            return ".safetensors"
        else:  # BitsAndBytes
            return ".safetensors"

    @property
    def method_family(self) -> str:
        """Quantization method family (GGUF, GPTQ, AWQ, BNB)."""
        if self.value.startswith("q"):
            return "GGUF"
        elif "gptq" in self.value:
            return "GPTQ"
        elif "awq" in self.value:
            return "AWQ"
        else:
            return "BitsAndBytes"


class QuantizationModule(Enum):
    """Quantization modules/tools."""

    LLAMA_CPP = "llama_cpp"
    AUTO_GPTQ = "auto_gptq"
    AUTO_AWQ = "auto_awq"
    OPTIMUM = "optimum"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
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
            "provider": str(self.model_info.provider),
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
        }


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

    @property
    def overall_score(self) -> float:
        """Combined score for sorting."""
        return (self.quality_score + self.speed_score) / 2
