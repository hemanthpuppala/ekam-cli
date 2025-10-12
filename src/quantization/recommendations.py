"""Hardware-based quantization recommendations."""

from typing import Optional

from ..models.model import ModelInfo
from ..models.system import SystemSpecs
from .models import QuantizationModule, QuantizationRecommendation, QuantizationType


def get_quantization_recommendations(
    model_info: ModelInfo,
    system_specs: SystemSpecs,
    method_family: str = "GGUF",
    use_gpu: bool = False,
) -> list[QuantizationRecommendation]:
    """Generate quantization recommendations based on model and system specs.

    Args:
        model_info: Model to quantize
        system_specs: System specifications
        method_family: Quantization method family (GGUF, GPTQ, AWQ, BNB)
        use_gpu: Whether GPU is available and will be used

    Returns:
        List of recommendations, sorted by overall score (best first)
    """
    recommendations = []

    if method_family == "GGUF":
        recommendations = _get_gguf_recommendations(model_info, system_specs)
    elif method_family == "GPTQ":
        recommendations = _get_gptq_recommendations(model_info, system_specs, use_gpu)
    elif method_family == "AWQ":
        recommendations = _get_awq_recommendations(model_info, system_specs, use_gpu)
    elif method_family == "BitsAndBytes":
        recommendations = _get_bnb_recommendations(model_info, system_specs, use_gpu)

    # Sort by overall score (best first)
    recommendations.sort(key=lambda r: r.overall_score, reverse=True)

    # Mark the best one as recommended
    if recommendations:
        recommendations[0].is_recommended = True

    return recommendations


def _get_gguf_recommendations(
    model_info: ModelInfo, system_specs: SystemSpecs
) -> list[QuantizationRecommendation]:
    """Get GGUF quantization recommendations."""
    recommendations = []
    original_size_gb = model_info.size_gb

    # Estimate sizes for different quantization levels
    size_factors = {
        QuantizationType.GGUF_Q4_K_M: 0.5,  # ~50% of original
        QuantizationType.GGUF_Q4_K_S: 0.45,  # Smaller variant
        QuantizationType.GGUF_Q5_K_M: 0.6,  # ~60% of original
        QuantizationType.GGUF_Q5_K_S: 0.55,
        QuantizationType.GGUF_Q6_K: 0.7,  # ~70% of original
        QuantizationType.GGUF_Q8_0: 0.9,  # ~90% of original
    }

    # Quality and speed scores (higher is better)
    scores = {
        QuantizationType.GGUF_Q4_K_M: (8, 9),  # (quality, speed)
        QuantizationType.GGUF_Q4_K_S: (7, 10),
        QuantizationType.GGUF_Q5_K_M: (9, 8),
        QuantizationType.GGUF_Q5_K_S: (8, 9),
        QuantizationType.GGUF_Q6_K: (9, 7),
        QuantizationType.GGUF_Q8_0: (10, 6),
    }

    # Best use cases
    best_for = {
        QuantizationType.GGUF_Q4_K_M: "Edge devices, Raspberry Pi",
        QuantizationType.GGUF_Q4_K_S: "Low-memory devices",
        QuantizationType.GGUF_Q5_K_M: "Desktop, balanced performance",
        QuantizationType.GGUF_Q5_K_S: "Desktop, memory-constrained",
        QuantizationType.GGUF_Q6_K: "Server, high-quality inference",
        QuantizationType.GGUF_Q8_0: "Archival, maximum quality",
    }

    available_ram_gb = system_specs.memory.available_gb

    for quant_type, factor in size_factors.items():
        estimated_size = original_size_gb * factor
        quality, speed = scores[quant_type]

        # Estimate time: ~2-4 minutes per GB depending on CPU
        cpu_cores = system_specs.cpu.physical_cores
        time_per_gb = 3.0 / (cpu_cores / 4)  # Scale with cores
        estimated_time = original_size_gb * time_per_gb

        # Determine if this fits in available RAM
        fits_in_ram = estimated_size < (available_ram_gb * 0.7)  # Leave 30% headroom

        # Reason for this quantization
        reason = f"{quant_type.display_name}: "
        if not fits_in_ram:
            reason += f"⚠️  May exceed available RAM ({available_ram_gb:.1f}GB). "
        else:
            reason += f"Fits in RAM ({estimated_size:.1f}GB / {available_ram_gb:.1f}GB available). "

        if quality >= 9:
            reason += "Excellent quality."
        elif quality >= 7:
            reason += "Good quality."
        else:
            reason += "Acceptable quality."

        recommendations.append(
            QuantizationRecommendation(
                quant_type=quant_type,
                module=QuantizationModule.LLAMA_CPP,
                reason=reason,
                estimated_size_gb=estimated_size,
                estimated_time_minutes=estimated_time,
                quality_score=quality,
                speed_score=speed,
                best_for=best_for[quant_type],
                requires_gpu=False,
            )
        )

    return recommendations


def _get_gptq_recommendations(
    model_info: ModelInfo, system_specs: SystemSpecs, use_gpu: bool
) -> list[QuantizationRecommendation]:
    """Get GPTQ quantization recommendations."""
    if not use_gpu:
        return []  # GPTQ requires GPU

    recommendations = []
    original_size_gb = model_info.size_gb

    gptq_configs = [
        (QuantizationType.GPTQ_4BIT, 0.35, 8, 9, "GPU inference, good quality"),
        (QuantizationType.GPTQ_3BIT, 0.25, 6, 10, "GPU inference, maximum compression"),
    ]

    for quant_type, factor, quality, speed, best_for in gptq_configs:
        estimated_size = original_size_gb * factor
        estimated_time = original_size_gb * 5.0  # GPTQ is slower to quantize

        reason = f"{quant_type.display_name}: Requires GPU. "
        if system_specs.gpu and system_specs.gpu.memory_gb:
            gpu_mem = system_specs.gpu.memory_gb
            if estimated_size < gpu_mem * 0.7:
                reason += f"Fits in GPU memory ({estimated_size:.1f}GB / {gpu_mem:.1f}GB)."
            else:
                reason += f"⚠️  May exceed GPU memory ({gpu_mem:.1f}GB)."
        else:
            reason += "GPU detected."

        recommendations.append(
            QuantizationRecommendation(
                quant_type=quant_type,
                module=QuantizationModule.AUTO_GPTQ,
                reason=reason,
                estimated_size_gb=estimated_size,
                estimated_time_minutes=estimated_time,
                quality_score=quality,
                speed_score=speed,
                best_for=best_for,
                requires_gpu=True,
            )
        )

    return recommendations


def _get_awq_recommendations(
    model_info: ModelInfo, system_specs: SystemSpecs, use_gpu: bool
) -> list[QuantizationRecommendation]:
    """Get AWQ quantization recommendations."""
    if not use_gpu:
        return []  # AWQ requires GPU

    original_size_gb = model_info.size_gb
    estimated_size = original_size_gb * 0.3
    estimated_time = original_size_gb * 6.0  # AWQ is slower but better quality

    reason = f"AWQ 4-bit: Requires GPU. Better quality than GPTQ. "
    if system_specs.gpu and system_specs.gpu.memory_gb:
        gpu_mem = system_specs.gpu.memory_gb
        if estimated_size < gpu_mem * 0.7:
            reason += f"Fits in GPU memory ({estimated_size:.1f}GB / {gpu_mem:.1f}GB)."
        else:
            reason += f"⚠️  May exceed GPU memory ({gpu_mem:.1f}GB)."

    return [
        QuantizationRecommendation(
            quant_type=QuantizationType.AWQ_4BIT,
            module=QuantizationModule.AUTO_AWQ,
            reason=reason,
            estimated_size_gb=estimated_size,
            estimated_time_minutes=estimated_time,
            quality_score=9,  # Better than GPTQ
            speed_score=9,
            best_for="GPU inference, high quality",
            requires_gpu=True,
        )
    ]


def _get_bnb_recommendations(
    model_info: ModelInfo, system_specs: SystemSpecs, use_gpu: bool
) -> list[QuantizationRecommendation]:
    """Get BitsAndBytes quantization recommendations."""
    recommendations = []
    original_size_gb = model_info.size_gb

    bnb_configs = [
        (QuantizationType.BNB_8BIT, 0.5, 9, 8, "HuggingFace, good quality", False),
        (QuantizationType.BNB_4BIT_NF4, 0.35, 8, 9, "HuggingFace, balanced", True),
    ]

    for quant_type, factor, quality, speed, best_for, requires_gpu in bnb_configs:
        if requires_gpu and not use_gpu:
            continue  # Skip GPU-only quantizations if no GPU

        estimated_size = original_size_gb * factor
        estimated_time = original_size_gb * 2.0  # BnB is fast

        reason = f"{quant_type.display_name}: Integrated with HuggingFace Transformers. "
        if requires_gpu:
            reason += "Requires GPU. "
        reason += f"Estimated size: {estimated_size:.1f}GB."

        recommendations.append(
            QuantizationRecommendation(
                quant_type=quant_type,
                module=QuantizationModule.OPTIMUM,
                reason=reason,
                estimated_size_gb=estimated_size,
                estimated_time_minutes=estimated_time,
                quality_score=quality,
                speed_score=speed,
                best_for=best_for,
                requires_gpu=requires_gpu,
            )
        )

    return recommendations


def check_can_quantize_multiple(system_specs: SystemSpecs) -> bool:
    """Check if system can handle multiple concurrent quantization jobs.

    Args:
        system_specs: System specifications

    Returns:
        True if system can handle multiple jobs
    """
    # Require at least 32GB RAM and 8+ CPU cores for multiple jobs
    has_enough_ram = system_specs.memory.total_gb >= 32
    has_enough_cores = system_specs.cpu.physical_cores >= 8

    return has_enough_ram and has_enough_cores
