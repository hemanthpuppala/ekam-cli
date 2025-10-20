"""Hardware-based quantization recommendations."""

from typing import Optional

from loguru import logger

from ...models.model import ModelInfo
from ...models.system import SystemSpecs
from ..models import QuantizationModule, QuantizationRecommendation, QuantizationType


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
        method_family: Quantization method family (GGUF, GPTQ, AWQ, BNB, Advanced, Generic)
        use_gpu: Whether GPU is available and will be used

    Returns:
        List of recommendations, sorted by overall score (best first)
    """
    recommendations = []

    if method_family == "Generic":
        recommendations = _get_generic_recommendations(model_info, system_specs, use_gpu)
    elif method_family == "GGUF":
        recommendations = _get_gguf_recommendations(model_info, system_specs)
    elif method_family == "MLX":
        recommendations = _get_mlx_recommendations(model_info, system_specs)
    elif method_family == "OpenVINO":
        recommendations = _get_openvino_recommendations(model_info, system_specs)
    elif method_family == "Advanced":
        # Advanced 4-bit quantization: Combine all advanced methods
        # GPTQ: GPU-optimized, good quality
        recommendations.extend(_get_gptq_recommendations(model_info, system_specs, use_gpu))
        # AWQ: Better quality than GPTQ
        recommendations.extend(_get_awq_recommendations(model_info, system_specs, use_gpu))
        # BnB: Works on CPU/GPU, HuggingFace integration
        recommendations.extend(_get_bnb_recommendations(model_info, system_specs, use_gpu))
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


def _get_generic_recommendations(
    model_info: ModelInfo, system_specs: SystemSpecs, use_gpu: bool
) -> list[QuantizationRecommendation]:
    """Get generic quantization recommendations (FP16, INT8, INT4).

    ALWAYS shows all 3 types, but marks INT8/INT4 as unavailable on non-CUDA systems.

    Note: INT8 and INT4 require CUDA GPU + bitsandbytes to save to disk.
    On CPU/Metal, only FP16 is available.
    """
    recommendations = []
    original_size_gb = model_info.size_gb

    # Check if CUDA is available for INT8/INT4
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    # FP16: Half precision (50% size reduction, minimal quality loss)
    # Works on ALL platforms (CPU, Metal, CUDA)
    fp16_size = original_size_gb * 0.5
    fp16_time = original_size_gb * 1.0  # Fast conversion

    reason_fp16 = "FP16 (Half Precision): Simple dtype conversion. "
    if fp16_size < system_specs.available_ram_gb * 0.7:
        reason_fp16 += f"Fits in RAM ({fp16_size:.1f}GB / {system_specs.available_ram_gb:.1f}GB). "
    reason_fp16 += "Minimal quality loss, works on all platforms."

    recommendations.append(
        QuantizationRecommendation(
            quant_type=QuantizationType.FP16,
            module=QuantizationModule.PYTORCH,
            reason=reason_fp16,
            estimated_size_gb=fp16_size,
            estimated_time_minutes=fp16_time,
            quality_score=10,  # No quality loss
            speed_score=10,  # Very fast
            best_for="Quick size reduction, all devices",
            requires_gpu=False,
            is_available=True,
            unavailable_reason="",
        )
    )

    # INT8: 8-bit integer (75% size reduction, small quality loss)
    # ALWAYS show, but mark as unavailable if no CUDA
    int8_size = original_size_gb * 0.25
    int8_time = original_size_gb * 2.0  # Requires calibration

    if has_cuda:
        reason_int8 = "INT8 Quantization: 8-bit integer weights (requires CUDA + bitsandbytes). "
        if int8_size < system_specs.available_ram_gb * 0.7:
            reason_int8 += f"Fits in RAM ({int8_size:.1f}GB / {system_specs.available_ram_gb:.1f}GB). "
        reason_int8 += "Good balance of size and quality."
        unavailable_reason_int8 = ""
        is_available_int8 = True
    else:
        reason_int8 = "INT8 Quantization: 8-bit integer weights. NOT available on Mac/CPU. "
        reason_int8 += "Requires NVIDIA CUDA GPU + bitsandbytes library."
        unavailable_reason_int8 = "Requires NVIDIA CUDA GPU (not available on your Mac/CPU system). PyTorch's CPU quantization cannot be saved to disk."
        is_available_int8 = False

    recommendations.append(
        QuantizationRecommendation(
            quant_type=QuantizationType.INT8,
            module=QuantizationModule.PYTORCH,
            reason=reason_int8,
            estimated_size_gb=int8_size,
            estimated_time_minutes=int8_time,
            quality_score=9,  # Minimal loss
            speed_score=9,  # Fast inference
            best_for="Production, edge devices (CUDA GPU required)",
            requires_gpu=True,
            is_available=is_available_int8,
            unavailable_reason=unavailable_reason_int8,
        )
    )

    # INT4: 4-bit integer (87.5% size reduction, moderate quality loss)
    # ALWAYS show, but mark as unavailable if no CUDA
    int4_size = original_size_gb * 0.125
    int4_time = original_size_gb * 3.0  # More complex quantization

    if has_cuda:
        reason_int4 = "INT4 Quantization: 4-bit integer weights (requires CUDA + bitsandbytes). "
        if int4_size < system_specs.available_ram_gb * 0.7:
            reason_int4 += f"Fits in RAM ({int4_size:.1f}GB / {system_specs.available_ram_gb:.1f}GB). "
        reason_int4 += "Maximum compression, acceptable quality."
        unavailable_reason_int4 = ""
        is_available_int4 = True
    else:
        reason_int4 = "INT4 Quantization: 4-bit integer weights. NOT available on Mac/CPU. "
        reason_int4 += "Requires NVIDIA CUDA GPU + bitsandbytes library."
        unavailable_reason_int4 = "Requires NVIDIA CUDA GPU (not available on your Mac/CPU system). PyTorch does not support persistent INT4 on CPU."
        is_available_int4 = False

    recommendations.append(
        QuantizationRecommendation(
            quant_type=QuantizationType.INT4,
            module=QuantizationModule.PYTORCH,
            reason=reason_int4,
            estimated_size_gb=int4_size,
            estimated_time_minutes=int4_time,
            quality_score=8,  # Moderate loss
            speed_score=10,  # Very fast
            best_for="Memory-constrained devices (CUDA GPU required)",
            requires_gpu=True,
            is_available=is_available_int4,
            unavailable_reason=unavailable_reason_int4,
        )
    )

    return recommendations


def _get_gguf_recommendations(
    model_info: ModelInfo, system_specs: SystemSpecs
) -> list[QuantizationRecommendation]:
    """Get GGUF quantization recommendations.

    ALWAYS shows all GGUF types, but marks as unavailable for VLMs.
    """
    recommendations = []
    original_size_gb = model_info.size_gb

    # Check if VLM (llama.cpp doesn't support VLMs yet)
    from ...models.endpoints import ModelType
    is_vlm = model_info.model_type == ModelType.VLM

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

    available_ram_gb = system_specs.available_ram_gb

    for quant_type, factor in size_factors.items():
        estimated_size = original_size_gb * factor
        quality, speed = scores[quant_type]

        # Estimate time: ~2-4 minutes per GB depending on CPU
        cpu_cores = system_specs.cpu_cores_physical
        time_per_gb = 3.0 / (cpu_cores / 4)  # Scale with cores
        estimated_time = original_size_gb * time_per_gb

        # Determine if this fits in available RAM
        fits_in_ram = estimated_size < (available_ram_gb * 0.7)  # Leave 30% headroom

        # Determine availability
        if is_vlm:
            is_available = False
            unavailable_reason = (
                "GGUF doesn't support Vision-Language Models (VLMs). "
                "llama.cpp cannot convert VLM architectures. "
                "Use Generic FP16 or BitsAndBytes quantization for VLMs instead."
            )
            reason = f"{quant_type.display_name}: NOT supported for VLMs. llama.cpp limitation."
        else:
            is_available = True
            unavailable_reason = ""
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
                is_available=is_available,
                unavailable_reason=unavailable_reason,
            )
        )

    return recommendations


def _get_gptq_recommendations(
    model_info: ModelInfo, system_specs: SystemSpecs, use_gpu: bool
) -> list[QuantizationRecommendation]:
    """Get GPTQ quantization recommendations.

    ALWAYS shows all GPTQ types, but marks as unavailable on non-CUDA systems or VLMs.
    """
    # Check CUDA availability
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    # Check if VLM
    from ...models.endpoints import ModelType
    is_vlm = model_info.model_type == ModelType.VLM

    recommendations = []
    original_size_gb = model_info.size_gb

    gptq_configs = [
        (QuantizationType.GPTQ_8BIT, 0.5, 9, 8, "GPU inference, high quality"),
        (QuantizationType.GPTQ_4BIT, 0.35, 8, 9, "GPU inference, good quality"),
        (QuantizationType.GPTQ_3BIT, 0.25, 6, 10, "GPU inference, maximum compression"),
    ]

    for quant_type, factor, quality, speed, best_for in gptq_configs:
        estimated_size = original_size_gb * factor
        estimated_time = original_size_gb * 5.0  # GPTQ is slower to quantize

        # Determine availability
        if not has_cuda:
            reason = f"{quant_type.display_name}: NOT available on Mac/CPU. Requires NVIDIA CUDA GPU."
            is_available = False
            unavailable_reason = "Requires NVIDIA CUDA GPU (not available on your Mac/CPU system). GPTQ quantization only works with CUDA."
        elif is_vlm:
            reason = f"{quant_type.display_name}: NOT supported for VLMs. GPTQ is designed for pure language models only."
            is_available = False
            unavailable_reason = "GPTQ doesn't support Vision-Language Models (VLMs). Use BitsAndBytes or Generic quantization for VLMs."
        else:
            reason = f"{quant_type.display_name}: Requires CUDA GPU. "
            if system_specs.gpu and system_specs.gpu.memory_gb:
                gpu_mem = system_specs.gpu.memory_gb
                if estimated_size < gpu_mem * 0.7:
                    reason += f"Fits in GPU memory ({estimated_size:.1f}GB / {gpu_mem:.1f}GB)."
                else:
                    reason += f"⚠️  May exceed GPU memory ({gpu_mem:.1f}GB)."
            else:
                reason += "CUDA GPU detected."
            is_available = True
            unavailable_reason = ""

        recommendations.append(
            QuantizationRecommendation(
                quant_type=quant_type,
                module=QuantizationModule.AUTO_GPTQ,
                reason=reason,
                estimated_size_gb=estimated_size,
                estimated_time_minutes=estimated_time,
                quality_score=quality,
                speed_score=speed,
                best_for=best_for + (" (CUDA GPU required)" if not has_cuda else "") + (" (LLMs only)" if is_vlm else ""),
                requires_gpu=True,
                is_available=is_available,
                unavailable_reason=unavailable_reason,
            )
        )

    return recommendations


def _get_awq_recommendations(
    model_info: ModelInfo, system_specs: SystemSpecs, use_gpu: bool
) -> list[QuantizationRecommendation]:
    """Get AWQ quantization recommendations.

    ALWAYS shows AWQ 4-bit, but marks as unavailable on non-CUDA systems or VLMs.
    """
    # Check CUDA availability
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    # Check if VLM
    from ...models.endpoints import ModelType
    is_vlm = model_info.model_type == ModelType.VLM

    original_size_gb = model_info.size_gb
    estimated_size = original_size_gb * 0.3
    estimated_time = original_size_gb * 6.0  # AWQ is slower but better quality

    # Determine availability
    if not has_cuda:
        is_available = False
        unavailable_reason = (
            "Requires NVIDIA CUDA GPU (not available on your Mac/CPU system). "
            "AWQ quantization only works with CUDA. Use Generic FP16 or GGUF instead."
        )
        reason = f"AWQ 4-bit: NOT available on Mac/CPU. Requires NVIDIA CUDA GPU."
    elif is_vlm:
        is_available = False
        unavailable_reason = (
            "AWQ doesn't support Vision-Language Models (VLMs). "
            "Use BitsAndBytes or Generic quantization for VLMs."
        )
        reason = f"AWQ 4-bit: NOT supported for VLMs. Use BnB or Generic instead."
    else:
        is_available = True
        unavailable_reason = ""
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
            is_available=is_available,
            unavailable_reason=unavailable_reason,
        )
    ]


def _get_bnb_recommendations(
    model_info: ModelInfo, system_specs: SystemSpecs, use_gpu: bool
) -> list[QuantizationRecommendation]:
    """Get BitsAndBytes quantization recommendations.

    ALWAYS shows all BnB types, but marks as unavailable on non-CUDA systems.
    """
    # Check CUDA availability
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    recommendations = []
    original_size_gb = model_info.size_gb

    bnb_configs = [
        (QuantizationType.BNB_8BIT, 0.5, 9, 8, "HuggingFace, good quality"),
        (QuantizationType.BNB_4BIT_NF4, 0.35, 8, 9, "HuggingFace, NormalFloat4 (recommended)"),
        (QuantizationType.BNB_4BIT_FP4, 0.35, 7, 10, "HuggingFace, Float4 (faster)"),
    ]

    for quant_type, factor, quality, speed, best_for in bnb_configs:
        estimated_size = original_size_gb * factor
        estimated_time = original_size_gb * 2.0  # BnB is fast

        if has_cuda:
            reason = f"{quant_type.display_name}: Integrated with HuggingFace Transformers. "
            reason += "Requires CUDA GPU. "
            reason += f"Estimated size: {estimated_size:.1f}GB."
            is_available = True
            unavailable_reason = ""
        else:
            reason = f"{quant_type.display_name}: NOT available on Mac/CPU. "
            reason += "Requires NVIDIA CUDA GPU + bitsandbytes library."
            is_available = False
            unavailable_reason = "Requires NVIDIA CUDA GPU (not available on your Mac/CPU system). BitsAndBytes quantization only works with CUDA."

        recommendations.append(
            QuantizationRecommendation(
                quant_type=quant_type,
                module=QuantizationModule.OPTIMUM,
                reason=reason,
                estimated_size_gb=estimated_size,
                estimated_time_minutes=estimated_time,
                quality_score=quality,
                speed_score=speed,
                best_for=best_for + (" (CUDA GPU required)" if not has_cuda else ""),
                requires_gpu=True,
                is_available=is_available,
                unavailable_reason=unavailable_reason,
            )
        )

    return recommendations


def _get_mlx_recommendations(
    model_info: ModelInfo, system_specs: SystemSpecs
) -> list[QuantizationRecommendation]:
    """Get MLX quantization recommendations (Apple Silicon only).
    
    MLX supports: INT2, INT3, INT4, INT6, INT8, FP16
    NOT supported: INT5 (not available in MLX framework)
    """
    import platform
    recommendations = []
    original_size_gb = model_info.size_gb

    is_macos = platform.system() == "Darwin"
    is_apple_silicon = platform.machine() in ["arm64", "aarch64"] and is_macos
    has_mlx = False
    try:
        import mlx.core  # noqa
        has_mlx = True
    except ImportError:
        pass

    # MLX quantization configurations: (type, factor, quality, speed, best_for)
    mlx_configs = [
        (QuantizationType.INT4, 0.25, 8, 10, "RECOMMENDED - Best balance of quality and size"),
        (QuantizationType.INT6, 0.375, 9, 9, "Good quality, moderate compression"),
        (QuantizationType.INT8, 0.5, 9, 9, "Highest quality integer quantization"),
        (QuantizationType.FP16, 0.5, 10, 10, "Baseline, minimal quality loss"),
        (QuantizationType.INT3, 0.1875, 7, 10, "High compression, quality tradeoff"),
        (QuantizationType.INT2, 0.125, 6, 10, "Maximum compression, significant quality loss"),
    ]

    for quant_type, factor, quality, speed, best_for in mlx_configs:
        estimated_size = original_size_gb * factor
        estimated_time = original_size_gb * 1.5

        if is_apple_silicon and has_mlx:
            reason = f"MLX {quant_type.display_name}: Native Apple Silicon quantization with Metal acceleration. "
            reason += f"Estimated size: {estimated_size:.1f}GB. "
            if quant_type == QuantizationType.INT4:
                reason += "✓ RECOMMENDED for most use cases."
            is_available = True
            unavailable_reason = ""
        else:
            reason = f"MLX {quant_type.display_name}: NOT available on this system. "
            if not is_macos:
                reason += "Requires macOS."
            elif not is_apple_silicon:
                reason += "Requires Apple Silicon (M1/M2/M3/M4)."
            elif not has_mlx:
                reason += "Requires: pip install mlx mlx-lm mlx-vlm"
            is_available = False
            unavailable_reason = "Requires macOS with Apple Silicon + MLX installed"

        recommendations.append(
            QuantizationRecommendation(
                quant_type=quant_type,
                module=QuantizationModule.MLX,
                reason=reason,
                estimated_size_gb=estimated_size,
                estimated_time_minutes=estimated_time,
                quality_score=quality,
                speed_score=speed,
                best_for=best_for + (" (Apple Silicon Mac)" if is_apple_silicon else ""),
                requires_gpu=False,
                is_available=is_available,
                unavailable_reason=unavailable_reason,
            )
        )

    return recommendations


def _get_openvino_recommendations(
    model_info: ModelInfo, system_specs: SystemSpecs
) -> list[QuantizationRecommendation]:
    """Get OpenVINO quantization recommendations (Intel CPU/iGPU).
    
    OpenVINO NNCF supports: INT4, INT8, FP16
    NOT supported: INT2, INT3, INT5, INT6 (not available in NNCF)
    """
    recommendations = []
    original_size_gb = model_info.size_gb

    has_openvino = False
    try:
        import openvino  # noqa
        has_openvino = True
    except ImportError:
        pass

    # OpenVINO quantization configurations: (type, factor, quality, speed, best_for)
    configs = [
        (QuantizationType.INT8, 0.5, 9, 8, "RECOMMENDED - Best quality/size balance"),
        (QuantizationType.INT4, 0.25, 8, 9, "Maximum compression for Intel hardware"),
        (QuantizationType.FP16, 0.5, 10, 8, "Baseline, no compression artifacts"),
    ]

    for quant_type, factor, quality, speed, best_for in configs:
        # Add 10% overhead for OpenVINO IR format
        estimated_size = (original_size_gb * factor) * 1.1
        estimated_time = original_size_gb * 2.0

        if has_openvino:
            reason = f"OpenVINO {quant_type.display_name}: Optimized for Intel CPUs (AVX-512, VNNI, AMX). "
            reason += f"Estimated size: {estimated_size:.1f}GB. "
            reason += "Supports Intel iGPU (XMX) and Arc GPU acceleration."
            if quant_type == QuantizationType.INT8:
                reason += " ✓ RECOMMENDED for production."
            is_available = True
            unavailable_reason = ""
        else:
            reason = f"OpenVINO {quant_type.display_name}: NOT available. "
            reason += "Requires: pip install openvino optimum[openvino]"
            is_available = False
            unavailable_reason = "Requires OpenVINO installation"

        recommendations.append(
            QuantizationRecommendation(
                quant_type=quant_type,
                module=QuantizationModule.OPENVINO,
                reason=reason,
                estimated_size_gb=estimated_size,
                estimated_time_minutes=estimated_time,
                quality_score=quality,
                speed_score=speed,
                best_for=best_for + " (Intel CPUs/iGPUs)",
                requires_gpu=False,
                is_available=is_available,
                unavailable_reason=unavailable_reason,
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
    has_enough_ram = system_specs.total_ram_gb >= 32
    has_enough_cores = system_specs.cpu_cores_physical >= 8

    return has_enough_ram and has_enough_cores
