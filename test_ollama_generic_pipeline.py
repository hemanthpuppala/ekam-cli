#!/usr/bin/env python3
"""
End-to-end test for Ollama→Generic quantization pipeline.

This test verifies the complete workflow:
1. Detect Ollama model (GGUF format)
2. Determine conversion path (GGUF_TO_GENERIC)
3. Execute orchestrator pipeline:
   - Step 1: Dequantize GGUF to FP16
   - Step 2: Convert FP16 GGUF to HuggingFace format
4. Apply Generic quantization (INT8 or FP16)
5. Verify output

Usage:
    python test_ollama_generic_pipeline.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
import torch

# Configure logger for testing
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")


def get_test_ollama_model() -> str:
    """Get the GGUF path for test Ollama model (gemma3:270m).

    Returns:
        Path to GGUF file, or None if not found
    """
    # Known gemma3:270m location
    gguf_path = Path("/Users/hemanth/.ollama/models/blobs/sha256-735af2139dc652bf01112746474883d79a52fa1c19038265d363e3d42556f7a2")

    if gguf_path.exists():
        logger.info(f"Found test Ollama model: {gguf_path}")
        logger.info(f"Size: {gguf_path.stat().st_size / (1024**3):.2f} GB")
        return str(gguf_path)

    logger.warning(f"Test Ollama model not found at: {gguf_path}")
    logger.info("Make sure you have gemma3:270m installed: ollama pull gemma3:270m")
    return None


def test_conversion_path_determination():
    """Test that orchestrator correctly determines GGUF_TO_GENERIC conversion path."""
    from src.quantization.orchestrator import QuantizationOrchestrator, ConversionPath
    from src.quantization.models import QuantizationType
    from src.models.model import ModelInfo
    from src.models.endpoints import ProviderType, ModelType, EndpointType, CompatibilityStatus

    print("\n" + "="*80)
    print("TEST 1: Conversion Path Determination")
    print("="*80)

    orchestrator = QuantizationOrchestrator()

    # Create mock Ollama model info
    gguf_path = get_test_ollama_model()
    if not gguf_path:
        logger.warning("Skipping test - no Ollama model available")
        return False

    model_info = ModelInfo(
        name="gemma3:270m",
        model_id="gemma3",
        provider=ProviderType.OLLAMA,
        size_gb=0.27,
        model_type=ModelType.LLM,
        capabilities=[EndpointType.TEXT],
        compatibility=CompatibilityStatus.PERFECT_FIT,
        compatibility_message="Test model for Ollama→Generic conversion",
        quantization="Q8_0",
        source_path=Path(gguf_path),
        architecture="gemma"
    )

    # Test: GGUF → Generic should use GGUF_TO_GENERIC path
    path = orchestrator.determine_conversion_path(model_info, QuantizationType.INT8)

    logger.info(f"Model provider: {model_info.provider}")
    logger.info(f"Target quantization: {QuantizationType.INT8.display_name}")
    logger.info(f"Determined conversion path: {path.value}")

    if path == ConversionPath.GGUF_TO_GENERIC:
        logger.info("✓ Correct path determined: GGUF_TO_GENERIC")
        return True
    else:
        logger.error(f"✗ Wrong path: expected GGUF_TO_GENERIC, got {path.value}")
        return False


def test_conversion_pipeline():
    """Test the conversion pipeline execution (GGUF → FP16 → HF)."""
    from src.quantization.orchestrator import QuantizationOrchestrator, ConversionPath
    from src.quantization.models import QuantizationTask, QuantizationType, TaskStatus, QuantizationModule
    from src.models.model import ModelInfo
    from src.models.endpoints import ProviderType, ModelType, EndpointType, CompatibilityStatus

    print("\n" + "="*80)
    print("TEST 2: Conversion Pipeline Execution (GGUF → FP16 → HF)")
    print("="*80)

    gguf_path = get_test_ollama_model()
    if not gguf_path:
        logger.warning("Skipping test - no Ollama model available")
        return False

    # Set up output directory
    output_dir = Path("results/test_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create orchestrator with test output directory
    orchestrator = QuantizationOrchestrator(intermediate_dir=output_dir / "intermediate")

    # Create model info
    model_info = ModelInfo(
        name="gemma3:270m",
        model_id="gemma3",
        provider=ProviderType.OLLAMA,
        size_gb=0.27,
        model_type=ModelType.LLM,
        capabilities=[EndpointType.TEXT],
        compatibility=CompatibilityStatus.PERFECT_FIT,
        compatibility_message="Test model for Ollama→Generic conversion",
        quantization="Q8_0",
        source_path=Path(gguf_path),
        architecture="gemma"
    )

    # Create quantization task
    task = QuantizationTask(
        task_id="test_ollama_to_generic",
        model_info=model_info,
        quant_type=QuantizationType.INT8,
        module=QuantizationModule.PYTORCH,
        output_path=output_dir / "gemma3_int8",
        status=TaskStatus.RUNNING
    )

    # Track progress
    progress_values = []
    def progress_callback(progress: float, eta: float = None):
        progress_values.append(progress)
        logger.info(f"Progress: {progress:.1f}%" + (f" (ETA: {eta:.0f}s)" if eta else ""))

    # Execute conversion pipeline
    logger.info(f"Source path: {gguf_path}")
    logger.info(f"Intermediate directory: {orchestrator.intermediate_dir}")

    result = orchestrator.execute_conversion_pipeline(
        task,
        ConversionPath.GGUF_TO_GENERIC,
        progress_callback
    )

    if result is None:
        logger.error("✗ Conversion pipeline failed")
        return False

    logger.info(f"✓ Conversion pipeline completed")
    logger.info(f"Output path: {result}")
    logger.info(f"Output exists: {result.exists()}")

    # Verify output structure
    if result.is_dir():
        config_file = result / "config.json"
        safetensors_file = list(result.glob("*.safetensors"))

        logger.info(f"  - config.json exists: {config_file.exists()}")
        logger.info(f"  - safetensors files found: {len(safetensors_file)}")

        if config_file.exists() and len(safetensors_file) > 0:
            logger.info("✓ HuggingFace format verified")
            return True

    logger.error("✗ Output doesn't appear to be valid HuggingFace format")
    return False


def test_generic_quantization():
    """Test Generic quantization on converted model."""
    from src.quantization.techniques.generic import GenericQuantizer
    from src.quantization.models import QuantizationTask, QuantizationType, TaskStatus, QuantizationModule
    from src.models.model import ModelInfo
    from src.models.endpoints import ProviderType, ModelType, EndpointType, CompatibilityStatus
    from pathlib import Path

    print("\n" + "="*80)
    print("TEST 3: Generic Quantization on Converted Model")
    print("="*80)

    # Check if converted model exists from previous test
    converted_model_dir = Path("results/test_pipeline/intermediate") / "gemma3_int8_hf"

    if not converted_model_dir.exists():
        logger.info("Skipping Generic quantization test - converted model not found")
        logger.info(f"Expected: {converted_model_dir}")
        return False

    # Create quantizer
    quantizer = GenericQuantizer()

    # Check availability
    available, message = quantizer.check_availability()
    logger.info(f"Generic quantizer available: {available}")
    if message:
        logger.info(f"  {message}")

    if not available:
        logger.warning("Generic quantizer not available, skipping test")
        return False

    # Create model info for converted model
    model_info = ModelInfo(
        name="gemma3:270m-int8",
        model_id="gemma3",
        provider=ProviderType.HUGGINGFACE,
        size_gb=1.0,  # Estimate
        model_type=ModelType.LLM,
        capabilities=[EndpointType.TEXT],
        compatibility=CompatibilityStatus.PERFECT_FIT,
        compatibility_message="Test model for Generic quantization",
        quantization="INT8",
        source_path=converted_model_dir,
        architecture="gemma"
    )

    # Create quantization task
    output_dir = Path("results/test_pipeline")
    task = QuantizationTask(
        task_id="test_generic_quantization",
        model_info=model_info,
        quant_type=QuantizationType.INT8,
        module=QuantizationModule.PYTORCH,
        output_path=output_dir / "gemma3_int8_quantized",
    )

    # Track progress
    def progress_callback(progress: float, eta: float = None):
        logger.info(f"Quantization progress: {progress:.1f}%" + (f" (ETA: {eta:.0f}s)" if eta else ""))

    # Execute quantization
    logger.info(f"Quantizing model from: {converted_model_dir}")
    success = quantizer.quantize(task, progress_callback)

    if success:
        logger.info("✓ Generic quantization completed successfully")
        logger.info(f"Output: {task.output_path}")

        if task.output_path.exists():
            size_gb = sum(f.stat().st_size for f in task.output_path.rglob("*") if f.is_file()) / (1024**3)
            logger.info(f"Output size: {size_gb:.2f} GB")
            return True
    else:
        logger.error(f"✗ Generic quantization failed: {task.error}")
        return False


def test_intermediate_files_preserved():
    """Verify that intermediate files are preserved for user access."""
    from pathlib import Path

    print("\n" + "="*80)
    print("TEST 4: Intermediate Files Preservation")
    print("="*80)

    intermediate_dir = Path("results/test_pipeline/intermediate")

    if not intermediate_dir.exists():
        logger.warning("Intermediate directory not found")
        return False

    logger.info(f"Checking intermediate directory: {intermediate_dir}")

    # Look for FP16 GGUF file
    fp16_files = list(intermediate_dir.glob("*_fp16.gguf"))

    logger.info(f"FP16 GGUF files found: {len(fp16_files)}")
    for fp16_file in fp16_files:
        size_gb = fp16_file.stat().st_size / (1024**3)
        logger.info(f"  - {fp16_file.name} ({size_gb:.2f} GB)")

    # Look for HF directories
    hf_dirs = [d for d in intermediate_dir.iterdir() if d.is_dir() and "_hf" in d.name]

    logger.info(f"HuggingFace directories found: {len(hf_dirs)}")
    for hf_dir in hf_dirs:
        config_file = hf_dir / "config.json"
        safetensors_files = list(hf_dir.glob("*.safetensors"))
        logger.info(f"  - {hf_dir.name}")
        logger.info(f"    - config.json: {config_file.exists()}")
        logger.info(f"    - safetensors files: {len(safetensors_files)}")

    if len(fp16_files) > 0 or len(hf_dirs) > 0:
        logger.info("✓ Intermediate files are preserved")
        return True
    else:
        logger.warning("✗ No intermediate files found")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("End-to-End Ollama→Generic Quantization Pipeline Test")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        "Conversion path determination": test_conversion_path_determination(),
        "Conversion pipeline execution": test_conversion_pipeline(),
        "Intermediate files preservation": test_intermediate_files_preserved(),
        "Generic quantization": test_generic_quantization(),
    }

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {test_name}")

    print("-" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("="*80)

    if passed == total:
        logger.info("✓ All tests passed!")
        return 0
    else:
        logger.warning(f"⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
