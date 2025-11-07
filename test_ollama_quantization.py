#!/usr/bin/env python3
"""Test script to verify Ollama model discovery and quantization pipeline."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger

# Configure logger for testing
logger.remove()
logger.add(sys.stderr, level="INFO")

def test_ollama_discovery():
    """Test Ollama model discovery."""
    from src.providers.ollama import OllamaProvider
    from src.models.provider import ProviderConfig
    from src.models.endpoints import ProviderType

    print("\n" + "="*80)
    print("TEST 1: Ollama Model Discovery")
    print("="*80)

    # Create Ollama provider
    config = ProviderConfig(
        name=ProviderType.OLLAMA,
        host="http://localhost:11434",
        timeout_seconds=30
    )
    provider = OllamaProvider(config)

    try:
        models = provider.discover_models()
        print(f"\n✓ Discovered {len(models)} Ollama models")

        if models:
            print("\nModels found:")
            for i, model in enumerate(models, 1):
                print(f"\n  {i}. {model.name}")
                print(f"     Provider: {model.provider} (type: {type(model.provider).__name__})")
                print(f"     Model Type: {model.model_type}")
                print(f"     Size: {model.size_gb:.2f} GB")
                print(f"     Quantization: {model.quantization}")
                print(f"     Source Path: {model.source_path}")
                print(f"     Architecture: {model.architecture}")
        else:
            print("\n⚠ No Ollama models found. Is Ollama running? Try: ollama list")

        return models
    except Exception as e:
        print(f"\n✗ Error discovering Ollama models: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_quantization_filtering(all_models):
    """Test quantization manager filtering."""
    from src.quantization.manager import QuantizationManager
    from src.models.system import SystemSpecs

    print("\n" + "="*80)
    print("TEST 2: Quantization Manager Filtering")
    print("="*80)

    # Create mock system specs
    system_specs = SystemSpecs(
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        total_vram_gb=0.0,
        available_vram_gb=0.0,
        cpu_count=8,
        platform="darwin",
        has_cuda=False,
        has_mps=True,
    )

    output_dir = Path("results/quantizations")
    quant_manager = QuantizationManager(system_specs, output_dir)

    print(f"\nTotal models to filter: {len(all_models)}")

    # Print details of all models
    print("\nAll models by provider:")
    from collections import defaultdict
    by_provider = defaultdict(list)
    for model in all_models:
        by_provider[str(model.provider)].append(model)

    for provider, models in by_provider.items():
        print(f"  {provider}: {len(models)} models")
        for model in models:
            print(f"    - {model.name} (source_path: {model.source_path is not None})")

    # Filter quantizable models
    quantizable = quant_manager.get_quantizable_models(all_models)

    print(f"\n✓ Quantizable models: {len(quantizable)}")

    if quantizable:
        print("\nQuantizable models by provider:")
        by_provider_quant = defaultdict(list)
        for model in quantizable:
            by_provider_quant[str(model.provider)].append(model)

        for provider, models in by_provider_quant.items():
            print(f"\n  {provider.upper()}: {len(models)} models")
            for model in models:
                print(f"    - {model.name}")
                print(f"      Source Path: {model.source_path}")
                print(f"      Quantization: {model.quantization}")
    else:
        print("\n⚠ No quantizable models found!")

    return quantizable


def test_provider_type_comparison():
    """Test provider type comparison."""
    from src.models.endpoints import ProviderType

    print("\n" + "="*80)
    print("TEST 3: Provider Type Comparison")
    print("="*80)

    # Test enum comparison
    print("\nProviderType.OLLAMA:", ProviderType.OLLAMA)
    print("Type:", type(ProviderType.OLLAMA))
    print("Value:", ProviderType.OLLAMA.value)

    # Test string comparison
    test_provider = ProviderType.OLLAMA
    print(f"\ntest_provider == ProviderType.OLLAMA: {test_provider == ProviderType.OLLAMA}")
    print(f"test_provider == 'ollama': {test_provider == 'ollama'}")
    print(f"str(test_provider): {str(test_provider)}")

    # Test in conditional
    if test_provider == ProviderType.OLLAMA:
        print("✓ Enum comparison works!")
    else:
        print("✗ Enum comparison FAILED!")


def test_ollama_file_locator():
    """Test Ollama file locator."""
    from src.utils.ollama_file_locator import OllamaFileLocator

    print("\n" + "="*80)
    print("TEST 4: Ollama File Locator")
    print("="*80)

    locator = OllamaFileLocator()

    # List available blobs
    blobs = locator.list_available_blobs()
    print(f"\nFound {len(blobs)} blob files in ~/.ollama/models/blobs/")

    if blobs:
        print("\nFirst 5 blobs:")
        for blob in blobs[:5]:
            print(f"  - {blob.name} ({blob.stat().st_size / (1024**3):.2f} GB)")

    # Try to get a model path
    print("\nTesting model path resolution...")
    print("Note: This will only work if you have Ollama models installed")

    # You can add specific model names to test here
    # test_models = ["llama3.2:3b", "mistral:7b"]
    # for model_name in test_models:
    #     path = locator.get_model_path(model_name)
    #     print(f"  {model_name}: {path}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("OLLAMA QUANTIZATION INTEGRATION TEST")
    print("="*80)

    # Test 3: Provider type comparison
    test_provider_type_comparison()

    # Test 4: File locator
    test_ollama_file_locator()

    # Test 1: Discover Ollama models
    ollama_models = test_ollama_discovery()

    # Create a mock model list (would normally include GGUF, HF models too)
    all_models = ollama_models.copy()

    # Test 2: Filter quantizable models
    if all_models:
        quantizable = test_quantization_filtering(all_models)

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total Ollama models discovered: {len(ollama_models)}")
        print(f"Total quantizable models: {len(quantizable)}")

        ollama_quantizable = [m for m in quantizable if str(m.provider) == 'ollama']
        print(f"Ollama models in quantizable list: {len(ollama_quantizable)}")

        if ollama_quantizable:
            print("\n✓ SUCCESS: Ollama models are showing in quantization pipeline!")
            print("\nThese Ollama models should appear in the UI:")
            for model in ollama_quantizable:
                print(f"  - {model.name}")
        else:
            print("\n✗ PROBLEM: Ollama models NOT appearing in quantizable list")
            print("\nDebugging info:")
            for model in ollama_models:
                print(f"\n  Model: {model.name}")
                print(f"    provider type: {type(model.provider)}")
                print(f"    provider value: {model.provider}")
                print(f"    source_path: {model.source_path}")
    else:
        print("\n⚠ No models to test with. Ensure Ollama is running with models installed.")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
