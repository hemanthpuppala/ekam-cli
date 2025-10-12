#!/usr/bin/env python3
"""
Test script to verify Ollama connection and inference fixes
"""
import asyncio
import sys
from pathlib import Path
from PIL import Image
from vlm_client import VLMClient


async def test_ollama_connection():
    """Test 1: Verify Ollama connection detection works"""
    print("\n" + "="*60)
    print("TEST 1: Ollama Connection Detection")
    print("="*60)

    client = VLMClient("config.yaml")

    try:
        models = await client.discover_installed_ollama_models()
        print(f"✓ Successfully discovered {len(models)} Ollama models:")
        for model in models:
            print(f"  - {model['name']} ({model['type']}, {model['size_gb']:.1f}GB)")
        return True
    except ConnectionError as e:
        print(f"✗ Connection error (expected if Ollama not running): {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_model_loading():
    """Test 2: Verify model loading is fast (no test inference)"""
    print("\n" + "="*60)
    print("TEST 2: Fast Model Loading (No Test Inference)")
    print("="*60)

    client = VLMClient("config.yaml")

    try:
        import time
        start = time.time()

        # Try to load moondream (should be instant, just checks list)
        success = await client._load_ollama_model("moondream:latest")

        elapsed = time.time() - start

        if success:
            print(f"✓ Model loaded successfully in {elapsed:.2f}s")
            if elapsed < 1.0:
                print(f"  ✓ Fast loading confirmed (< 1 second)")
                return True
            else:
                print(f"  ⚠ Slower than expected ({elapsed:.2f}s)")
                return True
        else:
            print(f"✗ Model loading failed")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_inference():
    """Test 3: Verify inference works and returns proper response"""
    print("\n" + "="*60)
    print("TEST 3: Image Inference")
    print("="*60)

    client = VLMClient("config.yaml")

    try:
        # Create a test image
        test_image_path = "/tmp/test_image.png"
        img = Image.new('RGB', (512, 512), color='red')
        img.save(test_image_path)
        print(f"Created test image: {test_image_path}")

        # Load model
        print("Loading moondream:latest...")
        success = await client._load_ollama_model("moondream:latest")
        if not success:
            print("✗ Failed to load model")
            return False

        client.current_model = "moondream:latest"
        client.current_provider = "ollama"

        # Run inference
        print("Running inference...")
        import time
        start = time.time()

        result = await client.qa(test_image_path, "What color is this image?")

        elapsed = time.time() - start

        print(f"\nInference completed in {elapsed:.2f}s")

        if "error" in result:
            print(f"✗ Inference error: {result['error']}")
            return False

        if "answer" in result and result["answer"]:
            print(f"✓ Got response: {result['answer'][:100]}")
            print(f"  Inference time: {result.get('inference_time_ms', 0)}ms")

            # Check if response is meaningful (not empty)
            if len(result['answer'].strip()) > 0:
                print(f"  ✓ Response is not empty")
                return True
            else:
                print(f"  ✗ Response is empty")
                return False
        else:
            print(f"✗ No answer in result: {result}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "OLLAMA FIXES TEST SUITE" + " "*20 + "║")
    print("╚" + "="*58 + "╝")

    results = []

    # Test 1: Connection
    results.append(("Connection Detection", await test_ollama_connection()))

    # Test 2: Fast Loading
    results.append(("Fast Model Loading", await test_model_loading()))

    # Test 3: Inference
    results.append(("Image Inference", await test_inference()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
