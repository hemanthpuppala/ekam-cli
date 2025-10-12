#!/usr/bin/env python3
"""
Test all endpoints with Moondream model using the fixed prompts
"""
import asyncio
import sys
from pathlib import Path
from vlm_client import VLMClient


async def test_moondream_endpoints():
    """Test all endpoints with moondream model"""
    print("\n" + "="*70)
    print("Testing Moondream Endpoints with Fixed Prompts")
    print("="*70)

    client = VLMClient("config.yaml")
    test_image = "/Users/hemanth/Desktop/road-condition-analyzer/vlm-tester/test_images/test1.jpeg"

    # Load moondream model
    print("\nLoading moondream:latest...")
    success = await client._load_ollama_model("moondream:latest")
    if not success:
        print("✗ Failed to load model")
        return False

    client.current_model = "moondream:latest"
    client.current_provider = "ollama"
    print("✓ Model loaded\n")

    results = []

    # Test 1: QA Endpoint
    print("="*70)
    print("TEST 1: QA Endpoint")
    print("="*70)
    question = "What objects do you see in this image?"
    print(f"Question: {question}")

    result = await client.qa(test_image, question)

    if "error" in result:
        print(f"✗ Error: {result['error']}")
        results.append(("QA", False))
    elif result.get('answer') and len(result['answer'].strip()) > 0:
        print(f"✓ Answer ({len(result['answer'])} chars): {result['answer'][:150]}...")
        print(f"  Time: {result['inference_time_ms']}ms")
        results.append(("QA", True))
    else:
        print(f"✗ Empty answer")
        results.append(("QA", False))

    # Test 2: Caption Endpoint
    print("\n" + "="*70)
    print("TEST 2: Caption Endpoint")
    print("="*70)

    result = await client.caption(test_image, detail_level="detailed")

    if "error" in result:
        print(f"✗ Error: {result['error']}")
        results.append(("Caption", False))
    elif result.get('caption') and len(result['caption'].strip()) > 0:
        print(f"✓ Caption ({len(result['caption'])} chars): {result['caption'][:150]}...")
        print(f"  Time: {result['inference_time_ms']}ms")
        results.append(("Caption", True))
    else:
        print(f"✗ Empty caption")
        results.append(("Caption", False))

    # Test 3: Detect Endpoint
    print("\n" + "="*70)
    print("TEST 3: Detect Endpoint")
    print("="*70)
    object_to_detect = "car"
    print(f"Detecting: {object_to_detect}")

    result = await client.detect(test_image, object_to_detect)

    if "error" in result:
        print(f"✗ Error: {result['error']}")
        results.append(("Detect", False))
    elif result.get('detections') and len(result['detections'].strip()) > 0:
        print(f"✓ Detection ({len(result['detections'])} chars): {result['detections'][:150]}...")
        print(f"  Time: {result['inference_time_ms']}ms")
        results.append(("Detect", True))
    else:
        print(f"✗ Empty detection")
        results.append(("Detect", False))

    # Test 4: Point Endpoint
    print("\n" + "="*70)
    print("TEST 4: Point Endpoint")
    print("="*70)
    object_to_locate = "car"
    print(f"Locating: {object_to_locate}")

    result = await client.point(test_image, object_to_locate)

    if "error" in result:
        print(f"✗ Error: {result['error']}")
        results.append(("Point", False))
    elif result.get('location') and len(result['location'].strip()) > 0:
        print(f"✓ Location ({len(result['location'])} chars): {result['location'][:150]}...")
        print(f"  Time: {result['inference_time_ms']}ms")
        results.append(("Point", True))
    else:
        print(f"✗ Empty location")
        results.append(("Point", False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for endpoint, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {endpoint}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All moondream endpoints work correctly!")
        return True
    else:
        print(f"\n⚠ {total_tests - total_passed} endpoint(s) still need fixing")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_moondream_endpoints())
    sys.exit(0 if success else 1)
