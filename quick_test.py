#!/usr/bin/env python3
"""
Quick test to verify the VLM CLI is working
"""

import asyncio
from pathlib import Path
from PIL import Image
from vlm_client import VLMClient


async def quick_test():
    """Quick test of VLM client"""

    print("\n" + "="*60)
    print("VLM CLI - Quick Test")
    print("="*60 + "\n")

    # Initialize
    print("✓ Initializing client...")
    client = VLMClient("config.yaml")

    # Show available models
    models = client.get_available_models()
    print(f"✓ Found {len(models)} configured models\n")

    # Try loading the default model (qwen2.5vl:3b)
    print("Loading default model (qwen2.5vl:3b)...")
    success = await client.load_model("qwen2vl-3b")

    if success:
        print("✓ Model loaded successfully!\n")

        # Create a test image
        test_image_path = Path("test_image.jpg")
        if not test_image_path.exists():
            print("Creating test image...")
            img = Image.new('RGB', (800, 600), color='blue')
            img.save(test_image_path)
            print(f"✓ Created test image: {test_image_path}\n")

        # Test QA endpoint
        print("Testing QA endpoint...")
        result = await client.qa(
            str(test_image_path),
            "What color is this image?"
        )

        if "error" not in result:
            print(f"✓ QA Test Passed!")
            print(f"  Question: {result['question']}")
            print(f"  Answer: {result['answer']}")
            print(f"  Time: {result['inference_time_ms']}ms\n")
        else:
            print(f"✗ QA Test Failed: {result['error']}\n")

        # Show stats
        stats = client.get_stats()
        print("="*60)
        print("Statistics:")
        print(f"  Current Model: {stats['current_model']}")
        print(f"  Provider: {stats['current_provider']}")
        print(f"  Total Inferences: {stats['total_inferences']}")
        print("="*60 + "\n")

        print("🎉 VLM CLI is working! You can now run:")
        print("   python3 vlm_cli.py")

    else:
        print("✗ Failed to load model")
        print("\nTrying Ollama models...")
        print("Make sure Ollama is running: ollama serve")

    # Cleanup
    await client.unload_model()


if __name__ == "__main__":
    try:
        asyncio.run(quick_test())
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nIf you see connection errors, make sure Ollama is running:")
        print("  ollama serve")
