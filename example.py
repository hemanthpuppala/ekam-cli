#!/usr/bin/env python3
"""
Example: Programmatic usage of VLM Client
This shows how to use the VLM client in your own scripts
"""

import asyncio
from vlm_client import VLMClient


async def main():
    """Example usage of VLM client"""

    # Initialize client
    print("Initializing VLM Client...")
    client = VLMClient("config.yaml")

    # Load a model
    print("Loading qwen2.5vl:3b model...")
    success = await client.load_model("qwen2vl-3b")

    if not success:
        print("Failed to load model")
        return

    print("Model loaded successfully!\n")

    # Example 1: Question Answering
    print("=" * 60)
    print("Example 1: Question Answering")
    print("=" * 60)

    result = await client.qa(
        image_path="path/to/your/image.jpg",
        question="What objects can you see in this image?"
    )

    if "error" not in result:
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Time: {result['inference_time_ms']}ms\n")
    else:
        print(f"Error: {result['error']}\n")

    # Example 2: Image Captioning
    print("=" * 60)
    print("Example 2: Image Captioning")
    print("=" * 60)

    result = await client.caption(
        image_path="path/to/your/image.jpg",
        detail_level="detailed"
    )

    if "error" not in result:
        print(f"Caption: {result['caption']}")
        print(f"Time: {result['inference_time_ms']}ms\n")
    else:
        print(f"Error: {result['error']}\n")

    # Example 3: Object Detection
    print("=" * 60)
    print("Example 3: Object Detection")
    print("=" * 60)

    result = await client.detect(
        image_path="path/to/your/image.jpg",
        object_name="car"
    )

    if "error" not in result:
        print(f"Object: {result['object']}")
        print(f"Detections: {result.get('detections', 'N/A')}")
        print(f"Time: {result['inference_time_ms']}ms\n")
    else:
        print(f"Error: {result['error']}\n")

    # Example 4: Object Pointing
    print("=" * 60)
    print("Example 4: Object Pointing")
    print("=" * 60)

    result = await client.point(
        image_path="path/to/your/image.jpg",
        object_name="person"
    )

    if "error" not in result:
        print(f"Object: {result['object']}")
        print(f"Location: {result.get('location', result.get('coordinates', 'N/A'))}")
        print(f"Time: {result['inference_time_ms']}ms\n")
    else:
        print(f"Error: {result['error']}\n")

    # Get statistics
    print("=" * 60)
    print("Statistics")
    print("=" * 60)
    stats = client.get_stats()
    print(f"Total inferences: {stats['total_inferences']}")
    print(f"Average time: {stats['average_time_ms']}ms")

    # Cleanup
    await client.unload_model()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
