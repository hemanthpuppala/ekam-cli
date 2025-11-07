"""Test script to measure tokenization overhead in llama-cpp-python.

This script validates the performance issue report by measuring how long
it takes to tokenize prompts of various lengths using llama-cpp-python.
"""

import time
from pathlib import Path

def test_tokenization_overhead():
    """Measure tokenization performance on CPU."""

    # Find the qwen2 model
    model_path = Path.home() / "models" / "gguf" / "qwen2-0_5b-instruct-q4_k_m.gguf"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    print("Loading llama-cpp-python model...")
    from llama_cpp import Llama

    # Load model (minimal config, no GPU to isolate CPU tokenization)
    model = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_gpu_layers=0,  # Force CPU to isolate tokenization overhead
        verbose=False
    )

    print(f"Model loaded: {model_path.name}\n")

    # Test cases with different prompt lengths
    test_cases = [
        ("Short (100 chars)", "hi" * 50),
        ("Medium (1K chars)", "This is a test message. " * 42),
        ("Long (4K chars)", "This is a test message with context. " * 108),
        ("Very Long (16K chars)", "This is a conversation history with multiple turns. " * 308),
    ]

    print("=" * 70)
    print("TOKENIZATION PERFORMANCE TEST (CPU-only)")
    print("=" * 70)

    for name, prompt in test_cases:
        prompt_len = len(prompt)

        # Time tokenization
        start = time.perf_counter()
        tokens = model.tokenize(prompt.encode('utf-8'))
        end = time.perf_counter()

        tokenization_time_ms = (end - start) * 1000
        token_count = len(tokens)

        print(f"\n{name}:")
        print(f"  Characters: {prompt_len:,}")
        print(f"  Tokens: {token_count:,}")
        print(f"  Tokenization time: {tokenization_time_ms:.2f}ms")
        print(f"  Tokens/sec: {token_count / (tokenization_time_ms / 1000):.0f}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
The performance issue report claims that tokenizing a 16K character prompt
(~4000 tokens with conversation history) takes ~6 seconds on CPU.

If your results show similar numbers (e.g., >2000ms for long prompts), the
report is accurate and the fix should be applied.

KEY INSIGHT:
- llama.cpp web UI doesn't tokenize the full prompt on CPU before inference
- It lets the model tokenize during inference (much faster with GPU)
- Ekam tokenizes purely to calculate token counts for safety margins
- This adds 5-7 seconds of pure overhead that could be avoided
""")

if __name__ == "__main__":
    test_tokenization_overhead()
