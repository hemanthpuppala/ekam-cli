"""Educational information about finetuning methods."""

from enum import Enum

from ..config import FinetuneMethod


class MethodInformation:
    """Professional documentation for each finetuning method."""

    INFO = {
        FinetuneMethod.LORA: {
            "title": "LoRA (Low-Rank Adaptation)",
            "description": """
Low-Rank Adaptation is a parameter-efficient finetuning method that adds small,
trainable "adapter" layers to your model while keeping the main weights frozen.

Instead of updating all 7 billion parameters in a 7B model, LoRA only updates
around 0.06% of parameters (~4.2 million), reducing memory usage significantly.
""",
            "how_it_works": """
1. For each layer you want to adapt, LoRA adds two small matrices A and B
2. During forward pass: output = original_output + scale * A @ B @ input
3. Only A and B matrices are updated during training (frozen original weights)
4. After training, A and B can be merged back into the model or kept separate

The "rank" (r) parameter controls the size of A and B:
  - r=8: ~0.06% trainable params, minimal quality loss
  - r=16: ~0.12% trainable params, better quality improvement
  - r=32: ~0.24% trainable params, marginal gains, more VRAM needed
""",
            "best_for": """
✓ Domain adaptation with moderate datasets (1K-100K examples)
✓ Finetuning on consumer hardware (8GB+ VRAM)
✓ Fine-grained control over adaptation (multiple adapters per model)
✓ Quick experimentation and iteration
✓ Production deployments (small adapter files ~5-50MB)
""",
            "memory_requirements": """
For 7B model (f16 precision):
  • Model size: ~14GB
  • LoRA adapters: ~50-100MB (tiny!)
  • Optimizer states: ~7GB
  • Activations: ~2GB
  TOTAL: ~23GB VRAM → With LoRA: ~13.6GB VRAM (57% reduction)
""",
            "recommendations": {
                "dataset_size": "1K - 100K examples (sweet spot: 10K-50K)",
                "learning_rate": "1e-4 to 5e-4 (start with 1e-4)",
                "rank_r": "8 (default) to 16 (high quality needed)",
                "batch_size": "4-8 for 8GB VRAM, 8-16 for 16GB+",
                "epochs": "3-5 epochs usually sufficient",
                "hardware": "RTX 3060 (12GB) minimum, RTX 4090+ recommended",
                "training_time": "2-4 hours for 10K examples on RTX 4090",
            },
            "data_needs": """
Finetuning data format: JSONL, JSON, or CSV with 'text' column
Example JSON format:
{
  "text": "Your training example text here..."
}

Data tips:
• Include task-specific examples your model should learn
• Domain: Legal docs, medical text, code, creative writing, etc.
• Quality > Quantity: 1K high-quality examples > 100K poor examples
• Aim for balance and diversity in examples
• Remove duplicates and low-quality/corrupted text
""",
            "training_duration": {
                "small": "1K examples: 5-15 mins",
                "medium": "10K examples: 30-60 mins",
                "large": "100K examples: 4-8 hours",
            },
            "quality": "Excellent - near-full finetuning quality with fraction of params",
            "speed": "Fast - 2-3x faster than full finetuning",
            "cost": "Low - minimal VRAM and storage needed",
        },
        FinetuneMethod.QLORA: {
            "title": "QLoRA (Quantized Low-Rank Adaptation)",
            "description": """
QLoRA combines 4-bit quantization with LoRA for extreme memory efficiency.
Quantization compresses model weights from 16-bit to 4-bit precision, reducing
size and VRAM by 4x. Combined with LoRA's parameter efficiency, you get the
most memory-efficient finetuning possible.

You can finetune a 70B parameter model with just 48GB VRAM!
""",
            "how_it_works": """
1. Load model in 4-bit quantized format (NormalFloat4 or Float4)
2. Unquantize only small LoRA adapter matrices for computation
3. Gradients flow back through the frozen quantized weights
4. After training, keep the small adapter files (100MB-300MB)

Quantization reduces:
  • Model weights: 16-bit → 4-bit (4x smaller)
  • Memory footprint: 14GB model → 3.5GB in 4-bit
  • But maintains reasonable quality through adapter learning
""",
            "best_for": """
✓ Large models on consumer hardware (7B-70B models with 8-48GB VRAM)
✓ Budget-conscious production deployments
✓ Quick prototyping without GPU investment
✓ Running multiple models simultaneously
✓ Edge deployment scenarios (small adapters only)
""",
            "memory_requirements": """
For 7B model (4-bit quantized):
  • Quantized model: ~3.5GB
  • LoRA adapters: ~50-100MB
  • Optimizer: ~1.8GB
  • Activations: ~1GB
  TOTAL: ~6.3GB VRAM! (vs 24GB for full finetuning)

For 13B model:
  • Total: ~8-10GB VRAM

For 70B model:
  • Total: ~45-50GB VRAM (vs 1.5TB+ for full finetuning)
""",
            "recommendations": {
                "dataset_size": "5K - 100K examples",
                "learning_rate": "1e-4 to 1e-5 (lower than LoRA)",
                "quantization_bits": "4 (recommended) or 8 (higher quality)",
                "rank_r": "8-16 (same as LoRA)",
                "batch_size": "4-8 (very memory efficient)",
                "epochs": "3-5 epochs",
                "hardware": "Any GPU with 8GB+ VRAM, even RTX 3060",
                "training_time": "1-2x slower than LoRA due to unquantization",
            },
            "data_needs": """
Same as LoRA, but even more effective with quantization:
• JSON/JSONL/CSV format with 'text' column
• Quality matters more than quantity for 4-bit models
• Start with 5K-20K examples for best results
• Higher quality examples compensate for quantization loss
""",
            "training_duration": {
                "small": "1K examples: 10-20 mins",
                "medium": "10K examples: 60-120 mins",
                "large": "100K examples: 10-20 hours",
            },
            "quality": "Good - slight quality loss from quantization, compensated by adapters",
            "speed": "Slower than LoRA (2-3x) due to unquantization overhead",
            "cost": "Ultra-low - works on 8GB consumer hardware",
        },
        FinetuneMethod.FULL: {
            "title": "Full Model Finetuning",
            "description": """
Full finetuning updates every parameter in the model. This is the original
finetuning method and provides maximum quality improvements, but at the cost
of significant memory and computational requirements.

Use this only when you have the hardware and need maximum accuracy.
""",
            "how_it_works": """
1. Load entire model (all parameters)
2. Initialize optimizer states for each parameter
3. During training: backward pass updates ALL weights
4. Gradients computed for all 7 billion (or more) parameters

This is the standard PyTorch/TensorFlow training loop:
  loss.backward()  # Compute gradients for all params
  optimizer.step()  # Update all params
""",
            "best_for": """
✓ Maximum quality on custom domains with unlimited data
✓ Complete architectural changes needed
✓ Research and benchmarking
✓ Enterprise deployments with ample hardware
✓ When single-digit percentage quality improvements matter
""",
            "memory_requirements": """
For 7B model (f16 precision):
  • Model weights: ~14GB
  • Optimizer states (Adam): ~28GB (2x model size)
  • Gradients: ~14GB
  • Activations: ~2-4GB
  TOTAL: ~58-60GB VRAM minimum (RTX A100 or H100)

For 13B model:
  • Total: ~120GB VRAM (A100 80GB + A100 40GB setup)

This is why full finetuning requires professional hardware!
""",
            "recommendations": {
                "dataset_size": "50K - 1M+ examples",
                "learning_rate": "5e-5 to 2e-5 (very conservative)",
                "batch_size": "16-32 (large batches for stability)",
                "gradient_accumulation": "1-4 steps",
                "epochs": "1-3 epochs (can overfit quickly)",
                "warmup_ratio": "10% of total steps",
                "hardware": "A100 (40GB+) or H100 (80GB), minimum 2x GPUs",
                "training_time": "1-7 days for million examples",
                "learning_rate_schedule": "Cosine with warmup (standard)",
            },
            "data_needs": """
Requires substantial, high-quality data:
• 50K-1M+ carefully curated examples
• JSON/JSONL/CSV format
• Domain-specific, representative of actual use cases
• Data cleaning, deduplication, filtering critical
• Class/topic balance important

Red flags if data < 50K:
  ⚠ Likely to overfit
  ⚠ Better off using LoRA or QLoRA
  ⚠ Wasting computational resources
""",
            "training_duration": {
                "small": "50K examples: 5-10 hours on A100",
                "medium": "200K examples: 1-2 days on A100",
                "large": "1M examples: 1-2 weeks on 8x A100",
            },
            "quality": "Excellent - best possible quality on custom data",
            "speed": "Slowest - requires significant compute time",
            "cost": "Very High - requires enterprise GPU hardware",
        },
    }

    @staticmethod
    def get_method_info(method: FinetuneMethod) -> dict:
        """Get complete information about a finetuning method.

        Args:
            method: FinetuneMethod enum value

        Returns:
            Dictionary with all method information
        """
        return MethodInformation.INFO.get(method, {})

    @staticmethod
    def get_comparison() -> str:
        """Get formatted comparison of all methods.

        Returns:
            Comparison table as string
        """
        comparison = """
╔════════════════════════════════════════════════════════════════════════════╗
║                  FINETUNING METHODS COMPARISON                              ║
╠════════════════════════════════════════════════════════════════════════════╣
║ Aspect         │ LoRA          │ QLoRA         │ Full           │          ║
╠════════════════════════════════════════════════════════════════════════════╣
║ Memory (7B)    │ 13.6GB        │ 6.3GB ⭐      │ 58-60GB        │          ║
║ VRAM Min       │ 12GB          │ 8GB ⭐        │ 24GB+          │          ║
║ Quality        │ ⭐⭐⭐⭐⭐     │ ⭐⭐⭐⭐      │ ⭐⭐⭐⭐⭐      │          ║
║ Speed          │ Fast ⭐⭐⭐⭐  │ Medium ⭐⭐⭐  │ Slow ⭐⭐       │          ║
║ Training Time  │ 30-60 min     │ 60-120 min    │ Hours-Days     │          ║
║ Adapter Size   │ 50-100MB      │ 100-300MB     │ ~14GB full     │          ║
║ Best for       │ Quick tuning  │ Large models  │ Max quality    │          ║
║ Hardware Req.  │ RTX 3060      │ Any GPU       │ A100/H100      │          ║
║ Cost Factor    │ Low ⭐        │ Very Low ⭐⭐ │ Very High ⭐   │          ║
╚════════════════════════════════════════════════════════════════════════════╝

RECOMMENDATIONS BY USE CASE:
────────────────────────────────────────────────────────────────────────────

📋 Document Classification (legal, medical) → LoRA (fast, accurate)
🔬 Scientific Paper Analysis → LoRA (balanced quality/speed)
💬 Chatbot/Custom Personality → LoRA or QLoRA (depends on hardware)
🏭 High-Volume Production → QLoRA (cost-efficient at scale)
📊 Data Analysis/SQL Generation → LoRA (straightforward)
💻 Code Generation / Refactoring → Full (if possible, best results)
🎯 Specialized Domain Adaptation → LoRA (best price/quality ratio)
🚀 Real-time Inference on Edge → QLoRA (compact adapters)
🔍 Maximum Accuracy Benchmark → Full (unlimited hardware)

QUICK DECISION TREE:
────────────────────────────────────────────────────────────────────────────

1. Do you have unlimited budget and 24GB+ VRAM?
   → Full Finetuning (best quality)

2. Do you have 12-16GB VRAM and reasonable dataset?
   → LoRA (best balance)

3. Do you have < 12GB VRAM or large models?
   → QLoRA (works with 8GB)

4. Is inference speed/size critical?
   → QLoRA (smallest adapters)

5. Testing/prototyping with limited data?
   → LoRA (fastest to validate)
"""
        return comparison
