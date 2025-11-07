#!/usr/bin/env python3
"""Standalone script to convert gemma3:270m Ollama model to HuggingFace format."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from src.quantization.converters.gguf_to_hf import GGUFToHFConverter

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


def main():
    """Convert gemma3:270m from Ollama to HuggingFace format."""
    print("=" * 80)
    print(" " * 20 + "GGUF → HuggingFace Standalone Converter")
    print("=" * 80)
    
    # Input: Ollama GGUF file
    source_path = Path("/Users/hemanth/.ollama/models/blobs/sha256-735af2139dc652bf01112746474883d79a52fa1c19038265d363e3d42556f7a2")
    
    # Output: HuggingFace safetensors directory
    output_path = Path("/Volumes/X9Pro/visualstudiocodes/vlm-tester - Copy/src/quantization/gemma3_270m_hf")
    
    print(f"\nSource (Ollama GGUF): {source_path}")
    print(f"Output (HF Safetensors): {output_path}")
    
    # Check source exists
    if not source_path.exists():
        print(f"\n✗ Source file not found: {source_path}")
        return 1
    
    size_gb = source_path.stat().st_size / (1024**3)
    print(f"Source size: {size_gb:.2f} GB")
    
    # Create converter
    converter = GGUFToHFConverter()
    
    # Check availability
    available, msg = converter.check_availability()
    print(f"\nConverter: {msg}")
    
    if not available:
        print("✗ Converter not available")
        return 1
    
    # Remove existing output if present
    if output_path.exists():
        import shutil
        print(f"\n⚠ Removing existing output: {output_path}")
        shutil.rmtree(output_path)
    
    # Progress callback
    def progress(percent, eta):
        bar_width = 40
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  Progress: [{bar}] {percent:5.1f}%", end="", flush=True)
    
    # Run conversion
    print("\n\nStarting conversion...")
    print("This will:")
    print("  1. Read GGUF file and extract metadata")
    print("  2. Detect architecture (gemma3)")
    print("  3. Load all tensors (FP16/quantized)")
    print("  4. Generate HuggingFace config.json")
    print("  5. Remap tensor names (GGUF → HF convention)")
    print("  6. Save as model.safetensors")
    print()
    
    success = converter.convert(
        source_path=source_path,
        output_path=output_path,
        progress_callback=progress
    )
    
    print()  # New line after progress bar
    
    if success:
        print("\n" + "=" * 80)
        print("✓ CONVERSION SUCCESSFUL")
        print("=" * 80)
        
        # Verify output
        print("\nOutput files:")
        
        config_file = output_path / "config.json"
        if config_file.exists():
            print(f"  ✓ config.json ({config_file.stat().st_size:,} bytes)")
            
            # Show config details
            import json
            with open(config_file) as f:
                config = json.load(f)
            
            print("\n  Configuration:")
            print(f"    Architecture: {config.get('architectures', ['N/A'])[0]}")
            print(f"    Model type: {config.get('model_type', 'N/A')}")
            print(f"    Hidden size: {config.get('hidden_size', 'N/A')}")
            print(f"    Num layers: {config.get('num_hidden_layers', 'N/A')}")
            print(f"    Num attention heads: {config.get('num_attention_heads', 'N/A')}")
            print(f"    Vocab size: {config.get('vocab_size', 'N/A')}")
        
        model_file = output_path / "model.safetensors"
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024**2)
            print(f"\n  ✓ model.safetensors ({size_mb:.2f} MB)")
            
            # Load and verify tensors
            try:
                from safetensors import safe_open
                
                with safe_open(model_file, framework="pt") as f:
                    tensor_names = list(f.keys())
                    print(f"\n  Tensors: {len(tensor_names)} total")
                    
                    # Show tensor statistics
                    total_params = 0
                    print("\n  Sample tensors:")
                    for i, name in enumerate(tensor_names[:10]):
                        tensor = f.get_tensor(name)
                        shape = tensor.shape
                        params = 1
                        for dim in shape:
                            params *= dim
                        total_params += params
                        
                        print(f"    {i+1:2}. {name:50} {str(list(shape)):20} {params:>12,} params")
                    
                    if len(tensor_names) > 10:
                        print(f"    ... and {len(tensor_names) - 10} more tensors")
                    
                    # Calculate remaining params
                    for name in tensor_names[10:]:
                        tensor = f.get_tensor(name)
                        params = 1
                        for dim in tensor.shape:
                            params *= dim
                        total_params += params
                    
                    print(f"\n  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
                    
            except Exception as e:
                print(f"  ⚠ Could not analyze tensors: {e}")
        
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print(f"\nThe converted model is ready for use with:")
        print("  • transformers library (AutoModel.from_pretrained)")
        print("  • Generic quantization (INT4, INT8, FP16)")
        print("  • GPTQ quantization")
        print("  • AWQ quantization")
        print("  • BitsAndBytes quantization")
        
        print(f"\nModel location: {output_path}")
        
        print("\nExample usage:")
        print("  from transformers import AutoModelForCausalLM")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{output_path}')")
        
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ CONVERSION FAILED")
        print("=" * 80)
        print("\nCheck the logs above for error details")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ Conversion cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Conversion crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
