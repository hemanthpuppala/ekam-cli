#!/usr/bin/env python3
"""
Quick script to check your HuggingFace models
"""
from pathlib import Path

def check_models():
    cache_dir = Path.home() / ".cache/huggingface/hub"
    
    if not cache_dir.exists():
        print("❌ HuggingFace cache not found!")
        print(f"   Expected: {cache_dir}")
        return
    
    print("=" * 70)
    print("YOUR HUGGINGFACE MODELS")
    print("=" * 70)
    print(f"\nLocation: {cache_dir}\n")
    
    models = []
    for model_dir in sorted(cache_dir.glob("models--*")):
        model_name = model_dir.name.replace("models--", "").replace("--", "/")
        
        # Check for actual model files
        has_weights = False
        weight_files = []
        
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot_dir in snapshots_dir.glob("*"):
                bins = list(snapshot_dir.glob("*.bin"))
                safetensors = list(snapshot_dir.glob("*.safetensors"))
                
                if bins or safetensors:
                    has_weights = True
                    weight_files.extend([f.name for f in bins + safetensors])
        
        # Calculate size
        size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
        size_gb = size / 1e9
        
        status = "✅ Complete" if has_weights else "⚠️  Incomplete (no weights)"
        models.append((model_name, size_gb, status, has_weights))
    
    # Sort by size
    models.sort(key=lambda x: x[1], reverse=True)
    
    # Display
    complete_count = sum(1 for m in models if m[3])
    incomplete_count = len(models) - complete_count
    
    print(f"Total Models: {len(models)}")
    print(f"Complete: {complete_count} | Incomplete: {incomplete_count}\n")
    print("-" * 70)
    
    for model_name, size_gb, status, _ in models:
        if size_gb < 0.001:
            size_str = f"{size_gb*1000:.0f}KB"
        elif size_gb < 0.1:
            size_str = f"{size_gb*1000:.1f}MB"
        else:
            size_str = f"{size_gb:.2f}GB"
        
        print(f"{model_name:45s} {size_str:>10s}  {status}")
    
    print("-" * 70)
    total_size = sum(m[1] for m in models)
    print(f"{'TOTAL DISK USAGE':45s} {total_size:>9.2f}GB")
    print("=" * 70)
    
    # Show incomplete models
    incomplete = [m for m in models if not m[3]]
    if incomplete:
        print("\n⚠️  INCOMPLETE MODELS (only configs, no weights):")
        for model_name, _, _, _ in incomplete:
            print(f"   • {model_name}")
        print("\n   To complete download, use: python3 vlm_cli.py")

if __name__ == "__main__":
    check_models()
