# Memory Management in VLM CLI

## Overview

The VLM CLI now includes **automatic memory management** to ensure accurate system resource readings and prevent memory bloat when switching between models.

## Features

### 1. Startup Cleanup ✅
- **Automatically unloads all models** when the application starts
- Clears GPU/CPU cache (CUDA, MPS for Apple Silicon)
- Provides **accurate system specs** with clean memory state
- Shows available RAM before any models are loaded

### 2. Model Switching ✅
- **Automatically unloads previous model** before loading a new one
- Frees memory from:
  - HuggingFace models (PyTorch tensors, processors, tokenizers)
  - GPU/CPU cache
- Displays unload status: `Unloading <model_name>...`

### 3. Provider Switching ✅
- Unloads current model when switching between Ollama and HuggingFace
- Ensures clean state before loading models from a different provider

### 4. Clean Exit ✅
- Unloads model before exiting the application
- Proper cleanup of resources

## How It Works

### Startup Process
```
1. Initialize VLM Client
2. Run cleanup_all_models()
   - Unload any loaded models
   - Clear GPU/CPU cache
   - Force garbage collection
3. Fetch system specs (clean state)
4. Display specs with accurate RAM availability
```

### Model Loading Process
```
1. User selects a model
2. Check if previous model is loaded
3. If yes:
   - Unload previous model
   - Free GPU/CPU memory
   - Clear cache
4. Load new model
5. Update current model info
```

### Memory Freed

#### HuggingFace Models
- **Full cleanup**: Model, processor, tokenizer removed from memory
- GPU/CPU cache cleared
- Garbage collection triggered
- **Significant memory freed** (especially for large models)

#### Ollama Models
- Client reference cleared
- Note: Ollama models are managed by the Ollama server
- Server may keep models in memory for performance
- Use `ollama stop` to fully unload Ollama models if needed

## System Specs Display

The system specs now show:
- **"Clean State"** in the title
- Available RAM after cleanup
- Note about automatic unloading when switching models

Example:
```
╭─ System Specifications (Clean State) ─╮
│ Platform: Darwin arm64                 │
│ CPU: 8 cores (8 threads)              │
│ RAM: 8.0GB total, 2.1GB available     │
│ GPU: Apple Silicon (MPS)              │
│ Recommended Model Size: ≤ 3.0GB       │
│                                        │
│ Note: Models are unloaded when        │
│       switching to free memory        │
╰────────────────────────────────────────╯
```

## API Methods

### `cleanup_all_models()`
- Unloads all loaded models
- Clears GPU/CPU cache
- Triggers garbage collection
- Called automatically at startup

### `unload_model()`
- Unloads currently loaded model
- Frees memory (HuggingFace models)
- Clears cache (CUDA, MPS)
- Called automatically when switching models

## Benefits

1. **Accurate Resource Monitoring**
   - System specs show true available memory
   - Compatibility checks based on clean state

2. **Prevent Memory Bloat**
   - No accumulation of loaded models
   - One model in memory at a time

3. **Better Performance**
   - More memory available for active model
   - Faster model loading

4. **Transparent to User**
   - Automatic cleanup
   - Informative status messages
   - No manual intervention needed

## Examples

### Switching from HuggingFace to Ollama
```
Unloading microsoft/Florence-2-base...
✓ Memory freed
Loading qwen2.5vl:3b...
✓ Model loaded successfully!
```

### Switching Between Ollama Models
```
Unloading llava:latest...
Loading qwen2.5vl:3b...
✓ Model loaded successfully!
```

### Startup Cleanup
```
Initializing VLM Client... ✓
Cleaning up memory... ✓

╭─ System Specifications (Clean State) ─╮
│ RAM: 8.0GB total, 5.2GB available     │
│ Recommended Model Size: ≤ 3.0GB       │
╰────────────────────────────────────────╯
```

## Technical Details

### Cleanup Implementation

**HuggingFace Models:**
```python
# Delete model components
del model_instance['model']
del model_instance['processor']
del model_instance['tokenizer']

# Clear cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
    torch.mps.synchronize()

# Garbage collection
gc.collect()
```

**Ollama Models:**
- Client reference cleared
- Models remain in Ollama server memory
- Server manages its own memory

## Notes

- **Ollama Server**: Models may remain in Ollama's memory even after unloading from the client. This is by design for faster re-loading.
- **HuggingFace**: Full cleanup of Python memory and GPU/CPU cache.
- **Apple Silicon (MPS)**: Properly handles MPS cache clearing.
- **CUDA**: Properly handles CUDA cache clearing and synchronization.

## Recommendations

1. **For Maximum Free Memory**:
   - Use HuggingFace models if you need frequent model switching with memory release
   - For Ollama, restart the Ollama service if you need to free server memory

2. **For Best Performance**:
   - Ollama is faster for repeated inference (server-side caching)
   - HuggingFace gives you full control over memory

3. **System Constraints**:
   - The recommended model size is calculated based on clean state
   - Leave 2-3GB for system operations
   - Monitor compatibility icons (✅⚠️💾❌) in model selection
