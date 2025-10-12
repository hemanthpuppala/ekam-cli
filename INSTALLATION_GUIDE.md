# VLM CLI - Model Installation Guide

## New Features

The VLM CLI now includes **dynamic model discovery** and **installation** capabilities!

### What's New

1. **Automatic Model Discovery**
   - Scans for installed Ollama vision models
   - Detects HuggingFace models in cache
   - Shows both configured and discovered models

2. **Install New Models**
   - Download Ollama models with progress tracking
   - Download HuggingFace models
   - Install from within the CLI

3. **Dynamic Model Loading**
   - Load any discovered model
   - No need to edit config files
   - Immediate availability after installation

## Usage

### Viewing Available Models

When you select **Option 6: Switch Model**, you'll see:

```
Available Models:
┌───┬─────────────────────┬────────────┬────────┬────────────────┐
│ # │ Model               │ Provider   │ Size   │ Status         │
├───┼─────────────────────┼────────────┼────────┼────────────────┤
│ 1 │ qwen2.5vl:3b        │ ollama     │ 3.2GB  │ ✓ Installed    │
│ 2 │ llava:latest        │ ollama     │ 4.7GB  │ ✓ Installed    │
│ 3 │ moondream:latest    │ ollama     │ 1.7GB  │ ✓ Installed    │
└───┴─────────────────────┴────────────┴────────┴────────────────┘
```

### Installing New Models

Select **Option 7: Install Model** from the main menu.

#### Install Ollama Model

1. Choose provider: `ollama`
2. Enter model name (examples):
   - `llama3.2-vision:11b`
   - `qwen2-vl:7b`
   - `llava:13b`
   - `bakllava:latest`

3. Confirm installation
4. Watch real-time download progress:
   ```
   Pulling model: llama3.2-vision:11b
   pulling manifest: 100.0% (10.5MB / 10.5MB)
   pulling model: 45.2% (2.3GB / 5.1GB)
   ```

5. Option to load immediately after installation

#### Install HuggingFace Model

1. Choose provider: `huggingface`
2. Enter model name (examples):
   - `microsoft/Florence-2-base`
   - `Salesforce/blip2-opt-2.7b`
   - `google/paligemma-3b-pt-224`

3. Confirm installation
4. Wait for download (shows progress indicators)
5. Model cached in `~/.cache/huggingface`

## Examples

### Example 1: Install and Use Llama Vision

```bash
$ python3 vlm_cli.py

Select an option: 7

═══ Install New Model ═══

Select provider [ollama/huggingface] (ollama): ollama

Example: llama3.2-vision:11b, qwen2-vl:7b, llava:13b
Enter Ollama model name: llama3.2-vision:11b

Install llama3.2-vision:11b from ollama? [Y/n]: y

Installing llama3.2-vision:11b...
Pulling model: llama3.2-vision:11b
pulling manifest: 100.0% (925MB / 925MB)
pulling 8eeb52dfaeac: 100.0% (6.7GB / 6.7GB)
verifying sha256 digest
writing manifest
success

✓ llama3.2-vision:11b installed successfully!

Load this model now? [Y/n]: y
Loading llama3.2-vision:11b (ollama)...
✓ Model loaded successfully!
✓ Model loaded!
```

### Example 2: Install Florence-2

```bash
Select an option: 7

═══ Install New Model ═══

Select provider [ollama/huggingface] (ollama): huggingface

Example: microsoft/Florence-2-base, Salesforce/blip2-opt-2.7b
Enter HuggingFace model name: microsoft/Florence-2-base

Install microsoft/Florence-2-base from huggingface? [Y/n]: y

Installing microsoft/Florence-2-base...
Downloading model: microsoft/Florence-2-base
(This may take a while...)
✓ Model downloaded successfully!

✓ microsoft/Florence-2-base installed successfully!

Load this model now? [Y/n]: n
```

## Recommended Models

### For 8GB RAM (Apple M3)

**Ollama:**
- `qwen2.5vl:3b` (3.2GB) - Excellent quality, fast
- `moondream:latest` (1.7GB) - Lightweight, fast
- `llava:7b` (4.7GB) - Good balance
- `qwen2-vl:7b` (4.4GB) - Better accuracy

**HuggingFace:**
- `microsoft/Florence-2-base` (230MB) - Ultra lightweight
- `Salesforce/blip2-opt-2.7b` (2.7GB) - Good quality

### For More RAM (>16GB)

**Ollama:**
- `llama3.2-vision:11b` (7GB) - High quality
- `llava:13b` (8GB) - Comprehensive
- `qwen2-vl:32b` (20GB) - Best quality

## Technical Details

### Discovery Process

The CLI automatically scans:

1. **Ollama Models**: Queries Ollama API for installed models
   - Filters for vision-capable models (keywords: vision, vl, llava, moondream)
   - Shows size, modification date

2. **HuggingFace Models**: Scans cache directory
   - Location: `~/.cache/huggingface/hub/`
   - Reads model snapshots
   - Calculates disk usage

### Installation Process

**Ollama:**
- Uses `ollama pull` with streaming
- Shows real-time progress (percentage, MB downloaded)
- Automatic resume if interrupted

**HuggingFace:**
- Downloads via transformers library
- Uses HuggingFace Hub
- Automatic caching
- Progress shown in terminal

### Model Storage

**Ollama:**
- Location: `~/.ollama/models/`
- Binary format (GGUF)
- Managed by Ollama service

**HuggingFace:**
- Location: `~/.cache/huggingface/hub/`
- PyTorch format (.safetensors)
- Shared across projects

## Troubleshooting

### "Failed to install model"

**Ollama:**
- Check Ollama is running: `ollama list`
- Verify model exists: Search on https://ollama.com/library
- Check network connection

**HuggingFace:**
- Verify model name is correct
- Check disk space
- Ensure internet connection

### "Model not showing in list"

- Restart the CLI
- For Ollama: Run `ollama list` to verify
- For HF: Check `~/.cache/huggingface/hub/`

### Progress not showing

- Normal for HuggingFace (downloads in background)
- Ollama shows detailed progress
- Check terminal for any error messages

## Tips

1. **Space Management**: Each model takes several GB
2. **Network**: Use stable connection for large models
3. **Patience**: Large models (>5GB) take 10-30 min
4. **Testing**: Start with smaller models first
5. **Cleanup**: Remove unused models to save space
   - Ollama: `ollama rm <model>`
   - HF: Manually delete from cache dir

## Advanced

### Custom Model Names

For Ollama, you can install any model from the library:
```bash
ollama pull <username>/<model>:<tag>
```

Then use it in the CLI!

### Model Quantization

Ollama models support quantization:
- `modelname:q4` - 4-bit quantization (smallest)
- `modelname:q5` - 5-bit
- `modelname:q8` - 8-bit (best quality)

Example: `llama3.2-vision:11b-q4`

## Summary

The VLM CLI now makes it easy to:
- ✅ Discover installed models automatically
- ✅ Install new models with one command
- ✅ Track download progress
- ✅ Load models immediately
- ✅ Switch between providers seamlessly

No more manual configuration needed!
