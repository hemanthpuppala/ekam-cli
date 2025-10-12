# VLM CLI - Vision Language Model Testing Tool

Production-ready command-line interface for testing Vision Language Models with multiple endpoints (QA, Caption, Detect, Point).

## Features

- **4 Core Endpoints**: Question Answering, Image Captioning, Object Detection, and Object Pointing
- **Multi-Provider Support**: Works with both Ollama and HuggingFace models
- **Interactive CLI**: Beautiful terminal interface with Rich library
- **Flexible Configuration**: Easy model switching via YAML config
- **Production Ready**: Error handling, logging, result saving
- **Moondream Optimized**: Special support for Moondream2's native methods
- **Extensible**: Easy to add new models and endpoints

## Quick Start

### 1. Installation

```bash
cd vlm-tester
pip install -r requirements.txt
```

### 2. Configure Models

Edit `config.yaml` to customize:
- Available models (Ollama & HuggingFace)
- Default model
- Inference parameters
- Endpoint templates

### 3. Run the CLI

```bash
python3 vlm_cli.py
```

Or make it executable:
```bash
chmod +x vlm_cli.py
./vlm_cli.py
```

## Usage

### Main Menu

Upon starting, you'll see:

```
╔══════════════════════════════════════════════════════════╗
║           VLM CLI - Vision Language Model Tester         ║
║                   Production-Ready v1.0                  ║
╚══════════════════════════════════════════════════════════╝

Main Menu
┌────────┬────────────────┬──────────────────────────────────────┐
│ Option │ Endpoint       │ Description                          │
├────────┼────────────────┼──────────────────────────────────────┤
│   1    │ QA             │ Question Answering - Ask questions   │
│   2    │ Caption        │ Image Captioning - Generate desc.    │
│   3    │ Detect         │ Object Detection - Locate objects    │
│   4    │ Point          │ Object Pointing - Get coordinates    │
│   5    │ Stats          │ View statistics and model info       │
│   6    │ Switch Model   │ Change the current VLM model         │
│   0    │ Exit           │ Quit the application                 │
└────────┴────────────────┴──────────────────────────────────────┘
```

### Endpoints

#### 1. Question Answering (QA)

Ask natural language questions about images:

```
Select: 1
Image path: /path/to/image.jpg
Question: What is in this image?
```

**Example Questions:**
- "What objects can you see in this image?"
- "What is the weather like?"
- "How many people are in the photo?"
- "What color is the car?"

#### 2. Image Captioning

Generate descriptions of images:

```
Select: 2
Image path: /path/to/image.jpg
Detail level: detailed (or short)
```

**Output:**
- **Detailed**: Full description with context
- **Short**: One-sentence summary

#### 3. Object Detection

Detect and locate specific objects:

```
Select: 3
Image path: /path/to/image.jpg
Object to detect: car
```

**Use Cases:**
- Count objects in images
- Verify presence of items
- Get approximate locations

#### 4. Object Pointing

Get precise coordinates for objects:

```
Select: 4
Image path: /path/to/image.jpg
Object to locate: stop sign
```

**Returns:** Coordinates or bounding box information (model-dependent)

### Model Management

#### View Available Models

Select option `6` to see all configured models:

```
Available Models:
┌───┬─────────────┬────────────┬──────┬─────────────────────────┐
│ # │ Model ID    │ Provider   │ Size │ Capabilities            │
├───┼─────────────┼────────────┼──────┼─────────────────────────┤
│ 1 │ moondream2  │ huggingface│ 1.6GB│ qa, caption, detect, pt │
│ 2 │ qwen2vl-7b  │ ollama     │ 4.4GB│ qa, caption, detect     │
│ 3 │ llama-vision│ ollama     │ 7.0GB│ qa, caption, detect     │
└───┴─────────────┴────────────┴──────┴─────────────────────────┘
```

#### Switch Models

1. Select option `6`
2. Choose model number
3. Wait for loading to complete

### Statistics

View usage statistics with option `5`:

```
Statistics
┌─────────────────────┬──────────────────────────────┐
│ Metric              │ Value                        │
├─────────────────────┼──────────────────────────────┤
│ Current Model       │ moondream2                   │
│ Provider            │ huggingface                  │
│ Total Inferences    │ 15                           │
│ Total Time          │ 23.4s                        │
│ Average Time        │ 1560ms                       │
└─────────────────────┴──────────────────────────────┘
```

## Configuration

### config.yaml Structure

```yaml
# Default model to load on startup
default_model: qwen2vl-3b

# Model definitions
models:
  qwen2vl-3b:
    provider: ollama
    name: "qwen2.5vl:3b"
    size_gb: 3.2
    capabilities: [qa, caption, detect]

# Inference parameters
inference:
  temperature: 0.3
  top_p: 0.9
  top_k: 40
  max_tokens: 500
  seed: 42

# Output settings
output:
  save_results: true
  results_dir: "./results"
  show_timing: true
```

### Adding New Models

To add a new model:

1. Open `config.yaml`
2. Add model definition under `models:`

```yaml
my-custom-model:
  provider: ollama  # or huggingface
  name: "model-name:tag"
  size_gb: 3.5
  capabilities: [qa, caption]
  description: "My custom VLM"
```

3. Run CLI and select your new model

## Providers

### Ollama

**Setup:**
1. Install Ollama: https://ollama.ai
2. Pull a vision model: `ollama pull llama3.2-vision`
3. Configure in `config.yaml`

**Supported Models:**
- qwen2.5vl:3b (lightweight, 2GB)
- qwen2-vl:7b (balanced, 4.4GB)
- llama3.2-vision:11b (powerful, 7GB)
- llava:13b (comprehensive, 8GB)

### HuggingFace

**Setup:**
1. Models auto-download on first use
2. Cached in `~/.cache/huggingface`

**Supported Models:**
- microsoft/Florence-2-base
- Salesforce/blip2-opt-2.7b

**Note:** Moondream2 currently has tokenizer compatibility issues and is disabled.

## System Requirements

### Minimum
- **RAM**: 4GB
- **Storage**: 2GB for models
- **Python**: 3.8+

### Recommended
- **RAM**: 8GB
- **GPU**: NVIDIA CUDA or Apple Silicon
- **Storage**: 10GB for multiple models

### For Your System (Apple M3, 8GB)

**Best Models:**
1. **qwen2.5vl:3b** (3.2GB) - Recommended, works perfectly
2. **qwen2-vl:7b** (4.4GB) - Better accuracy
3. **llava:latest** (4.7GB) - Comprehensive features
4. **florence-2** (0.2GB) - Ultra lightweight (HuggingFace)

## Results & Output

### Automatic Saving

Results can be saved to JSON:

```
results/
├── qa_20241011_143022.json
├── caption_20241011_143145.json
└── detect_20241011_143301.json
```

### Result Format

```json
{
  "endpoint": "qa",
  "question": "What is in this image?",
  "answer": "A red car parked on a street...",
  "inference_time_ms": 1234.56,
  "model": "moondream2",
  "provider": "huggingface",
  "metadata": {
    "gpu_memory_used_mb": 1456.78
  }
}
```

## Troubleshooting

### Model Not Loading

**Ollama models:**
```bash
# Pull the model first
ollama pull qwen2.5vl:3b

# Check Ollama is running
ollama list
```

**HuggingFace models:**
- First run downloads the model
- Check internet connection
- Verify disk space in `~/.cache/huggingface`

### Memory Errors

If you encounter OOM (Out of Memory):

1. Use smaller models (moondream2, florence-2)
2. Close other applications
3. Reduce `max_tokens` in config
4. For HuggingFace, enable 8-bit quantization

### Import Errors

```bash
# Install missing dependencies
pip install -r requirements.txt

# For GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Development

### Project Structure

```
vlm-tester/
├── vlm_cli.py          # Main CLI interface
├── vlm_client.py       # VLM client library
├── config.yaml         # Configuration
├── requirements.txt    # Dependencies
├── README.md           # This file
└── results/            # Saved results
```

### Extending

**Add new endpoint:**

1. Edit `vlm_client.py`, add method:
```python
async def my_endpoint(self, image_path: str, ...):
    # Implementation
    pass
```

2. Edit `vlm_cli.py`, add menu option and handler:
```python
async def run_my_endpoint(self):
    # CLI interaction
    result = await self.client.my_endpoint(...)
    self.display_result(result, "My Endpoint")
```

3. Update `config.yaml` with endpoint config

## Performance Tips

1. **Use GPU**: Significantly faster inference
2. **Batch Images**: Process multiple images in one session
3. **Adjust Parameters**: Lower `max_tokens` for speed
4. **Model Selection**: Smaller models = faster inference
5. **Cache Models**: Download once, use multiple times

## Examples

### Example Session

```bash
$ python3 vlm_cli.py

Loading default model: qwen2.5vl:3b
✓ Model loaded successfully!

Select option: 1
Image path: ~/photos/street.jpg
Question: What vehicles can you see?

╔═══════════════════════════════════════════════╗
║                  QA Result                    ║
╠═══════════════════════════════════════════════╣
║ Answer: I can see two cars - a red sedan     ║
║ and a blue SUV - along with a bicycle        ║
║ parked near the sidewalk.                     ║
║                                               ║
║ ⏱  Inference Time: 1234ms                    ║
║ 🤖 Model: qwen2.5vl:3b (ollama)              ║
╚═══════════════════════════════════════════════╝

Save result to JSON? [y/n]: y
✓ Saved to results/qa_20241011_143022.json
```

## License

MIT License - See parent project for details

## Support

- **Issues**: Report bugs in the parent repository
- **Docs**: See Claude Code documentation
- **Models**: Check Ollama/HuggingFace for model-specific help

## Credits

Built for production VLM testing with:
- Rich - Beautiful terminal UI
- Ollama - Local LLM serving
- HuggingFace - Model hub and transformers
- Moondream2 - Lightweight VLM

---

**Ready to test your VLMs! 🚀**

Run `python3 vlm_cli.py` to get started.
