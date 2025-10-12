# Quickstart Guide

**Feature**: High-Performance Multi-Provider VLM/LLM CLI with Resource Management
**Branch**: `001-build-an-cli`
**Target Users**: Developers, researchers, data scientists testing VLM/LLM models locally

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 4GB | 8GB+ |
| **Disk Space** | 10GB free | 50GB+ (for model storage) |
| **Python** | 3.8+ | 3.10+ |
| **OS** | Linux, macOS, Windows | Linux or macOS |
| **GPU** (optional) | CUDA 11.8+ or Metal | CUDA 12+ or M1/M2/M3 |

### Supported Platforms

✅ **Edge Devices**: Raspberry Pi 4 (4GB+), Jetson Nano/Orin
✅ **Laptops**: MacBook Air/Pro M1+, x86_64 laptops with 8GB+ RAM
✅ **Desktops**: Any x86_64 or ARM64 with 8GB+ RAM
✅ **Servers**: Any Linux server with 16GB+ RAM

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/vlm-cli.git
cd vlm-cli
git checkout 001-build-an-cli
```

### Step 2: Run Launcher (Automated Setup)

The `run.sh` script performs **automatic detection and installation assistance** for all providers (per FR-010, FR-011):

```bash
chmod +x run.sh
./run.sh
```

**What the launcher does**:
1. Checks Python version (requires 3.8+)
2. Creates virtual environment if missing
3. Detects installed providers (Ollama, LM Studio, HuggingFace, GGUF support)
4. **Offers to install missing components** with user confirmation:
   - Ollama → Link to https://ollama.ai
   - LM Studio → Link to https://lmstudio.ai
   - Python packages (transformers, llama-cpp-python) → Auto-install via pip
5. Launches CLI after validation

**Example interactive session**:
```
🔍 Checking prerequisites...

✅ Python 3.10.8 detected
✅ Virtual environment ready
✅ Ollama detected at http://localhost:11434
❌ LM Studio not detected

Would you like to download LM Studio? [y/N]: n
Skipping LM Studio installation.

❌ llama-cpp-python not installed

Install llama-cpp-python for GGUF support? [y/N]: y
Installing llama-cpp-python...
✅ llama-cpp-python installed successfully

✅ HuggingFace Transformers detected

═══════════════════════════════════════════════════
Provider Status:
  • Ollama: ✅ Ready
  • HuggingFace: ✅ Ready
  • LM Studio: ❌ Disabled
  • GGUF: ✅ Ready
═══════════════════════════════════════════════════

Press [Enter] to launch CLI...
```

### Step 3: Manual Installation (Alternative)

If you prefer manual setup without the launcher:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install base dependencies
pip install -r requirements.txt

# Install provider-specific dependencies

# Ollama: No Python packages needed (REST API only)
# Download from https://ollama.ai and start service:
ollama serve

# LM Studio: No Python packages needed (REST API only)
# Download from https://lmstudio.ai

# HuggingFace: Already in requirements.txt

# GGUF support: Install llama-cpp-python
# CPU-only (Raspberry Pi, minimal)
pip install llama-cpp-python

# With NVIDIA GPU (Jetson, desktop)
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# With Apple Silicon GPU (MacBook)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

---

## Configuration

### Edit config.yaml

The CLI is **configuration-driven** (per CL-002). Edit `config.yaml` before first run:

```yaml
providers:
  ollama:
    enabled: true
    host: "http://localhost:11434"  # Change if Ollama runs on different port

  huggingface:
    enabled: true
    cache_dir: "~/.cache/huggingface/hub/"  # Change if using custom cache
    device_preference: ["cuda", "mps", "cpu"]  # Priority order for GPU selection

  lm_studio:
    enabled: false  # Set to true if you have LM Studio installed
    host: "http://localhost:1234"

  gguf:
    enabled: true
    models_dir: "~/models/gguf/"  # Directory for GGUF files
    device_preference: ["cuda", "mps", "cpu"]

system:
  max_image_dimension: 1920  # Auto-resize larger images
  memory_safety_margin_gb: 1.0  # Reserve RAM for OS
  default_timeout_seconds: 120
```

**Key Configuration Notes**:
- **Ollama**: Requires `ollama serve` running before CLI launch
- **LM Studio**: Requires LM Studio app running with local server enabled
- **HuggingFace**: Automatically uses `~/.cache/huggingface/` if exists
- **GGUF**: Create `~/models/gguf/` directory and place `.gguf` files inside

---

## First Run: Complete Workflow

### 1. Launch CLI

```bash
./run.sh
# Or manually: python3 -m src.core.app
```

### 2. View System Specifications

On startup, the CLI displays detected hardware (per FR-012):

```
═══════════════════════════════════════════════════════════
               VLM/LLM CLI - System Information
═══════════════════════════════════════════════════════════

Platform: Darwin (arm64)
Device Category: LAPTOP

CPU:
  Physical Cores: 8
  Logical Cores: 8

Memory:
  Total RAM: 16.00 GB
  Available RAM: 12.30 GB
  Recommended Model Size: ≤2.00 GB

GPU:
  Apple Silicon GPU (mps)
  Estimated Memory: 12.00 GB

═══════════════════════════════════════════════════════════
```

**Interpreting Results**:
- **Recommended Model Size**: Maximum model size without memory pressure (formula: `min(2.0, max(0.5, total_ram * 0.1))`)
- **Compatibility Icons** in model list:
  - ✅ **PERFECT_FIT**: <70% of recommended size (safe)
  - ⚠️ **TIGHT_FIT**: 70-100% of recommended size (may impact performance)
  - ❌ **TOO_LARGE**: >100% of recommended size (requires confirmation to load)

### 3. Select Provider

```
═══════════════════════════════════════════════════════════
Main Menu
═══════════════════════════════════════════════════════════

Available Providers:
  1. Ollama (✅ 10 models available)
  2. HuggingFace Transformers (✅ 7 models cached)
  3. LM Studio (❌ Not running)
  4. GGUF (✅ 3 models found)
  5. [Back] Return to system info
  6. [Exit] Quit application

Select provider [1-6]:
```

**Enter**: `1` (for Ollama example)

### 4. Select Model

```
═══════════════════════════════════════════════════════════
Models - Ollama Provider
═══════════════════════════════════════════════════════════

Discovered 10 models (sorted by compatibility):

┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━┓
┃ # ┃ Model Name            ┃ Type   ┃ Size         ┃ Status ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━┩
│ 1   │ ✅ llama3.2:3b         │ LLM    │ 1.7 GB       │ PERFECT │
│ 2   │ ✅ moondream:latest    │ VLM    │ 1.8 GB       │ PERFECT │
│ 3   │ ⚠️ llava:7b            │ VLM    │ 4.1 GB       │ TIGHT   │
│ 4   │ ❌ llama3.1:70b        │ LLM    │ 39.0 GB      │ TOO_LARGE │
└─────┴───────────────────────┴────────┴──────────────┴────────┘

Actions:
  [1-10] Select model
  [i] Install new model
  [r] Refresh model list
  [b] Back to provider menu

Select action:
```

**Enter**: `2` (select moondream VLM)

### 5. Model Loading

```
Loading model moondream:latest...

Device: mps (Apple Silicon GPU)
Estimated Memory: 2.2 GB (includes 20% safety margin)
Available RAM: 12.30 GB → 10.10 GB after loading

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 100%

✅ Model loaded successfully in 3.2 seconds
```

### 6. Select Endpoint

The CLI dynamically shows only **supported endpoints** for the loaded model (per FR-034, CL-011):

```
═══════════════════════════════════════════════════════════
Endpoint Menu - moondream:latest (VLM)
═══════════════════════════════════════════════════════════

Available Endpoints:
  1. 🖼️ Question Answering (QA)
  2. 📝 Caption
  3. 🔍 Detect
  4. 📍 Point
  5. [Stats] View session statistics
  6. [Switch] Change model
  7. [Back] Return to model selection
  8. [Exit] Quit application

Select endpoint [1-8]:
```

**Note**: If an LLM was selected, only `Text` endpoint would show (per FR-027).

**Enter**: `1` (QA endpoint)

### 7. Run Inference (QA Example)

```
════════════════════════════════════════════════════════════
Question Answering (QA)
════════════════════════════════════════════════════════════

Enter image path: /path/to/your/image.jpg
✅ Image validated (1920x1080, RGB)

Enter question: What objects are visible in this image?

Processing...

▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ Inference complete

════════════════════════════════════════════════════════════
Result
════════════════════════════════════════════════════════════

Answer: The image contains a red car parked on a street, with
a stop sign visible in the background. There are also trees
and a building visible in the scene.

⏱️ Inference Time: 847ms

════════════════════════════════════════════════════════════

Save result to JSON? [y/N]:
```

**Enter**: `y` (save result)

```
✅ Result saved to: results/20251011_143052_001234.json
```

### 8. Continue or Switch

```
Continue with same model? [Y/n]:
```

**Options**:
- Press **Enter** or `y`: Return to endpoint menu for another inference
- Enter `n`: Return to model selection menu

---

## Example Workflows

### Workflow 1: Test Multiple Endpoints on Same Image

**Goal**: Run QA, Caption, Detect, Point on `test_image.jpg` with single VLM

**Steps**:
1. Launch CLI → Select Ollama → Select `llava:7b` VLM
2. Select **QA** → Image: `test_image.jpg` → Question: "What is this?"
3. After result, press **Enter** (continue with same model)
4. Select **Caption** → Image: `test_image.jpg` → Detail: "detailed"
5. After result, press **Enter**
6. Select **Detect** → Image: `test_image.jpg` → Object: "car"
7. After result, press **Enter**
8. Select **Point** → Image: `test_image.jpg` → Object: "stop sign"
9. View saved annotated images in `results/annotated_images/`

**Expected Output**:
- 4 JSON result files with timestamps
- 2 annotated images (Detect and Point endpoints save images per FR-040)
- Session statistics showing 4 inferences

### Workflow 2: Compare Two Models on Same Task

**Goal**: Compare moondream vs llava caption quality

**Steps**:
1. Launch CLI → Select Ollama → Select `moondream:latest`
2. Select **Caption** → Image: `test.jpg` → Detail: "detailed"
3. Note inference time and caption quality
4. After result, enter `n` (switch model)
5. Select `llava:7b` from model list
6. Select **Caption** → Image: `test.jpg` → Detail: "detailed"
7. Note inference time and caption quality
8. Select **Stats** from endpoint menu to compare metrics

**Expected Output**:
```
═══════════════════════════════════════════════════════════
Session Statistics
═══════════════════════════════════════════════════════════

Session Duration: 2.5 minutes
Current Model: llava:7b
Current Provider: ollama

Total Inferences: 2
Total Time: 1.83 seconds
Average Time: 915ms per inference

Breakdown:
  caption: 2

═══════════════════════════════════════════════════════════
```

### Workflow 3: Install and Test New Model

**Goal**: Download llama3.2-vision from Ollama and test

**Steps**:
1. Launch CLI → Select Ollama → Select **[i] Install new model**
2. Enter model name: `llama3.2-vision:11b`
3. Review compatibility assessment:
   ```
   Model: llama3.2-vision:11b
   Estimated Size: 6.7 GB
   Compatibility: ⚠️ TIGHT_FIT (uses 85% of recommended size)

   Warning: This model may use significant RAM. Performance may be affected.

   Continue installation? [y/N]:
   ```
4. Enter `y` (confirm installation)
5. Monitor download progress:
   ```
   Downloading llama3.2-vision:11b...
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░ 52% (3.5GB / 6.7GB)
   ```
6. After download completes, model auto-loads (per FR-004 acceptance scenario 4)
7. Test with QA or Caption endpoint

---

## Annotated Images & Clickable Paths

For **Detect** and **Point** endpoints, the CLI saves annotated images with bounding boxes/coordinates drawn on the original image (per CL-009):

```
════════════════════════════════════════════════════════════
Detect Result
════════════════════════════════════════════════════════════

Detections:
  • car (confidence: 0.94) - bbox: [120, 340, 580, 720]
  • car (confidence: 0.89) - bbox: [650, 380, 1020, 690]

🖼️ Annotated Image: file:///path/to/results/annotated_images/20251011_143052_detect_car.jpg

⏱️ Inference Time: 1240ms

════════════════════════════════════════════════════════════
```

**Clickable Paths**: If your terminal supports OSC 8 hyperlinks (iTerm2, WezTerm, Windows Terminal), Cmd+Click (macOS) or Ctrl+Click (Linux/Windows) on the file path to open the annotated image directly.

---

## Managing Models

### View Installed Models

From any provider's model selection menu, all discovered models are listed with:
- **Compatibility status** (✅/⚠️/❌)
- **Type** (VLM/LLM/Embedding)
- **Size** (GB)
- **Installation status** (installed vs available-to-download)

### Install New Models

**Ollama**:
```
Select action: i
Enter model name (e.g., 'llama3.2:3b'): llama3.2:3b
```

**HuggingFace**:
```
Select action: i
Enter model ID (e.g., 'llava-hf/llava-1.5-7b-hf'): llava-hf/llava-1.5-7b-hf
```

**GGUF**:
```
Select action: i
Enter HuggingFace repo (e.g., 'TheBloke/Llama-2-7B-GGUF'): TheBloke/Llama-2-7B-GGUF
Select quantization (Q4_K_M, Q8_0, etc.): Q4_K_M
```

### Delete Models (Future Feature)

Not yet implemented in Phase 1, but spec includes this (FR-008, User Story 5). Will support provider-specific deletion commands.

---

## Troubleshooting

### Issue: "Ollama not accessible"

**Symptoms**:
```
❌ Ollama provider unavailable
Error: Connection refused at http://localhost:11434
```

**Solutions**:
1. Check if Ollama is running: `ps aux | grep ollama`
2. Start Ollama: `ollama serve`
3. Verify port: `curl http://localhost:11434/api/tags`
4. If using custom port, update `config.yaml`:
   ```yaml
   providers:
     ollama:
       host: "http://localhost:PORT"
   ```

### Issue: "Model too large" error

**Symptoms**:
```
❌ Model llama3.1:70b exceeds recommended size

Model Size: 39.0 GB
Available RAM: 12.30 GB
Recommended Size: ≤2.00 GB

This model may cause out-of-memory errors. Suggested alternatives:
  • llama3.2:3b (1.7 GB) - Similar architecture, smaller size
  • mistral:7b (4.1 GB) - High-quality text generation

Proceed anyway? [y/N]:
```

**Solutions**:
1. **Advisory Warning** (per CL-008): Enter `y` to proceed if you have swap space or want to test anyway
2. **Select smaller model**: Enter `n` and choose a model with ✅ or ⚠️ status
3. **Use quantized GGUF**: Download Q4 or Q8 quantized version (50-70% size reduction)
4. **Close other applications**: Free up RAM, then refresh model list with `[r]`

### Issue: "Image not found" error

**Symptoms**:
```
❌ Image validation failed
Error: /path/to/image.jpg does not exist
```

**Solutions**:
1. Verify path with absolute path: `/Users/username/images/test.jpg`
2. Use tab completion if terminal supports it
3. Check file permissions: `ls -l /path/to/image.jpg`
4. Ensure file is a valid image format (JPEG, PNG, WebP, etc.)

### Issue: Slow inference on CPU

**Symptoms**:
```
⏱️ Inference Time: 47,320ms (47 seconds)
```

**Solutions**:
1. **Enable GPU**: Install GPU-specific dependencies (see Installation step 3)
2. **Use smaller model**: Switch to 3B or 7B parameter models instead of 13B+
3. **Use quantized GGUF**: Q4_K_M quantization offers 3-5x speedup with minimal quality loss
4. **Reduce image resolution**: Edit `config.yaml`:
   ```yaml
   system:
     max_image_dimension: 1024  # Reduce from default 1920
   ```

### Issue: "llama-cpp-python build failed"

**Symptoms**:
```
ERROR: Failed building wheel for llama-cpp-python
```

**Solutions**:

**On Raspberry Pi / ARM Linux**:
```bash
# Install build dependencies
sudo apt-get install build-essential cmake

# Install with pre-built wheel (if available)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

**On macOS**:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install with Metal support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
```

**On NVIDIA Jetson**:
```bash
# Install CUDA toolkit first
sudo apt-get install nvidia-cuda-toolkit

# Install with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --no-cache-dir
```

---

## Performance Optimization

### For Edge Devices (Raspberry Pi, Jetson Nano)

**Configuration**:
```yaml
providers:
  gguf:
    enabled: true
    device_preference: ["cpu"]  # Force CPU if GPU memory limited

system:
  max_image_dimension: 1024  # Reduce memory usage
  memory_safety_margin_gb: 0.5  # Tighten margin on constrained devices
```

**Recommended Models**:
- **moondream:latest** (Ollama) - 1.8GB, optimized for vision
- **TinyLlama GGUF Q4** - 637MB, basic text generation
- **LLaVA 7B GGUF Q4** - 4.2GB VLM (if 8GB RAM available)

### For Laptops (8-16GB RAM)

**Recommended Models**:
- **llama3.2:3b** (Ollama) - 1.7GB, fast text generation
- **llava:7b** (Ollama) - 4.1GB VLM, good balance
- **moondream** (HuggingFace) - 1.8GB, specialized vision tasks

### For Desktops (16GB+ RAM, GPU)

**Configuration**:
```yaml
providers:
  huggingface:
    device_preference: ["cuda", "mps", "cpu"]  # Prioritize GPU

  gguf:
    device_preference: ["cuda", "cpu"]
```

**Recommended Models**:
- **llava-hf/llava-1.5-13b-hf** (HuggingFace) - 13GB VLM, high quality
- **llama3.1:70b** (Ollama, if 32GB+ RAM) - 39GB, best quality
- **LLaVA 13B GGUF Q8** - 13.8GB, near-FP16 quality with quantization

---

## Next Steps

After completing this quickstart, you can:

1. **Explore all endpoints**: Test QA, Caption, Detect, Point on various images
2. **Compare models**: Use session statistics to benchmark different VLMs
3. **Contribute providers**: Implement new provider integrations (see `contracts/provider.py`)
4. **Customize config**: Tune performance settings for your hardware
5. **Read architecture docs**: See `specs/001-build-an-cli/research.md` for technology decisions

---

## Getting Help

- **Logs**: Check `logs/vlm_cli_YYYYMMDD_HHMMSS.log` for detailed error traces
- **Issues**: Report bugs at https://github.com/your-org/vlm-cli/issues
- **Discussions**: Ask questions at https://github.com/your-org/vlm-cli/discussions
- **Constitution**: Review project principles at `.specify/memory/constitution.md`

**Quickstart Guide Complete** ✅
