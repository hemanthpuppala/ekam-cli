# HuggingFace Models Explained - Complete Guide

## 📍 Where Are HuggingFace Models Stored?

### Primary Location
```bash
~/.cache/huggingface/hub/
```

**On your system:**
```
/Users/hemanth/.cache/huggingface/hub/
```

### Your Installed Models (by size)
```
8KB     models--microsoft--Florence-2-base/          ⚠️ INCOMPLETE!
8KB     models--Salesforce--blip2-opt-2.7b/          ⚠️ INCOMPLETE!
2.7MB   models--gpt2/                                ✅ Complete (tokenizer only)
3.5MB   models--moondream--starmie-v1/               ⚠️ INCOMPLETE!
87MB    models--sentence-transformers--all-MiniLM-L6-v2/  ✅ Complete
233MB   models--t5-small/                            ✅ Complete
336MB   models--microsoft--DialoGPT-small/           ✅ Complete
339MB   models--distilgpt2/                          ✅ Complete
418MB   models--sentence-transformers--all-mpnet-base-v2/ ✅ Complete
4.9GB   models--tiiuae--falcon-rw-1b/                ✅ Complete
```

**Note:** Models marked "INCOMPLETE" only have config/tokenizer files downloaded, not the actual model weights!

---

## 🏗️ HuggingFace Storage Structure

### Git-Like Architecture

HuggingFace uses a **content-addressable storage** system similar to Git:

```
models--<organization>--<model-name>/
├── blobs/                    # Actual file content (like Git objects)
│   ├── 3a0d68f0...          # Model weights (2.4GB) - pytorch_model.bin
│   ├── 608af9d4...          # Model weights (2.4GB) - model.safetensors
│   ├── 8406492...           # config.json
│   ├── 5de8eff...           # tokenizer.json
│   └── ...
├── snapshots/                # Version snapshots (like Git commits)
│   └── e4b9872.../          # Hash of this version
│       ├── config.json -> ../../blobs/8406492...
│       ├── pytorch_model.bin -> ../../blobs/3a0d68f0...
│       ├── tokenizer.json -> ../../blobs/5de8eff...
│       └── ...
└── refs/                     # Branch references
    └── main
```

### Why This Structure?

**Benefits:**
1. **Deduplication**: Same file across models = stored once
2. **Version Control**: Can have multiple snapshots of same model
3. **Atomic Updates**: Download to blobs, then atomically create symlinks
4. **Resume Downloads**: Can resume interrupted downloads

**Example:**
```bash
# The actual model file (2.4GB)
~/.cache/huggingface/hub/models--tiiuae--falcon-rw-1b/blobs/3a0d68f0...

# Symlink pointing to it
~/.cache/huggingface/hub/models--tiiuae--falcon-rw-1b/snapshots/e4b9872.../pytorch_model.bin
-> ../../blobs/3a0d68f0...
```

---

## 📦 Model File Formats

### HuggingFace Uses These Formats:

#### 1. **PyTorch Binary (.bin)**
```bash
pytorch_model.bin      # 2.4GB - model weights
```
- Traditional PyTorch format
- Used by older models
- **Not GGUF!**

#### 2. **SafeTensors (.safetensors)**
```bash
model.safetensors      # 2.4GB - model weights
```
- Newer, safer format
- Faster loading
- Better security (prevents pickle exploits)
- **Recommended format**

#### 3. **Configuration Files**
```bash
config.json            # Model architecture config
tokenizer.json         # Tokenizer configuration
vocab.json             # Vocabulary
merges.txt             # BPE merges
```

### ❌ NOT GGUF!

**GGUF is a different format:**
- Used by: **Ollama**, llama.cpp, LM Studio
- Location: `~/.ollama/models/`
- Purpose: Optimized for CPU inference with quantization
- File extension: `.gguf`

**HuggingFace models are:**
- **PyTorch models** (.bin or .safetensors)
- Designed for GPU/Python inference
- Larger but more accurate

---

## 🔍 How to Check What's Installed

### Method 1: Using Our CLI
```bash
python3 vlm_cli.py

# Select "HuggingFace" provider
# Shows only COMPLETE models with weights
```

### Method 2: Command Line
```bash
# List all models
ls -lh ~/.cache/huggingface/hub/

# Show sizes
du -sh ~/.cache/huggingface/hub/models--*/ | sort -h

# Check if model has weights
ls -lh ~/.cache/huggingface/hub/models--tiiuae--falcon-rw-1b/snapshots/*/
```

### Method 3: Python Script
```python
from pathlib import Path

cache_dir = Path.home() / ".cache/huggingface/hub"

for model_dir in cache_dir.glob("models--*"):
    model_name = model_dir.name.replace("models--", "").replace("--", "/")

    # Check for actual model files
    has_weights = False
    for snapshot_dir in (model_dir / "snapshots").glob("*"):
        if list(snapshot_dir.glob("*.bin")) or list(snapshot_dir.glob("*.safetensors")):
            has_weights = True
            break

    size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())

    status = "✅ Complete" if has_weights else "⚠️ Incomplete"
    print(f"{model_name:40s} {size/1e9:6.2f}GB {status}")
```

---

## 🚀 How HuggingFace Models Work Locally

### Download Process

#### Step 1: Model Discovery
```python
from transformers import AutoModel

# This triggers download if not cached
model = AutoModel.from_pretrained("microsoft/Florence-2-base")
```

#### Step 2: Download Files
```
Downloading (1/8): config.json           ✓
Downloading (2/8): processor_config.json ✓
Downloading (3/8): tokenizer.json        ✓
Downloading (4/8): model.safetensors     ✓ (Largest file!)
...
```

Files are saved to `blobs/` with content-based hashes.

#### Step 3: Create Snapshot
Creates symlinks in `snapshots/<commit-hash>/` pointing to blobs.

#### Step 4: Store Reference
Updates `refs/main` to point to latest snapshot.

### Loading Process

#### Step 1: Check Cache
```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "tiiuae/falcon-rw-1b",
    cache_dir="~/.cache/huggingface"  # Default location
)
```

Checks:
1. Is model in cache?
2. Which snapshot to use? (refs/main → latest)
3. Follow symlinks to blobs

#### Step 2: Load Weights
```python
# Load model weights from .bin or .safetensors
# Into PyTorch tensors in RAM/GPU
```

#### Step 3: Ready for Inference
```python
# Model is now in memory
outputs = model(inputs)
```

---

## 🆚 HuggingFace vs Ollama Storage

| Feature | HuggingFace | Ollama |
|---------|-------------|--------|
| **Location** | `~/.cache/huggingface/hub/` | `~/.ollama/models/` |
| **Format** | PyTorch (.bin, .safetensors) | GGUF |
| **Structure** | Git-like (blobs + snapshots) | Simple blob storage |
| **Size** | Larger (full precision) | Smaller (quantized) |
| **Loading** | Python/PyTorch | Ollama server |
| **Memory** | Managed by Python | Managed by server |
| **GPU Support** | Direct PyTorch (CUDA/MPS) | Via server |
| **Quantization** | Manual | Built-in |

### Example: Same Model, Different Storage

**GPT-2 on HuggingFace:**
```
~/.cache/huggingface/hub/models--gpt2/
├── blobs/
│   └── pytorch_model.bin (548MB)
└── snapshots/...
```

**GPT-2 on Ollama (if available):**
```
~/.ollama/models/
└── blobs/
    └── sha256-xxx (GGUF, ~300MB quantized)
```

---

## 🛠️ Common Operations

### Check Disk Usage
```bash
# Total HuggingFace cache
du -sh ~/.cache/huggingface/

# Per model
du -sh ~/.cache/huggingface/hub/models--*/
```

### Delete a Model
```bash
# Manual deletion
rm -rf ~/.cache/huggingface/hub/models--microsoft--Florence-2-base/

# Or use our CLI
python3 vlm_cli.py
# Select "Delete a Model"
```

### Clear All Cache
```bash
# ⚠️ WARNING: Deletes ALL downloaded models!
rm -rf ~/.cache/huggingface/
```

### Download Specific Model
```python
from transformers import AutoModel, AutoProcessor

# Download model
model = AutoModel.from_pretrained(
    "microsoft/Florence-2-base",
    cache_dir="~/.cache/huggingface",
    trust_remote_code=True  # Required for some models
)

# Download processor
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base",
    cache_dir="~/.cache/huggingface",
    trust_remote_code=True
)
```

### Check What's Actually Downloaded
```bash
# Check for model weights
find ~/.cache/huggingface/hub/models--microsoft--Florence-2-base \
  -name "*.bin" -o -name "*.safetensors"

# If empty = only configs downloaded, not weights!
```

---

## 🔧 Your System Status

### Complete Models (Ready to Use)
```
✅ tiiuae/falcon-rw-1b                    4.9GB  (LLM)
✅ sentence-transformers/all-mpnet-base-v2  418MB (Embedding)
✅ distilgpt2                              339MB (LLM)
✅ microsoft/DialoGPT-small                336MB (LLM)
✅ t5-small                                233MB (LLM)
✅ sentence-transformers/all-MiniLM-L6-v2  87MB  (Embedding)
```

### Incomplete Downloads (Only Configs)
```
⚠️ microsoft/Florence-2-base               8KB   (VLM)
⚠️ Salesforce/blip2-opt-2.7b              8KB   (VLM)
⚠️ moondream/starmie-v1                   3.5MB (VLM)
```

**Total Used:** ~6.2GB

---

## 💡 Why Are Florence-2 and BLIP2 Incomplete?

These models only have config files downloaded but **no model weights**. This happens when:

1. **Download was interrupted**
2. **Only config was requested** (for inspection)
3. **Model architecture inspection** without full download

### How to Complete the Download

#### Option 1: Using Our CLI
```bash
python3 vlm_cli.py
# Select HuggingFace
# Select "Install New Model"
# Enter: microsoft/Florence-2-base
```

#### Option 2: Manual Python
```python
from transformers import AutoModel, AutoProcessor

# This will download the missing weights
model = AutoModel.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True
)

print("✓ Model downloaded completely!")
```

**Expected size after complete download:**
- Florence-2-base: ~230MB
- BLIP2-opt-2.7b: ~2.7GB

---

## 🎯 Key Takeaways

1. **Location**: `~/.cache/huggingface/hub/`
2. **Format**: PyTorch (.bin or .safetensors), **NOT GGUF**
3. **Structure**: Git-like (blobs + snapshots)
4. **Check Status**: Use our CLI or `du -sh` command
5. **Incomplete Models**: Only have configs, need re-download
6. **Different from Ollama**: PyTorch vs GGUF, Python vs Server

---

## 📚 Further Reading

- [HuggingFace Hub Documentation](https://huggingface.co/docs/huggingface_hub)
- [Transformers Library](https://huggingface.co/docs/transformers)
- [SafeTensors Format](https://github.com/huggingface/safetensors)
- [Cache Management](https://huggingface.co/docs/transformers/installation#cache-setup)

---

## 🆘 Troubleshooting

### "Model not found"
```bash
# Check if it's actually downloaded
ls -lh ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/*/

# Look for .bin or .safetensors files
```

### "Out of disk space"
```bash
# Check usage
du -sh ~/.cache/huggingface/

# Delete unused models
rm -rf ~/.cache/huggingface/hub/models--<name>
```

### "Model loads but gives errors"
- Check if download completed
- Look for .safetensors or .bin files
- If only configs exist, re-download

---

**Happy Model Management! 🚀**
