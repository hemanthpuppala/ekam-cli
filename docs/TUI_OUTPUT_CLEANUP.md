# TUI Output Cleanup - No More Metadata Leakage

## Problem
Unwanted metadata and library messages were appearing in the TUI during:
- Model installation/downloads
- Model loading
- Quantization operations
- Inference runs

## Root Causes
1. **HuggingFace Transformers** - Progress bars, download info, warnings
2. **PyTorch/CUDA** - Initialization messages
3. **BitsAndBytes** - Welcome messages
4. **Python Logging** - Library INFO/DEBUG messages leaking to console
5. **Environment Variables** - Not set to suppress output

## Solution Implemented

### 1. Enhanced Logging Setup (`src/core/logging_setup.py`)

**Suppresses ALL third-party library output:**

```python
# Environment variables for clean TUI
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
# ... and 7+ more
```

**Configures Python logging to silence noisy libraries:**
- transformers, huggingface_hub, torch, bitsandbytes, etc.
- All set to ERROR level (only errors shown)

### 2. Output Suppression Utilities (`src/utils/output_suppressor.py`)

**Three context managers for targeted suppression:**

#### `suppress_stdout_stderr()`
Complete stdout/stderr suppression for critical operations:
```python
with suppress_stdout_stderr():
    # Nothing prints to terminal
    dangerous_operation()
```

#### `suppress_transformers_output()` ⭐ **RECOMMENDED**
Targeted suppression for HuggingFace operations:
```python
with suppress_transformers_output():
    model = AutoModel.from_pretrained(...)  # Clean!
```

#### `capture_output()`
Capture output to strings for logging:
```python
with capture_output() as (out, err):
    operation()
    
captured = out.getvalue()
```

### 3. Updated Model Loading (`src/providers/huggingface.py`)

**All model loading now wrapped with suppressors:**

```python
# VLM loading
with suppress_transformers_output():
    processor = AutoProcessor.from_pretrained(...)
    model = AutoModel.from_pretrained(...)

# LLM loading  
with suppress_transformers_output():
    tokenizer = AutoTokenizer.from_pretrained(...)
    model = AutoModel.from_pretrained(...)
```

### 4. Updated Quantization (`src/quantization/techniques/bnb.py`)

**BitsAndBytes quantization now silent:**

```python
with suppress_transformers_output():
    model = AutoModelForCausalLM.from_pretrained(
        model_identifier,
        quantization_config=bnb_config,
        device_map="auto",
        ...
    )
```

## What's Suppressed

### ✅ Completely Suppressed
- [x] HuggingFace download progress bars
- [x] "Loading checkpoint shards" messages
- [x] Transformers info messages
- [x] PyTorch CUDA initialization
- [x] BitsAndBytes welcome message
- [x] Tokenizer parallelism warnings
- [x] Symlink warnings
- [x] Library deprecation warnings
- [x] TensorFlow messages (if present)

### ⚠️ Still Shown (As Intended)
- [x] **Errors** - Critical issues always shown
- [x] **Your TUI** - All Rich panels, menus, progress bars
- [x] **User prompts** - Input requests
- [x] **Results** - Inference output, model info

### 📝 Logged to File (Not Console)
- [x] All INFO/DEBUG messages
- [x] Library initialization
- [x] Model loading details
- [x] Quantization progress
- [x] Everything for debugging

## Testing Checklist

### Test Scenarios
Run each and verify **NO metadata appears** in terminal:

#### 1. Model Installation
```bash
./run.sh
# [1] Inference → Select provider → [i] Install model
# Should see: Only your TUI, NO download progress bars
```

#### 2. Model Loading
```bash
./run.sh
# Load any HuggingFace model
# Should see: Only "Loading..." screen, NO transformers messages
```

#### 3. Quantization
```bash
./run.sh  
# [2] Quantization → Start → BitsAndBytes 4-bit
# Should see: Only progress UI, NO library output
```

#### 4. Inference
```bash
./run.sh
# Run any inference (QA, Caption, Text)
# Should see: Only TUI prompts and results, NO warnings
```

### What to Look For

**✅ GOOD - Clean TUI:**
```
┌─────────────────────────────────┐
│ Select Model                    │
│                                 │
│ [1] Model 1                     │
│ [2] Model 2                     │
└─────────────────────────────────┘
Choose [1-2/b/q]:
```

**❌ BAD - Metadata Leakage:**
```
Loading checkpoint shards: 100%|████| 2/2
Some weights were not initialized: [....]
The model was loaded with use_flash_attention_2=True
┌─────────────────────────────────┐
│ Select Model                    │  <-- TUI mixed with library output!
```

## Files Modified

1. `src/core/logging_setup.py` - Enhanced with 8+ env vars + logger suppression
2. `src/utils/output_suppressor.py` - NEW: Context managers for output control
3. `src/providers/huggingface.py` - Wrapped all loading with suppressors
4. `src/quantization/techniques/bnb.py` - Suppressed BnB output

## Log Files

All suppressed output goes to log files for debugging:
- Location: `logs/vlm_cli_YYYY-MM-DD.log`
- Level: DEBUG (everything)
- Rotation: 100 MB per file
- Retention: 30 days
- Compression: ZIP after rotation

**View logs:**
```bash
tail -f logs/vlm_cli_*.log
```

## Troubleshooting

### Still Seeing Output?

1. **Check if it's an ERROR** - Errors are intentionally shown
2. **Check log file** - Verify suppression is working (should see messages in log)
3. **Restart application** - Env vars set at startup
4. **Check specific library** - Add to `noisy_loggers` list if needed

### Need to See More Output (Debugging)?

**Temporarily enable verbose mode:**

```python
# In src/core/logging_setup.py, change:
setup_logging(console_level="ERROR")  # Normal
# To:
setup_logging(console_level="DEBUG")  # Verbose
```

Or set environment variable before running:
```bash
export TRANSFORMERS_VERBOSITY=info
./run.sh
```

## Production Notes

### Environment Variables Set
These are automatically configured in `setup_logging()`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `TRANSFORMERS_VERBOSITY` | `error` | Suppress transformers info |
| `HF_HUB_DISABLE_PROGRESS_BARS` | `1` | No download bars |
| `TOKENIZERS_PARALLELISM` | `false` | No tokenizer warnings |
| `BITSANDBYTES_NOWELCOME` | `1` | No BnB welcome |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | `1` | No symlink warnings |
| `TF_CPP_MIN_LOG_LEVEL` | `3` | TF errors only |
| `DATASETS_VERBOSITY` | `error` | Datasets quiet |
| `ACCELERATE_VERBOSITY` | `error` | Accelerate quiet |

### Python Logging Suppressed
These loggers are set to ERROR level:
- transformers (all submodules)
- huggingface_hub
- torch, torch.nn, torch.cuda
- bitsandbytes
- urllib3, filelock

### Future Additions
If new libraries print unwanted output:

1. Add env var to `setup_logging()`
2. Add logger name to `noisy_loggers` list
3. Consider wrapping calls with `suppress_transformers_output()`

## Summary

**Before:**
```
Loading checkpoint shards: 100%|███████| 2/2 [00:01<00:00]
Some weights of the model were not initialized...
WARNING: tokenizers parallelism is enabled...
┌─ Your TUI Here (mixed with library spam) ─┐
```

**After:**
```
┌─ Your Clean TUI ─┐
│                  │
│ Only what you    │
│ designed shows!  │
└──────────────────┘
```

All metadata → `logs/` directory for debugging when needed! 🎉
