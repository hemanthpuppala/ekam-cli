# Bug Fix: OLLAMA Models Not Showing in Quantization Pipeline on Windows

**Date:** 2025-11-06
**Issue:** OLLAMA models were not appearing in the quantization model registry on Windows systems
**Status:** ✅ Fixed

---

## Error

When running the quantization pipeline on Windows, OLLAMA models were completely absent from the model selection screen, while HUGGINGFACE and GGUF models appeared correctly.

```
╭────────────────────────────────────────────── Large Language Models ──────────────────────────────────────────────╮
│ ┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓ │
│ ┃    # ┃ Model Name                     ┃   Provider   ┃   Params ┃   Quant   ┃      RAM ┃     Size ┃    Fit     ┃ │
│ ┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩ │
│ │    1 │ Qwen/Qwen3-4B-Instruct-2507    │ HUGGINGFACE  │     4.0B │   BF16    │   10.1GB │    3.9GB │  OPTIMAL   │ │
│ └──────┴────────────────────────────────┴──────────────┴──────────┴───────────┴──────────┴──────────┴────────────┘ │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

# No OLLAMA models visible, even though they are installed locally
```

### Observed Symptoms

1. OLLAMA models were discovered by the provider
2. Models had `source_path = None` in the model info
3. Quantization manager filtered out all OLLAMA models
4. Warning logs showed: `"Could not parse blob hash from modelfile for {model_name}"`

---

## Root Cause

The `OllamaFileLocator` class failed to parse blob hashes from Ollama modelfiles on Windows systems due to **path separator incompatibility**.

### Technical Details

**File:** `src/utils/ollama_file_locator.py`
**Function:** `_parse_blob_hash()` (line 145-181)

**The Problem:**

1. Ollama on Windows generates modelfiles with **Windows-style paths** (backslashes):
   ```
   FROM C:\Users\hpuppala\.ollama\models\blobs\sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa
   ```

2. The original regex pattern only matched **Unix-style paths** (forward slashes):
   ```python
   pattern1 = r"FROM\s+(?:.*/)?(sha256-[a-f0-9]+)"
                           ^
                           Only matches forward slash /
   ```

3. This caused the regex to fail on Windows, resulting in:
   - `blob_hash = None`
   - `source_path = None` in `ModelInfo`
   - Models filtered out by `get_quantizable_models()` (requires `source_path` for OLLAMA)

### Why OLLAMA Models Require `source_path`

The quantization manager has provider-specific filtering logic in `src/quantization/manager.py` (lines 62-67):

```python
elif model.provider == ProviderType.OLLAMA:
    # Only include if we have source_path (blob location)
    if model.source_path:
        quantizable.append(model)
    else:
        logger.debug(f"Skipping Ollama model {model.name} - no source_path")
```

OLLAMA models must have a valid `source_path` pointing to their GGUF blob files because:
- Quantization requires access to the underlying GGUF model file
- OLLAMA stores models in a content-addressable blob storage (`~/.ollama/models/blobs/`)
- Without the blob path, the quantization process cannot access the model weights

---

## Solution

### Changes Made

**File:** `src/utils/ollama_file_locator.py:163`

**Before:**
```python
# Pattern 1: Full path to blob file
# FROM /Users/user/.ollama/models/blobs/sha256-abc123...
pattern1 = r"FROM\s+(?:.*/)?(sha256-[a-f0-9]+)"
```

**After:**
```python
# Pattern 1: Full path to blob file (Unix or Windows)
# FROM /Users/user/.ollama/models/blobs/sha256-abc123...
# FROM C:\Users\user\.ollama\models\blobs\sha256-abc123...
pattern1 = r"FROM\s+(?:.*[/\\])?(sha256-[a-f0-9]+)"
```

### What Changed

The regex pattern was updated to match **both path separator types**:
- `[/\\]` - Matches either forward slash `/` (Unix/Linux/macOS) or backslash `\\` (Windows)
- This makes the code cross-platform compatible

The updated docstring also documents both path formats:
```python
def _parse_blob_hash(self, modelfile: str) -> Optional[str]:
    """Parse blob hash from modelfile content.

    Expected formats:
        FROM /path/to/.ollama/models/blobs/sha256-abc123def456...
        FROM C:\path\to\.ollama\models\blobs\sha256-abc123def456...  # Added
        FROM blob:sha256-abc123def456...
        FROM @sha256:abc123def456...
    """
```

---

## Verification

### Test Results

After applying the fix, running `test_ollama_quantization.py` showed:

```
[OK] Discovered 3 Ollama models

Models found:

  1. qwen3:0.6b
     Provider: ollama (type: str)
     Model Type: llm
     Size: 0.49 GB
     Quantization: Q4_K_M
     Source Path: C:\Users\hpuppala\.ollama\models\blobs\sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa
     Architecture: qwen3

  2. qwen2.5vl:3b
     Provider: ollama (type: str)
     Model Type: vlm
     Size: 2.98 GB
     Quantization: Q4_K_M
     Source Path: C:\Users\hpuppala\.ollama\models\blobs\sha256-e9758e589d443f653821b7be9bb9092c1bf7434522b70ec6e83591b1320fdb4d
     Architecture: qwen25vl

  3. gemma3:1b
     Provider: ollama (type: str)
     Model Type: llm
     Size: 0.76 GB
     Quantization: Q4_K_M
     Source Path: C:\Users\hpuppala\.ollama\models\blobs\sha256-7cd4618c1faf8b7233c6c906dac1694b6a47684b37b8895d470ac688520b9c01
     Architecture: gemma3
```

✅ All models now have valid `source_path` values
✅ No parsing warnings in logs
✅ Models pass quantization filtering

### Success Logs

```
[INFO] Located Ollama model qwen3:0.6b at C:\Users\hpuppala\.ollama\models\blobs\sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa
[INFO] Located Ollama model qwen2.5vl:3b at C:\Users\hpuppala\.ollama\models\blobs\sha256-e9758e589d443f653821b7be9bb9092c1bf7434522b70ec6e83591b1320fdb4d
[INFO] Located Ollama model gemma3:1b at C:\Users\hpuppala\.ollama\models\blobs\sha256-7cd4618c1faf8b7233c6c906dac1694b6a47684b37b8895d470ac688520b9c01
```

---

## Impact

### Affected Systems
- **Windows 10/11** - Primary impact
- **WSL (Windows Subsystem for Linux)** - May have been affected if accessing Windows Ollama installation
- **Unix/Linux/macOS** - No impact (forward slashes already worked)

### Affected Functionality
- Quantization pipeline model discovery
- OLLAMA model requantization workflows
- Multi-format conversion from OLLAMA models

### Benefits of Fix
1. **Cross-platform compatibility** - Works on all operating systems
2. **Complete model discovery** - OLLAMA models now visible alongside HUGGINGFACE/GGUF models
3. **Unlocks quantization features** - Users can now requantize OLLAMA models to different formats
4. **Better error handling** - No silent failures on Windows

---

## Additional Notes

### Why This Bug Existed
- The codebase was likely developed/tested primarily on Unix-based systems (Linux/macOS)
- Ollama's behavior differs by platform - Windows uses native Windows paths
- The regex pattern worked perfectly on Unix systems, so the issue went unnoticed

### Testing Recommendations
For future cross-platform features:
1. Test path handling on both Windows and Unix systems
2. Use `Path` objects from `pathlib` for cross-platform path manipulation
3. When using regex for path matching, always consider both `/` and `\\` separators
4. Include Windows-specific test cases in CI/CD pipelines

### Related Files
- `src/utils/ollama_file_locator.py` - Fixed file
- `src/quantization/manager.py` - Uses `source_path` for filtering
- `src/providers/ollama.py` - Calls the file locator
- `test_ollama_quantization.py` - Test script used to verify fix

---

## Conclusion

This was a **platform-specific path separator issue** that prevented OLLAMA model discovery on Windows. The fix is minimal (single character class addition to regex) but critical for Windows users.

The solution is **backwards compatible** - it doesn't break existing Unix/Linux/macOS functionality while adding Windows support.
