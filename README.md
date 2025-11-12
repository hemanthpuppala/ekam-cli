# Ekam-CLI - Unified AI Model Interface

**Ekam** (एकम्) means "Unity" in Sanskrit - representing the unified interface for all your AI model testing needs.

A production-ready CLI for testing Vision-Language Models (VLMs) and Large Language Models (LLMs) across multiple providers with automatic resource management, quantization, and comprehensive benchmarking.

## Key Features

- **Dynamic VLM Detection**: Automatically detects vision capabilities via API introspection
- **Multi-Provider Support**: Ollama, HuggingFace, LM Studio, GGUF (coming soon)
- **Smart Memory Management**: Intelligent RAM calculation based on GPU/CPU architecture
- **Automatic Cleanup**: Models are always unloaded on exit (normal quit, Ctrl+C, Ctrl+Z, kill)
- **Clean TUI**: Dynamic terminal interface with responsive layouts
- **5 VLM Endpoints**: Question Answering, Captioning, Detection, Pointing, Text Chat

## Quick Start
```bash
./run.sh
```

See [quickstart.md](specs/001-build-an-cli/quickstart.md) for detailed installation and usage.

## Resource Cleanup

Models are automatically unloaded in **ALL** exit scenarios:

1. ✅ **Normal exit** - User quits via menu
2. ✅ **Ctrl+C (SIGINT)** - Interrupt signal
3. ✅ **Ctrl+Z (SIGTSTP)** - Suspend signal (unloads before suspending)
4. ✅ **Kill/Terminate (SIGTERM)** - Termination signal
5. ✅ **Unexpected errors** - Exception handler triggers cleanup

No manual cleanup required!

## VLM GGUF Compatibility Notes

- Ekam uses llama.cpp’s `convert_hf_to_gguf.py` with `--mmproj` for VLMs, producing a text model GGUF and a `mmproj-*.gguf` vision projector.
- After conversion, Ekam inspects the mmproj to detect the expected input dimension of the projector MLP:
  - 1152 → single-crop features (fully compatible)
  - 2304 → multi-crop (global + regional) features (e.g., some Moondream2 variants)

If 2304 is detected, Ekam logs a warning. The bundled llama.cpp in this repository includes a safe concat-zero fallback in `tools/mtmd/clip.cpp` that prevents crashes on Metal and other backends (quality may be slightly degraded vs. multi-crop training).

### Optional VLM Smoke Test

You can run a quick local sanity test after conversion by enabling:

```bash
EKAM_VLM_SMOKETEST=1 ekam quantize --to gguf <hf_model_id>
```

Requirements:
- A locally built `llama.cpp` binary (`llama-mtmd-cli` or `llava-cli`)
- A test image in `assets/` (e.g., `assets/image_3.jpg`)
