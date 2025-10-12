# VLM/LLM CLI - High-Performance Multi-Provider Testing Tool

A production-ready CLI for testing Vision-Language Models (VLMs) and Large Language Models (LLMs) across multiple providers with automatic resource management.

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
