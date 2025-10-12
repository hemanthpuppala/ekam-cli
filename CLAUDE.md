# vlm-tester Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-10-11

## Active Technologies

**Core Stack** (001-build-an-cli):
- Python 3.8+ (broad edge device compatibility: RPi, Jetson, IPC)
- Rich 13.7+ (terminal formatting with ~5MB overhead)
- Pydantic 2.5+ (data validation with Rust core)
- PyYAML 6.0+ (configuration management)

**Provider Libraries**:
- httpx 0.25+ (async+sync HTTP for Ollama, LM Studio APIs)
- transformers 4.35+ (HuggingFace local inference)
- torch 2.1+ (PyTorch for model execution)
- llama-cpp-python 0.2.20+ (GGUF inference with CPU/GPU optimization)

**System & Utilities**:
- Pillow 10.1+ (image preprocessing, ~2MB overhead)
- psutil 5.9+ (cross-platform system metrics)
- loguru 0.7+ (production logging with rotation)

**Development**:
- pytest 7.4+ (testing framework)
- ruff 0.1+ (fast linting & formatting, Rust-based)

## Project Structure

```
src/
├── models/              # Pydantic data models (ProviderConfig, ModelInfo, etc.)
├── providers/           # Provider implementations (Ollama, HF, LM Studio, GGUF)
│   ├── base.py         # BaseProvider interface
│   ├── ollama.py
│   ├── huggingface.py
│   ├── lm_studio.py
│   └── gguf.py
├── services/            # Core services (model discovery, resource management)
│   ├── model_discovery.py
│   ├── resource_manager.py
│   └── session.py
├── cli/                 # CLI UI logic (Rich-based menus)
│   ├── menus.py
│   ├── prompts.py
│   └── display.py
└── core/                # Application entry point
    └── app.py

tests/
├── unit/                # Fast isolated tests (Pydantic validation, utils)
├── integration/         # Provider interaction tests (requires running services)
└── contract/            # Provider interface compliance tests

config.yaml              # Configuration (provider hosts, device preferences)
run.sh                   # Launcher script with auto-detection & installation
requirements.txt         # Production dependencies
requirements-dev.txt     # Development dependencies (pytest, ruff)
requirements-edge.txt    # Minimal deps for edge devices (no GPU libs)
```

## Commands

**Setup**:
```bash
# Automated setup with provider detection
./run.sh

# Manual setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Run Application**:
```bash
./run.sh                    # Recommended (checks prerequisites)
python3 -m src.core.app     # Direct launch
```

**Testing**:
```bash
pytest                      # All tests
pytest tests/unit/          # Fast unit tests only
pytest tests/integration/   # Requires Ollama/HF models
pytest -v --tb=short        # Verbose with short tracebacks
```

**Code Quality**:
```bash
ruff check .                # Lint codebase
ruff format .               # Format code
ruff check --fix .          # Auto-fix violations
```

**Provider-Specific**:
```bash
# Ollama (required for Ollama provider)
ollama serve                # Start Ollama server

# GGUF with GPU support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python  # NVIDIA
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python   # Apple Silicon
```

## Code Style

**Python 3.8+**: PEP 8 compliance via Ruff, 100-char line length

**Key Conventions**:
- **Type hints**: Required for all function signatures (enforced by Ruff ANN rules)
- **Pydantic models**: Use for all data entities (ModelInfo, SystemSpecs, etc.)
- **Docstrings**: Google-style for public APIs
- **Error handling**: Use try/except with loguru logging, never silent failures
- **O(1) operations**: Use dict/set for lookups (per FR-049), not list scans

**Example**:
```python
from pydantic import BaseModel, Field
from loguru import logger

class ModelInfo(BaseModel):
    """Complete metadata for a discoverable model"""
    model_id: str = Field(description="Unique identifier")
    size_gb: float = Field(gt=0)

def get_model(model_id: str) -> ModelInfo | None:
    """Retrieve model by ID with O(1) lookup.

    Args:
        model_id: Unique model identifier

    Returns:
        ModelInfo if found, None otherwise
    """
    try:
        return model_registry[model_id]  # O(1) dict lookup
    except KeyError:
        logger.warning(f"Model {model_id} not found in registry")
        return None
```

## Architecture Principles (from Constitution)

1. **Provider Abstraction**: All providers implement BaseProvider interface
2. **Configuration-Driven**: No hardcoded URLs/paths, all in config.yaml
3. **Production-Ready Reliability**: Comprehensive error handling, timing, persistence
4. **Interactive UX**: Rich formatting, progress bars, clear feedback
5. **Extensibility First**: New endpoints via single method, new models via config

## Recent Changes

- 001-build-an-cli: Full tech stack defined (Python 3.8+, Rich, Pydantic, httpx, transformers, llama-cpp-python)
- 001-build-an-cli: Project structure finalized (providers/, services/, cli/, core/)
- 001-build-an-cli: Removed Textual TUI in favor of Rich + input() (sync CLI model)

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
- do not create md file documents for every change just give a summary of what you did in text format in chat