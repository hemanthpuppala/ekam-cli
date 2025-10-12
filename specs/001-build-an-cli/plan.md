# Implementation Plan: High-Performance Multi-Provider VLM/LLM CLI with Resource Management

**Branch**: `001-build-an-cli` | **Date**: 2025-10-11 | **Spec**: [spec.md](./spec.md)

## Summary

Build a production-ready CLI application for testing VLM/LLM models locally across multiple providers (Ollama, HuggingFace Transformers, LM Studio, GGUF). The application enables O(1) menu operations, dynamic resource-aware model loading/unloading, multi-endpoint testing (QA, Caption, Detect, Point, Text), and comprehensive performance tracking. Optimized for edge devices (Raspberry Pi, Jetson) through minimal memory overhead (~731MB base) and intelligent model compatibility assessment before loading.

**Technical Approach**: Python 3.8+ with Rich CLI (not Textual TUI), Pydantic v2 validation, httpx for API providers, transformers/llama-cpp-python for local inference. Configuration-driven architecture with YAML config, provider abstraction interface, and lazy loading patterns for memory efficiency. Advisory memory warnings with user override capability (per clarifications).

---

## Technical Context

**Language/Version**: Python 3.8+ (broad edge device compatibility: Raspberry Pi, Jetson Nano/Orin, IPC devices, laptops)

**Primary Dependencies**:
- Rich 13.7+ (terminal formatting, ~5MB overhead)
- Pydantic 2.5+ (data validation with Rust core, ~8MB overhead)
- httpx 0.25+ (HTTP client for Ollama/LM Studio APIs, ~2MB overhead)
- transformers 4.35+ (HuggingFace local inference, ~150MB overhead)
- torch 2.1+ (PyTorch model execution, ~500MB overhead)
- llama-cpp-python 0.2.20+ (GGUF inference with CPU/GPU optimization, ~10MB overhead)
- Pillow 10.1+ (image preprocessing, ~2MB overhead)
- PyYAML 6.0+ (configuration management, ~1MB overhead)
- psutil 5.9+ (system metrics, ~2MB overhead)
- loguru 0.7+ (production logging, ~1MB overhead)

**Storage**: File-based (JSON for results persistence, YAML for config, local model caches per provider)

**Testing**: pytest 7.4+ with contract, integration, and unit test suites

**Target Platform**: Cross-platform (Linux, macOS, Windows) with focus on edge devices (ARM64, x86_64)

**Project Type**: Single CLI application

**Performance Goals**:
- CLI startup: <3 seconds (including system detection and model discovery)
- Menu operations: <100ms (O(1) lookups via hash maps per FR-049)
- Model switching: <10 seconds (unload + load for models <5GB)
- Memory overhead: <200MB for CLI itself (excludes loaded models)

**Constraints**:
- Local inference only (no cloud providers per CL-001)
- Single loaded model at a time (per FR-016)
- Advisory memory warnings with user override (per CL-008)
- O(1) complexity for all non-inference operations (per FR-049)

**Scale/Scope**:
- 4 provider implementations (Ollama, HF, LM Studio, GGUF)
- 5 endpoints (QA, Caption, Detect, Point, Text)
- ~50 models discoverable across all providers (10-20 per provider)

---

## Constitution Check

✅ **I. Provider Abstraction** - PASS
- All providers implement `BaseProvider` interface (see contracts/)
- Unified `ModelInfo` with capabilities list enables polymorphic endpoint routing
- New providers addable without modifying existing code

✅ **II. Configuration-Driven Architecture** - PASS
- All provider settings in `config.yaml` (hosts, cache_dirs, device preferences)
- No hardcoded URLs or model names in source code
- Runtime behavior fully configurable (timeouts, image resizing, memory margins)

✅ **III. Production-Ready Reliability** - PASS
- Comprehensive error handling via try/except with loguru logging
- Results persistable to JSON with full metadata (FR-042)
- Inference timing tracked for all operations (FR-029)
- Graceful degradation when providers unavailable (FR-044)

✅ **IV. Interactive User Experience** - PASS
- Rich library provides formatted tables, progress bars, panels
- Standard input() provides simple synchronous control flow
- Clear memory warnings with explicit confirmation prompts (CL-008)
- Back navigation in all menus (CL-010)

✅ **V. Extensibility First** - PASS
- New endpoints addable via single method per provider
- New models addable via config or provider-specific install commands
- Model-specific optimizations supported (e.g., Moondream native methods) via capability detection

**Gate Status**: ✅ All principles satisfied, no violations to justify

---

## Project Structure

### Documentation (this feature)

```
specs/001-build-an-cli/
├── spec.md              # Feature specification (complete)
├── plan.md              # This file (implementation plan)
├── research.md          # Phase 0: Technology decisions & SOTA choices
├── data-model.md        # Phase 1: Pydantic schemas for all entities
├── quickstart.md        # Phase 1: User onboarding guide
└── tasks.md             # Phase 2: NOT YET CREATED (use /speckit.tasks)
```

### Source Code (repository root)

```
src/
├── models/                      # Pydantic data models (~8 files)
│   ├── __init__.py
│   ├── provider.py              # ProviderConfig, ProviderType
│   ├── model.py                 # ModelInfo, ModelType, CompatibilityStatus
│   ├── system.py                # SystemSpecs, GPUInfo
│   ├── session.py               # SessionState, SessionStatistics, LoadedModel
│   ├── inference.py             # InferenceInput, InferenceOutput, InferenceResult
│   └── endpoints.py             # EndpointType enum
│
├── providers/                   # Provider implementations (~5 files)
│   ├── __init__.py
│   ├── base.py                  # BaseProvider abstract class
│   ├── ollama.py                # OllamaProvider (httpx client)
│   ├── huggingface.py           # HuggingFaceProvider (transformers pipelines)
│   ├── lm_studio.py             # LMStudioProvider (OpenAI-compatible API)
│   └── gguf.py                  # GGUFProvider (llama-cpp-python)
│
├── services/                    # Core business logic (~4 files)
│   ├── __init__.py
│   ├── model_discovery.py       # ModelDiscoveryService (aggregates from all providers)
│   ├── resource_manager.py      # ResourceManager (memory checks, compatibility assessment)
│   └── session.py               # SessionManager (state coordination, statistics)
│
├── cli/                         # CLI UI logic (~5 files)
│   ├── __init__.py
│   ├── menus.py                 # Menu classes (MainMenu, ModelMenu, EndpointMenu)
│   ├── prompts.py               # Input prompt wrappers (image_path, question, etc.)
│   ├── display.py               # Rich formatters (tables, panels, progress bars)
│   └── hyperlinks.py            # OSC 8 hyperlink generation for clickable paths
│
├── utils/                       # Shared utilities (~3 files)
│   ├── __init__.py
│   ├── image.py                 # Image preprocessing (resize, convert)
│   └── persistence.py           # JSON save/load for results
│
└── core/                        # Application entry point (~1 file)
    ├── __init__.py
    └── app.py                   # Main CLI loop, provider coordination

tests/
├── unit/                        # Fast isolated tests (~10 files)
│   ├── test_models.py           # Pydantic validation
│   ├── test_resource_manager.py # Memory calculations
│   ├── test_image_utils.py      # Image preprocessing
│   └── test_session.py          # SessionState logic
│
├── integration/                 # Provider interaction tests (~4 files)
│   ├── test_ollama.py           # Requires Ollama server running
│   ├── test_huggingface.py      # Requires HF models cached
│   ├── test_lm_studio.py        # Requires LM Studio running
│   └── test_gguf.py             # Requires GGUF files present
│
└── contract/                    # Provider interface compliance (~1 file)
    └── test_provider_contract.py # Verifies all providers implement BaseProvider

config.yaml                      # Configuration (provider hosts, device preferences)
run.sh                           # Launcher script with provider detection & installation
requirements.txt                 # Production dependencies
requirements-dev.txt             # Development dependencies (pytest, ruff)
requirements-edge.txt            # Minimal deps for edge devices (CPU-only, no GPU libs)
logs/                            # Application logs (auto-created, gitignored)
results/                         # Inference results (auto-created, gitignored)
├── *.json                       # InferenceResult JSON files
└── annotated_images/            # Detect/Point output images
```

**Structure Decision**: Single CLI application (Option 1) selected because:
- No frontend/backend separation needed (CLI-only)
- No mobile app requirements
- Unified codebase simplifies deployment to edge devices
- Clear separation of concerns via subdirectories (models, providers, services, cli, core)

**File Count Estimate**: ~35 Python files total (8 models, 5 providers, 4 services, 5 CLI, 3 utils, 1 core, ~15 tests)

---

## Detailed Phase Breakdown

### Phase 0: Research & Technology Selection ✅ COMPLETE

**Artifacts Generated**:
- `research.md`: Technology decisions with rationale for all dependencies
- Memory footprint analysis (~731MB total CLI overhead)
- Provider-specific API patterns documented
- Optimization strategies defined (lazy loading, cache clearing, O(1) lookups)

**Key Decisions Made**:
- Rich + input() instead of Textual (40MB overhead saved, simpler threading)
- Pydantic v2 for data validation (10x faster than v1)
- httpx for unified HTTP client (Ollama + LM Studio)
- llama-cpp-python for GGUF (in-process, no server overhead)
- ruff for linting/formatting (10-100x faster than Black+Flake8)

### Phase 1: Design & Contracts ✅ COMPLETE

**Artifacts Generated**:
- `data-model.md`: Complete Pydantic schemas for 7 core entities
- `quickstart.md`: End-to-end user onboarding guide with examples
- `CLAUDE.md`: Updated agent context with final tech stack

**Entity Models Defined**:
1. `ProviderConfig`: Configuration per provider (host, cache_dir, device_preference)
2. `ModelInfo`: Model metadata with dynamic capabilities detection
3. `SystemSpecs`: Hardware detection (RAM, GPU, recommended model size)
4. `LoadedModel`: Active model state with memory tracking
5. `InferenceResult`: Output with timing and full metadata
6. `SessionStatistics`: Cumulative performance metrics
7. `SessionState`: Global application state coordinator

**Validation Rules**:
- All models use Pydantic v2 `Field()` constraints
- `@computed_field` for derived properties (compatibility_icon, average_time_ms)
- O(1) model lookups via dict-based registries

### Phase 2: Task Generation (NOT YET EXECUTED)

**Next Command**: `/speckit.tasks`

**Expected Output**: `tasks.md` with dependency-ordered implementation tasks

**Task Categories to Generate**:
1. **Foundation** (P0): Project setup, config loading, logging
2. **Models** (P0): Pydantic schemas, validation
3. **Providers** (P1): BaseProvider interface, 4 concrete implementations
4. **Services** (P1): Model discovery, resource management, session
5. **CLI** (P2): Menus, prompts, display formatters
6. **Integration** (P2): Provider coordination, model loading workflow
7. **Testing** (P3): Unit, integration, contract tests
8. **Documentation** (P3): Inline docstrings, README

---

## Constitution Compliance Matrix

| Principle | Implementation Strategy | Validation Method |
|-----------|------------------------|-------------------|
| **I. Provider Abstraction** | `BaseProvider` ABC with required methods | Contract tests verify all providers implement interface |
| **II. Configuration-Driven** | `config.yaml` loaded via PyYAML, validated with Pydantic | No hardcoded URLs/paths in source (enforced by code review) |
| **III. Production-Ready Reliability** | Try/except + loguru, JSON persistence, timing | Integration tests simulate failures, check error messages |
| **IV. Interactive User Experience** | Rich tables/panels, clear prompts, back navigation | Manual UX testing, SC-010 (first inference <5 min) |
| **V. Extensibility First** | New endpoints via method, new models via config | Add test model/endpoint without modifying existing code |

---

## Risk Analysis

### Risk 1: llama-cpp-python Build Failures on Edge Devices

**Likelihood**: Medium (compiler dependencies vary by platform)

**Impact**: High (GGUF provider unavailable)

**Mitigation**:
- Provide pre-built wheels for Raspberry Pi, Jetson (GitHub releases)
- Fallback to CPU-only build if GPU compilation fails
- Document manual build process in quickstart.md
- Make GGUF provider optional (disable if not needed)

### Risk 2: HuggingFace Model Download Timeouts

**Likelihood**: Medium (network issues, large models)

**Impact**: Medium (blocks first-time setup)

**Mitigation**:
- Implement retry logic with exponential backoff (httpx retries)
- Display progress bars during downloads (transformers has built-in support)
- Provide offline mode with pre-downloaded model list
- Cache successful downloads to avoid re-downloads

### Risk 3: Memory Estimation Inaccuracy

**Likelihood**: Low (psutil reliable, estimates conservative)

**Impact**: Medium (users may load too-large models)

**Mitigation**:
- Conservative estimates with 20% safety margin (FR-013 formula)
- Dynamic memory monitoring during load (abort if exceeds threshold)
- Clear warning messages with alternative model suggestions
- User override with explicit confirmation (CL-008)

### Risk 4: O(1) Lookup Performance Degradation with Many Models

**Likelihood**: Low (dict lookups are O(1) in Python)

**Impact**: Low (violates FR-049 if degraded)

**Mitigation**:
- Use Python dict (hash table, guaranteed O(1) average case)
- Benchmark with 1000+ models in registry (stress test)
- Profile with cProfile to verify no hidden O(N) scans
- Document max model count (>10k should still be <1ms lookup)

---

## Performance Optimization Strategy

### Lazy Loading for Provider Libraries

**Problem**: Loading all provider libraries upfront wastes memory if user only uses one

**Solution**: Dynamic imports on provider selection
```python
def get_provider(provider_type: ProviderType):
    if provider_type == ProviderType.HUGGINGFACE:
        from .providers.huggingface import HuggingFaceProvider  # Lazy import
        return HuggingFaceProvider()
```

**Memory Saved**: ~150MB Transformers not loaded if user selects Ollama

### Cache Clearing After Model Unload

**Problem**: PyTorch caches GPU memory, doesn't release until explicit clear

**Solution**: Call `torch.cuda.empty_cache()` / `torch.mps.empty_cache()` + `gc.collect()` (per FR-017)

**Memory Recovered**: ~95% of model RAM freed within 2 seconds

### Image Streaming for Large Files

**Problem**: Loading 4K images into RAM before resize wastes memory

**Solution**: Use Pillow's `thumbnail()` method for in-place modification

**Memory Saved**: 75% reduction for 4K→1080p downsampling (16MB → 4MB)

### Precomputed Model Indices

**Problem**: FR-049 requires O(1) model lookups

**Solution**: Index models by `model_id` in dict on discovery, separate indices for provider filtering

**Performance**: Menu selection <1ms regardless of model count

---

## Next Steps

### Immediate (this session):
1. ✅ Generate `research.md` with tech stack decisions
2. ✅ Generate `data-model.md` with Pydantic schemas
3. ✅ Generate `quickstart.md` with user workflows
4. ✅ Update `CLAUDE.md` with final tech stack
5. ⏳ Delete old code files (per user request)

### Next Session:
1. Run `/speckit.tasks` to generate dependency-ordered task list
2. Implement foundation (P0): Project structure, config loading, logging
3. Implement models (P0): All Pydantic schemas with validation
4. Implement providers (P1): Start with BaseProvider + OllamaProvider

**Phase 1 Complete** ✅

**Branch**: `001-build-an-cli`
**Planning Artifacts**: research.md, data-model.md, quickstart.md, plan.md (this file)
**Next Command**: `/speckit.tasks` to generate implementation tasks
