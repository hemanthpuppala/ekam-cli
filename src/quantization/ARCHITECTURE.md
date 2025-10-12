# Quantization Module Architecture

## Overview
Modular quantization system with clear separation of concerns.

## Directory Structure

```
src/quantization/
├── __init__.py                    # Main module exports
├── models.py                      # Data models (QuantizationType, QuantizationTask, etc.)
├── manager.py                     # Main orchestration layer
├── ARCHITECTURE.md               # This file
│
├── core/                         # Core functionality
│   ├── __init__.py
│   ├── background.py            # Background job management with threading
│   └── recommendations.py        # Hardware-aware recommendation engine
│
├── techniques/                   # Quantization technique implementations
│   ├── __init__.py
│   ├── base.py                  # BaseQuantizer abstract interface
│   ├── gguf.py                  # GGUF quantization (llama.cpp)
│   ├── gptq.py                  # GPTQ quantization (Phase 2)
│   ├── awq.py                   # AWQ quantization (Phase 2)
│   └── bnb.py                   # BitsAndBytes quantization (Phase 2)
│
├── methods/                      # Quantization methods and utilities
│   ├── __init__.py
│   ├── calibration.py           # Calibration methods (Phase 2)
│   └── validation.py            # Model validation methods (Phase 2)
│
└── ui/                          # User interface components
    ├── __init__.py
    ├── display.py               # Display utilities and prompts
    ├── workflow.py              # Main quantization workflow
    └── monitoring.py            # Background job monitoring UI (future)
```

## Module Responsibilities

### Root Level

**models.py**
- Data structures: `QuantizationType`, `QuantizationModule`, `QuantizationTask`, `QuantizationRecommendation`
- Enums with display names and properties
- Task state management
- JSON serialization

**manager.py**
- Main `QuantizationManager` class
- Orchestrates all quantization operations
- Interfaces with background job manager
- Filters quantizable models
- Generates recommendations

### core/ - Core Functionality

**background.py**
- `BackgroundJobManager` class
- Threading-based job execution
- JSON persistence for job state
- Progress tracking and callbacks
- Status bar generation

**recommendations.py**
- Hardware-based recommendation engine
- Analyzes system specs (RAM, CPU, GPU)
- Generates recommendations for each quantization type
- Estimates size, time, quality, speed scores
- Methods: `get_quantization_recommendations()`, `check_can_quantize_multiple()`

### techniques/ - Quantization Implementations

**base.py**
- `BaseQuantizer` abstract class
- Interface all quantizers must implement:
  - `check_availability()` - Check if technique is available
  - `get_supported_types()` - List supported quantization types
  - `get_source_model_path()` - Get source model file path
  - `quantize()` - Perform quantization
  - `estimate_output_size()` - Estimate output size
  - `validate_compatibility()` - Check model compatibility

**gguf.py** (Phase 1)
- `GGUFQuantizer` class extends `BaseQuantizer`
- Implements GGUF quantization using llama.cpp
- Supports: Q4_K_M/S, Q5_K_M/S, Q6_K, Q8_0
- Real-time progress parsing
- Phase 1: GGUF models only

**gptq.py** (Phase 2)
- GPTQ quantization for GPU inference
- 3-bit and 4-bit quantization
- Requires CUDA

**awq.py** (Phase 2)
- AWQ quantization (better quality than GPTQ)
- 4-bit quantization
- Requires CUDA

**bnb.py** (Phase 2)
- BitsAndBytes quantization
- Integrated with HuggingFace Transformers
- 4-bit and 8-bit options

### methods/ - Quantization Methods

**calibration.py** (Phase 2)
- Calibration dataset generation
- Calibration methods for improved accuracy
- Dataset sampling strategies

**validation.py** (Phase 2)
- Model validation after quantization
- Quality metrics computation
- Perplexity testing
- Inference speed benchmarks

### ui/ - User Interface

**display.py**
- Display components:
  - `show_quantization_intro()` - Introduction panel
  - `select_model_to_quantize()` - Model selection menu
  - `select_quantization_type()` - Quantization type selection with recommendations
  - `ask_gpu_preference()` - CPU/GPU selection
  - `confirm_quantization()` - Confirmation dialog
  - `ask_background_mode()` - Live vs background mode
  - `show_live_progress()` - Real-time progress display

**workflow.py**
- Main workflow orchestration:
  - `run_quantization_workflow()` - Complete end-to-end workflow
  - `show_background_jobs_monitor()` - Background jobs monitor

**monitoring.py** (Phase 2)
- Real-time background job monitoring
- `/background` command implementation
- Multi-job status display

## Data Flow

1. **User selects Quantization mode** → `ui/workflow.py`
2. **Select model** → `manager.py` filters quantizable models
3. **Get recommendations** → `core/recommendations.py` analyzes hardware
4. **Select quantization type** → User chooses from recommendations
5. **Create task** → `manager.py` creates `QuantizationTask`
6. **Submit task** → `core/background.py` starts background thread
7. **Execute quantization** → `techniques/gguf.py` performs quantization
8. **Update progress** → Callbacks update task state
9. **Complete** → Save quantized model to `results/quantizations/`

## Integration Points

### Main App Integration
- `src/core/app_quantization.py` - Main app integration
- Add to main menu after system specs
- Initialize `QuantizationManager` with system specs

### Status Bar
- Show across all screens when jobs are active
- Format: `🔧 Quantizing model_name [Q4_K_M] ████████▒▒ 80% ETA: 2m 30s`

### Commands
- `/background` - Monitor background jobs
- `/cancel <task_id>` - Cancel running job

## Future Phases

### Phase 2: HuggingFace Support
- Add HF→GGUF conversion pipeline
- Implement GPTQ and AWQ quantizers
- Add calibration methods

### Phase 3: Ollama Support
- Add Ollama→GGUF export
- Integration with Ollama model management

## Dependencies

### Required
- `llama-cpp-python` - GGUF quantization (already in requirements.txt)

### Phase 2
- `auto-gptq` - GPTQ quantization
- `autoawq` - AWQ quantization
- `optimum` - HuggingFace quantization tools
- `bitsandbytes` - BnB quantization

## Configuration

### Quantization Output
- Default: `results/quantizations/`
- Naming: `{model_id}_{quant_type}_{timestamp}.gguf`
- State file: `results/quantizations/.quantization_state.json`

### System Requirements
- Minimum: 8GB RAM, 4 CPU cores
- Recommended: 16GB RAM, 8 CPU cores
- Multiple jobs: 32GB RAM, 8+ cores

## Error Handling

All modules follow consistent error handling:
1. Validate inputs
2. Check availability/compatibility
3. Try operation with detailed logging
4. Update task status on failure
5. Provide user-friendly error messages

## Logging

- Module: `loguru` logger
- Levels:
  - DEBUG: Progress updates, detailed steps
  - INFO: Major operations, task lifecycle
  - WARNING: Non-fatal issues
  - ERROR: Failures with details

## Testing Strategy

### Unit Tests
- `test_quantization_models.py` - Data model tests
- `test_recommendations.py` - Recommendation logic
- `test_gguf_quantizer.py` - GGUF quantizer

### Integration Tests
- `test_quantization_workflow.py` - End-to-end workflow
- `test_background_jobs.py` - Background job management

### Contract Tests
- `test_base_quantizer.py` - BaseQuantizer interface compliance
