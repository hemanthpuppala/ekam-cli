# Benchmarking Feature Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture Structure](#architecture-structure)
3. [UI/UX Pipeline](#uiux-pipeline)
4. [Suite 2: Speed & Throughput Analysis](#suite-2-speed--throughput-analysis)
5. [Implementation Tasks](#implementation-tasks)

---

## Overview

**Feature:** Benchmarking Tool for LLMs and VLMs
**Purpose:** Measure and compare model performance across different metrics (speed, resources, quality, stress)
**Integration:** Independent pipeline, can run in foreground (real-time UI) or background (async with /background monitoring)
**Architecture:** Modular, focused metrics per suite, reusable components for Full Profile

---

## Architecture Structure

### Directory Layout
```
src/benchmarking/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── benchmark_runner.py         # Main orchestrator, handles foreground/background
│   ├── metrics_collector.py        # Universal metric collection (reusable across suites)
│   ├── system_monitor.py           # Wrapper around SystemMonitor (reusable)
│   └── result_formatter.py         # Format results for display/export (reusable)
│
├── suites/
│   ├── __init__.py
│   ├── base_suite.py               # Abstract base class (reusable template)
│   ├── suite_speed.py              # Suite 2: Speed & Throughput (FOCUSED)
│   ├── suite_resources.py          # Suite 3: Resource Efficiency (FOCUSED)
│   ├── suite_quality.py            # Suite 4: Quality & Consistency (FOCUSED)
│   ├── suite_stress.py             # Suite 5: Stress & Endurance (FOCUSED)
│   └── suite_complete.py           # Suite 1: Complete System (imports all above)
│
├── handlers/
│   ├── __init__.py
│   ├── llm_handler.py              # LLM-specific endpoint handling
│   ├── vlm_handler.py              # VLM-specific endpoint handling
│   └── endpoint_executor.py        # Generic endpoint invocation (reusable)
│
├── datasets/
│   ├── __init__.py
│   ├── defaults.py                 # Static default prompts/images
│   ├── llm_datasets.py             # LLM test prompts
│   └── vlm_datasets.py             # VLM test images + prompts
│
├── reporters/
│   ├── __init__.py
│   ├── json_reporter.py            # Export to JSON
│   ├── csv_reporter.py             # Export to CSV
│   ├── html_reporter.py            # Generate HTML with graphs
│   └── console_formatter.py        # TUI display formatting (reusable)
│
├── ui/
│   ├── __init__.py
│   ├── benchmark_menu.py           # Main menu + suite selection
│   ├── model_selection.py          # Model selection screen
│   ├── endpoint_selection.py       # Endpoint selection (type-aware)
│   ├── test_data_config.py         # Test data configuration
│   ├── benchmark_config.py         # Benchmark settings (params, runs, etc.)
│   ├── foreground_display.py       # Real-time foreground UI
│   ├── results_viewer.py           # Results summary + export
│   └── background_monitor.py       # Background task monitoring (/background)
│
└── models/
    ├── __init__.py
    ├── benchmark_config.py         # BenchmarkConfig pydantic model
    ├── suite_result.py             # SuiteResult data structure
    └── metric_types.py             # Metric enums and types

```

### Core Benchmarking Principles
- **Per-Endpoint Isolation**: Each benchmark run tests ONLY ONE endpoint (never combine multiple endpoints)
- **Sequential Model Loading**: When testing multiple models, load one model → run all inferences → unload → load next model
- **Fair Comparison**: When testing multiple models on the same endpoint, ALL models receive IDENTICAL inputs (same prompts, same images)
- **Complete Transparency**: Show all measurements (baseline, per-run, cleanup) with no hidden data

### Modular Design Principle
- **Core modules** (metrics_collector, system_monitor, result_formatter): Reusable across ALL suites
- **Base suite class**: Template for all suite implementations
- **Individual suites**: Focused on specific metrics, import core + handlers
- **Full Profile suite**: Imports all other suites + aggregates results
- **Handlers**: Abstract endpoint execution, reusable for all suites

### Reusability Example
```python
# Suite 2 uses:
from ..core.metrics_collector import MetricsCollector
from ..core.system_monitor import SystemMonitorWrapper
from ..handlers.llm_handler import LLMHandler
from ..handlers.vlm_handler import VLMHandler

# Suite 1 (Full Profile) uses:
from .suite_speed import SpeedSuite
from .suite_resources import ResourceSuite
from .suite_quality import QualitySuite
from .suite_stress import StressSuite
# Plus aggregation logic
```

---

## UI/UX Pipeline

### Screen Flow

```
┌─────────────────────────────────────────────────────┐
│ Screen 1: Model Type Selection (LLM vs VLM)        │
│ ───────────────────────────────────────────────────│
│ 1. [LLM] Text-only Language Models                │
│ 2. [VLM] Vision-Language Models                   │
│ ───────────────────────────────────────────────────│
│ Selection: 1                                      │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Screen 2: Select Benchmark Suite                  │
│ ───────────────────────────────────────────────────│
│ 1. [✨] Complete System Analysis (all metrics)    │
│ 2. [⚡] Speed & Throughput (latency, tokens/sec)  │
│ 3. [💾] Resource Efficiency (CPU, GPU, memory)    │
│ 4. [🎯] Quality & Consistency (accuracy, IoU)     │
│ 5. [🔥] Stress & Endurance (long-run stability)   │
│ ───────────────────────────────────────────────────│
│ Selection: 2                                      │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Screen 3: Select Models to Benchmark              │
│ ───────────────────────────────────────────────────│
│ Ollama (LLM only):                               │
│  1. llama2:7b (LLM)                              │
│  2. neural-chat:7b (LLM)                         │
│                                                  │
│ HuggingFace (LLM only):                          │
│  3. meta-llama/Llama-2-7b (LLM)                  │
│  4. mistralai/Mistral-7B-v0.1 (LLM)             │
│                                                  │
│ GGUF (LLM only):                                │
│  5. mistral-7b.gguf (LLM)                       │
│ ───────────────────────────────────────────────────│
│ Select models (e.g., "1 3 5"): 1 3              │
│ Selected: 2 models (llama2:7b, Llama-2-7b)      │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Screen 4: Select Endpoints to Test                │
│ ───────────────────────────────────────────────────│
│ For llama2:7b (LLM):                             │
│  ☑ [1] Text/Chat (text generation)               │
│                                                  │
│ For Llama-2-7b (LLM):                            │
│  ☑ [2] Text/Chat (text generation)               │
│                                                  │
│ Selected endpoints: Text (for all models)        │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Screen 5: Configure Test Data                     │
│ ───────────────────────────────────────────────────│
│                                                  │
│ [For LLM endpoints]                             │
│ ☑ [1] Use Default Prompts                       │
│ ☐ [2] Use Custom Prompts                        │
│ ☐ [3] Both (static + custom)                    │
│                                                  │
│ [For VLM endpoints]                             │
│ Step 1: Select Image Source                     │
│ ─────────────────────────────────────────────────│
│ ☑ [1] Use Default Images                        │
│    └─ src/benchmarking/datasets/images/{endpoint}/│
│ ☐ [2] Use Custom Image Directory                │
│    └─ Enter directory path:                     │
│       [/path/to/custom/images]                  │
│                                                  │
│ Step 2: Select Prompt Strategy                  │
│ ─────────────────────────────────────────────────│
│ ☑ [1] Same Prompt for All 10 Images             │
│    └─ Enter prompt (copy/paste OK, multiline):  │
│       [What is in this image?]                  │
│                                                  │
│ ☐ [2] Different Prompt per Image                │
│    └─ Will show each image path, enter prompt   │
│       Image 1: /path/to/image_1.jpg             │
│       [Prompt: What is in this image?]          │
│       Image 2: /path/to/image_2.jpg             │
│       [Prompt: Describe the objects...]         │
│       ... (and so on for all images)            │
│                                                  │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Screen 6: Benchmark Configuration                 │
│ ───────────────────────────────────────────────────│
│ Number of Runs: 10 (per endpoint)                │
│   └─ 2 warmup + 8 counted (warmup shown, marked)│
│                                                  │
│ Model Parameters:                               │
│  1. [✓] Use Default Parameters (all models)    │
│  2. [ ] Configure Parameters (bulk edit)        │
│  3. [ ] Configure Per Model (individual)        │
│                                                  │
│ Current Defaults:                               │
│  temperature: 0.7                               │
│  top_p: 0.9                                     │
│  top_k: 50                                      │
│  seed: 42                                       │
│  max_tokens: auto                               │
│  repeat_penalty: 1.0                            │
│                                                  │
│ Options:                                        │
│  [e] Edit (Option 2: applies to all models)    │
│  [p] Per Model (Option 3: different per model) │
│  [c] Continue with defaults                     │
│                                                  │
│ Selection: c                                    │
│                                                  │
│ Estimated Time: ~8 minutes                      │
│ Save Results: Yes (CSV + graphs + metadata)    │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Screen 7: Foreground vs Background Selection      │
│ ───────────────────────────────────────────────────│
│ 1. [▶️] Run Now (Foreground)                      │
│    └─ Watch real-time progress + results        │
│    └─ Results shown when complete               │
│                                                  │
│ 2. [⏲️] Run in Background                        │
│    └─ Continue using app                        │
│    └─ Monitor with /background command          │
│                                                  │
│ Selection: 1                                    │
└─────────────────────────────────────────────────────┘
                    ↓
           START BENCHMARK RUN
```

### Foreground Display (Real-time)

**During Each Inference:**
```
╔═════════════════════════════════════════════════════╗
║   Speed & Throughput Suite - Running               ║
║   Models: llama2:7b, Llama-2-7b | LLM              ║
╠═════════════════════════════════════════════════════╣
║                                                     ║
║  [████████████░░░░░░░░░░░░░░░░░░░░░░░░░] 35%      ║
║                                                     ║
║  ┌─ Current Model: llama2:7b ─────────────────────┐║
║  │ Endpoint: Text                                  ││
║  │ Run: 5/10 (warmup runs: 2)                      ││
║  │                                                 ││
║  │ Current Inference Results:                      ││
║  │ ├─ Input: "Explain quantum computing..."       ││
║  │ ├─ ⏱️  Latency: 1234 ms                         ││
║  │ ├─ 📊 Tokens Generated: 145 tokens              ││
║  │ ├─ ⚡ Throughput: 117.5 tokens/sec             ││
║  │ ├─ 🚀 Time-to-First-Token: 105 ms             ││
║  │ ├─ 🖥️  CPU Usage: 42.3%                        ││
║  │ ├─ 🎮 GPU Usage: 68.5%, Memory: 2.8GB          ││
║  │ └─ 🌡️  Temp: 58°C (↑6°C from baseline)        ││
║  │                                                 ││
║  │ Running Averages (Runs 1-4):                   ││
║  │ ├─ Avg Latency: 1205 ms (σ=45ms)              ││
║  │ ├─ Avg Tokens/sec: 120.2 (σ=2.1)              ││
║  │ └─ Avg 1st Token: 102 ms (σ=3ms)              ││
║  └─────────────────────────────────────────────────┘║
║                                                     ║
║  Progress Summary:                                  ║
║  ├─ ✓ llama2:7b - Text (completed)                 ║
║  │   └─ Avg Latency: 1205ms                        ║
║  │   └─ Avg Tokens/sec: 120.2                      ║
║  │                                                 ║
║  └─ ⏳ Llama-2-7b - Text (5/10 runs)               ║
║     └─ Current latency: 1234ms                     ║
║                                                     ║
║  Time Elapsed: 3m 42s | Remaining: ~4m 15s        ║
║  Press Ctrl+C to cancel                            ║
║                                                     ║
╚═════════════════════════════════════════════════════╝
```

### Results Display (After Completion - Foreground)

**Summary View:**
```
╔═════════════════════════════════════════════════════╗
║   Benchmark Results - Speed & Throughput Suite      ║
║   Models: llama2:7b, Llama-2-7b | LLM              ║
║   Date: 2025-01-20 14:45:30 | Duration: 7m 52s    ║
╠═════════════════════════════════════════════════════╣
║                                                     ║
║  📊 RESULTS SUMMARY                                 ║
║  ─────────────────────────────────────────────────  ║
║                                                     ║
║  llama2:7b (Ollama):                               ║
║  ├─ Endpoint: Text                                 ║
║  ├─ Runs: 10 (2 warmup, 8 counted)                ║
║  ├─ Avg Latency: 1205 ms (σ=45ms)                ║
║  ├─ Avg Tokens/sec: 120.2 (σ=2.1)                ║
║  ├─ Avg 1st Token: 102 ms (σ=3ms)                ║
║  └─ Min/Max Latency: 1150ms / 1280ms             ║
║                                                     ║
║  Llama-2-7b (HuggingFace):                        ║
║  ├─ Endpoint: Text                                 ║
║  ├─ Runs: 10 (2 warmup, 8 counted)                ║
║  ├─ Avg Latency: 1345 ms (σ=62ms)                ║
║  ├─ Avg Tokens/sec: 108.5 (σ=3.5)                ║
║  ├─ Avg 1st Token: 125 ms (σ=5ms)                ║
║  └─ Min/Max Latency: 1260ms / 1450ms             ║
║                                                     ║
║  1. [📈] View Detailed Tables                       ║
║  2. [📊] View Graphs & Visualizations               ║
║  3. [💾] Export to CSV (with graphs & data)        ║
║  4. [🔄] Compare with Previous Run                  ║
║  5. [↩️] Back to Suite Selection                    ║
║                                                     ║
╚═════════════════════════════════════════════════════╝
```

**Option 1: Detailed Tables View**
```
╔═════════════════════════════════════════════════════╗
║   Detailed Results Table - llama2:7b               ║
╠═════════════════════════════════════════════════════╣
║                                                     ║
║  Run │ Latency │ Tokens │ T/sec │ 1stToken │ Temp ║
║  ────┼─────────┼────────┼───────┼──────────┼──────║
║   1  │ 1180ms  │  148   │ 125.4 │  105ms   │ 58°C ║
║   2  │ 1195ms  │  146   │ 122.1 │  103ms   │ 59°C ║
║   3  │ 1215ms  │  144   │ 118.5 │  108ms   │ 60°C ║
║   4  │ 1240ms  │  142   │ 114.5 │  110ms   │ 61°C ║
║  ...                                              ║
║  10  │ 1260ms  │  140   │ 111.1 │  107ms   │ 62°C ║
║  ────┼─────────┼────────┼───────┼──────────┼──────║
║  Avg │ 1205ms  │  144.5 │ 120.2 │  102ms   │ 60°C ║
║  σ   │  45ms   │ 3.2    │  2.1  │   3ms    │ 1.2°C║
║                                                     ║
║ [b] Back | [↩️] Return to Results Summary          ║
║                                                     ║
╚═════════════════════════════════════════════════════╝
```

**Option 3: Export to CSV**
```
File: benchmark_speed_2025-01-20_144530.csv
Saved to: results/benchmarks/

Contents:
- Full detailed results table (all runs, all metrics)
- Per-endpoint statistics
- Graphs (PNG) in same directory
- Summary JSON metadata

Option to:
[✓] Save results | [↩️] Back
```

### Background Monitoring (/background command)

**Task List View:**
```
╔═════════════════════════════════════════════════════╗
║   Background Tasks Monitor                         ║
╠═════════════════════════════════════════════════════╣
║                                                     ║
║  🎯 QUANTIZATION TASKS (1)                         ║
║  └─ [1] llama-2-7b → 4-bit (35% done, ~2m remain) ║
║                                                     ║
║  📊 BENCHMARKING TASKS (1)    ← NEW!               ║
║  └─ [2] Speed Suite (llama2:7b) (55% done, ~3m)   ║
║                                                     ║
║  Select task to view details [1-2] or 'q' quit:  ║
║  _____________________________________________     ║
║                                                     ║
╚═════════════════════════════════════════════════════╝
```

**Detailed Benchmark Task View (Select 2):**
```
╔═════════════════════════════════════════════════════╗
║   Benchmark Task Details - Speed Suite             ║
╠═════════════════════════════════════════════════════╣
║                                                     ║
║  Configuration:                                     ║
║  ├─ Suite: Speed & Throughput                      ║
║  ├─ Models: llama2:7b (1 of 1)                     ║
║  ├─ Endpoints: Text (1 of 1)                       ║
║  ├─ Started: 14:32:10                              ║
║  └─ Elapsed: 4m 15s                                ║
║                                                     ║
║  [████████████░░░░░░░░░░░░░░░░░░] 55%             ║
║                                                     ║
║  Current Progress:                                  ║
║  ├─ Model: llama2:7b                               ║
║  ├─ Endpoint: Text                                 ║
║  ├─ Run: 5/10 (warmup: 2)                          ║
║  │                                                 ║
║  │ System Metrics (Real-time):                     ║
║  │ ├─ CPU Usage: 45.2%                            ║
║  │ ├─ GPU Usage: 72.3%, Memory: 3.1GB             ║
║  │ ├─ System Memory: 8.2GB / 16GB (51%)           ║
║  │ ├─ Temperature: 62°C                            ║
║  │ └─ Throttling: None                             ║
║  │                                                 ║
║  │ Current Inference:                              ║
║  │ ├─ Latency: 1234ms                             ║
║  │ ├─ Tokens/sec: 118.3                           ║
║  │ └─ Avg so far: 1205ms, 120.2 t/s               ║
║                                                     ║
║  Estimated Completion: 14:40:45 (~4m 30s remain) ║
║                                                     ║
║  [↩️] Back | [q] Quit                              ║
║                                                     ║
╚═════════════════════════════════════════════════════╝
```

---

## Suite 2: Speed & Throughput Analysis

### Purpose
Measure inference speed and generation throughput. Answer: "How fast is this model?"

### Metrics Collected

#### Primary Metrics (per inference run)
```
1. Total Latency (ms)
   └─ Time from input submission to complete output received

2. Time-to-First-Token (ms)
   └─ Latency before first output token appears
   └─ Indicates model responsiveness

3. Tokens Per Second (tokens/sec)
   └─ For LLM/VLM text output: tokens_generated / total_latency_seconds
   └─ For Detection/Pointing: detections_per_second or points_per_second

4. Generation Rate (tokens/sec after first token)
   └─ (tokens_after_first) / (total_latency - first_token_latency)
```

#### Per-Endpoint Behavior

**LLM - Text Endpoint:**
```python
Input: "What is machine learning?"
Provider: Ollama (llama2:7b)

Metrics:
├─ Total Latency: 1500 ms
├─ First Token: 120 ms
├─ Tokens Generated: 150
├─ Throughput: 150 tokens / 1.5 sec = 100 tokens/sec
├─ Generation Rate: (150-1) tokens / (1500-120)ms = 149/1380ms = 108 tokens/sec
└─ Variance: σ of latencies across 10 runs

Output Captured:
{
  "model_id": "ollama/llama2:7b",
  "provider": "ollama",
  "endpoint": "text",
  "latency_ms": 1500,
  "first_token_ms": 120,
  "tokens_generated": 150,
  "tokens_per_second": 100.0,
  "generation_rate_tokens_per_sec": 108.0,
  "input": "What is machine learning?",
  "warmup_run": false
}
```

**VLM - QA Endpoint:**
```python
Input: Image + "What's in this image?"
Provider: HuggingFace (llava-hf/llava-1.5-7b)

Metrics:
├─ Total Latency: 1250 ms (includes image encoding)
├─ First Token: 180 ms (higher due to image processing)
├─ Tokens Generated: 85
├─ Throughput: 85 tokens / 1.25 sec = 68 tokens/sec
├─ Image Overhead: ~180ms estimated
└─ Variance: σ of latencies across 10 runs

Output Captured:
{
  "model_id": "huggingface/llava-hf/llava-1.5-7b",
  "provider": "huggingface",
  "endpoint": "qa",
  "image_path": "/path/to/test/image.jpg",
  "latency_ms": 1250,
  "first_token_ms": 180,
  "tokens_generated": 85,
  "tokens_per_second": 68.0,
  "generation_rate_tokens_per_sec": 62.5,
  "input": "What's in this image?",
  "warmup_run": false
}
```

**VLM - Caption Endpoint:**
```python
Input: Image (no explicit prompt)
Provider: HuggingFace (llava-hf/llava-1.5-7b)

Metrics:
├─ Total Latency: 900 ms
├─ First Token: 150 ms
├─ Tokens Generated: 50
├─ Throughput: 50 tokens / 0.9 sec = 55.5 tokens/sec
└─ Variance: σ of latencies

Output Captured:
{
  "model_id": "huggingface/llava-hf/llava-1.5-7b",
  "provider": "huggingface",
  "endpoint": "caption",
  "image_path": "/path/to/test/image.jpg",
  "latency_ms": 900,
  "first_token_ms": 150,
  "tokens_generated": 50,
  "tokens_per_second": 55.5,
  "generation_rate_tokens_per_sec": 50.0,
  "input": "Describe this image", # Default prompt
  "warmup_run": false
}
```

**VLM - Detection Endpoint:**
```python
Input: Image + "Find cars"
Provider: HuggingFace (custom detection model)

Metrics:
├─ Total Latency: 800 ms
├─ Detections: 5 objects found
├─ Detections Per Second: 5 / 0.8 = 6.25 detections/sec
└─ Note: No tokens/sec - different type of output

Output Captured:
{
  "model_id": "huggingface/custom-detection",
  "provider": "huggingface",
  "endpoint": "detect",
  "image_path": "/path/to/test/image.jpg",
  "latency_ms": 800,
  "detections_count": 5,
  "detections_per_second": 6.25,
  "tokens_applicable": false,
  "input": "Find cars",
  "warmup_run": false
}
```

**VLM - Pointing Endpoint:**
```python
Input: Image + "Locate the dog"
Provider: HuggingFace (pointing model)

Metrics:
├─ Total Latency: 750 ms
├─ Points Generated: 1
├─ Points Per Second: 1 / 0.75 = 1.33 points/sec
└─ Note: Usually single point output

Output Captured:
{
  "model_id": "huggingface/custom-pointing",
  "provider": "huggingface",
  "endpoint": "point",
  "image_path": "/path/to/test/image.jpg",
  "latency_ms": 750,
  "points_generated": 1,
  "points_per_second": 1.33,
  "tokens_applicable": false,
  "input": "Locate the dog",
  "warmup_run": false
}
```

### Test Data Requirements

#### Suite 2 Default Test Data Structure

**LLM Prompts (10 different prompts for 10 runs - hardcoded defaults):**
```python
DEFAULT_LLM_PROMPTS = [
    "What is machine learning?",
    "Explain quantum computing briefly.",
    "How does photosynthesis work?",
    "What are the benefits of renewable energy?",
    "Describe the water cycle.",
    "What is artificial intelligence?",
    "Explain deep learning in simple terms.",
    "How do neural networks work?",
    "What is natural language processing?",
    "Describe the Internet of Things."
]
```

**VLM Images (10 different test images per endpoint - default directory):**
```
src/benchmarking/datasets/images/
├── qa/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── ... (up to 10+ images)
├── caption/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── ... (up to 10+ images)
├── detect/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── ... (up to 10+ images)
└── point/
    ├── image_001.jpg
    ├── image_002.jpg
    ├── ... (up to 10+ images)
```

**VLM Prompts (per endpoint type):**
```python
DEFAULT_VLM_PROMPTS = {
    "qa": [
        "What is in this image?",
        "Describe the main objects you see.",
        "What colors dominate this image?",
        "Is there any text visible in this image?",
        "What is the main subject?",
        "Describe the background.",
        "What is the lighting condition?",
        "Are there people in this image?",
        "What objects can you identify?",
        "Describe the composition of this image."
    ],
    "caption": [
        # No explicit prompt needed - model generates caption
        # Use model's default caption generation
    ],
    "detect": [
        "Find all objects.",
        "Detect people.",
        "Locate vehicles.",
        "Identify animals.",
        "Find text regions.",
        "Detect faces.",
        "Locate buildings.",
        "Find electronic devices.",
        "Detect all animals.",
        "Find objects of interest."
    ],
    "point": [
        "Where is the main object?",
        "Locate the person.",
        "Point to the largest object.",
        "Find the focal point.",
        "Where is the subject?",
        "Locate the center of attention.",
        "Point to the vehicle.",
        "Find the animal.",
        "Locate the object.",
        "Where is the object of interest?"
    ]
}
```

#### Configuration Menu Options

**For LLM Endpoints - Custom Prompt Input:**
Using `professional_prompt.get_text()` from `src/cli/text_input.py`:
1. User selects "Use Custom Prompts"
2. System shows prompt entry for EACH of the 10 runs:
   ```
   Run 1 - Enter prompt:
   [Professional text input with multiline support]

   Features:
   ├─ Copy/Paste support: Paste longer prompts directly
   ├─ Arrow keys: Navigate left/right within text (Ctrl+A/E for line start/end)
   ├─ Ctrl+J: Add newline (multiline support)
   ├─ Up/Down: Cycle through prompt history
   └─ Proper parsing: All input preserved exactly as entered

   Run 2 - Enter prompt:
   [Professional text input]
   ... (all 10 runs)
   ```
3. All 10 prompts collected and used for all models
4. Option to combine with defaults (run some with defaults, some with custom)

**For VLM Endpoints - Image Source & Prompt Strategy:**

**Step 1: Select Image Directory**
```
Option A: Use Default Images
  └─ src/benchmarking/datasets/images/{endpoint}/
     └─ Automatically uses 10+ images from endpoint-specific folder

Option B: Use Custom Image Directory
  └─ User enters path: [Professional text input]
     └─ Copy/paste directory path, arrow key navigation
     └─ Any number of images (10+, 100+, etc. - no minimum)
     └─ System loads all valid image formats (.jpg, .png, .webp, .bmp)
```

**Step 2: Choose Prompt Strategy (for QA, Detect, Point endpoints)**
```
Option 1: Same Prompt for All Images
  └─ User enters ONE prompt:
     [Professional text input with multiline support]

     Features:
     ├─ Copy/Paste: Paste longer prompts
     ├─ Arrow keys: Navigate text cursor
     ├─ Ctrl+J: Multiline prompts supported
     └─ Proper parsing: All formatting preserved

     Applied to all images: image_1, image_2, ..., image_N

Option 2: Different Prompt per Image
  └─ For EACH image in directory:
     Image 1: /path/to/image_001.jpg
     Enter prompt for this image:
     [Professional text input]

     Image 2: /path/to/image_002.jpg
     Enter prompt for this image:
     [Professional text input]

     ... (repeats for all images in directory)

     Result: [image_1, prompt_1] → [image_2, prompt_2] → ... → [image_N, prompt_N]
```

**Caption Endpoint Special Case:**
```
No prompt needed - uses model's default caption generation
├─ All images processed with same default prompt
└─ Caption endpoint generates its own captions
```

**Fair Comparison Guarantee:**
- All models in a single run receive IDENTICAL inputs
- Same 10 LLM prompts for all LLM models
- Same images for all VLM models (directory shared)
- Same prompts (if applicable) for all VLM models
- **This ensures true, valid performance comparison**

### Hardcoded Default Parameters

```python
DEFAULT_INFERENCE_PARAMS = {
    "temperature": 0.7,      # Balanced creativity
    "top_p": 0.9,            # Nucleus sampling
    "top_k": 50,             # Top-K filtering
    "seed": 42,              # Reproducibility
    "max_tokens": "auto",    # Let model decide
    "repeat_penalty": 1.0    # No repetition penalty
}

BENCHMARK_CONFIG = {
    "num_runs": 10,                      # Per endpoint: 2 warmup + 8 counted
    "warmup_runs": 2,                    # First 2 runs - marked in results, shown in detailed table
    "timeout_seconds": 120,              # Per inference: 120 seconds
    "configurable": True                 # All these values configurable in config menu
}

### Configuration Menu for Suite 2

**Number of Runs:**
- Default: 10 (2 warmup + 8 counted)
- Configurable: User can set any value
- Display: Show breakdown (e.g., "2 warmup + 8 counted = 10 total")

**Model Parameters:**
- Option 1: Use Default Parameters (for all models)
- Option 2: Configure Parameters (single set applies to all models)
- Option 3: Configure Per Model (different params per model)

Where parameters are: temperature, top_p, top_k, seed, max_tokens, repeat_penalty

**Test Data (already detailed above):**
- LLM: Default prompts vs custom
- VLM: Default images vs custom, plus prompt strategy choice
```

### Result Structure (Per Suite Run)

```python
SuiteResult {
    "suite_name": "Speed & Throughput",
    "timestamp": "2025-01-20T14:45:30Z",
    "duration_seconds": 472,

    "configuration": {
        "models": ["ollama/llama2:7b", "huggingface/llava-hf/llava-1.5-7b"],
        "endpoints": ["text", "qa"],
        "num_runs": 10,
        "warmup_runs": 2,
        "test_data_source": "static_defaults",
        "parameters": {...}
    },

    "results": [
        {
            "model_id": "ollama/llama2:7b",
            "provider": "ollama",
            "model_type": "llm",
            "endpoints": [
                {
                    "endpoint_name": "text",
                    "runs": [
                        # Individual run data (detailed)
                        {
                            "run_number": 1,
                            "warmup": false,
                            "latency_ms": 1500,
                            "first_token_ms": 120,
                            "tokens_generated": 150,
                            "tokens_per_second": 100.0,
                            "input_text": "What is machine learning?"
                        },
                        # ... more runs
                    ],
                    "statistics": {
                        "avg_latency_ms": 1205,
                        "std_dev_latency_ms": 45,
                        "min_latency_ms": 1150,
                        "max_latency_ms": 1280,
                        "p50_latency_ms": 1210,
                        "p95_latency_ms": 1265,

                        "avg_tokens_per_sec": 120.2,
                        "std_dev_tokens_per_sec": 2.1,

                        "avg_first_token_ms": 102,
                        "std_dev_first_token_ms": 3
                    }
                }
            ]
        },
        # ... more models
    ],

    "summary": {
        "total_inferences": 20,  # 2 models * 1 endpoint * 10 runs
        "total_tokens_generated": 3000,
        "total_time_seconds": 472,

        "fastest_model": "ollama/llama2:7b",
        "fastest_latency_ms": 1150,

        "highest_throughput_model": "ollama/llama2:7b",
        "highest_throughput_tokens_per_sec": 125.4
    }
}
```

### Error Handling

```python
# Timeout (>120 seconds for single inference)
├─ Mark run as TIMEOUT in results
├─ Log error details
├─ Do NOT count in statistics
└─ Continue with next run (no interruption)

# Out of Memory (OOM)
├─ Mark as OOM in results
├─ Attempt to unload/offload model from memory
├─ Continue with next model (sequential loading)
└─ Note: OOM ends testing for THIS model, but continues with other models

# Model Load Failure
├─ Mark as LOAD_ERROR
├─ Log error details
├─ Skip remaining runs for this model
└─ Continue with next model

# Model Not Found / Invalid Provider
├─ Mark as NOT_FOUND
├─ Log error
└─ Continue with other models

# Invalid Input / Prompt Error
├─ Log error details
├─ Use fallback/default input
└─ Retry current run with safe input

# Partial Failure (some runs fail, some succeed)
├─ Include all runs in detailed results with status
├─ Calculate statistics excluding failed runs
├─ Show failure count in summary
├─ Mark model as "Partially Completed" if applicable
└─ Highlight errors in results viewer
```

### Display Format (Real-time + Results)

**Per-Inference Display (running):**
```
Run 5/10: Text Endpoint
Input: "What is machine learning?"
⏱️  Latency: 1234 ms
📊 Tokens: 145
⚡ Throughput: 117.5 tokens/sec
🚀 First Token: 105 ms
Avg so far: 1205ms, 120.2 t/s, σ=45ms
```

**Results Summary Table:**
```
Model: llama2:7b (Ollama, LLM)
Endpoint: Text
Runs: 10 (2 warmup, 8 counted)

Metric             │ Value         │ Std Dev
─────────────────────────────────────────────
Avg Latency        │ 1205 ms       │ ±45 ms
Avg Throughput     │ 120.2 t/s     │ ±2.1 t/s
Avg 1st Token      │ 102 ms        │ ±3 ms
Min / Max Latency  │ 1150 / 1280ms │ -
P95 Latency        │ 1265 ms       │ -
```

### Suite 2 CSV Export Format

**Wide Format with All Metrics and Full Raw Responses**

**File Name:** `benchmark_speed_TIMESTAMP.csv`

**CSV Columns (in order):**
```
Part 1: Model & Run Metadata
  model_id | provider | model_type | endpoint_name | run_number | is_warmup | timestamp

Part 2: Timing Metrics (Core Suite 2 Data)
  latency_ms | first_token_ms | tokens_generated | tokens_per_second |
  generation_rate_tokens_per_sec | response_length_chars

Part 3: Input & Output
  input_text | raw_response (complete, untruncated, all formatting preserved)

Part 4: Per-Run Statistics
  latency_vs_avg_percent | latency_std_dev_so_far | throughput_std_dev_so_far
```

**Example Rows:**

LLM Text Endpoint:
```
ollama/llama2:7b | ollama | llm | text | 1 | true | 2025-01-20T14:30:05Z | 1500 | 120 | 150 | 100.0 | 108.0 | 2847 | "What is machine learning?" | "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It involves developing algorithms..."
```

VLM QA Endpoint:
```
huggingface/llava | huggingface | vlm | qa | 3 | false | 2025-01-20T14:31:15Z | 1250 | 180 | 85 | 68.0 | 62.5 | 1956 | "/benchmark/datasets/images/qa/image_003.jpg + What's in this image?" | "This image shows a person sitting on a bench in a park. The person is wearing casual clothing. In the background, there are trees and a playground..."
```

**Key Features:**
- Wide format: All metrics as columns for easy spreadsheet analysis
- Full raw responses: Complete, untruncated text with all original formatting preserved in CSV cells
- All runs included: Both warmup (is_warmup=true) and counted runs (is_warmup=false)
- Separated statistics: Individual run data + aggregated statistics in separate section
- Timestamps: Precise timing for each run
- Copy-paste ready: Can be immediately used in Excel, pandas, R, etc.

**Additional Sections in CSV:**
```
After all run data, append:

=== STATISTICS SUMMARY ===
Metric,Counted Runs,All Runs,Mean,Std Dev,Min,Max,P50,P95
Latency (ms),8,10,1205,45,1150,1280,1210,1265
Tokens/Sec,8,10,120.2,2.1,115.5,125.4,120.5,123.2
First Token (ms),8,10,102,3,98,110,102,108
Generation Rate,8,10,108.5,3.2,103.1,114.5,108.8,111.9
...
```

### Suite 2 Graphs and Visualizations

**Graphs Generated (Option B - Comprehensive Statistical Analysis):**
1. **Latency Comparison** - Bar chart (model vs avg latency)
2. **Latency Distribution** - Box plot per model (min, Q1, median, Q3, max)
3. **Latency Time Series** - Line chart showing latency across all 10 runs per model
4. **Throughput Comparison** - Bar chart (model vs tokens/sec)
5. **Throughput Trends** - Line chart showing tokens/sec across all 10 runs per model
6. **First Token Time Comparison** - Bar chart (model vs first token latency)
7. **Statistical Summary** - Table showing all statistics per model (mean, std dev, percentiles)
8. **Performance Ranking** - Sorted ranking of models by latency and throughput

**Format:** PNG images saved to results directory with timestamp
**Interactivity:** User can view all graphs in results viewer or export with CSV

---

## Suite 3: 💾 Resource Efficiency

### Purpose
Measure resource consumption during inference. Answer: "How much CPU, GPU, memory, and power does this model use?"

### Metrics Collected

#### Primary Metrics (per inference run)
```
1. CPU Usage (%)
   └─ Average CPU % during inference
   └─ Peak CPU %
   └─ Per-core utilization (if available)

2. Memory (RAM)
   └─ Process memory (MB) - Python process footprint
   └─ System RAM used (GB)
   └─ System RAM peak (GB)
   └─ Memory % of total system RAM

3. GPU Usage (if available)
   └─ GPU utilization %
   └─ GPU memory used (GB)
   └─ GPU memory peak (GB)
   └─ GPU temperature (°C) if available

4. Power Draw (NVIDIA GPU only)
   └─ Average watts during inference
   └─ Peak watts
   └─ Energy per token (millijoules)

5. Disk I/O
   └─ MB/sec read speed during model loading (first load)
   └─ Total MB read during model loading
   └─ Disk I/O during inference (if any swapping occurs)

6. Device-Aware Efficiency Ratios (based on SystemSpecs device type)
   └─ Memory per token generated (MB/token)
   └─ Power per token (watts/token) - NVIDIA only
   └─ CPU:GPU ratio - shows which is bottleneck
   └─ Device-specific efficiency metrics (MPS-aware, TPU-aware, etc.)
```

#### Baseline & Cleanup Measurements (transparent)
```
Baseline (before any inference):
├─ System CPU usage
├─ System memory usage
├─ GPU memory (if available)
└─ Temperature (if available)

Cleanup (after all inferences complete):
├─ System memory usage
├─ GPU memory usage
└─ Check if resources were properly freed
```

#### Per-Endpoint Behavior
Each endpoint measured separately. Example for LLM Text:
```
Model: ollama/llama2:7b
Endpoint: text
Runs: 7 (2 warmup, 5 counted)

Baseline:
├─ System Memory: 7.2 GB / 16 GB (45%)
├─ GPU Memory: Not available
└─ CPU: 15% (idle)

Run 1:
├─ Duration: 1234 ms
├─ Tokens: 150
├─ CPU Avg: 75.3%, Peak: 88.2%
├─ Memory Avg: 8.2 GB (51%), Peak: 8.5 GB
├─ GPU Memory: 3.1 GB / 8 GB, Temp: 58°C
├─ Disk I/O: 0 MB (model already loaded)
├─ Power (NVIDIA): Avg 120W, Peak 150W
└─ Efficiency: 20.7 MB/token, 0.8 W/token, CPU:GPU ratio 1.15

Cleanup (after all 7 runs):
├─ System Memory: 8.2 GB (freed 0 GB - possible leak)
├─ GPU Memory: 3.1 GB (not freed)
└─ Temperature: 62°C
```

### Test Data Requirements

Same as Suite 2:
- Use **DEFAULT_LLM_PROMPTS** for LLM endpoints
- Use **DEFAULT_TEST_IMAGE_PATH** for VLM endpoints
- Static defaults or custom user-provided (configurable)

### Configuration & Defaults

```
RESOURCE_SUITE_CONFIG = {
    "num_runs": 7,              # User-configurable (default: 7)
    "warmup_runs": 2,           # Excluded from final statistics
    "counted_runs": 5,          # Used for averaging

    "measure_baseline": true,   # Before first inference
    "measure_cleanup": true,    # After all inferences
    "monitoring_interval_ms": 500,  # How often to sample metrics

    "inference_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "seed": 42,
        "max_tokens": "auto",
        "repeat_penalty": 1.0
    }
}
```

### Result Structure (Per Suite Run)

Each endpoint produces:
```
Baseline Snapshot:
├─ timestamp
├─ cpu_percent
├─ system_memory_gb
├─ gpu_memory_gb (if available)
└─ temperature_celsius (if available)

Per-Run Metrics (5 runs counted):
├─ run_number
├─ latency_ms
├─ tokens_generated
├─ cpu_metrics: {avg_percent, peak_percent, per_core[]}
├─ memory_metrics: {process_mb, system_ram_gb, peak_gb, percent_of_total}
├─ gpu_metrics: {utilization_percent, memory_gb, peak_gb, temperature_celsius}
├─ power_metrics: {avg_watts, peak_watts, energy_per_token_mj} (NVIDIA only)
├─ disk_io_metrics: {mb_per_sec, total_mb_read} (model loading only)
└─ efficiency_metrics: {memory_per_token_mb, power_per_token_watts, cpu_gpu_ratio}

Aggregated Statistics (across 5 counted runs):
├─ cpu: {avg_percent, std_dev, peak, min}
├─ memory: {avg_process_mb, avg_system_ram_gb, peak_gb, std_dev}
├─ gpu: {avg_utilization_percent, avg_memory_gb, peak_gb, avg_temp, max_temp}
├─ power: {avg_watts, peak_watts, avg_energy_per_token} (NVIDIA only)
├─ disk_io: {avg_mb_per_sec, total_mb}
├─ efficiency: {avg_memory_per_token, avg_power_per_token, cpu_gpu_ratio_avg}
└─ resource_trends: {leaked_memory_gb, temperature_increase, throttling_detected}

Cleanup Snapshot:
├─ system_memory_gb
├─ gpu_memory_gb
├─ delta_from_baseline (what was/wasn't freed)
└─ leak_detected (boolean)
```

### Results Display & Export

**Transparency:** All measurements (baseline, per-run, cleanup) visible to user

**Display Format:**
```
In Real-time (during benchmark):
├─ Run X/7: Endpoint name
├─ Current metrics: CPU %, Memory, GPU %, Temp
└─ Running averages so far

After Completion:
├─ Summary table (all metrics, avg, peak, std dev)
├─ Comparative insights (which endpoint used most resources, ranking)
├─ Graphs/visualizations (CPU trend, Memory trend, GPU usage, etc.)
├─ Baseline vs Cleanup comparison (what was freed)
└─ Single-click "Export to CSV" (with all data + graphs)
```

### Error Handling

```
GPU not available:
├─ Set gpu_metrics to empty
└─ Continue with CPU/Memory metrics

Disk I/O not measurable:
├─ Mark as unavailable
└─ Continue with other metrics

Memory leak detected:
├─ Highlight in results
└─ Recommend investigation

Thermal throttling:
├─ Mark in results
└─ Flag as potential performance impact

Power metrics unavailable (non-NVIDIA):
├─ Skip power calculations
└─ Continue with efficiency ratios
```

### Suite 3 CSV Export Format

**Wide Format with All Resource Metrics**

**File Name:** `benchmark_resources_TIMESTAMP.csv`

**CSV Columns (in order):**
```
Part 1: Model & Run Metadata
  model_id | provider | model_type | endpoint_name | run_number | is_warmup | timestamp

Part 2: CPU Metrics
  cpu_avg_percent | cpu_peak_percent | cpu_per_core_json (if available)

Part 3: Memory Metrics (System)
  memory_system_gb | memory_system_percent | memory_peak_gb | memory_process_mb |
  memory_freed_gb_after_cleanup | memory_leaked_gb

Part 4: GPU Metrics (if available)
  gpu_utilization_percent | gpu_memory_gb | gpu_memory_peak_gb | gpu_temperature_celsius

Part 5: Power Metrics (NVIDIA only)
  power_avg_watts | power_peak_watts | energy_per_token_millijoules

Part 6: Disk I/O
  disk_io_mb_per_sec | disk_io_total_mb

Part 7: Efficiency Ratios
  memory_per_token_mb | power_per_token_watts | cpu_gpu_ratio |
  efficiency_score (custom calculation based on device type)

Part 8: Baseline & Cleanup
  baseline_cpu_percent | baseline_memory_gb | baseline_temperature_celsius |
  cleanup_memory_gb | cleanup_gpu_memory_gb | thermal_throttling_detected

Part 9: Input & Inference Time
  input_text | inference_time_ms | latency_ms | tokens_generated
```

**Example Row (LLM Text with GPU):**
```
ollama/llama2:7b | ollama | llm | text | 1 | true | 2025-01-20T14:30:05Z | 75.3 | 88.2 | ... | 8.2 | 51.2 | 8.5 | 245 | 0.1 | 0 | 65.4 | 3.2 | 62.5 | 58 | 120.5 | 150.0 | 0.82 | 20.7 | 0.8 | 1.15 | 0.85 | 75.0 | 8.0 | 52 | 8.3 | 3.2 | false | "What is machine learning?" | 1500 | 150 | 100.0
```

**Key Features:**
- All resource consumption data in single row per inference
- Baseline measurements transparent (shown in columns)
- Cleanup measurements included (memory freed vs leaked)
- Device-aware metrics (GPU fields empty if not available)
- Efficiency ratios computed for each run
- Thermal throttling flagged per run
- Timestamps for temporal analysis

**Additional Sections in CSV:**
```
After all run data, append:

=== RESOURCE STATISTICS (Counted Runs Only) ===
Metric,Mean,Std Dev,Min,Max,Peak
CPU (%), 75.8, 2.1, 73.2, 78.5, 88.2
Memory System (GB), 8.35, 0.15, 8.1, 8.6, 8.6
GPU Util (%), 65.2, 3.1, 60.5, 72.3, 72.3
GPU Memory (GB), 3.25, 0.12, 3.0, 3.5, 3.5
GPU Temp (C), 58.5, 2.3, 54, 62, 62
Power (W), 125.5, 8.3, 115, 150, 150

=== EFFICIENCY SUMMARY ===
Memory per Token (MB), 0.205
Power per Token (W), 0.0084
CPU:GPU Ratio, 1.15
Memory Leak Detected, No
Thermal Throttling, No
Recommendation, Efficient, good thermal margins
```

### Suite 3 Graphs and Visualizations

**Graphs Generated (Maximum Statistical Visualization):**
1. **CPU Usage Trend** - Line chart (CPU % over all runs)
2. **Memory Growth Trend** - Line chart (Memory GB over runs, with baseline/cleanup markers)
3. **Memory per Token** - Bar chart (efficiency comparison if multiple models)
4. **GPU Utilization Over Time** - Line chart (% over runs)
5. **GPU Memory Growth** - Line chart (GB over runs)
6. **GPU Temperature Trend** - Line chart (°C over runs, highlight throttling threshold)
7. **Power Consumption Trend** - Line chart (Watts over runs, NVIDIA only)
8. **CPU vs GPU Activity** - Dual-axis chart (CPU % and GPU % on same plot)
9. **Thermal Performance** - Gauge/dial showing peak temperature vs safe limit
10. **Resource Efficiency Comparison** - Radar/spider chart (CPU%, Memory, GPU%, Temp, Power)
11. **Distribution Charts** - Box plots for CPU, Memory, GPU metrics (showing variance)
12. **Correlation Heatmap** - Shows relationships between CPU, GPU, Memory, Temp, Power
13. **Memory Leak Detection Graph** - Area chart highlighting memory delta baseline-to-cleanup

**Format:** PNG images + interactive HTML if possible
**Directory:** `results/benchmarks/{timestamp}/suite3_graphs/`
**Count:** 12-13 graphs (depending on available hardware)

---

## Suite 4: 🎯 Quality & Consistency

### Purpose
Collect raw model outputs and their consistency across multiple runs. Answer: "What does this model output, and is it consistent?"

### Metrics Collected

#### Primary Metrics (per run/inference)
```
1. Raw Output (Complete, Untruncated)
   └─ Full text response (for Text, QA, Caption endpoints)
   └─ Full detection results with bboxes (for Detection endpoint)
   └─ Full coordinates (for Point endpoint)

2. Inference Time (ms)
   └─ Time taken for that specific inference
   └─ Allows correlation between speed and output quality

3. Detection/Point Results (per image, per model)
   └─ For Detection: Bounding boxes [x1, y1, x2, y2]
   └─ For Point: Coordinates {x, y}
   └─ Annotated images saved for visual verification
```

#### Per-Endpoint Behavior

**Text Endpoints (LLM Text or VLM Text):**
```
Run Configuration:
├─ Default: 10 runs per endpoint
├─ User-configurable in config menu
└─ Same prompt/input each time

Example Output:
Run | Model A (ollama/llama2:7b)      | Model B (huggingface/llama2)     | Inference Time
────────────────────────────────────────────────────────────────────────────────────────
1   | "Machine learning is a subset..." | "ML is a form of artificial..." | 1200ms
2   | "Machine learning is a form..."   | "Machine learning is a subset..." | 1210ms
3   | "ML is a branch of computer..."   | "ML is a type of AI that..."     | 1195ms
... (10 runs total)
```

**Caption Endpoint (VLM):**
```
Configuration:
├─ User provides dataset directory (images)
├─ Default: 2 inferences per image per model
├─ User-configurable in config menu
└─ Image format validation: .jpg, .png, .webp, .bmp only

Example Output:
Image Path       | Model A (ollama/llava)        | Model B (huggingface/llava)      | Inference Time
─────────────────────────────────────────────────────────────────────────────────────────────
dataset/img1.jpg | "A dog playing in a park"     | "A brown dog in grass"           | 950ms, 960ms
dataset/img2.jpg | "A cat sitting on a table"    | "A feline on furniture"          | 880ms, 875ms
dataset/img3.jpg | "A sunset over mountains"     | "Sun setting behind hills"       | 1050ms, 1040ms
... (per image shown with 2 inference times)
```

**QA Endpoint (VLM):**
```
Configuration:
├─ User provides dataset directory (images)
├─ Default: 2 inferences per image per model
├─ User-configurable in config menu
├─ Same question asked for each image
└─ Image format validation: .jpg, .png, .webp, .bmp only

Example Output:
Image Path       | Model A (ollama/llava)        | Model B (huggingface/llava)      | Inference Time
─────────────────────────────────────────────────────────────────────────────────────────────
dataset/img1.jpg | "There is a dog in the image"| "A dog is visible here"          | 1150ms, 1160ms
dataset/img2.jpg | "Yes, a cat is shown"        | "The image shows a cat"          | 1080ms, 1090ms
dataset/img3.jpg | "I see mountains and sunset" | "A sunset over mountain range"   | 1200ms, 1210ms
... (per image, 2 inference times)
```

**Detection Endpoint (VLM):**
```
Configuration:
├─ User provides dataset directory (images)
├─ Default: 2 inferences per image per model
├─ User-configurable in config menu
├─ Same detection prompt ("Find objects", "Detect cars", etc.)
└─ Image format validation: .jpg, .png, .webp, .bmp only

Output Format:
Image Path       | Model A (ollama/llava)              | Model B (huggingface/custom)     | Inference Time
─────────────────────────────────────────────────────────────────────────────────────────────────
dataset/img1.jpg | [100,150,300,350], [400,200,600,400]| [102,152,298,348], [402,202,598,398] | 800ms, 810ms
dataset/img2.jpg | [120,180,420,320]                   | [118,182,418,318]                | 820ms, 815ms
dataset/img3.jpg | (no detections)                     | [250,200,450,400]                | 780ms, 790ms

Plus: Annotated images saved to results/benchmarking/detections/
├─ detection_img1_modelA_run1.jpg
├─ detection_img1_modelA_run2.jpg
├─ detection_img1_modelB_run1.jpg
└─ ... (one annotated image per image per model per run)
```

**Point Endpoint (VLM):**
```
Configuration:
├─ User provides dataset directory (images)
├─ Default: 2 inferences per image per model
├─ User-configurable in config menu
├─ Same pointing prompt ("Where is the dog?", "Locate the car", etc.)
└─ Image format validation: .jpg, .png, .webp, .bmp only

Output Format:
Image Path       | Model A (ollama/llava)   | Model B (huggingface/custom)| Inference Time
──────────────────────────────────────────────────────────────────────────────────
dataset/img1.jpg | x:250, y:350             | x:248, y:352               | 750ms, 760ms
dataset/img2.jpg | x:120, y:180             | x:118, y:182               | 740ms, 745ms
dataset/img3.jpg | x:400, y:250             | x:402, y:248               | 780ms, 775ms

Plus: Annotated images with point markers saved to results/benchmarking/point/
├─ point_img1_modelA_run1.jpg
├─ point_img1_modelA_run2.jpg
├─ point_img1_modelB_run1.jpg
└─ ... (one annotated image per image per model per run)
```

### Test Data Requirements

**Text Endpoints:**
- Use **DEFAULT_LLM_PROMPTS** for LLM Text
- Use **DEFAULT_VLM_PROMPTS["qa"]** for QA
- Use **DEFAULT_VLM_PROMPTS["caption"]** for Caption
- Or custom user-provided prompts

**VLM Endpoints (Caption, QA, Detection, Pointing):**
- **User-provided dataset directory** (required)
- Image format validation: .jpg, .png, .webp, .bmp
- Encouragement message: "More images provide more accurate consistency statistics"
- No maximum limit - user can provide as many images as needed

### Configuration & Defaults

```
QUALITY_SUITE_CONFIG = {
    "text_runs": 10,                # User-configurable (default: 10)
    "vlm_inferences_per_image": 2,  # User-configurable (default: 2)

    "text_prompts": [               # Default prompts for text endpoints
        "What is machine learning?",
        "Explain quantum computing briefly.",
        "How does photosynthesis work?"
    ],

    "vlm_dataset_path": None,       # User must provide for VLM endpoints
    "vlm_dataset_validation": {
        "allowed_formats": [".jpg", ".png", ".webp", ".bmp"],
        "validate_format": true
    },

    "inference_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "seed": 42,
        "max_tokens": "auto",
        "repeat_penalty": 1.0
    }
}
```

### Result Structure (Per Suite Run)

**For Text Endpoints:**
```
Results: {
    "suite_name": "Quality & Consistency",
    "endpoints": [
        {
            "endpoint_name": "text",
            "models": ["ollama/llama2:7b", "huggingface/llama2"],
            "configuration": {
                "num_runs": 10,
                "prompt": "What is machine learning?"
            },

            "runs": [
                {
                    "run_number": 1,
                    "model_outputs": {
                        "ollama/llama2:7b": {
                            "raw_output": "Machine learning is a subset of artificial intelligence...",
                            "inference_time_ms": 1200
                        },
                        "huggingface/llama2": {
                            "raw_output": "ML is a form of artificial intelligence that...",
                            "inference_time_ms": 1210
                        }
                    }
                },
                # ... 10 runs total
            ],

            "summary": {
                "total_runs": 10,
                "models_tested": 2,
                "avg_inference_time_ms": 1205
            }
        }
    ]
}
```

**For VLM Endpoints (Caption, QA, Detection, Point):**
```
Results: {
    "suite_name": "Quality & Consistency",
    "endpoints": [
        {
            "endpoint_name": "detection",
            "dataset_path": "path/to/dataset",
            "models": ["ollama/llava", "huggingface/custom-detect"],
            "configuration": {
                "inferences_per_image": 2,
                "total_images": 5,
                "detection_prompt": "Find cars"
            },

            "images": [
                {
                    "image_path": "dataset/img1.jpg",
                    "inferences": [
                        {
                            "inference_number": 1,
                            "model_outputs": {
                                "ollama/llava": {
                                    "detections": [[100,150,300,350], [400,200,600,400]],
                                    "inference_time_ms": 800,
                                    "annotated_image_path": "results/benchmarking/detections/detection_img1_ollama_1.jpg"
                                },
                                "huggingface/custom-detect": {
                                    "detections": [[102,152,298,348], [402,202,598,398]],
                                    "inference_time_ms": 810,
                                    "annotated_image_path": "results/benchmarking/detections/detection_img1_huggingface_1.jpg"
                                }
                            }
                        },
                        {
                            "inference_number": 2,
                            # ... second inference for same image
                        }
                    ]
                },
                # ... more images
            ],

            "summary": {
                "total_images": 5,
                "total_inferences": 10,  # 5 images × 2 inferences
                "models_tested": 2,
                "avg_inference_time_ms": 805,
                "annotated_images_saved": 10
            }
        }
    ]
}
```

### Results Display & Export

**Transparency:** All raw outputs visible to user (complete, untruncated)

**Live Display (during benchmark):**
```
Run 5/10: Text Endpoint
Input: "What is machine learning?"

Model A Output: "Machine learning is a form of..." (truncated to 4-5 words for readability)
Model B Output: "ML is when computers learn..." (truncated)
Inference Times: Model A: 1200ms, Model B: 1210ms
```

**Final Results:**
```
Summary Table (Text Endpoints):
├─ Run | Model A (complete raw output) | Model B (complete raw output) | Times
├─ 1   | "Machine learning is a subset of artificial intelligence..." | "ML is a form of AI that..." | 1200ms, 1210ms
└─ ... (10 runs)

Summary Table (VLM Endpoints):
├─ Image | Model A (results) | Model B (results) | Times
├─ img1  | [100,150,300,350] | [102,152,298,348] | 800ms, 810ms
└─ ... (per image)

Annotated Images:
├─ results/benchmarking/detections/detection_img1_modelA_run1.jpg
├─ results/benchmarking/detections/detection_img1_modelB_run1.jpg
└─ ... (all annotated images)

CSV Export:
├─ All raw outputs (models as columns, runs/images as rows)
├─ All inference times
├─ All bbox/coordinate data
└─ Links to annotated images
```

### Error Handling

```
Invalid image format (VLM endpoints):
├─ Skip invalid image
├─ Log warning
└─ Continue with valid images

Image loading fails:
├─ Skip that image
├─ Log error
└─ Continue with other images

Dataset directory not found:
├─ Prompt user to provide valid path
└─ Cannot proceed without dataset

No valid images in directory:
├─ Error: "No valid images found"
└─ Cannot proceed

Inference timeout:
├─ Log error
├─ Mark as (error) in results
└─ Continue to next run

Invalid format detected:
├─ Show error: "Only .jpg, .png, .webp, .bmp supported"
├─ List unsupported files
└─ Ask user to remove them
```

### Suite 4 CSV Export Format

**Wide Format with Complete Raw Outputs & Detection/Point Data**

**File Name:** `benchmark_quality_TIMESTAMP.csv`

**CSV Columns (in order):**

**For Text Endpoints (LLM Text, VLM QA, Caption):**
```
Part 1: Model & Run Metadata
  model_id | provider | model_type | endpoint_name | run_number | timestamp

Part 2: Input Data
  input_text (or prompt used)

Part 3: Raw Model Output (COMPLETE, UNTRUNCATED)
  raw_output | output_length_tokens | output_length_chars

Part 4: Inference Metrics
  inference_time_ms | model_confidence (if available)

Part 5: Consistency Tracking
  output_matches_previous_run (yes/no) | output_similarity_percent (0-100%)
```

**For Detection Endpoint:**
```
Part 1: Model & Run Metadata
  model_id | provider | endpoint_name | image_path | inference_number

Part 2: Detection Data
  detection_count | detections_json (full bounding boxes: [[x1,y1,x2,y2], ...])
  detection_confidence_scores (if available)

Part 3: Raw Output
  raw_output_text

Part 4: Results Files
  annotated_image_path

Part 5: Inference Metrics
  inference_time_ms | image_width | image_height
```

**For Point Endpoint:**
```
Part 1: Model & Run Metadata
  model_id | provider | endpoint_name | image_path | inference_number

Part 2: Point Data
  point_x | point_y | point_confidence (0-100%)

Part 3: Raw Output
  raw_output_text

Part 4: Results Files
  annotated_image_path

Part 5: Inference Metrics
  inference_time_ms | image_width | image_height
```

**Example Rows:**

Text Endpoint (Run 1 & 2 of same model for consistency comparison):
```
ollama/llava | ollama | vlm | qa | 1 | 2025-01-20T14:30:05Z | "What is in this image?" | "This image shows a person holding a golden retriever dog in a park. The background features green grass and trees. The weather appears sunny and clear." | 256 | 8845 | 1200 | 0.92 | no | 100
ollama/llava | ollama | vlm | qa | 2 | 2025-01-20T14:31:10Z | "What is in this image?" | "A woman is holding a golden retriever in a park setting. The image has trees in the background and sunny weather. There's grass visible on the ground." | 242 | 8421 | 1210 | 0.90 | partial | 94
```

Detection Endpoint:
```
huggingface/detect | huggingface | detect | /path/to/image_001.jpg | 1 | 3 | [[152,243,445,578], [601,289,723,412], [50,100,200,350]] | [0.95, 0.87, 0.92] | "Found 3 cars in the image..." | results/detections/detect_img001_hf_run1.jpg | 850 | 1920 | 1080
```

**Key Features:**
- **Complete raw outputs**: Full text untruncated, preserved exactly as generated
- **Per-run consistency tracking**: Shows variation between runs of same input
- **Detection data structured**: Full bounding boxes in JSON for parsing
- **Annotated images linked**: Direct path to visual verification
- **Inference timing**: Performance per run for consistency analysis
- **All raw data preserved**: No filtering or modification

**Additional Sections in CSV:**
```
After all run data, append:

=== CONSISTENCY ANALYSIS ===
Model,Endpoint,Total Runs,Exact Matches,Partial Matches,Similarity (Avg %),Variance
ollama/llava,qa,10,2,6,87.3,8.2
huggingface/llava,qa,10,1,5,83.1,12.5

=== OUTPUT LENGTH STATISTICS ===
Model,Endpoint,Avg Tokens,Avg Chars,Min,Max,Std Dev
ollama/llava,qa,256,8456,180,312,45
huggingface/llava,qa,238,7932,150,289,52

=== DETECTION CONSISTENCY (Detection/Point Endpoints) ===
Model,Avg Detection Count,Detection Count Variance,Bbox Stability (%)
custom-detect,5.2,±1.3,78.5
...
```

### Suite 4 Graphs and Visualizations

**Graphs Generated (Complete Quality Visualization):**
1. **Output Length Distribution** - Histogram (tokens/chars across all runs, all models)
2. **Inference Time Trend** - Line chart (time per run, showing consistency)
3. **Output Similarity Heatmap** - Matrix showing pairwise similarity between all runs
4. **Consistency Score Over Runs** - Line chart (similarity to first output)
5. **Detection Count Consistency** - Bar chart (average detection count ± std dev per model)
6. **Bbox Stability Heatmap** - Shows which regions have stable/unstable detections
7. **Point Location Scatter Plot** - Shows coordinate distribution across runs (clustering)
8. **Model Comparison** - Side-by-side output length, consistency, inference time
9. **Variation Over Time** - Shows if consistency degrades as inference progresses
10. **Output Quality Metrics** - Custom metrics like token diversity, vocabulary coverage
11. **Confidence Score Distribution** - If available, show detection confidence scores

**Format:** PNG images + HTML report with output samples
**Directory:** `results/benchmarks/{timestamp}/suite4_graphs/`
**Count:** 11 graphs + HTML report with full output samples

---

## Suite 5: 🔥 Stress & Endurance

### Purpose
Test long-running stability and find system limits. Answer: "How reliable is this model under sustained load? Where does it break?"

### Metrics Collected

#### Primary Metrics (tracked every 5 inferences)
```
1. Performance Degradation
   └─ Current latency vs baseline
   └─ % slowdown from first 5 inferences
   └─ Tokens/sec (to detect throughput loss)
   └─ First-token latency

2. Resource Trends
   └─ CPU usage (avg, peak)
   └─ GPU usage (avg, peak)
   └─ Memory usage (current, peak so far)
   └─ Temperature (current, peak so far)
   └─ Power draw (if available)

3. Thermal Behavior
   └─ When thermal throttling detected
   └─ Duration of throttling
   └─ Temperature trend (increasing, stable, decreasing)

4. Error Tracking
   └─ Number of errors so far
   └─ Error types (OOM, timeout, etc.)
   └─ Error rate % (errors / inferences)
   └─ Timestamp of first error

5. System Stability
   └─ Is memory leaking? (trend: linear, exponential, stable)
   └─ Is degradation accelerating or stabilizing?
   └─ At what point did system start struggling?

6. Breaking Points Detected
   └─ OOM occurred (yes/no, at which inference)
   └─ Timeout occurred (yes/no, at which inference)
```

#### Sampling Frequency
```
Every 5 inferences:
├─ Collect all metrics above
├─ Example: After inference 5, 10, 15, 20, 25, etc.
└─ Create snapshot with current state
```

#### Per-Endpoint Behavior

**LLM Text Endpoint (20+ inferences):**
```
Inference Batch | Latency | Throughput | CPU % | Memory | Temp | Errors | Degradation
─────────────────────────────────────────────────────────────────────────────────────
1-5    (Baseline) | 1205ms | 120.2 t/s  | 75%   | 8.2GB  | 52°C | 0      | 0%
6-10             | 1215ms | 119.5 t/s  | 76%   | 8.3GB  | 54°C | 0      | +0.8%
11-15            | 1240ms | 117.1 t/s  | 78%   | 8.5GB  | 58°C | 0      | +2.9%
16-20            | 1280ms | 112.8 t/s  | 80%   | 8.6GB  | 62°C | 1      | +6.2%
```

**VLM Caption Endpoint (20+ inferences, with images):**
```
Inference Batch | Latency | Throughput | GPU % | GPU Mem | Temp | Memory Leak? | Errors
─────────────────────────────────────────────────────────────────────────────────────────
1-5   (Baseline) | 950ms  | 105.3 t/s  | 72%   | 4.2GB   | 54°C | No         | 0
6-10            | 960ms  | 104.2 t/s  | 74%   | 4.3GB   | 56°C | Slight     | 0
11-15           | 1020ms | 98.0 t/s   | 78%   | 4.8GB   | 61°C | Moderate   | 0
16-20           | 1100ms | 90.9 t/s   | 82%   | 5.2GB   | 66°C | Significant| 0

Memory Leak Alert: Growing from 4.2GB → 5.2GB over 20 inferences
```

### Test Data Requirements

Same as Suite 2 and 3:
- Use **DEFAULT_LLM_PROMPTS** for LLM endpoints
- Use **DEFAULT_TEST_IMAGE_PATH** for VLM endpoints
- Static defaults (same input repeated for stress testing)

### Configuration & Defaults

```
STRESS_SUITE_CONFIG = {
    "num_inferences_default": 20,          # Default for both LLM and VLM
    "num_inferences_strict_options": [50, 100],  # User can increase
    "num_inferences_custom": None,         # Or custom value

    "metric_sampling_interval": 5,         # Collect metrics every 5 inferences

    "breaking_points": {
        "oom_detection": true,             # Stop on Out of Memory
        "timeout_seconds": 180,            # Stop if inference >180s
        "max_acceptable_errors": None      # No limit, just track
    },

    "inference_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "seed": 42,
        "max_tokens": "auto",
        "repeat_penalty": 1.0
    }
}

STRESS_TEST_PROMPT = "Same prompt repeated (for LLM)"
STRESS_TEST_IMAGE = DEFAULT_TEST_IMAGE_PATH  # Same image repeated (for VLM)
```

### Result Structure (Per Suite Run)

```
Results: {
    "suite_name": "Stress & Endurance",
    "timestamp": "2025-01-20T17:00:00Z",
    "duration_seconds": 450,  # Total test duration

    "configuration": {
        "model": "ollama/llama2:7b",
        "endpoint": "text",
        "total_inferences": 20,
        "user_set_strict": false  # Did user increase from default 20?
    },

    "summary": {
        "baseline_metrics": {
            "latency_ms": 1205,
            "throughput_tokens_per_sec": 120.2,
            "cpu_percent": 75,
            "memory_gb": 8.2,
            "temperature_celsius": 52
        },

        "final_metrics": {
            "latency_ms": 1280,
            "throughput_tokens_per_sec": 112.8,
            "cpu_percent": 80,
            "memory_gb": 8.6,
            "temperature_celsius": 62
        },

        "degradation": {
            "latency_increase_percent": 6.2,
            "throughput_loss_percent": 6.1,
            "memory_growth_gb": 0.4,
            "temperature_increase_celsius": 10
        },

        "breaking_points": {
            "oom_detected": false,
            "timeout_detected": false,
            "first_error_at_inference": None,
            "total_errors": 1,
            "system_reached_limit": false,
            "recommendation": "System stable under 20 inferences"
        },

        "stability_analysis": {
            "memory_leak_detected": false,
            "thermal_throttling_detected": false,
            "throttling_started_at_inference": None,
            "error_rate_percent": 5.0,
            "overall_stability": "good"  # good, moderate, poor, critical
        }
    },

    "timeline": [
        {
            "inference_range": "1-5",
            "baseline": true,
            "latency_ms": 1205,
            "throughput_tokens_per_sec": 120.2,
            "cpu_percent": 75.0,
            "memory_gb": 8.2,
            "temperature_celsius": 52,
            "errors": 0,
            "degradation_percent": 0.0
        },
        {
            "inference_range": "6-10",
            "baseline": false,
            "latency_ms": 1215,
            "throughput_tokens_per_sec": 119.5,
            "cpu_percent": 76.0,
            "memory_gb": 8.3,
            "temperature_celsius": 54,
            "errors": 0,
            "degradation_percent": 0.8,
            "notes": "Normal operation"
        },
        {
            "inference_range": "11-15",
            "baseline": false,
            "latency_ms": 1240,
            "throughput_tokens_per_sec": 117.1,
            "cpu_percent": 78.0,
            "memory_gb": 8.5,
            "temperature_celsius": 58,
            "errors": 0,
            "degradation_percent": 2.9,
            "thermal_throttling": false,
            "notes": "Slight degradation visible"
        },
        {
            "inference_range": "16-20",
            "baseline": false,
            "latency_ms": 1280,
            "throughput_tokens_per_sec": 112.8,
            "cpu_percent": 80.0,
            "memory_gb": 8.6,
            "temperature_celsius": 62,
            "errors": 1,
            "degradation_percent": 6.2,
            "error_types": ["timeout"],
            "thermal_throttling": false,
            "notes": "Degradation accelerating, first timeout"
        }
    ],

    "detailed_analysis": {
        "memory_behavior": {
            "initial_gb": 8.2,
            "peak_gb": 8.6,
            "final_gb": 8.6,
            "leaked_gb": 0.4,
            "leak_pattern": "linear",
            "leak_severity": "minor"  # minor, moderate, severe
        },

        "thermal_behavior": {
            "initial_celsius": 52,
            "peak_celsius": 62,
            "final_celsius": 62,
            "temperature_increase": 10,
            "throttling_detected": false,
            "sustained_high_temp_inferences": 0
        },

        "performance_degradation": {
            "latency_degradation_percent": 6.2,
            "throughput_loss_percent": 6.1,
            "degradation_pattern": "linear",  # linear, exponential, sudden
            "degradation_severity": "minor"
        },

        "error_analysis": {
            "total_errors": 1,
            "error_types": {
                "timeout": 1,
                "oom": 0
            },
            "error_rate_percent": 5.0,
            "first_error_at_inference": 20
        }
    }
}
```

### Results Display & Export

**Transparency:** All intervals tracked and visible

**Summary View:**
```
Stress Test Results: llama2:7b (Text endpoint)
Total Inferences: 20 | Duration: 7m 30s

Baseline (1-5 inferences):
├─ Latency: 1205 ms
├─ Throughput: 120.2 tokens/sec
└─ Resources: CPU 75%, Memory 8.2GB, Temp 52°C

Final State (16-20 inferences):
├─ Latency: 1280 ms (↑6.2% degradation)
├─ Throughput: 112.8 tokens/sec (↓6.1%)
└─ Resources: CPU 80%, Memory 8.6GB, Temp 62°C

System Status: STABLE
├─ Memory leak: None
├─ Thermal throttling: No
├─ Errors: 1 timeout at inference 20
└─ Recommendation: Safe for sustained use
```

**Timeline Table (all intervals):**
```
Inference | Latency | Throughput | CPU % | Memory | Temp | Errors | Degradation
──────────────────────────────────────────────────────────────────────────────────
1-5       | 1205ms  | 120.2 t/s  | 75%   | 8.2GB  | 52°C | 0      | 0%
6-10      | 1215ms  | 119.5 t/s  | 76%   | 8.3GB  | 54°C | 0      | +0.8%
11-15     | 1240ms  | 117.1 t/s  | 78%   | 8.5GB  | 58°C | 0      | +2.9%
16-20     | 1280ms  | 112.8 t/s  | 80%   | 8.6GB  | 62°C | 1      | +6.2%
```

**Graphs:**
```
├─ Latency Trend (over inferences)
├─ Memory Growth (over inferences)
├─ Temperature Climb (over inferences)
├─ CPU Usage (over inferences)
└─ Error Count (over inferences)
```

**CSV Export:**
```
All timeline data professionally formatted:
├─ Every interval as row
├─ All metrics as columns
├─ Summary metrics
├─ Breaking point information
└─ Linked to graphs
```

### Error Handling

```
Out of Memory (OOM):
├─ Immediately stop test
├─ Mark: "Breaking point reached at inference X"
├─ Log OOM details
└─ Show in results as critical

Inference Timeout (>180 seconds):
├─ Immediately stop test
├─ Mark: "Timeout at inference X"
├─ Log timeout details
└─ Show in results as critical

Memory Leak Detected (significant growth):
├─ Continue test
├─ Mark in results: "Memory leak detected"
├─ Show leak severity (minor/moderate/severe)
└─ Recommend investigation

Thermal Throttling:
├─ Continue test
├─ Mark when throttling occurred
├─ Track impact on performance
└─ Show in results

Partial Failures (some inferences fail):
├─ Log each failure
├─ Calculate error rate
├─ Continue test
└─ Show in timeline

Test Completion:
├─ If all inferences complete: "Test completed successfully"
├─ If stopped early: "Test stopped at inference X (reason: OOM/timeout)"
└─ Show summary with all collected data
```

### Suite 5 CSV Export Format

**Wide Format with Stress Test Progression & Breaking Points**

**File Name:** `benchmark_stress_TIMESTAMP.csv`

**CSV Columns (in order):**

**Per-Inference Detail Rows:**
```
Part 1: Inference Metadata
  model_id | provider | endpoint_name | inference_number | batch_number (1-5, 6-10, etc.) |
  timestamp | inference_batch_label (1-5, 6-10, etc.)

Part 2: Performance Metrics
  latency_ms | first_token_ms | tokens_generated | tokens_per_second |
  latency_vs_baseline_percent | throughput_vs_baseline_percent

Part 3: Resource Metrics (Current Run)
  cpu_percent | cpu_peak_percent | memory_gb | memory_peak_gb | gpu_utilization_percent |
  gpu_memory_gb | temperature_celsius

Part 4: Trend Analysis
  memory_growth_since_start_gb | temperature_growth_since_start_celsius |
  cpu_trend_direction (increasing/stable/decreasing)

Part 5: Error & Status
  inference_status (success/timeout/oom/error) | error_message | errors_so_far_count

Part 6: Batch Sampling Data (Every 5 inferences)
  batch_performance_summary | batch_resource_summary | batch_degradation_percent
```

**Example Rows (Sampling every 5 inferences):**
```
Batch 1-5 (Baseline):
ollama/llama2:7b | ollama | text | 5 | 1 | 2025-01-20T14:30:05Z | 1-5 | 1205 | 120 | 150 | 120.2 | 0.0 | 0.0 | 75.3 | 88.2 | 8.2 | 8.5 | 65.4 | 3.2 | 52 | 0.0 | 0.0 | increasing | success | null | 0 | Baseline established

Batch 6-10:
ollama/llama2:7b | ollama | text | 10 | 2 | 2025-01-20T14:31:15Z | 6-10 | 1215 | 122 | 150 | 119.5 | 0.8 | 0.6 | 76.2 | 89.5 | 8.3 | 8.6 | 66.8 | 3.3 | 54 | 0.1 | 2.0 | increasing | success | null | 0 | Normal operation

Batch 11-15:
ollama/llama2:7b | ollama | text | 15 | 3 | 2025-01-20T14:32:30Z | 11-15 | 1240 | 125 | 148 | 117.1 | 2.9 | 2.5 | 78.1 | 91.2 | 8.5 | 8.8 | 68.5 | 3.8 | 58 | 0.3 | 6.0 | increasing | success | null | 0 | Degradation visible

Batch 16-20 (Breaking point):
ollama/llama2:7b | ollama | text | 20 | 4 | 2025-01-20T14:33:45Z | 16-20 | 1350 | 135 | 140 | 103.7 | 12.0 | 13.8 | 82.5 | 95.0 | 9.2 | 9.8 | 74.2 | 5.1 | 66 | 1.0 | 14.0 | accelerating | timeout | "Inference 20 exceeded 180s timeout" | 1 | CRITICAL: Breaking point reached
```

**Key Features:**
- **Batch-level sampling**: Every 5 inferences consolidated (efficient for 20+ runs)
- **Trend tracking**: Degradation %, growth %, direction changes visible
- **Breaking point data**: Exact inference where OOM/timeout occurred
- **Full timeline**: See exact moment system struggled
- **Thermal tracking**: Temperature progression throughout test
- **Error tracking**: All failures logged with timestamps

**Additional Sections in CSV:**
```
After all run data, append:

=== STRESS TEST SUMMARY ===
Total Inferences,20
Completed Successfully,19
Failures,1
Error Type,Timeout at inference 20
Test Status,STOPPED (Breaking point reached)

=== DEGRADATION ANALYSIS ===
Metric,Baseline (1-5),Final (16-20),Degradation %,Pattern
Latency (ms),1205,1350,12.0,Linear
Throughput (tokens/sec),120.2,103.7,13.8,Linear
CPU (%),75.3,82.5,9.6,Linear
Memory (GB),8.2,9.2,12.2,Linear
Temperature (C),52,66,26.9,Linear

=== MEMORY & THERMAL ANALYSIS ===
Memory Leak Detected,Yes
Leaked Memory (GB),1.0
Leak Pattern,Linear
Leak Severity,Moderate
Thermal Throttling,No
Peak Temperature,66°C
Safe Margin,14°C (before throttle)

=== SYSTEM STABILITY ASSESSMENT ===
Overall Stability,Moderate
Recommendation,Model stable for 20 inferences. May struggle with 50+. Investigate memory leak.
Safe Run Count,20 (all completions below safe limits)
Max Safe Load,Unknown (not reached in this test)
```

### Suite 5 Graphs and Visualizations

**Graphs Generated (Maximum Stress Analysis Visualization):**
1. **Latency Degradation** - Line chart showing latency increase across all inferences (marks baseline, breaking point)
2. **Throughput Loss** - Line chart showing tokens/sec decrease (with trendline)
3. **Memory Consumption Trend** - Line chart showing memory growth (highlights leak detection point)
4. **Temperature Progression** - Line chart showing thermal climb (marks throttling threshold)
5. **CPU & GPU Over Time** - Dual-axis chart showing resource escalation
6. **Performance Degradation Rate** - Bar chart showing % loss per 5-inference interval
7. **Memory Leak Severity** - Area chart showing memory delta from baseline
8. **Error Timeline** - Timeline/scatter plot showing when errors occurred
9. **Resource Saturation Heatmap** - Color-coded intensity showing which resources saturating first
10. **Stability Score Over Time** - Line chart with overall system stability metric (0-100%)
11. **Batch-by-Batch Comparison** - Multi-series chart comparing all batches side-by-side (latency, throughput, memory, CPU)
12. **Breaking Point Analysis** - Zone highlighting showing exactly where/when system broke
13. **Prediction Extrapolation** - If linear degradation detected, project when system would fail at higher runs (50, 100, etc.)

**Format:** PNG images + interactive HTML with timeline slider
**Directory:** `results/benchmarks/{timestamp}/suite5_graphs/`
**Count:** 13 graphs + interactive HTML timeline

---

## Suite 1: ✨ Complete System Analysis

### Purpose
Run all focused benchmarks (Speed, Resources, Quality) in one comprehensive test. Answer: "What is the complete performance profile of this model?"

### How It Works

**Execution:**
- Runs Suites 2, 3, 4 **sequentially** (NOT Suite 5 - stress testing is separate)
- Same inference runs shared across all three suites
- All metrics collected per inference
- One unified comprehensive result

**Suites Included:**
- Suite 2: Speed & Throughput (latency, tokens/sec, first-token, throughput)
- Suite 3: Resource Efficiency (CPU, GPU, memory, power, disk I/O, efficiency ratios)
- Suite 4: Quality & Consistency (raw outputs, inference times, detection/point results)

**NOT Included:**
- Suite 5: Stress & Endurance (optional separate test for long-running stability)

### Configuration & Defaults

```
COMPLETE_SUITE_CONFIG = {
    "suites_included": [2, 3, 4],  # Speed, Resource, Quality

    "suite_2_config": {
        "num_runs": 7,  # Default (user-configurable)
        "warmup_runs": 2,
        "counted_runs": 5
    },

    "suite_3_config": {
        "num_runs": 7,  # Default (user-configurable)
        "warmup_runs": 2,
        "counted_runs": 5,
        "measure_baseline": true,
        "measure_cleanup": true
    },

    "suite_4_config": {
        "text_runs": 10,  # Default (user-configurable)
        "vlm_inferences_per_image": 2,  # Default (user-configurable)
        "vlm_dataset_required": true  # For VLM endpoints
    },

    "inference_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "seed": 42,
        "max_tokens": "auto",
        "repeat_penalty": 1.0
    }
}
```

### Result Structure

**Unified Comprehensive CSV Format - Wide with Maximum Columns**

```
Column Structure:
├─ Model Metadata
│  ├─ model_id
│  ├─ provider
│  ├─ model_type (LLM or VLM)
│  └─ model_size_gb
│
├─ Endpoint & Run Info
│  ├─ endpoint_name
│  ├─ run_number
│  ├─ is_warmup (true/false)
│  ├─ timestamp
│  └─ input_text / image_path
│
├─ SUITE 2: Speed & Throughput Metrics
│  ├─ latency_ms
│  ├─ tokens_generated
│  ├─ tokens_per_second
│  ├─ first_token_ms
│  ├─ generation_rate_tokens_per_sec
│  └─ speed_variance_sigma
│
├─ SUITE 3: Resource Efficiency Metrics
│  ├─ cpu_avg_percent
│  ├─ cpu_peak_percent
│  ├─ cpu_per_core (if available)
│  ├─ memory_process_mb
│  ├─ memory_system_gb
│  ├─ memory_system_percent
│  ├─ memory_peak_gb
│  ├─ gpu_utilization_percent
│  ├─ gpu_memory_gb
│  ├─ gpu_memory_peak_gb
│  ├─ gpu_temperature_celsius
│  ├─ power_avg_watts (NVIDIA only)
│  ├─ power_peak_watts (NVIDIA only)
│  ├─ disk_io_mb_per_sec (model loading)
│  ├─ disk_io_total_mb (model loading)
│  ├─ memory_per_token_mb (efficiency)
│  ├─ power_per_token_watts (efficiency, NVIDIA only)
│  ├─ cpu_gpu_ratio (efficiency)
│  ├─ baseline_cpu_percent
│  ├─ baseline_memory_gb
│  ├─ baseline_temperature_celsius
│  ├─ cleanup_memory_gb
│  ├─ cleanup_gpu_memory_gb
│  ├─ memory_freed_gb (cleanup delta)
│  ├─ memory_leaked_gb (cleanup delta)
│  └─ thermal_throttling_detected
│
├─ SUITE 4: Quality & Consistency Metrics
│  ├─ raw_output (complete, untruncated text)
│  ├─ raw_output_length_tokens (for text endpoints)
│  ├─ raw_detections (full bbox list [x1,y1,x2,y2])
│  ├─ raw_coordinates (full x, y data)
│  ├─ detection_count (if applicable)
│  ├─ annotated_image_path (if saved)
│  ├─ suite4_inference_time_ms
│  └─ suite4_errors_if_any
│
└─ Aggregation & Status
   ├─ overall_status (success/partial_failure/critical_failure)
   ├─ notes
   └─ timestamp_completed
```

### Example CSV Row (Text Endpoint, 1 inference)

```
model_id | provider | model_type | endpoint_name | run_number | latency_ms | tokens_generated | tokens_per_second | first_token_ms | cpu_avg_percent | memory_system_gb | gpu_utilization_percent | temperature_celsius | raw_output | ...more columns
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
ollama/llama2:7b | ollama | LLM | text | 1 | 1200 | 150 | 125.0 | 105 | 75.3 | 8.2 | 65.4 | 58 | "Machine learning is a subset of artificial intelligence that..." | ...
```

### Results Display & Export

**Summary View:**
```
Complete System Analysis Results
Models tested: 2 | Endpoints: 3 | Total inferences: 42

Quick Overview:
├─ Fastest model: Model A (avg 1200ms)
├─ Most efficient: Model B (75% CPU, 3.1GB memory)
├─ Most stable outputs: Model A (10 runs consistent)
└─ Overall recommendation: Model A for speed, Model B for efficiency

Detailed results available in:
├─ CSV export (all metrics, all inferences)
├─ Suite 2 detailed results (speed breakdown)
├─ Suite 3 detailed results (resource breakdown)
└─ Suite 4 detailed results (output samples, detection images)
```

**Comprehensive CSV Export:**
```
File: complete_analysis_TIMESTAMP.csv
Format: Wide (all columns)
Content:
├─ Every inference as one row
├─ All metrics from all 3 suites as columns
├─ Complete raw outputs in dedicated cells
├─ Full bbox/coordinate data for detection/point
├─ No data duplication
└─ Professional formatting maintained in text cells
```

### Error Handling

```
Inference fails in any suite:
├─ Mark row with error status
├─ Log error details
└─ Continue with other inferences

Partial suite failure:
├─ Complete available suites
├─ Mark incomplete suites
└─ Include all successful data in CSV

Model loading fails:
├─ Cannot proceed
└─ Inform user before starting

Test interrupted:
├─ Save partial results collected so far
├─ Include all completed inferences in CSV
└─ Mark test as incomplete
```

### Suite 1 CSV Export Format

**Comprehensive Wide Format - Maximum Columns from All Suites**

**File Name:** `benchmark_complete_TIMESTAMP.csv`

**CSV Structure: All metrics on single wide row**
```
Part 1: Model Metadata (Suite 1 columns)
  model_id | provider | model_type | model_size_gb | endpoint_name | run_number | is_warmup

Part 2: Timestamp & Input
  timestamp | input_text | image_path_if_vlm

Part 3: SUITE 2 - Speed & Throughput Metrics
  latency_ms | first_token_ms | tokens_generated | tokens_per_second |
  generation_rate_tokens_per_sec | response_length_chars

Part 4: SUITE 3 - Resource Efficiency Metrics
  cpu_avg_percent | cpu_peak_percent | memory_system_gb | memory_peak_gb |
  gpu_utilization_percent | gpu_memory_gb | gpu_temperature_celsius |
  power_avg_watts | power_peak_watts | memory_per_token_mb | cpu_gpu_ratio |
  baseline_cpu_percent | baseline_memory_gb | cleanup_memory_gb |
  memory_leaked_gb | thermal_throttling_detected

Part 5: SUITE 4 - Quality & Consistency Metrics
  raw_output (complete untruncated text) | output_length_tokens |
  output_similarity_percent | consistency_score |
  detection_count_if_applicable | detections_json_if_applicable |
  point_x_if_applicable | point_y_if_applicable |
  annotated_image_path_if_applicable

Part 6: Aggregation & Status
  inference_status | error_message | overall_quality_score |
  timestamp_completed
```

**Example Row (Text Endpoint - one inference):**
```
ollama/llama2:7b | ollama | llm | 3.8 | text | 1 | true | 2025-01-20T14:30:05Z | "What is machine learning?" | null | 1500 | 120 | 150 | 100.0 | 108.0 | 2847 | 75.3 | 88.2 | 8.2 | 8.5 | 0 | 0 | 52 | 0 | 0 | 20.7 | 1.15 | 75.0 | 8.0 | 8.1 | 0.1 | false | "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed..." | 256 | 100 | 94 | "success" | null | 0.92 | 2025-01-20T14:30:05Z
```

**Example Row (QA Endpoint - VLM):**
```
huggingface/llava | huggingface | vlm | 7.2 | qa | 3 | false | 2025-01-20T14:31:15Z | "What's in this image?" | /dataset/images/qa/image_003.jpg | 1250 | 180 | 85 | 68.0 | 62.5 | 1956 | 65.4 | 72.1 | 8.5 | 9.2 | 48.5 | 3.8 | 58 | 125.0 | 150.0 | 0.45 | 0.82 | 65.0 | 8.2 | 8.6 | 0.4 | false | "This image shows a person sitting in a park with trees in the background. The weather appears to be sunny..." | 142 | 92 | 88 | "success" | null | 0.88 | 2025-01-20T14:31:15Z
```

**Key Features:**
- **Single comprehensive CSV**: All data in one file with maximum columns
- **No data duplication**: Each metric appears once, aggregated from all suites
- **Complete preservation**: Raw outputs, detection data, all metrics included
- **Easy analysis**: One row = one inference across all suites
- **Spreadsheet ready**: Can pivot, filter, compare in Excel/pandas/R

**Additional Sections in CSV:**
```
After all run data, append:

=== COMPLETE SYSTEM SUMMARY ===
Total Inferences,30 (3 suites × 10 runs)
Models Tested,2
Endpoints Tested,3
Total Duration,15m 42s

Model Performance Comparison:
Model,Endpoint,Avg Latency,Avg Throughput,Avg CPU,Avg Memory,Avg GPU Util,Consistency %,Overall Score
ollama/llama2:7b,text,1205,120.2,75.3,8.2,65.4,94,0.92
huggingface/llava,qa,1250,68.0,65.4,8.5,48.5,88,0.88

=== KEY INSIGHTS ===
Fastest Model,ollama/llama2:7b (1205ms avg)
Most Efficient,huggingface/llava (8.5GB memory avg, 48.5% GPU util)
Most Consistent,ollama/llama2:7b (94% consistency)
Best Overall,ollama/llama2:7b (score: 0.92)
Memory Concern,Minor leak detected in Suite 3 (0.1-0.4GB per run)
Thermal Status,All safe (peak 58°C, margin 32°C)

=== RECOMMENDATIONS ===
- Use ollama/llama2:7b for speed-critical applications
- Use huggingface/llava for memory-constrained environments
- Investigate memory leak in huggingface provider
- Both models safe for sustained 20+ inference loads
```

### Suite 1 Graphs and Visualizations

**Master Graphs (Combining all suites - maximum insight):**

**Speed Comparisons:**
1. **Model Performance Ranking** - Bar chart ranking models by latency, throughput, consistency
2. **Latency vs Efficiency Trade-off** - Scatter plot (latency vs memory, bubbles = model)
3. **Speed Across Endpoints** - Grouped bar chart (model A vs B vs C for each endpoint)

**Resource Analysis:**
4. **Resource Consumption Comparison** - Grouped bar (CPU, Memory, GPU, Power by model)
5. **Efficiency Metrics Dashboard** - Heatmap showing efficiency ratios per model
6. **Memory Footprint vs Performance** - Scatter plot (memory vs tokens/sec)
7. **Thermal Profile Comparison** - Temperature ranges for each model

**Quality & Consistency:**
8. **Consistency Scorecard** - Radar chart comparing consistency % for all models
9. **Output Quality Distribution** - Box plots for output length, similarity across runs
10. **Detection Stability** - For VLM: scatter showing coordinate variance per model

**Combined Analysis:**
11. **Model Capability Matrix** - 3D or bubbleplot (Speed vs Resources vs Quality)
12. **Overall Performance Score** - Gauge/dial for each model (aggregate score 0-100)
13. **Trade-off Analysis** - Parallel coordinates showing all metrics for each model
14. **Benchmark Summary Table** - Professional table with key metrics, color-coded performance

**Temporal Analysis:**
15. **All Inferences Timeline** - Timeline showing all 30 inferences with color coding (model/endpoint)
16. **System Load Over Time** - Stacked area showing cumulative CPU, GPU, Memory

**Export Report:**
17. **Interactive HTML Dashboard** - Single-page dashboard with all charts, filtering, comparison tools

**Format:** PNG images + Interactive HTML dashboard
**Directory:** `results/benchmarks/{timestamp}/suite1_graphs/` + `suite1_dashboard.html`
**Count:** 17 graphs + interactive HTML with comparison tools

---

## Implementation Tasks

### Phase 1: Core Infrastructure (Weeks 1-2)

#### 1.1 Create Base Structures
- [ ] Create `src/benchmarking/models/benchmark_config.py`
  - [ ] BenchmarkConfig pydantic model
  - [ ] TestDataConfig
  - [ ] SuiteConfig
- [ ] Create `src/benchmarking/models/suite_result.py`
  - [ ] SuiteResult data structure
  - [ ] InferenceMetric
  - [ ] EndpointResults
  - [ ] StatisticsBundle

#### 1.2 Core Module Implementation
- [ ] Create `src/benchmarking/core/metrics_collector.py`
  - [ ] MetricsCollector class (reusable)
  - [ ] Methods: start_inference_timing(), stop_timing(), collect_metrics()
  - [ ] Handle token counting for different providers
- [ ] Create `src/benchmarking/core/system_monitor.py`
  - [ ] SystemMonitorWrapper (wraps SystemMonitor from system_specs)
  - [ ] Methods: get_system_snapshot(), start_monitoring(), stop_monitoring()
  - [ ] Integration with SystemMonitor/SystemSpecsDetector
- [ ] Create `src/benchmarking/core/result_formatter.py`
  - [ ] Format results for TUI display
  - [ ] Calculate statistics (avg, std dev, min, max, percentiles)
  - [ ] Handle different metric types (tokens/sec, latency, etc.)

#### 1.3 Handler Implementation
- [ ] Create `src/benchmarking/handlers/endpoint_executor.py`
  - [ ] Generic endpoint invocation wrapper
  - [ ] Timing capture around provider.run_X() calls
  - [ ] Error handling and retry logic
- [ ] Create `src/benchmarking/handlers/llm_handler.py`
  - [ ] LLMHandler class
  - [ ] Endpoint: TEXT only
  - [ ] Token counting for LLM outputs
- [ ] Create `src/benchmarking/handlers/vlm_handler.py`
  - [ ] VLMHandler class
  - [ ] Endpoints: QA, CAPTION, DETECT, POINT, TEXT
  - [ ] Token counting for vision endpoints
  - [ ] Detection/point metric handling

#### 1.4 Dataset Setup
- [ ] Create `src/benchmarking/datasets/defaults.py`
  - [ ] DEFAULT_LLM_PROMPTS list
  - [ ] DEFAULT_VLM_PROMPTS dict
  - [ ] DEFAULT_INFERENCE_PARAMS dict
  - [ ] BENCHMARK_CONFIG dict
- [ ] Create `src/benchmarking/datasets/llm_datasets.py`
  - [ ] Load LLM test prompts
  - [ ] Support custom prompt input
- [ ] Create `src/benchmarking/datasets/vlm_datasets.py`
  - [ ] Load VLM test images
  - [ ] Load VLM prompts per endpoint
  - [ ] Support custom image/prompt input
- [ ] Add default test image to `benchmarking/datasets/default_test_image.jpg`

### Phase 2: Suite 2 Implementation (Weeks 2-3)

#### 2.1 Suite 2 Base
- [ ] Create `src/benchmarking/suites/base_suite.py`
  - [ ] BaseSuite abstract class
  - [ ] Methods: run(), collect_metrics(), format_results()
  - [ ] Common error handling
- [ ] Create `src/benchmarking/suites/suite_speed.py`
  - [ ] SpeedSuite class (implements BaseSuite)
  - [ ] run(models, endpoints, test_data, params) method
  - [ ] Metric collection: latency, tokens/sec, first-token, generation rate
  - [ ] Per-endpoint behavior handling
  - [ ] Result aggregation and statistics

#### 2.2 Suite 2 Integration
- [ ] Integrate with EndpointExecutor for timing
- [ ] Integrate with MetricsCollector for system metrics (if needed for display)
- [ ] Integrate with result formatter for statistics
- [ ] Test with LLM and VLM models
- [ ] Test with all endpoints (TEXT, QA, CAPTION, DETECT, POINT)

### Phase 3: UI/UX Implementation (Weeks 3-4)

#### 3.1 Menu Screens
- [ ] Create `src/benchmarking/ui/benchmark_menu.py`
  - [ ] Screen 1: LLM vs VLM selection
  - [ ] Screen 2: Suite selection
  - [ ] Menu navigation logic
- [ ] Create `src/benchmarking/ui/model_selection.py`
  - [ ] Screen 3: Model filtering by type
  - [ ] Multi-select support
  - [ ] Model discovery integration
- [ ] Create `src/benchmarking/ui/endpoint_selection.py`
  - [ ] Screen 4: Type-aware endpoint selection
  - [ ] LLM shows TEXT only
  - [ ] VLM shows all 5 endpoints
  - [ ] Multi-select support
- [ ] Create `src/benchmarking/ui/test_data_config.py`
  - [ ] Screen 5: Static vs custom data selection
  - [ ] Custom prompt input
  - [ ] Custom image path input
- [ ] Create `src/benchmarking/ui/benchmark_config.py`
  - [ ] Screen 6: Num runs, warmup, parameters
  - [ ] Display hardcoded defaults
  - [ ] Allow parameter override
  - [ ] Time estimation
- [ ] Create `src/benchmarking/ui/mode_selection.py`
  - [ ] Screen 7: Foreground vs Background selection

#### 3.2 Display Components
- [ ] Create `src/benchmarking/ui/foreground_display.py`
  - [ ] Real-time inference display
  - [ ] Progress bar and elapsed time
  - [ ] Per-run metrics display
  - [ ] Running averages
- [ ] Create `src/benchmarking/ui/results_viewer.py`
  - [ ] Results summary view
  - [ ] Detailed tables view
  - [ ] Graphs visualization (optional Phase 3+)
  - [ ] Export options
- [ ] Create `src/benchmarking/ui/background_monitor.py`
  - [ ] /background command integration
  - [ ] Task list display
  - [ ] Detailed task view with metrics
  - [ ] ETA display

#### 3.3 Console Formatters
- [ ] Create `src/benchmarking/ui/console_formatter.py`
  - [ ] TUI panel creation (reusable across screens)
  - [ ] Table formatting for results
  - [ ] Progress bar formatting
  - [ ] Metric display formatting

### Phase 4: Reporting (Weeks 4-5)

#### 4.1 Export Reporters
- [ ] Create `src/benchmarking/reporters/csv_reporter.py`
  - [ ] Export detailed results to CSV
  - [ ] Include all runs + statistics
  - [ ] Per-endpoint CSV files
- [ ] Create `src/benchmarking/reporters/json_reporter.py`
  - [ ] Export full result structure to JSON
  - [ ] Preserve all metadata
- [ ] Create `src/benchmarking/reporters/html_reporter.py`
  - [ ] Generate HTML report with tables
  - [ ] Embed graphs if available
- [ ] Create `src/benchmarking/reporters/console_formatter.py`
  - [ ] Shared formatting logic for display

### Phase 5: Background Task Integration (Weeks 5-6)

#### 5.1 Background Runner
- [ ] Create `src/benchmarking/core/benchmark_runner.py`
  - [ ] BenchmarkRunner orchestrator
  - [ ] Foreground execution
  - [ ] Background task queuing
  - [ ] Task persistence for /background monitoring
- [ ] Integrate with existing quantization background task system
  - [ ] Queue benchmark tasks alongside quant/finetune
  - [ ] Store task state for monitoring
  - [ ] Handle task completion/failure notifications

#### 5.2 /background Integration
- [ ] Modify existing `/background` command handler
  - [ ] Add benchmarking task type (alongside quant, finetune)
  - [ ] Show benchmark tasks in task list
  - [ ] Implement detailed benchmark task view
  - [ ] Show system metrics during background run
  - [ ] Show ETA calculation

### Phase 6: Suites 3-5 Implementation (Weeks 6-8)

- [ ] Suite 3: Resource Efficiency
- [ ] Suite 4: Quality & Consistency
- [ ] Suite 5: Stress & Endurance

### Phase 7: Suite 1 - Complete System Analysis (Weeks 8-9)

- [ ] Suite 1: Complete System Analysis (imports and combines 2-5)

### Phase 8: Testing & Refinement (Weeks 9-10)

- [ ] Integration testing with all suites
- [ ] Multi-model benchmarking
- [ ] Multi-endpoint benchmarking
- [ ] Background task integration testing
- [ ] Export and reporting testing
- [ ] Performance optimization

---

## Key Design Decisions

1. **Modular Suites**: Each suite is independent, can be imported into Full Profile
2. **Hardcoded Defaults**: Sensible defaults per provider/model type, but user-overridable
3. **Reusable Components**: Core modules used across all suites
4. **Type-Aware Handling**: Different UX for LLM vs VLM
5. **Foreground/Background**: Real-time display or async monitoring via /background
6. **One-Click Export**: Results include CSV + graphs + metadata
7. **Error Resilience**: Partial failures don't stop entire benchmark

---

## Success Criteria

- ✅ Suite 2 fully functional with all metrics
- ✅ Real-time display shows all required metrics
- ✅ Results export includes CSV with graphs
- ✅ Background monitoring shows progress + ETA
- ✅ Works with LLM and VLM models
- ✅ Works with all endpoints
- ✅ Modular code (can import suites into Full Profile)
- ✅ Same TUI style as existing app
