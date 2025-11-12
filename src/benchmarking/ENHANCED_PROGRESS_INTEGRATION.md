# Enhanced Progress Tracking Integration Guide

## Overview

This document describes the enhanced progress tracking system with:
- ✅ O(1) time/space complexity for all operations
- ✅ Per-run status tracking with text badges, colored symbols, and progress indicators
- ✅ Queue modal showing current/next/completed/failed models
- ✅ Global batch retry for failed runs
- ✅ Nested panels always visible during execution
- ✅ Dynamic support for n models
- ✅ Matching UI colors and vibe

## Architecture

### Data Structures (O(1) Lookups)

```python
# RunStatus: PENDING, RUNNING, SUCCESS, FAILED, RETRYING, RETRY_SUCCESS, RETRY_FAILED
# ModelStatus: QUEUED, RUNNING, COMPLETED, FAILED, RETRYING

# RunInfo: Tracks individual run with status, latency, error, retry_count
# ModelInfo: Tracks model with dictionary of runs (run_number -> RunInfo)
# ProgressState: Central state with models dict (model_id -> ModelInfo)
```

### Key Methods

1. **Initialization** (called once, O(n)):
   ```python
   progress_display.initialize_models(model_ids, total_runs_per_model)
   ```

2. **Tracking** (all O(1)):
   ```python
   progress_display.start_model(model_id)
   progress_display.start_run(model_id, run_number)
   progress_display.record_run_success(model_id, run_number, latency_ms, is_retry)
   progress_display.record_run_failure(model_id, run_number, error, is_retry)
   progress_display.complete_model(model_id)
   ```

3. **Display** (called periodically):
   ```python
   progress_display.render_run_status_panel(model_id)  # Run-by-run status
   progress_display.render_queue_modal()  # All models queue
   ```

4. **Retry** (global batch):
   ```python
   failed_runs = progress_display.get_failed_runs()  # O(1)
   # Process retries...
   progress_display.clear_retry_queue()  # O(1)
   ```

## Integration Pattern

### Step 1: Initialize Tracking

```python
def run(self, config: BenchmarkConfig) -> SuiteResult:
    # Get model IDs
    model_ids = config.models

    # Start progress with model initialization
    self._progress_start(
        total_runs=num_runs,
        num_models=len(model_ids),
        model_ids=model_ids  # NEW: Pass model_ids
    )

    # Display queue modal at start
    self._display_queue_status()
```

### Step 2: Track Each Model

```python
for model_idx, model_id in enumerate(model_ids, start=1):
    # Mark model as running
    self._track_model_start(model_id)

    # Update progress display
    self._progress_update_model(model_id, model_idx)

    # ... load model ...

    # Run benchmarks with tracking
    for run_num in range(1, num_runs + 1):
        self._track_run_start(model_id, run_num)

        try:
            # Execute run
            result = execute_inference(...)

            # Track success
            latency_ms = result.get('latency_ms')
            self._track_run_success(model_id, run_num, latency_ms)

        except Exception as e:
            # Track failure (queued for retry)
            self._track_run_failure(model_id, run_num, str(e))

    # Mark model complete
    self._track_model_complete(model_id)

    # Display run status panel after completing model
    self._display_run_status(model_id)
```

### Step 3: Global Batch Retry

```python
# After all models complete, retry failures
failed_runs = self._get_failed_runs_for_retry()

if failed_runs:
    logger.info(f"Retrying {len(failed_runs)} failed runs...")
    self._display_queue_status()

    for model_id, run_number in failed_runs:
        # Update status to RETRYING
        self._track_run_start(model_id, run_number)

        try:
            # Re-execute run
            result = execute_inference(...)

            # Track retry success
            latency_ms = result.get('latency_ms')
            self._track_run_success(model_id, run_number, latency_ms, is_retry=True)

        except Exception as e:
            # Track retry failure
            self._track_run_failure(model_id, run_number, str(e), is_retry=True)

    # Clear retry queue
    self._clear_retry_queue()

    # Display final status
    self._display_queue_status()
```

### Step 4: Periodic Status Display

```python
# Display queue modal periodically (e.g., every N models)
if model_idx % 2 == 0:  # Every 2 models
    self._display_queue_status()
```

## Visual Output Examples

### Run Status Panel (Per Model)

```
╭─ Run Status: Qwen_Qwen3-VL-2B-Instruct_iq4_xs_iq4_xs_language.gguf ─╮
│ Run  Status             Symbol  Latency     Progress               │
│ ──────────────────────────────────────────────────────────────────  │
│  1   ✓ OK              ●       1234.56ms   ██████████             │
│  2   ✓ OK              ●       1156.23ms   ██████████             │
│  3   ✗ FAIL            ●       -           ██████████             │
│  4   ✓ OK              ●       1298.45ms   ██████████             │
│  5   ↻ RETRY           ◐       -           █████░░░░░             │
│                                                                      │
│ Total  4 ✓ 1 ✗                            ████████░░             │
╰──────────────────────────────────────────────────────────────────────╯
```

### Queue Modal (All Models)

```
╭────────────────────── Benchmark Queue ──────────────────────────────╮
│ #    Model                              Status      Progress  Summary │
│ ──────────────────────────────────────────────────────────────────    │
│ →    CURRENT: model1.gguf                                             │
│ ↓    NEXT: model2.gguf                                                │
│                                                                        │
│ 1    model1.gguf                       ▶ RUNNING   ████████░░  4/5 ✓│
│ 2    model2.gguf                       ⏸ QUEUED    ░░░░░░░░░░  0/5  │
│                                                                        │
│      Total: 2 | Done: 0 | Failed: 0 | Pending: 1                    │
│      ⚠ 1 runs queued for retry                                       │
╰──────────────────────────────────────────────────────────────────────╯
```

## Suite-Specific Enhancements

### Speed Suite
- Display latency percentiles in run status
- Show tokens/second for each run
- Highlight fastest/slowest runs

### Resources Suite
- Show memory usage per run
- Display GPU utilization
- Track peak resource usage

### Quality Suite
- Show accuracy/score per run
- Display average quality metrics
- Highlight outliers

### Stress Suite
- Show error rate progression
- Display stability metrics
- Track degradation over time

## Benefits

1. **O(1) Complexity**: All tracking operations are constant time
2. **Memory Efficient**: Dictionary-based lookups, no redundant data
3. **Real-Time Visibility**: See exactly what's happening per-run
4. **Smart Retry**: Only retry once, after all models complete
5. **Comprehensive Status**: Know current/next/completed/failed at a glance
6. **Consistent UI**: Matches existing Rich-based theme and colors
7. **Suite Agnostic**: Works identically across all benchmark suites
8. **Extensible**: Easy to add suite-specific metrics to panels

## Implementation Status

✅ **Complete**:
- Enhanced data structures (RunInfo, ModelInfo, ProgressState)
- O(1) tracking methods (start/success/failure/complete)
- Render methods (run_status_panel, queue_modal)
- Base suite integration (helper methods)
- Global retry queue management

📋 **TODO** (for full integration):
- Update each suite's `run()` method to call tracking methods
- Add periodic queue modal display
- Implement retry logic in each suite
- Add suite-specific metrics to panels
- Test with multiple models
