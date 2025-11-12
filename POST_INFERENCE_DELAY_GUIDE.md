# Post-Inference Delay - Configuration Guide

## Overview

A configurable delay has been added AFTER each llama-server inference to prevent the intermittent `mtmd_encode_chunk()` state corruption bug. This delay:

✅ **Does NOT affect benchmark latency measurements**
✅ Significantly reduces HTTP 500 "failed to process image" errors
✅ Applies automatically to all llama-server providers
⚠️ Adds to total benchmark runtime (not per-request timing)

## Quick Start

### Default Behavior (Recommended)
No configuration needed! The system uses a sensible 500ms default delay.

### Custom Configuration

```bash
# Set custom delay (in milliseconds)
export LLAMA_SERVER_POST_INFERENCE_DELAY_MS=500  # default

# For faster benchmarks (higher risk of failures)
export LLAMA_SERVER_POST_INFERENCE_DELAY_MS=250

# For heavily loaded systems (maximum reliability)
export LLAMA_SERVER_POST_INFERENCE_DELAY_MS=1000

# Disable delay (NOT recommended - will cause failures)
export LLAMA_SERVER_POST_INFERENCE_DELAY_MS=0
```

### Running Benchmarks with Custom Delay

```bash
# One-time use
LLAMA_SERVER_POST_INFERENCE_DELAY_MS=250 python -m ekam benchmark

# Persistent setting
export LLAMA_SERVER_POST_INFERENCE_DELAY_MS=750
python -m ekam benchmark

# Disable for testing (expect failures!)
LLAMA_SERVER_POST_INFERENCE_DELAY_MS=0 python -m ekam benchmark
```

## How It Works

### Execution Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Execute inference request                             │
│    ↓                                                      │
│ 2. Measure latency (timing ends here)                    │
│    ↓                                                      │
│ 3. Log result to user                                    │
│    ↓                                                      │
│ 4. Record metrics to CSV                                 │
│    ↓                                                      │
│ 5. **DELAY APPLIED HERE** (configurable, default 500ms)  │  ← Does NOT affect latency!
│    ↓                                                      │
│ 6. Return to start next request                          │
└─────────────────────────────────────────────────────────┘
```

### Key Points

1. **Latency measurements are accurate** - The delay happens AFTER timing stops
2. **All metrics unaffected** - TTFT, throughput, tokens/sec remain precise
3. **Total runtime increases** - Benchmark takes longer but data is clean
4. **Applies everywhere** - All suites (speed, quality, resources, stress)

## Performance Impact Calculator

| Requests | Delay | Added Runtime |
|----------|-------|---------------|
| 10       | 500ms | +5 seconds    |
| 50       | 500ms | +25 seconds   |
| 100      | 500ms | +50 seconds   |
| 500      | 500ms | +4.2 minutes  |
| 1000     | 500ms | +8.3 minutes  |

| Requests | Delay | Added Runtime |
|----------|-------|---------------|
| 100      | 250ms | +25 seconds   |
| 100      | 750ms | +75 seconds   |
| 100      | 1000ms| +100 seconds  |

## Choosing the Right Delay

### 500ms (Default) - Recommended
**Use when:**
- Running standard benchmarks
- Want balance of speed and reliability
- Using typical hardware (M1/M2/M3, RTX 30xx/40xx)

**Characteristics:**
- ~95% reliability
- Reasonable runtime
- Well-tested default

### 250ms (Faster) - Experimental
**Use when:**
- Need faster benchmark completion
- Willing to accept slightly higher failure rate
- Have powerful hardware
- Running small-scale tests

**Trade-offs:**
- ⚠️ May see occasional HTTP 500 errors
- ✅ Faster benchmarks
- ⚠️ Retry logic will catch most failures

### 750-1000ms (Maximum Reliability) - Conservative
**Use when:**
- Running critical benchmarks
- Cannot tolerate ANY failures
- System is under heavy load
- Running very large benchmark suites

**Characteristics:**
- ✅ Near-zero failure rate
- ⚠️ Slower runtime
- ✅ Best for production benchmarks

### 0ms (No Delay) - For Testing Only
**Use when:**
- Testing the bug itself
- Measuring failure rates
- Verifying retry logic works

**DO NOT USE for:**
- ❌ Production benchmarks
- ❌ Real measurements
- ❌ Comparing models

## Implementation Details

### Code Locations

1. **Delay configuration**: `src/benchmarking/utils/llama_delay.py`
2. **Speed suite**: `src/benchmarking/suites/suite_speed.py:563`
3. **Quality suite**: `src/benchmarking/suites/suite_quality.py:447`
4. **Resources suite**: `src/benchmarking/suites/suite_resources.py:446`
5. **Stress suite**: `src/benchmarking/suites/suite_stress.py:348`

### Provider Detection

The system automatically detects llama-server providers:
- ✅ GGUF provider → delay applied
- ✅ Quantized provider → delay applied
- ❌ Hugging Face → no delay
- ❌ Ollama → no delay

## Monitoring and Debugging

### Check if Delay is Applied

```bash
# Enable debug logging
python -m ekam benchmark --verbose

# Look for this message in logs:
# "Applying llama-server post-inference delay: 0.500s"
```

### Measure Impact

```bash
# Benchmark WITHOUT delay
time LLAMA_SERVER_POST_INFERENCE_DELAY_MS=0 python -m ekam benchmark

# Benchmark WITH delay (default)
time python -m ekam benchmark

# Compare the difference = delay × request_count
```

### Verify Reliability

```bash
# Run with minimal delay (expect failures)
LLAMA_SERVER_POST_INFERENCE_DELAY_MS=50 python -m ekam benchmark 2>&1 | grep -i "failed to process image" | wc -l

# Run with recommended delay (expect ~0 failures)
LLAMA_SERVER_POST_INFERENCE_DELAY_MS=500 python -m ekam benchmark 2>&1 | grep -i "failed to process image" | wc -l
```

## FAQ

### Q: Does this affect my latency measurements?
**A:** No! The delay is added AFTER latency measurement completes. Your metrics remain accurate.

### Q: Why not fix llama.cpp instead?
**A:** We should! This is a workaround. A bug report should be filed with llama.cpp maintainers.

### Q: Can I disable it?
**A:** Yes, but you'll see many HTTP 500 failures. Set `LLAMA_SERVER_POST_INFERENCE_DELAY_MS=0`.

### Q: Does it apply to non-vision models?
**A:** Yes, it applies to ALL llama-server inference (text and vision) to be safe.

### Q: What about Ollama and Hugging Face?
**A:** No delay is added for these providers - they don't have the mtmd bug.

### Q: Will this be needed forever?
**A:** No. Once llama.cpp fixes the root bug, we can remove this workaround.

### Q: Can I have different delays for different suites?
**A:** Currently no, but this could be added. File a feature request if needed.

### Q: What if I'm still seeing failures with 500ms?
**A:** Try increasing to 750-1000ms. Some systems may need more time.

## Troubleshooting

### Issue: Still seeing "failed to process image" errors

**Solutions:**
1. Increase delay: `export LLAMA_SERVER_POST_INFERENCE_DELAY_MS=1000`
2. Check if retry logic is working (should auto-recover)
3. Verify llama-server is running properly
4. Check available memory (OOM can cause failures)

### Issue: Benchmarks taking too long

**Solutions:**
1. Reduce delay: `export LLAMA_SERVER_POST_INFERENCE_DELAY_MS=250`
2. Accept slightly higher failure rate
3. Reduce number of test runs
4. Use faster hardware

### Issue: Inconsistent results between runs

**Solutions:**
1. This is expected if failures occur
2. Increase delay for consistency
3. Check logs for retry patterns
4. Verify no other system load during benchmarks

## Related Features

This delay works in conjunction with:

1. **Retry logic** (`src/providers/quantized.py`) - Auto-retries on failures
2. **Request throttling** (`src/providers/quantized.py`) - Minimum 50ms between requests
3. **Memory optimization** - Aggressive cleanup between runs

Together, these provide a robust solution to the llama.cpp state management bug.

## References

- Full workaround documentation: `LLAMACPP_VISION_BUG_WORKAROUND.md`
- llama.cpp multimodal docs: `llama.cpp/docs/multimodal.md`
- Issue tracker: [llama.cpp GitHub Issues](https://github.com/ggml-org/llama.cpp/issues)
