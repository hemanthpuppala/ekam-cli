# Dynamic KV Cache Implementation

## Overview

The system now intelligently enables/disables KV cache based on:
1. **Device type** (Metal/CUDA/ROCm/CPU)
2. **Context** (Benchmarking vs Inference Pipeline)

## Policy (`src/utils/kv_cache_policy.py`)

```python
def should_use_kv_cache(device: str, is_benchmark: bool = False) -> bool:
    """
    Returns True if KV cache should be enabled.

    Rules:
    - Benchmarking: Always False (stateless for consistent measurements)
    - Metal (mps): False (memory fragmentation workaround)
    - CUDA/ROCm/CPU: True (stable, faster with conversation history)
    """
```

## How It Works

### Inference Pipeline (Chat/QA with History)

**When user enables conversation history:**

1. **Device Check** (`app_chat.py:287`, `app_qa.py:357`)
   ```python
   device = session_manager.state.loaded_model.device
   use_kv_cache = should_use_kv_cache(device, is_benchmark=False)
   ```

2. **Session ID Assignment**
   - **Metal/MPS**: `session_id=None` → KV cache disabled
   - **CUDA/ROCm/CPU**: `session_id=active_session.session_id` → KV cache enabled

3. **Provider Call** (`app_chat.py:354`, `app_qa.py:411`)
   ```python
   response = provider.run_qa(
       model_handle, image, question,
       conversation_history,
       custom_parameters,
       stream_callback=_stream_cb,
       session_id=session_id_for_inference  # ← Conditionally set
   )
   ```

4. **llama-server** (`llama_cli_server_manager.py:640-641`)
   ```python
   if session_id is None:
       payload["cache_prompt"] = False  # ← Clears KV cache
   else:
       # cache_prompt defaults to True → KV cache enabled
   ```

### Benchmarking Pipeline

**Always stateless:**
- Never passes `session_id` parameter
- Defaults to `None`
- KV cache always disabled for consistent measurements

## User-Visible Behavior

### On Metal/MPS (e.g., Apple Silicon)

**Chat with history enabled:**
```
💾 KV cache disabled (mps - workaround for memory fragmentation)
```

- ⚠️ **Slower**: Re-processes full conversation history each time
- ✅ **Stable**: No memory allocation failures
- ✅ **Works correctly**: Context preserved via conversation_history

### On CUDA/ROCm/CPU

**Chat with history enabled:**
```
💾 KV cache enabled (cuda - faster with conversation history)
```

- ⚡ **Faster**: Only processes new input, reuses cached context
- ✅ **Stable**: No fragmentation issues on these backends
- ✅ **Efficient**: Long conversations don't slow down

### Benchmarking (All Devices)

```
KV cache disabled (benchmarking mode - stateless measurements)
```

- ✅ **Consistent**: Each inference is independent
- ✅ **Fair**: No caching advantages
- ✅ **Repeatable**: Results are reproducible

## Benefits

| Scenario | Before | After |
|----------|--------|-------|
| **Metal + Chat (10 turns)** | Fails after 3-4 images | ✅ Works, stable (slower) |
| **CUDA + Chat (10 turns)** | Re-processes 200+ tokens/turn | ⚡ Only processes ~20 new tokens |
| **Benchmarking** | Stateless ✓ | Stateless ✓ (unchanged) |

## Technical Details

### Conversation History vs KV Cache

**Without KV cache (Metal):**
```
Turn 1: Process "Hello" → Generate response
Turn 2: Process ["Hello", "Hi!", "What's up?"] → Generate response (ALL reprocessed)
Turn 3: Process [8 messages, ~150 tokens] → Generate response (ALL reprocessed)
```

**With KV cache (CUDA):**
```
Turn 1: Process "Hello" → Cache state → Generate response
Turn 2: Reuse cached "Hello" → Process "What's up?" → Cache state → Generate response
Turn 3: Reuse cached history → Process new input → Cache state → Generate response
```

### Files Modified

1. **`src/utils/kv_cache_policy.py`** *(new)* - Policy logic
2. **`src/core/app_chat.py`** - Dynamic session_id for text chat
3. **`src/core/app_qa.py`** - Dynamic session_id for visual QA
4. **`src/providers/quantized.py`** - Accept and forward session_id
5. **`src/services/llama_cli_server_manager.py`** - Already handles session_id correctly

### Backward Compatibility

- ✅ Old code without session_id: Works (defaults to None)
- ✅ Benchmarking: Unchanged behavior
- ✅ Non-GGUF providers: Gracefully ignore session_id parameter

## Testing

### Verify KV Cache is Working (CUDA/ROCm/CPU)

1. Start chat with history enabled
2. Have a 5+ turn conversation
3. Check logs for:
   ```
   💾 KV cache enabled (cuda - faster with conversation history)
   Using session ID for KV cache: abc123def456
   ```
4. Later turns should be noticeably faster

### Verify Metal Stays Stable

1. Run vision benchmarks on Apple Silicon
2. Should complete without "failed to process image" errors
3. Check logs for:
   ```
   ⚠️  KV cache disabled (mps - workaround for memory fragmentation)
   ```

## Future Improvements

- Monitor llama.cpp issues for Metal backend fixes
- Once Metal fragmentation is fixed upstream, remove Metal exception
- Add user override option in config.yaml: `force_kv_cache: true/false/auto`
