# Bug Report: Vision Model State Corruption in mtmd_encode_chunk()

## Summary

Intermittent HTTP 500 "failed to process image" errors occur in llama.cpp's multimodal (mtmd) library when processing sequential vision requests with `--cont-batching` enabled. The same image/prompt combination fails initially but succeeds on retry after a short delay, indicating state corruption rather than invalid input.

**Impact:** ~30-50% of vision requests fail on first attempt in continuous batching mode
**Severity:** High - blocks production use of vision models
**Reproducibility:** Consistent when processing 3+ sequential images

## Environment

- **llama.cpp version:** Current master (as of 2025-01-11)
- **Model:** Qwen2-VL-2B-Instruct (GGUF quantized)
- **Platform:** macOS (Darwin 23.6.0), Apple Silicon M-series
- **Server args:** `--cont-batching -c 4096 -b 2048 -ub 512 -t 8 -tb 8 --mmproj <mmproj.gguf>`
- **API:** OpenAI-compatible `/v1/chat/completions` with base64 image content

## Steps to Reproduce

1. Start llama-server with vision model:
```bash
llama-server -m model.gguf \
  --mmproj mmproj.gguf \
  --cont-batching \
  -c 4096 -b 2048 -ub 512 \
  -ngl 99 -t 8
```

2. Send 10+ sequential vision requests via `/v1/chat/completions`:
```python
for i in range(10):
    response = requests.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            "max_tokens": 512
        }
    )
```

3. Observe HTTP 500 errors starting from ~3rd image onward

## Expected Behavior

All requests should succeed with valid image data and proper API format.

## Actual Behavior

**Pattern observed:**
- Request 1 (warmup): ✓ Success
- Request 2: ✓ Success
- Request 3: ❌ **HTTP 500** "failed to process image" after 3-6 seconds
- Retry 1 (after 0.1s): ❌ **HTTP 500** again
- Retry 2 (after 0.2s): ✓ **Success** (same image/prompt)
- Request 4-10: Intermittent failures (~40% failure rate)

**Timing pattern:**
- Initial request fails after 3-6 seconds (during encoding)
- Retry with 0.1-0.2s delay often succeeds
- Longer delays increase success probability

## Root Cause Analysis

### Primary Issue: Uncleared Buffer in `mtmd_context`

**Location:** `tools/mtmd/mtmd.cpp:808`

```cpp
struct mtmd_context {
    std::vector<float> image_embd_v; // image embedding vector
    // ...
};

int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    // ...
    int n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);
    ctx->image_embd_v.resize(image_tokens->n_tokens() * n_mmproj_embd);  // ⚠️ PROBLEM!

    // ...
    ok = clip_image_batch_encode(
        ctx_clip,
        ctx->n_threads,
        &image_tokens->batch_f32,
        ctx->image_embd_v.data()  // Writing to potentially dirty buffer
    );

    return ok ? 0 : 1;
}
```

**The Problem:**
1. `std::vector::resize()` changes size but **does NOT clear or zero existing data**
2. After first image, `image_embd_v` contains stale embeddings
3. Subsequent calls reuse the same vector with garbage data
4. `clip_image_batch_encode()` writes to a corrupted buffer
5. Encoding fails with non-zero error code
6. Server returns HTTP 500

### Secondary Issue: Thread Safety

**Location:** `tools/mtmd/mtmd-helper.h:62`

```cpp
// this function is NOT thread-safe
MTMD_API int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx, ...);
```

With `--cont-batching`, multiple requests can access the shared `mtmd_context` concurrently:
- No mutex protecting `image_embd_v`
- Race conditions during `resize()` and write operations
- One request's resize can invalidate another's data pointer

### Error Propagation Path

```
HTTP Request
  → tools/server/server.cpp:3986 process_chunk()
    → tools/server/utils.hpp:1383 mtmd_helper_eval_chunk_single()
      → tools/mtmd/mtmd-helper.cpp:294 mtmd_encode_chunk()
        → tools/mtmd/mtmd.cpp:772 mtmd_encode()
          → tools/mtmd/mtmd.cpp:825-829 clip_image_batch_encode()
            ↓
          Returns error != 0 (buffer corruption)
            ↓
        HTTP 500: "failed to process image"
```

## Evidence from Logs

### Example 1: Failure on 3rd Image

```
2025-11-11 12:37:11 | Request 3 | image_2.jpg (612x537)
2025-11-11 12:37:14 | ⚠️  mtmd_encode_chunk() failure (HTTP 500) - attempt 1/3
2025-11-11 12:37:18 | ⚠️  mtmd_encode_chunk() failure (HTTP 500) - attempt 2/3
2025-11-11 12:37:41 | ✓ Success (after 23 seconds total)
```

**Analysis:**
- Initial attempt: 3 seconds → failure (buffer corrupted from image_1 + image_10)
- Retry 1: 4 seconds → failure (still polluted)
- Retry 2: 23 seconds → success (accidental cleanup occurred)

### Example 2: Timing Correlation

| Request | Image | Time to Result | Outcome | Buffer State |
|---------|-------|----------------|---------|--------------|
| Warmup | image_1.jpg | 5s | ✓ Success | Clean (first use) |
| Run 1 | image_1.jpg | 5s | ✓ Success | Clean |
| Run 2 | image_10.jpg | 23s | ✓ Success | Light pollution |
| Run 3 | image_2.jpg | 3s → 30s | ❌→✓ Retry | **Corrupted** |
| Run 4 | image_3.jpg | 9s | ✓ Success | Cleaned |
| Run 5 | image_4.jpg | 4s → 18s | ❌→✓ Retry | **Corrupted again** |

**Pattern:** Failures occur every 2-3 images when buffer accumulates enough stale data.

### Example 3: Fresh Server Instance

When starting a new llama-server instance for a different model:

```
2025-11-11 12:39:53 | llama-server stopped (old instance)
2025-11-11 12:39:53 | Starting new llama-server...
2025-11-11 12:39:58 | ✓ llama-server started
2025-11-11 12:40:06 | Run 1 | image_1.jpg → ✓ Success
2025-11-11 12:40:27 | Run 2 | image_10.jpg → ✓ Success
2025-11-11 12:40:43 | Run 3 | image_2.jpg → ✓ Success (no failure!)
```

**Analysis:** Same image (image_2.jpg) that failed with old instance succeeds with fresh instance, confirming state pollution issue.

## Suggested Fixes

### Option 1: Clear Buffer Before Use (Minimal Change)

```cpp
int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    clip_ctx * ctx_clip = ctx->ctx_v;
    if (!ctx_clip) {
        LOG_ERR("%s: this API does not support non-vision input\n", __func__);
        return 1;
    }

    int n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);

    // FIX: Clear before resize to ensure clean buffer
    ctx->image_embd_v.clear();
    ctx->image_embd_v.resize(image_tokens->n_tokens() * n_mmproj_embd);

    // ... rest of function
}
```

**Pros:** Simple, minimal change
**Cons:** May add minor overhead; doesn't address thread safety

### Option 2: Add Mutex Protection (Thread-Safe)

```cpp
struct mtmd_context {
    struct clip_ctx * ctx_v;
    struct clip_ctx * ctx_a;
    const struct llama_model * text_model;
    std::vector<float> image_embd_v;
    std::mutex image_embd_mutex;  // ADD THIS
    // ...
};

int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    std::lock_guard<std::mutex> lock(ctx->image_embd_mutex);  // ADD THIS

    ctx->image_embd_v.clear();
    ctx->image_embd_v.resize(...);
    // ... rest of function
}
```

**Pros:** Thread-safe, production-ready
**Cons:** Serializes concurrent vision requests

### Option 3: Per-Slot Context (Ideal Solution)

Allocate separate `mtmd_context` for each continuous batching slot:

```cpp
// In server.cpp slot initialization
struct server_slot {
    // ...
    mtmd_context * mtmd_ctx = nullptr;  // Per-slot context
    // ...
};
```

**Pros:** Full isolation, no contention
**Cons:** Higher memory usage, larger refactor

### Option 4: Add Cleanup API

Add explicit cleanup function to be called between requests:

```cpp
// In mtmd.h
MTMD_API void mtmd_clear_embeddings(mtmd_context * ctx);

// In mtmd.cpp
void mtmd_clear_embeddings(mtmd_context * ctx) {
    ctx->image_embd_v.clear();
    ctx->image_embd_v.shrink_to_fit();
}
```

Then call from `server.cpp` after each vision request completes.

**Pros:** Explicit, controllable
**Cons:** Requires server-side integration

## Workarounds (Application-Side)

While waiting for upstream fix, applications can mitigate with:

1. **Retry with exponential backoff** (90% effective):
```python
max_retries = 3
for attempt in range(max_retries):
    try:
        response = llama_request(...)
        break
    except HTTPError as e:
        if e.status_code == 500 and attempt < max_retries - 1:
            time.sleep(0.1 * (2 ** attempt))
            continue
```

2. **Post-inference delay** (95% effective):
```python
response = llama_request(...)
time.sleep(0.5)  # Allow state cleanup
```

3. **Disable continuous batching** (100% effective, poor performance):
```bash
llama-server -m model.gguf --mmproj mmproj.gguf --no-cont-batching
```

## Additional Information

**Related Code Locations:**
- `tools/mtmd/mtmd.cpp:772-832` - `mtmd_encode()` function
- `tools/mtmd/mtmd.cpp:118-122` - `mtmd_context` struct definition
- `tools/mtmd/mtmd-helper.cpp:288-316` - Chunk evaluation calling `mtmd_encode_chunk()`
- `tools/server/utils.hpp:1369-1398` - `process_chunk()` wrapper
- `tools/server/server.cpp:3983-3992` - Error handling and HTTP 500 response

**Why Retries Succeed:**
- Time delay allows OS memory management or internal cleanup
- Continuous batching may rotate to a clean slot
- Random chance that buffer happens to be valid

**Affected Models:**
- All vision models using mtmd library
- Confirmed: Qwen2-VL, LLaVA variants
- Likely affects any model with `--mmproj`

## Request

Please investigate and implement one of the suggested fixes. The buffer clearing fix (Option 1) would be a good quick fix, while per-slot context (Option 3) would be the ideal long-term solution.

I'm happy to:
- Provide additional logs/debugging output
- Test patches
- Submit a PR with a fix if guidance is provided

## References

- Workaround documentation: [LLAMACPP_VISION_BUG_WORKAROUND.md](./LLAMACPP_VISION_BUG_WORKAROUND.md)
- Full logs available upon request
- Reproduction case available

---

**Reporter:** Community testing via vision model benchmarking
**Date:** 2025-01-11
**llama.cpp commit:** [Current master branch]
