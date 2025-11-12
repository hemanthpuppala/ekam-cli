# VLM Inference Pipeline Analysis Report

## 1. Executive Summary

The investigation into incorrect Vision Language Model (VLM) responses has revealed a critical issue in how our application communicates with the `llama.cpp` server. The root cause is the use of an incorrect and likely deprecated API request format for sending image data.

This issue is exacerbated by a faulty caching mechanism that prevents the application from self-correcting. As a result, both the benchmarking and main inference pipelines produce hallucinated, nonsensical results for VLM tasks.

This report details the technical specifics of the failure and provides clear, actionable recommendations to resolve the issue. The primary recommendation is to prioritize the modern, OpenAI-compatible API format for all VLM requests to the `llama.cpp` server.

## 2. The Problem: Incorrect VLM Responses

The initial symptom was observed during VLM benchmarking. When provided with a valid image (e.g., a lion with its prey), the model returned a completely unrelated and hallucinated description ("a hand holding a small, white, rectangular object") followed by an erroneous claim that the image was invalid.

This behavior was confirmed to be present in both the **benchmarking pipeline** and the **main interactive inference pipeline**.

## 3. Root Cause Analysis

The investigation traced the problem to the `QuantizedProvider` and its interaction with the `llama.cpp` server for GGUF-based VLMs. The issue is not with the model itself, but with the data it receives.

### 3.1. The Two Competing Formats

Research into the `llama.cpp` source code and community documentation revealed two different formats for sending VLM requests:

**A. The "Proprietary" Format (What We Use)**

This format appears to be a legacy method used by the `llama.cpp` web UI. It involves sending a text prompt with an `[img-ID]` placeholder and a separate, top-level `image_data` array in the JSON payload.

```json
{
  "prompt": "USER:[img-1]Describe the image...",
  "image_data": [
    {
      "data": "<BASE64_STRING>",
      "id": 1
    }
  ]
}
```

**B. The "OpenAI-Compatible" Format (The Correct Method)**

This is the modern, standard, and officially tested format for the `/v1/chat/completions` endpoint. It mirrors the OpenAI Vision API, embedding the image data directly into the `messages` array.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Describe the image." },
        { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } }
      ]
    }
  ]
}
```

Our investigation of `llama.cpp/tools/server/tests/unit/test_vision_api.py` confirms that the **OpenAI format is the primary, validated method** for API interactions, while the "proprietary" format is not covered in these crucial tests.

### 3.2. Our Application's Mistake

The `_run_qa_gguf_server` method within `src/providers/quantized.py` is responsible for VLM inference. This method contains logic to try multiple formats. However, it incorrectly prioritizes the "proprietary" format.

For the VLM being tested (a `Qwen3-VL` variant), this "proprietary" format is not correctly interpreted by the `llama.cpp` server. The server does not crash, but it fails to process the image, leading the model to receive invalid input and subsequently hallucinate.

### 3.3. The Faulty Caching Mechanism

The problem is compounded by a critical flaw in the format detection logic. After the first attempt with the "proprietary" format, the code checks only for a successful HTTP response, not for the quality of the content returned.

Since the server returns a 200 OK status with a hallucinated response, our code incorrectly assumes the format is valid and **caches it as the working method**.

```python
# In src/providers/quantized.py

if isinstance(response, dict) and "choices" in response:
    answer = response["choices"][0]["message"]["content"]
    # !!! BUG: Caches the format as successful without validating the answer !!!
    if model_key:
        self._vision_api_format_cache[model_key] = format_name
    return answer.strip()
```

This faulty cache prevents the application from ever trying the correct "OpenAI" format on subsequent requests.

## 4. Impact on Pipelines

The issue affects both pipelines due to the shared `QuantizedProvider` instance and its stateful cache.

### 4.1. Benchmarking Pipeline
The benchmark runs first, triggering the format detection. It tries the "proprietary" format, gets a garbage response, and "poisons the well" by incorrectly caching this failing format as a success.

### 4.2. Main Inference Pipeline
When the user later runs the main QA endpoint, it uses the same provider instance. The code finds the "proprietary" format in the cache and uses it, resulting in the same hallucinated output seen during the benchmark. The pipeline is a victim of the bad state left by the benchmark run.

## 5. LLM vs. VLM Inference

This issue is **strictly limited to VLM inference** for GGUF models. Standard LLM (text-only) inference is unaffected because it uses a much simpler API payload that does not involve passing image data.

## 6. Recommendations

To resolve this issue and make our VLM inference pipeline robust, the following changes are required:

### 6.1. Prioritize OpenAI-Compatible Format (Critical)

The format detection logic in `_run_qa_gguf_server` within `src/providers/quantized.py` must be reordered to prioritize the most reliable and standard format.

**Action:** Modify the `formats_to_try` list to attempt the `"openai"` format **first**.

```python
# Change this:
formats_to_try = [
    ("proprietary", ...),
    ("openai", ...),
    ("simple", ...),
]

# To this:
formats_to_try = [
    ("openai", messages, None),
    ("proprietary", proprietary_messages, proprietary_image_data),
    ("simple", simple_messages, None),
]
```

### 6.2. Implement Response Validation (Highly Recommended)

To prevent the faulty caching from happening again, the code should validate the content of the VLM's response before caching the format.

**Action:** Before caching a format as successful, check the response for keywords that indicate an error or inability to process the image.

```python
# In src/providers/quantized.py, before caching

answer = response["choices"][0]["message"]["content"]
error_keywords = ["cannot", "unable to process", "not a valid image", "do not see an image"]

# Check if the response looks like a hallucinated error message
is_error_response = any(keyword in answer.lower() for keyword in error_keywords)

if not is_error_response:
    # Only cache the format if the response seems valid
    if model_key:
        self._vision_api_format_cache[model_key] = format_name
    logger.info(f"✓ GGUF VLM inference successful ({format_name} format)")
    return answer.strip()
else:
    # Log the failure and allow the loop to try the next format
    logger.warning(f"Format {format_name} produced a suspected error response: {answer}")
    last_error = RuntimeError(f"Format {format_name} failed with VLM error response.")
    continue # This is crucial
```

### 6.3. Clear the Cache

To apply the fix immediately without restarting the application, the faulty cache entry must be cleared. The code change should ideally handle this, but as a principle, the `_vision_api_format_cache` dictionary on the `QuantizedProvider` instance needs to be reset.
