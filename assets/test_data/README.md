# Test Data Structure for VLM Benchmarking

This directory contains the structured test data for VLM (Vision Language Model) benchmarking with endpoint-specific image-prompt mapping.

## Directory Structure

```
assets/test_data/
├── visual_qa/
│   └── images/          (Add .jpg, .png, .jpeg images here)
├── caption/
│   └── images/          (Add .jpg, .png, .jpeg images here)
├── detect/
│   └── images/          (Add .jpg, .png, .jpeg images here)
├── point/
│   └── images/          (Add .jpg, .png, .jpeg images here)
├── test_data_config.json (Configuration file - DO NOT DELETE)
└── README.md (This file)
```

## Endpoints and Prompts

### 1. Visual Q&A (`visual_qa/`)
**Endpoint:** `vision/qa`

**Purpose:** Ask questions about images

**Prompts:**
- Describe this image in detail.
- What objects can you see in this image?
- What is the main subject of this image?
- What colors are prominent in this image?
- What activity or scene is depicted in this image?
- Count how many people are in this image.
- Describe the setting or location shown.
- What is the mood or atmosphere of this image?
- Identify any brands or logos visible.

**How to use:**
1. Add images to `visual_qa/images/`
2. Images will be automatically paired with prompts (cycling if needed)
3. Example:
   - `image_001.jpg` → paired with prompt 1
   - `image_002.jpg` → paired with prompt 2
   - `image_003.jpg` → paired with prompt 3 (cycles back if more images than prompts)

### 2. Image Captioning (`caption/`)
**Endpoint:** `vision/caption`

**Purpose:** Generate captions/descriptions for images

**Prompts:**
- Provide a brief caption for this image.
- Write a one-sentence description of what's in this image.
- Summarize the main content of this image in a few words.
- Create a descriptive title for this image.
- Describe the primary subject in a concise way.
- What would be a good caption for this social media post?
- Write a news headline for this image.

### 3. Object Detection (`detect/`)
**Endpoint:** `vision/detect`

**Purpose:** Identify and list objects in images

**Prompts:**
- What objects are in the foreground?
- What objects are in the background?
- Identify any vehicles in the image.
- Are there any animals in this image? If so, how many?
- What furniture can you see?
- List all the distinct objects you can identify.
- What is the largest object in this image?

### 4. Object Pointing (`point/`)
**Endpoint:** `vision/point`

**Purpose:** Locate objects and describe spatial layout

**Prompts:**
- Where is the main subject located in this image?
- Point to the center of the image and describe what's there.
- What is positioned in the top-left corner?
- Identify the location of any people in this image.
- Where are the largest objects positioned?
- Describe the spatial layout of objects in this image.

## Adding Images

1. Choose the appropriate endpoint directory based on your use case
2. Place images in the `images/` subdirectory
3. Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.webp`
4. Images are automatically discovered and sorted alphabetically
5. Each image will be paired with prompts cyclically

### Example

If you want to test Visual Q&A:

```bash
cp your_images/*.jpg assets/test_data/visual_qa/images/
```

Then during benchmarking:
- Step 4: Select endpoint → "Visual Q&A"
- Step 5: Select test data → "Structured test data (N images)"

## Configuration

The `test_data_config.json` file defines all prompts and endpoints. **Do not delete this file.**

To view the current configuration:
```bash
cat assets/test_data/test_data_config.json
```

## Important Notes

- **Permanent Structure:** This directory is a permanent fixture. Do not delete it.
- **Image-Prompt Pairing:** Images are automatically paired 1-to-1 with prompts
- **Cycling:** If you have more images than prompts, prompts will cycle
- **Auto-Discovery:** Images are automatically detected when benchmarking runs
- **Sorting:** Images are sorted alphabetically for consistency

## Usage in Benchmarking

During the benchmark configuration:

1. Select "Vision Language Models (image + text)" as model type
2. Select your benchmark suite
3. Select your VLM models
4. Select the endpoint (will determine which test data directory is used)
5. In Step 5 (Test Data Configuration):
   - If images exist in the selected endpoint's directory, "Structured test data" option will appear
   - Select "Structured test data" to use images + prompts from `test_data_config.json`

## Troubleshooting

**"Structured test data" option not appearing?**
- Ensure images are in the correct subdirectory: `assets/test_data/{endpoint}/images/`
- Check that file extensions are supported (.jpg, .png, .jpeg, etc.)
- Restart the application if you just added images

**Images not being detected?**
- Check file extensions (case-sensitive on some systems)
- Ensure images are in the `images/` subdirectory, not the endpoint root
- Use `ls assets/test_data/visual_qa/images/` to verify images are present




 How VLM Quantization Works

  Architecture

  VLMs have TWO separate components:

  1. Language Decoder (LLM part)
    - The main transformer model that generates text
    - Can be aggressively quantized: Q4_K_M, Q5_K_M, Q6_K, Q8_0
  2. Vision Encoder (mmproj/CLIP part)
    - Converts images to embeddings the LLM understands
    - More sensitive to quantization than the language part
    - Typically kept at F16 or Q8_0 for quality

  Why Separate Files?

  From llama.cpp developers:
  "The multimodal model uses another network to extract features, and this process is more sensitive to disturbances. If a picture is compressed to 
  fewer tokens, the impact on vision part quantization is much greater than the impact on LLM."

  Conversion Workflow for HuggingFace VLMs

  # Step 1: Convert HF model to GGUF (creates F16 base)
  python convert_hf_to_gguf.py ./Qwen2-VL-2B-Instruct \
    --outfile ./qwen2-vl-2b-instruct-f16.gguf

  # Step 2: Run "surgery" to extract vision encoder
  # This creates TWO files:
  #   - qwen2-vl-2b-instruct-f16.gguf (language model only)
  #   - qwen2-vl-2b-instruct-mmproj-f16.gguf (vision encoder)
  python examples/llava/qwen2_vl_surgery.py ./Qwen2-VL-2B-Instruct

  # Step 3: Quantize language model (aggressive)
  ./llama-quantize qwen2-vl-2b-instruct-f16.gguf \
    qwen2-vl-2b-instruct-q4_k_m.gguf Q4_K_M

  # Step 4: Optionally quantize vision encoder (conservative)
  ./llama-quantize qwen2-vl-2b-instruct-mmproj-f16.gguf \
    qwen2-vl-2b-instruct-mmproj-q8_0.gguf Q8_0

  Final Files Structure

  models/
  ├── qwen2-vl-2b-instruct-q4_k_m.gguf       # Language: Q4_K_M (2.5GB)
  └── qwen2-vl-2b-instruct-mmproj-q8_0.gguf  # Vision: Q8_0 (1GB)
