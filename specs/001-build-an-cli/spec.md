# Feature Specification: High-Performance Multi-Provider VLM/LLM CLI with Resource Management

**Feature Branch**: `001-build-an-cli`
**Created**: 2025-10-11
**Status**: Draft
**Input**: User description: "build an CLI application with O(1) complexity in every single operation which can help user to select which provider to select -> which model to select from that provider which are pre installed and an option to install any new model required by the user -> use all endpoints supported by the vlm/llm (like qa, caption, point, detect, text only for vlms) or just text enpoint for llms -> give image path (if a vlm is selected) or a prompt for a LLM -> give response with infernce speed -> again show the endpoint table. (For reference use this MVP on this project: /Users/hemanth/Desktop/road-condition-analyzer/vlm-tester/Refernce files/ read the code files in this path). Should load and unload models dynamically for best memory allocation based on the system constraints - take consideration of all cpu/gpu/memopry and then prompt the user to warn about their memory constraints before loading any model or before installing any new model. dynammically check which models are preinstalled in user's memoery like with all providfers ollama, hf, and any other providers and show them in th eCLI. should be production ready and  proper modular sturcutre. On every start up of this application unload all the models and providers and then load upon usage."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Quick Model Selection with Resource Awareness (Priority: P1)

A data scientist wants to quickly test different VLM models on their local machine without running out of memory. They need to see at a glance which models are installed, which will fit in their available RAM, and switch between models efficiently.

**Why this priority**: Core value proposition - enables users to work within system constraints without crashes or performance degradation. This is the foundation for all other features.

**Independent Test**: Can be fully tested by launching the CLI, viewing system specs with available memory, seeing installed models with compatibility status (✅⚠️❌), and selecting a model that loads successfully. Delivers immediate value by preventing memory issues.

**Acceptance Scenarios**:

1. **Given** the CLI starts with 8GB total RAM and 6GB available, **When** user launches the application, **Then** system displays CPU, GPU, and memory specs with recommended model size threshold
2. **Given** user has Ollama and HuggingFace models installed, **When** viewing the model selection menu, **Then** all discovered models are listed with type (VLM/LLM/Embedding), size, and CompatibilityStatus (✅ PERFECT_FIT, ⚠️ TIGHT_FIT, ❌ TOO_LARGE)
3. **Given** a model is marked as "too large" for current system, **When** user attempts to select it, **Then** system displays prominent warning with specific memory requirement vs. available memory, suggests alternatives, and requires explicit "yes, proceed anyway" confirmation before loading
4. **Given** user selects a compatible model, **When** the model loads, **Then** loading completes in reasonable time and endpoint menu appears
5. **Given** a different model is already loaded, **When** user switches models, **Then** previous model is unloaded first, memory is freed, then new model loads

---

### User Story 2 - Multi-Endpoint Vision Testing Workflow (Priority: P1)

A computer vision researcher needs to test different VLM capabilities (question answering, captioning, object detection, object pointing) on the same image to evaluate model performance across tasks.

**Why this priority**: Primary use case for VLM testing - enables comprehensive evaluation of vision-language capabilities. Essential for MVP.

**Independent Test**: Can be fully tested by selecting a VLM model, running QA with an image and question, seeing inference time, then immediately running caption on same image, then detect, then point - all showing results and timing independently.

**Acceptance Scenarios**:

1. **Given** a VLM is loaded, **When** user selects QA endpoint and provides image path and question, **Then** system displays answer with inference time in milliseconds
2. **Given** the same VLM session, **When** user selects Caption endpoint with "detailed" level, **Then** system generates detailed image description with inference time
3. **Given** the same VLM session, **When** user selects Detect endpoint with object name "car", **Then** system locates and describes car instances with inference time, saves annotated image, and displays clickable file path
4. **Given** the same VLM session, **When** user selects Point endpoint with object name "stop sign", **Then** system returns coordinates or location information with inference time, saves annotated image, and displays clickable file path
5. **Given** any VLM endpoint completes, **When** results are displayed, **Then** user is prompted "Continue with same model? [Y/n]" and can either continue with endpoints or return to model selection
6. **Given** user completes an inference, **When** prompted to save results, **Then** results are saved to JSON file with timestamp, model info, and all metadata

---

### User Story 3 - LLM Text-Only Workflow (Priority: P2)

A developer wants to test pure text-based language models for natural language tasks without vision capabilities, using the same interface as VLM testing.

**Why this priority**: Extends the tool's utility to pure LLMs, enabling broader model testing. Important but not critical for initial VLM-focused MVP.

**Independent Test**: Can be fully tested by selecting an LLM (non-vision) model, seeing that only text endpoint is available (no VLM endpoints), providing a text prompt, and receiving a text response with timing.

**Acceptance Scenarios**:

1. **Given** an LLM model is selected, **When** endpoint menu displays, **Then** only "Text" endpoint is shown (no QA, Caption, Detect, Point options)
2. **Given** LLM's Text endpoint is selected, **When** user provides a text prompt, **Then** system generates text response with inference time without requiring image input
3. **Given** an LLM model is loaded, **When** user attempts to access VLM endpoints, **Then** system prevents access and displays clear message that model lacks vision capabilities

---

### User Story 4 - Model Installation with Resource Pre-Check (Priority: P2)

A user discovers they need a specific model that isn't installed yet. They want to install new models directly from the CLI while being warned about resource requirements before downloading large files.

**Why this priority**: Enables self-service model management and prevents wasted downloads of incompatible models. Enhances user experience but core functionality works without it.

**Independent Test**: Can be fully tested by selecting "Install New Model" option, entering a model name, seeing estimated size and CompatibilityStatus assessment, confirming installation, and having the model download and auto-load.

**Acceptance Scenarios**:

1. **Given** user selects "Install New Model" from model menu, **When** they enter a model name (e.g., "llama3.2-vision:11b"), **Then** system estimates model size and shows compatibility assessment before download
2. **Given** a model size estimate shows "⚠️ tight fit", **When** user reviews warnings, **Then** specific warnings are displayed (e.g., "May use 90%+ RAM", "Performance may be affected")
3. **Given** a model is estimated as too large, **When** CompatibilityStatus assessment returns TOO_LARGE, **Then** system displays prominent warning with clear explanation and alternative suggestions, then prompts user for explicit "yes, proceed anyway" confirmation before allowing installation
4. **Given** a model passes CompatibilityStatus assessment, **When** download completes successfully, **Then** model is automatically loaded and ready for immediate use
5. **Given** Ollama models are being installed, **When** download is in progress, **Then** system displays real-time progress with percentage and MB downloaded

---

### User Story 5 - Model Cleanup and Management (Priority: P3)

A user with limited disk space wants to remove unused models to free up storage and manage their local model collection efficiently.

**Why this priority**: Housekeeping feature that improves long-term usability. Nice to have but not essential for core testing workflow.

**Independent Test**: Can be fully tested by selecting "Delete a Model" option, choosing a model from the list, confirming deletion, and verifying the model is removed from both disk and the available models list.

**Acceptance Scenarios**:

1. **Given** user selects "Delete a Model" from model menu, **When** model list displays, **Then** all installed models are shown with their disk sizes
2. **Given** user selects a model to delete, **When** confirmation prompt appears, **Then** model name and size are clearly displayed before requiring confirmation
3. **Given** user confirms deletion, **When** deletion completes, **Then** disk space is freed and model no longer appears in available models list
4. **Given** the currently loaded model is deleted, **When** deletion completes, **Then** model is unloaded first, then deleted, then user is prompted to select a different model

---

### User Story 6 - Performance Monitoring and Statistics (Priority: P3)

A researcher running multiple experiments wants to track overall usage statistics including total inferences run, cumulative time spent, and average inference speeds across different models.

**Why this priority**: Analytical feature for power users conducting extensive testing. Useful for research but not required for basic testing workflow.

**Independent Test**: Can be fully tested by running several inferences across different endpoints, then viewing statistics screen that shows total inference count, total time, average time, current model, and provider.

**Acceptance Scenarios**:

1. **Given** user has run 15 inferences, **When** viewing statistics screen, **Then** total count, cumulative time, and average time per inference are displayed
2. **Given** user switches models, **When** statistics are viewed, **Then** current model name and provider are accurately reflected
3. **Given** user runs inferences with different models, **When** statistics are calculated, **Then** average time reflects all inferences across all models in the session

---

### Edge Cases

- **What happens when Ollama service is not running?** System displays clear error message indicating Ollama is not accessible with link to installation/startup instructions, prevents hanging or cryptic errors
- **What happens when user provides invalid image path?** System validates path immediately and shows "Image not found" error with prompt to re-enter, does not attempt inference
- **What happens when model loading fails mid-process?** System catches failure, displays specific error message, frees any partially loaded resources, and returns user to model selection menu
- **What happens when available RAM drops during operation?** System checks memory before each model load; if insufficient, displays warning and suggests closing other applications or selecting smaller model
- **What happens when HuggingFace cache directory doesn't exist?** System creates cache directory structure automatically on first HuggingFace model installation
- **What happens when user interrupts (Ctrl+C) during inference?** System catches interrupt, displays "Interrupted" message, safely returns to endpoint menu without crashing
- **What happens when GPU is available but user wants CPU-only?** System respects configuration settings for device selection (GPU/CPU/MPS) from config file
- **What happens when trying to delete a model that's currently loaded?** System automatically unloads model first, confirms unload succeeded, then proceeds with deletion

## Requirements *(mandatory)*

### Functional Requirements

#### Provider & Model Management

- **FR-001**: System MUST discover and display all installed models from Ollama provider by querying Ollama's REST API at configured host (default: localhost:11434)
- **FR-002**: System MUST discover and display all installed models from HuggingFace Transformers (local) by scanning configured cache directory (default: ~/.cache/huggingface/hub/)
- **FR-003**: System MUST discover and display all installed models from LM Studio provider by querying LM Studio's OpenAI-compatible API at configured host (default: localhost:1234)
- **FR-004**: System MUST discover and display all installed GGUF model files by scanning configured local directories and optionally querying HuggingFace repositories for available GGUF models
- **FR-005**: System MUST detect and classify each model as VLM (Vision Language Model), LLM (Language Model), or Embedding model based on model metadata and naming patterns
- **FR-005a**: System MUST query and detect specific endpoint capabilities (QA, Caption, Detect, Point, Text) for each model on load by checking provider metadata and model configuration
- **FR-006**: System MUST display model information including name, provider, type, size in GB, and compatibility status for all discovered models
- **FR-007**: Users MUST be able to install new models from any provider through the CLI interface (Ollama: via ollama pull, HF: via transformers download, GGUF: via HuggingFace Hub download to local cache)
- **FR-008**: Users MUST be able to delete installed models to free disk space (provider-dependent: some providers like LM Studio may not support programmatic deletion)
- **FR-009**: System MUST display prominent warning for models that exceed available system memory (TOO_LARGE status) with CompatibilityStatus assessment before download, and require explicit user confirmation ("yes, proceed anyway") to continue installation
- **FR-010**: Launcher script (run.sh) MUST detect installation status of all four providers (Ollama, HuggingFace Transformers, LM Studio, llama-cpp-python) before CLI launch
- **FR-011**: Launcher script MUST offer interactive installation assistance for missing providers with user confirmation (download links for Ollama/LM Studio, pip install commands for Python packages)

#### Resource Management & System Awareness

- **FR-012**: System MUST detect and display system specifications on startup including platform, CPU cores, total RAM, available RAM, GPU name (if present), and GPU type (CUDA/MPS/none)
- **FR-013**: System MUST calculate and display recommended maximum model size based on available RAM minus safety margin (formula: min(2.0, max(0.5, total_ram * 0.1)) GB)
- **FR-014**: System MUST check model compatibility before loading using three-tier system: Perfect Fit (✅ <70% of recommended size), Tight Fit (⚠️ 70-100%), or Too Large (❌ >100%), and require explicit user confirmation for TOO_LARGE models before proceeding
- **FR-015**: System MUST unload all models on application startup to establish clean memory baseline
- **FR-016**: System MUST unload previous model before loading new model when user switches models
- **FR-017**: System MUST free GPU/CPU memory caches after unloading models (torch.cuda.empty_cache for NVIDIA, torch.mps.empty_cache for Apple Silicon)
- **FR-018**: System MUST display real-time memory warnings when attempting to load models that will use >90% of available RAM, with advisory notices that allow user to proceed after explicit confirmation

#### VLM Endpoints (Vision Language Models)

- **FR-019**: System MUST provide Question Answering (QA) endpoint that accepts image path and question text, returns answer (only displayed in menu if model supports it)
- **FR-020**: System MUST provide Caption endpoint that accepts image path and detail level (detailed/short), returns image description (only displayed in menu if model supports it)
- **FR-021**: System MUST provide Detect endpoint that accepts image path and object name, returns object detection results (only displayed in menu if model supports it)
- **FR-022**: System MUST provide Point endpoint that accepts image path and object name, returns object center coordinates (x, y) in pixel space, or bounding box (x1, y1, x2, y2) if exact center unavailable (only displayed in menu if model supports it)
- **FR-023**: System MUST validate image paths before attempting inference and reject invalid paths immediately
- **FR-024**: System MUST auto-resize images exceeding maximum dimension (default 1920px) to prevent memory issues, preserving aspect ratio using bicubic interpolation
- **FR-025**: System MUST convert images to RGB format if needed for model compatibility

#### LLM Endpoints (Text-Only Models)

- **FR-026**: System MUST provide Text endpoint capability accessible through all LLM-supporting providers that accepts text prompt and returns text response
- **FR-027**: System MUST hide VLM endpoints (QA, Caption, Detect, Point) when LLM model is loaded
- **FR-028**: System MUST display clear message when user attempts VLM endpoints with LLM model

#### Performance & Timing

- **FR-029**: System MUST measure and display inference time in milliseconds for every endpoint execution
- **FR-030**: System MUST track cumulative statistics including total inferences, total time spent, and average time per inference
- **FR-031**: System MUST display statistics on demand showing current model, provider, inference count, cumulative time, and average time

#### User Interface & Workflow

- **FR-032**: System MUST display main menu with clear provider selection (Ollama, HuggingFace Transformers, LM Studio, GGUF) using Rich library for formatted tables and standard input() for user interaction, with back/exit options
- **FR-033**: System MUST display model selection menu with numbered options, model details in Rich-formatted table, with back option to return to provider selection
- **FR-034**: System MUST display endpoint selection menu after model load with only the endpoints supported by the loaded model (dynamically determined from model capabilities), using Rich panels and standard input() prompts, with back option to return to model selection
- **FR-035**: System MUST prompt after each inference completion "Continue with same model? [Y/n]" - if yes, redisplay endpoint menu; if no, return to model selection menu
- **FR-036**: System MUST provide back navigation option in all menus to return to previous menu level
- **FR-037**: System MUST provide "View Statistics" option accessible from endpoint menu
- **FR-038**: System MUST provide "Exit" option that cleanly unloads models and terminates application, accessible from all menu levels
- **FR-039**: System MUST handle keyboard interrupts (Ctrl+C) gracefully without crashes
- **FR-040**: System MUST save annotated/processed images to files (for Detect/Point endpoints) and display full file paths as output, using terminal hyperlinks (OSC 8 escape sequences) when supported to make paths clickable

#### Results & Persistence

- **FR-041**: System MUST offer to save results to JSON file after each inference
- **FR-042**: System MUST save results with complete metadata including endpoint type, question/prompt, answer/response, inference time, model name, provider, timestamp, and image file paths (for VLM endpoints)
- **FR-043**: System MUST save results to organized directory structure with timestamp-based filenames

#### Error Handling & Reliability

- **FR-044**: System MUST display actionable error messages when Ollama service is not accessible, including command to start service (`ollama serve`) and link to https://ollama.ai/docs if not installed
- **FR-045**: System MUST display specific error messages when model loading fails (network, memory, compatibility issues)
- **FR-046**: System MUST recover gracefully from inference failures without crashing the application
- **FR-047**: System MUST validate all user inputs before processing: image paths (existence, format), model names (valid for provider), endpoint selections (compatible with model type)
- **FR-048**: System MUST provide helpful examples for model installation (valid model names for each provider)
- **FR-049**: System MUST perform menu navigation and model selection operations in O(1) constant time (inference time excluded as model-dependent); specifically: provider selection, model lookup from pre-loaded list, endpoint routing, and statistics queries must use constant-time data structures (hash maps, direct array access)

### Performance Assumptions

- **Assumption PA-001**: CLI operations (menus, lookups) achieve O(1) via index-based access and hash maps (see FR-049)
- **Assumption PA-002**: Model selection from pre-loaded list is O(1) via index-based array access
- **Assumption PA-003**: System spec lookup is O(1) as it queries system APIs once and caches results
- **Assumption PA-004**: Endpoint routing to appropriate model method is O(1) via direct function mapping
- **Assumption PA-005**: The actual inference time is model-dependent and cannot be O(1), but system minimizes overhead to ensure inference time dominates total execution time

### Key Entities

- **Provider**: Represents a model provider (Ollama or HuggingFace). Attributes include name, connection host/URL, cache directory path, availability status.
- **Model**: Represents an AI model. Attributes include name, provider reference, model type (VLM/LLM/Embedding), size in GB, installation status, CompatibilityStatus (PERFECT_FIT/TIGHT_FIT/TOO_LARGE), capabilities list (dynamically detected: qa, caption, detect, point, text).
- **System Specs**: Represents hardware capabilities. Attributes include platform name, CPU core count, total RAM, available RAM, GPU availability, GPU name, recommended model size threshold.
- **Inference Result**: Represents the output of a model operation. Attributes include endpoint type, input (question/prompt/object name), output (answer/caption/detections/coordinates), inference time, model name, provider, timestamp, metadata.
- **Session Statistics**: Represents cumulative usage data. Attributes include current model, current provider, total inference count, total time, average time.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can identify compatible models for their system within 5 seconds of application launch by viewing system specs and model compatibility status
- **SC-002**: Users can switch between models in under 10 seconds including unload and load time for models under 5GB
- **SC-003**: System warns users about 100% of potentially incompatible models (TOO_LARGE status) before loading/installation, requiring explicit confirmation to proceed, reducing unintended out-of-memory errors while allowing advanced users to override
- **SC-004**: Users can complete full testing workflow (select provider → select model → run 4 different endpoints → save results) in under 3 minutes for typical VLM models
- **SC-005**: Application startup completes in under 3 seconds on systems with 8GB RAM, including system spec detection and model discovery
- **SC-006**: Memory overhead for the CLI application itself is less than 200MB, with remaining RAM available for model loading
- **SC-007**: Users can run continuous testing sessions for 2+ hours without memory leaks causing degradation
- **SC-008**: 95% of user actions (menu selections, endpoint routing) complete in under 100ms, excluding actual model inference time
- **SC-009**: Error messages provide actionable next steps in 100% of failure cases (invalid paths, incompatible models, connection failures)
- **SC-010**: New users can successfully complete their first VLM inference within 5 minutes without external documentation by following in-app prompts and examples

### Acceptance Criteria for "Production Ready"

- **AC-001**: No uncaught exceptions that crash the application during normal operation
- **AC-002**: All user inputs are validated before processing
- **AC-003**: All external dependencies (Ollama, HuggingFace cache) are checked for availability with helpful error messages when missing
- **AC-004**: Memory is properly released when switching or unloading models
- **AC-005**: Results are persistently saved with all necessary metadata for reproducibility
- **AC-006**: Application handles edge cases gracefully (missing images, model load failures, connection timeouts)
- **AC-007**: User interface provides clear feedback for all operations (loading states, progress indicators, completion messages)
- **AC-008**: System supports multiple hardware configurations (CPU-only, NVIDIA GPU, Apple Silicon GPU) without code changes

## Assumptions

- **ASM-001**: Users have either Ollama or HuggingFace Transformers library installed; the system will guide them if missing
- **ASM-002**: Users understand basic ML terminology (model, inference, VLM vs LLM concepts)
- **ASM-003**: Model inference time varies by model size and hardware; "O(1)" applies only to CLI operations, not inference itself
- **ASM-004**: Users have at least 4GB of RAM available; the system will work but heavily constrain model choices below this threshold
- **ASM-005**: Internet connection is available for model installation but not required for using pre-installed models
- **ASM-006**: Standard Python environment (3.8+) with common libraries (PIL, PyTorch, Transformers) is available or installable
- **ASM-007**: Users have appropriate permissions to read/write to cache directories and model storage locations
- **ASM-008**: Terminal supports rich formatting; graceful fallback to plain text if rich library is unavailable

## Clarifications *(captured 2025-10-11)*

### CL-001: Local-Only Provider Architecture (Critical)

**Question**: Should the application support cloud/API-based providers or only local inference?

**Answer**: **LOCAL INFERENCE ONLY**. All providers must run models locally on the user's hardware without any cloud/API inference. No cloud providers (HuggingFace API, Replicate, Together AI, Groq, Fal.ai) should be integrated.

**Rationale**: User explicitly specified "all the providers and llm/vlm from any providers should be installed locally and run locally without any api usage or cloud inference."

**Impact**:
- Removes all cloud provider implementations from architecture
- Ensures complete data privacy (no data sent to external APIs)
- Eliminates dependency on internet connection for inference (only needed for model downloads)
- Focuses optimization efforts on local execution efficiency

### CL-002: Supported Local Providers

**Question**: Which local model providers should be integrated?

**Answer**: **Four local providers**:
1. **Ollama** - Primary local provider with REST API, automatic model management
2. **HuggingFace Transformers (local)** - Python library for local model execution via transformers pipeline
3. **LM Studio** - Desktop application with OpenAI-compatible API for local inference
4. **Direct GGUF Loading** - Native support for GGUF model files without external provider dependencies

**Rationale**: These providers cover the full spectrum of local inference options:
- Ollama: User-friendly, handles downloads, popular in community
- HuggingFace local: Maximum model compatibility, Python-native
- LM Studio: GUI users who want programmatic access
- GGUF: Direct access to quantized models, maximum flexibility

**Impact**:
- Four provider implementations required (all implementing BaseProvider interface)
- Launcher script (run.sh) must check installation status of all four
- Configuration schema needs sections for each provider
- Model discovery logic differs per provider (API vs filesystem vs cache scanning)

### CL-003: GGUF Model Discovery & Management

**Question**: How should the application discover and manage GGUF model files?

**Answer**: **Hybrid approach with HuggingFace integration**:
- Support local GGUF directories specified in config.yaml (e.g., `~/models/gguf/`)
- Support downloading GGUF models directly from HuggingFace repositories (e.g., `TheBloke/Llama-2-7B-GGUF`)
- Cache downloaded models in local directory for reuse
- Scan and index GGUF files with metadata extraction from filenames

**Rationale**: Provides flexibility for users who already have GGUF files locally while enabling easy discovery and download of popular quantized models from HuggingFace.

**Impact**:
- GGUF provider must implement both filesystem scanning and HuggingFace API integration
- Model discovery scans both local directories and can query HF repos
- Installation workflow downloads .gguf files from HuggingFace to local cache
- Metadata extraction needed from GGUF filenames (model name, parameter count, quantization level)

### CL-004: GGUF Inference Backend

**Question**: Which backend library should execute GGUF model inference?

**Answer**: **llama-cpp-python** (Python bindings for llama.cpp)

**Rationale**: Selected for optimal edge device support:
- In-process execution (no separate server overhead, critical for RAM-constrained devices like Raspberry Pi)
- Highly optimized C++ backend with CPU/GPU acceleration
- Native GGUF support with built-in quantization (4-bit, 8-bit)
- Universal compatibility (x86_64, ARM, Apple Silicon)
- Automatic Metal (macOS) and CUDA (NVIDIA) support
- Simpler deployment (no external process management)

**Threading Strategy**: Async wrapper using threading to prevent blocking main CLI thread during inference.

**Impact**:
- Add llama-cpp-python to requirements.txt (with optional CUDA/Metal builds)
- GGUF provider wraps llama-cpp-python API
- Edge device optimization automatically handled by llama.cpp backend
- No separate llama-server process management needed

### CL-005: Launcher Pre-flight Checks & Interactive Installation

**Question**: What level of installation checking should the launcher (run.sh) perform?

**Answer**: **Interactive installation assistance**:
- Detect missing providers (Ollama, LM Studio, llama-cpp-python)
- Test connectivity for API-based providers (Ollama server running, LM Studio server accessible)
- Check Python package availability (transformers, llama-cpp-python)
- **Offer to install/download missing components** with user confirmation:
  - "Ollama not found. Visit https://ollama.ai to download? [y/N]"
  - "LM Studio not detected. Download from https://lmstudio.ai? [y/N]"
  - "llama-cpp-python not installed. Install via pip now? [y/N]"
- Auto-install Python packages if user confirms
- Display clear status summary before launching CLI

**Rationale**: Reduces friction for new users by guiding them through setup process rather than failing silently. Interactive prompts respect user control while providing helpful automation.

**Impact**:
- Launcher script becomes more complex with installation detection and offering logic
- Must handle different installation methods per provider (external downloads vs pip installs)
- Needs internet connectivity checks before offering downloads
- Should cache installation status to avoid repeated prompts
- Provides superior onboarding experience for first-time users

### CL-006: Provider Architecture Summary

**Final Provider Set** (all local-only per CL-001):
1. **Ollama** - Local server with REST API
2. **HuggingFace Transformers (local)** - Python library for local inference
3. **LM Studio** - Desktop app with OpenAI-compatible local API
4. **GGUF direct loading** - llama-cpp-python for edge-optimized inference

All provider requirements have been integrated into FR-001 through FR-011 above.

### Session 2025-10-11 (Clarification Update)

- Q: UI implementation approach - remove Textual library and use normal CLI as in reference files? → A: Rich library for formatted tables/menus with standard input() for interaction (hybrid approach)
- Q: Should system constraints block model loading/installation or only warn? → A: Advisory warnings with user override - show warning, require explicit "yes, proceed anyway" confirmation for TOO_LARGE models
- Q: How should images be displayed when users run VLM endpoints? → A: Don't display images in terminal - save annotated images to files and print full file paths (clickable if terminal supports hyperlinks)
- Q: How should the continuous workflow loop work after each inference? → A: After each inference, prompt "Continue with same model? [Y/n]" - yes shows endpoints, no returns to model selection. All menus must have back option
- Q: How should the system handle VLM models that don't support all four endpoints? → A: Query provider/model capabilities on load, only display supported endpoints in menu (dynamic per model)
