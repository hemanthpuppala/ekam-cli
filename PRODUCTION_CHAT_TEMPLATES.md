# Production-Level Chat Templates

## Overview

The application now uses **production-level role-based chat templates** like ChatGPT, Claude, and DeepSeek. The system automatically detects the right template format based on the model being used.

## What Changed

### ❌ Old System (Simple Format)
```
User: hi
Assistant: Hello!
User: how are you
Assistant:
```

**Problems**:
- Too simple for modern LLMs
- Models treated it as text to analyze, not conversation to continue
- DeepSeek-R1 showed meta-commentary about the conversation instead of responding naturally

### ✅ New System (Production Templates)
```xml
<|im_start|>system
You are a helpful AI assistant. Provide clear, accurate, and concise responses.<|im_end|>
<|im_start|>user
hi<|im_end|>
<|im_start|>assistant
Hello!<|im_end|>
<|im_start|>user
how are you<|im_end|>
<|im_start|>assistant
```

**Benefits**:
- ✅ Matches training format of modern LLMs
- ✅ Models understand this is a conversation, not text to analyze
- ✅ Natural, context-aware responses
- ✅ System prompts for better instruction following
- ✅ Auto-detects correct format per model

---

## Supported Templates

### 1. ChatML (Default for Modern Models)

**Used by**: DeepSeek, Qwen, Qwen2, Yi, OpenChat, Mixtral, Mistral, Phi, GPT-4

**Format**:
```xml
<|im_start|>system
You are a helpful AI assistant. Provide clear, accurate, and concise responses.<|im_end|>
<|im_start|>user
what is a LLM<|im_end|>
<|im_start|>assistant
A Large Language Model (LLM) is...<|im_end|>
<|im_start|>user
what is transformers<|im_end|>
<|im_start|>assistant
```

**Why**: Most modern models trained with this format. It's the de-facto standard for chat models.

**Models**:
- ✅ `deepseek-r1:1.5b`
- ✅ `qwen2.5:7b`
- ✅ `mistral:7b`
- ✅ `phi-3:mini`

---

### 2. Llama-2-Chat

**Used by**: Meta Llama-2-Chat models

**Format**:
```
<s>[INST] <<SYS>>
You are a helpful AI assistant. Provide clear, accurate, and helpful responses.
<</SYS>>

what is a LLM [/INST] A Large Language Model is... </s><s>[INST] what is transformers [/INST]
```

**Why**: Llama-2-Chat models were trained with this specific format.

**Models**:
- ✅ `llama-2-7b-chat`
- ✅ `llama-2-13b-chat`

---

### 3. Alpaca

**Used by**: Alpaca, WizardLM, many fine-tuned models

**Format**:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
what is a LLM

### Response:
A Large Language Model is...

### Instruction:
what is transformers

### Response:
```

**Why**: Popular format for instruction-tuned models.

**Models**:
- ✅ `alpaca-7b`
- ✅ `wizardlm-7b`

---

### 4. Vicuna/FastChat

**Used by**: Vicuna, FastChat models

**Format**:
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: what is a LLM
ASSISTANT: A Large Language Model is...
USER: what is transformers
ASSISTANT:
```

**Why**: Format used by Vicuna family of models.

**Models**:
- ✅ `vicuna-7b`
- ✅ `vicuna-13b`

---

### 5. Simple (Fallback)

**Used by**: Unknown models, basic LLMs

**Format**:
```
System: You are a helpful assistant.

User: what is a LLM
Assistant: A Large Language Model is...
User: what is transformers
Assistant:
```

**Why**: Universal fallback that works with most models.

---

## Automatic Template Detection

The system **automatically detects** which template to use based on model name:

```python
# DeepSeek detected -> ChatML
"deepseek-r1:1.5b" → ChatML Template

# Llama-2-Chat detected -> Llama2
"llama-2-7b-chat" → Llama-2-Chat Template

# Alpaca detected -> Alpaca
"alpaca-7b" → Alpaca Template

# Unknown model -> ChatML (modern default)
"my-custom-model" → ChatML Template
```

### Detection Logic

**ChatML Models** (most common):
- `deepseek`, `qwen`, `qwen2`, `yi`, `openchat`, `starling`
- `dolphin`, `mixtral`, `mistral`, `phi`, `gpt`
- `llama3`, `gemma`, `command`

**Llama-2-Chat Models**:
- Model name contains both `llama-2` AND `chat`

**Vicuna Models**:
- `vicuna`, `fastchat`

**Alpaca Models**:
- `alpaca`, `wizardlm`

**Default**: ChatML (for modern models) or Simple (for older/unknown models)

---

## System Prompts

All templates include a **system prompt** for better instruction following:

**Default System Prompt**:
```
You are a helpful AI assistant. Provide clear, accurate, and concise responses.
```

**Benefits**:
- ✅ Better instruction following
- ✅ More consistent behavior
- ✅ Reduced hallucinations
- ✅ Professional, helpful tone

**Custom System Prompts** (future):
```python
format_conversation_history(
    conversation_history=history,
    current_prompt="Hello",
    system_prompt="You are a medical AI assistant specialized in diagnostics."
)
```

---

## Implementation Details

### Code Location

**Main Module**: `src/utils/history_formatter.py`

**Key Functions**:
```python
def format_conversation_history(
    conversation_history: Optional[list[tuple[str, str]]],
    current_prompt: str,
    max_turns: int = 5,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,  # For auto-detection
    template: Optional[ChatTemplate] = None,  # Force specific template
) -> str:
    """Format with production-level templates."""
```

### Provider Integration

**Ollama** (`src/providers/ollama.py`):
```python
def run_text(self, handle, prompt, conversation_history=None):
    model_id = handle
    full_prompt = format_conversation_history(
        conversation_history=conversation_history,
        current_prompt=prompt,
        model_name=model_id  # Passes "deepseek-r1:1.5b"
    )
    # deepseek detected → ChatML template automatically used
```

**HuggingFace** (`src/providers/huggingface.py`):
```python
def run_text(self, handle, prompt, conversation_history=None):
    model, tokenizer = handle
    model_name = model.config._name_or_path  # Extract model name
    full_prompt = format_conversation_history(
        conversation_history=conversation_history,
        current_prompt=prompt,
        model_name=model_name  # Passes "meta-llama/Llama-2-7b-chat-hf"
    )
    # llama-2-chat detected → Llama-2-Chat template automatically used
```

**GGUF** (`src/providers/gguf.py`):
```python
def run_text(self, handle, prompt, conversation_history=None):
    full_prompt = format_conversation_history(
        conversation_history=conversation_history,
        current_prompt=prompt,
        model_name=None  # No easy access to model name
    )
    # Defaults to ChatML (works well with most GGUF models)
```

---

## Testing

### Test with DeepSeek-R1

**Before** (Simple format):
```
Session 11338fbf • 2 exchanges

You: hi
AI: Hello! How can I assist you today?

You: how are you
AI: <think>
Okay, the user said "hi" and then I responded with a friendly greeting. I noticed they used a simple "hi." Maybe I should keep it concise...

The second line was a bit confusing with the assistant saying "hi" again and asking for clarification...
</think>

Hello! How can I assist you today?
```

**Issue**: Model showing meta-commentary, treating conversation as text to analyze.

---

**After** (ChatML format):
```
Session 11338fbf • 2 exchanges

You: hi
AI: Hello! How can I assist you today? 😊

You: how are you
AI: I'm doing well, thank you for asking! I'm here and ready to help you with any questions or tasks you have. How can I assist you today?
```

**Fixed**: Natural, context-aware conversation without meta-commentary!

---

## Conversation History Examples

### Example 1: Context Retention

**Conversation**:
```
User: what is a LLM
AI: A Large Language Model (LLM) is an AI model trained on vast amounts of text data...

User: what is transformers
AI: Transformers are the neural network architecture that LLMs use. They were introduced in 2017...

User: im asking about movie
AI: Ah, I apologize for the confusion! You're asking about the Transformers movie franchise...
```

**Formatted Prompt** (what model sees):
```xml
<|im_start|>system
You are a helpful AI assistant. Provide clear, accurate, and concise responses.<|im_end|>
<|im_start|>user
what is a LLM<|im_end|>
<|im_start|>assistant
A Large Language Model (LLM) is an AI model trained on vast amounts of text data...<|im_end|>
<|im_start|>user
what is transformers<|im_end|>
<|im_start|>assistant
Transformers are the neural network architecture that LLMs use...<|im_end|>
<|im_start|>user
im asking about movie<|im_end|>
<|im_start|>assistant
```

**Result**: Model understands clarification and switches context appropriately.

---

### Example 2: Follow-up Questions

**Conversation**:
```
User: explain quantum computing
AI: Quantum computing uses quantum mechanics principles like superposition and entanglement...

User: give me an example
AI: A classic example is Shor's algorithm, which can factor large numbers exponentially faster than classical computers...

User: why is that important
AI: This is important for cryptography. Most encryption relies on the difficulty of factoring large numbers...
```

**Key**: Each response builds on previous context naturally.

---

## Configuration

### Max Turns (History Window)

**Default**: 5 turns (last 5 user-AI exchanges)

**Why 5**:
- ✅ Keeps recent context relevant
- ✅ Prevents token limit issues
- ✅ Optimal balance for most conversations
- ✅ Prevents ancient context from confusing model

**Example**:
```
Turn 1: "hi" → "Hello!"
Turn 2: "how are you" → "I'm well!"
Turn 3: "what's 2+2" → "4"
Turn 4: "what's 3+3" → "6"
Turn 5: "what's 4+4" → "8"
Turn 6: "what was my first question" → Remembers "hi" (turn 1 kept)
Turn 7: "what was my second question" → Turn 1 dropped, turn 2-7 kept
```

---

## Benefits Summary

### 🎯 Production-Ready

✅ **ChatGPT-like Experience**:
- Role-based chat templates
- System prompts for instruction following
- Natural conversation flow

✅ **Model-Specific Optimization**:
- ChatML for DeepSeek, Qwen, modern models
- Llama-2-Chat for Meta models
- Alpaca for instruction-tuned models
- Automatic detection

✅ **Context Awareness**:
- Models remember previous exchanges
- Can handle follow-up questions
- Understands clarifications and corrections

✅ **Better Responses**:
- No more meta-commentary
- Models treat it as conversation, not text analysis
- More natural, helpful responses

---

## Troubleshooting

### Model Still Showing Weird Behavior?

1. **Check Template Detection**:
   - Look for debug log: `"Using chatml template | History turns: 2/5"`
   - Verify correct template detected for your model

2. **Force Specific Template** (if auto-detection fails):
   ```python
   from ..utils.history_formatter import ChatTemplate

   full_prompt = format_conversation_history(
       conversation_history=history,
       current_prompt="Hello",
       template=ChatTemplate.CHATML  # Force ChatML
   )
   ```

3. **Check Logs**:
   - `loguru` logs show template selection
   - Look for: `"Detected ChatML template for deepseek-r1:1.5b"`

### Model Not Remembering Context?

1. **Verify History Enabled**:
   - In TUI, select "[1] WITH History" when starting chat
   - Check session indicator shows "Session: xxx • N exchanges"

2. **Check History Being Passed**:
   - Log should show: `"Passing 2 history turns to provider"`
   - Verify `conversation_history` not `None`

---

## Migration Notes

### No Breaking Changes

✅ **Backward Compatible**:
- All provider APIs unchanged
- Session storage unchanged
- Statistics tracking unchanged
- Existing sessions work as-is

✅ **Automatic**:
- No configuration needed
- Templates auto-detected
- Works out of the box

---

## Summary

The production-level chat template system brings ChatGPT/Claude-quality conversation handling to your application. It automatically detects and uses the right format for each model, includes system prompts for better instruction following, and maintains full conversation context for natural interactions.

**Key Points**:
1. **5 Template Types**: ChatML, Llama-2, Alpaca, Vicuna, Simple
2. **Auto-Detection**: Based on model name
3. **System Prompts**: Better instruction following
4. **Context Window**: Last 5 turns retained
5. **Production-Ready**: Like ChatGPT/Claude

Test with DeepSeek-R1 or any modern LLM to see the difference!
