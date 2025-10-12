"""Production-level conversation history formatting with role-based chat templates.

This module provides chat template formatting used by production LLMs like
ChatGPT, Claude, DeepSeek, Qwen, etc. Supports multiple template formats
with automatic model detection.
"""

from enum import Enum
from typing import Optional

from loguru import logger


class ChatTemplate(Enum):
    """Chat template formats used by different model families."""

    CHATML = "chatml"  # GPT-4, DeepSeek, Qwen, modern models
    ALPACA = "alpaca"  # Alpaca, many fine-tuned models
    VICUNA = "vicuna"  # Vicuna, FastChat models
    LLAMA2 = "llama2"  # Meta Llama-2-Chat
    SIMPLE = "simple"  # Fallback for basic models


def get_stop_tokens_for_template(template: ChatTemplate) -> list[str]:
    """Get appropriate stop tokens for a chat template.

    Stop tokens tell the model when to stop generating to prevent
    it from continuing past its response or generating fake conversation turns.

    Args:
        template: The chat template being used

    Returns:
        List of stop token strings

    Examples:
        >>> get_stop_tokens_for_template(ChatTemplate.CHATML)
        ['<|im_end|>', '<|im_start|>']

        >>> get_stop_tokens_for_template(ChatTemplate.LLAMA2)
        ['[/INST]', '[INST]', '</s>']
    """
    stop_tokens_map = {
        ChatTemplate.CHATML: ["<|im_end|>", "<|im_start|>"],
        ChatTemplate.ALPACA: ["###"],
        ChatTemplate.VICUNA: ["USER:", "ASSISTANT:"],
        ChatTemplate.LLAMA2: ["[/INST]", "[INST]", "</s>"],
        ChatTemplate.SIMPLE: ["\nUser:", "\nAssistant:"],
    }
    return stop_tokens_map.get(template, [])


def get_stop_tokens(model_name: Optional[str] = None, template: Optional[ChatTemplate] = None) -> list[str]:
    """Get stop tokens for a model or template.

    Convenience function that auto-detects template from model name or uses provided template.

    Args:
        model_name: Model identifier for auto-detection (optional)
        template: Force specific template (optional)

    Returns:
        List of stop token strings

    Examples:
        >>> get_stop_tokens(model_name="deepseek-r1:1.5b")
        ['<|im_end|>', '<|im_start|>']

        >>> get_stop_tokens(template=ChatTemplate.CHATML)
        ['<|im_end|>', '<|im_start|>']
    """
    if template is None:
        if model_name:
            template = detect_chat_template(model_name)
        else:
            template = ChatTemplate.CHATML  # Default

    return get_stop_tokens_for_template(template)


def detect_chat_template(model_name: str) -> ChatTemplate:
    """Detect appropriate chat template based on model name.

    Args:
        model_name: Model identifier or name

    Returns:
        ChatTemplate enum for the model
    """
    model_lower = model_name.lower()

    # ChatML format (most modern models)
    chatml_models = [
        "deepseek", "qwen", "qwen2", "yi", "openchat", "starling",
        "dolphin", "mixtral", "mistral", "phi", "gpt"
    ]
    if any(keyword in model_lower for keyword in chatml_models):
        logger.debug(f"Detected ChatML template for {model_name}")
        return ChatTemplate.CHATML

    # Llama-2-Chat format
    if "llama-2" in model_lower and "chat" in model_lower:
        logger.debug(f"Detected Llama-2-Chat template for {model_name}")
        return ChatTemplate.LLAMA2

    # Vicuna format
    if "vicuna" in model_lower or "fastchat" in model_lower:
        logger.debug(f"Detected Vicuna template for {model_name}")
        return ChatTemplate.VICUNA

    # Alpaca format
    if "alpaca" in model_lower or "wizardlm" in model_lower:
        logger.debug(f"Detected Alpaca template for {model_name}")
        return ChatTemplate.ALPACA

    # Default to ChatML for unknown modern models, SIMPLE for others
    if any(x in model_lower for x in ["llama3", "gemma", "command"]):
        logger.debug(f"Using ChatML (default) for {model_name}")
        return ChatTemplate.CHATML
    else:
        logger.debug(f"Using SIMPLE (fallback) for {model_name}")
        return ChatTemplate.SIMPLE


def format_with_chatml(
    conversation_history: Optional[list[tuple[str, str]]],
    current_prompt: str,
    max_turns: int,
    system_prompt: Optional[str] = None,
) -> str:
    """Format with ChatML template (GPT-4, DeepSeek, Qwen style).

    Format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        Hi there!<|im_end|>
        <|im_start|>user
        Current message<|im_end|>
        <|im_start|>assistant
    """
    parts = []

    # System prompt
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
    else:
        parts.append("<|im_start|>system\nYou are a helpful AI assistant. Provide clear, accurate, and concise responses.<|im_end|>\n")

    # Conversation history
    if conversation_history:
        recent_history = conversation_history[-max_turns:]
        for user_msg, ai_response in recent_history:
            parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>\n")
            parts.append(f"<|im_start|>assistant\n{ai_response}<|im_end|>\n")

    # Current prompt
    parts.append(f"<|im_start|>user\n{current_prompt}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


def format_with_alpaca(
    conversation_history: Optional[list[tuple[str, str]]],
    current_prompt: str,
    max_turns: int,
    system_prompt: Optional[str] = None,
) -> str:
    """Format with Alpaca template.

    Format:
        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        Hello

        ### Response:
        Hi there!

        ### Instruction:
        Current message

        ### Response:
    """
    parts = []

    # System prompt
    if system_prompt:
        parts.append(f"{system_prompt}\n\n")
    else:
        parts.append("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n")

    # Conversation history
    if conversation_history:
        recent_history = conversation_history[-max_turns:]
        for user_msg, ai_response in recent_history:
            parts.append(f"### Instruction:\n{user_msg}\n\n")
            parts.append(f"### Response:\n{ai_response}\n\n")

    # Current prompt
    parts.append(f"### Instruction:\n{current_prompt}\n\n")
    parts.append("### Response:\n")

    return "".join(parts)


def format_with_vicuna(
    conversation_history: Optional[list[tuple[str, str]]],
    current_prompt: str,
    max_turns: int,
    system_prompt: Optional[str] = None,
) -> str:
    """Format with Vicuna/FastChat template.

    Format:
        A chat between a curious user and an artificial intelligence assistant.

        USER: Hello
        ASSISTANT: Hi there!
        USER: Current message
        ASSISTANT:
    """
    parts = []

    # System prompt
    if system_prompt:
        parts.append(f"{system_prompt}\n\n")
    else:
        parts.append("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n")

    # Conversation history
    if conversation_history:
        recent_history = conversation_history[-max_turns:]
        for user_msg, ai_response in recent_history:
            parts.append(f"USER: {user_msg}\n")
            parts.append(f"ASSISTANT: {ai_response}\n")

    # Current prompt
    parts.append(f"USER: {current_prompt}\n")
    parts.append("ASSISTANT: ")

    return "".join(parts)


def format_with_llama2(
    conversation_history: Optional[list[tuple[str, str]]],
    current_prompt: str,
    max_turns: int,
    system_prompt: Optional[str] = None,
) -> str:
    """Format with Llama-2-Chat template.

    Format:
        <s>[INST] <<SYS>>
        You are a helpful assistant.
        <</SYS>>

        Hello [/INST] Hi there! </s><s>[INST] Current message [/INST]
    """
    parts = []

    # System prompt (only in first message)
    if system_prompt:
        sys_msg = system_prompt
    else:
        sys_msg = "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."

    # Conversation history with special handling for first turn
    if conversation_history:
        recent_history = conversation_history[-max_turns:]

        # First turn includes system prompt
        if recent_history:
            first_user, first_ai = recent_history[0]
            parts.append(f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{first_user} [/INST] {first_ai} </s>")

            # Remaining turns
            for user_msg, ai_response in recent_history[1:]:
                parts.append(f"<s>[INST] {user_msg} [/INST] {ai_response} </s>")
    else:
        # No history - include system in current turn
        parts.append(f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{current_prompt} [/INST]")
        return "".join(parts)

    # Current prompt
    parts.append(f"<s>[INST] {current_prompt} [/INST]")

    return "".join(parts)


def format_with_simple(
    conversation_history: Optional[list[tuple[str, str]]],
    current_prompt: str,
    max_turns: int,
    system_prompt: Optional[str] = None,
) -> str:
    """Format with simple User/Assistant template (fallback).

    Format:
        System: You are a helpful assistant.

        User: Hello
        Assistant: Hi there!
        User: Current message
        Assistant:
    """
    parts = []

    # System prompt
    if system_prompt:
        parts.append(f"System: {system_prompt}\n\n")

    # Conversation history
    if conversation_history:
        recent_history = conversation_history[-max_turns:]
        for user_msg, ai_response in recent_history:
            parts.append(f"User: {user_msg}\n")
            parts.append(f"Assistant: {ai_response}\n")

    # Current prompt
    parts.append(f"User: {current_prompt}\n")
    parts.append("Assistant:")

    return "".join(parts)


def format_conversation_history(
    conversation_history: Optional[list[tuple[str, str]]],
    current_prompt: str,
    max_turns: int = 5,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    template: Optional[ChatTemplate] = None,
) -> str:
    """Format conversation history with production-level role-based templates.

    This is the main formatting function used across all providers.
    Automatically detects and uses the appropriate chat template for the model.

    Args:
        conversation_history: List of (user_message, ai_response) tuples
        current_prompt: The current user message/question
        max_turns: Maximum number of previous conversation turns (default: 5)
        system_prompt: Optional system prompt for instruction following
        model_name: Model identifier for template detection (optional)
        template: Force specific template (optional, auto-detected if not provided)

    Returns:
        Formatted prompt string with role-based conversation context

    Examples:
        >>> # Auto-detect template from model name
        >>> format_conversation_history(history, "Hello", model_name="deepseek-r1:1.5b")
        '<|im_start|>system\\n...\\n<|im_start|>assistant\\n'

        >>> # Force specific template
        >>> format_conversation_history(history, "Hello", template=ChatTemplate.CHATML)
        '<|im_start|>system\\n...\\n<|im_start|>assistant\\n'
    """
    # Detect or use provided template
    if template is None:
        if model_name:
            template = detect_chat_template(model_name)
        else:
            # Default to ChatML for modern models
            template = ChatTemplate.CHATML
            logger.debug("No model name provided, defaulting to ChatML template")

    # Log template selection
    logger.debug(
        f"Using {template.value} template | "
        f"History turns: {len(conversation_history) if conversation_history else 0}/{max_turns}"
    )

    # Format with selected template
    if template == ChatTemplate.CHATML:
        formatted = format_with_chatml(conversation_history, current_prompt, max_turns, system_prompt)
    elif template == ChatTemplate.ALPACA:
        formatted = format_with_alpaca(conversation_history, current_prompt, max_turns, system_prompt)
    elif template == ChatTemplate.VICUNA:
        formatted = format_with_vicuna(conversation_history, current_prompt, max_turns, system_prompt)
    elif template == ChatTemplate.LLAMA2:
        formatted = format_with_llama2(conversation_history, current_prompt, max_turns, system_prompt)
    else:  # SIMPLE
        formatted = format_with_simple(conversation_history, current_prompt, max_turns, system_prompt)

    logger.debug(f"Formatted prompt length: {len(formatted)} chars")
    return formatted


def format_qa_history(
    conversation_history: Optional[list[tuple[str, str]]],
    current_question: str,
    max_turns: int = 5,
) -> str:
    """Format conversation history for image Q&A tasks (VLMs).

    Uses Q&A format which is more natural for vision models.
    This format works well across all VLM providers.

    Args:
        conversation_history: List of (question, answer) tuples
        current_question: The current question about the image
        max_turns: Maximum number of previous Q&A turns

    Returns:
        Formatted Q&A prompt string

    Format:
        Q: previous question 1
        A: previous answer 1
        Q: previous question 2
        A: previous answer 2
        Q: current question
        A:

    Examples:
        >>> history = [("What's in the image?", "A red car"), ("What color?", "Red")]
        >>> format_qa_history(history, "Any people?")
        "Q: What's in the image?\\nA: A red car\\nQ: What color?\\nA: Red\\nQ: Any people?\\nA:"
    """
    parts = []

    # Add conversation history if provided
    if conversation_history:
        # Limit to last N turns
        recent_history = conversation_history[-max_turns:]
        logger.debug(f"Formatting {len(recent_history)} Q&A turns (max: {max_turns})")

        for question, answer in recent_history:
            parts.append(f"Q: {question}\nA: {answer}\n")

    # Add current question
    parts.append(f"Q: {current_question}\nA:")

    formatted_prompt = "".join(parts)
    logger.debug(f"Formatted Q&A prompt length: {len(formatted_prompt)} chars")

    return formatted_prompt
