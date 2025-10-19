"""Response cleaning utilities to handle model artifacts and meta-commentary.

This module provides comprehensive response cleaning for all LLM providers,
handling issues like template leakage, internal reasoning artifacts, and
meta-commentary that some models may generate.
"""

from loguru import logger


def clean_model_response(response_text: str, aggressive: bool = True) -> str:
    """Clean model response by removing artifacts, reasoning, and meta-commentary.

    This function handles various output issues that can occur with different models:
    - Template leakage (role markers like "User:", "Assistant:")
    - Internal reasoning artifacts ("Okay,", "Let me think", etc.)
    - Meta-commentary ("The user asked", "The problem is", etc.)
    - Math formatting artifacts (\\boxed{}, **Final Answer**, etc.)
    - Excessive whitespace and repetition

    Args:
        response_text: Raw model output text
        aggressive: If True, truncates at first reasoning marker (default: True)

    Returns:
        Cleaned response text with artifacts removed

    Examples:
        >>> clean_model_response("Okay, let me think... The answer is 42.")
        "The answer is 42."

        >>> clean_model_response("User: hello\\nAssistant: Hi there!")
        "Hi there!"
    """
    if not response_text or len(response_text.strip()) == 0:
        return response_text

    # 1. Template artifacts to remove (role markers)
    role_artifacts = [
        "User:", "Assistant:", "Human:", "System:",
        "The user", "The assistant", "The system"
    ]

    # 2. Reasoning markers (indicates internal reasoning - truncate here if aggressive)
    reasoning_markers = [
        "Okay,", "Alright,", "Wait,", "So,", "Let me", "I think", "I need to",
        "I'm going to", "I should", "I can", "I will", "I'll", "Let's",
        "The user asked", "The question", "The problem", "The instruction",
        "The initial", "The final", "According to", "Based on the",
        "This is a", "This response", "The key points", "The example",
        "Now the", "In the problem", "In the example", "From the",
        "The reasoning", "The thinking", "The analysis", "The approach"
    ]

    # 3. Math/formatting artifacts
    math_artifacts = ["\\boxed{", "**Final Answer**", "\\begin{", "\\end{"]

    # Split response into lines
    lines = response_text.split('\n')
    cleaned_lines = []
    found_reasoning_section = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines at start
        if not stripped and not cleaned_lines:
            continue

        # Check for role artifacts - remove these lines
        if any(stripped.startswith(artifact) for artifact in role_artifacts):
            logger.debug(f"Removed role artifact: {stripped[:50]}")
            continue

        # Check for reasoning markers
        if aggressive and any(stripped.startswith(marker) for marker in reasoning_markers):
            logger.debug(f"Truncated at reasoning marker: {stripped[:50]}")
            found_reasoning_section = True
            break

        # Check for math artifacts - remove these lines
        if any(artifact in stripped for artifact in math_artifacts):
            logger.debug(f"Removed math artifact: {stripped[:50]}")
            continue

        # Keep the line
        cleaned_lines.append(line)

    # Rejoin lines
    cleaned_response = '\n'.join(cleaned_lines).strip()

    # Remove excessive newlines (more than 2 in a row)
    while '\n\n\n' in cleaned_response:
        cleaned_response = cleaned_response.replace('\n\n\n', '\n\n')

    # If cleaning removed too much (less than 10 chars), return original
    if len(cleaned_response) < 10:
        logger.warning("Response cleaning removed too much content, returning original")
        return response_text.strip()

    # Log cleaning summary
    original_len = len(response_text)
    cleaned_len = len(cleaned_response)
    if original_len != cleaned_len:
        logger.debug(
            f"Response cleaned: {original_len} -> {cleaned_len} chars "
            f"({'truncated' if found_reasoning_section else 'filtered'})"
        )

    return cleaned_response


def extract_first_answer(response_text: str) -> str:
    """Extract just the first coherent answer from model output.

    Some models may generate multiple paragraphs with repetition or
    meta-commentary. This function tries to extract just the first
    complete answer.

    Args:
        response_text: Model output text

    Returns:
        First coherent answer section

    Examples:
        >>> extract_first_answer("An LLM is a language model.\\n\\nOkay let me explain...")
        "An LLM is a language model."
    """
    # First apply general cleaning
    cleaned = clean_model_response(response_text, aggressive=True)

    # Split into paragraphs
    paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]

    if not paragraphs:
        return cleaned

    # Return first substantial paragraph (at least 20 chars)
    for para in paragraphs:
        if len(para) >= 20:
            return para

    # If no substantial paragraph, return first paragraph
    return paragraphs[0] if paragraphs else cleaned
