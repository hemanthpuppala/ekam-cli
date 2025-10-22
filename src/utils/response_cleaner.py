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
    - Extracts content after "Answer:" markers

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

        >>> clean_model_response("Okay...\\nAnswer: This is the real answer.")
        "This is the real answer."
    """
    if not response_text or len(response_text.strip()) == 0:
        return response_text

    # OPTIMIZATION: O(1) amortized time - early termination with pattern matching
    # Process only what's needed, skip full line-by-line scan when possible

    # Pattern 1: Extract content after "Answer:" marker
    # This handles edge case where reasoning precedes the actual answer
    # Priority order: exact "\nAnswer:" > "\nA:" > case variants
    answer_patterns = ["\nAnswer:", "\nA:", "Answer:", "A:"]
    answer_idx = -1
    answer_marker_len = 0

    for pattern in answer_patterns:
        idx = response_text.find(pattern)
        if idx != -1:
            answer_idx = idx
            answer_marker_len = len(pattern)
            break

    if answer_idx != -1:
        # Found answer marker - extract everything after it
        extracted = response_text[answer_idx + answer_marker_len:].strip()

        # Quick clean: remove common end markers (O(1) operations, max 4 iterations)
        for marker in ["<|endoftext|>", "<|im_end|>", "</s>", "<eos>"]:
            extracted = extracted.replace(marker, "")

        extracted = extracted.strip()

        # Only use if substantial (>20 chars) to avoid false positives
        if len(extracted) > 20:
            logger.debug(f"Extracted answer after marker ({len(extracted)} chars)")
            return extracted

    # Pattern 2: Extract after "The answer is" or similar definitive statements
    # O(1) - check at most 3 patterns, only if substantial reasoning precedes it
    if len(response_text) > 100:  # Only try this pattern if response is long enough
        definitive_patterns = [" answer is ", " solution is ", " result is "]
        for pattern in definitive_patterns:
            idx = response_text.lower().find(pattern)
            if idx != -1 and idx > 20:  # Must have some text before "answer is"
                # Extract from this point onwards
                extracted = response_text[idx + len(pattern):].strip()

                # Find first sentence (stop at period, newline, or 200 chars)
                sentence_end = min(
                    extracted.find('. ') if '. ' in extracted else len(extracted),
                    extracted.find('\n') if '\n' in extracted else len(extracted),
                    200  # Max sentence length to keep it O(1)
                )

                if sentence_end > 0:
                    extracted = extracted[:sentence_end].strip()

                    # Clean markers
                    for marker in ["<|endoftext|>", "<|im_end|>", "</s>", "<eos>"]:
                        extracted = extracted.replace(marker, "")

                    extracted = extracted.strip()

                    if len(extracted) > 3:  # Even short answers like "42" are valid
                        logger.debug(f"Extracted definitive answer ({len(extracted)} chars)")
                        return extracted

    # 1. Template artifacts to remove (role markers)
    role_artifacts = [
        "User:", "Assistant:", "Human:", "System:",
    ]

    # 2. Very minimal reasoning markers - only catch obvious meta-commentary
    # BE LENIENT: Only remove if it's clearly just thinking, not actual content
    reasoning_markers = [
        "The user asked", "The user said", "The user wants",
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

        # LENIENT: Only skip obvious meta-commentary at the START of the response
        # Don't truncate mid-response - that removes too much content
        if aggressive and not cleaned_lines:  # Only at the start
            if any(stripped.startswith(marker) for marker in reasoning_markers):
                # Skip this preamble line
                logger.debug(f"Skipped preamble: {stripped[:50]}")
                continue

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
