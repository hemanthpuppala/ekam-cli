"""
Benchmarking initialization module.

Handles one-time setup tasks required for benchmarking features.
This is automatically called at application startup via app.py.
"""

from loguru import logger


def initialize_nltk_data() -> None:
    """
    Initialize NLTK data required for quality metrics.

    Downloads punkt and punkt_tab tokenizers if not already present.
    Required for BLEU score computation in Quality Suite.

    This function is idempotent - safe to call multiple times.
    """
    try:
        import nltk

        # Check and download punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
            logger.debug("NLTK punkt tokenizer already installed")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer data...")
            nltk.download('punkt', quiet=True)
            logger.info("✓ NLTK punkt tokenizer installed")

        # Check and download punkt_tab tokenizer (newer version)
        try:
            nltk.data.find('tokenizers/punkt_tab')
            logger.debug("NLTK punkt_tab tokenizer already installed")
        except LookupError:
            logger.info("Downloading NLTK punkt_tab tokenizer data...")
            nltk.download('punkt_tab', quiet=True)
            logger.info("✓ NLTK punkt_tab tokenizer installed")

    except ImportError:
        # NLTK not installed - this is fine, BLEU scores will be skipped
        logger.debug("NLTK not installed - BLEU scores will not be available")
    except Exception as e:
        # Non-fatal error - benchmarking will still work without BLEU
        logger.warning(f"Failed to initialize NLTK data: {e}")


def initialize_benchmarking() -> None:
    """
    Initialize all benchmarking dependencies.

    Called once at application startup to ensure all required data
    and resources are available for benchmarking features.
    """
    logger.info("Initializing benchmarking dependencies...")

    # Initialize NLTK data for BLEU scores
    initialize_nltk_data()

    # Future: Add other initialization tasks here
    # - Download sentence-transformers models?
    # - Initialize any other NLP resources?

    logger.info("Benchmarking initialization complete")


if __name__ == "__main__":
    # Allow running this module directly for testing
    from loguru import logger
    initialize_benchmarking()
