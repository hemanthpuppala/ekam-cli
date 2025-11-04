"""
Pytest configuration and root-level fixtures for benchmarking tests.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def temp_results_dir(tmp_path_factory):
    """Create temporary directory for test results."""
    return tmp_path_factory.mktemp("test_results")


@pytest.fixture(autouse=True)
def reset_loggers():
    """Reset loggers between tests to avoid interference."""
    import logging
    logging.root.handlers = []
    yield
    logging.root.handlers = []
