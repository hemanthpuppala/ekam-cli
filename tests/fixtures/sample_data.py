"""
Sample test data for benchmarking.

Provides test prompts, images, and benchmark configurations.
"""

from typing import List, Dict


# Sample LLM prompts for different categories
DEFAULT_LLM_PROMPTS: List[str] = [
    "What is artificial intelligence?",
    "Explain the concept of machine learning in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
    "Translate 'Hello, world!' into Spanish, French, and German.",
    "Summarize the key benefits of cloud computing.",
]

# Category-specific prompts
CREATIVE_PROMPTS: List[str] = [
    "Write a haiku about coding.",
    "Create a short story about a robot learning to paint.",
    "Compose a limerick about debugging.",
]

ANALYTICAL_PROMPTS: List[str] = [
    "Analyze the pros and cons of remote work.",
    "Compare and contrast SQL and NoSQL databases.",
    "Explain the trade-offs between monolithic and microservices architecture.",
]

TECHNICAL_PROMPTS: List[str] = [
    "Explain how a hash table works.",
    "What is the difference between TCP and UDP?",
    "Describe the CAP theorem in distributed systems.",
]

MATH_PROMPTS: List[str] = [
    "Calculate 15 * 24.",
    "What is the square root of 144?",
    "Solve for x: 2x + 5 = 15",
]


# Sample VLM prompts (image description tasks)
DEFAULT_VLM_PROMPTS: List[str] = [
    "Describe this image in detail.",
    "What objects can you see in this image?",
    "What is the main subject of this image?",
    "Describe the colors and composition of this image.",
    "What activity or scene is depicted in this image?",
]

# VLM analysis prompts
VLM_ANALYSIS_PROMPTS: List[str] = [
    "Identify any text visible in this image.",
    "Count the number of people in this image.",
    "What is the setting or location of this image?",
    "Describe the lighting and mood of this image.",
]


# Sample benchmark test data configurations
BENCHMARK_TEST_DATA: Dict[str, Dict] = {
    "speed_llm": {
        "prompts": DEFAULT_LLM_PROMPTS[:3],
        "num_runs": 5,
        "num_warmup": 1,
    },
    "speed_vlm": {
        "prompts": DEFAULT_VLM_PROMPTS[:3],
        "images": ["sample_1.jpg", "sample_2.jpg", "sample_3.jpg"],
        "num_runs": 5,
        "num_warmup": 1,
    },
    "quality_llm": {
        "prompts": ANALYTICAL_PROMPTS,
        "num_runs": 10,
        "num_warmup": 2,
    },
    "stress_llm": {
        "prompts": DEFAULT_LLM_PROMPTS,
        "duration_minutes": 30,
        "num_warmup": 3,
    },
}


# Sample expected outputs for quality testing
EXPECTED_OUTPUTS: Dict[str, str] = {
    "What is 2 + 2?": "4",
    "What is the capital of France?": "Paris",
    "Translate 'Hello' to Spanish": "Hola",
}


def get_prompts_by_category(category: str) -> List[str]:
    """Get prompts by category."""
    categories = {
        "default": DEFAULT_LLM_PROMPTS,
        "creative": CREATIVE_PROMPTS,
        "analytical": ANALYTICAL_PROMPTS,
        "technical": TECHNICAL_PROMPTS,
        "math": MATH_PROMPTS,
        "vlm_default": DEFAULT_VLM_PROMPTS,
        "vlm_analysis": VLM_ANALYSIS_PROMPTS,
    }
    return categories.get(category, DEFAULT_LLM_PROMPTS)


def get_test_data_config(test_type: str) -> Dict:
    """Get test data configuration for a specific test type."""
    return BENCHMARK_TEST_DATA.get(test_type, {})
