"""
Default test data for benchmarking.

Provides ready-to-use prompts and test configurations for LLM and VLM benchmarks.
Includes structured test data management with image-prompt pairing.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from loguru import logger


# ============================================
# LLM Default Prompts
# ============================================

DEFAULT_LLM_PROMPTS: List[str] = [
    "What is artificial intelligence?",
    "Explain the concept of machine learning in simple terms.",
    "Write a Python function to calculate the nth Fibonacci number.",
    "What are the main differences between SQL and NoSQL databases?",
    "Describe three benefits of cloud computing.",
]

# Short prompts for speed testing
SPEED_TEST_PROMPTS: List[str] = [
    "What is Python?",
    "Define recursion.",
    "Explain HTTP.",
    "What is JSON?",
    "Define API.",
]

# Long prompts for stress testing
STRESS_TEST_PROMPTS: List[str] = [
    "Explain in detail the history, architecture, and key features of microservices, "
    "including their advantages over monolithic architectures, common design patterns, "
    "and best practices for implementation in production environments.",

    "Write a comprehensive tutorial on building a REST API with authentication, "
    "including code examples, security considerations, database integration, "
    "error handling, and deployment strategies.",

    "Analyze the evolution of programming languages from the 1950s to present day, "
    "discussing major paradigm shifts, influential languages, and how modern languages "
    "incorporate lessons from their predecessors.",
]

# Quality test prompts (deterministic answers)
QUALITY_TEST_PROMPTS: List[str] = [
    "What is 2 + 2?",
    "What is the capital of France?",
    "How many days are in a week?",
    "What is the boiling point of water in Celsius?",
    "What year did World War II end?",
]


# ============================================
# VLM Default Prompts - By Endpoint
# ============================================

# Visual Q&A Endpoint Prompts
VLM_VISION_QA_PROMPTS: List[str] = [
    "Describe this image in detail.",
    "What objects can you see in this image?",
    "What animals are in this image, and what each animal is doing?",
    "What colors are prominent in this image?",
    "What activity or scene is depicted in this image?",
    "Count how many people are in this image.",
    "Describe the setting or location shown.",
    "What is the mood or atmosphere of this image?",
    "Identify any brands or logos visible.",
]

# Image Captioning Endpoint Prompts
VLM_CAPTION_PROMPTS: List[str] = [
    "Provide a brief caption for this image.",
    "Write a one-sentence description of what's in this image.",
    "Summarize the main content of this image in a few words.",
    "Create a descriptive title for this image.",
    "Describe the primary subject in a concise way.",
    "What would be a good caption for this social media post?",
    "Write a news headline for this image.",
]

# Object Detection Endpoint Prompts
VLM_DETECT_PROMPTS: List[str] = [
    "What objects are in the foreground?",
    "What objects are in the background?",
    "Identify any vehicles in the image.",
    "Are there any animals in this image? If so, how many?",
    "What furniture can you see?",
    "List all the distinct objects you can identify.",
    "What is the largest object in this image?",
]

# Object Pointing Endpoint Prompts
VLM_POINT_PROMPTS: List[str] = [
    "Where is the main subject located in this image?",
    "Point to the center of the image and describe what's there.",
    "What is positioned in the top-left corner?",
    "Identify the location of any people in this image.",
    "Where are the largest objects positioned?",
    "Describe the spatial layout of objects in this image.",
]

# Generic VLM prompts (fallback)
DEFAULT_VLM_PROMPTS: List[str] = VLM_VISION_QA_PROMPTS[:5]

# Specific analysis prompts (legacy)
VLM_ANALYSIS_PROMPTS: List[str] = [
    "List all the text visible in this image.",
    "Count how many people are in this image.",
    "Describe the setting or location shown.",
    "What is the mood or atmosphere of this image?",
    "Identify any brands or logos visible.",
]

# Object detection prompts (legacy)
VLM_OBJECT_PROMPTS: List[str] = [
    "What objects are in the foreground?",
    "What objects are in the background?",
    "Identify any vehicles in the image.",
    "Are there any animals in this image?",
    "What furniture can you see?",
]


# ============================================
# Default Test Configurations
# ============================================

DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "prompts": DEFAULT_LLM_PROMPTS,
    "parameters": {
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9
    },
    "num_runs": 5,
    "num_warmup": 2
}

DEFAULT_VLM_CONFIG: Dict[str, Any] = {
    "prompts": DEFAULT_VLM_PROMPTS,
    "images": [],  # User must provide images
    "parameters": {
        "temperature": 0.5,
        "max_tokens": 256
    },
    "num_runs": 3,
    "num_warmup": 1
}

SPEED_BENCHMARK_CONFIG: Dict[str, Any] = {
    "prompts": SPEED_TEST_PROMPTS,
    "parameters": {
        "temperature": 0.7,
        "max_tokens": 100
    },
    "num_runs": 10,
    "num_warmup": 2
}

QUALITY_BENCHMARK_CONFIG: Dict[str, Any] = {
    "prompts": QUALITY_TEST_PROMPTS,
    "parameters": {
        "temperature": 0.1,  # Low temp for consistency
        "max_tokens": 50
    },
    "num_runs": 10,
    "num_warmup": 1
}

STRESS_BENCHMARK_CONFIG: Dict[str, Any] = {
    "prompts": STRESS_TEST_PROMPTS,
    "parameters": {
        "temperature": 0.7,
        "max_tokens": 2048
    },
    "num_runs": 20,
    "num_warmup": 3
}


# ============================================
# Helper Functions
# ============================================

def get_default_prompts(model_type: str, count: int = 5) -> List[str]:
    """
    Get default prompts for a model type.

    Args:
        model_type: 'llm' or 'vlm'
        count: Number of prompts to return

    Returns:
        List of prompts
    """
    if model_type.lower() == "llm":
        return DEFAULT_LLM_PROMPTS[:count]
    elif model_type.lower() == "vlm":
        return DEFAULT_VLM_PROMPTS[:count]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_prompts_by_endpoint(endpoint: str) -> List[str]:
    """
    Get default prompts for a specific VLM endpoint.

    Args:
        endpoint: Endpoint name ('vision/qa', 'vision/caption', 'vision/detect', 'vision/point')

    Returns:
        List of prompts for the endpoint
    """
    endpoint = endpoint.lower()

    if endpoint == "vision/qa":
        return VLM_VISION_QA_PROMPTS
    elif endpoint == "vision/caption":
        return VLM_CAPTION_PROMPTS
    elif endpoint == "vision/detect":
        return VLM_DETECT_PROMPTS
    elif endpoint == "vision/point":
        return VLM_POINT_PROMPTS
    else:
        return DEFAULT_VLM_PROMPTS


def get_prompts_for_suite(suite_type: str, model_type: str = "llm", endpoint: str = None) -> List[str]:
    """
    Get appropriate prompts for a benchmark suite.

    Args:
        suite_type: 'speed', 'quality', 'stress', 'resources', or 'complete'
        model_type: 'llm' or 'vlm'
        endpoint: Optional VLM endpoint name for endpoint-specific prompts

    Returns:
        List of prompts
    """
    suite_type = suite_type.lower()
    model_type = model_type.lower()

    if model_type == "llm":
        if suite_type == "speed":
            return SPEED_TEST_PROMPTS
        elif suite_type == "quality":
            return QUALITY_TEST_PROMPTS
        elif suite_type == "stress":
            return STRESS_TEST_PROMPTS
        else:
            return DEFAULT_LLM_PROMPTS
    else:  # VLM
        # Use endpoint-specific prompts if provided
        if endpoint:
            return get_prompts_by_endpoint(endpoint)

        # Fallback based on suite type
        if suite_type == "speed":
            return DEFAULT_VLM_PROMPTS[:3]
        elif suite_type == "quality":
            return DEFAULT_VLM_PROMPTS
        elif suite_type == "stress":
            return DEFAULT_VLM_PROMPTS
        elif suite_type == "resources":
            return DEFAULT_VLM_PROMPTS
        elif suite_type == "complete":
            return DEFAULT_VLM_PROMPTS
        else:
            return DEFAULT_VLM_PROMPTS


def get_default_config(suite_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a suite type.

    Args:
        suite_type: Suite type identifier

    Returns:
        Configuration dictionary
    """
    configs = {
        "speed": SPEED_BENCHMARK_CONFIG,
        "quality": QUALITY_BENCHMARK_CONFIG,
        "stress": STRESS_BENCHMARK_CONFIG,
        "llm": DEFAULT_LLM_CONFIG,
        "vlm": DEFAULT_VLM_CONFIG
    }

    return configs.get(suite_type.lower(), DEFAULT_LLM_CONFIG)


# ============================================
# Expected Outputs (for quality testing)
# ============================================

EXPECTED_OUTPUTS: Dict[str, List[str]] = {
    "What is 2 + 2?": ["4", "four"],
    "What is the capital of France?": ["Paris"],
    "How many days are in a week?": ["7", "seven"],
    "What is the boiling point of water in Celsius?": ["100", "100°C", "100 degrees"],
    "What year did World War II end?": ["1945"],
}


def check_output_matches_expected(prompt: str, output: str) -> bool:
    """
    Check if output matches expected answer for quality testing.

    Args:
        prompt: Input prompt
        output: Model output

    Returns:
        True if output matches expected answer
    """
    if prompt not in EXPECTED_OUTPUTS:
        return False

    expected = EXPECTED_OUTPUTS[prompt]
    output_lower = output.lower().strip()

    return any(exp.lower() in output_lower for exp in expected)


# ============================================
# Test Data Management
# ============================================

# Test data directory structure
TEST_DATA_ROOT = Path(__file__).parent.parent.parent.parent / "assets" / "test_data"

ENDPOINT_DIRS = {
    "vision/qa": TEST_DATA_ROOT / "visual_qa",
    "vision/caption": TEST_DATA_ROOT / "caption",
    "vision/detect": TEST_DATA_ROOT / "detect",
    "vision/point": TEST_DATA_ROOT / "point",
}

TEST_DATA_CONFIG = TEST_DATA_ROOT / "test_data_config.json"


def get_endpoint_images(endpoint: str) -> List[Path]:
    """
    Get all images for a specific endpoint.

    Args:
        endpoint: Endpoint name ('vision/qa', 'vision/caption', etc.)

    Returns:
        List of image paths
    """
    endpoint_dir = ENDPOINT_DIRS.get(endpoint)
    if not endpoint_dir:
        logger.warning(f"Unknown endpoint: {endpoint}")
        return []

    images_dir = endpoint_dir / "images"
    if not images_dir.exists():
        return []

    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    images = [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]
    images.sort()

    return images


def get_image_prompt_pairs(endpoint: str) -> List[Tuple[Path, str]]:
    """
    Get image-prompt pairs for a specific endpoint.

    1-to-1 mapping: image[i] -> prompt[i] (cycles if needed).

    Args:
        endpoint: Endpoint name

    Returns:
        List of (image_path, prompt) tuples
    """
    images = get_endpoint_images(endpoint)
    prompts = get_prompts_by_endpoint(endpoint)

    if not images or not prompts:
        return []

    # Pair images with prompts (cycle if more images than prompts)
    pairs = []
    for i, image in enumerate(images):
        prompt = prompts[i % len(prompts)]
        pairs.append((image, prompt))

    return pairs


def get_test_data_for_endpoint(endpoint: str, count: Optional[int] = None) -> Dict[str, Any]:
    """
    Get complete test data for an endpoint.

    Args:
        endpoint: Endpoint name
        count: Limit number of image-prompt pairs (None = all)

    Returns:
        Dictionary with images, prompts, and pairs
    """
    pairs = get_image_prompt_pairs(endpoint)

    if count:
        pairs = pairs[:count]

    return {
        "endpoint": endpoint,
        "num_pairs": len(pairs),
        "images": [str(img) for img, _ in pairs],
        "prompts": [prompt for _, prompt in pairs],
        "pairs": [(str(img), prompt) for img, prompt in pairs],
    }


def validate_endpoint_data(endpoint: str) -> Dict[str, Any]:
    """
    Validate test data structure for an endpoint.

    Args:
        endpoint: Endpoint name

    Returns:
        Validation result dictionary
    """
    endpoint_dir = ENDPOINT_DIRS.get(endpoint)
    images = get_endpoint_images(endpoint)
    prompts = get_prompts_by_endpoint(endpoint)

    return {
        "endpoint": endpoint,
        "images_dir_exists": (endpoint_dir / "images").exists() if endpoint_dir else False,
        "image_count": len(images),
        "prompt_count": len(prompts),
        "is_ready": len(images) > 0 and len(prompts) > 0,
        "image_files": [f.name for f in images][:5],  # First 5
    }


def get_all_endpoints_status() -> Dict[str, Dict[str, Any]]:
    """Get status of test data for all endpoints."""
    return {endpoint: validate_endpoint_data(endpoint) for endpoint in ENDPOINT_DIRS.keys()}


def init_test_data_structure() -> None:
    """Initialize test data directory structure."""
    TEST_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for endpoint_dir in ENDPOINT_DIRS.values():
        endpoint_dir.mkdir(parents=True, exist_ok=True)
        images_dir = endpoint_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
