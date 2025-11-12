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
    # Complexity Pattern: Easy, Medium, Hard, cycling
    # 1-Easy
    "What is Python?",
    # 2-Medium
    "Explain the difference between lists and tuples in Python.",
    # 3-Hard
    "How does Python's Global Interpreter Lock (GIL) affect multi-threaded performance?",
    # 4-Easy
    "What is an API?",
    # 5-Medium
    "Describe how REST APIs differ from GraphQL APIs.",
    # 6-Hard
    "Explain the CAP theorem and its implications for distributed systems.",
    # 7-Easy
    "Define machine learning.",
    # 8-Medium
    "What are the key differences between supervised and unsupervised learning?",
    # 9-Hard
    "Explain the mathematical foundation of backpropagation in neural networks.",
    # 10-Easy
    "What is a database?",
    # 11-Medium
    "Compare SQL and NoSQL databases in terms of use cases.",
    # 12-Hard
    "Describe database normalization and denormalization trade-offs in OLTP vs OLAP systems.",
    # 13-Easy
    "What is cloud computing?",
    # 14-Medium
    "Explain the differences between IaaS, PaaS, and SaaS.",
    # 15-Hard
    "How do microservices architectures handle distributed transactions and eventual consistency?",
    # 16-Easy
    "What is version control?",
    # 17-Medium
    "Explain git branching strategies for team collaboration.",
    # 18-Hard
    "Describe the internals of git's object storage and how it achieves deduplication.",
    # 19-Easy
    "What is cybersecurity?",
    # 20-Medium
    "Explain common web vulnerabilities like XSS and SQL injection.",
]

# Short prompts for speed testing (Easy/Medium/Hard pattern)
SPEED_TEST_PROMPTS: List[str] = [
    # 1-Easy
    "Define AI.",
    # 2-Medium
    "Explain recursion briefly.",
    # 3-Hard
    "What is P vs NP?",
    # 4-Easy
    "What is HTTP?",
    # 5-Medium
    "Difference between HTTP and HTTPS?",
    # 6-Hard
    "How does TLS handshake work?",
    # 7-Easy
    "What is RAM?",
    # 8-Medium
    "Explain virtual memory.",
    # 9-Hard
    "Describe cache coherence protocols.",
    # 10-Easy
    "What is JSON?",
    # 11-Medium
    "JSON vs XML comparison.",
    # 12-Hard
    "Explain JSON Schema validation.",
    # 13-Easy
    "Define algorithm.",
    # 14-Medium
    "What is Big O notation?",
    # 15-Hard
    "Analyze quicksort time complexity.",
    # 16-Easy
    "What is a function?",
    # 17-Medium
    "Explain higher-order functions.",
    # 18-Hard
    "Describe function currying benefits.",
    # 19-Easy
    "What is encryption?",
    # 20-Medium
    "Symmetric vs asymmetric encryption?",
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

# Quality test prompts (deterministic answers - Easy/Medium/Hard pattern)
QUALITY_TEST_PROMPTS: List[str] = [
    # 1-Easy
    "What is 5 + 7?",
    # 2-Medium
    "Calculate 15 percent of 200.",
    # 3-Hard
    "What is the square root of 144?",
    # 4-Easy
    "What is the capital of France?",
    # 5-Medium
    "Name the largest ocean on Earth.",
    # 6-Hard
    "What year did the first human land on the moon?",
    # 7-Easy
    "How many hours in a day?",
    # 8-Medium
    "How many weeks in a year?",
    # 9-Hard
    "How many seconds in a week?",
    # 10-Easy
    "What is H2O?",
    # 11-Medium
    "What is the atomic number of carbon?",
    # 12-Hard
    "What is Avogadro's number?",
    # 13-Easy
    "Spell 'necessary'.",
    # 14-Medium
    "What is the past tense of 'go'?",
    # 15-Hard
    "Define 'onomatopoeia'.",
    # 16-Easy
    "What color is the sky?",
    # 17-Medium
    "Name the primary colors.",
    # 18-Hard
    "Explain RGB color model.",
    # 19-Easy
    "How many continents exist?",
    # 20-Medium
    "Name the smallest country by area.",
]


# ============================================
# VLM Default Prompts - By Endpoint
# ============================================

# Visual Q&A Endpoint Prompts (Easy/Medium/Hard pattern)
VLM_VISION_QA_PROMPTS: List[str] = [
    # 1-Easy
    "Is there are watermelon in this image? Respond in a single sentence.",
    # 2-Medium
    "Describe this image, and the items in this image.",
    # 3-Hard
    "Analyze this image and explain the relationships between the objects present.",
    # 4-Easy
    "What colors are most prominent?",
    # 5-Medium
    "Identify the number of people and their activities in the image.",
    # 6-Hard
    "What cultural or historical context can you infer from this location?",
    # 7-Easy
    "Are there any people in this image?",
    # 8-Medium
    "Assess the pavement condition.",
    # 9-Hard
    "Analyze the item type, color, pattern, closure, handle-length and its brand based on visible logos or design features.",
    # 10-Easy
    "What animals can you see?",
    # 11-Medium
    "Identify the species and behavior of animals present.",
    # 12-Hard
    "What can you deduce about the animals' habitat or environment?",
    # 13-Easy
    "Is this indoors or outdoors?",
    # 14-Medium
    "What time of day does this appear to be taken?",
    # 15-Hard
    "Estimate the season and weather conditions based on visual cues.",
    # 16-Easy
    "What text is visible?",
    # 17-Medium
    "Read and interpret any signs, labels, or written content.",
    # 18-Hard
    "What language is the text in, and what does it communicate?",
    # 19-Easy
    "Count how many objects are present.",
    # 20-Medium
    "Identify and categorize the different types of objects.",
]

# Image Captioning Endpoint Prompts
VLM_CAPTION_PROMPTS: List[str] = [
    # Complexity Pattern: Easy, Medium, Hard, cycling
    # 1-Easy
    "Describe this image.",
    # 2-Medium
    "Write a detailed caption explaining the scene and context.",
    # 3-Hard
    "Create a caption that describes the image composition, lighting, mood, and any symbolic elements.",
    # 4-Easy
    "What do you see?",
    # 5-Medium
    "Generate a social media caption with hashtags for this image.",
    # 6-Hard
    "Provide a professional photography critique describing technique, subject matter, and artistic choices.",
    # 7-Easy
    "Caption this photo.",
    # 8-Medium
    "Write a news headline and brief description for this image.",
    # 9-Hard
    "Describe the image in a way that would help a blind person understand the visual elements, spatial relationships, and atmosphere.",
    # 10-Easy
    "What is happening here?",
    # 11-Medium
    "Create an engaging caption for this image targeting a specific audience.",
    # 12-Hard
    "Analyze the image composition using the rule of thirds, leading lines, and color theory, then provide a caption.",
    # 13-Easy
    "Summarize this image.",
    # 14-Medium
    "Write a caption that tells a story about what might have happened before or after this moment.",
    # 15-Hard
    "Provide a technical description including camera settings, perspective, depth of field, and post-processing effects visible in the image.",
    # 16-Easy
    "Describe the scene.",
    # 17-Medium
    "Create a poetic caption that captures the emotion and atmosphere of this image.",
    # 18-Hard
    "Write a comprehensive caption analyzing the cultural context, symbolism, and potential interpretations of this image.",
    # 19-Easy
    "What's in this picture?",
    # 20-Medium
    "Generate an SEO-optimized caption with relevant keywords for this image.",
]

# Object Detection Endpoint Prompts
VLM_DETECT_PROMPTS: List[str] = [
    # Complexity Pattern: Easy, Medium, Hard, cycling
    # 1-Easy
    "What objects do you see?",
    # 2-Medium
    "List all objects in the foreground and background separately.",
    # 3-Hard
    "Identify all objects with their positions, sizes, and spatial relationships to each other.",
    # 4-Easy
    "Are there any people?",
    # 5-Medium
    "Count and categorize all living things (people, animals, plants) in the image.",
    # 6-Hard
    "Detect all objects and classify them by material composition (metal, wood, fabric, etc.) and purpose.",
    # 7-Easy
    "What vehicles can you see?",
    # 8-Medium
    "Identify all transportation-related objects and their states (moving, parked, etc.).",
    # 9-Hard
    "Detect all mechanical and electronic devices, estimate their models, and describe their condition.",
    # 10-Easy
    "Is there any furniture?",
    # 11-Medium
    "List all furniture items with their approximate styles and materials.",
    # 12-Hard
    "Identify all architectural elements, furniture, and fixtures with their historical period and design influences.",
    # 13-Easy
    "What animals are present?",
    # 14-Medium
    "Detect and count all animals, identifying species and their activities.",
    # 15-Hard
    "Identify all biological entities with taxonomic classification, estimated age, and behavioral analysis.",
    # 16-Easy
    "What colors are dominant?",
    # 17-Medium
    "List all distinct objects with their primary and secondary colors.",
    # 18-Hard
    "Detect all objects, provide color palettes in hex codes, and analyze color relationships and harmony.",
    # 19-Easy
    "What's the main object?",
    # 20-Medium
    "Identify the focal point object and all supporting elements in the composition.",
]

# Object Pointing Endpoint Prompts
VLM_POINT_PROMPTS: List[str] = [
    # Complexity Pattern: Easy, Medium, Hard, cycling
    # 1-Easy
    "Where is the main object?",
    # 2-Medium
    "Identify the coordinates and size of all people in the image.",
    # 3-Hard
    "Provide bounding box coordinates for all objects with their relative positions using a grid system.",
    # 4-Easy
    "What's in the center?",
    # 5-Medium
    "Locate all text elements and describe their positions relative to image quadrants.",
    # 6-Hard
    "Map the spatial distribution of all elements using percentage-based coordinates from top-left origin.",
    # 7-Easy
    "Where are the people?",
    # 8-Medium
    "Identify the location and orientation of all faces in the image.",
    # 9-Hard
    "Provide precise landmark points for all human figures including head, shoulders, hands, and feet positions.",
    # 10-Easy
    "What's in the corner?",
    # 11-Medium
    "Describe the z-depth ordering of objects from foreground to background.",
    # 12-Hard
    "Create a spatial map showing relative distances between all objects using proportional measurements.",
    # 13-Easy
    "Where is the vehicle?",
    # 14-Medium
    "Point to all interactive elements (buttons, signs, controls) and describe their accessibility.",
    # 15-Hard
    "Provide a complete spatial scene graph with object relationships, proximities, and directional vectors.",
    # 16-Easy
    "What's at the top?",
    # 17-Medium
    "Identify the vanishing points and describe the perspective geometry of the scene.",
    # 18-Hard
    "Calculate the optical center, rule-of-thirds intersections, and golden ratio points in the composition.",
    # 19-Easy
    "Where's the animal?",
    # 20-Medium
    "Map all regions of interest using attention heatmap coordinates.",
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


def get_default_prompts_for_suite(suite_name: str, model_type, count: int = 20) -> List[str]:
    """
    Get a specific number of default prompts for a suite.

    Args:
        suite_name: Suite name ('speed', 'quality', 'stress', 'resources', 'complete')
        model_type: ModelType enum (ModelType.LLM or ModelType.VLM)
        count: Number of prompts to return (1-20)

    Returns:
        List of prompts (up to count items)
    """
    from src.benchmarking.models.metric_types import ModelType

    # Convert ModelType enum to string
    model_type_str = "llm" if model_type == ModelType.LLM else "vlm"

    # Get full suite prompts list
    all_prompts = get_prompts_for_suite(suite_name, model_type_str)

    # Return first N prompts
    return all_prompts[:min(count, len(all_prompts))]


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
# Use repository-local dataset under src/benchmarking/datasets/test_data
TEST_DATA_ROOT = Path(__file__).parent / "test_data"

ENDPOINT_DIRS = {
    "vision/qa": TEST_DATA_ROOT / "vision_qa",
    "vision/caption": TEST_DATA_ROOT / "vision_caption",
    "vision/detect": TEST_DATA_ROOT / "vision_detect",
    "vision/point": TEST_DATA_ROOT / "vision_point",
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

    # Get all image files, ignore hidden and AppleDouble files (e.g., ._image.jpg)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    images = [
        f for f in images_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in image_extensions
        and not f.name.startswith('.')  # excludes .DS_Store, ._ files, hidden files
    ]
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
