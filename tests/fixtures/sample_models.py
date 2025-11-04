"""
Sample model definitions for testing.

Provides mock model configurations and metadata.
"""

from typing import Dict, List, Any


# Sample LLM models
SAMPLE_LLM_MODELS: Dict[str, Dict[str, Any]] = {
    "llama2:7b": {
        "provider": "ollama",
        "type": "llm",
        "endpoints": ["text/chat", "text/completion"],
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9
        },
        "metadata": {
            "size_gb": 3.8,
            "quantization": "q4_0",
            "context_length": 4096
        }
    },
    "mistral:7b": {
        "provider": "ollama",
        "type": "llm",
        "endpoints": ["text/chat", "text/completion"],
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "metadata": {
            "size_gb": 4.1,
            "quantization": "q4_0",
            "context_length": 8192
        }
    },
    "phi-2": {
        "provider": "ollama",
        "type": "llm",
        "endpoints": ["text/chat"],
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 1024
        },
        "metadata": {
            "size_gb": 1.6,
            "quantization": "q4_0",
            "context_length": 2048
        }
    }
}


# Sample VLM models
SAMPLE_VLM_MODELS: Dict[str, Dict[str, Any]] = {
    "llava:7b": {
        "provider": "ollama",
        "type": "vlm",
        "endpoints": ["vision/analyze", "vision/describe"],
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 512
        },
        "metadata": {
            "size_gb": 4.5,
            "quantization": "q4_0",
            "vision_encoder": "clip",
            "context_length": 2048
        }
    },
    "llava:13b": {
        "provider": "ollama",
        "type": "vlm",
        "endpoints": ["vision/analyze", "vision/describe"],
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 1024
        },
        "metadata": {
            "size_gb": 7.3,
            "quantization": "q4_0",
            "vision_encoder": "clip",
            "context_length": 4096
        }
    },
    "bakllava": {
        "provider": "ollama",
        "type": "vlm",
        "endpoints": ["vision/analyze"],
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 512
        },
        "metadata": {
            "size_gb": 4.4,
            "quantization": "q4_0",
            "vision_encoder": "clip",
            "context_length": 2048
        }
    }
}


# Combined registry
ALL_SAMPLE_MODELS: Dict[str, Dict[str, Any]] = {
    **SAMPLE_LLM_MODELS,
    **SAMPLE_VLM_MODELS
}


def get_model_by_id(model_id: str) -> Dict[str, Any]:
    """Get model configuration by ID."""
    return ALL_SAMPLE_MODELS.get(model_id, {})


def get_models_by_type(model_type: str) -> Dict[str, Dict[str, Any]]:
    """Get all models of a specific type (llm or vlm)."""
    return {
        model_id: config
        for model_id, config in ALL_SAMPLE_MODELS.items()
        if config.get("type") == model_type
    }


def get_available_endpoints(model_id: str) -> List[str]:
    """Get available endpoints for a model."""
    model = get_model_by_id(model_id)
    return model.get("endpoints", [])
