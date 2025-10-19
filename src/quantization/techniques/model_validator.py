"""Model validation and repair utilities for quantization.

Validates HuggingFace model directories and attempts to auto-fix common issues
like missing config.json files.
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger


class ModelValidator:
    """Validates and repairs HuggingFace model directories."""

    # Known model architectures and their HuggingFace identifiers
    MODEL_ARCH_MAPPING = {
        "minicpm": "openbmb/MiniCPM-2B-dpo-fp16",
        "llama": "meta-llama/Llama-2-7b-hf",
        "qwen": "Qwen/Qwen-7B",
        "phi": "microsoft/phi-2",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "gemma": "google/gemma-2b",
        "yi": "01-ai/Yi-6B",
        "deepseek": "deepseek-ai/deepseek-llm-7b-base",
    }

    @staticmethod
    def validate_model_directory(model_path: Path) -> tuple[bool, str, dict]:
        """Validate a HuggingFace model directory.

        Args:
            model_path: Path to model directory

        Returns:
            Tuple of (is_valid, error_message, validation_info)
        """
        validation_info = {
            "has_config": False,
            "has_weights": False,
            "has_tokenizer": False,
            "config_path": None,
            "weight_files": [],
            "issues": [],
        }

        if not model_path.exists():
            return False, f"Model directory does not exist: {model_path}", validation_info

        if not model_path.is_dir():
            return False, f"Model path is not a directory: {model_path}", validation_info

        # Check for config.json
        config_path = model_path / "config.json"
        if config_path.exists():
            validation_info["has_config"] = True
            validation_info["config_path"] = config_path
        else:
            validation_info["issues"].append("Missing config.json")

        # Check for model weights
        weight_patterns = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model-*.bin",
            "model-*.safetensors",
        ]

        weight_files = []
        for pattern in weight_patterns:
            weight_files.extend(model_path.glob(pattern))

        if weight_files:
            validation_info["has_weights"] = True
            validation_info["weight_files"] = [str(f.name) for f in weight_files]
        else:
            validation_info["issues"].append("Missing model weights (pytorch_model.bin or model.safetensors)")

        # Check for tokenizer files
        tokenizer_files = list(model_path.glob("tokenizer*"))
        if tokenizer_files:
            validation_info["has_tokenizer"] = True

        # Validate config.json content if it exists
        if validation_info["has_config"]:
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)

                if "model_type" not in config:
                    validation_info["issues"].append("config.json missing 'model_type' field")

                if "architectures" not in config:
                    validation_info["issues"].append("config.json missing 'architectures' field")

            except json.JSONDecodeError as e:
                validation_info["issues"].append(f"config.json is invalid JSON: {e}")
            except Exception as e:
                validation_info["issues"].append(f"Error reading config.json: {e}")

        # Determine if valid
        is_valid = validation_info["has_config"] and validation_info["has_weights"]
        error_message = ""

        if not is_valid:
            error_message = "Invalid model directory:\n"
            for issue in validation_info["issues"]:
                error_message += f"  - {issue}\n"

        return is_valid, error_message, validation_info

    @staticmethod
    def infer_model_architecture(model_path: Path) -> Optional[str]:
        """Infer model architecture from directory name.

        Args:
            model_path: Path to model directory

        Returns:
            Model architecture identifier or None
        """
        model_name = model_path.name.lower()

        for arch_key, hf_id in ModelValidator.MODEL_ARCH_MAPPING.items():
            if arch_key in model_name:
                logger.info(f"Inferred model architecture '{arch_key}' from name: {model_name}")
                return hf_id

        return None

    @staticmethod
    def attempt_config_repair(model_path: Path) -> bool:
        """Attempt to repair missing or invalid config.json.

        Args:
            model_path: Path to model directory

        Returns:
            True if repair was successful
        """
        config_path = model_path / "config.json"

        # If config exists and is valid, no repair needed
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                if "model_type" in config:
                    return True  # Already valid
            except:
                pass

        # Try to infer architecture and fetch config
        hf_model_id = ModelValidator.infer_model_architecture(model_path)

        if hf_model_id:
            logger.info(f"Attempting to fetch config.json from HuggingFace: {hf_model_id}")

            try:
                from huggingface_hub import hf_hub_download

                # Download config.json from HuggingFace
                downloaded_config = hf_hub_download(
                    repo_id=hf_model_id,
                    filename="config.json",
                    cache_dir=model_path / ".cache",
                )

                # Copy to model directory
                import shutil
                shutil.copy(downloaded_config, config_path)

                logger.info(f"Successfully downloaded and installed config.json")

                # Also try to get tokenizer files if missing
                tokenizer_files = ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]
                for tokenizer_file in tokenizer_files:
                    if not (model_path / tokenizer_file).exists():
                        try:
                            downloaded_file = hf_hub_download(
                                repo_id=hf_model_id,
                                filename=tokenizer_file,
                                cache_dir=model_path / ".cache",
                            )
                            shutil.copy(downloaded_file, model_path / tokenizer_file)
                            logger.debug(f"Downloaded {tokenizer_file}")
                        except Exception as e:
                            logger.debug(f"Could not download {tokenizer_file}: {e}")

                return True

            except Exception as e:
                logger.warning(f"Could not auto-repair config.json: {e}")
                return False

        logger.warning("Could not infer model architecture for auto-repair")
        return False

    @staticmethod
    def get_helpful_error_message(model_path: Path, validation_info: dict) -> str:
        """Generate a helpful error message for invalid models.

        Args:
            model_path: Path to model directory
            validation_info: Validation info from validate_model_directory

        Returns:
            Helpful error message with suggestions
        """
        msg = f"\n{'='*60}\n"
        msg += "MODEL VALIDATION FAILED\n"
        msg += f"{'='*60}\n\n"
        msg += f"Model path: {model_path}\n\n"
        msg += "Issues found:\n"

        for issue in validation_info["issues"]:
            msg += f"  ❌ {issue}\n"

        msg += "\n"
        msg += f"{'='*60}\n"
        msg += "HOW TO FIX\n"
        msg += f"{'='*60}\n\n"

        if not validation_info["has_config"]:
            msg += "Missing config.json:\n"
            msg += "  1. Download the full model from HuggingFace (including config.json)\n"
            msg += "  2. Or use the model's HuggingFace identifier instead of local path\n"
            msg += "  3. Or manually create config.json with the model architecture\n\n"

            # Check if we can infer the model
            hf_id = ModelValidator.infer_model_architecture(model_path)
            if hf_id:
                msg += f"  💡 Auto-detected: This looks like a '{hf_id}' model\n"
                msg += f"     Try downloading from: https://huggingface.co/{hf_id}\n\n"

        if not validation_info["has_weights"]:
            msg += "Missing model weights:\n"
            msg += "  1. Ensure pytorch_model.bin or model.safetensors exists\n"
            msg += "  2. Check if the model was fully downloaded\n\n"

        msg += f"{'='*60}\n"

        return msg
