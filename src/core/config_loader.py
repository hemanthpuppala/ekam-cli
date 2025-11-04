"""Configuration loading and validation."""

from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from ..models.endpoints import ProviderType
from ..models.provider import ProviderConfig


def load_config(path: Path = Path("config.yaml")) -> dict[ProviderType, ProviderConfig]:
    """Load and validate config.yaml using PyYAML + Pydantic.

    Args:
        path: Path to config.yaml file

    Returns:
        Dict mapping ProviderType to ProviderConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config is invalid
        ValueError: If YAML is malformed
    """
    # Check if file exists
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        logger.info("Using default configuration")
        return _get_default_config()

    # Load YAML
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {path}: {e}")
        raise ValueError(f"Invalid YAML in config file: {e}")

    if not data or "providers" not in data:
        logger.warning("Config file missing 'providers' section, using defaults")
        return _get_default_config()

    # Parse and validate providers
    provider_configs = {}

    for provider_name, provider_data in data["providers"].items():
        try:
            # Add provider name to data
            provider_data["name"] = provider_name

            # Create ProviderConfig
            config = ProviderConfig(**provider_data)
            provider_type = ProviderType(provider_name)
            provider_configs[provider_type] = config

            logger.debug(f"Loaded config for {provider_name}: enabled={config.enabled}")

        except ValidationError as e:
            logger.error(f"Invalid config for provider {provider_name}: {e}")
            # Continue with other providers
        except ValueError as e:
            logger.warning(f"Unknown provider type: {provider_name}, skipping")

    if not provider_configs:
        logger.warning("No valid provider configs found, using defaults")
        return _get_default_config()

    logger.info(f"Loaded configuration for {len(provider_configs)} providers")
    return provider_configs


def _get_default_config() -> dict[ProviderType, ProviderConfig]:
    """Get default configuration with Ollama only.

    Returns:
        Dict with default Ollama configuration
    """
    return {
        ProviderType.OLLAMA: ProviderConfig(
            name=ProviderType.OLLAMA,
            enabled=True,
            host="http://localhost:11434",
            timeout_seconds=300,  # 5 minutes for VLM inference on slower devices (increased from 120s)
        )
    }


def create_example_config(path: Path = Path("config.yaml")) -> None:
    """Create example config file from template.

    Args:
        path: Path where to create config file
    """
    template_path = Path("config.yaml.example")

    if not template_path.exists():
        logger.error("config.yaml.example not found")
        return

    if path.exists():
        logger.warning(f"{path} already exists, not overwriting")
        return

    # Copy template to config
    import shutil

    shutil.copy(template_path, path)
    logger.info(f"Created {path} from template")
