"""Automatic dependency installer for missing Python packages.

Handles automatic installation of missing dependencies with proper error handling.
"""

import subprocess
import sys
from typing import Optional, Union

from loguru import logger


class DependencyInstaller:
    """Auto-install missing Python packages."""

    @staticmethod
    def install_package(
        package_name: str,
        pip_name: Optional[str] = None,
        version: Optional[str] = None,
        auto_install: bool = True,
        quiet: bool = False
    ) -> bool:
        """Install a Python package if it's missing.

        Args:
            package_name: Name of the package to import (e.g., "transformers")
            pip_name: Name of the package on PyPI (defaults to package_name)
            version: Specific version to install (e.g., ">=4.30.0")
            auto_install: Whether to auto-install without prompting
            quiet: Suppress installation output

        Returns:
            True if package is available (was already installed or successfully installed)

        Example:
            >>> DependencyInstaller.install_package("transformers", version=">=4.30.0")
            >>> DependencyInstaller.install_package("PIL", pip_name="Pillow")
        """
        pip_package_name = pip_name or package_name

        # Check if already installed
        try:
            __import__(package_name)
            return True
        except ImportError:
            pass

        if not auto_install:
            logger.warning(
                f"Package '{package_name}' is not installed.\n"
                f"Install with: pip install {pip_package_name}"
            )
            return False

        # Build pip install command
        if version:
            install_target = f"{pip_package_name}{version}"
        else:
            install_target = pip_package_name

        logger.info(f"Installing {install_target}...")

        try:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                install_target,
            ]

            if quiet:
                cmd.append("--quiet")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"✓ Successfully installed {pip_package_name}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {pip_package_name}: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error installing {pip_package_name}: {e}")
            return False

    @staticmethod
    def ensure_packages(
        packages: Union[str, list[str], dict[str, str]],
        auto_install: bool = True,
        quiet: bool = False
    ) -> bool:
        """Ensure multiple packages are installed.

        Args:
            packages: Package name(s) or dict of {import_name: pip_name}
            auto_install: Whether to auto-install without prompting
            quiet: Suppress installation output

        Returns:
            True if all packages are available

        Example:
            >>> DependencyInstaller.ensure_packages(["torch", "transformers"])
            >>> DependencyInstaller.ensure_packages({"PIL": "Pillow", "cv2": "opencv-python"})
        """
        if isinstance(packages, str):
            packages = [packages]

        if isinstance(packages, dict):
            # Dict format: {import_name: pip_name}
            results = []
            for import_name, pip_name in packages.items():
                result = DependencyInstaller.install_package(
                    import_name,
                    pip_name=pip_name,
                    auto_install=auto_install,
                    quiet=quiet
                )
                results.append(result)
            return all(results)
        else:
            # List format: [package1, package2, ...]
            results = []
            for package in packages:
                result = DependencyInstaller.install_package(
                    package,
                    auto_install=auto_install,
                    quiet=quiet
                )
                results.append(result)
            return all(results)

    @staticmethod
    def check_and_install_transformers_dependency(
        original_model_id: str,
        auto_install: bool = True
    ) -> bool:
        """Check if a model requires specific transformers version and install if needed.

        Args:
            original_model_id: Original HuggingFace model ID
            auto_install: Whether to auto-install

        Returns:
            True if dependencies are satisfied
        """
        # Special cases for models requiring newer transformers
        if any(pattern in original_model_id.lower() for pattern in ["qwen3", "gemma3", "llama4"]):
            logger.info(f"Model {original_model_id} may require transformers>=4.40.0")
            return DependencyInstaller.install_package(
                "transformers",
                version=">=4.40.0",
                auto_install=auto_install,
                quiet=True
            )

        return True
