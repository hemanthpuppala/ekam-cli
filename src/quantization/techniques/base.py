"""Base quantizer interface for all quantization techniques."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from ..models import QuantizationTask, QuantizationType
from ...models.model import ModelInfo


class BaseQuantizer(ABC):
    """Abstract base class for quantization implementations."""

    def check_and_install_dependencies(self, auto_install: bool = True, show_progress: bool = True) -> tuple[bool, str]:
        """Check availability and auto-install missing dependencies if needed.
        
        This method wraps check_availability() and adds automatic dependency installation.
        
        Args:
            auto_install: If True, attempt to install missing dependencies
            show_progress: If True, show TUI progress panels during installation
            
        Returns:
            (available, message) tuple where:
            - available: True if technique can be used after installation
            - message: Description of availability status or installation results
        """
        # First check if already available
        available, message = self.check_availability()
        
        if available:
            return True, message
        
        # Not available - try to extract missing packages from error message
        if not auto_install:
            return False, message
        
        # Try to detect missing packages from the error message
        missing_packages = self._extract_missing_packages(message)
        
        if not missing_packages:
            # Can't auto-install if we don't know what's missing
            return False, message
        
        logger.info(f"Attempting to install missing dependencies: {missing_packages}")
        
        try:
            # Use architecture registry for installation
            from ...utils.architecture_registry import get_architecture_registry
            registry = get_architecture_registry()
            
            success, installed, failed = registry.install_dependencies(
                dependencies=missing_packages,
                auto_approve=True,  # Auto-approve for quantization dependencies
                show_progress=show_progress
            )
            
            if success:
                # Re-check availability after installation
                available, new_message = self.check_availability()
                if available:
                    installed_str = ', '.join(installed)
                    return True, f"✓ Installed {installed_str}. {new_message}"
                else:
                    return False, f"Installation succeeded but technique still unavailable: {new_message}"
            else:
                failed_str = ', '.join(failed) if failed else 'unknown packages'
                return False, f"Failed to install dependencies: {failed_str}. {message}"
                
        except Exception as e:
            logger.error(f"Error during dependency installation: {e}")
            return False, f"Dependency installation failed: {e}. {message}"
    
    def _extract_missing_packages(self, error_message: str) -> list[str]:
        """Extract package names from error message.
        
        Args:
            error_message: Error message from check_availability()
            
        Returns:
            List of package names to install
        """
        packages = []
        
        # Common patterns in error messages
        # "MLX not installed. Install with: pip install mlx mlx-lm mlx-vlm"
        # "Missing packages: mlx-lm, mlx-vlm. Install with: pip install mlx-lm mlx-vlm"
        # "OpenVINO not installed. Install with: pip install openvino openvino-dev"
        
        import re
        
        # Pattern 1: "pip install <packages>"
        pip_match = re.search(r'pip install ([a-zA-Z0-9\-_ ]+)', error_message)
        if pip_match:
            pkg_str = pip_match.group(1).strip()
            packages.extend([p.strip() for p in pkg_str.split() if p.strip()])
        
        # Pattern 2: "Missing packages: pkg1, pkg2"
        missing_match = re.search(r'Missing packages?: ([a-zA-Z0-9\-_, ]+)', error_message, re.IGNORECASE)
        if missing_match:
            pkg_str = missing_match.group(1).strip()
            packages.extend([p.strip() for p in pkg_str.replace(',', ' ').split() if p.strip()])
        
        # Pattern 3: "<package> not installed"
        not_installed_match = re.search(r'(\w+(?:-\w+)*)\s+not installed', error_message, re.IGNORECASE)
        if not_installed_match:
            pkg = not_installed_match.group(1).strip()
            if pkg and pkg not in packages:
                packages.append(pkg)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_packages = []
        for pkg in packages:
            if pkg.lower() not in seen:
                seen.add(pkg.lower())
                unique_packages.append(pkg)
        
        logger.debug(f"Extracted packages from error message: {unique_packages}")
        return unique_packages

    @abstractmethod
    def check_availability(self) -> tuple[bool, str]:
        """Check if this quantization technique is available.

        Returns:
            (available, message) tuple where:
            - available: True if technique can be used
            - message: Description of availability status
        """
        pass

    @abstractmethod
    def get_supported_types(self) -> list[QuantizationType]:
        """Get list of supported quantization types.

        Returns:
            List of QuantizationType enums this quantizer supports
        """
        pass

    @abstractmethod
    def get_source_model_path(self, model_info: ModelInfo) -> Optional[Path]:
        """Get source model path for quantization.

        Args:
            model_info: Model to quantize

        Returns:
            Path to model file, or None if not available/compatible
        """
        pass

    @abstractmethod
    def quantize(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[float, Optional[float]], None]] = None,
    ) -> bool:
        """Perform quantization.

        Args:
            task: Quantization task with all parameters
            progress_callback: Optional callback(progress%, eta_seconds)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def estimate_output_size(
        self, model_info: ModelInfo, quant_type: QuantizationType
    ) -> float:
        """Estimate output file size in GB.

        Args:
            model_info: Source model
            quant_type: Quantization type

        Returns:
            Estimated size in GB
        """
        pass

    def validate_compatibility(self, model_info: ModelInfo) -> tuple[bool, str]:
        """Validate if model is compatible with this quantizer.

        Args:
            model_info: Model to check

        Returns:
            (compatible, reason) tuple
        """
        source_path = self.get_source_model_path(model_info)
        if source_path is None:
            return False, "Model not available or incompatible format"
        if not source_path.exists():
            return False, f"Model file not found: {source_path}"
        return True, "Model is compatible"
