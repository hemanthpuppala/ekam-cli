"""Dynamic architecture registry for automatic dependency management.

This module provides:
- Automatic detection of missing dependencies from model loading errors
- Dynamic installation of required packages (respects virtual environments)
- Persistent storage of architecture requirements across sessions
- Machine-wide configuration for future-proof architecture support
"""

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class ArchitectureRegistry:
    """Manage architecture requirements and dependencies dynamically."""

    # Machine-wide registry path (persists across all sessions)
    REGISTRY_PATH = Path.home() / ".cache" / "vlm-tester" / "architecture_registry.json"

    # Platform and hardware compatibility matrix
    # Format: package_name -> {"requires": [...], "incompatible_with": [...]}
    PACKAGE_COMPATIBILITY = {
        # CUDA-only packages (require NVIDIA GPU + CUDA)
        "triton": {
            "requires": ["cuda", "nvidia_gpu"],
            "incompatible_with": ["darwin_cpu", "darwin_mps", "windows_cpu", "linux_cpu"],
            "optional": True,  # Model can work without it (fallback available)
            "fallback_message": "Using CPU/MPS fallback (slower but functional)"
        },
        "flash-attn": {
            "requires": ["cuda", "nvidia_gpu"],
            "incompatible_with": ["darwin_cpu", "darwin_mps", "windows_cpu", "linux_cpu"],
            "optional": True,
            "fallback_message": "Using standard attention (slower but functional)"
        },
        # Platform-specific packages
        "mps": {
            "requires": ["darwin", "apple_silicon"],
            "incompatible_with": ["windows", "linux"],
            "optional": True,
            "fallback_message": "Using CPU or CUDA instead"
        },
        # Windows-specific issues
        "bitsandbytes": {
            "requires": [],
            "incompatible_with": ["windows"],  # No official Windows support
            "optional": True,
            "fallback_message": "Quantization unavailable on Windows"
        },
    }

    # Known architecture patterns and their dependencies
    DEFAULT_ARCHITECTURES = {
        "transnormer": ["einops"],
        "mamba": ["mamba-ssm", "einops"],
        "retnet": ["einops"],
        "rwkv": ["rwkv"],
        "megabyte": ["einops"],
        "hyena": ["einops", "flash-attn"],
        "s4": ["einops"],
        "gla": ["einops"],
        "flashattention": ["flash-attn"],
        "ring-attention": ["ring-attention"],
        "longnet": ["einops"],
        "striped-hyena": ["einops"],
        "jamba": ["mamba-ssm", "einops"],
        # Common dependencies
        "vision_transformer": ["timm"],
        "clip": ["ftfy", "regex"],
        "blip": ["timm"],
        "sam": ["segment-anything"],
        "dino": ["timm"],
    }

    def __init__(self):
        """Initialize architecture registry with persistent storage."""
        self.registry_path = self.REGISTRY_PATH
        registry = self._load_registry()
        
        # Separate architectures and package mappings
        self.architectures = registry.get('architectures', {})
        self.package_mappings = registry.get('package_mappings', {})
        logger.debug(f"Loaded {len(self.package_mappings)} learned package mappings")

        # Detect and log Python environment info
        self.python_env = self._detect_python_environment()
        logger.debug(f"Python environment: {self.python_env}")

        # Detect platform capabilities once at initialization
        self.platform_capabilities = self._detect_platform_capabilities()
        logger.debug(f"Platform capabilities: {self.platform_capabilities}")

    def _load_registry(self) -> dict[str, Any]:
        """Load architecture registry from disk or create default.

        Returns:
            Dictionary with 'architectures' and 'package_mappings' keys
        """
        # Ensure registry directory exists
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    registry = json.load(f)
                logger.debug(f"Loaded architecture registry from {self.registry_path}")
                
                # Ensure both sections exist (backward compatibility)
                if isinstance(registry, dict) and 'architectures' not in registry:
                    # Old format: just architectures, migrate to new format
                    registry = {
                        'architectures': registry,
                        'package_mappings': {}  # Learned package name mappings
                    }
                    self._save_registry(registry)
                    logger.info("Migrated registry to new format with package mappings")
                
                return registry
            except Exception as e:
                logger.warning(f"Failed to load registry, using defaults: {e}")

        # Create default registry with both sections
        default_registry = {
            'architectures': self.DEFAULT_ARCHITECTURES.copy(),
            'package_mappings': {}  # Will be populated dynamically
        }
        logger.info(f"Creating new architecture registry at {self.registry_path}")
        self._save_registry(default_registry)
        return default_registry

    def _save_registry(self, registry: Optional[dict[str, Any]] = None) -> None:
        """Save architecture registry to disk.

        Args:
            registry: Registry to save (constructs from self.architectures and self.package_mappings if None)
        """
        if registry is None:
            registry = {
                'architectures': self.architectures,
                'package_mappings': self.package_mappings
            }

        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2, sort_keys=True)
            logger.debug(f"Saved architecture registry to {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to save architecture registry: {e}")

    def _detect_python_environment(self) -> dict[str, Any]:
        """Detect Python environment type and installation target.

        Returns:
            Dict with environment information
        """
        env_info = {
            "python_executable": sys.executable,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "sys_prefix": sys.prefix,
            "is_virtualenv": False,
            "is_conda": False,
            "env_type": "system",
            "pip_install_target": None,
        }

        # Detect virtual environment (venv/virtualenv)
        # Check if we're in a venv by comparing prefix with base_prefix
        if hasattr(sys, 'real_prefix'):
            # Old virtualenv
            env_info["is_virtualenv"] = True
            env_info["env_type"] = "virtualenv"
        elif hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
            # Modern venv or virtualenv
            env_info["is_virtualenv"] = True
            env_info["env_type"] = "venv"

        # Detect conda environment
        if "conda" in sys.executable.lower() or os.environ.get("CONDA_DEFAULT_ENV"):
            env_info["is_conda"] = True
            env_info["env_type"] = "conda"
            env_info["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV", "base")

        # Determine pip installation target
        if env_info["is_virtualenv"] or env_info["is_conda"]:
            env_info["pip_install_target"] = "virtual environment"
        else:
            env_info["pip_install_target"] = "system (use --user flag)"

        # Verify pip is available
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                env_info["pip_available"] = True
                env_info["pip_version"] = result.stdout.strip()
            else:
                env_info["pip_available"] = False
                logger.warning("pip not available in this Python environment")
        except Exception as e:
            env_info["pip_available"] = False
            logger.error(f"Failed to check pip availability: {e}")

        return env_info

    def _detect_platform_capabilities(self) -> set[str]:
        """Detect current platform and hardware capabilities.

        Returns:
            Set of capability strings (e.g., {"linux", "cuda", "nvidia_gpu"})
        """
        capabilities = set()

        # Detect OS platform
        system = platform.system().lower()
        capabilities.add(system)

        # More specific platform detection
        if system == "darwin":
            capabilities.add("macos")
            # Detect Apple Silicon vs Intel
            machine = platform.machine().lower()
            if machine in ["arm64", "aarch64"]:
                capabilities.add("apple_silicon")
                capabilities.add("mps_capable")  # Metal Performance Shaders
            else:
                capabilities.add("intel_mac")
        elif system == "linux":
            # Detect Linux distribution
            try:
                import distro
                dist_id = distro.id().lower()
                capabilities.add(dist_id)  # ubuntu, centos, fedora, etc.
            except ImportError:
                pass
        elif system == "windows":
            # Detect Windows version
            capabilities.add("win32")

        # Detect GPU and CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                capabilities.add("cuda")
                capabilities.add("nvidia_gpu")
                # Get CUDA version
                cuda_version = torch.version.cuda
                if cuda_version:
                    capabilities.add(f"cuda_{cuda_version.replace('.', '_')}")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                capabilities.add("mps")
                capabilities.add("apple_gpu")
        except ImportError:
            logger.debug("torch not available for GPU detection")

        # Detect CPU-only mode
        if "cuda" not in capabilities and "mps" not in capabilities:
            capabilities.add("cpu_only")
            # Add platform-specific CPU-only tags
            if system == "darwin":
                capabilities.add("darwin_cpu")
            elif system == "windows":
                capabilities.add("windows_cpu")
            elif system == "linux":
                capabilities.add("linux_cpu")

        # Detect MPS-only mode (Apple Silicon without CUDA)
        if "mps" in capabilities and "cuda" not in capabilities:
            capabilities.add("darwin_mps")

        return capabilities

    def _is_package_compatible(self, package: str) -> tuple[bool, Optional[str]]:
        """Check if a package is compatible with the current platform.

        Args:
            package: Package name to check

        Returns:
            Tuple of (is_compatible, reason/fallback_message)
        """
        # If package not in compatibility matrix, assume it's compatible
        if package not in self.PACKAGE_COMPATIBILITY:
            return True, None

        compat_info = self.PACKAGE_COMPATIBILITY[package]

        # Check if package has specific requirements
        if compat_info.get("requires"):
            required = set(compat_info["requires"])
            if not required.issubset(self.platform_capabilities):
                missing = required - self.platform_capabilities
                reason = f"requires {', '.join(missing)}"
                fallback = compat_info.get("fallback_message")
                return False, fallback or reason

        # Check if package is incompatible with current platform
        if compat_info.get("incompatible_with"):
            incompatible = set(compat_info["incompatible_with"])
            if incompatible.intersection(self.platform_capabilities):
                matched = incompatible.intersection(self.platform_capabilities)
                reason = f"incompatible with {', '.join(matched)}"
                fallback = compat_info.get("fallback_message")
                return False, fallback or reason

        return True, None

    def detect_missing_dependencies(self, error_message: str) -> list[str]:
        """Detect missing dependencies from error message.

        Parses error messages like:
        - "This modeling file requires the following packages that were not found in your environment: einops"
        - "No module named 'mamba_ssm'"
        - "ImportError: cannot import name 'FlashAttention' from 'flash_attn'"

        Args:
            error_message: Error message from model loading

        Returns:
            List of missing package names
        """
        missing_deps = []

        # Pattern 1: HuggingFace transformers error (most common)
        # "This modeling file requires the following packages that were not found in your environment: einops"
        hf_pattern = r"packages that were not found in your environment:\s*([a-zA-Z0-9_,\s-]+)"
        match = re.search(hf_pattern, error_message, re.IGNORECASE)
        if match:
            deps_str = match.group(1)
            # Split by comma or space
            deps = [d.strip() for d in re.split(r'[,\s]+', deps_str) if d.strip()]
            missing_deps.extend(deps)
            logger.debug(f"Detected missing packages (HF pattern): {deps}")

        # Pattern 2: Standard Python import error
        # "No module named 'einops'"
        # "cannot import name 'X' from 'einops'"
        import_pattern = r"(?:No module named|cannot import name .+ from)\s+['\"]([a-zA-Z0-9_-]+)['\"]"
        matches = re.findall(import_pattern, error_message)
        if matches:
            missing_deps.extend(matches)
            logger.debug(f"Detected missing packages (import pattern): {matches}")

        # Pattern 3: ModuleNotFoundError
        # "ModuleNotFoundError: No module named 'mamba_ssm'"
        module_pattern = r"ModuleNotFoundError:.*['\"]([a-zA-Z0-9_-]+)['\"]"
        matches = re.findall(module_pattern, error_message)
        if matches:
            missing_deps.extend(matches)
            logger.debug(f"Detected missing packages (ModuleNotFoundError): {matches}")

        # Normalize package names (handle underscores and dashes)
        # mamba_ssm → mamba-ssm (pip uses dashes)
        normalized = []
        for dep in missing_deps:
            # Keep original name but also try dash variant
            normalized.append(dep)
            if '_' in dep:
                normalized.append(dep.replace('_', '-'))

        return list(set(normalized))  # Remove duplicates

    def detect_architecture(self, model_id: str, error_message: str = "") -> Optional[str]:
        """Detect architecture from model ID or error message.

        Args:
            model_id: HuggingFace model ID (e.g., "OpenNLPLab/TransNormerLLM-1B")
            error_message: Optional error message for additional context

        Returns:
            Detected architecture name (lowercase) or None
        """
        model_lower = model_id.lower()

        # Check known architectures by keyword matching
        for arch_name in self.architectures.keys():
            # Match architecture name in model ID
            if arch_name.lower() in model_lower:
                logger.debug(f"Detected architecture '{arch_name}' from model ID")
                return arch_name.lower()

        # Try to extract architecture from model name patterns
        # TransNormerLLM → transnormer
        arch_patterns = [
            r'([a-zA-Z]+)LLM',  # TransNormerLLM → TransNormer
            r'([a-zA-Z]+)-\d+[BM]',  # Mamba-1B → Mamba
            r'/([a-zA-Z]+)-',  # org/architecture-version
        ]

        for pattern in arch_patterns:
            match = re.search(pattern, model_id, re.IGNORECASE)
            if match:
                arch_name = match.group(1).lower()
                logger.debug(f"Extracted potential architecture '{arch_name}' from pattern")
                return arch_name

        return None

    def get_dependencies(self, architecture: str) -> list[str]:
        """Get required dependencies for an architecture.

        Args:
            architecture: Architecture name (case-insensitive)

        Returns:
            List of required package names
        """
        arch_lower = architecture.lower()
        return self.architectures.get(arch_lower, [])

    def register_architecture(self, architecture: str, dependencies: list[str]) -> None:
        """Register a new architecture with its dependencies.

        Args:
            architecture: Architecture name
            dependencies: List of required package names
        """
        arch_lower = architecture.lower()

        # Merge with existing dependencies
        existing_deps = set(self.architectures.get(arch_lower, []))
        new_deps = set(dependencies)
        merged_deps = sorted(list(existing_deps | new_deps))

        self.architectures[arch_lower] = merged_deps
        self._save_registry()

        logger.info(f"Registered architecture '{arch_lower}' with dependencies: {merged_deps}")

    def _filter_platform_compatible_deps(
        self,
        dependencies: list[str]
    ) -> tuple[list[str], dict[str, str]]:
        """Filter dependencies based on platform and hardware compatibility.

        Args:
            dependencies: List of package names

        Returns:
            Tuple of (compatible_packages, skipped_packages_with_reasons)
        """
        compatible = []
        skipped = {}  # package -> reason

        for dep in dependencies:
            is_compatible, reason = self._is_package_compatible(dep)

            if is_compatible:
                compatible.append(dep)
            else:
                skipped[dep] = reason
                logger.warning(f"Skipping {dep}: {reason}")

        return compatible, skipped

    def _generate_package_name_alternatives(self, package_name: str) -> list[str]:
        """Generate alternative package names to try.
        
        Args:
            package_name: Original package name from error
            
        Returns:
            List of alternative package names to try, in order of likelihood
        """
        alternatives = []
        
        # 1. Check learned mappings first
        if package_name in self.package_mappings:
            learned_name = self.package_mappings[package_name]
            alternatives.append(learned_name)
            logger.debug(f"Found learned mapping: {package_name} -> {learned_name}")
        
        # 2. Try common transformations
        # Underscore to dash (e.g., hf_olmo -> hf-olmo)
        if '_' in package_name:
            alternatives.append(package_name.replace('_', '-'))
        
        # Dash to underscore (e.g., hf-olmo -> hf_olmo)
        if '-' in package_name:
            alternatives.append(package_name.replace('-', '_'))
        
        # 3. Try common prefixes (e.g., hf_olmo -> ai2-olmo, olmo)
        # Remove common prefixes like hf_, hf-, transformers-, etc.
        prefixes_to_try = ['hf_', 'hf-', 'transformers_', 'transformers-', 'pytorch_', 'pytorch-']
        for prefix in prefixes_to_try:
            if package_name.startswith(prefix):
                base_name = package_name[len(prefix):]
                alternatives.append(base_name)
                # Also try with ai2- prefix (common for AI2 packages)
                alternatives.append(f"ai2-{base_name}")
                # Try without any prefix
                if '_' in base_name:
                    alternatives.append(base_name.replace('_', '-'))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_alternatives = []
        for alt in alternatives:
            if alt not in seen and alt != package_name:
                seen.add(alt)
                unique_alternatives.append(alt)
        
        return unique_alternatives

    def _try_install_package(self, package_name: str) -> tuple[bool, Optional[str]]:
        """Try to install a single package.
        
        Args:
            package_name: Package name to install
            
        Returns:
            Tuple of (success, error_message)
        """
        pip_cmd = [sys.executable, "-m", "pip", "install", package_name, "--quiet"]
        
        # Add --user flag if not in virtual environment
        if not (self.python_env["is_virtualenv"] or self.python_env["is_conda"]):
            pip_cmd.append("--user")
        
        try:
            result = subprocess.run(
                pip_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return True, None
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return False, error_msg
                
        except Exception as e:
            return False, str(e)

    def _learn_package_mapping(self, original_name: str, actual_name: str) -> None:
        """Learn and save a successful package name mapping.
        
        Args:
            original_name: Package name that appeared in error
            actual_name: Actual package name that worked
        """
        if original_name != actual_name:
            self.package_mappings[original_name] = actual_name
            self._save_registry()
            logger.info(f"✓ Learned package mapping: {original_name} -> {actual_name}")

    def install_dependencies(
        self,
        dependencies: list[str],
        auto_approve: bool = False,
        show_progress: bool = True
    ) -> tuple[bool, list[str], list[str]]:
        """Install missing dependencies using pip.

        Args:
            dependencies: List of package names to install
            auto_approve: If True, install without user confirmation
            show_progress: If True, show TUI progress panels

        Returns:
            Tuple of (success, installed_packages, failed_packages)
        """
        if not dependencies:
            return True, [], []

        # Filter platform-incompatible dependencies
        compatible_deps, skipped_deps_dict = self._filter_platform_compatible_deps(dependencies)

        if skipped_deps_dict:
            logger.info(f"Skipped platform-incompatible packages: {list(skipped_deps_dict.keys())}")

        if not compatible_deps:
            logger.info("No compatible dependencies to install")
            # All dependencies were skipped - this is OK if they're optional
            if show_progress and skipped_deps_dict:
                try:
                    from ..cli.tui_manager import tui
                    skip_msg = (
                        f"[bold yellow]Dependencies Skipped[/bold yellow]\n\n"
                        f"The following packages are incompatible with your platform:\n\n"
                    )
                    for pkg, reason in skipped_deps_dict.items():
                        skip_msg += f"  • [cyan]{pkg}[/cyan]: {reason}\n"
                    skip_msg += "\n[green]The model will use fallback implementations.[/green]"
                    tui.show_panel(skip_msg, title="Platform Compatibility", border_style="yellow")
                    tui.prompt("Press Enter to continue...", style="dim")
                except ImportError:
                    pass
            return True, [], []

        installed = []
        failed = []

        logger.info(f"Installing {len(compatible_deps)} compatible dependencies: {compatible_deps}")

        # Show TUI panel if available
        if show_progress:
            try:
                from ..cli.tui_manager import tui
                panel_msg = (
                    f"[bold cyan]Installing Missing Dependencies[/bold cyan]\n\n"
                    f"The model requires the following packages:\n"
                    + "\n".join([f"  • {pkg}" for pkg in compatible_deps])
                )
                if skipped_deps_dict:
                    panel_msg += f"\n\n[yellow]Skipped (platform/hardware incompatible):[/yellow]\n"
                    for pkg, reason in skipped_deps_dict.items():
                        panel_msg += f"  • [dim]{pkg}[/dim]: {reason}\n"
                panel_msg += (
                    f"\n\nInstalling {len(compatible_deps)} package(s)...\n"
                    f"[dim]This may take a few moments[/dim]"
                )
                tui.show_panel(panel_msg, title="Dependency Installation", border_style="cyan")
            except ImportError:
                pass

        for i, package in enumerate(compatible_deps, 1):
            try:
                if show_progress:
                    try:
                        from ..cli.tui_manager import tui
                        tui.console.print(f"[{i}/{len(compatible_deps)}] Installing {package}...", style="cyan")
                    except ImportError:
                        logger.info(f"[{i}/{len(compatible_deps)}] Installing {package}...")
                else:
                    logger.info(f"Installing {package}...")

                # Try installing the package as-is first
                success, error_msg = self._try_install_package(package)
                
                if success:
                    installed.append(package)
                    if show_progress:
                        try:
                            from ..cli.tui_manager import tui
                            tui.console.print(f"  ✓ Successfully installed {package}", style="green")
                        except ImportError:
                            logger.info(f"✓ Successfully installed {package}")
                    else:
                        logger.info(f"✓ Successfully installed {package}")
                else:
                    # First attempt failed - try alternatives
                    logger.debug(f"Package '{package}' failed to install, trying alternatives...")
                    alternatives = self._generate_package_name_alternatives(package)
                    
                    package_installed = False
                    actual_package_name = None
                    
                    for alt_name in alternatives:
                        logger.debug(f"Trying alternative: {alt_name}")
                        alt_success, alt_error = self._try_install_package(alt_name)
                        
                        if alt_success:
                            # Alternative worked! Learn this mapping
                            installed.append(alt_name)
                            package_installed = True
                            actual_package_name = alt_name
                            self._learn_package_mapping(package, alt_name)
                            
                            if show_progress:
                                try:
                                    from ..cli.tui_manager import tui
                                    tui.console.print(
                                        f"  ✓ Installed {alt_name} (alternative for {package})", 
                                        style="green"
                                    )
                                except ImportError:
                                    logger.info(f"✓ Installed {alt_name} (alternative for {package})")
                            else:
                                logger.info(f"✓ Installed {alt_name} (alternative for {package})")
                            break
                    
                    if not package_installed:
                        # All attempts failed
                        failed.append(package)
                        if show_progress:
                            try:
                                from ..cli.tui_manager import tui
                                tui.console.print(f"  ✗ Failed to install {package}", style="red")
                            except ImportError:
                                logger.error(f"✗ Failed to install {package}: {error_msg}")
                        else:
                            logger.error(f"✗ Failed to install {package}: {error_msg}")

            except subprocess.TimeoutExpired:
                failed.append(package)
                logger.error(f"✗ Timeout installing {package}")
            except Exception as e:
                failed.append(package)
                logger.error(f"✗ Error installing {package}: {e}")

        success = len(failed) == 0

        # Show completion message
        if show_progress:
            try:
                from ..cli.tui_manager import tui
                if success:
                    tui.console.print(f"\n✓ All dependencies installed successfully!\n", style="bold green")
                else:
                    tui.console.print(f"\n⚠ Some dependencies failed to install: {failed}\n", style="yellow")
                tui.prompt("Press Enter to continue...", style="dim")
            except ImportError:
                pass

        return success, installed, failed

    def handle_model_load_error(
        self,
        model_id: str,
        error_message: str,
        auto_install: bool = True
    ) -> tuple[bool, list[str], dict[str, str]]:
        """Handle model loading error by detecting and installing dependencies.

        Args:
            model_id: HuggingFace model ID
            error_message: Error message from model loading
            auto_install: If True, automatically install dependencies

        Returns:
            Tuple of (should_retry, installed_packages, failure_info)
            - should_retry: Whether to retry loading the model
            - installed_packages: List of successfully installed packages
            - failure_info: Dict of {package: error_reason} for failed/skipped packages
        """
        # Detect missing dependencies
        missing_deps = self.detect_missing_dependencies(error_message)

        if not missing_deps:
            logger.debug("No missing dependencies detected in error message")
            logger.error(f"Model load error (no missing deps detected): {error_message}")
            return False, [], {}

        logger.warning(f"Model {model_id} requires missing dependencies: {missing_deps}")

        # Detect architecture
        architecture = self.detect_architecture(model_id, error_message)

        # Install dependencies
        if auto_install:
            success, installed, failed = self.install_dependencies(missing_deps, auto_approve=True)

            # Collect failure information (including skipped packages)
            failure_info = {}
            for pkg in failed:
                failure_info[pkg] = "Installation failed"

            # Get skipped packages info
            _, skipped_dict = self._filter_platform_compatible_deps(missing_deps)
            for pkg, reason in skipped_dict.items():
                if pkg not in installed:
                    failure_info[pkg] = f"Skipped: {reason}"

            # Register architecture with ALL dependencies for future use
            # (even failed ones, so we know what's needed)
            if architecture and (installed or failure_info):
                all_attempted = installed + list(failure_info.keys())
                self.register_architecture(architecture, all_attempted)

            if success and installed:
                logger.info(f"✓ All compatible dependencies installed successfully, retry loading model")
                return True, installed, failure_info
            elif installed and not failed:
                # Some were skipped but none failed - still worth retrying
                logger.info(f"✓ Installed {len(installed)} dependencies (some skipped as optional), retry loading model")
                return True, installed, failure_info
            else:
                # Some installations actually failed
                logger.error(f"✗ Failed to install some dependencies: {failed}")
                logger.error(f"Skipped dependencies: {list(skipped_dict.keys())}")
                return False, installed, failure_info
        else:
            logger.info(f"Auto-install disabled. Please install: pip install {' '.join(missing_deps)}")
            return False, [], {}

    def get_registry_info(self) -> dict[str, Any]:
        """Get registry information for debugging.

        Returns:
            Dictionary with registry metadata
        """
        return {
            "registry_path": str(self.registry_path),
            "registry_exists": self.registry_path.exists(),
            "architectures_count": len(self.architectures),
            "architectures": self.architectures,
            "package_mappings_count": len(self.package_mappings),
            "package_mappings": self.package_mappings,
        }


# Global registry instance (singleton)
_registry_instance: Optional[ArchitectureRegistry] = None


def get_architecture_registry() -> ArchitectureRegistry:
    """Get global architecture registry instance.

    Returns:
        ArchitectureRegistry singleton instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ArchitectureRegistry()
    return _registry_instance
