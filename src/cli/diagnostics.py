"""System diagnostics and capability reporting for production transparency."""

import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from loguru import logger

from ..quantization.techniques.platform_detector import PlatformDetector, DeviceCapabilities


class SystemDiagnostics:
    """Production-grade system diagnostics and reporting."""

    def __init__(self):
        self.console = Console()
        self.capabilities: Optional[DeviceCapabilities] = None

    def run_full_diagnostics(self, output_format: str = "rich") -> DeviceCapabilities:
        """Run complete system diagnostics.

        Args:
            output_format: Output format - "rich", "json", or "text"

        Returns:
            DeviceCapabilities with all detected information
        """
        logger.info("Running system diagnostics...")
        
        # Detect all capabilities
        self.capabilities = PlatformDetector.detect_all()

        # Display based on format
        if output_format == "json":
            self._display_json()
        elif output_format == "text":
            self._display_text()
        else:
            self._display_rich()

        return self.capabilities

    def _display_rich(self):
        """Display diagnostics with rich formatting."""
        if not self.capabilities:
            return

        caps = self.capabilities
        
        # Title
        self.console.print("\n")
        self.console.print(Panel.fit(
            "[bold cyan]System Diagnostics & Capability Report[/bold cyan]",
            border_style="cyan"
        ))

        # Platform Information
        platform_table = Table(title="Platform Information", show_header=True, header_style="bold magenta")
        platform_table.add_column("Component", style="cyan")
        platform_table.add_column("Details", style="white")

        platform_table.add_row("Operating System", caps.platform.value)
        platform_table.add_row("CPU Architecture", caps.cpu_arch)
        platform_table.add_row("CPU Cores", str(caps.cpu_cores))
        platform_table.add_row("Total RAM", f"{caps.total_ram_gb:.1f} GB")
        platform_table.add_row("Available RAM", f"{caps.available_ram_gb:.1f} GB")
        platform_table.add_row("Python Version", caps.python_version)

        if caps.is_edge_device:
            platform_table.add_row("Device Type", "[yellow]Edge Device[/yellow]")
        if caps.is_raspberry_pi:
            platform_table.add_row("Platform", "[yellow]Raspberry Pi[/yellow]")
        if caps.is_jetson:
            platform_table.add_row("Platform", "[yellow]NVIDIA Jetson[/yellow]")

        self.console.print(platform_table)

        # GPU Information
        if caps.has_gpu:
            gpu_table = Table(title="GPU Information", show_header=True, header_style="bold green")
            gpu_table.add_column("Component", style="cyan")
            gpu_table.add_column("Details", style="white")

            gpu_table.add_row("Device Type", caps.device_type.value.upper())
            gpu_table.add_row("GPU Name", caps.gpu_name or "Unknown")
            if caps.gpu_memory_gb:
                gpu_table.add_row("GPU Memory", f"{caps.gpu_memory_gb:.1f} GB")
            
            if caps.supports_cuda:
                gpu_table.add_row("CUDA", f"[green]✓ Available[/green] ({caps.cuda_version or 'version unknown'})")
            if caps.supports_metal:
                gpu_table.add_row("Metal (MPS)", "[green]✓ Available[/green]")
            if caps.supports_rocm:
                gpu_table.add_row("ROCm", f"[green]✓ Available[/green] ({caps.rocm_version or 'version unknown'})")

            self.console.print(gpu_table)
        else:
            self.console.print(Panel(
                "[yellow]No GPU detected - CPU-only mode[/yellow]",
                title="GPU Information",
                border_style="yellow"
            ))

        # Software Versions
        version_table = Table(title="Software Versions", show_header=True, header_style="bold blue")
        version_table.add_column("Package", style="cyan")
        version_table.add_column("Version", style="white")
        version_table.add_column("Status", style="white")

        version_table.add_row("PyTorch", caps.torch_version or "Not installed", 
                             "[green]✓[/green]" if caps.torch_version else "[red]✗[/red]")
        version_table.add_row("Transformers", caps.transformers_version or "Not installed",
                             "[green]✓[/green]" if caps.transformers_version else "[red]✗[/red]")
        version_table.add_row("BitsAndBytes", caps.bitsandbytes_version or "Not installed",
                             "[green]✓[/green]" if caps.bitsandbytes_version else "[yellow]○[/yellow]")
        version_table.add_row("llama-cpp-python", caps.llama_cpp_version or "Not installed",
                             "[green]✓[/green]" if caps.llama_cpp_version else "[yellow]○[/yellow]")
        version_table.add_row("AutoAWQ", caps.awq_version or "Not installed",
                             "[green]✓[/green]" if caps.awq_version else "[yellow]○[/yellow]")

        self.console.print(version_table)

        # Quantization Capabilities
        quant_table = Table(title="Quantization Capabilities", show_header=True, header_style="bold yellow")
        quant_table.add_column("Method", style="cyan")
        quant_table.add_column("Supported", style="white")
        quant_table.add_column("Notes", style="white")

        quant_table.add_row("FP16 (Generic)", 
                           "[green]✓ Yes[/green]",
                           "Works on all platforms")
        
        quant_table.add_row("INT8 (BitsAndBytes)",
                           "[green]✓ Yes[/green]" if caps.can_quantize_int8 else "[red]✗ No[/red]",
                           "Requires CUDA GPU" if not caps.can_quantize_int8 else "CUDA available")
        
        quant_table.add_row("INT4 (BitsAndBytes)",
                           "[green]✓ Yes[/green]" if caps.can_quantize_int4 else "[red]✗ No[/red]",
                           "Requires CUDA GPU" if not caps.can_quantize_int4 else "CUDA available")
        
        quant_table.add_row("GGUF (llama.cpp)",
                           "[green]✓ Yes[/green]" if caps.can_quantize_gguf else "[red]✗ No[/red]",
                           "Install llama.cpp" if not caps.can_quantize_gguf else "llama.cpp tools found")
        
        quant_table.add_row("AWQ",
                           "[green]✓ Yes[/green]" if caps.can_quantize_awq else "[red]✗ No[/red]",
                           "Requires CUDA + AutoAWQ" if not caps.can_quantize_awq else "Available")
        
        quant_table.add_row("GPTQ",
                           "[green]✓ Yes[/green]" if caps.can_quantize_gptq else "[red]✗ No[/red]",
                           "Requires CUDA + Optimum" if not caps.can_quantize_gptq else "Available")

        self.console.print(quant_table)

        # Performance Recommendations
        perf_table = Table(title="Recommended Settings", show_header=True, header_style="bold green")
        perf_table.add_column("Parameter", style="cyan")
        perf_table.add_column("Recommended Value", style="white")

        perf_table.add_row("Batch Size", str(caps.optimal_batch_size))
        perf_table.add_row("Context Length", f"{caps.recommended_context_length} tokens")

        self.console.print(perf_table)

        # Warnings
        if caps.warnings:
            warning_text = Text()
            for warning in caps.warnings:
                warning_text.append(f"⚠️  {warning}\n", style="yellow")
            
            self.console.print(Panel(
                warning_text,
                title="[yellow]Warnings[/yellow]",
                border_style="yellow"
            ))

        # Recommendations
        if caps.recommendations:
            rec_text = Text()
            for rec in caps.recommendations:
                rec_text.append(f"💡 {rec}\n", style="cyan")
            
            self.console.print(Panel(
                rec_text,
                title="[cyan]Recommendations[/cyan]",
                border_style="cyan"
            ))

        # Summary
        summary = self._generate_summary()
        self.console.print(Panel(
            summary,
            title="[bold green]Summary[/bold green]",
            border_style="green"
        ))

        self.console.print()

    def _display_json(self):
        """Display diagnostics as JSON."""
        if not self.capabilities:
            return

        caps = self.capabilities
        
        data = {
            "platform": {
                "os": caps.platform.value,
                "cpu_arch": caps.cpu_arch,
                "cpu_cores": caps.cpu_cores,
                "total_ram_gb": caps.total_ram_gb,
                "available_ram_gb": caps.available_ram_gb,
                "is_edge_device": caps.is_edge_device,
            },
            "gpu": {
                "has_gpu": caps.has_gpu,
                "device_type": caps.device_type.value,
                "gpu_name": caps.gpu_name,
                "gpu_memory_gb": caps.gpu_memory_gb,
                "supports_cuda": caps.supports_cuda,
                "supports_metal": caps.supports_metal,
                "supports_rocm": caps.supports_rocm,
            },
            "versions": {
                "python": caps.python_version,
                "torch": caps.torch_version,
                "transformers": caps.transformers_version,
                "cuda": caps.cuda_version,
                "rocm": caps.rocm_version,
                "bitsandbytes": caps.bitsandbytes_version,
                "llama_cpp": caps.llama_cpp_version,
                "awq": caps.awq_version,
            },
            "quantization_capabilities": {
                "fp16": caps.can_quantize_fp16,
                "int8": caps.can_quantize_int8,
                "int4": caps.can_quantize_int4,
                "gguf": caps.can_quantize_gguf,
                "awq": caps.can_quantize_awq,
                "gptq": caps.can_quantize_gptq,
                "bitsandbytes": caps.can_quantize_bnb,
            },
            "recommended_settings": {
                "batch_size": caps.optimal_batch_size,
                "context_length": caps.recommended_context_length,
            },
            "warnings": caps.warnings,
            "recommendations": caps.recommendations,
            "backends": caps.backends,
        }

        print(json.dumps(data, indent=2))

    def _display_text(self):
        """Display diagnostics as plain text."""
        if not self.capabilities:
            return

        caps = self.capabilities
        
        print("\n" + "="*60)
        print("SYSTEM DIAGNOSTICS & CAPABILITY REPORT")
        print("="*60)

        print("\nPLATFORM INFORMATION:")
        print(f"  OS: {caps.platform.value}")
        print(f"  CPU: {caps.cpu_arch} ({caps.cpu_cores} cores)")
        print(f"  RAM: {caps.total_ram_gb:.1f}GB total, {caps.available_ram_gb:.1f}GB available")
        print(f"  Python: {caps.python_version}")

        if caps.has_gpu:
            print("\nGPU INFORMATION:")
            print(f"  Type: {caps.device_type.value.upper()}")
            print(f"  Name: {caps.gpu_name}")
            if caps.gpu_memory_gb:
                print(f"  Memory: {caps.gpu_memory_gb:.1f}GB")
            if caps.cuda_version:
                print(f"  CUDA: {caps.cuda_version}")

        print("\nSOFTWARE VERSIONS:")
        print(f"  PyTorch: {caps.torch_version or 'Not installed'}")
        print(f"  Transformers: {caps.transformers_version or 'Not installed'}")
        print(f"  BitsAndBytes: {caps.bitsandbytes_version or 'Not installed'}")
        print(f"  llama-cpp-python: {caps.llama_cpp_version or 'Not installed'}")

        print("\nQUANTIZATION CAPABILITIES:")
        print(f"  FP16: {'✓' if caps.can_quantize_fp16 else '✗'}")
        print(f"  INT8: {'✓' if caps.can_quantize_int8 else '✗'}")
        print(f"  INT4: {'✓' if caps.can_quantize_int4 else '✗'}")
        print(f"  GGUF: {'✓' if caps.can_quantize_gguf else '✗'}")
        print(f"  AWQ: {'✓' if caps.can_quantize_awq else '✗'}")

        if caps.warnings:
            print("\nWARNINGS:")
            for warning in caps.warnings:
                print(f"  ⚠️  {warning}")

        if caps.recommendations:
            print("\nRECOMMENDATIONS:")
            for rec in caps.recommendations:
                print(f"  💡 {rec}")

        print("\n" + "="*60 + "\n")

    def _generate_summary(self) -> str:
        """Generate a summary of system capabilities."""
        if not self.capabilities:
            return ""

        caps = self.capabilities
        
        summary_parts = []
        
        # Platform
        summary_parts.append(f"Platform: {caps.platform.value}")
        
        # GPU status
        if caps.has_gpu:
            summary_parts.append(f"GPU: {caps.device_type.value.upper()} ({caps.gpu_name})")
        else:
            summary_parts.append("GPU: None (CPU-only)")
        
        # Available quantization methods
        available_methods = []
        if caps.can_quantize_fp16:
            available_methods.append("FP16")
        if caps.can_quantize_int8:
            available_methods.append("INT8")
        if caps.can_quantize_int4:
            available_methods.append("INT4")
        if caps.can_quantize_gguf:
            available_methods.append("GGUF")
        if caps.can_quantize_awq:
            available_methods.append("AWQ")
        
        if available_methods:
            summary_parts.append(f"Quantization: {', '.join(available_methods)}")
        else:
            summary_parts.append("Quantization: Limited (FP16 only)")
        
        # Status
        if caps.warnings:
            summary_parts.append(f"Status: ⚠️  {len(caps.warnings)} warning(s)")
        else:
            summary_parts.append("Status: ✓ All systems operational")

        return "\n".join(summary_parts)

    def export_to_file(self, output_path: Path, format: str = "json"):
        """Export diagnostics to file.

        Args:
            output_path: Path to output file
            format: Export format - "json" or "text"
        """
        if not self.capabilities:
            self.run_full_diagnostics(output_format="text")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                caps = self.capabilities
                data = {
                    "platform": caps.platform.value,
                    "device_type": caps.device_type.value,
                    "cpu_arch": caps.cpu_arch,
                    "cpu_cores": caps.cpu_cores,
                    "total_ram_gb": caps.total_ram_gb,
                    "available_ram_gb": caps.available_ram_gb,
                    "has_gpu": caps.has_gpu,
                    "gpu_name": caps.gpu_name,
                    "gpu_memory_gb": caps.gpu_memory_gb,
                    "python_version": caps.python_version,
                    "torch_version": caps.torch_version,
                    "transformers_version": caps.transformers_version,
                    "cuda_version": caps.cuda_version,
                    "backends": caps.backends,
                    "quantization": {
                        "fp16": caps.can_quantize_fp16,
                        "int8": caps.can_quantize_int8,
                        "int4": caps.can_quantize_int4,
                        "gguf": caps.can_quantize_gguf,
                        "awq": caps.can_quantize_awq,
                    },
                    "warnings": caps.warnings,
                    "recommendations": caps.recommendations,
                }
                json.dump(data, f, indent=2)
        else:
            with open(output_path, "w") as f:
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = f
                self._display_text()
                sys.stdout = old_stdout

        logger.info(f"Diagnostics exported to {output_path}")


def run_diagnostics_cli(output_format: str = "rich", export_path: Optional[str] = None):
    """CLI entry point for diagnostics.

    Args:
        output_format: Display format - "rich", "json", or "text"
        export_path: Optional path to export diagnostics
    """
    diag = SystemDiagnostics()
    diag.run_full_diagnostics(output_format=output_format)

    if export_path:
        export_format = "json" if export_path.endswith(".json") else "text"
        diag.export_to_file(Path(export_path), format=export_format)
