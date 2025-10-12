#!/usr/bin/env python3
"""
VLM CLI - Interactive testing tool for Vision Language Models
Professional workflow: Provider → Model → Endpoints → Inference
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' not installed. Install with: pip install rich")

from vlm_client import VLMClient


class VLM_CLI:
    """Interactive CLI for VLM testing with professional workflow"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.client = None
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)

        # Local model storage
        self.local_models_dir = Path("./local_models")
        self.local_models_dir.mkdir(exist_ok=True)

        # Current session state
        self.current_provider = None
        self.current_model_info = None
        self.system_specs = None

    def print(self, *args, **kwargs):
        """Print with rich if available, otherwise standard print"""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name != 'nt' else 'cls')

    def show_banner(self):
        """Display application banner"""
        if RICH_AVAILABLE:
            banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║           VLM CLI - Vision Language Model Tester         ║
    ║                   Production-Ready v2.0                  ║
    ╚══════════════════════════════════════════════════════════╝
            """
            self.console.print(banner, style="bold cyan")
        else:
            print("\n" + "="*60)
            print("     VLM CLI - Vision Language Model Tester v2.0")
            print("="*60 + "\n")

    async def initialize(self):
        """Initialize the VLM client and fetch system specs"""
        try:
            config_path = Path(__file__).parent / "config.yaml"

            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task("Initializing VLM Client...", total=None)
                    self.client = VLMClient(str(config_path))
                    progress.update(task, completed=True)

                    # Cleanup any loaded models to get accurate system specs
                    task2 = progress.add_task("Cleaning up memory...", total=None)
                    await self.client.cleanup_all_models()
                    progress.update(task2, completed=True)
            else:
                print("Initializing VLM Client...")
                self.client = VLMClient(str(config_path))
                print("Cleaning up memory...")
                await self.client.cleanup_all_models()

            # Fetch system specs (now with clean memory state)
            self.system_specs = self.client.get_system_specs()
            self._display_system_specs()

            return True

        except Exception as e:
            self.print(f"[bold red]Error initializing client: {e}[/bold red]")
            return False

    def _display_system_specs(self):
        """Display system specifications"""
        if not self.system_specs:
            return

        if RICH_AVAILABLE:
            specs_content = []

            # Platform
            specs_content.append(f"[bold cyan]Platform:[/bold cyan] {self.system_specs.get('platform', 'Unknown')} {self.system_specs.get('architecture', '')}")

            # CPU
            cpu_count = self.system_specs.get('cpu_count', 0)
            cpu_logical = self.system_specs.get('cpu_count_logical', 0)
            specs_content.append(f"[bold cyan]CPU:[/bold cyan] {cpu_count} cores ({cpu_logical} threads)")

            # RAM
            ram_total = self.system_specs.get('ram_total_gb', 0)
            ram_available = self.system_specs.get('ram_available_gb', 0)
            specs_content.append(f"[bold cyan]RAM:[/bold cyan] {ram_total:.1f}GB total, {ram_available:.1f}GB available")

            # GPU
            gpu_available = self.system_specs.get('gpu_available', False)
            if gpu_available:
                gpu_name = self.system_specs.get('gpu_name', 'Unknown')
                specs_content.append(f"[bold cyan]GPU:[/bold cyan] {gpu_name}")
            else:
                specs_content.append(f"[bold cyan]GPU:[/bold cyan] None (CPU only)")

            # Recommendation
            recommended = self.system_specs.get('recommended_model_size_gb', 0)
            specs_content.append(f"[bold green]Recommended Model Size:[/bold green] ≤ {recommended:.1f}GB")

            # Memory management note
            specs_content.append(f"\n[dim]Note: Models are unloaded when switching to free memory[/dim]")

            panel = Panel(
                "\n".join(specs_content),
                title="[bold white]System Specifications (Clean State)[/bold white]",
                border_style="cyan",
                box=box.ROUNDED
            )
            self.console.print("\n", panel)
        else:
            print("\n=== System Specifications ===")
            print(f"Platform: {self.system_specs.get('platform', 'Unknown')}")
            print(f"CPU: {self.system_specs.get('cpu_count', 0)} cores")
            print(f"RAM: {self.system_specs.get('ram_total_gb', 0):.1f}GB")
            print(f"GPU: {self.system_specs.get('gpu_name', 'None')}")
            print(f"Recommended Model Size: ≤ {self.system_specs.get('recommended_model_size_gb', 0):.1f}GB")
            print("="*30 + "\n")

    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size based on name (rough estimates)"""
        model_lower = model_name.lower()

        # Ollama models (from their size in name)
        if ':' in model_lower:
            # Extract size indicator
            if '0.5b' in model_lower or '500m' in model_lower:
                return 0.5
            elif '1b' in model_lower or '1.5b' in model_lower:
                return 1.5
            elif '2b' in model_lower or '2.5b' in model_lower or '2.7b' in model_lower:
                return 2.7
            elif '3b' in model_lower:
                return 3.2
            elif '7b' in model_lower:
                return 4.5
            elif '11b' in model_lower:
                return 7.0
            elif '13b' in model_lower:
                return 8.0
            elif '32b' in model_lower:
                return 20.0
            elif '70b' in model_lower:
                return 40.0

        # HuggingFace models (common ones)
        if 'florence-2-base' in model_lower:
            return 0.2
        elif 'florence-2-large' in model_lower:
            return 0.8
        elif 'blip2-opt-2.7b' in model_lower:
            return 2.7
        elif 'blip2-opt-6.7b' in model_lower:
            return 6.7
        elif 'llava' in model_lower and '7b' in model_lower:
            return 4.7
        elif 'llava' in model_lower and '13b' in model_lower:
            return 8.0
        elif 'moondream' in model_lower:
            return 1.7

        # Default: unable to estimate
        return 0.0

    async def select_provider(self) -> str:
        """Step 1: Select provider (Ollama or HuggingFace)"""
        self.print("\n[bold yellow]Step 1: Select Provider[/bold yellow]\n")

        if RICH_AVAILABLE:
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="cyan", justify="center", width=5)
            table.add_column("Provider", style="green", width=20)
            table.add_column("Description", style="white", width=40)

            table.add_row("1", "Ollama", "Local models with Ollama server")
            table.add_row("2", "HuggingFace", "Models from HuggingFace Hub")

            self.console.print(table)

            choice = Prompt.ask(
                "\n[bold cyan]Select provider[/bold cyan]",
                choices=["1", "2"],
                default="1"
            )
            provider = "ollama" if choice == "1" else "huggingface"
        else:
            print("1. Ollama - Local models")
            print("2. HuggingFace - Hub models")
            choice = input("\nSelect provider (1/2): ").strip()
            provider = "huggingface" if choice == "2" else "ollama"

        self.current_provider = provider
        return provider

    async def select_model(self, provider: str) -> Optional[Dict[str, Any]]:
        """Step 2: Select model from provider (with install option)"""
        self.print(f"\n[bold yellow]Step 2: Select Model ({provider})[/bold yellow]\n")

        # Get ALL installed models for this provider (dynamically discovered)
        try:
            if provider == "ollama":
                all_models = await self.client.discover_installed_ollama_models()
            else:
                all_models = await self.client.discover_installed_hf_models()
        except ConnectionError as e:
            # Ollama connection failed
            self.print(f"[bold red]Error:[/bold red] {e}")
            return None

        if not all_models:
            self.print(f"[yellow]No {provider} models found. Install one now?[/yellow]")
            return await self.install_model_flow(provider)

        # Display models with Type and Compatibility
        if RICH_AVAILABLE:
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="cyan", justify="center", width=5)
            table.add_column("Model", style="green", width=35)
            table.add_column("Type", style="magenta", justify="center", width=10)
            table.add_column("Size", style="yellow", width=10)
            table.add_column("Compat", style="white", justify="center", width=8)

            for idx, model in enumerate(all_models, 1):
                # Get model type with color
                model_type = model.get('type', 'LLM')
                if model_type == 'VLM':
                    type_display = "[bold green]VLM[/bold green]"
                elif model_type == 'Embedding':
                    type_display = "[bold blue]Embed[/bold blue]"
                else:
                    type_display = "[bold yellow]LLM[/bold yellow]"

                # Get compatibility
                compat = self.client.check_model_compatibility(model['size_gb'])
                compat_icon = compat['icon']

                table.add_row(
                    str(idx),
                    model['name'],
                    type_display,
                    f"{model['size_gb']:.1f}GB",
                    compat_icon
                )

            # Add "Install New" and "Delete" options
            table.add_row("", "", "", "", "")  # Separator
            table.add_row(
                str(len(all_models) + 1),
                "Install New Model",
                "[dim]-[/dim]",
                "-",
                "[blue]⬇[/blue]"
            )
            table.add_row(
                str(len(all_models) + 2),
                "Delete a Model",
                "[dim]-[/dim]",
                "-",
                "[red]🗑[/red]"
            )

            self.console.print(table)

            choice = Prompt.ask(
                "\n[bold cyan]Select model[/bold cyan]",
                choices=[str(i) for i in range(1, len(all_models) + 3)],
                default="1"
            )
        else:
            for idx, model in enumerate(all_models, 1):
                print(f"{idx}. {model['name']} ({model['type']}) - {model['size_gb']:.1f}GB")
            print(f"\n{len(all_models) + 1}. Install New Model")
            print(f"{len(all_models) + 2}. Delete a Model")

            choice = input(f"\nSelect model (1-{len(all_models) + 2}): ").strip()

        model_idx = int(choice) - 1

        # Check if "Install New" was selected
        if model_idx == len(all_models):
            return await self.install_model_flow(provider)

        # Check if "Delete" was selected
        if model_idx == len(all_models) + 1:
            await self.delete_model_flow(provider, all_models)
            # After deletion, show model list again
            return await self.select_model(provider)

        selected_model = all_models[model_idx]

        # Warn if it's not a VLM
        if selected_model.get('type') != 'VLM':
            self.print(f"\n[yellow]Note: {selected_model['name']} is a {selected_model.get('type', 'LLM')}, not a VLM.[/yellow]")
            self.print("[yellow]It cannot process images. Only text-based inference is supported.[/yellow]")

        # Unload any previously loaded model first
        if self.client.model_instance:
            old_model = self.client.current_model
            self.print(f"\n[dim]Unloading {old_model}...[/dim]")
            await self.client.unload_model()

            # Show memory freed (especially useful for HuggingFace models)
            if self.client.current_provider == 'huggingface':
                self.print("[dim]✓ Memory freed[/dim]")

        # Load the model
        self.print(f"[dim]Loading {selected_model['name']}...[/dim]")

        # Load discovered model
        if provider == 'ollama':
            success = await self.client._load_ollama_model(selected_model['name'])
            if success:
                self.client.current_model = selected_model['name']
                self.client.current_provider = provider
        else:
            success = await self.client._load_huggingface_model(selected_model['name'], {})
            if success:
                self.client.current_model = selected_model['name']
                self.client.current_provider = provider

        if not success:
            self.print("[bold red]✗ Failed to load model[/bold red]")
            return None

        self.print("[bold green]✓ Model loaded successfully![/bold green]")

        # Add model type and capabilities to model info
        selected_model['is_vlm'] = (selected_model.get('type') == 'VLM')
        selected_model['capabilities'] = ['qa', 'caption', 'detect', 'point'] if selected_model['is_vlm'] else []

        self.current_model_info = selected_model
        return selected_model

    async def delete_model_flow(self, provider: str, models: List[Dict[str, Any]]):
        """Flow for deleting a model"""
        self.print(f"\n[bold yellow]═══ Delete {provider.title()} Model ═══[/bold yellow]\n")

        if not models:
            self.print("[yellow]No models to delete![/yellow]")
            return

        # Show models for selection
        if RICH_AVAILABLE:
            table = Table(show_header=True, header_style="bold red", box=box.ROUNDED)
            table.add_column("#", style="cyan", justify="center", width=5)
            table.add_column("Model", style="white", width=40)
            table.add_column("Size", style="yellow", width=12)

            for idx, model in enumerate(models, 1):
                table.add_row(
                    str(idx),
                    model['name'],
                    f"{model['size_gb']:.1f}GB"
                )

            self.console.print(table)

            choice = Prompt.ask(
                "\n[bold red]Select model to delete (or 0 to cancel)[/bold red]",
                choices=["0"] + [str(i) for i in range(1, len(models) + 1)],
                default="0"
            )
        else:
            for idx, model in enumerate(models, 1):
                print(f"{idx}. {model['name']} ({model['size_gb']:.1f}GB)")
            choice = input("\nSelect model to delete (or 0 to cancel): ").strip()

        if choice == "0":
            self.print("[dim]Cancelled[/dim]")
            return

        model_idx = int(choice) - 1
        selected_model = models[model_idx]

        # Confirm deletion
        if RICH_AVAILABLE:
            confirm = Confirm.ask(
                f"\n[bold red]Delete {selected_model['name']} ({selected_model['size_gb']:.1f}GB)?[/bold red]",
                default=False
            )
        else:
            response = input(f"\nDelete {selected_model['name']}? (y/n): ").strip().lower()
            confirm = response == 'y'

        if not confirm:
            self.print("[dim]Cancelled[/dim]")
            return

        # Delete the model
        self.print(f"\n[dim]Deleting {selected_model['name']}...[/dim]")

        if provider == "ollama":
            success = await self.client.delete_ollama_model(selected_model['name'])
        else:
            success = await self.client.delete_hf_model(selected_model['name'])

        if success:
            self.print(f"[bold green]✓ Successfully deleted {selected_model['name']}[/bold green]")
        else:
            self.print(f"[bold red]✗ Failed to delete {selected_model['name']}[/bold red]")

    async def install_model_flow(self, provider: str) -> Optional[Dict[str, Any]]:
        """Flow for installing a new model"""
        self.print(f"\n[bold yellow]═══ Install New {provider.title()} Model ═══[/bold yellow]\n")

        # Show examples
        if provider == "ollama":
            self.print("[dim]Examples: llama3.2-vision:11b, qwen2-vl:7b, llava:13b, moondream:latest[/dim]")
        else:
            self.print("[dim]Examples: microsoft/Florence-2-base, Salesforce/blip2-opt-2.7b[/dim]")

        # Get model name
        if RICH_AVAILABLE:
            model_name = Prompt.ask(f"[cyan]Enter {provider} model name[/cyan]")
        else:
            model_name = input(f"Enter {provider} model name: ").strip()

        if not model_name:
            self.print("[bold red]Model name cannot be empty![/bold red]")
            return None

        # For known models, check compatibility before confirming
        # Estimate size based on model name (rough estimates)
        estimated_size = self._estimate_model_size(model_name)
        if estimated_size > 0:
            compat = self.client.check_model_compatibility(estimated_size)
            self.print(f"\n[dim]Estimated size: {estimated_size:.1f}GB[/dim]")
            self.print(f"[dim]Compatibility: {compat['icon']} {compat['message']}[/dim]")

            if not compat['compatible']:
                self.print(f"\n[bold red]⚠ Cannot install: {compat['message']}[/bold red]")
                for warning in compat['warnings']:
                    self.print(f"[red]  • {warning}[/red]")
                return None

            if compat['warnings']:
                self.print(f"[yellow]⚠ Warnings:[/yellow]")
                for warning in compat['warnings']:
                    self.print(f"[yellow]  • {warning}[/yellow]")

        # Confirm
        if RICH_AVAILABLE:
            confirm = Confirm.ask(f"\n[cyan]Download {model_name}?[/cyan]", default=True)
        else:
            response = input(f"\nDownload {model_name}? (y/n): ").strip().lower()
            confirm = response == 'y'

        if not confirm:
            return None

        # Install
        self.print(f"\n[bold cyan]Downloading {model_name}...[/bold cyan]")

        if provider == "ollama":
            success = await self.client.install_ollama_model(model_name)
        else:
            success = await self.client.install_hf_model(model_name)

        if not success:
            self.print("[bold red]✗ Installation failed[/bold red]")
            return None

        self.print("\n[bold green]✓ Model installed successfully![/bold green]")

        # Detect model type
        model_type = self.client.detect_model_type(model_name, provider)

        # Unload any previously loaded model first
        if self.client.model_instance:
            self.print(f"\n[dim]Unloading previous model...[/dim]")
            await self.client.unload_model()

        # Load immediately
        self.print(f"[dim]Loading {model_name}...[/dim]")
        if provider == "ollama":
            await self.client._load_ollama_model(model_name)
            self.client.current_model = model_name
            self.client.current_provider = provider
        else:
            await self.client._load_huggingface_model(model_name, {})
            self.client.current_model = model_name
            self.client.current_provider = provider

        is_vlm = (model_type == 'VLM')

        return {
            'name': model_name,
            'provider': provider,
            'type': model_type,
            'is_vlm': is_vlm,
            'capabilities': ['qa', 'caption', 'detect', 'point'] if is_vlm else [],
            'size_gb': estimated_size if estimated_size > 0 else 0.0
        }

    def show_endpoints(self, model_info: Dict[str, Any]) -> str:
        """Step 3: Show available endpoints for the model"""
        self.print("\n[bold yellow]Available Endpoints:[/bold yellow]\n")

        capabilities = model_info.get('capabilities', ['qa', 'caption'])

        if RICH_AVAILABLE:
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="cyan", justify="center", width=5)
            table.add_column("Endpoint", style="green", width=20)
            table.add_column("Description", style="white", width=35)

            options = []
            if 'qa' in capabilities:
                table.add_row("1", "QA", "Question Answering about images")
                options.append("1")

            if 'caption' in capabilities:
                table.add_row("2", "Caption", "Generate image descriptions")
                options.append("2")

            if 'detect' in capabilities:
                table.add_row("3", "Detect", "Detect objects in images")
                options.append("3")

            if 'point' in capabilities:
                table.add_row("4", "Point", "Locate objects (coordinates)")
                options.append("4")

            table.add_row("8", "Switch Model", "Change to different model")
            table.add_row("9", "Stats", "View statistics")
            table.add_row("0", "Exit", "Quit application")

            options.extend(["8", "9", "0"])

            self.console.print(table)

            choice = Prompt.ask(
                "\n[bold cyan]Select option[/bold cyan]",
                choices=options,
                default="1" if "1" in options else options[0]
            )
        else:
            print("Available Options:")
            options = []
            if 'qa' in capabilities:
                print("1. QA - Question Answering")
                options.append("1")
            if 'caption' in capabilities:
                print("2. Caption - Image Descriptions")
                options.append("2")
            if 'detect' in capabilities:
                print("3. Detect - Object Detection")
                options.append("3")
            if 'point' in capabilities:
                print("4. Point - Object Localization")
                options.append("4")

            print("8. Switch Model")
            print("9. Stats")
            print("0. Exit")

            choice = input("\nSelect option: ").strip()

        return choice

    async def run_endpoint(self, endpoint: str, is_vlm: bool = True):
        """Step 4: Run the selected endpoint"""

        # Check if model is VLM
        if not is_vlm:
            self.print("\n[bold red]Error: This model is not a VLM (Vision Language Model)[/bold red]")
            self.print("[yellow]LLM and Embedding models cannot process images.[/yellow]")
            self.print("[dim]Please select a VLM model with visual capabilities.[/dim]")
            return

        # Get image path for VLM models
        image_path = None
        if RICH_AVAILABLE:
            image_path = Prompt.ask("\n[cyan]Image path[/cyan]")
        else:
            image_path = input("\nImage path: ").strip()

        if not Path(image_path).exists():
            self.print("[bold red]Error: Image not found![/bold red]")
            return

        # Get prompt/question based on endpoint
        if endpoint == "qa":
            self.print("\n[bold yellow]═══ Question Answering ═══[/bold yellow]")
            if RICH_AVAILABLE:
                question = Prompt.ask("\n[cyan]Question[/cyan]")
            else:
                question = input("\nQuestion: ").strip()

            self.print("\n[dim]Processing...[/dim]")
            result = await self.client.qa(image_path, question)
            self.display_result(result, "QA Result")

        elif endpoint == "caption":
            self.print("\n[bold yellow]═══ Image Captioning ═══[/bold yellow]")
            if RICH_AVAILABLE:
                detail = Prompt.ask(
                    "\n[cyan]Detail level[/cyan]",
                    choices=["detailed", "short"],
                    default="detailed"
                )
            else:
                detail = input("\nDetail level (detailed/short): ").strip() or "detailed"

            self.print("\n[dim]Generating caption...[/dim]")
            result = await self.client.caption(image_path, detail_level=detail)
            self.display_result(result, "Caption Result")

        elif endpoint == "detect":
            self.print("\n[bold yellow]═══ Object Detection ═══[/bold yellow]")
            if RICH_AVAILABLE:
                object_name = Prompt.ask("\n[cyan]Object to detect[/cyan]")
            else:
                object_name = input("\nObject to detect: ").strip()

            self.print("\n[dim]Detecting...[/dim]")
            result = await self.client.detect(image_path, object_name)
            self.display_result(result, "Detection Result")

        elif endpoint == "point":
            self.print("\n[bold yellow]═══ Object Pointing ═══[/bold yellow]")
            if RICH_AVAILABLE:
                object_name = Prompt.ask("\n[cyan]Object to locate[/cyan]")
            else:
                object_name = input("\nObject to locate: ").strip()

            self.print("\n[dim]Locating...[/dim]")
            result = await self.client.point(image_path, object_name)
            self.display_result(result, "Point Result")

        # Ask to save
        if RICH_AVAILABLE:
            save = Confirm.ask("\n[cyan]Save result to JSON?[/cyan]", default=False)
        else:
            save = input("\nSave result to JSON? (y/n): ").strip().lower() == 'y'

        if save:
            self.save_result(result, endpoint)

    def display_result(self, result: Dict[str, Any], title: str):
        """Display inference result"""
        if "error" in result:
            self.print(f"\n[bold red]Error: {result['error']}[/bold red]")
            return

        if RICH_AVAILABLE:
            content = []

            # Main response
            if "answer" in result:
                content.append(f"[bold green]Answer:[/bold green] {result['answer']}")
            elif "caption" in result:
                content.append(f"[bold green]Caption:[/bold green] {result['caption']}")
            elif "detections" in result:
                content.append(f"[bold green]Detections:[/bold green]\n{result['detections']}")
            elif "location" in result:
                content.append(f"[bold green]Location:[/bold green] {result['location']}")
            elif "coordinates" in result:
                content.append(f"[bold green]Coordinates:[/bold green] {result['coordinates']}")

            # Timing
            if "inference_time_ms" in result:
                content.append(f"\n[dim]⏱  Time: {result['inference_time_ms']}ms[/dim]")

            # Model info
            if "model" in result:
                content.append(f"[dim]🤖 Model: {result['model']} ({result.get('provider', 'unknown')})[/dim]")

            panel = Panel(
                "\n".join(content),
                title=f"[bold cyan]{title}[/bold cyan]",
                border_style="green",
                box=box.ROUNDED
            )
            self.console.print("\n", panel)
        else:
            print(f"\n--- {title} ---")
            print(json.dumps(result, indent=2))

    def save_result(self, result: Dict[str, Any], endpoint: str):
        """Save result to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{endpoint}_{timestamp}.json"
            filepath = self.results_dir / filename

            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)

            self.print(f"[bold green]✓ Saved to {filepath}[/bold green]")

        except Exception as e:
            self.print(f"[bold red]Error saving: {e}[/bold red]")

    def show_stats(self):
        """Display statistics"""
        stats = self.client.get_stats()

        if RICH_AVAILABLE:
            table = Table(title="Statistics", show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("Metric", style="cyan", width=25)
            table.add_column("Value", style="green", width=30)

            table.add_row("Current Model", str(stats.get('current_model', 'None')))
            table.add_row("Provider", str(stats.get('current_provider', 'None')))
            table.add_row("Total Inferences", str(stats.get('total_inferences', 0)))
            table.add_row("Total Time", f"{stats.get('total_time_seconds', 0)}s")
            table.add_row("Average Time", f"{stats.get('average_time_ms', 0)}ms")

            self.console.print("\n", table)
        else:
            print("\n--- Statistics ---")
            print(json.dumps(stats, indent=2))

    async def run(self):
        """Main application loop with new workflow"""
        self.clear_screen()
        self.show_banner()

        # Initialize
        if not await self.initialize():
            return

        # Step 1: Select Provider
        provider = await self.select_provider()

        # Step 2: Select Model
        model_info = await self.select_model(provider)
        if not model_info:
            self.print("[bold red]No model selected. Exiting.[/bold red]")
            return

        # Get is_vlm from model_info
        is_vlm = model_info.get('is_vlm', False)

        # Main loop: Step 3 & 4
        while True:
            try:
                # Step 3: Show endpoints
                choice = self.show_endpoints(model_info)

                if choice == "0":
                    # Cleanup before exit
                    if self.client.model_instance:
                        self.print("\n[dim]Unloading model...[/dim]")
                        await self.client.unload_model()
                    self.print("[bold cyan]Thanks for using VLM CLI! Goodbye! 👋[/bold cyan]\n")
                    break

                elif choice == "8":
                    # Switch model - unload current model first
                    if self.client.model_instance:
                        self.print("\n[dim]Unloading current model...[/dim]")
                        await self.client.unload_model()

                    provider = await self.select_provider()
                    model_info = await self.select_model(provider)
                    if not model_info:
                        break
                    # Update is_vlm flag
                    is_vlm = model_info.get('is_vlm', False)

                elif choice == "9":
                    # Stats
                    self.show_stats()

                elif choice in ["1", "2", "3", "4"]:
                    # Run endpoint
                    endpoint_map = {"1": "qa", "2": "caption", "3": "detect", "4": "point"}
                    await self.run_endpoint(endpoint_map[choice], is_vlm)

                else:
                    self.print("[bold red]Invalid option![/bold red]")

                # Wait before continuing
                if RICH_AVAILABLE:
                    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
                else:
                    input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                self.print("\n\n[bold yellow]Interrupted[/bold yellow]")
                if RICH_AVAILABLE:
                    if Confirm.ask("[cyan]Exit?[/cyan]", default=True):
                        break
                else:
                    if input("Exit? (y/n): ").strip().lower() == 'y':
                        break

            except Exception as e:
                self.print(f"\n[bold red]Error: {e}[/bold red]")
                if RICH_AVAILABLE:
                    Prompt.ask("[dim]Press Enter to continue[/dim]", default="")
                else:
                    input("Press Enter to continue...")


def main():
    """Entry point"""
    cli = VLM_CLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
