#!/usr/bin/env python3
"""
VLM CLI - Interactive testing tool for Vision Language Models
Production-ready CLI with support for QA, Caption, Detect, and Point endpoints
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' not installed. Install with: pip install rich")
    print("Falling back to basic output...")

from vlm_client import VLMClient


class VLM_CLI:
    """Interactive CLI for VLM testing"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.client = None
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)

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
    ║                   Production-Ready v1.0                  ║
    ╚══════════════════════════════════════════════════════════╝
            """
            self.console.print(banner, style="bold cyan")
        else:
            print("\n" + "="*60)
            print("     VLM CLI - Vision Language Model Tester v1.0")
            print("="*60 + "\n")

    def show_main_menu(self) -> str:
        """Display main menu and get user choice"""
        if RICH_AVAILABLE:
            self.console.print("\n[bold yellow]Main Menu[/bold yellow]", justify="center")

            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("Option", style="cyan", justify="center", width=10)
            table.add_column("Endpoint", style="green", width=20)
            table.add_column("Description", style="white", width=40)

            table.add_row("1", "QA", "Question Answering - Ask questions about images")
            table.add_row("2", "Caption", "Image Captioning - Generate descriptions")
            table.add_row("3", "Detect", "Object Detection - Locate objects in images")
            table.add_row("4", "Point", "Object Pointing - Get object coordinates")
            table.add_row("5", "Stats", "View statistics and current model info")
            table.add_row("6", "Switch Model", "Change the current VLM model")
            table.add_row("7", "Install Model", "Download and install a new model")
            table.add_row("0", "Exit", "Quit the application")

            self.console.print(table)

            choice = Prompt.ask(
                "\n[bold cyan]Select an option[/bold cyan]",
                choices=["0", "1", "2", "3", "4", "5", "6", "7"],
                default="1"
            )
        else:
            print("\n--- Main Menu ---")
            print("1. QA - Question Answering")
            print("2. Caption - Image Captioning")
            print("3. Detect - Object Detection")
            print("4. Point - Object Pointing")
            print("5. Stats - View statistics")
            print("6. Switch Model - Change model")
            print("7. Install Model - Download new model")
            print("0. Exit")
            choice = input("\nSelect option (0-7): ").strip()

        return choice

    async def initialize(self):
        """Initialize the VLM client"""
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
            else:
                print("Initializing VLM Client...")
                self.client = VLMClient(str(config_path))

            return True

        except Exception as e:
            self.print(f"[bold red]Error initializing client: {e}[/bold red]")
            return False

    async def select_and_load_model(self):
        """Show model selection menu and load chosen model"""
        # Get configured models
        configured_models = self.client.get_available_models()

        # Discover installed models
        self.print("\n[dim]Scanning for installed models...[/dim]")
        ollama_installed = await self.client.discover_installed_ollama_models()
        hf_installed = await self.client.discover_installed_hf_models()

        # Combine all models
        all_models = []

        # Add configured models first
        for model in configured_models:
            all_models.append({
                'display_id': model['id'],
                'name': model['name'],
                'provider': model['provider'],
                'size_gb': model['size_gb'],
                'capabilities': model.get('capabilities', []),
                'installed': True,  # Assume configured models are available
                'type': 'configured'
            })

        # Add discovered Ollama models not in config
        for model in ollama_installed:
            # Check if not already in configured
            if not any(m['name'] == model['name'] for m in all_models):
                all_models.append({
                    'display_id': f"ollama-{model['name']}",
                    'name': model['name'],
                    'provider': 'ollama',
                    'size_gb': model['size_gb'],
                    'capabilities': ['qa', 'caption'],  # Default capabilities
                    'installed': True,
                    'type': 'discovered'
                })

        if not all_models:
            self.print("[bold red]No models found![/bold red]")
            return False

        self.print("\n[bold yellow]Available Models:[/bold yellow]\n")

        if RICH_AVAILABLE:
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("#", style="cyan", justify="center", width=5)
            table.add_column("Model", style="green", width=25)
            table.add_column("Provider", style="blue", width=12)
            table.add_column("Size", style="yellow", width=10)
            table.add_column("Status", style="white", width=12)

            for idx, model in enumerate(all_models, 1):
                status = "[green]✓ Installed[/green]" if model['installed'] else "[yellow]Available[/yellow]"
                table.add_row(
                    str(idx),
                    model['name'],
                    model['provider'],
                    f"{model['size_gb']:.1f}GB",
                    status
                )

            self.console.print(table)

            choice = Prompt.ask(
                "\n[bold cyan]Select model number[/bold cyan]",
                choices=[str(i) for i in range(1, len(all_models) + 1)],
                default="1"
            )
            model_idx = int(choice) - 1
        else:
            for idx, model in enumerate(all_models, 1):
                status = "✓ Installed" if model['installed'] else "Available"
                print(f"{idx}. {model['name']} ({model['provider']}) - {model['size_gb']:.1f}GB [{status}]")

            choice = input(f"\nSelect model (1-{len(all_models)}): ").strip()
            model_idx = int(choice) - 1 if choice.isdigit() else 0

        selected_model = all_models[model_idx]

        # Load model - need to handle both configured and discovered
        if selected_model['type'] == 'configured':
            model_id = selected_model['display_id']
        else:
            # For discovered models, load directly
            model_id = selected_model['name']

        # Load model
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Loading {selected_model['name']}...", total=None)

                if selected_model['type'] == 'configured':
                    success = await self.client.load_model(model_id)
                else:
                    # Load discovered model directly
                    if selected_model['provider'] == 'ollama':
                        success = await self.client._load_ollama_model(selected_model['name'])
                        if success:
                            self.client.current_model = model_id
                            self.client.current_provider = 'ollama'
                    else:
                        self.print("[yellow]Discovered HF models need to be configured first[/yellow]")
                        success = False

                progress.update(task, completed=True)
        else:
            print(f"\nLoading {selected_model['name']}...")
            if selected_model['type'] == 'configured':
                success = await self.client.load_model(model_id)
            else:
                if selected_model['provider'] == 'ollama':
                    success = await self.client._load_ollama_model(selected_model['name'])
                    if success:
                        self.client.current_model = model_id
                        self.client.current_provider = 'ollama'
                else:
                    print("Discovered HF models need to be configured first")
                    success = False

        if success:
            self.print(f"\n[bold green]✓ Model loaded successfully![/bold green]")
        else:
            self.print(f"\n[bold red]✗ Failed to load model[/bold red]")

        return success

    async def run_qa(self):
        """Run QA endpoint"""
        self.print("\n[bold yellow]═══ Question Answering ═══[/bold yellow]\n")

        # Get image path
        if RICH_AVAILABLE:
            image_path = Prompt.ask("[cyan]Image path[/cyan]")
        else:
            image_path = input("Image path: ").strip()

        if not Path(image_path).exists():
            self.print("[bold red]Error: Image not found![/bold red]")
            return

        # Get question
        if RICH_AVAILABLE:
            question = Prompt.ask("[cyan]Question[/cyan]")
        else:
            question = input("Question: ").strip()

        # Run inference
        self.print("\n[dim]Running inference...[/dim]")
        result = await self.client.qa(image_path, question)

        # Display result
        self.display_result(result, "QA Result")

        # Save option
        if self._ask_save():
            self.save_result(result, "qa")

    async def run_caption(self):
        """Run Caption endpoint"""
        self.print("\n[bold yellow]═══ Image Captioning ═══[/bold yellow]\n")

        # Get image path
        if RICH_AVAILABLE:
            image_path = Prompt.ask("[cyan]Image path[/cyan]")
        else:
            image_path = input("Image path: ").strip()

        if not Path(image_path).exists():
            self.print("[bold red]Error: Image not found![/bold red]")
            return

        # Get detail level
        if RICH_AVAILABLE:
            detail = Prompt.ask(
                "[cyan]Detail level[/cyan]",
                choices=["detailed", "short"],
                default="detailed"
            )
        else:
            print("Detail level: 1) detailed  2) short")
            choice = input("Choose (1/2): ").strip()
            detail = "short" if choice == "2" else "detailed"

        # Run inference
        self.print("\n[dim]Generating caption...[/dim]")
        result = await self.client.caption(image_path, detail_level=detail)

        # Display result
        self.display_result(result, "Caption Result")

        # Save option
        if self._ask_save():
            self.save_result(result, "caption")

    async def run_detect(self):
        """Run Detect endpoint"""
        self.print("\n[bold yellow]═══ Object Detection ═══[/bold yellow]\n")

        # Get image path
        if RICH_AVAILABLE:
            image_path = Prompt.ask("[cyan]Image path[/cyan]")
        else:
            image_path = input("Image path: ").strip()

        if not Path(image_path).exists():
            self.print("[bold red]Error: Image not found![/bold red]")
            return

        # Get object name
        if RICH_AVAILABLE:
            object_name = Prompt.ask("[cyan]Object to detect[/cyan]")
        else:
            object_name = input("Object to detect: ").strip()

        # Run inference
        self.print("\n[dim]Detecting objects...[/dim]")
        result = await self.client.detect(image_path, object_name)

        # Display result
        self.display_result(result, "Detection Result")

        # Save option
        if self._ask_save():
            self.save_result(result, "detect")

    async def run_point(self):
        """Run Point endpoint"""
        self.print("\n[bold yellow]═══ Object Pointing ═══[/bold yellow]\n")

        # Get image path
        if RICH_AVAILABLE:
            image_path = Prompt.ask("[cyan]Image path[/cyan]")
        else:
            image_path = input("Image path: ").strip()

        if not Path(image_path).exists():
            self.print("[bold red]Error: Image not found![/bold red]")
            return

        # Get object name
        if RICH_AVAILABLE:
            object_name = Prompt.ask("[cyan]Object to locate[/cyan]")
        else:
            object_name = input("Object to locate: ").strip()

        # Run inference
        self.print("\n[dim]Locating object...[/dim]")
        result = await self.client.point(image_path, object_name)

        # Display result
        self.display_result(result, "Point Result")

        # Save option
        if self._ask_save():
            self.save_result(result, "point")

    def display_result(self, result: Dict[str, Any], title: str):
        """Display inference result"""
        if "error" in result:
            self.print(f"\n[bold red]Error: {result['error']}[/bold red]")
            return

        if RICH_AVAILABLE:
            # Create panel with result
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
                content.append(f"\n[dim]⏱  Inference Time: {result['inference_time_ms']}ms[/dim]")

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

    def show_stats(self):
        """Display client statistics"""
        if not self.client:
            self.print("[bold red]Client not initialized![/bold red]")
            return

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

    def _ask_save(self) -> bool:
        """Ask if user wants to save result"""
        if RICH_AVAILABLE:
            return Confirm.ask("\n[cyan]Save result to JSON?[/cyan]", default=False)
        else:
            response = input("\nSave result to JSON? (y/n): ").strip().lower()
            return response == 'y'

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
            self.print(f"[bold red]Error saving result: {e}[/bold red]")

    async def install_new_model(self):
        """Install a new model"""
        self.print("\n[bold yellow]═══ Install New Model ═══[/bold yellow]\n")

        # Select provider
        if RICH_AVAILABLE:
            provider = Prompt.ask(
                "[cyan]Select provider[/cyan]",
                choices=["ollama", "huggingface"],
                default="ollama"
            )
        else:
            print("Provider: 1) ollama  2) huggingface")
            choice = input("Choose (1/2): ").strip()
            provider = "huggingface" if choice == "2" else "ollama"

        # Get model name
        if provider == "ollama":
            self.print("\n[dim]Example: llama3.2-vision:11b, qwen2-vl:7b, llava:13b[/dim]")
            if RICH_AVAILABLE:
                model_name = Prompt.ask("[cyan]Enter Ollama model name[/cyan]")
            else:
                model_name = input("Enter Ollama model name: ").strip()
        else:
            self.print("\n[dim]Example: microsoft/Florence-2-base, Salesforce/blip2-opt-2.7b[/dim]")
            if RICH_AVAILABLE:
                model_name = Prompt.ask("[cyan]Enter HuggingFace model name[/cyan]")
            else:
                model_name = input("Enter HuggingFace model name: ").strip()

        if not model_name:
            self.print("[bold red]Model name cannot be empty![/bold red]")
            return

        # Confirm installation
        if RICH_AVAILABLE:
            confirm = Confirm.ask(f"\n[cyan]Install {model_name} from {provider}?[/cyan]", default=True)
        else:
            response = input(f"\nInstall {model_name} from {provider}? (y/n): ").strip().lower()
            confirm = response == 'y'

        if not confirm:
            self.print("[yellow]Installation cancelled[/yellow]")
            return

        # Install model
        self.print(f"\n[bold cyan]Installing {model_name}...[/bold cyan]")

        try:
            if provider == "ollama":
                success = await self.client.install_ollama_model(model_name)
            else:
                success = await self.client.install_hf_model(model_name)

            if success:
                self.print(f"\n[bold green]✓ {model_name} installed successfully![/bold green]")

                # Ask if user wants to load it now
                if RICH_AVAILABLE:
                    load_now = Confirm.ask("\n[cyan]Load this model now?[/cyan]", default=True)
                else:
                    response = input("\nLoad this model now? (y/n): ").strip().lower()
                    load_now = response == 'y'

                if load_now:
                    if provider == "ollama":
                        success = await self.client._load_ollama_model(model_name)
                        if success:
                            self.client.current_model = model_name
                            self.client.current_provider = 'ollama'
                            self.print("[bold green]✓ Model loaded![/bold green]")
                    else:
                        self.print("[yellow]Please restart to use the new HuggingFace model[/yellow]")
            else:
                self.print(f"\n[bold red]✗ Failed to install {model_name}[/bold red]")

        except Exception as e:
            self.print(f"\n[bold red]Error: {e}[/bold red]")

    async def run(self):
        """Main application loop"""
        self.clear_screen()
        self.show_banner()

        # Initialize client
        if not await self.initialize():
            return

        # Load default model
        self.print("\n[bold cyan]Loading default model...[/bold cyan]")
        default_model = self.client.config.get('default_model', 'qwen2vl-3b')

        if not await self.client.load_model(default_model):
            self.print("[bold yellow]Default model failed to load. Please select a model.[/bold yellow]")
            if not await self.select_and_load_model():
                self.print("[bold red]No model loaded. Exiting.[/bold red]")
                return

        # Main loop
        while True:
            try:
                choice = self.show_main_menu()

                if choice == "0":
                    self.print("\n[bold cyan]Thanks for using VLM CLI! Goodbye! 👋[/bold cyan]\n")
                    break
                elif choice == "1":
                    await self.run_qa()
                elif choice == "2":
                    await self.run_caption()
                elif choice == "3":
                    await self.run_detect()
                elif choice == "4":
                    await self.run_point()
                elif choice == "5":
                    self.show_stats()
                elif choice == "6":
                    await self.select_and_load_model()
                elif choice == "7":
                    await self.install_new_model()
                else:
                    self.print("[bold red]Invalid option![/bold red]")

                # Wait for user before continuing
                if RICH_AVAILABLE:
                    Prompt.ask("\n[dim]Press Enter to continue[/dim]", default="")
                else:
                    input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                self.print("\n\n[bold yellow]Interrupted by user[/bold yellow]")
                if RICH_AVAILABLE:
                    if Confirm.ask("[cyan]Exit application?[/cyan]", default=True):
                        break
                else:
                    response = input("Exit application? (y/n): ").strip().lower()
                    if response == 'y':
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
