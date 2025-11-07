"""Enhanced live monitoring display with resource metrics."""

import time
from typing import Optional

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from ...cli.tui_manager import tui
from ..models import QuantizationTask
from ..models_extended import QuantizationMetrics
from ..core.resource_monitor import ResourceMonitor


def show_enhanced_live_progress(
    task: QuantizationTask,
    use_gpu: bool = False
) -> None:
    """Show enhanced live progress with resource monitoring.

    Args:
        task: QuantizationTask being monitored
        use_gpu: Whether GPU is being used
    """
    # Initialize resource monitor
    monitor = ResourceMonitor(track_gpu=use_gpu)

    # Create rich progress bar
    progress_display = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="bold green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:.1f}%"),
        TimeRemainingColumn(),
        expand=True
    )

    task_progress = progress_display.add_task(
        f"Quantizing {task.model_info.name}",
        total=100
    )

    # Track metrics history for calculations
    last_progress = -1
    start_time = time.time()
    metrics_history = []

    with Live(
        refresh_per_second=4,  # 4 updates per second
        console=tui.console
    ) as live:
        while task.is_active:
            # Update progress if changed
            if task.progress != last_progress:
                progress_display.update(task_progress, completed=task.progress)
                last_progress = task.progress

            # Get current resource metrics
            resource_metrics = monitor.get_all_metrics()

            # Build metrics table
            metrics_table = Table.grid(padding=(0, 2))
            metrics_table.add_column(style="cyan", justify="right")
            metrics_table.add_column(style="white")

            # Stage information
            stage = getattr(task, 'stage', 'Processing')
            substage = getattr(task, 'substage', None)
            stage_text = f"[bold]{stage}[/bold]"
            if substage:
                stage_text += f" [dim]({substage})[/dim]"
            metrics_table.add_row("Stage:", stage_text)

            # CPU metrics
            cpu_pct = resource_metrics.get('cpu_percent', 0.0)
            cpu_sparkline = resource_metrics.get('cpu_sparkline', '')
            cpu_color = "green" if cpu_pct < 50 else "yellow" if cpu_pct < 80 else "red"
            metrics_table.add_row(
                "CPU:",
                f"[{cpu_color}]{cpu_pct:.1f}%[/{cpu_color}] {cpu_sparkline}"
            )

            # Memory metrics - Show both process and system-wide
            mem_mb = resource_metrics.get('memory_mb', 0.0)
            mem_pct = resource_metrics.get('memory_percent', 0.0)
            mem_sparkline = resource_metrics.get('memory_sparkline', '')

            # System memory delta (more accurate for large models)
            sys_delta_mb = resource_metrics.get('system_memory_delta_mb', 0.0)
            sys_pct = resource_metrics.get('system_memory_percent', 0.0)

            # Use system percentage for color (what user sees in Activity Monitor)
            mem_color = "green" if sys_pct < 60 else "yellow" if sys_pct < 80 else "red"

            # Show process memory + system delta (most informative)
            if sys_delta_mb > 100:  # If significant system memory change
                metrics_table.add_row(
                    "Memory:",
                    f"[{mem_color}]Process: {mem_mb:.0f} MB | System: +{sys_delta_mb:.0f} MB ({sys_pct:.1f}%)[/{mem_color}] {mem_sparkline}"
                )
            else:
                # Fall back to process-only if system delta is small
                metrics_table.add_row(
                    "Memory:",
                    f"[{mem_color}]{mem_mb:.0f} MB ({mem_pct:.1f}%)[/{mem_color}] {mem_sparkline}"
                )

            # GPU metrics (if available)
            if 'gpu_memory_mb' in resource_metrics:
                gpu_mb = resource_metrics['gpu_memory_mb']
                gpu_pct = resource_metrics.get('gpu_memory_percent', 0.0)
                gpu_color = "green" if gpu_pct < 70 else "yellow" if gpu_pct < 90 else "red"
                if gpu_pct > 0:
                    metrics_table.add_row(
                        "GPU VRAM:",
                        f"[{gpu_color}]{gpu_mb:.0f} MB ({gpu_pct:.1f}%)[/{gpu_color}]"
                    )
                else:
                    # MPS doesn't report percentage
                    metrics_table.add_row(
                        "GPU VRAM:",
                        f"[{gpu_color}]{gpu_mb:.0f} MB[/{gpu_color}]"
                    )

            # Elapsed time
            elapsed = time.time() - start_time
            mins = int(elapsed / 60)
            secs = int(elapsed % 60)
            metrics_table.add_row("Elapsed:", f"{mins}m {secs}s")

            # ETA
            if task.eta_seconds and task.eta_seconds > 0:
                eta_mins = int(task.eta_seconds / 60)
                eta_secs = int(task.eta_seconds % 60)
                metrics_table.add_row("ETA:", f"~{eta_mins}m {eta_secs}s")

            # Size information (if available)
            if hasattr(task, 'estimated_size_gb') and task.estimated_size_gb:
                original_size = task.model_info.size_gb
                estimated_size = task.estimated_size_gb
                reduction = ((original_size - estimated_size) / original_size) * 100
                metrics_table.add_row(
                    "Size:",
                    f"{estimated_size:.2f} GB [dim](~{reduction:.0f}% reduction)[/dim]"
                )

            # Disk write speed (if available)
            if hasattr(task, 'write_speed_mbps') and task.write_speed_mbps:
                metrics_table.add_row(
                    "Write Speed:",
                    f"{task.write_speed_mbps:.1f} MB/s"
                )

            # Build the complete display
            content_group = Group(
                progress_display,
                "",  # Spacing
                Panel(
                    metrics_table,
                    title="[bold]📊 Resource Metrics[/bold]",
                    border_style="blue",
                    padding=(0, 1)
                ),
                "",  # Spacing
                Text("Press Ctrl+C to cancel (not recommended - may corrupt output)", style="dim")
            )

            main_panel = Panel(
                content_group,
                title=f"[bold cyan]🔧 Quantization: {task.model_info.name}[/bold cyan]",
                subtitle=f"[dim]{task.quant_type.display_name}[/dim]",
                border_style="cyan",
                padding=(1, 2)
            )

            live.update(main_panel)

            # Store metrics for trend analysis
            metrics_history.append({
                'time': time.time(),
                'progress': task.progress,
                'cpu': resource_metrics.get('cpu_percent', 0),
                'memory': resource_metrics.get('memory_mb', 0)
            })

            # Keep only last 100 measurements
            if len(metrics_history) > 100:
                metrics_history.pop(0)

            # Poll at 250ms intervals
            time.sleep(0.25)

    # Show final status
    tui.console.print("\n")

    if task.status.value == "completed":
        # Calculate actual size (handle both files and directories)
        quantized_size_gb = 0.0
        if task.output_path.exists():
            if task.output_path.is_file():
                quantized_size_gb = task.output_path.stat().st_size / (1024 ** 3)
            elif task.output_path.is_dir():
                total_bytes = sum(f.stat().st_size for f in task.output_path.rglob("*") if f.is_file())
                quantized_size_gb = total_bytes / (1024 ** 3)

        # Calculate size reduction
        original_size = task.model_info.size_gb
        reduction_pct = ((original_size - quantized_size_gb) / original_size) * 100

        # Calculate average metrics
        avg_cpu = sum(m['cpu'] for m in metrics_history) / len(metrics_history) if metrics_history else 0
        peak_memory = max(m['memory'] for m in metrics_history) if metrics_history else 0

        tui.show_message(
            f"""[bold green]✅ Quantization Completed![/bold green]

[bold]Output:[/bold] {task.output_path}
[bold]Final Size:[/bold] {quantized_size_gb:.2f} GB (original: {original_size:.2f} GB)
[bold]Size Reduction:[/bold] {reduction_pct:.1f}%
[bold]Time Taken:[/bold] {int(elapsed / 60)}m {int(elapsed % 60)}s

[bold cyan]Resource Usage:[/bold cyan]
  • Average CPU: {avg_cpu:.1f}%
  • Peak Memory: {peak_memory:.0f} MB

The quantized model has been saved and is ready to use.""",
            title="Success",
            style="green",
        )
    else:
        tui.show_error(f"Quantization failed: {task.error}")

    tui.prompt("Press Enter to continue...", style="dim")
