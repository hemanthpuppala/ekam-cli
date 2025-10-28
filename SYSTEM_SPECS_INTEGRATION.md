# System Specifications & Monitoring Integration Guide

## Overview

The new `system_specs` module provides comprehensive hardware detection and real-time monitoring capabilities for VLM testing and inference.

## Quick Start

### 1. Detect System Specifications at Startup

```python
from src.system_specs import SystemSpecsDetector

# Detect all system specs
detector = SystemSpecsDetector()
specs = detector.detect_all()

# Print summary
summary = detector.get_summary(specs)
print(summary)

# Access specific information
if 'cpu' in specs:
    print(f"CPU Cores: {specs['cpu']['cores_physical']}")

if 'gpus' in specs:
    for gpu in specs['gpus']:
        print(f"GPU: {gpu['device_name']} - {gpu['total_memory_gb']:.1f} GB")

if 'temperature' in specs:
    if 'cpu_celsius' in specs['temperature']:
        print(f"CPU Temperature: {specs['temperature']['cpu_celsius']:.1f}°C")
```

### 2. Real-Time Monitoring

```python
from src.system_specs import SystemMonitor, MonitoringConfig, MonitoringLevel

# Configure monitoring
config = MonitoringConfig(
    level=MonitoringLevel.DETAILED,
    track_temperature=True,
    track_per_core=True,
    track_gpu=True,
    interval_sec=1.0
)

# Create monitor
monitor = SystemMonitor(config)

# Get single snapshot
snapshot = monitor.get_snapshot()
print(f"CPU Usage: {snapshot.cpu.usage_percent:.1f}%")
print(f"Memory Used: {snapshot.memory.used_ram_gb:.1f} GB")

# Start continuous monitoring
monitor.start_continuous_monitoring(interval_sec=0.5)

# ... do work ...

# Stop monitoring
monitor.stop_continuous_monitoring()

# Get recent snapshots
recent = monitor.get_recent_snapshots(count=10)
```

### 3. Track Inference Metrics

```python
from src.system_specs import SystemMonitor, MetricsExporter
from pathlib import Path

monitor = SystemMonitor()

# Use context manager for automatic tracking
with monitor.start_inference_tracking("llava-v1.5-7b") as tracker:
    # Run your inference here
    result = model.generate(prompt, image)
    tokens_generated = len(result)

# Get metrics after inference
metrics = tracker.get_metrics(tokens_generated=tokens_generated)

print(f"Inference Time: {metrics.inference_time_sec:.2f}s")
print(f"Tokens/Second: {metrics.tokens_per_second:.1f}")
print(f"Peak CPU: {metrics.peak_cpu_percent:.1f}%")
print(f"Peak GPU Memory: {metrics.peak_gpu_memory_mb:.1f} MB")
print(f"CPU Temp Delta: {metrics.cpu_temp_delta:.1f}°C")

# Export metrics
MetricsExporter.to_json(metrics, Path("metrics/inference_results.json"))
MetricsExporter.inference_metrics_to_csv(metrics, Path("metrics/inference_summary.csv"))
```

## Integration with Existing Inference Pipeline

### In `src/core/inference_handler.py`

```python
from src.system_specs import SystemMonitor, MetricsExporter
from pathlib import Path
from loguru import logger

class InferenceHandler:
    def __init__(self):
        self.monitor = SystemMonitor()
        self.metrics_dir = Path("metrics")
        self.metrics_dir.mkdir(exist_ok=True)

    def run_inference(self, model_name: str, prompt: str, image_path: str = None):
        """Run inference with automatic monitoring."""

        # Start tracking
        with self.monitor.start_inference_tracking(model_name) as tracker:
            # Your existing inference code
            result = self._run_model(prompt, image_path)
            tokens = self._count_tokens(result)

        # Get and save metrics
        metrics = tracker.get_metrics(tokens_generated=tokens)

        # Log key metrics
        logger.info(f"Inference completed in {metrics.inference_time_sec:.2f}s")
        logger.info(f"Tokens/sec: {metrics.tokens_per_second:.1f}")
        logger.info(f"Peak GPU Memory: {metrics.peak_gpu_memory_mb:.1f} MB")

        if metrics.throttling_occurred:
            logger.warning("⚠️ Thermal throttling detected during inference!")

        # Export metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.metrics_dir / f"inference_{model_name}_{timestamp}.json"
        MetricsExporter.to_json(metrics, metrics_file)

        return result, metrics
```

### In `src/core/app.py` (Startup)

```python
from src.system_specs import SystemSpecsDetector
from loguru import logger

class Application:
    def __init__(self):
        # Detect system specs at startup
        logger.info("Detecting system specifications...")
        detector = SystemSpecsDetector()
        self.system_specs = detector.detect_all()

        # Log system info
        summary = detector.get_summary(self.system_specs)
        logger.info(f"System Specs:\n{summary}")

        # Check for issues
        if 'temperature' in self.system_specs:
            temp = self.system_specs['temperature']
            if temp.get('throttling'):
                logger.warning("⚠️ System is currently throttling due to high temperature!")

        # Initialize monitor
        self.monitor = SystemMonitor()
```

## Export Formats

### JSON Export
```python
# Export single snapshot
snapshot = monitor.get_snapshot()
MetricsExporter.to_json(snapshot, Path("snapshot.json"))

# Export inference metrics
MetricsExporter.to_json(metrics, Path("inference_results.json"))
```

### CSV Export
```python
# Export snapshots to CSV
snapshots = monitor.get_recent_snapshots(count=100)
MetricsExporter.snapshots_to_csv(snapshots, Path("snapshots.csv"))

# Export inference summary to CSV
MetricsExporter.inference_metrics_to_csv(metrics, Path("inference_summary.csv"))
```

### Prometheus Export
```python
# Export to Prometheus format
snapshot = monitor.get_snapshot()
prometheus_text = MetricsExporter.to_prometheus(snapshot, metric_prefix="vlm")

# Or save to file
MetricsExporter.save_prometheus(snapshot, Path("metrics.prom"))
```

## CLI Integration

### Add to main CLI

```python
# In src/cli/menus.py or main CLI handler

def show_system_diagnostics():
    """Show comprehensive system diagnostics."""
    from src.system_specs import SystemSpecsDetector
    from rich.console import Console
    from rich.table import Table

    console = Console()
    detector = SystemSpecsDetector()
    specs = detector.detect_all()

    # Display in rich table format
    table = Table(title="System Specifications")
    table.add_column("Component", style="cyan")
    table.add_column("Details", style="white")

    # Add rows based on detected specs
    if 'cpu' in specs:
        cpu = specs['cpu']
        table.add_row("CPU", f"{cpu['cores_physical']} cores @ {cpu.get('frequency', {}).get('current_ghz', 0):.2f} GHz")

    if 'memory' in specs:
        mem = specs['memory']
        table.add_row("RAM", f"{mem['total_ram_gb']:.1f} GB ({mem['available_ram_gb']:.1f} GB available)")

    if 'gpus' in specs:
        for i, gpu in enumerate(specs['gpus']):
            table.add_row(f"GPU {i}", f"{gpu['device_name']} ({gpu['total_memory_gb']:.1f} GB)")

    if 'temperature' in specs:
        temp = specs['temperature']
        if 'cpu_celsius' in temp:
            table.add_row("CPU Temp", f"{temp['cpu_celsius']:.1f}°C")

    console.print(table)
```

## Advanced Usage

### Custom Callbacks for Real-Time Monitoring

```python
def on_snapshot(snapshot):
    """Called for each snapshot during monitoring."""
    if snapshot.temperatures and snapshot.temperatures.cpu_temp_celsius > 80.0:
        logger.warning(f"High CPU temperature: {snapshot.temperatures.cpu_temp_celsius:.1f}°C")

    if snapshot.gpus:
        for gpu in snapshot.gpus:
            if gpu.gpu_utilization_percent > 95.0:
                logger.info(f"GPU {gpu.device_id} at {gpu.gpu_utilization_percent:.1f}% utilization")

# Start monitoring with callback
monitor.start_continuous_monitoring(interval_sec=1.0, callback=on_snapshot)
```

### Monitoring Specific GPUs

```python
from src.system_specs.backends import NvidiaBackend

nvidia = NvidiaBackend()
if nvidia.is_available():
    # Get specific GPU metrics
    gpu_0 = nvidia.get_gpu_metrics(device_id=0)
    print(f"GPU 0: {gpu_0.device_name}")
    print(f"  Memory: {gpu_0.used_memory_gb:.1f} / {gpu_0.total_memory_gb:.1f} GB")
    print(f"  Utilization: {gpu_0.gpu_utilization_percent:.1f}%")
    print(f"  Temperature: {gpu_0.temperature_celsius:.1f}°C")
    print(f"  Power: {gpu_0.power_draw_watts:.1f} W")
```

## Testing

```python
# Test system detection
def test_system_detection():
    detector = SystemSpecsDetector()
    specs = detector.detect_all()

    assert 'platform' in specs
    assert 'cpu' in specs
    assert 'memory' in specs

    print("✅ System detection working")
    print(detector.get_summary(specs))

# Test monitoring
def test_monitoring():
    monitor = SystemMonitor()

    snapshot = monitor.get_snapshot()
    assert snapshot.cpu is not None
    assert snapshot.memory is not None

    print("✅ Monitoring working")
    print(f"CPU: {snapshot.cpu.usage_percent:.1f}%")
    print(f"Memory: {snapshot.memory.used_ram_gb:.1f} GB")

# Test inference tracking
def test_inference_tracking():
    monitor = SystemMonitor()

    with monitor.start_inference_tracking("test-model") as tracker:
        import time
        time.sleep(2)  # Simulate inference

    metrics = tracker.get_metrics(tokens_generated=100)

    assert metrics.inference_time_sec >= 2.0
    assert metrics.tokens_per_second > 0

    print("✅ Inference tracking working")
    print(f"Time: {metrics.inference_time_sec:.2f}s")
    print(f"Tokens/sec: {metrics.tokens_per_second:.1f}")

if __name__ == "__main__":
    test_system_detection()
    test_monitoring()
    test_inference_tracking()
```

## Dependencies

The system_specs module uses:
- `psutil` - Cross-platform system metrics (required)
- `pynvml` (nvidia-ml-py) - NVIDIA GPU metrics (optional, for detailed GPU stats)
- `torch` - GPU detection (already in project)

Install optional dependency:
```bash
pip install nvidia-ml-py
```

## Platform Support

| Feature | Linux | macOS | Windows | Raspberry Pi | Jetson |
|---------|-------|-------|---------|--------------|--------|
| CPU metrics | ✅ | ✅ | ✅ | ✅ | ✅ |
| Memory metrics | ✅ | ✅ | ✅ | ✅ | ✅ |
| Storage metrics | ✅ | ✅ | ✅ | ✅ | ✅ |
| CPU temperature | ✅ | ⚠️ | ⚠️ | ✅ | ✅ |
| GPU metrics (NVIDIA) | ✅ | ⚠️ | ✅ | ❌ | ✅ |
| GPU metrics (AMD) | ⚠️ | ❌ | ⚠️ | ❌ | ❌ |
| GPU metrics (Apple) | ❌ | ✅ | ❌ | ❌ | ❌ |
| Per-core CPU | ✅ | ✅ | ✅ | ✅ | ✅ |
| Network I/O | ✅ | ✅ | ✅ | ✅ | ✅ |

✅ Full support | ⚠️ Partial support | ❌ Not supported
