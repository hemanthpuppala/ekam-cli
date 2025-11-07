"""Export system metrics to various formats."""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from .base import SystemHealthSnapshot, InferenceMetrics


class MetricsExporter:
    """Export system metrics to various formats."""

    @staticmethod
    def to_json(data: Any, output_path: Path, pretty: bool = True):
        """Export data to JSON file.

        Args:
            data: Data to export (SystemHealthSnapshot, InferenceMetrics, or dict)
            output_path: Path to output file
            pretty: Whether to pretty-print JSON
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, (SystemHealthSnapshot, InferenceMetrics)):
            data_dict = MetricsExporter._to_serializable_dict(data)
        elif isinstance(data, list):
            data_dict = [MetricsExporter._to_serializable_dict(item) for item in data]
        else:
            data_dict = data

        with open(output_path, 'w') as f:
            if pretty:
                json.dump(data_dict, f, indent=2, default=str)
            else:
                json.dump(data_dict, f, default=str)

    @staticmethod
    def _to_serializable_dict(obj: Any) -> Dict:
        """Convert object to JSON-serializable dictionary."""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, (list, tuple)):
                    result[key] = [MetricsExporter._to_serializable_dict(item) if hasattr(item, '__dict__') else item
                                  for item in value]
                elif hasattr(value, '__dict__'):
                    result[key] = MetricsExporter._to_serializable_dict(value)
                else:
                    result[key] = value
            return result
        return obj

    @staticmethod
    def snapshots_to_csv(snapshots: List[SystemHealthSnapshot], output_path: Path):
        """Export snapshots to CSV file.

        Args:
            snapshots: List of system health snapshots
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not snapshots:
            return

        with open(output_path, 'w', newline='') as f:
            # Define fields
            fieldnames = [
                'timestamp',
                'cpu_usage_percent',
                'cpu_frequency_ghz',
                'memory_used_gb',
                'memory_percent',
                'gpu_0_utilization',
                'gpu_0_memory_gb',
                'gpu_0_temperature',
                'cpu_temperature',
                'disk_read_mb',
                'disk_write_mb',
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for snapshot in snapshots:
                row = {
                    'timestamp': datetime.fromtimestamp(snapshot.timestamp).isoformat(),
                    'cpu_usage_percent': snapshot.cpu.usage_percent if snapshot.cpu else 0.0,
                    'cpu_frequency_ghz': snapshot.cpu.frequency.cpu_current_ghz if snapshot.cpu and snapshot.cpu.frequency else 0.0,
                    'memory_used_gb': snapshot.memory.used_ram_gb if snapshot.memory else 0.0,
                    'memory_percent': snapshot.memory.percent_used if snapshot.memory else 0.0,
                    'gpu_0_utilization': snapshot.gpus[0].gpu_utilization_percent if snapshot.gpus else 0.0,
                    'gpu_0_memory_gb': snapshot.gpus[0].used_memory_gb if snapshot.gpus else 0.0,
                    'gpu_0_temperature': snapshot.gpus[0].temperature_celsius if snapshot.gpus else None,
                    'cpu_temperature': snapshot.temperatures.cpu_temp_celsius if snapshot.temperatures else None,
                    'disk_read_mb': snapshot.storage.total_read_mb if snapshot.storage else 0.0,
                    'disk_write_mb': snapshot.storage.total_write_mb if snapshot.storage else 0.0,
                }
                writer.writerow(row)

    @staticmethod
    def inference_metrics_to_csv(metrics: InferenceMetrics, output_path: Path):
        """Export inference metrics summary to CSV.

        Args:
            metrics: Inference metrics
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            fieldnames = [
                'model_name',
                'inference_time_sec',
                'tokens_generated',
                'tokens_per_second',
                'peak_cpu_percent',
                'peak_memory_mb',
                'peak_gpu_memory_mb',
                'peak_gpu_utilization',
                'avg_cpu_percent',
                'avg_memory_mb',
                'avg_gpu_memory_mb',
                'avg_gpu_utilization',
                'cpu_temp_start',
                'cpu_temp_end',
                'cpu_temp_delta',
                'cpu_temp_peak',
                'throttling_occurred',
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            row = {
                'model_name': metrics.model_name,
                'inference_time_sec': metrics.inference_time_sec,
                'tokens_generated': metrics.tokens_generated,
                'tokens_per_second': metrics.tokens_per_second,
                'peak_cpu_percent': metrics.peak_cpu_percent,
                'peak_memory_mb': metrics.peak_memory_mb,
                'peak_gpu_memory_mb': metrics.peak_gpu_memory_mb,
                'peak_gpu_utilization': metrics.peak_gpu_utilization,
                'avg_cpu_percent': metrics.avg_cpu_percent,
                'avg_memory_mb': metrics.avg_memory_mb,
                'avg_gpu_memory_mb': metrics.avg_gpu_memory_mb,
                'avg_gpu_utilization': metrics.avg_gpu_utilization,
                'cpu_temp_start': metrics.cpu_temp_start,
                'cpu_temp_end': metrics.cpu_temp_end,
                'cpu_temp_delta': metrics.cpu_temp_delta,
                'cpu_temp_peak': metrics.cpu_temp_peak,
                'throttling_occurred': metrics.throttling_occurred,
            }
            writer.writerow(row)

    @staticmethod
    def to_prometheus(snapshot: SystemHealthSnapshot, metric_prefix: str = "vlm") -> str:
        """Export snapshot to Prometheus text format.

        Args:
            snapshot: System health snapshot
            metric_prefix: Prefix for metric names

        Returns:
            Prometheus-formatted text
        """
        lines = []

        # CPU metrics
        if snapshot.cpu:
            lines.append(f'# HELP {metric_prefix}_cpu_usage_percent CPU usage percentage')
            lines.append(f'# TYPE {metric_prefix}_cpu_usage_percent gauge')
            lines.append(f'{metric_prefix}_cpu_usage_percent {snapshot.cpu.usage_percent}')

            if snapshot.cpu.frequency:
                lines.append(f'# HELP {metric_prefix}_cpu_frequency_ghz CPU frequency in GHz')
                lines.append(f'# TYPE {metric_prefix}_cpu_frequency_ghz gauge')
                lines.append(f'{metric_prefix}_cpu_frequency_ghz {snapshot.cpu.frequency.cpu_current_ghz}')

            if snapshot.cpu.load_avg_1min:
                lines.append(f'# HELP {metric_prefix}_cpu_load_1min CPU load average 1 minute')
                lines.append(f'# TYPE {metric_prefix}_cpu_load_1min gauge')
                lines.append(f'{metric_prefix}_cpu_load_1min {snapshot.cpu.load_avg_1min}')

        # Memory metrics
        if snapshot.memory:
            lines.append(f'# HELP {metric_prefix}_memory_used_gb Memory used in GB')
            lines.append(f'# TYPE {metric_prefix}_memory_used_gb gauge')
            lines.append(f'{metric_prefix}_memory_used_gb {snapshot.memory.used_ram_gb}')

            lines.append(f'# HELP {metric_prefix}_memory_percent Memory usage percentage')
            lines.append(f'# TYPE {metric_prefix}_memory_percent gauge')
            lines.append(f'{metric_prefix}_memory_percent {snapshot.memory.percent_used}')

        # GPU metrics
        if snapshot.gpus:
            for gpu in snapshot.gpus:
                labels = f'device_id="{gpu.device_id}",device_name="{gpu.device_name}"'

                lines.append(f'# HELP {metric_prefix}_gpu_utilization GPU utilization percentage')
                lines.append(f'# TYPE {metric_prefix}_gpu_utilization gauge')
                lines.append(f'{metric_prefix}_gpu_utilization{{{labels}}} {gpu.gpu_utilization_percent}')

                lines.append(f'# HELP {metric_prefix}_gpu_memory_used_gb GPU memory used in GB')
                lines.append(f'# TYPE {metric_prefix}_gpu_memory_used_gb gauge')
                lines.append(f'{metric_prefix}_gpu_memory_used_gb{{{labels}}} {gpu.used_memory_gb}')

                if gpu.temperature_celsius:
                    lines.append(f'# HELP {metric_prefix}_gpu_temperature_celsius GPU temperature in Celsius')
                    lines.append(f'# TYPE {metric_prefix}_gpu_temperature_celsius gauge')
                    lines.append(f'{metric_prefix}_gpu_temperature_celsius{{{labels}}} {gpu.temperature_celsius}')

                if gpu.power_draw_watts:
                    lines.append(f'# HELP {metric_prefix}_gpu_power_watts GPU power draw in watts')
                    lines.append(f'# TYPE {metric_prefix}_gpu_power_watts gauge')
                    lines.append(f'{metric_prefix}_gpu_power_watts{{{labels}}} {gpu.power_draw_watts}')

        # Temperature metrics
        if snapshot.temperatures:
            if snapshot.temperatures.cpu_temp_celsius:
                lines.append(f'# HELP {metric_prefix}_cpu_temperature_celsius CPU temperature in Celsius')
                lines.append(f'# TYPE {metric_prefix}_cpu_temperature_celsius gauge')
                lines.append(f'{metric_prefix}_cpu_temperature_celsius {snapshot.temperatures.cpu_temp_celsius}')

        # Storage metrics
        if snapshot.storage:
            lines.append(f'# HELP {metric_prefix}_disk_read_mb Disk read in MB')
            lines.append(f'# TYPE {metric_prefix}_disk_read_mb counter')
            lines.append(f'{metric_prefix}_disk_read_mb {snapshot.storage.total_read_mb}')

            lines.append(f'# HELP {metric_prefix}_disk_write_mb Disk write in MB')
            lines.append(f'# TYPE {metric_prefix}_disk_write_mb counter')
            lines.append(f'{metric_prefix}_disk_write_mb {snapshot.storage.total_write_mb}')

        # Network metrics
        if snapshot.network:
            lines.append(f'# HELP {metric_prefix}_network_bytes_sent Network bytes sent')
            lines.append(f'# TYPE {metric_prefix}_network_bytes_sent counter')
            lines.append(f'{metric_prefix}_network_bytes_sent {snapshot.network.bytes_sent}')

            lines.append(f'# HELP {metric_prefix}_network_bytes_recv Network bytes received')
            lines.append(f'# TYPE {metric_prefix}_network_bytes_recv counter')
            lines.append(f'{metric_prefix}_network_bytes_recv {snapshot.network.bytes_recv}')

        return '\n'.join(lines) + '\n'

    @staticmethod
    def save_prometheus(snapshot: SystemHealthSnapshot, output_path: Path, metric_prefix: str = "vlm"):
        """Save snapshot to Prometheus text format file.

        Args:
            snapshot: System health snapshot
            output_path: Path to output file
            metric_prefix: Prefix for metric names
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        prometheus_text = MetricsExporter.to_prometheus(snapshot, metric_prefix)

        with open(output_path, 'w') as f:
            f.write(prometheus_text)
