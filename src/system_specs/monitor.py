"""Real-time system monitoring for inference and runtime tracking."""

import time
import threading
from typing import Optional, Callable, List
from collections import deque
from loguru import logger

from .base import (
    SystemHealthSnapshot,
    InferenceMetrics,
    MonitoringConfig,
    MonitoringLevel,
    CPUMetrics,
    MemoryMetrics,
    StorageMetrics,
    GPUMetrics,
)
from .backends import (
    PSUtilBackend,
    NvidiaBackend,
    LinuxBackend,
    MacOSBackend,
    WindowsBackend,
)


class SystemMonitor:
    """Real-time system monitoring for tracking resource usage."""

    def __init__(self, config: Optional[MonitoringConfig] = None):
        """Initialize system monitor.

        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        self.backends = self._initialize_backends()

        # Monitoring state
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.snapshots: deque = deque(maxlen=1000)  # Keep last 1000 snapshots

        # Callbacks
        self.callbacks: List[Callable] = []

    def _initialize_backends(self):
        """Initialize backends based on config."""
        backends = {}

        # Always initialize psutil (baseline)
        psutil = PSUtilBackend()
        if psutil.is_available():
            backends['psutil'] = psutil

        # GPU backends
        if self.config.track_gpu:
            nvidia = NvidiaBackend()
            if nvidia.is_available():
                backends['nvidia'] = nvidia

        # Platform-specific backends
        linux = LinuxBackend()
        if linux.is_available():
            backends['linux'] = linux

        macos = MacOSBackend()
        if macos.is_available():
            backends['macos'] = macos

        windows = WindowsBackend()
        if windows.is_available():
            backends['windows'] = windows

        logger.debug(f"Initialized monitoring backends: {list(backends.keys())}")
        return backends

    def get_snapshot(self) -> SystemHealthSnapshot:
        """Get current system health snapshot.

        Returns:
            Current system state
        """
        timestamp = time.time()

        # Get CPU metrics
        cpu = self._get_cpu_metrics()

        # Get memory metrics
        memory = self._get_memory_metrics()

        # Get storage metrics
        storage = self._get_storage_metrics()

        # Get GPU metrics
        gpus = self._get_gpu_metrics()

        # Get temperatures
        temperatures = None
        if self.config.track_temperature:
            temperatures = self._get_temperature_info()

        # Get network metrics
        network = None
        if self.config.track_network:
            network = self._get_network_metrics()

        # Get top processes
        top_processes = []
        if self.config.track_processes:
            if 'psutil' in self.backends:
                top_processes = self.backends['psutil'].get_top_processes(limit=10)

        # Get battery info
        battery_percent = None
        battery_plugged = None
        if 'psutil' in self.backends:
            battery = self.backends['psutil'].get_battery_info()
            if battery:
                battery_percent = battery.get('percent')
                battery_plugged = battery.get('plugged')

        # Get uptime
        uptime = None
        if 'psutil' in self.backends:
            uptime = self.backends['psutil'].get_system_uptime()

        snapshot = SystemHealthSnapshot(
            timestamp=timestamp,
            cpu=cpu,
            memory=memory,
            storage=storage,
            gpus=gpus,
            temperatures=temperatures,
            network=network,
            top_processes=top_processes,
            battery_percent=battery_percent,
            battery_plugged=battery_plugged,
            uptime_seconds=uptime,
        )

        return snapshot

    def _get_cpu_metrics(self) -> CPUMetrics:
        """Get CPU metrics from backends."""
        if 'psutil' in self.backends:
            return self.backends['psutil'].get_cpu_metrics() or CPUMetrics()
        return CPUMetrics()

    def _get_memory_metrics(self) -> MemoryMetrics:
        """Get memory metrics from backends."""
        if 'psutil' in self.backends:
            return self.backends['psutil'].get_memory_metrics() or MemoryMetrics()
        return MemoryMetrics()

    def _get_storage_metrics(self) -> StorageMetrics:
        """Get storage metrics from backends."""
        if 'psutil' in self.backends:
            return self.backends['psutil'].get_storage_metrics() or StorageMetrics()
        return StorageMetrics()

    def _get_gpu_metrics(self) -> List[GPUMetrics]:
        """Get GPU metrics from backends."""
        gpus = []

        if 'nvidia' in self.backends:
            gpus.extend(self.backends['nvidia'].get_all_gpu_metrics())

        return gpus

    def _get_temperature_info(self):
        """Get temperature info from backends."""
        # Try platform-specific backends first
        for backend_name in ['linux', 'macos', 'windows']:
            if backend_name in self.backends:
                temp = self.backends[backend_name].get_temperature_info()
                if temp:
                    return temp

        # Try psutil as fallback
        if 'psutil' in self.backends:
            return self.backends['psutil'].get_temperature_info()

        return None

    def _get_network_metrics(self):
        """Get network metrics from backends."""
        if 'psutil' in self.backends:
            return self.backends['psutil'].get_network_metrics()
        return None

    def start_continuous_monitoring(self, interval_sec: Optional[float] = None,
                                   callback: Optional[Callable] = None):
        """Start continuous background monitoring.

        Args:
            interval_sec: Sampling interval in seconds (uses config default if None)
            callback: Optional callback function called with each snapshot
        """
        if self.monitoring:
            logger.warning("Monitoring already running")
            return

        interval = interval_sec or self.config.interval_sec

        if callback:
            self.callbacks.append(callback)

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started continuous monitoring (interval: {interval}s)")

    def _monitoring_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                snapshot = self.get_snapshot()
                self.snapshots.append(snapshot)

                # Call callbacks
                for callback in self.callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.error(f"Error in monitoring callback: {e}")

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(interval)

    def stop_continuous_monitoring(self):
        """Stop continuous monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None

        logger.info("Stopped continuous monitoring")

    def get_recent_snapshots(self, count: int = 10) -> List[SystemHealthSnapshot]:
        """Get recent snapshots.

        Args:
            count: Number of recent snapshots to return

        Returns:
            List of recent snapshots
        """
        return list(self.snapshots)[-count:]

    def clear_snapshots(self):
        """Clear stored snapshots."""
        self.snapshots.clear()

    def start_inference_tracking(self, model_name: str) -> 'InferenceTracker':
        """Start tracking metrics for an inference session.

        Args:
            model_name: Name of the model being used

        Returns:
            InferenceTracker context manager
        """
        return InferenceTracker(self, model_name)


class InferenceTracker:
    """Context manager for tracking inference metrics."""

    def __init__(self, monitor: SystemMonitor, model_name: str):
        """Initialize inference tracker.

        Args:
            monitor: SystemMonitor instance
            model_name: Name of the model
        """
        self.monitor = monitor
        self.model_name = model_name

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.snapshots: List[SystemHealthSnapshot] = []

        # Baseline metrics
        self.baseline_network_bytes: int = 0
        self.baseline_disk_read_mb: float = 0.0
        self.baseline_disk_write_mb: float = 0.0

    def __enter__(self) -> 'InferenceTracker':
        """Start tracking."""
        self.start_time = time.time()

        # Get baseline snapshot
        baseline = self.monitor.get_snapshot()
        self.snapshots.append(baseline)

        # Store baseline I/O
        if baseline.network:
            self.baseline_network_bytes = baseline.network.bytes_sent + baseline.network.bytes_recv
        if baseline.storage:
            self.baseline_disk_read_mb = baseline.storage.total_read_mb
            self.baseline_disk_write_mb = baseline.storage.total_write_mb

        # Start continuous monitoring if not already running
        if not self.monitor.monitoring:
            self.monitor.start_continuous_monitoring(
                interval_sec=0.5,  # Fast sampling during inference
                callback=lambda s: self.snapshots.append(s)
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking and compute metrics."""
        self.end_time = time.time()

        # Get final snapshot
        final_snapshot = self.monitor.get_snapshot()
        self.snapshots.append(final_snapshot)

    def get_metrics(self, tokens_generated: Optional[int] = None) -> InferenceMetrics:
        """Compute inference metrics from collected snapshots.

        Args:
            tokens_generated: Number of tokens generated (optional)

        Returns:
            Computed inference metrics
        """
        if not self.start_time or not self.end_time:
            raise ValueError("Inference tracking not completed")

        inference_time = self.end_time - self.start_time

        # Compute peak and average resource usage
        cpu_usage = []
        memory_usage = []
        gpu_memory_usage = []
        gpu_utilization = []
        temperatures = []

        for snapshot in self.snapshots:
            if snapshot.cpu:
                cpu_usage.append(snapshot.cpu.usage_percent)

            if snapshot.memory:
                memory_usage.append(snapshot.memory.used_ram_gb * 1024)  # Convert to MB

            if snapshot.gpus:
                for gpu in snapshot.gpus:
                    gpu_memory_usage.append(gpu.used_memory_gb * 1024)  # Convert to MB
                    gpu_utilization.append(gpu.gpu_utilization_percent)

            if snapshot.temperatures and snapshot.temperatures.cpu_temp_celsius:
                temperatures.append(snapshot.temperatures.cpu_temp_celsius)

        # Compute statistics
        peak_cpu = max(cpu_usage) if cpu_usage else 0.0
        avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0.0

        peak_memory = max(memory_usage) if memory_usage else 0.0
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0.0

        peak_gpu_memory = max(gpu_memory_usage) if gpu_memory_usage else 0.0
        avg_gpu_memory = sum(gpu_memory_usage) / len(gpu_memory_usage) if gpu_memory_usage else 0.0

        peak_gpu_util = max(gpu_utilization) if gpu_utilization else 0.0
        avg_gpu_util = sum(gpu_utilization) / len(gpu_utilization) if gpu_utilization else 0.0

        # Temperature stats
        cpu_temp_start = None
        cpu_temp_end = None
        cpu_temp_peak = max(temperatures) if temperatures else None

        if self.snapshots:
            first = self.snapshots[0]
            last = self.snapshots[-1]

            if first.temperatures and first.temperatures.cpu_temp_celsius:
                cpu_temp_start = first.temperatures.cpu_temp_celsius

            if last.temperatures and last.temperatures.cpu_temp_celsius:
                cpu_temp_end = last.temperatures.cpu_temp_celsius

        cpu_temp_delta = None
        if cpu_temp_start and cpu_temp_end:
            cpu_temp_delta = cpu_temp_end - cpu_temp_start

        # Throttling detection
        throttling = any(
            s.temperatures and s.temperatures.throttling
            for s in self.snapshots
            if s.temperatures
        )

        # I/O metrics
        disk_read_mb = 0.0
        disk_write_mb = 0.0
        network_bytes = 0

        if self.snapshots:
            last = self.snapshots[-1]
            if last.storage:
                disk_read_mb = last.storage.total_read_mb - self.baseline_disk_read_mb
                disk_write_mb = last.storage.total_write_mb - self.baseline_disk_write_mb
            if last.network:
                network_bytes = (last.network.bytes_sent + last.network.bytes_recv) - self.baseline_network_bytes

        # Tokens per second
        tokens_per_second = None
        if tokens_generated and inference_time > 0:
            tokens_per_second = tokens_generated / inference_time

        metrics = InferenceMetrics(
            model_name=self.model_name,
            start_time=self.start_time,
            end_time=self.end_time,
            inference_time_sec=inference_time,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            peak_cpu_percent=peak_cpu,
            peak_memory_mb=peak_memory,
            peak_gpu_memory_mb=peak_gpu_memory,
            peak_gpu_utilization=peak_gpu_util,
            avg_cpu_percent=avg_cpu,
            avg_memory_mb=avg_memory,
            avg_gpu_memory_mb=avg_gpu_memory,
            avg_gpu_utilization=avg_gpu_util,
            cpu_usage_timeline=cpu_usage,
            memory_usage_timeline=memory_usage,
            gpu_memory_timeline=gpu_memory_usage,
            gpu_usage_timeline=gpu_utilization,
            temperature_timeline=temperatures,
            cpu_temp_start=cpu_temp_start,
            cpu_temp_end=cpu_temp_end,
            cpu_temp_delta=cpu_temp_delta,
            cpu_temp_peak=cpu_temp_peak,
            throttling_occurred=throttling,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            network_bytes=network_bytes,
            snapshots=self.snapshots,
        )

        return metrics
