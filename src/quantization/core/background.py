"""Background job management for quantization tasks."""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

from loguru import logger

from ..techniques.gguf import GGUFQuantizer
from ..techniques.generic import GenericQuantizer
from ..techniques.gptq import GPTQQuantizer
from ..techniques.awq import AWQQuantizer
from ..techniques.bnb import BitsAndBytesQuantizer
from ..techniques.mlx import MLXQuantizer
from ..techniques.openvino import OpenVINOQuantizer
from ..techniques.dequantization import DequantizationQuantizer
from ..models import QuantizationTask, TaskStatus, QuantizationModule
from ..orchestrator import QuantizationOrchestrator


class BackgroundJobManager:
    """Manage background quantization jobs."""

    def __init__(self, state_file: Path):
        """Initialize background job manager.

        Args:
            state_file: Path to state file for persistence
        """
        self.state_file = state_file
        self.tasks: Dict[str, QuantizationTask] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()

        # Initialize quantizers
        self.generic_quantizer = GenericQuantizer()
        self.gguf_quantizer = GGUFQuantizer()
        self.gptq_quantizer = GPTQQuantizer()
        self.awq_quantizer = AWQQuantizer()
        self.bnb_quantizer = BitsAndBytesQuantizer()
        self.mlx_quantizer = MLXQuantizer()
        self.openvino_quantizer = OpenVINOQuantizer()
        self.dequantization_quantizer = DequantizationQuantizer()

        # Initialize orchestrator for multi-step conversions
        self.orchestrator = QuantizationOrchestrator()

        # Load saved state
        self._load_state()

    def submit_task(
        self,
        task: QuantizationTask,
        progress_callback: Optional[Callable[[str, float, Optional[float]], None]] = None,
    ) -> str:
        """Submit a quantization task for background processing.

        Args:
            task: Quantization task to execute
            progress_callback: Optional callback(task_id, progress%, eta_seconds)

        Returns:
            Task ID
        """
        with self.lock:
            task.started_at = datetime.now()
            task.status = TaskStatus.PREPARING
            self.tasks[task.task_id] = task
            self._save_state()

        # Create thread for this task
        def run_task():
            try:
                logger.info(f"Starting quantization task {task.task_id}")

                # Define progress wrapper that updates task AND saves state
                def progress_wrapper(progress: float, eta: Optional[float]):
                    task.progress = progress
                    task.eta_seconds = eta
                    task.status = TaskStatus.RUNNING

                    # Save state periodically (every 5% progress)
                    if int(progress) % 5 == 0:
                        with self.lock:
                            self._save_state()

                    # Call external callback if provided
                    if progress_callback:
                        progress_callback(task.task_id, progress, eta)

                # STEP 1: Check if conversion is needed (Ollama/GGUF → other formats)
                # Orchestrator now returns BOTH path AND directive
                # NOTE: For VLMs, orchestrator may raise ValueError if unsupported quantization is requested
                try:
                    conversion_path, quantizer_directive = self.orchestrator.determine_conversion_path(
                        task.model_info, task.quant_type
                    )
                    logger.debug(f"Conversion path: {conversion_path.value}")
                    logger.debug(f"Quantizer directive: {quantizer_directive.value}")

                    # Store directive on task for quantizer to check
                    task.quantizer_directive = quantizer_directive.value

                except ValueError as e:
                    # Orchestrator rejected this combination (e.g., VLM + GGUF)
                    logger.error(str(e))
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    success = False
                    return

                converted_path = None
                if self.orchestrator.needs_conversion(conversion_path):
                    logger.info(f"Multi-step conversion required: {conversion_path.value}")

                    # Execute conversion pipeline (GGUF → FP16, etc.)
                    converted_path = self.orchestrator.execute_conversion_pipeline(
                        task, conversion_path, progress_wrapper
                    )

                    if not converted_path:
                        task.status = TaskStatus.FAILED
                        task.error = "Format conversion failed"
                        success = False
                        return

                    logger.info(f"Conversion completed: {converted_path}")

                    # Update task to use converted model
                    # Store original source_path for cleanup
                    original_source_path = task.model_info.source_path
                    task.model_info.source_path = converted_path

                # STEP 2: Check orchestrator directive before proceeding with quantization
                # FP16 is NOT a quantization - it's just dtype conversion
                # After format conversion (GGUF → FP16 Safetensors), we're DONE for FP16
                # For INT8/INT4/etc., we need to apply additional quantization
                if task.quantizer_directive == "skip":
                    logger.info(
                        f"Quantizer directive is SKIP - conversion output is final\n"
                        f"Final output: {converted_path}"
                    )
                    # CRITICAL: Update task.output_path to actual converted directory
                    # so metadata is saved to the correct location
                    if converted_path:
                        task.output_path = converted_path
                        logger.debug(f"Updated task.output_path to converted directory: {converted_path}")
                    task.status = TaskStatus.COMPLETED
                    success = True
                else:
                    # STEP 3: Execute quantization based on module (takes priority) or method family
                    # Check module first for platform-specific quantizers
                    if task.module == QuantizationModule.MLX:
                        success = self.mlx_quantizer.quantize(task, progress_wrapper)
                    elif task.module == QuantizationModule.OPENVINO:
                        success = self.openvino_quantizer.quantize(task, progress_wrapper)
                    else:
                        # Fall back to method family routing
                        method_family = task.quant_type.method_family

                        if method_family == "Generic":
                            success = self.generic_quantizer.quantize(task, progress_wrapper)
                        elif method_family == "GGUF":
                            success = self.gguf_quantizer.quantize(task, progress_wrapper)
                        elif method_family == "GPTQ":
                            success = self.gptq_quantizer.quantize(task, progress_wrapper)
                        elif method_family == "AWQ":
                            success = self.awq_quantizer.quantize(task, progress_wrapper)
                        elif method_family == "BitsAndBytes":
                            success = self.bnb_quantizer.quantize(task, progress_wrapper)
                        elif method_family == "Dequantization":
                            success = self.dequantization_quantizer.quantize(task, progress_wrapper)
                        else:
                            task.status = TaskStatus.FAILED
                            task.error = f"Quantization method {method_family} not yet implemented"
                            success = False

                # STEP 3: Keep intermediate files for user (don't cleanup)
                # Users may want to use the FP16 GGUF file for other purposes
                if converted_path and success:
                    logger.info(f"Intermediate file preserved: {converted_path}")
                    logger.info("You can use this FP16 GGUF file with llama.cpp or Ollama")
                elif converted_path and not success:
                    logger.info(f"Quantization failed, but intermediate file saved: {converted_path}")
                    logger.info("You can use this FP16 GGUF file directly or try a different quantization method")

                # Update completion time
                if success:
                    task.completed_at = datetime.now()
                    logger.info(f"Task {task.task_id} completed successfully")

                    # Save quantization metadata
                    self._save_quantization_metadata(task)
                else:
                    logger.error(f"Task {task.task_id} failed: {task.error}")

                # Save final state
                with self.lock:
                    self._save_state()

            except Exception as e:
                logger.error(f"Task {task.task_id} crashed: {e}", exc_info=True)
                task.status = TaskStatus.FAILED
                task.error = f"Unexpected error: {str(e)}"
                with self.lock:
                    self._save_state()

        thread = threading.Thread(target=run_task, daemon=True, name=f"quant-{task.task_id}")
        thread.start()

        with self.lock:
            self.threads[task.task_id] = thread

        return task.task_id

    def get_task(self, task_id: str) -> Optional[QuantizationTask]:
        """Get task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task if found, None otherwise
        """
        with self.lock:
            return self.tasks.get(task_id)

    def get_active_tasks(self) -> list[QuantizationTask]:
        """Get all active (running or pending) tasks.

        Returns:
            List of active tasks
        """
        with self.lock:
            return [task for task in self.tasks.values() if task.is_active]

    def get_all_tasks(self) -> list[QuantizationTask]:
        """Get all tasks.

        Returns:
            List of all tasks
        """
        with self.lock:
            return list(self.tasks.values())

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if cancelled, False if not found or already finished
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task or task.is_finished:
                return False

            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            self._save_state()

            # Note: We can't actually kill the thread in Python safely
            # The thread will check task.status and stop on its own
            logger.info(f"Task {task_id} marked for cancellation")
            return True

    def remove_task(self, task_id: str) -> bool:
        """Remove a completed or failed task from the job list.

        Args:
            task_id: Task ID to remove

        Returns:
            True if removed, False if not found or still active
        """
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            # Only allow removal of finished tasks
            if not task.is_finished:
                logger.warning(f"Cannot remove active task {task_id}")
                return False

            # Remove from tasks dict
            del self.tasks[task_id]
            self._save_state()

            logger.info(f"Task {task_id} removed from job list")
            return True

    def has_active_jobs(self) -> bool:
        """Check if there are any active jobs.

        Returns:
            True if there are active jobs
        """
        return len(self.get_active_tasks()) > 0

    def get_status_summary(self) -> str:
        """Get a summary of current job status.

        Returns:
            Status string for display
        """
        active = self.get_active_tasks()
        if not active:
            return ""

        if len(active) == 1:
            task = active[0]
            progress_bar = self._make_progress_bar(task.progress)
            eta_str = self._format_eta(task.eta_seconds)
            return f"{task.status.emoji} Quantizing {task.model_info.name} [{task.quant_type.value.upper()}] {progress_bar} {task.progress:.0f}% {eta_str}"
        else:
            return f"🔧 {len(active)} quantization jobs running"

    def _make_progress_bar(self, progress: float, width: int = 10) -> str:
        """Create a text progress bar.

        Args:
            progress: Progress percentage (0-100)
            width: Width of progress bar in characters

        Returns:
            Progress bar string
        """
        filled = int((progress / 100.0) * width)
        empty = width - filled
        return "█" * filled + "▒" * empty

    def _format_eta(self, eta_seconds: Optional[float]) -> str:
        """Format ETA for display.

        Args:
            eta_seconds: Estimated time remaining in seconds

        Returns:
            Formatted string like "ETA: 3m 45s"
        """
        if eta_seconds is None or eta_seconds <= 0:
            return ""

        if eta_seconds < 60:
            return f"ETA: {int(eta_seconds)}s"
        elif eta_seconds < 3600:
            minutes = int(eta_seconds / 60)
            seconds = int(eta_seconds % 60)
            return f"ETA: {minutes}m {seconds}s"
        else:
            hours = int(eta_seconds / 3600)
            minutes = int((eta_seconds % 3600) / 60)
            return f"ETA: {hours}h {minutes}m"

    def _save_state(self):
        """Save current state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "tasks": [task.to_dict() for task in self.tasks.values()],
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

            logger.debug(f"Saved state to {self.state_file}")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self):
        """Load state from disk and reconstruct tasks."""
        if not self.state_file.exists():
            logger.debug("No state file found, starting fresh")
            return

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)

            # Reconstruct tasks from saved state
            tasks_loaded = 0
            for task_dict in state.get("tasks", []):
                try:
                    task = QuantizationTask.from_dict(task_dict)
                    self.tasks[task.task_id] = task
                    tasks_loaded += 1

                    # Log active tasks that were reloaded
                    if task.is_active:
                        logger.info(
                            f"Reloaded active task {task.task_id}: "
                            f"{task.model_info.name} → {task.quant_type.display_name} "
                            f"({task.status.value}, {task.progress:.1f}%)"
                        )
                    elif task.is_finished:
                        logger.debug(
                            f"Reloaded finished task {task.task_id}: {task.status.value}"
                        )

                except Exception as e:
                    logger.error(f"Failed to reconstruct task from dict: {e}")
                    continue

            logger.info(f"Loaded {tasks_loaded} tasks from {self.state_file}")

            # Note: Background threads can't be restored (Python limitation)
            # Active tasks will show their last saved state, but can't be rejoined
            active_count = len([t for t in self.tasks.values() if t.is_active])
            if active_count > 0:
                logger.warning(
                    f"{active_count} active tasks found in state, but their "
                    f"background threads cannot be restored. They may still be "
                    f"running in the background and will complete normally."
                )

        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def _save_quantization_metadata(self, task: QuantizationTask):
        """Save metadata file for quantized model.

        This allows the QuantizedProvider to discover and display info about
        quantized models.

        Args:
            task: Completed quantization task
        """
        try:
            # Calculate actual size (handle both files and directories)
            quantized_size_gb = 0.0
            if task.output_path.exists():
                if task.output_path.is_file():
                    # Single file (e.g., GGUF)
                    quantized_size_gb = task.output_path.stat().st_size / (1024 ** 3)
                elif task.output_path.is_dir():
                    # Directory (e.g., HuggingFace format) - sum all files
                    total_bytes = 0
                    for file_path in task.output_path.rglob("*"):
                        if file_path.is_file():
                            total_bytes += file_path.stat().st_size
                    quantized_size_gb = total_bytes / (1024 ** 3)

            metadata = {
                "quant_type": task.quant_type.value,
                "quant_display_name": task.quant_type.display_name,
                "method_family": task.quant_type.method_family,
                "module": task.module.value,
                "original_model": task.model_info.model_id,
                "original_model_name": task.model_info.name,
                "original_provider": str(task.model_info.provider),
                "original_size_gb": task.model_info.size_gb,
                "quantized_size_gb": quantized_size_gb,
                "task_id": task.task_id,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "elapsed_seconds": task.elapsed_seconds,
                "use_gpu": task.use_gpu,
            }

            # Determine metadata file path based on output type
            if task.output_path.is_file():
                # GGUF file - save metadata as .json next to it
                metadata_file = task.output_path.with_suffix(".json")
            else:
                # Directory (HF format) - save inside directory
                metadata_file = task.output_path / "quantization_metadata.json"

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved quantization metadata: {metadata_file}")

        except Exception as e:
            logger.error(f"Failed to save quantization metadata: {e}")
