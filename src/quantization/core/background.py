"""Background job management for quantization tasks."""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

from loguru import logger

from ..techniques.gguf import GGUFQuantizer
from ..models import QuantizationTask, TaskStatus


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
        self.gguf_quantizer = GGUFQuantizer()

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

                # Define progress wrapper
                def progress_wrapper(progress: float, eta: Optional[float]):
                    if progress_callback:
                        progress_callback(task.task_id, progress, eta)

                # Execute quantization based on method family
                if task.quant_type.method_family == "GGUF":
                    success = self.gguf_quantizer.quantize(task, progress_wrapper)
                else:
                    task.status = TaskStatus.FAILED
                    task.error = f"Quantization method {task.quant_type.method_family} not yet implemented"
                    success = False

                # Update completion time
                if success:
                    task.completed_at = datetime.now()
                    logger.info(f"Task {task.task_id} completed successfully")
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
        """Load state from disk."""
        if not self.state_file.exists():
            logger.debug("No state file found, starting fresh")
            return

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)

            # Load tasks (but don't restart them - just for history)
            for task_dict in state.get("tasks", []):
                # TODO: Reconstruct QuantizationTask from dict
                # For now, skip loading old tasks
                pass

            logger.info(f"Loaded state from {self.state_file}")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
