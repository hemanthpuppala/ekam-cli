"""
Error handling and recovery for benchmark execution.

Provides:
- Error classification and severity levels
- Retry logic with exponential backoff
- Recovery strategies for different error types
- Circuit breaker pattern for repeated failures
- Comprehensive error logging
"""

from typing import Optional, Callable, Any, TypeVar, Generic
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import traceback

from loguru import logger

T = TypeVar('T')


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"              # Minor issues, can continue
    MEDIUM = "medium"        # Significant but recoverable
    HIGH = "high"            # Critical, affects results
    FATAL = "fatal"          # Unrecoverable, must stop


class ErrorCategory(str, Enum):
    """Error categories for classification."""

    MODEL_LOADING = "model_loading"
    INFERENCE = "inference"
    RESOURCE = "resource"
    VALIDATION = "validation"
    SYSTEM = "system"
    NETWORK = "network"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Recovery strategies for errors."""

    RETRY = "retry"                    # Retry the operation
    SKIP = "skip"                      # Skip and continue
    FALLBACK = "fallback"              # Use fallback value/method
    ABORT = "abort"                    # Abort benchmark
    CONTINUE_WITH_WARNING = "warning"  # Continue but log warning


@dataclass
class BenchmarkError:
    """
    Structured error information.

    Attributes:
        category: Error category
        severity: Error severity level
        message: Human-readable error message
        exception: Original exception (if any)
        context: Additional context dict
        timestamp: When error occurred
        recovery_strategy: Recommended recovery strategy
        retry_count: Number of retries attempted
    """

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    context: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.ABORT
    retry_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dict for logging/reporting."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "exception_message": str(self.exception) if self.exception else None,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "recovery_strategy": self.recovery_strategy.value,
            "retry_count": self.retry_count
        }


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing repeated failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests blocked
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        name: str = "circuit_breaker"
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout_seconds: Seconds to wait before attempting recovery
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout_seconds)
        self.name = name

        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = "CLOSED"

    @property
    def state(self) -> str:
        """Get current circuit state."""
        # Check if we should transition to HALF_OPEN
        if (
            self._state == "OPEN"
            and self._last_failure_time
            and datetime.now() - self._last_failure_time >= self.recovery_timeout
        ):
            logger.info(f"Circuit breaker [{self.name}] transitioning to HALF_OPEN")
            self._state = "HALF_OPEN"

        return self._state

    def record_success(self) -> None:
        """Record successful operation."""
        if self._state == "HALF_OPEN":
            logger.info(f"Circuit breaker [{self.name}] transitioning to CLOSED after success")
            self._state = "CLOSED"

        self._failure_count = 0
        self._last_failure_time = None

    def record_failure(self) -> None:
        """Record failed operation."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._failure_count >= self.failure_threshold:
            if self._state != "OPEN":
                logger.warning(
                    f"Circuit breaker [{self.name}] OPEN after {self._failure_count} failures"
                )
            self._state = "OPEN"

    def is_allowed(self) -> bool:
        """Check if operation is allowed."""
        return self.state in ("CLOSED", "HALF_OPEN")

    def reset(self) -> None:
        """Reset circuit breaker."""
        logger.info(f"Circuit breaker [{self.name}] manually reset")
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "CLOSED"


class ErrorHandler:
    """
    Error handling and recovery coordinator.

    Provides:
    - Error classification
    - Retry logic with exponential backoff
    - Circuit breaker integration
    - Recovery strategy execution
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        backoff_multiplier: float = 2.0,
        enable_circuit_breaker: bool = True
    ):
        """
        Initialize error handler.

        Args:
            max_retries: Maximum retry attempts
            retry_delay_seconds: Initial retry delay
            backoff_multiplier: Exponential backoff multiplier
            enable_circuit_breaker: Whether to use circuit breaker
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay_seconds
        self.backoff_multiplier = backoff_multiplier
        self.enable_circuit_breaker = enable_circuit_breaker

        # Circuit breakers per category
        self._circuit_breakers: dict[ErrorCategory, CircuitBreaker] = {}

        # Error history for analysis
        self._error_history: list[BenchmarkError] = []

    def classify_error(self, exception: Exception, context: Optional[dict] = None) -> BenchmarkError:
        """
        Classify an exception into a BenchmarkError.

        Args:
            exception: Exception to classify
            context: Additional context

        Returns:
            BenchmarkError with classification
        """
        context = context or {}
        exc_str = str(exception).lower()
        exc_type = type(exception).__name__

        # Classify by exception type and message
        if "timeout" in exc_str or exc_type == "TimeoutError":
            return BenchmarkError(
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                message=f"Operation timed out: {exception}",
                exception=exception,
                context=context,
                recovery_strategy=RecoveryStrategy.RETRY
            )

        elif "out of memory" in exc_str or "oom" in exc_str or "cuda out of memory" in exc_str:
            return BenchmarkError(
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.HIGH,
                message=f"Out of memory: {exception}",
                exception=exception,
                context=context,
                recovery_strategy=RecoveryStrategy.SKIP
            )

        elif "model" in exc_str and ("load" in exc_str or "not found" in exc_str):
            return BenchmarkError(
                category=ErrorCategory.MODEL_LOADING,
                severity=ErrorSeverity.HIGH,
                message=f"Model loading failed: {exception}",
                exception=exception,
                context=context,
                recovery_strategy=RecoveryStrategy.SKIP
            )

        elif "connection" in exc_str or "network" in exc_str:
            return BenchmarkError(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                message=f"Network error: {exception}",
                exception=exception,
                context=context,
                recovery_strategy=RecoveryStrategy.RETRY
            )

        elif exc_type in ("ValueError", "ValidationError"):
            return BenchmarkError(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                message=f"Validation error: {exception}",
                exception=exception,
                context=context,
                recovery_strategy=RecoveryStrategy.SKIP
            )

        else:
            # Unknown error - default to inference category
            return BenchmarkError(
                category=ErrorCategory.INFERENCE,
                severity=ErrorSeverity.MEDIUM,
                message=f"Inference error: {exception}",
                exception=exception,
                context=context,
                recovery_strategy=RecoveryStrategy.RETRY
            )

    def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str = "operation",
        context: Optional[dict] = None,
        max_retries: Optional[int] = None
    ) -> tuple[Optional[T], Optional[BenchmarkError]]:
        """
        Execute operation with retry logic.

        Args:
            operation: Callable to execute
            operation_name: Name for logging
            context: Additional context
            max_retries: Override default max retries

        Returns:
            Tuple of (result, error). If successful, error is None.
        """
        max_attempts = (max_retries or self.max_retries) + 1
        context = context or {}
        context["operation_name"] = operation_name

        for attempt in range(max_attempts):
            try:
                # Check circuit breaker
                category = context.get("error_category", ErrorCategory.UNKNOWN)
                if self.enable_circuit_breaker and not self._get_circuit_breaker(category).is_allowed():
                    error = BenchmarkError(
                        category=category,
                        severity=ErrorSeverity.HIGH,
                        message=f"Circuit breaker OPEN for {category.value}",
                        context=context,
                        recovery_strategy=RecoveryStrategy.ABORT
                    )
                    logger.error(f"[{operation_name}] Circuit breaker blocking operation")
                    return None, error

                # Execute operation
                result = operation()

                # Success - record for circuit breaker
                if self.enable_circuit_breaker:
                    self._get_circuit_breaker(category).record_success()

                if attempt > 0:
                    logger.info(f"[{operation_name}] Succeeded after {attempt} retries")

                return result, None

            except Exception as e:
                # Classify error
                error = self.classify_error(e, context)
                error.retry_count = attempt

                # Log error
                logger.error(
                    f"[{operation_name}] Attempt {attempt + 1}/{max_attempts} failed: {error.message}"
                )
                if attempt == 0:
                    logger.debug(f"Full traceback: {traceback.format_exc()}")

                # Record for circuit breaker
                if self.enable_circuit_breaker:
                    self._get_circuit_breaker(error.category).record_failure()

                # Store in history
                self._error_history.append(error)

                # Check if should retry
                if attempt < max_attempts - 1 and error.recovery_strategy == RecoveryStrategy.RETRY:
                    delay = self.retry_delay * (self.backoff_multiplier ** attempt)
                    logger.info(f"[{operation_name}] Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue

                # No more retries or non-retryable error
                logger.error(f"[{operation_name}] Failed permanently: {error.message}")
                return None, error

        # Should not reach here
        return None, error

    def _get_circuit_breaker(self, category: ErrorCategory) -> CircuitBreaker:
        """Get or create circuit breaker for category."""
        if category not in self._circuit_breakers:
            self._circuit_breakers[category] = CircuitBreaker(
                name=f"CB-{category.value}",
                failure_threshold=5,
                recovery_timeout_seconds=60
            )
        return self._circuit_breakers[category]

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        for cb in self._circuit_breakers.values():
            cb.reset()
        logger.info("All circuit breakers reset")

    def get_error_summary(self) -> dict:
        """
        Get summary of errors encountered.

        Returns:
            Dict with error statistics
        """
        if not self._error_history:
            return {
                "total_errors": 0,
                "by_category": {},
                "by_severity": {}
            }

        by_category = {}
        by_severity = {}

        for error in self._error_history:
            # Count by category
            cat = error.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            # Count by severity
            sev = error.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total_errors": len(self._error_history),
            "by_category": by_category,
            "by_severity": by_severity,
            "circuit_breaker_states": {
                cat.value: cb.state
                for cat, cb in self._circuit_breakers.items()
            }
        }

    def clear_history(self) -> None:
        """Clear error history."""
        self._error_history.clear()
        logger.debug("Error history cleared")
