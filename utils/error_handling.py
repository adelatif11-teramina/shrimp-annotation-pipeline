"""
Error handling utilities with retry logic and circuit breaker pattern
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Optional, Callable, Any, Dict, Type
from functools import wraps
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Categorize errors for proper handling"""
    TRANSIENT = "transient"  # Can retry
    PERMANENT = "permanent"  # Don't retry
    RATE_LIMIT = "rate_limit"  # Backoff required
    AUTHENTICATION = "authentication"  # Need new credentials
    VALIDATION = "validation"  # Bad input
    SYSTEM = "system"  # Internal error

@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff"""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random())
        
        return delay

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.success_count = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try resetting"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:  # Need 2 successes to fully close
                self.state = CircuitBreakerState.CLOSED
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} is now CLOSED")
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} is now OPEN (half-open test failed)")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(
                f"Circuit breaker {self.name} is now OPEN "
                f"(failures: {self.failure_count})"
            )

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass

def categorize_error(error: Exception) -> ErrorCategory:
    """Categorize an error for proper handling"""
    error_type = type(error).__name__
    error_msg = str(error).lower()
    
    # Rate limiting errors
    if "rate limit" in error_msg or "too many requests" in error_msg:
        return ErrorCategory.RATE_LIMIT
    
    # Authentication errors
    if any(term in error_msg for term in ["unauthorized", "forbidden", "invalid token", "authentication"]):
        return ErrorCategory.AUTHENTICATION
    
    # Validation errors
    if any(term in error_msg for term in ["invalid", "validation", "bad request", "missing required"]):
        return ErrorCategory.VALIDATION
    
    # Transient errors (can retry)
    transient_errors = [
        "timeout", "connection", "temporary", "unavailable",
        "service unavailable", "gateway timeout", "network"
    ]
    if any(term in error_msg for term in transient_errors):
        return ErrorCategory.TRANSIENT
    
    # HTTP status codes
    if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
        status = error.response.status_code
        if status in [429, 503, 504]:  # Rate limit or temporary unavailability
            return ErrorCategory.RATE_LIMIT if status == 429 else ErrorCategory.TRANSIENT
        elif status in [401, 403]:
            return ErrorCategory.AUTHENTICATION
        elif status in [400, 422]:
            return ErrorCategory.VALIDATION
        elif status >= 500:
            return ErrorCategory.SYSTEM
    
    # Default to permanent error (don't retry)
    return ErrorCategory.PERMANENT

def retry_with_backoff(
    func: Optional[Callable] = None,
    *,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable] = None,
    on_failure: Optional[Callable] = None
):
    """
    Decorator for retry logic with exponential backoff
    
    Usage:
        @retry_with_backoff(config=RetryConfig(max_attempts=5))
        async def fetch_data():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(f):
        @wraps(f)
        async def async_wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await f(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    category = categorize_error(e)
                    
                    # Don't retry permanent errors
                    if category == ErrorCategory.PERMANENT:
                        logger.error(f"Permanent error, not retrying: {e}")
                        raise
                    
                    # Don't retry authentication errors
                    if category == ErrorCategory.AUTHENTICATION:
                        logger.error(f"Authentication error, not retrying: {e}")
                        raise
                    
                    # Don't retry validation errors
                    if category == ErrorCategory.VALIDATION:
                        logger.error(f"Validation error, not retrying: {e}")
                        raise
                    
                    # Calculate delay
                    if attempt < config.max_attempts - 1:
                        delay = config.get_delay(attempt)
                        
                        # Longer delay for rate limits
                        if category == ErrorCategory.RATE_LIMIT:
                            delay = max(delay, 30.0)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        
                        if on_retry:
                            on_retry(attempt, e, delay)
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {config.max_attempts} attempts failed")
                        if on_failure:
                            on_failure(last_error)
            
            raise last_error
        
        @wraps(f)
        def sync_wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(config.max_attempts):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    category = categorize_error(e)
                    
                    # Don't retry certain error categories
                    if category in [ErrorCategory.PERMANENT, ErrorCategory.AUTHENTICATION, ErrorCategory.VALIDATION]:
                        logger.error(f"{category.value} error, not retrying: {e}")
                        raise
                    
                    if attempt < config.max_attempts - 1:
                        delay = config.get_delay(attempt)
                        
                        if category == ErrorCategory.RATE_LIMIT:
                            delay = max(delay, 30.0)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        
                        if on_retry:
                            on_retry(attempt, e, delay)
                        
                        time.sleep(delay)
                    else:
                        logger.error(f"All {config.max_attempts} attempts failed")
                        if on_failure:
                            on_failure(last_error)
            
            raise last_error
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        else:
            return sync_wrapper
    
    # Handle direct decoration
    if func is not None:
        return decorator(func)
    
    return decorator

class ErrorHandler:
    """Centralized error handler for the application"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def get_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
        return self.circuit_breakers[name]
    
    def handle_api_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API errors with proper response"""
        category = categorize_error(error)
        
        # Log the error with context
        logger.error(
            f"API error in {context.get('endpoint', 'unknown')}: {error}",
            extra={"context": context, "category": category.value}
        )
        
        # Prepare error response
        error_response = {
            "error": True,
            "category": category.value,
            "message": self._get_user_friendly_message(error, category),
            "details": None
        }
        
        # Add details for non-production environments
        from config.settings import settings
        if not settings.is_production:
            error_response["details"] = {
                "type": type(error).__name__,
                "original_message": str(error),
                "context": context
            }
        
        # Determine HTTP status code
        status_code = self._get_status_code(category, error)
        
        return {
            "status_code": status_code,
            "response": error_response
        }
    
    def _get_user_friendly_message(self, error: Exception, category: ErrorCategory) -> str:
        """Get user-friendly error message"""
        messages = {
            ErrorCategory.TRANSIENT: "Service temporarily unavailable. Please try again.",
            ErrorCategory.RATE_LIMIT: "Too many requests. Please slow down.",
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please check your credentials.",
            ErrorCategory.VALIDATION: "Invalid request. Please check your input.",
            ErrorCategory.PERMANENT: "An error occurred. Please contact support if this persists.",
            ErrorCategory.SYSTEM: "Internal system error. Our team has been notified."
        }
        return messages.get(category, "An unexpected error occurred.")
    
    def _get_status_code(self, category: ErrorCategory, error: Exception) -> int:
        """Get appropriate HTTP status code"""
        if hasattr(error, 'status_code'):
            return error.status_code
        
        status_map = {
            ErrorCategory.TRANSIENT: 503,
            ErrorCategory.RATE_LIMIT: 429,
            ErrorCategory.AUTHENTICATION: 401,
            ErrorCategory.VALIDATION: 400,
            ErrorCategory.PERMANENT: 500,
            ErrorCategory.SYSTEM: 500
        }
        return status_map.get(category, 500)

# Global error handler instance
error_handler = ErrorHandler()

# Export commonly used functions
__all__ = [
    'ErrorCategory',
    'RetryConfig',
    'CircuitBreaker',
    'CircuitBreakerOpenError',
    'categorize_error',
    'retry_with_backoff',
    'ErrorHandler',
    'error_handler'
]