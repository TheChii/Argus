"""Rate limiting and retry utilities."""
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger("rate_limiter")


class RateLimiter:
    """Simple rate limiter ensuring min delay between requests."""
    def __init__(self, calls_per_second: float):
        self.min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 0.0
        self.last_call = 0.0

    def wait(self) -> None:
        """Blocks until the required interval has passed since the last call."""
        if self.min_interval <= 0:
            return
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_call = time.time()


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable:
    """Decorator to retry a function call with exponential backoff and jitter."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts. Last error: {e}")
                        raise
                    delay = base_delay * (backoff_factor ** (attempt - 1)) + random.uniform(0.1, 0.5)
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
        return wrapper
    return decorator
