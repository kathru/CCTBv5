"""
Retry policy with exponential back-off + jitter.

Why jitter? Without it, multiple orders failing simultaneously
will all retry at the same interval — causing a thundering herd
that hammers the exchange and can trigger rate limits or duplicate
order submissions.
"""

import asyncio
import logging
import random

logger = logging.getLogger(__name__)


class RetryPolicy:
    """
    Exponential back-off with full jitter.

    Formula: sleep = random(0, min(cap, base * 2^attempt))

    Defaults:
      - max_attempts : 4   (1 initial + 3 retries)
      - base_delay   : 1s
      - cap          : 30s (never wait more than this)
    """

    def __init__(
        self,
        max_attempts: int = 4,
        base_delay: float = 1.0,
        cap: float = 30.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.cap = cap

    def should_retry(self, attempt: int) -> bool:
        """True if there are attempts remaining."""
        return attempt < self.max_attempts

    def delay(self, attempt: int) -> float:
        """Return jittered delay in seconds for this attempt."""
        ceiling = min(self.cap, self.base_delay * (2 ** attempt))
        return random.uniform(0, ceiling)

    async def wait(self, attempt: int) -> None:
        """Async sleep with calculated jitter delay."""
        seconds = self.delay(attempt)
        logger.debug("Retry attempt=%d waiting=%.2fs", attempt, seconds)
        await asyncio.sleep(seconds)
