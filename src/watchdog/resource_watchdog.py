"""
Resource Watchdog — monitors CPU and memory usage.

Triggers soft kill switch if resources are critically high,
which could indicate a memory leak or runaway computation.

Thresholds (conservative for Oracle Free Tier):
  - CPU    > 85% for 3 consecutive checks → alert
  - Memory > 80% → alert
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# Try to import psutil — optional dependency
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed — ResourceWatchdog will be limited")

CPU_ALERT_THRESHOLD = 85.0    # %
MEM_ALERT_THRESHOLD = 80.0    # %
CPU_CONSECUTIVE_LIMIT = 3     # checks in a row before alerting


class ResourceWatchdog:
    """
    Monitors system CPU and memory usage.
    Alerts via kill switch if thresholds are exceeded.
    """

    def __init__(
        self,
        interval_seconds: int = 30,
        kill_switch=None,
        cpu_threshold: float = CPU_ALERT_THRESHOLD,
        mem_threshold: float = MEM_ALERT_THRESHOLD,
    ) -> None:
        self._interval = interval_seconds
        self._kill_switch = kill_switch
        self._cpu_threshold = cpu_threshold
        self._mem_threshold = mem_threshold

        self._running = False
        self._task: asyncio.Task | None = None
        self._high_cpu_count = 0
        self._alert_count = 0

        # Latest readings
        self._last_cpu: float = 0.0
        self._last_mem: float = 0.0

    async def start(self) -> None:
        if self._running:
            return
        if not PSUTIL_AVAILABLE:
            logger.warning(
                "ResourceWatchdog: psutil unavailable — not starting. "
                "Install with: pip install psutil"
            )
            return
        self._running = True
        self._task = asyncio.create_task(
            self._loop(), name="resource_watchdog"
        )
        logger.info(
            "ResourceWatchdog started cpu_threshold=%.0f%% mem_threshold=%.0f%%",
            self._cpu_threshold, self._mem_threshold,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._interval)
            try:
                await self._check()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("ResourceWatchdog error")

    async def _check(self) -> None:
        # Run blocking psutil calls in thread pool
        loop = asyncio.get_running_loop()
        cpu = await loop.run_in_executor(
            None, lambda: psutil.cpu_percent(interval=1)
        )
        mem = await loop.run_in_executor(
            None, lambda: psutil.virtual_memory().percent
        )

        self._last_cpu = cpu
        self._last_mem = mem

        logger.debug("Resources cpu=%.1f%% mem=%.1f%%", cpu, mem)

        # CPU check
        if cpu > self._cpu_threshold:
            self._high_cpu_count += 1
            logger.warning(
                "High CPU cpu=%.1f%% consecutive=%d",
                cpu, self._high_cpu_count,
            )
            if self._high_cpu_count >= CPU_CONSECUTIVE_LIMIT:
                self._alert_count += 1
                logger.error(
                    "CPU ALERT: %.1f%% for %d consecutive checks",
                    cpu, self._high_cpu_count,
                )
                if self._kill_switch:
                    self._kill_switch.trigger_soft(
                        reason=f"high_cpu_{cpu:.0f}pct"
                    )
        else:
            self._high_cpu_count = 0

        # Memory check
        if mem > self._mem_threshold:
            self._alert_count += 1
            logger.error("MEMORY ALERT: %.1f%%", mem)
            if self._kill_switch:
                self._kill_switch.trigger_soft(
                    reason=f"high_memory_{mem:.0f}pct"
                )

    def status(self) -> dict:
        return {
            "running": self._running,
            "psutil_available": PSUTIL_AVAILABLE,
            "cpu_pct": self._last_cpu,
            "mem_pct": self._last_mem,
            "high_cpu_count": self._high_cpu_count,
            "alert_count": self._alert_count,
        }
