from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from trainctl.core.event_bus import EventBus
from trainctl.core.events import (
    CheckpointSaved, DiskLow, EvalCompleted, EvalFailed, Event, GpuWarning,
    RunCompleted, RunFailed, RunStarted,
)
from trainctl.integrations.base import NotificationAdapter

logger = logging.getLogger(__name__)
NOTIFICATION_EVENTS = (
    RunStarted, RunCompleted, RunFailed, CheckpointSaved,
    EvalCompleted, EvalFailed, DiskLow, GpuWarning,
)


class NotificationRouter:
    """Dispatch events to independent adapters and isolate channel failures."""

    def __init__(self, event_bus: EventBus, adapters: Iterable[NotificationAdapter]) -> None:
        self.adapters = tuple(adapters)
        self._unsubscribers = [event_bus.subscribe(kind, self.dispatch) for kind in NOTIFICATION_EVENTS]

    async def dispatch(self, event: Event) -> None:
        async def deliver(adapter: NotificationAdapter) -> None:
            try:
                await adapter(event)
            except Exception:
                logger.exception("Notification adapter %s failed for %s", type(adapter).__name__, type(event).__name__)

        if self.adapters:
            await asyncio.gather(*(deliver(adapter) for adapter in self.adapters))

    async def close(self, timeout: float = 6.0) -> None:
        """Unsubscribe and close transports within a bounded shutdown window."""
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        async def close_adapter(adapter: NotificationAdapter) -> None:
            try:
                await adapter.close()
            except Exception:
                logger.exception("Could not close notification adapter %s", type(adapter).__name__)

        closers = [close_adapter(adapter) for adapter in self.adapters]
        if not closers:
            return
        try:
            await asyncio.wait_for(asyncio.gather(*closers, return_exceptions=True), timeout=timeout)
        except TimeoutError:
            logger.warning("Timed out while closing notification adapters")
