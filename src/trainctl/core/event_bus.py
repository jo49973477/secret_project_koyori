from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .events import Event

logger = logging.getLogger(__name__)
E = TypeVar("E", bound=Event)
EventHandler = Callable[[E], Awaitable[None] | None]


class EventBus:
    """Small in-process asynchronous publish/subscribe bus.

    Handlers are isolated: one failing subscriber is logged without preventing
    the remaining subscribers from receiving the event.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[EventHandler[Event]]] = defaultdict(list)

    def subscribe(self, event_type: type[E], handler: EventHandler[E]) -> Callable[[], None]:
        handlers = self._handlers[event_type]
        handlers.append(handler)  # type: ignore[arg-type]

        def unsubscribe() -> None:
            if handler in handlers:
                handlers.remove(handler)  # type: ignore[arg-type]

        return unsubscribe

    async def publish(self, event: Event) -> None:
        handlers: list[EventHandler[Event]] = []
        for event_type, registered in self._handlers.items():
            if isinstance(event, event_type):
                handlers.extend(registered)
        if not handlers:
            return
        await asyncio.gather(
            *(self._invoke(handler, event) for handler in tuple(handlers)),
            return_exceptions=False,
        )

    async def _invoke(self, handler: EventHandler[Event], event: Event) -> None:
        try:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("Event subscriber failed for %s", type(event).__name__)

