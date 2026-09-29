from __future__ import annotations

from trainctl.storage.base import StateStore

from .event_bus import EventBus
from .events import MetricUpdated
from .models import MetricPoint


class MetricStore:
    def __init__(self, store: StateStore, event_bus: EventBus | None = None) -> None:
        self.store = store
        self.event_bus = event_bus

    async def record(self, run_id: str, name: str, value: float, step: int | None = None) -> MetricPoint:
        point = MetricPoint(run_id=run_id, name=name, value=float(value), step=step)
        self.store.save_metric(point)
        if self.event_bus:
            await self.event_bus.publish(
                MetricUpdated(run_id=run_id, name=name, value=float(value), step=step)
            )
        return point

    def latest(self, run_id: str, name: str) -> MetricPoint | None:
        return self.store.latest_metric(run_id, name)

    def history(self, run_id: str, name: str, limit: int | None = None) -> list[MetricPoint]:
        return self.store.metric_history(run_id, name, limit)

