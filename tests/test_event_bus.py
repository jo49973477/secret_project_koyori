from __future__ import annotations

from trainctl.core.event_bus import EventBus
from trainctl.core.events import RunCompleted


async def test_event_bus_publishes_to_typed_subscribers() -> None:
    bus = EventBus()
    received: list[str] = []

    async def handler(event: RunCompleted) -> None:
        received.append(event.run_id)

    unsubscribe = bus.subscribe(RunCompleted, handler)
    await bus.publish(RunCompleted(run_id="run-1"))
    unsubscribe()
    await bus.publish(RunCompleted(run_id="run-2"))
    assert received == ["run-1"]


async def test_event_bus_isolates_failing_subscriber() -> None:
    bus = EventBus()
    received: list[str] = []

    def broken(event: RunCompleted) -> None:
        raise RuntimeError("boom")

    bus.subscribe(RunCompleted, broken)
    bus.subscribe(RunCompleted, lambda event: received.append(event.run_id))
    await bus.publish(RunCompleted(run_id="still-delivered"))
    assert received == ["still-delivered"]

