from trainctl.core.event_bus import EventBus
from trainctl.core.events import RunFailed
from trainctl.integrations.notification import NotificationRouter


async def test_channels_deliver_independently():
    bus = EventBus()
    received = []

    class Adapter:
        def __init__(self, fail=False):
            self.fail = fail
        async def __call__(self, event):
            if self.fail:
                raise RuntimeError("channel failure")
            received.append(type(self).__name__ + event.run_id)
        async def close(self):
            pass

    router = NotificationRouter(bus, [Adapter(fail=True), Adapter()])
    await bus.publish(RunFailed(run_id="terminal", exit_code=2))
    assert received == ["Adapterterminal"]
    await router.close()


async def test_terminal_event_dispatch_is_awaited():
    bus = EventBus()
    received = []
    class Adapter:
        async def __call__(self, event):
            received.append(event.run_id)
        async def close(self):
            pass
    router = NotificationRouter(bus, [Adapter()])
    await bus.publish(RunFailed(run_id="before-shutdown", exit_code=1))
    assert received == ["before-shutdown"]
    await router.close()


async def test_core_works_with_notifications_disabled():
    bus = EventBus()
    router = NotificationRouter(bus, [])
    await bus.publish(RunFailed(run_id="no-channels", exit_code=1))
    await router.close()
