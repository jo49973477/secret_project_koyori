from trainctl.core.event_bus import EventBus
from trainctl.core.metric_store import MetricStore
from trainctl.core.run_manager import RunManager
from trainctl.storage.sqlite import SQLiteStore


async def test_metric_persistence_and_queries(tmp_path) -> None:
    storage = SQLiteStore(tmp_path / "metrics.db")
    run = RunManager(storage, EventBus()).create("demo", ["python"])
    metrics = MetricStore(storage)
    await metrics.record(run.run_id, "loss", 1.0, 1)
    await metrics.record(run.run_id, "loss", 0.5, 2)
    assert metrics.latest(run.run_id, "loss").value == 0.5  # type: ignore[union-attr]
    assert [point.value for point in metrics.history(run.run_id, "loss")] == [1.0, 0.5]
    assert [point.value for point in metrics.history(run.run_id, "loss", limit=1)] == [0.5]

