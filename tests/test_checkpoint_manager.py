from trainctl.core.checkpoint_manager import CheckpointManager
from trainctl.core.event_bus import EventBus
from trainctl.core.run_manager import RunManager
from trainctl.storage.sqlite import SQLiteStore


async def test_checkpoint_registration_and_lookup(tmp_path) -> None:
    storage = SQLiteStore(tmp_path / "checkpoints.db")
    bus = EventBus()
    run = RunManager(storage, bus).create("demo", ["python"])
    manager = CheckpointManager(storage, bus)
    first = await manager.register(run.run_id, tmp_path / "checkpoint-100", 100, {"score": 0.7})
    best = await manager.register(run.run_id, tmp_path / "checkpoint-200", 200, {"score": 0.9})
    duplicate = await manager.register(run.run_id, tmp_path / "checkpoint-200", 200)
    assert duplicate.checkpoint_id == best.checkpoint_id
    assert manager.latest(run.run_id).step == 200  # type: ignore[union-attr]
    assert manager.by_step(run.run_id, 100) == first
    assert manager.best(run.run_id, "score") == best

