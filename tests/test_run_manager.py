from __future__ import annotations

import pytest

from trainctl.core.event_bus import EventBus
from trainctl.core.models import RunStatus
from trainctl.core.run_manager import InvalidRunTransition, RunManager
from trainctl.storage.sqlite import SQLiteStore


async def test_run_state_transitions(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "state.db")
    manager = RunManager(store, EventBus())
    run = manager.create("demo", ["python", "train.py"])
    assert run.status is RunStatus.CREATED
    await manager.starting(run)
    await manager.started(run, pid=1234, pgid=1234)
    assert run.status is RunStatus.RUNNING
    await manager.completed(run)
    assert store.get_run(run.run_id).status is RunStatus.COMPLETED  # type: ignore[union-attr]


def test_invalid_transition(tmp_path) -> None:
    manager = RunManager(SQLiteStore(tmp_path / "state.db"), EventBus())
    run = manager.create("demo", ["python"])
    with pytest.raises(InvalidRunTransition):
        manager.transition(run, RunStatus.COMPLETED)

