from __future__ import annotations

import sys

from trainctl.agent.supervisor import Supervisor
from trainctl.core.event_bus import EventBus
from trainctl.core.metric_store import MetricStore
from trainctl.core.models import RunStatus
from trainctl.core.run_manager import RunManager
from trainctl.integrations.generic import GenericTrainingAdapter
from trainctl.storage.sqlite import SQLiteStore


async def test_supervisor_captures_logs_metrics_and_completion(tmp_path) -> None:
    storage = SQLiteStore(tmp_path / "state.db")
    bus = EventBus()
    manager = RunManager(storage, bus)
    metrics = MetricStore(storage, bus)
    code = "for step in range(3): print(f'step={step} loss={1/(step+1)}', flush=True)"
    run = manager.create(
        "tiny", [sys.executable, "-u", "-c", code],
        log_path=tmp_path / "tiny.log",
    )
    supervisor = Supervisor(manager, metrics, GenericTrainingAdapter())
    result = await supervisor.supervise(run)

    assert result.status is RunStatus.COMPLETED
    assert result.exit_code == 0
    assert result.step == 2
    assert result.metrics["loss"] == 1 / 3
    assert len(metrics.history(run.run_id, "loss")) == 3
    assert "step=2" in (tmp_path / "tiny.log").read_text()
    assert len(supervisor.tail(run.run_id, 2)) == 2

