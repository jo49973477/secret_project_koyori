from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path

from trainctl.core.models import EvalJob
from trainctl.storage.base import StateStore


class EvalManager:
    """Creates trusted, local-config-driven evaluation jobs.

    Job execution is deliberately deferred to a later release. Commands must
    originate from local configuration, never free-form Telegram input.
    """

    def __init__(self, store: StateStore, command_templates: Mapping[str, list[str]] | None = None) -> None:
        self.store = store
        self.command_templates = dict(command_templates or {})

    def create(self, kind: str, run_id: str, checkpoint_path: Path) -> EvalJob:
        if kind not in self.command_templates:
            raise KeyError(f"No trusted evaluator configured for {kind!r}")
        command = [part.replace("{checkpoint}", str(checkpoint_path)) for part in self.command_templates[kind]]
        job = EvalJob(
            job_id=uuid.uuid4().hex[:12], run_id=run_id,
            checkpoint_path=checkpoint_path, command=command,
        )
        self.store.save_eval_job(job)
        return job

