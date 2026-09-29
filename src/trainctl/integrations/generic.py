from __future__ import annotations

import re

from trainctl.core.events import Event, MetricUpdated
from trainctl.core.models import Checkpoint, Run

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_STEP_RE = re.compile(r"\b(?:global_)?step\s*[:=]\s*(\d+)\b", re.IGNORECASE)
_METRIC_RE = re.compile(
    rf"\b(?P<name>loss|learning_rate|lr|success_rate|validation_loss|val_loss)"
    rf"\s*[:=]\s*(?P<value>{_NUMBER})",
    re.IGNORECASE,
)
_NAMES = {"lr": "learning_rate", "val_loss": "validation_loss"}


class GenericTrainingAdapter:
    """Conservative parser for common key/value training output."""

    def parse_line(self, run_id: str, line: str) -> list[Event]:
        step_match = _STEP_RE.search(line)
        step = int(step_match.group(1)) if step_match else None
        events: list[Event] = []
        for match in _METRIC_RE.finditer(line):
            raw_name = match.group("name").lower()
            name = _NAMES.get(raw_name, raw_name)
            events.append(
                MetricUpdated(
                    run_id=run_id, name=name,
                    value=float(match.group("value")), step=step,
                )
            )
        return events

    def discover_checkpoints(self, run: Run) -> list[Checkpoint]:
        # Filesystem discovery is handled by the configurable CheckpointWatcher.
        return []

