from __future__ import annotations

from trainctl.core.models import MetricPoint


def metric_series(points: list[MetricPoint]) -> tuple[list[int], list[float]]:
    """Return plot-ready x/y data without imposing a plotting dependency."""
    x = [point.step if point.step is not None else index for index, point in enumerate(points)]
    return x, [point.value for point in points]

