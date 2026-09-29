from trainctl.core.events import MetricUpdated
from trainctl.integrations.generic import GenericTrainingAdapter


def test_parses_step_and_multiple_metrics() -> None:
    events = GenericTrainingAdapter().parse_line(
        "abc", "step=1200 loss=0.923 learning_rate=1e-4"
    )
    assert [(event.run_id, event.name, event.value, event.step) for event in events] == [
        ("abc", "loss", 0.923, 1200),
        ("abc", "learning_rate", 1e-4, 1200),
    ]


def test_parses_colon_and_aliases() -> None:
    events = GenericTrainingAdapter().parse_line("abc", "loss: 0.5 lr=0.0001")
    assert [(event.name, event.value) for event in events] == [
        ("loss", 0.5), ("learning_rate", 0.0001)
    ]
