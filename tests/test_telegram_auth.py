from trainctl.integrations.telegram.commands import is_authorized


def test_telegram_authorization_is_deny_by_default() -> None:
    assert not is_authorized(123, set())
    assert not is_authorized(None, {123})
    assert not is_authorized(999, {123})
    assert is_authorized(123, {123})

