from trainctl.config import load_config
from trainctl.core.event_bus import EventBus
from trainctl.cli import _enable_run_notifications
import logging


def test_discord_environment_precedence(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[notifications.discord]\nenabled = true\nwebhook_url = "https://discord.com/api/webhooks/toml"\n')
    monkeypatch.setenv("TRAINCTL_DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/primary")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/fallback")
    config = load_config(path)
    assert config.discord.enabled
    assert config.discord.webhook_url.endswith("primary")


def test_discord_legacy_env_fallback_and_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAINCTL_DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/fallback")
    config = load_config(tmp_path / "absent.toml")
    assert config.discord.enabled
    assert config.discord.webhook_url.endswith("fallback")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL")
    config = load_config(tmp_path / "absent.toml")
    assert not config.discord.enabled
    assert config.discord.webhook_url is None


def test_missing_discord_url_disables_channel_with_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("TRAINCTL_DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    path = tmp_path / "config.toml"
    path.write_text('[notifications.discord]\nenabled = true\n')
    config = load_config(path)
    with caplog.at_level(logging.WARNING):
        router = _enable_run_notifications(config, EventBus())
    assert router.adapters == ()
    assert "webhook URL is missing" in caplog.text
