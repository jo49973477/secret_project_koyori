from __future__ import annotations

import logging

import pytest

from trainctl.core.events import RunFailed, RunStarted
from trainctl.integrations.discord.formatter import format_event
from trainctl.integrations.discord.webhook import DiscordWebhookAdapter


class Response:
    def __init__(self, status_code=204, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload or {}

    def json(self):
        return self._payload


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


async def test_discord_success_and_mentions_are_suppressed():
    client = Client([Response()])
    adapter = DiscordWebhookAdapter("https://discord.com/api/webhooks/123/secret", client=client)
    await adapter(RunStarted(run_id="r1", name="demo", pid=42))
    assert client.calls[0][1]["json"]["allowed_mentions"] == {"parse": []}
    assert client.calls[0][1]["json"]["embeds"][0]["color"] == 0x3498DB


async def test_429_honors_retry_after(monkeypatch):
    delays = []
    async def sleep(delay):
        delays.append(delay)
    monkeypatch.setattr("trainctl.integrations.discord.webhook.asyncio.sleep", sleep)
    client = Client([Response(429, {"Retry-After": "0.4"}), Response()])
    adapter = DiscordWebhookAdapter("https://discord.com/api/webhooks/123/secret", client=client)
    await adapter.send_message("hello")
    assert len(client.calls) == 2
    assert delays == [0.4]


async def test_5xx_retries_and_invalid_webhook_does_not_retry(caplog):
    async def no_sleep(_):
        pass
    client = Client([Response(500), Response()])
    adapter = DiscordWebhookAdapter("https://discord.com/api/webhooks/123/secret", client=client)
    # Keep retry tests fast and deterministic.
    from unittest.mock import patch
    with patch("trainctl.integrations.discord.webhook.asyncio.sleep", no_sleep):
        await adapter.send_message("hello")
    assert len(client.calls) == 2
    client = Client([Response(404)])
    adapter = DiscordWebhookAdapter("https://discord.com/api/webhooks/123/secret", client=client)
    with caplog.at_level(logging.ERROR):
        await adapter.send_message("hello")
    assert len(client.calls) == 1
    assert "secret" not in caplog.text
    assert "discord.com/api" not in caplog.text


async def test_timeout_is_logged_without_url(caplog):
    client = Client([TimeoutError("timeout")])
    adapter = DiscordWebhookAdapter("https://discord.com/api/webhooks/123/secret", client=client, retries=0)
    with caplog.at_level(logging.WARNING):
        await adapter.send_message("hello")
    assert "TimeoutException" in caplog.text
    assert "secret" not in caplog.text


async def test_png_attachment_is_multipart(tmp_path):
    png = tmp_path / "loss.png"
    png.write_bytes(b"fake-png")
    client = Client([Response()])
    adapter = DiscordWebhookAdapter("https://discord.com/api/webhooks/123/secret", client=client)
    await adapter.send_file(png, "Loss curve")
    kwargs = client.calls[0][1]
    assert kwargs["files"]["files[0]"][0] == "loss.png"
    assert kwargs["files"]["files[0]"][2] == "image/png"
    assert "payload_json" in kwargs["data"]


def test_formatter_truncates_large_logs():
    message = format_event(RunFailed(run_id="r", exit_code=1, tail=("x" * 2000,)))
    assert message
    assert len(message.payload()["embeds"][0]["fields"][-1]["value"]) <= 1024


def test_malformed_webhook_url_is_rejected_without_echoing_secret():
    with pytest.raises(ValueError, match="HTTPS Discord webhook URL"):
        DiscordWebhookAdapter("https://discord.com/api/webhooks/private-token")
