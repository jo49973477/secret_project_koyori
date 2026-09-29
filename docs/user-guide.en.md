# trainctl User Guide (English)

## What is trainctl?

On a remote GPU server, a typical workflow is to SSH in, start a long training
job, disconnect, wonder whether it crashed, reconnect to inspect logs and GPU
usage, and search manually for checkpoints. `trainctl` supervises an ordinary
local command, captures its output, records run state and parsed metrics in
SQLite, watches a checkpoint directory, and can notify you when important
events occur.

Experiment tracking records and explores metrics, artifacts, and comparisons.
Experiment orchestration starts and supervises work and reports its lifecycle.
trainctl is focused on the latter and offers only lightweight local metric
history. It is not intended to replace WandB or TensorBoard.

## Installation

Python 3.10–3.13 is declared supported. The current implementation targets
Linux/Unix-style training servers; process groups use POSIX features when
available. CPU-only systems work. NVIDIA monitoring is optional and uses NVML
when installed, with an `nvidia-smi` fallback. Telegram and Discord are
optional integrations (`telegram` and `discord` extras); the core needs neither.

From a repository checkout:

```bash
git clone <repository-url>
cd <repository-directory>
uv sync
uv pip install -e .
```

Optional dependencies can be installed with `uv sync --extra telegram --extra
discord --extra nvml`, or with `pip install '.[telegram,discord,nvml]'` from a
checkout. The package is not currently published on PyPI, so the command
`pip install trainctl` is not yet a supported installation method.

## Telegram setup

1. In Telegram, open **@BotFather**, send `/newbot`, and follow its prompts.
2. Copy the Bot Token. Keep it secret.
3. Find your numeric user ID using a trusted ID lookup bot or your Telegram
   client tooling. Only IDs in the allowlist can use the bot.
4. Install the optional client and export settings:

```bash
uv pip install -e '.[telegram]'
export TRAINCTL_TELEGRAM_TOKEN='replace-with-your-bot-token'
export TRAINCTL_TELEGRAM_ALLOWED_USERS='123456789'
```

5. Start the interactive frontend with `trainctl daemon`. To receive lifecycle
   events for a process launched by `trainctl run`, configure the same settings
   in the shell that launches that command.
6. Start a run and check Telegram for its start and terminal notification.

Implemented bot commands are `/status`, `/gpu`, `/tail 20`, and `/disk`.
`/plot`, `/eval`, `/best`, `/compare`, `/pause`, and `/resume` are registered
placeholders and return a planned-feature message.

## Discord setup

Discord webhooks send messages to one channel; they do not need a Discord Bot
Token and do not provide interactive slash commands.

1. Open Discord and select your server and target text channel.
2. Open the channel's **Edit Channel** settings.
3. Select **Integrations**, then **Webhooks**.
4. Choose **New Webhook**, give it a name, and select the target channel.
5. Copy the webhook URL and keep it private.
6. Configure it and enable the channel:

```bash
export TRAINCTL_DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'
```

```toml
[notifications.discord]
enabled = true
```

When the URL is supplied by the environment and TOML has no explicit `enabled`
setting, Discord is enabled automatically. A future interactive Discord
control interface would need a separate Discord Bot integration.

## Training experiments

`trainctl run` passes arguments after `--` directly to a child process (no
shell is involved). For example:

```bash
trainctl run --name experiment-01 -- python train.py
```

For a two-GPU distributed job:

```bash
trainctl run \
  --name groot-finetune \
  -- \
  torchrun --nproc_per_node=2 train.py
```

Checkpoint and step options are available:

```bash
trainctl run --name experiment-01 \
  --checkpoint-dir ./outputs --total-steps 2000 \
  -- python train.py
```

Use `trainctl status` to inspect the latest stored run and `trainctl config
show` to inspect effective non-secret settings.

## Working with tmux

Environment variables are inherited when a process starts. Load secrets before
creating a tmux session:

```bash
source ~/.config/trainctl/secrets.env
tmux new -s training
```

If the tmux server/session was started earlier, source the file inside that
session too:

```bash
source ~/.config/trainctl/secrets.env
```

Check presence without printing the secret:

```bash
test -n "$TRAINCTL_DISCORD_WEBHOOK_URL" && echo "Discord configured"
```

You may run `trainctl run ...` in a tmux session when you want an attached,
recoverable terminal. `trainctl run` itself supervises the child and writes
logs, so do not start a second `trainctl run` for the same training process.
Start `trainctl daemon` only when you need Telegram's interactive commands; it
is not required for run supervision or Discord webhook delivery. One Telegram
daemon per account/server is generally enough.

## Notifications

The current EventBus defines and the notification router supports: training
started, completed, and failed; checkpoint saved; evaluation completed and
failed; disk low; and GPU warning. Run and checkpoint events are currently
produced by implemented paths. Evaluation events have event types but no
evaluation manager produces them yet. Disk/GPU warning event production is
also not currently wired to periodic policies.

Set `[notifications.telegram].enabled = true` or
`[notifications.discord].enabled = true` to enable channels. Telegram also
requires a token and a non-empty user allowlist. Discord requires a webhook
URL. If one channel fails, delivery to the other is still attempted.

## Configuration reference

The default configuration file is `~/.config/trainctl/config.toml`. Environment
variables override TOML. Legacy `[telegram]` remains accepted for compatibility.

| Configuration key | Environment variable | Default | Required? | Description |
| --- | --- | --- | --- | --- |
| `notifications.telegram.enabled` | — | `false` | Optional | Enable Telegram notifications |
| `notifications.telegram.allowed_users` | `TRAINCTL_TELEGRAM_ALLOWED_USERS` | `[]` | Required for Telegram | Comma-separated numeric user IDs in the environment |
| Telegram token | `TRAINCTL_TELEGRAM_TOKEN` | unset | Required for Telegram | Bot credential; environment only |
| `notifications.discord.enabled` | — | `false` | Optional | Enable Discord notifications |
| `notifications.discord.webhook_url` | `TRAINCTL_DISCORD_WEBHOOK_URL`, then `DISCORD_WEBHOOK_URL` | unset | Required for Discord | Webhook URL; environment values override TOML, preferred variable wins |
| `storage.database` | `TRAINCTL_DATABASE` | `~/.local/share/trainctl/trainctl.db` | Optional | SQLite database path |
| `logging.directory` | `TRAINCTL_LOG_DIR` | `~/.local/share/trainctl/logs` | Optional | Child process log directory |
| `logging.level` | — | `INFO` | Optional | Python logging level |
| `monitoring.disk_warning_percent` | — | `90.0` | Optional | Threshold setting; periodic warning policy is not wired yet |
| `monitoring.disk_paths` | — | log and database directories | Optional | Paths included in resource queries |
| `checkpoints.poll_interval_seconds` | — | `5.0` | Optional | Checkpoint polling interval |
| `checkpoints.pattern` | — | `(?:checkpoint|ckpt)[-_]?(?P<step>\d+)` | Optional | Filename pattern for checkpoint steps |
| — | `TRAINCTL_CONFIG` | default file above | Optional | Alternate TOML file |
| — | `TRAINCTL_HOME` | `~/.local/share/trainctl` | Optional | Base directory used for defaults |

Example complete file:

```toml
[notifications.telegram]
enabled = true
allowed_users = [123456789]

[notifications.discord]
enabled = true

[storage]
database = "~/.local/share/trainctl/trainctl.db"

[logging]
directory = "~/.local/share/trainctl/logs"
level = "INFO"

[monitoring]
disk_warning_percent = 90
disk_paths = ["~/datasets", "~/checkpoints"]

[checkpoints]
poll_interval_seconds = 5
pattern = '(?:checkpoint|ckpt)[-_]?(?P<step>\d+)'
```

Put tokens and webhook URLs in the process environment, not TOML. `trainctl
config show` reports only whether credentials are configured.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Telegram notification not received | Confirm the run command shell has the token and allowlist; install `.[telegram]`; check `trainctl config show`; message the bot once so Telegram can open the chat. |
| Discord webhook not configured | Set `TRAINCTL_DISCORD_WEBHOOK_URL`, source the environment file, set `[notifications.discord].enabled = true`, then run `trainctl config show`. |
| Discord HTTP 429 | The adapter respects `Retry-After` and makes bounded retries. Persistent rate limits mean reduce event volume or use a dedicated channel. |
| Discord notification failure | Check logs for HTTP status or transport exception. 401/403/404 usually means the webhook was revoked or copied incorrectly. Logs never include the URL. |
| Training exited without a notification | Run terminal events are sent synchronously before `trainctl run` exits. Check channel configuration in that exact shell and inspect the run with `trainctl status`. |
| Environment variables missing inside tmux | Run `source ~/.config/trainctl/secrets.env` inside the session and verify with `test -n "$TRAINCTL_DISCORD_WEBHOOK_URL" && echo configured`. |
| GPU information unavailable | Install NVIDIA drivers and `nvidia-smi`; optionally install `nvidia-ml-py`. CPU-only operation is supported. |
| Checkpoint not detected | Set `--checkpoint-dir`, ensure filenames match the configured pattern, and allow at least one polling interval. |
| Daemon already running | Avoid launching duplicate polling bots for the same Telegram token. Stop the old daemon or reuse it. |
| No active training experiment | `trainctl status` displays the latest stored run, including completed runs. Start a run with `trainctl run --name ... -- python train.py`. |

Useful diagnostics:

```bash
trainctl --help
trainctl config show
trainctl status
test -n "$TRAINCTL_TELEGRAM_TOKEN" && echo "Telegram token configured"
test -n "$TRAINCTL_DISCORD_WEBHOOK_URL" && echo "Discord configured"
```

## Architecture overview

```text
                    trainctl
                       |
                 Orchestrator
                       |
                    EventBus
                       |
               NotificationRouter
                /           \\
           Telegram        Discord
```

The core publishes typed events. A small router calls independent adapters;
the Telegram library and Discord HTTP client stay at the integration edges.
This keeps the supervisor usable without either optional package and prevents
a remote service outage from changing the training process result.

## Security

Keep Telegram tokens and Discord webhook URLs in environment variables or an
untracked secret file with restrictive permissions (for example `chmod 600
~/.config/trainctl/secrets.env`). Telegram commands deny users outside the
numeric allowlist. Discord payloads suppress mention parsing by default.
trainctl does not support arbitrary remote shell execution. Never commit `.env`
or secret files; `.env.example` contains placeholders only.

## Known limitations and roadmap

**Currently implemented:** local command supervision; SQLite run and metric
records; generic metric parsing; checkpoint polling; started/completed/failed
and checkpoint events; outgoing Telegram and Discord run notifications;
Telegram status/resource commands.

**Partially implemented:** typed evaluation, disk, and GPU events and formatters
exist, but corresponding event producers/policies are not wired. Cross-process
delivery from the Telegram daemon to separately launched CLI processes is not
implemented. Notification retries are in-memory and bounded, not durable.

**Planned:** persistent event outbox and daemon recovery, periodic disk/GPU
warning policy, evaluation execution and commands, metric plots, cooperative
pause/resume, and an interactive Discord Bot integration. Discord webhooks are
outgoing only; slash commands are not implemented.
