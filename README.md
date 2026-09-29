# trainctl

Lightweight, framework-agnostic experiment orchestration for deep-learning
jobs on remote servers. It supervises ordinary commands, records lifecycle and
metrics locally, captures logs, watches checkpoints, and sends notifications.

## Documentation

- [English User Guide](docs/user-guide.en.md)
- [한국어 사용 가이드](docs/user-guide.ko.md)

## Architecture

```text
                    trainctl
                       │
                 Orchestrator
                       │
                    EventBus
                       │
              NotificationRouter
                /           \\
           Telegram        Discord
```

Telegram and Discord are independent optional channels. Either can be enabled
alone, both can receive the same event, or notifications can be disabled. The
core supervisor remains usable without either integration.

> **Project status:** early alpha. The local supervisor, persistence, generic
> metric parsing, resource queries, Telegram commands, and outgoing Discord
> webhook notifications are working.
> The package is not yet published to PyPI.

## Motivation

A familiar workflow is: SSH into a GPU server, start a multi-hour experiment,
leave the computer, wonder whether it crashed, SSH back in, run `nvidia-smi`,
tail logs, look for checkpoints, and launch evaluations manually. `trainctl`
aims to make that loop observable from a safe remote UI without imposing a
cloud service, Docker, WandB, or changes to the training script.

## Features

Implemented:

- `trainctl run -- <command>` with async, shell-free child-process supervision
- run PID/PGID, command, timestamps, status, exit code, and paths in SQLite
- combined stdout/stderr log files and a 200-line in-memory ring buffer
- generic `step`, `loss`, learning-rate, validation-loss, and success-rate parsing
- persistent metric histories and latest values
- configurable checkpoint polling and checkpoint registry
- typed in-process event bus and lifecycle/checkpoint notifications
- independent Telegram and Discord webhook notification adapters
- NVML GPU monitoring with `nvidia-smi` fallback; graceful CPU-only behavior
- RAM and configured-path disk usage
- Telegram allowlist plus `/status`, `/gpu`, `/tail [N]`, and `/disk`
- stale-PID reconciliation after restart

Planned:

- persistent cross-process event delivery and richer daemon recovery
- `/plot`, `/best`, `/eval`, `/compare`, and evaluation execution
- cooperative distributed-training pause/resume
- multi-run selection and additional remote frontends
- interactive Discord Bot commands (webhooks are outgoing only)
- periodic GPU-failure and disk-warning policies

Telegram handlers never call a shell or manipulate training processes
directly. They authorize callers and invoke typed services. See
[the detailed design](docs/architecture.md).

## Installation

From a clone:

```bash
git clone <repository-url>
cd trainctl
uv sync
uv pip install -e .
```

Install optional Telegram, Discord, and NVML support with:

```bash
uv sync --extra telegram --extra discord --extra nvml
```

Python 3.10 or newer is required.

## Quick start

Run any argv-based training command; arguments after `--` are passed directly
without a shell:

```bash
trainctl run --name example -- python train.py

trainctl run \
  --name groot-univtac \
  --checkpoint-dir ./outputs \
  --total-steps 2000 \
  -- \
  torchrun --nproc_per_node=2 train.py --config config.yaml
```

Inspect the latest persisted run:

```bash
trainctl status
trainctl config show
```

The repository includes a smoke-test program:

```bash
uv run trainctl run --name demo -- python examples/dummy_train.py
```

## Telegram setup and usage

Install the optional extra, create a bot with Telegram's BotFather, then export
the token and a comma-separated allowlist of numeric Telegram user IDs:

```bash
export TRAINCTL_TELEGRAM_TOKEN='replace-with-real-token'
export TRAINCTL_TELEGRAM_ALLOWED_USERS='123456789'
trainctl daemon
```

Available commands:

```text
/status       latest persisted run and progress
/gpu          GPU and system RAM summary
/tail 20      latest training log lines (maximum 200)
/disk         configured filesystem usage
```

`/plot`, `/eval`, `/best`, `/compare`, `/pause`, and `/resume` are registered as
explicit placeholders. They do not accept or execute arbitrary commands.

When `trainctl run` has Telegram credentials and the optional extra installed,
its event subscriber sends started, completed, failed, and checkpoint messages.
The daemon handles interactive commands. Cross-process event queuing is a known
alpha limitation; see [the roadmap](docs/roadmap.md).

## Configuration

Non-secret settings can live in `~/.config/trainctl/config.toml`; environment
variables override TOML values. The token is accepted only from the environment.
See [configuration documentation](docs/configuration.md) and `.env.example`.

## Security

- There is **no arbitrary remote shell execution**.
- Telegram access is denied unless the numeric user ID is allowlisted.
- The token belongs in `TRAINCTL_TELEGRAM_TOKEN`, never source control.
- Local training and evaluator commands are argv lists and do not use `shell=True`.
- Evaluation templates will come only from trusted local configuration.
- Low-level signals are not exposed through Telegram in this release.

## Roadmap

### v0.1

`trainctl run`, notifications, `/status`, `/gpu`, `/tail`, lifecycle events, and
checkpoint detection. Most of this foundation is implemented in the current alpha.

### v0.2

Metric plotting, scheduled disk warnings, `/plot`, and `/best`.

### v0.3

Trusted evaluation job execution, `/eval`, and `/compare`.

### v0.4

Cooperative `/pause` and `/resume`, multiple active experiments, multiple
servers, and additional frontends.

## Development

```bash
uv sync
uv run pytest
uv run trainctl --help
uv build
```

See [development notes](docs/development.md). Contributions should keep the core
framework-neutral and Telegram-independent.
