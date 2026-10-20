# trainctl Library Usage Dictionary

This document is a practical quick-reference guide for using trainctl in real
deep-learning workflows.

## 1. What trainctl does

trainctl supervises an ordinary training command from the outside.

```text
Before:
python train.py

With trainctl:
trainctl run -- python train.py
```

It can observe:

- run lifecycle (started/completed/failed)
- stdout/stderr logs
- common metrics such as step/loss
- checkpoint creation
- GPU/RAM/disk status
- Telegram and Discord notifications

The training script does not need to import trainctl.

## 2. Installation

Base package:

```bash
pip install trainctl
```

Telegram support:

```bash
pip install 'trainctl[telegram]'
```

Discord webhook support:

```bash
pip install 'trainctl[discord]'
```

Both:

```bash
pip install 'trainctl[telegram,discord]'
```

Add NVML GPU support:

```bash
pip install 'trainctl[telegram,discord,nvml]'
```

Development checkout:

```bash
uv sync --extra telegram --extra discord --extra nvml
uv pip install -e .
```

## 3. Basic command

```bash
trainctl run --name my-experiment -- python train.py
```

Everything after `--` is the original training command.

For torchrun:

```bash
trainctl run \
  --name groot-finetune \
  --total-steps 2000 \
  -- \
  torchrun --nproc_per_node=2 train.py --config config.yaml
```

## 4. Checkpoint monitoring

Pass the directory in which your training job creates checkpoints:

```bash
trainctl run \
  --name my-experiment \
  --checkpoint-dir ./outputs \
  --total-steps 2000 \
  -- \
  python train.py
```

Default checkpoint names include patterns such as:

```text
checkpoint-1000
checkpoint_1000
ckpt-1000
ckpt_1000
checkpoint-1000.pt
```

Detected checkpoints can generate Telegram/Discord notifications.

## 5. Metric logging

The generic parser understands simple key/value stdout:

```text
step=100 loss=0.823 learning_rate=0.0001
```

Recognized fields include:

```text
step
loss
learning_rate
lr
validation_loss
val_loss
success_rate
```

A normal Python print is sufficient:

```python
print(
    f"step={step} loss={loss.item():.6f} learning_rate={lr:.8g}",
    flush=True,
)
```

## 6. Telegram

Fish:

```fish
set -gx TRAINCTL_TELEGRAM_TOKEN "YOUR_BOT_TOKEN"
set -gx TRAINCTL_TELEGRAM_ALLOWED_USERS "123456789"
```

Bash/zsh:

```bash
export TRAINCTL_TELEGRAM_TOKEN="YOUR_BOT_TOKEN"
export TRAINCTL_TELEGRAM_ALLOWED_USERS="123456789"
```

Multiple user IDs are comma-separated.

For interactive commands, run:

```bash
trainctl daemon
```

Current commands:

```text
/status
/gpu
/tail 20
/disk
```

Planned:

```text
/plot
/eval
/best
/compare
/pause
/resume
```

## 7. Discord webhook

Fish:

```fish
set -gx TRAINCTL_DISCORD_WEBHOOK_URL "https://discord.com/api/webhooks/..."
```

Bash/zsh:

```bash
export TRAINCTL_DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

The current Discord integration is outgoing-only. It can receive lifecycle,
checkpoint, evaluation, disk, and GPU-warning notifications, but it does not
yet provide interactive slash commands.

## 8. Telegram and Discord together

Configure both sets of environment variables, then run your experiment normally:

```bash
trainctl run --name demo -- python train.py
```

The same event can be delivered independently to both channels.

## 9. Inspect effective configuration

```bash
trainctl config show
```

This reports whether secrets are configured without printing the secret values.

## 10. Check latest run

```bash
trainctl status
```

Or, with the Telegram daemon active, send:

```text
/status
```

## 11. tmux

Prefer exporting notification environment variables before creating the tmux
session. If the tmux session already exists, set/source the variables inside
that session as well.

Do not print secrets directly when checking configuration.

## 12. Examples

See:

- `examples/alexnet_cifar10.py`
- `examples/vae_mnist.py`
- `examples/README.md`

Both examples support download-free FakeData smoke tests and intentional
failure injection for testing notifications.

## 13. Security

Never commit:

- Telegram bot tokens
- Discord webhook URLs
- API tokens
- local secret files

trainctl intentionally does not expose arbitrary remote shell execution through
Telegram.

## 14. Mental model

```text
your normal training command
            ↓
trainctl run -- <command>
            ↓
logs / metrics / checkpoints
            ↓
Telegram / Discord
```

The main idea is to wrap existing training commands rather than rewrite
training code around trainctl.
