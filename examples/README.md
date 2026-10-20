# trainctl examples

These examples demonstrate the intended **zero/low-integration** workflow:
the training script is ordinary PyTorch code, while `trainctl run` supervises
it from the outside.

The scripts print conservative key/value logs such as:

```text
step=12 loss=0.8421 learning_rate=0.001
```

so trainctl can persist progress without requiring a trainctl SDK inside the
training code.

## What notifications can I receive?

When the corresponding environment variables are configured, the same
`trainctl run` can send lifecycle/checkpoint notifications to:

- **Telegram bot**: outgoing run/checkpoint notifications, plus interactive
  `/status`, `/gpu`, `/tail`, and `/disk` when `trainctl daemon` is running.
- **Discord webhook**: outgoing notifications only. The current Discord
  integration is a webhook, not an interactive Discord slash-command bot.

Telegram and Discord can be enabled **at the same time**.

## 1. Install example dependencies

The main trainctl package intentionally does not depend on PyTorch.

With pip:

```bash
pip install 'trainctl[telegram,discord]' torch torchvision
```

With uv:

```bash
uv pip install 'trainctl[telegram,discord]' torch torchvision
```

If you only want one notification channel, install only that extra.

## 2. Configure notifications

### Fish shell

Telegram:

```fish
set -gx TRAINCTL_TELEGRAM_TOKEN "replace-with-your-bot-token"
set -gx TRAINCTL_TELEGRAM_ALLOWED_USERS "123456789"
```

Before trainctl can message you, open the bot in Telegram and send it a message
once (for example `/status` or `/start`) so Telegram permits the bot/user chat.

Discord:

```fish
set -gx TRAINCTL_DISCORD_WEBHOOK_URL "https://discord.com/api/webhooks/..."
```

Set all three variables to receive both Telegram and Discord notifications.

### Bash / zsh

```bash
export TRAINCTL_TELEGRAM_TOKEN="replace-with-your-bot-token"
export TRAINCTL_TELEGRAM_ALLOWED_USERS="123456789"
export TRAINCTL_DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

Do not commit any real bot token or webhook URL.

You can verify configuration without printing secrets:

```bash
trainctl config show
```

## 3. AlexNet-style CIFAR example

A fast, download-free smoke test uses torchvision `FakeData` by default:

```bash
mkdir -p outputs/alexnet

trainctl run \
  --name alexnet-demo \
  --checkpoint-dir ./outputs/alexnet \
  --total-steps 30 \
  -- \
  python examples/alexnet_cifar10.py \
    --dataset fake \
    --steps 30 \
    --checkpoint-every 10 \
    --checkpoint-dir ./outputs/alexnet
```

For real CIFAR-10, change:

```bash
--dataset fake
```

to:

```bash
--dataset cifar10
```

The first real-data run downloads CIFAR-10.

## 4. MNIST VAE example

Download-free smoke test:

```bash
mkdir -p outputs/vae

trainctl run \
  --name vae-demo \
  --checkpoint-dir ./outputs/vae \
  --total-steps 40 \
  -- \
  python examples/vae_mnist.py \
    --dataset fake \
    --steps 40 \
    --checkpoint-every 10 \
    --checkpoint-dir ./outputs/vae
```

Use the real MNIST dataset with:

```bash
--dataset mnist
```

## 5. Test a failure notification on purpose

Both examples support `--fail-at-step`.

For example:

```bash
trainctl run \
  --name vae-failure-demo \
  --total-steps 20 \
  -- \
  python examples/vae_mnist.py \
    --dataset fake \
    --steps 20 \
    --fail-at-step 7
```

At step 7 the child process intentionally raises an exception. trainctl should
mark the run as failed and send a `RunFailed` notification (including recent
log lines) to every configured notification channel.

The non-zero exit status is expected for this demo.

## 6. Telegram interactive commands

Lifecycle notifications from `trainctl run` do **not** require the Telegram
daemon to be running. The run command can send notifications directly when the
Telegram environment variables are configured.

For interactive Telegram commands, start the daemon in another terminal or
tmux pane:

```bash
trainctl daemon
```

Then send the bot:

```text
/status
/gpu
/tail 20
/disk
```

Discord webhooks are currently one-way, so these commands are Telegram-only.

## 7. Check the run locally

You can always inspect the latest persisted run without any messaging service:

```bash
trainctl status
```

Checkpoint files named like `checkpoint-10.pt` match trainctl's default
checkpoint pattern and generate checkpoint events when `--checkpoint-dir` is
passed to `trainctl run`.

## Notes

- `trainctl` watches stdout/stderr and the child process; the examples do not
  import trainctl.
- The generic parser currently recognizes common fields including `step`,
  `loss`, `learning_rate`/`lr`, `validation_loss`/`val_loss`, and
  `success_rate`.
- The VAE also prints `reconstruction_loss` and `kl_loss` for humans. Those
  two fields are not persisted by the current generic parser.
- `FakeData` is useful for testing trainctl itself. It is not a meaningful
  benchmark for model quality.
