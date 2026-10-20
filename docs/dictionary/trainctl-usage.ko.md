# trainctl 라이브러리 사용 사전

이 문서는 trainctl을 처음 사용하는 사람이 **무엇을 설치하고, 어떤 환경변수를 설정하고, 어떤 명령을 실행해야 하는지** 빠르게 찾아볼 수 있도록 만든 실전형 사용 가이드입니다.

## 1. trainctl은 무엇을 하나요?

trainctl은 딥러닝 학습 코드를 직접 바꾸지 않고 바깥에서 감싸서 실행하는 경량 실험 관리자입니다.

```text
기존 방식
python train.py

trainctl 방식
trainctl run -- python train.py
```

trainctl은 자식 프로세스의 다음 정보를 감시합니다.

- 시작/종료/실패 상태
- stdout/stderr 로그
- step/loss 등의 일반적인 metric
- checkpoint 생성
- GPU/메모리/디스크 상태
- Telegram / Discord 알림

학습 코드 안에서 `import trainctl`을 하지 않아도 사용할 수 있습니다.

---

## 2. 설치

기본 설치:

```bash
pip install trainctl
```

Telegram까지 사용:

```bash
pip install 'trainctl[telegram]'
```

Discord Webhook까지 사용:

```bash
pip install 'trainctl[discord]'
```

둘 다 사용:

```bash
pip install 'trainctl[telegram,discord]'
```

GPU 모니터링용 NVML도 같이 설치:

```bash
pip install 'trainctl[telegram,discord,nvml]'
```

개발 중인 repository를 직접 사용하는 경우:

```bash
uv sync --extra telegram --extra discord --extra nvml
uv pip install -e .
```

---

## 3. 가장 기본적인 사용법

기존 명령이:

```bash
python train.py
```

였다면:

```bash
trainctl run --name my-experiment -- python train.py
```

처럼 실행합니다.

`--` 뒤의 내용이 실제 학습 명령입니다.

예를 들어 torchrun:

```bash
trainctl run \
  --name groot-finetune \
  --total-steps 2000 \
  -- \
  torchrun --nproc_per_node=2 train.py --config config.yaml
```

---

## 4. checkpoint 감시

checkpoint가 저장되는 디렉터리를 trainctl에 알려주면 checkpoint 생성 이벤트를 감지할 수 있습니다.

```bash
trainctl run \
  --name my-experiment \
  --checkpoint-dir ./outputs \
  --total-steps 2000 \
  -- \
  python train.py
```

기본적으로 다음과 같은 이름을 인식합니다.

```text
checkpoint-1000
checkpoint_1000
ckpt-1000
ckpt_1000
checkpoint-1000.pt
```

checkpoint가 감지되면 활성화된 Telegram/Discord 채널로 알림을 보낼 수 있습니다.

---

## 5. log / metric 작성 규칙

trainctl은 일반적인 stdout 로그를 파싱합니다.

추천 형식:

```text
step=100 loss=0.823 learning_rate=0.0001
```

현재 generic parser가 인식하는 주요 필드:

```text
step
loss
learning_rate
lr
validation_loss
val_loss
success_rate
```

즉 학습 코드에서는 다음처럼 평범하게 출력하면 됩니다.

```python
print(
    f"step={step} loss={loss.item():.6f} learning_rate={lr:.8g}",
    flush=True,
)
```

추가적인 metric을 출력해도 문제는 없지만, generic parser가 모르는 이름은 아직 SQLite metric으로 저장되지 않을 수 있습니다.

---

## 6. Telegram 설정

필요한 값:

- Telegram Bot Token
- 알림을 받을 Telegram numeric user ID

Fish shell:

```fish
set -gx TRAINCTL_TELEGRAM_TOKEN "YOUR_BOT_TOKEN"
set -gx TRAINCTL_TELEGRAM_ALLOWED_USERS "123456789"
```

Bash / zsh:

```bash
export TRAINCTL_TELEGRAM_TOKEN="YOUR_BOT_TOKEN"
export TRAINCTL_TELEGRAM_ALLOWED_USERS="123456789"
```

여러 명을 허용하려면 쉼표로 구분합니다.

```bash
export TRAINCTL_TELEGRAM_ALLOWED_USERS="123456789,987654321"
```

Telegram 봇과 먼저 한 번 대화를 시작해야 봇이 해당 사용자에게 메시지를 보낼 수 있습니다.

---

## 7. Telegram에서 사용할 수 있는 명령

Telegram 명령을 사용하려면 별도 terminal 또는 tmux pane에서 daemon을 실행합니다.

```bash
trainctl daemon
```

현재 사용 가능한 명령:

```text
/status
/gpu
/tail 20
/disk
```

의미:

- `/status`: 가장 최근 실험의 상태/step/loss 확인
- `/gpu`: GPU utilization / VRAM / 시스템 RAM 확인
- `/tail 20`: 최근 학습 로그 20줄 확인
- `/disk`: 설정된 경로의 디스크 사용량 확인

다음 기능은 계획 중입니다.

```text
/plot
/eval
/best
/compare
/pause
/resume
```

---

## 8. Discord 설정

현재 Discord 연동은 **Webhook 기반 단방향 알림**입니다.

Fish shell:

```fish
set -gx TRAINCTL_DISCORD_WEBHOOK_URL "https://discord.com/api/webhooks/..."
```

Bash / zsh:

```bash
export TRAINCTL_DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

Discord에서는 현재 다음 종류의 알림을 받을 수 있습니다.

- Training started
- Training completed
- Training failed
- Checkpoint saved
- Evaluation completed / failed
- Disk warning
- GPU warning

Discord Webhook은 입력을 받을 수 없기 때문에 `/status` 같은 command는 아직 사용할 수 없습니다.

---

## 9. Telegram + Discord 동시에 사용

둘 다 설정하면 같은 이벤트를 독립적으로 받습니다.

```text
RunFailed
   ├── Telegram
   └── Discord
```

Fish 예시:

```fish
set -gx TRAINCTL_TELEGRAM_TOKEN "..."
set -gx TRAINCTL_TELEGRAM_ALLOWED_USERS "123456789"
set -gx TRAINCTL_DISCORD_WEBHOOK_URL "https://discord.com/api/webhooks/..."
```

그 뒤 평소처럼:

```bash
trainctl run --name demo -- python train.py
```

하면 됩니다.

한 채널에서 알림 전송이 실패하더라도 다른 채널에는 독립적으로 전송됩니다.

---

## 10. 현재 설정 확인

secret 값을 직접 출력하지 않고 설정 상태를 확인하려면:

```bash
trainctl config show
```

예상 형태:

```json
{
  "telegram": {
    "enabled": true,
    "token_configured": true
  },
  "discord": {
    "enabled": true,
    "webhook_configured": true
  }
}
```

---

## 11. 가장 최근 실험 상태 확인

local terminal에서는:

```bash
trainctl status
```

을 사용합니다.

Telegram daemon을 실행 중이라면 Telegram에서:

```text
/status
```

를 보내도 됩니다.

---

## 12. tmux에서 사용할 때

tmux를 시작하기 전에 환경변수를 설정하는 편이 가장 단순합니다.

Fish:

```fish
set -gx TRAINCTL_TELEGRAM_TOKEN "..."
set -gx TRAINCTL_TELEGRAM_ALLOWED_USERS "123456789"
set -gx TRAINCTL_DISCORD_WEBHOOK_URL "https://discord.com/api/webhooks/..."

tmux new -s training
```

이미 tmux session이 만들어진 뒤 환경변수를 추가했다면, 그 tmux pane 안에서도 다시 export/set 해주는 것이 안전합니다.

설정 여부만 확인하려면:

```fish
test -n "$TRAINCTL_DISCORD_WEBHOOK_URL"; and echo "Discord configured"
test -n "$TRAINCTL_TELEGRAM_TOKEN"; and echo "Telegram configured"
```

secret 자체를 `echo`로 출력하지 않는 것을 권장합니다.

---

## 13. 예제 실행

repository 안에는 실제 PyTorch 예제가 있습니다.

AlexNet-style:

```bash
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

VAE:

```bash
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

자세한 내용은 `examples/README.md`를 참고하세요.

---

## 14. 실패 알림 테스트

일부 예제는 의도적으로 실패하도록 만들 수 있습니다.

```bash
trainctl run \
  --name failure-demo \
  --total-steps 20 \
  -- \
  python examples/vae_mnist.py \
    --dataset fake \
    --steps 20 \
    --fail-at-step 7
```

step 7에서 예외가 발생하고 trainctl은 이를 `RunFailed`로 처리합니다.

Telegram/Discord가 설정되어 있다면 실패 알림과 최근 로그를 받을 수 있습니다.

---

## 15. 자주 쓰는 명령 요약

```bash
# help
trainctl --help

# 학습 실행
trainctl run --name EXPERIMENT -- python train.py

# checkpoint 감시
trainctl run \
  --name EXPERIMENT \
  --checkpoint-dir ./checkpoints \
  --total-steps 10000 \
  -- \
  python train.py

# 최근 run 상태
trainctl status

# 설정 확인
trainctl config show

# Telegram command daemon
trainctl daemon
```

---

## 16. 보안 주의사항

다음 값은 절대로 Git repository에 commit하지 마세요.

- Telegram Bot Token
- Discord Webhook URL
- 기타 API token

다음 파일도 보통 commit하지 않습니다.

```text
.env
.env.local
secrets.env
```

trainctl은 Telegram에서 임의의 shell command를 실행하는 기능을 제공하지 않습니다.

이는 remote experiment manager를 안전하게 유지하기 위한 의도적인 제한입니다.

---

## 17. 핵심 사용 패턴

가장 중요한 사용법은 이것 하나로 요약할 수 있습니다.

```text
원래 실행하던 학습 명령
        ↓
trainctl run -- <원래 명령>
        ↓
log / metric / checkpoint 감시
        ↓
Telegram / Discord 알림
```

즉 기존 training pipeline을 trainctl 전용으로 다시 작성하는 것이 아니라, **기존 명령을 바깥에서 감싸는 것**이 기본 철학입니다.
