# trainctl 사용자 가이드

## trainctl이란?

원격 GPU 서버에서 흔히 겪는 흐름은 이렇습니다. SSH로 접속해 긴 학습을
시작하고, 터미널 연결을 끊은 뒤, 학습이 중간에 죽었는지 걱정합니다. 다시
접속해 로그와 GPU 사용량을 확인하고 체크포인트를 직접 찾기도 합니다.
`trainctl`은 일반적인 학습 명령을 감시하고 출력 로그를 저장하며, 실행 상태와
파싱한 지표를 SQLite에 기록하고, 체크포인트 디렉터리를 확인해 주요 이벤트를
알려 줍니다.

실험 추적은 지표·아티팩트·실험 간 비교를 기록하고 탐색하는 일입니다. 실험
오케스트레이션은 작업을 시작하고 실행 과정을 관리하며 상태 변화를 전달하는
일입니다. trainctl은 오케스트레이션에 초점을 둔 가벼운 도구이며, 제한적인 로컬
지표 기록을 제공합니다. WandB나 TensorBoard를 대체하려는 도구는 아닙니다.

## 설치

선언된 지원 Python 버전은 3.10–3.13입니다. 현재 구현은 Linux 및 Unix 계열
학습 서버를 중심으로 하며, 가능한 경우 POSIX 프로세스 그룹 기능을 사용합니다.
CPU 전용 환경도 사용할 수 있습니다. NVIDIA GPU 모니터링은 선택 사항이며,
설치된 경우 NVML을 사용하고 `nvidia-smi`를 대체 경로로 이용합니다. Telegram과
Discord는 각각 선택 설치 기능(`telegram`, `discord` extra)입니다.

저장소에서 설치하려면 다음을 실행합니다.

```bash
git clone <repository-url>
cd <repository-directory>
uv sync
uv pip install -e .
```

선택 의존성은 `uv sync --extra telegram --extra discord --extra nvml`로 추가할 수
있습니다. 저장소에서 `pip install '.[telegram,discord,nvml]'`을 사용해도 됩니다.
현재 PyPI에 배포되지 않았으므로 `pip install trainctl`은 아직 사용할 수 있는
설치 방법이 아닙니다.

## Telegram 설정

1. Telegram에서 **@BotFather**를 열고 `/newbot`을 보낸 뒤 안내를 따릅니다.
2. Bot Token을 복사합니다. 토큰은 비밀번호처럼 다뤄야 합니다.
3. 신뢰할 수 있는 사용자 ID 조회 도구나 Telegram 클라이언트 기능으로 본인의
   숫자형 사용자 ID를 확인합니다. 허용 목록에 있는 사용자만 봇을 쓸 수 있습니다.
4. 선택 의존성을 설치하고 환경 변수를 설정합니다.

```bash
uv pip install -e '.[telegram]'
export TRAINCTL_TELEGRAM_TOKEN='실제-토큰-대신-입력'
export TRAINCTL_TELEGRAM_ALLOWED_USERS='123456789'
```

5. 대화형 기능을 사용하려면 `trainctl daemon`을 실행합니다. `trainctl run`이
   시작·종료 이벤트를 Telegram으로 보내려면 해당 명령을 실행하는 셸에도 같은
   설정이 있어야 합니다.
6. 학습을 시작하고 Telegram에서 시작 및 종료 알림을 확인합니다.

구현된 봇 명령은 `/status`, `/gpu`, `/tail 20`, `/disk`입니다. `/plot`, `/eval`,
`/best`, `/compare`, `/pause`, `/resume`은 명령 이름만 등록되어 있으며 현재는
미구현 안내를 반환합니다.

## Discord 설정

Discord 웹훅은 지정한 채널로 메시지를 보내는 기능입니다. Discord Bot Token이
필요하지 않으며, 대화형 슬래시 명령도 제공하지 않습니다.

1. Discord를 열고 서버와 알림을 받을 텍스트 채널을 선택합니다.
2. 채널 **편집** 설정을 엽니다.
3. **연동(Integrations)** 에서 **웹훅(Webhooks)** 으로 이동합니다.
4. **새 웹훅(New Webhook)** 을 만들고 이름과 채널을 확인합니다.
5. 웹훅 URL을 복사해 비밀로 보관합니다.
6. 환경 변수로 URL을 지정하고 채널을 활성화합니다.

```bash
export TRAINCTL_DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'
```

```toml
[notifications.discord]
enabled = true
```

환경 변수에 URL이 있고 TOML에 `enabled`를 명시하지 않았다면 Discord 채널은
자동으로 활성화됩니다. Discord에서 상호작용형 제어를 제공하려면 향후 별도의
Discord Bot 연동이 필요합니다.

## 학습 실험 실행

`trainctl run`은 `--` 뒤의 인자를 셸을 거치지 않고 자식 프로세스에 전달합니다.

```bash
trainctl run --name experiment-01 -- python train.py
```

2개 GPU를 사용하는 분산 실행 예시는 다음과 같습니다.

```bash
trainctl run \
  --name groot-finetune \
  -- \
  torchrun --nproc_per_node=2 train.py
```

체크포인트 디렉터리와 전체 스텝 수를 지정할 수도 있습니다.

```bash
trainctl run --name experiment-01 \
  --checkpoint-dir ./outputs --total-steps 2000 \
  -- python train.py
```

가장 최근 실행은 `trainctl status`로 확인하고, 비밀을 제외한 적용 설정은
`trainctl config show`로 확인합니다.

## tmux 사용

환경 변수는 프로세스가 시작될 때 전달됩니다. tmux 세션을 만들기 전에 비밀
설정 파일을 불러오면 편리합니다.

```bash
source ~/.config/trainctl/secrets.env
tmux new -s training
```

tmux 서버나 세션을 먼저 시작했다면 해당 세션 안에서도 파일을 불러와야 합니다.

```bash
source ~/.config/trainctl/secrets.env
```

비밀값을 화면에 출력하지 않고 설정 여부를 확인할 수 있습니다.

```bash
test -n "$TRAINCTL_DISCORD_WEBHOOK_URL" && echo "Discord 설정됨"
```

터미널을 분리해 두고 싶다면 tmux 안에서 `trainctl run ...`을 실행할 수 있습니다.
trainctl이 자식 프로세스를 감독하고 로그를 기록하므로 같은 학습에 대해 `trainctl
run`을 중복 실행할 필요는 없습니다. Telegram 대화형 명령을 쓸 때만
`trainctl daemon`을 별도로 실행하면 됩니다. run 감독이나 Discord 웹훅 전송에는
daemon이 필요하지 않습니다. 같은 봇 토큰으로 daemon을 여러 개 띄우지 마세요.

## 알림

현재 EventBus와 라우터는 학습 시작·완료·실패, 체크포인트 저장, 평가 완료·실패,
디스크 부족, GPU 경고 이벤트를 지원합니다. 학습 및 체크포인트 이벤트는 현재
실행 경로에서 발생합니다. 평가 이벤트 형식은 있지만 평가 매니저가 아직 이벤트를
발행하지 않습니다. 디스크와 GPU 경고도 주기적으로 감시해 이벤트를 만드는 정책은
아직 연결되지 않았습니다.

`[notifications.telegram].enabled = true` 또는
`[notifications.discord].enabled = true`로 채널을 각각 켭니다. Telegram에는 토큰과
비어 있지 않은 사용자 허용 목록이 필요합니다. Discord에는 웹훅 URL이 필요합니다.
한 채널에서 오류가 나더라도 다른 채널로의 전송은 계속 시도합니다.

## 설정 항목

기본 설정 파일은 `~/.config/trainctl/config.toml`이며 환경 변수가 TOML보다
우선합니다. 기존 호환성을 위해 최상위 `[telegram]`도 읽을 수 있습니다.

| 설정 키 | 환경 변수 | 기본값 | 필수 여부 | 설명 |
| --- | --- | --- | --- | --- |
| `notifications.telegram.enabled` | — | `false` | 선택 | Telegram 알림 활성화 |
| `notifications.telegram.allowed_users` | `TRAINCTL_TELEGRAM_ALLOWED_USERS` | `[]` | Telegram 사용 시 필수 | 환경 변수에서는 쉼표로 구분한 숫자 ID 사용 |
| Telegram 토큰 | `TRAINCTL_TELEGRAM_TOKEN` | 미설정 | Telegram 사용 시 필수 | Bot Token, 환경 변수로만 설정 |
| `notifications.discord.enabled` | — | `false` | 선택 | Discord 알림 활성화 |
| `notifications.discord.webhook_url` | `TRAINCTL_DISCORD_WEBHOOK_URL`, 이후 `DISCORD_WEBHOOK_URL` | 미설정 | Discord 사용 시 필수 | 환경 변수가 TOML보다 우선하며 앞의 변수가 우선 |
| `storage.database` | `TRAINCTL_DATABASE` | `~/.local/share/trainctl/trainctl.db` | 선택 | SQLite 파일 경로 |
| `logging.directory` | `TRAINCTL_LOG_DIR` | `~/.local/share/trainctl/logs` | 선택 | 학습 로그 저장 경로 |
| `logging.level` | — | `INFO` | 선택 | Python 로그 수준 |
| `monitoring.disk_warning_percent` | — | `90.0` | 선택 | 임계값 설정. 주기 경고 정책은 아직 미연결 |
| `monitoring.disk_paths` | — | 로그·DB 디렉터리 | 선택 | 자원 조회 대상 경로 |
| `checkpoints.poll_interval_seconds` | — | `5.0` | 선택 | 체크포인트 확인 주기 |
| `checkpoints.pattern` | — | `(?:checkpoint|ckpt)[-_]?(?P<step>\d+)` | 선택 | 체크포인트 파일명에서 step을 찾는 패턴 |
| — | `TRAINCTL_CONFIG` | 위 기본 파일 | 선택 | 다른 TOML 파일 지정 |
| — | `TRAINCTL_HOME` | `~/.local/share/trainctl` | 선택 | 기본 데이터 경로 기준 디렉터리 |

전체 예시:

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

토큰과 웹훅 주소는 TOML에 넣지 말고 환경 변수로 전달하세요. `trainctl config
show`는 비밀값 자체를 출력하지 않고 설정 여부만 보여 줍니다.

## 문제 해결

| 증상 | 확인할 내용 |
| --- | --- |
| Telegram 알림이 오지 않음 | `trainctl run`을 실행한 셸에 토큰과 ID가 있는지, `.[telegram]` 설치 여부, 봇과 대화를 시작했는지 확인합니다. `trainctl config show`도 확인하세요. |
| Discord 웹훅 미설정 | `TRAINCTL_DISCORD_WEBHOOK_URL`을 지정하고 환경 파일을 불러온 뒤 `enabled = true`와 `trainctl config show`를 확인합니다. |
| Discord HTTP 429 | 어댑터가 `Retry-After`를 따르고 제한된 횟수로 재시도합니다. 계속 발생하면 알림 빈도를 줄이거나 전용 채널을 사용하세요. |
| Discord 전송 오류 | 로그의 HTTP 상태나 통신 예외를 확인합니다. 401/403/404는 URL 오류 또는 웹훅 폐기 가능성이 큽니다. 로그에 URL은 남기지 않습니다. |
| 학습은 종료됐는데 알림이 없음 | 종료 이벤트는 `trainctl run` 종료 전에 동기적으로 전달됩니다. 해당 셸의 채널 설정과 `trainctl status`를 점검하세요. |
| tmux 안에서 환경 변수가 없음 | 세션 내부에서 `source ~/.config/trainctl/secrets.env`를 실행하고 `test -n "$TRAINCTL_DISCORD_WEBHOOK_URL" && echo 설정됨`으로 확인합니다. |
| GPU 정보를 가져오지 못함 | NVIDIA 드라이버와 `nvidia-smi`를 확인하고 필요하면 `nvidia-ml-py`를 설치합니다. CPU 전용 사용은 가능합니다. |
| 체크포인트를 찾지 못함 | `--checkpoint-dir`, 파일명 패턴, 최소 한 번의 polling 주기를 확인합니다. |
| daemon이 이미 실행 중 | 같은 Telegram 토큰으로 polling 봇을 중복 실행하지 마세요. 기존 daemon을 재사용하거나 종료합니다. |
| 활성 학습 실험이 없다고 표시됨 | `trainctl status`는 최근 저장된 실행을 보여 줍니다. `trainctl run --name ... -- python train.py`로 시작하세요. |

진단 명령:

```bash
trainctl --help
trainctl config show
trainctl status
test -n "$TRAINCTL_TELEGRAM_TOKEN" && echo "Telegram 토큰 설정됨"
test -n "$TRAINCTL_DISCORD_WEBHOOK_URL" && echo "Discord 설정됨"
```

## 아키텍처

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

코어는 타입이 지정된 이벤트를 발행합니다. 라우터는 서로 독립적인 어댑터를
호출하며 Telegram 라이브러리와 Discord HTTP 클라이언트는 연동 계층에만 둡니다.
따라서 두 연동이 없어도 감독 기능을 사용할 수 있고 외부 알림 서비스 장애가 학습
프로세스의 결과를 바꾸지 않습니다.

## 보안

Telegram 토큰과 Discord 웹훅 URL은 환경 변수 또는 Git에 추가하지 않은 비밀 파일에
보관하고 파일 권한을 제한하세요(예: `chmod 600
~/.config/trainctl/secrets.env`). Telegram은 숫자형 허용 목록 밖의 사용자를 거부하고,
Discord 메시지는 기본적으로 멘션 파싱을 끕니다. trainctl은 원격 임의 셸 실행을
지원하지 않습니다. `.env`와 비밀 파일을 저장소에 커밋하지 마세요. `.env.example`에는
예시 자리표시자만 있습니다.

## 현재 제한과 로드맵

**현재 구현:** 로컬 명령 감독, SQLite 실행·지표 기록, 일반 로그 지표 파싱,
체크포인트 polling, 실행 및 체크포인트 이벤트, Telegram/Discord 실행 알림,
Telegram 상태·자원 명령.

**부분 구현:** 평가·디스크·GPU 이벤트 타입과 formatter는 있지만 해당 이벤트를
생성하는 평가 매니저와 주기 정책은 연결되지 않았습니다. 별도 CLI 프로세스에서
발생한 이벤트를 Telegram daemon으로 전달하는 기능도 없습니다. 재시도는 메모리에서
제한적으로 이뤄지며 영속적인 전송 큐는 없습니다.

**계획:** 영속 이벤트 outbox와 daemon 복구, 디스크/GPU 경고 정책, 평가 실행 및 명령,
지표 그래프, 협력형 pause/resume, 대화형 Discord Bot 연동. Discord 웹훅은 발신 전용이며
슬래시 명령은 구현되지 않았습니다.
