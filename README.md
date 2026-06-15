# physical-ai-vanila

xArm6 real robot에서 LeRobot 형식의 demonstration 데이터를 수집하고, SSH 서버의 VLA policy server로 실제 로봇 eval을 돌리기 위한 repo입니다.

MuJoCo PPO/SAC 학습 코드는 보조 기능으로만 남깁니다. RL 관련 내용은 [xarm_rl/README.md](xarm_rl/README.md)를 봅니다.

## Workflow

기본 흐름은 아래 순서입니다.

```bash
cd /home/mlic/data_collection/physical-ai-vanila
vim sh_scripts/env.sh

./sh_scripts/00_check_robot.sh
./sh_scripts/10_collect_delta_dataset.sh
./sh_scripts/20_sync_data_to_ssh.sh
./sh_scripts/30_remote_policy_server.sh
./sh_scripts/40_open_policy_tunnel.sh
./sh_scripts/50_vla_no_motion.sh
./sh_scripts/60_vla_real_eval.sh
```

`50_vla_no_motion.sh`가 정상일 때만 `60_vla_real_eval.sh`를 실행합니다.

## Config

연구원이 보통 바꾸는 값은 [sh_scripts/env.sh](sh_scripts/env.sh)에만 둡니다. joystick sign, deadzone, camera resolution, servo gain, safety clamp, gripper button 같은 실행 디테일은 각 script 안에 고정되어 있습니다.

자주 바꾸는 값:

| 변수 | 용도 |
|---|---|
| `ROBOT_IP` | xArm controller IP |
| `WRIST_CAMERA_SERIAL` | wrist RealSense serial |
| `FRONT_CAMERA_SERIAL` | front RealSense serial |
| `USE_FRONT_CAMERA` | `1`이면 wrist+front, `0`이면 wrist만 사용 |
| `TASK` | data collection/eval language instruction |
| `DATA_ROOT` | local LeRobot dataset 저장 root |
| `REPO_ID` | LeRobot dataset repo id |
| `TASK_ID` | dataset task folder |
| `DATA_ACTION_MODE` | demo collection action 저장 방식, 기본 `both` |
| `ACTION_MODE` | real eval action 실행 방식, 기본 `delta` |
| `FPS` | 수집/eval FPS |
| `REMOTE` | SSH alias, 기본 `10server` |
| `REMOTE_LEROBOT_ROOT` | SSH 서버의 LeRobot repo 경로 |
| `REMOTE_DATA_BASE` | 수집 데이터가 복사될 SSH 서버 경로 |
| `REMOTE_SETUP` | 서버 venv/conda 활성화 명령 |
| `REMOTE_POLICY_PORT` | SSH 서버 policy server port |
| `LOCAL_POLICY_PORT` | robot PC tunnel local port |
| `POLICY_TYPE` | `pi05`, `smolvla` 등 |
| `POLICY_PATH` | SSH 서버 filesystem의 checkpoint path |
| `POLICY_DEVICE` | `cuda` 또는 `cpu` |

예시:

```bash
TASK="pick up the white bottle and place it in the dark brown box"
TASK_ID="TASK_DELTA"
DATA_ROOT="${PROJECT_ROOT}/data/xarm6_delta_demo"
DATA_ACTION_MODE=both

POLICY_TYPE=pi05
POLICY_PATH=/home/mlic/mingukang/lerobot/outputs/train/pi05_real_full_wandb_20260612_160420/checkpoints/025000/pretrained_model
LOCAL_POLICY_PORT=8080
```

`POLICY_PATH`는 로봇 PC 경로가 아니라 SSH 서버에서 보이는 checkpoint 경로입니다.

## Setup

로봇 PC에서:

```bash
cd /home/mlic/data_collection/physical-ai-vanila
source .venv/bin/activate
pip install -e ".[real]"
pip install pyrealsense2 opencv-python grpcio lerobot
```

로봇과 카메라 확인:

```bash
./sh_scripts/00_check_robot.sh
```

실행 전 체크:

- xArm Studio에서 motor enable, error clear
- e-stop이 손 닿는 위치에 있는지 확인
- RealSense Viewer, 이전 VLA client, 이전 teleop process 종료
- camera serial이 [sh_scripts/env.sh](sh_scripts/env.sh)의 값과 맞는지 확인

## Data Collection

수집은 이 스크립트로 실행합니다.

```bash
./sh_scripts/10_collect_delta_dataset.sh
```

시작 전에 [05_release_cameras.sh](sh_scripts/05_release_cameras.sh)가 `/dev/video*`를 잡고 있는 이전 process를 정리합니다.

### 조작

조이스틱은 xArm TCP를 Cartesian delta로 움직입니다. 모든 조작은 버튼/스틱을 놓으면 멈춥니다.

| 입력 | 동작 |
|---|---|
| left stick 위 | TCP X+ 방향 |
| left stick 아래 | TCP X- 방향 |
| left stick 왼쪽 | TCP Y+ 방향 |
| left stick 오른쪽 | TCP Y- 방향 |
| LB | TCP Z+ 방향, 위로 이동 |
| LT | TCP Z- 방향, 아래로 이동 |
| right stick 왼쪽/오른쪽 | TCP roll -/+ |
| right stick 위/아래 | TCP pitch +/- |
| RB | TCP yaw + |
| RT | TCP yaw - |
| BTN_Y / triangle | 누르고 있는 동안 gripper 닫기 |
| BTN_X | 누르고 있는 동안 gripper 열기 |
| SELECT | episode 저장 후 종료 |
| Enter | episode 저장, home 복귀, 종료 |
| Esc | episode 폐기, home 복귀, 종료 |
| START | emergency stop, episode 폐기 |

기본 속도는 TCP position `80 mm/s`, rotation `25 deg/s`, gripper `300 unit/s`입니다. 실제 움직임 방향이 현장 기준과 반대로 느껴지면 [10_collect_delta_dataset.sh](sh_scripts/10_collect_delta_dataset.sh)의 `--x-sign`, `--y-sign`, `--z-sign`, `--roll-sign`, `--pitch-sign`, `--yaw-sign`만 조정합니다.

기록 중 episode 처리는 이렇게 합니다.

| 입력 | 결과 |
|---|---|
| SELECT | 지금까지 기록한 episode 저장, 로봇은 현재 위치에 둔 채 종료 |
| Enter | episode 저장, PPO joint home으로 복귀, 종료 |
| Esc | episode 폐기, PPO joint home으로 복귀, 종료 |
| START | emergency stop, episode 폐기 |

gripper는 세모/네모를 누르고 있는 동안 계속 움직이고, gripper delta도 action에 기록됩니다.

세모/네모를 눌렀는데 gripper가 안 움직이면 먼저 터미널 로그를 봅니다. `gripper close button pressed` 또는 `gripper open button pressed`가 찍히면 joystick 입력은 들어온 것입니다. 이후 `target`, `actual`, `ret`를 확인합니다. `ret`가 0이 아니면 xArm gripper command가 실패한 것이고, `actual`이 변하지 않으면서 `min-limit` 또는 `max-limit`가 뜨면 이미 gripper 범위 끝에 붙어 있는 상태입니다.

### 복귀 자세

Enter 또는 Esc를 누르면 script 시작 TCP 위치가 아니라 PPO 학습/real deploy에서 쓰던 joint home으로 돌아갑니다.

```text
HOME_QPOS_RAD = [0.0, -0.3, -1.2, 0.0, 1.5, 0.0]
```

이 값은 [xarm_rl/envs/base_env.py](xarm_rl/envs/base_env.py)의 `HOME_QPOS`와 같습니다.

### Action Contract

demo collection의 기본 `DATA_ACTION_MODE`는 `both`입니다. 이때 `action`에는 delta 7DoF가 저장되고, `action.absolute`에는 absolute target 7DoF가 함께 저장됩니다.

```text
observation.images.wrist   RGB video
observation.images.front   RGB video, front camera enabled
observation.state          float32[7], current TCP xyz/rpy + gripper
action                     float32[7], target - observation.state
action.absolute            float32[7], target TCP xyz/rpy + gripper
task                       language instruction
```

`action` 순서와 단위:

```text
delta_tcp_x_mm
delta_tcp_y_mm
delta_tcp_z_mm
delta_tcp_roll_rad
delta_tcp_pitch_rad
delta_tcp_yaw_rad
delta_gripper_pos
```

`action.absolute` 순서와 단위:

```text
target_tcp_x_mm
target_tcp_y_mm
target_tcp_z_mm
target_tcp_roll_rad
target_tcp_pitch_rad
target_tcp_yaw_rad
target_gripper_pos
```

한쪽만 저장하고 싶을 때만 [10_collect_delta_dataset.sh](sh_scripts/10_collect_delta_dataset.sh) 실행 전에 `DATA_ACTION_MODE=delta` 또는 `DATA_ACTION_MODE=absolute`를 지정합니다. `absolute`만 저장하면 `action` 필드 자체가 absolute target이 됩니다.

기존 action schema와 다른 dataset에는 episode를 섞지 않습니다. 기존 `meta/info.json`에 `action.absolute`가 없는데 `DATA_ACTION_MODE=both`로 이어붙이거나, 반대로 `both` dataset에 `delta`/`absolute`만 이어붙이려고 하면 수집을 시작하지 않고 에러를 냅니다. 다른 schema는 새 `DATA_ROOT` 또는 `TASK_ID`로 분리합니다.

같은 `TASK_ID`로 이어서 수집할 때 instruction이 섞이지 않도록, dataset root 아래에 `task_instruction.txt`를 저장합니다.

```text
data/xarm6_delta_demo/TASK_DELTA/task_instruction.txt
```

이미 존재하는 task folder에 다시 수집할 경우, 이 파일의 instruction과 현재 `TASK`가 다르면 수집을 시작하지 않고 에러를 냅니다. 다른 instruction은 새 `TASK_ID` 또는 새 `DATA_ROOT`로 분리합니다.

### Dataset

LeRobot dataset은 대략 다음 구조로 생성됩니다.

```text
data/xarm6_delta_demo/
  TASK_DELTA/
    task_instruction.txt
    data/
      chunk-000/
        file-000.parquet
    videos/
      observation.images.wrist/
        chunk-000/
          file-000.mp4
      observation.images.front/
        chunk-000/
          file-000.mp4
    meta/
      info.json
      stats.json
      tasks.parquet
      episodes/
        chunk-000/
          file-000.parquet
```

VLA fine-tuning에서는 `observation.state`, `observation.images.*`, `task`를 input으로 쓰고 `action`을 예측 대상으로 씁니다.

## Sync To SSH Server

수집이 끝나면 로봇 PC에서:

```bash
./sh_scripts/20_sync_data_to_ssh.sh
```

기본 위치:

```text
local : physical-ai-vanila/data
remote: 10server:/home/mlic/mingukang/lerobot/collected_demo/data
```

필요하면 [sh_scripts/env.sh](sh_scripts/env.sh)의 `REMOTE`, `REMOTE_DATA_BASE`를 수정합니다.

## Remote VLA Eval

### 1. Policy Server

로봇 PC에서 서버 실행 스크립트를 호출합니다.

```bash
./sh_scripts/30_remote_policy_server.sh
```

이 스크립트는 `REMOTE`에 SSH로 접속해서 LeRobot async policy server를 실행합니다. 서버는 client handshake 때 `POLICY_TYPE`과 `POLICY_PATH`를 받아 checkpoint를 로드합니다.

서버에서 venv/conda 활성화가 필요하면 [sh_scripts/env.sh](sh_scripts/env.sh)에 넣습니다.

```bash
REMOTE_SETUP="source .venv/bin/activate &&"
```

### 2. Tunnel

다른 터미널에서:

```bash
./sh_scripts/40_open_policy_tunnel.sh
```

기본 연결:

```text
robot PC 127.0.0.1:8080 -> 10server 127.0.0.1:8080
```

포트 번호는 tunnel과 client가 같아야 합니다. `LOCAL_POLICY_PORT=18080`을 쓰면 tunnel도 client도 `18080`으로 맞춥니다.

포트가 꼬였을 때:

```bash
fuser -v 8080/tcp 18080/tcp
fuser -k 8080/tcp 18080/tcp
```

### 3. No-Motion

실제 로봇을 움직이기 전에 먼저 실행합니다.

```bash
./sh_scripts/50_vla_no_motion.sh
```

확인할 것:

- policy server handshake 성공
- checkpoint가 SSH 서버에서 로드됨
- RealSense image와 `observation.state` 정상
- action 값이 demo action range와 크게 어긋나지 않음
- delta checkpoint면 `ACTION_MODE=delta`

### 4. Real Eval

no-motion에서 action이 정상일 때만 실제 실행합니다.

```bash
./sh_scripts/60_vla_real_eval.sh
```

`60_vla_real_eval.sh`는 보수적인 step limiter로 시작합니다. 속도나 step 제한을 바꿔야 할 때만 script 내부의 `--max-pos-step-mm`, `--max-rot-step-rad`, `--max-gripper-step`, `--servo-speed`, `--servo-acc` 값을 수정합니다.

## Repo Layout

```text
physical-ai-vanila/
  joy_stick/
    joy_telecontrol_serial.py   # xArm6 teleop + LeRobot dataset recording
  scripts/
    vla_xarm_client.py          # real xArm6 client for remote VLA policy server
    camera_ready.py             # RealSense quick check
    diag_servo.py               # xArm servo sanity check
  sh_scripts/
    env.sh                      # experiment config
    00_check_robot.sh
    05_release_cameras.sh
    10_collect_delta_dataset.sh
    20_sync_data_to_ssh.sh
    30_remote_policy_server.sh
    40_open_policy_tunnel.sh
    50_vla_no_motion.sh
    60_vla_real_eval.sh
  xarm_rl/                      # optional MuJoCo PPO/SAC envs
  data/                         # local LeRobot datasets
  outputs/                      # local diagnostics and reports
```

## Troubleshooting

| 증상 | 확인 |
|---|---|
| `Address already in use` | local tunnel port가 이미 사용 중입니다. `fuser -v 8080/tcp 18080/tcp` |
| `timed out before receiving SETTINGS frame` | tunnel은 열렸지만 remote `127.0.0.1:8080`이 LeRobot gRPC policy server가 아닐 수 있습니다. |
| `'PI05Config' object has no attribute 'vlm_model_name'` | PI0.5 checkpoint를 `smolvla`로 로드한 경우입니다. `POLICY_TYPE=pi05` |
| `VIDIOC_S_FMT errno=16 Device or resource busy` | 이전 client나 RealSense Viewer가 카메라를 잡고 있습니다. `./sh_scripts/05_release_cameras.sh` |
| RealSense serial 없음 | `./sh_scripts/00_check_robot.sh`로 연결 serial 확인 |
| xArm error code | xArm Studio에서 error clear 후 재시도 |
