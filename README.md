# physical-ai-vanila

`physical-ai-vanila`는 xArm6 real robot에서 VLA fine-tuning용 LeRobot 데이터를 수집하고, 학습된 VLA checkpoint를 실제 로봇에서 평가하기 위한 작업 repo입니다.

주요 흐름은 다음과 같습니다.

1. 로봇 PC에서 joystick + RealSense로 xArm6 demonstration 데이터를 수집합니다.
2. 수집된 LeRobot dataset을 SSH 서버로 동기화합니다.
3. SSH 서버에서 LeRobot async policy server를 띄웁니다.
4. 로봇 PC는 SSH tunnel을 통해 policy server에 observation을 보내고 action chunk를 받아 실행합니다.

MuJoCo PPO/SAC 학습 코드는 보조 기능으로만 유지합니다. 관련 내용은 [xarm_rl/README.md](xarm_rl/README.md)를 보세요.

## Repo 구조

```text
physical-ai-vanila/
  joy_stick/
    joy_telecontrol_serial.py   # xArm6 teleop + LeRobot dataset recording
  scripts/
    vla_xarm_client.py          # remote VLA policy server client for real xArm6 eval
    camera_ready.py             # RealSense quick check
    diag_servo.py               # xArm servo sanity check
    train.py, eval_headless.py  # legacy/support xarm_rl experiments
  sh_scripts/
    env.sh                      # shared local config
    00_check_robot.sh
    10_collect_delta_dataset.sh
    20_sync_data_to_ssh.sh
    30_remote_policy_server.sh
    40_open_policy_tunnel.sh
    50_vla_no_motion.sh
    60_vla_real_eval.sh
  xarm_rl/                      # optional MuJoCo PPO/SAC envs
  data/                         # local LeRobot datasets
  outputs/                      # diagnostics, old RL outputs, reports
```

긴 명령은 되도록 `sh_scripts/`에 넣었습니다. 먼저 [sh_scripts/env.sh](sh_scripts/env.sh)를 열어서 robot IP, camera serial, task, model path, port를 현재 실험에 맞게 수정하세요.

## 설치와 기본 확인

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

기본 xArm controller IP는 `192.168.1.199`입니다. 바뀌면 [sh_scripts/env.sh](sh_scripts/env.sh)의 `ROBOT_IP`를 수정합니다.

실제 로봇 실행 전에는 항상 xArm Studio에서 motor enable, error clear, home pose, e-stop 위치를 확인합니다.

## 데이터 수집

데이터 수집은 [joy_stick/joy_telecontrol_serial.py](joy_stick/joy_telecontrol_serial.py)를 사용합니다. 기본 action contract는 `delta`입니다.

```text
observation.images.wrist   RGB video
observation.images.front   RGB video, front camera를 켠 경우
observation.state          float32[7], current TCP xyz/rpy + gripper
action                     float32[7], target - observation.state
task                       language instruction
```

`action`의 순서와 단위:

```text
delta_tcp_x_mm
delta_tcp_y_mm
delta_tcp_z_mm
delta_tcp_roll_rad
delta_tcp_pitch_rad
delta_tcp_yaw_rad
delta_gripper_pos
```

수집 시작:

```bash
./sh_scripts/10_collect_delta_dataset.sh
```

조작 중 `SELECT`는 episode 저장 후 종료, `START`는 emergency stop 후 episode buffer 폐기입니다. `Enter`는 저장 후 초기 pose로 복귀하고 종료합니다.

기존 absolute target action 데이터셋에 delta episode를 이어 붙이지 마세요. action 의미가 섞이지 않도록 새 `DATA_ROOT` 또는 `TASK_ID`로 수집합니다.

## Dataset 트리

LeRobot dataset은 대략 다음 구조로 생성됩니다.

```text
data/xarm6_delta_demo/
  data/
    chunk-000/
      file-000.parquet
      file-001.parquet
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

VLA fine-tuning에서는 `observation.state`, `observation.images.*`, `task`를 input으로 쓰고 `action`을 예측 대상으로 씁니다. delta로 수집한 checkpoint는 real eval에서도 `--action-mode delta`로 실행해야 합니다.

## SSH 서버로 데이터 보내기

수집이 끝나면 로봇 PC에서:

```bash
./sh_scripts/20_sync_data_to_ssh.sh
```

기본 동기화 위치는 다음과 같습니다.

```text
local : physical-ai-vanila/data
remote: 10server:/home/mlic/mingukang/lerobot/collected_demo/data
```

필요하면 [sh_scripts/env.sh](sh_scripts/env.sh)의 `REMOTE`, `REMOTE_DATA_BASE`를 수정합니다.

## Remote VLA inference

### 1. 모델 등록

로봇 PC의 [sh_scripts/env.sh](sh_scripts/env.sh)에서 inference할 checkpoint를 등록합니다.

```bash
POLICY_TYPE=pi05
POLICY_PATH=/home/mlic/mingukang/lerobot/outputs/train/pi05_real_full_wandb_20260612_160420/checkpoints/025000/pretrained_model
ACTION_MODE=delta
LOCAL_POLICY_PORT=8080
```

`POLICY_PATH`는 로봇 PC 경로가 아니라 SSH 서버 filesystem의 checkpoint 경로입니다. `POLICY_TYPE`과 checkpoint 종류가 맞아야 합니다. 예를 들어 PI0.5 checkpoint는 `pi05`, SmolVLA checkpoint는 `smolvla`입니다.

서버에서 별도 venv/conda 활성화가 필요하면 `REMOTE_SETUP`에 넣습니다.

```bash
REMOTE_SETUP="source .venv/bin/activate &&"
```

### 2. SSH 서버에서 policy server 실행

`10server`에 LeRobot 환경이 준비되어 있다면 로봇 PC에서 다음 스크립트로 policy server를 띄웁니다.

```bash
./sh_scripts/30_remote_policy_server.sh
```

서버는 처음에는 빈 policy server입니다. 로봇 PC client가 handshake할 때 `POLICY_TYPE`과 `POLICY_PATH`를 보내고, 서버가 자기 디스크에서 checkpoint를 로드합니다.

### 3. 로봇 PC에서 tunnel 열기

다른 터미널에서:

```bash
./sh_scripts/40_open_policy_tunnel.sh
```

기본 연결은 다음과 같습니다.

```text
robot PC 127.0.0.1:8080 -> 10server 127.0.0.1:8080
```

포트 번호는 반드시 client와 tunnel이 같아야 합니다. `LOCAL_POLICY_PORT=8080`이면 client도 `127.0.0.1:8080`을 사용합니다. 8080이 이미 사용 중이면 [sh_scripts/env.sh](sh_scripts/env.sh)에서 `LOCAL_POLICY_PORT=18080`으로 바꾸고 tunnel과 client를 둘 다 같은 설정으로 실행합니다.

포트가 꼬였을 때:

```bash
fuser -v 8080/tcp 18080/tcp
fuser -k 8080/tcp 18080/tcp
```

### 4. no-motion 1차 테스트

실제 로봇을 움직이기 전에 항상 no-motion 진단을 먼저 돌립니다.

```bash
./sh_scripts/50_vla_no_motion.sh
```

이 단계에서 확인할 것:

- policy server handshake가 되는지
- checkpoint가 서버에서 로드되는지
- RealSense image와 `observation.state`가 정상인지
- action 값이 demo action range와 크게 어긋나지 않는지
- delta checkpoint는 `ACTION_MODE=delta`로 실행되는지

### 5. 실제 로봇 eval

no-motion에서 action이 정상 출력될 때만 실제 실행합니다.

```bash
./sh_scripts/60_vla_real_eval.sh
```

처음에는 [sh_scripts/env.sh](sh_scripts/env.sh)의 step limiter를 작게 두고 시작합니다.

```bash
MAX_POS_STEP_MM=2
MAX_ROT_STEP_RAD=0.015
MAX_GRIPPER_STEP=30
SERVO_SPEED=20
SERVO_ACC=150
```

안정적으로 확인한 뒤 천천히 올립니다.

## Troubleshooting

| 증상 | 확인 |
|---|---|
| `Address already in use` | local tunnel port가 이미 사용 중입니다. `fuser -v 8080/tcp 18080/tcp`로 확인합니다. |
| `timed out before receiving SETTINGS frame` | tunnel은 열렸지만 remote `127.0.0.1:8080`이 LeRobot gRPC policy server가 아닐 수 있습니다. 서버 쪽 policy server를 확인합니다. |
| `'PI05Config' object has no attribute 'vlm_model_name'` | PI0.5 checkpoint를 `smolvla`로 로드한 경우입니다. `POLICY_TYPE=pi05`로 맞춥니다. |
| `VIDIOC_S_FMT errno=16 Device or resource busy` | 이전 client나 RealSense Viewer가 카메라를 잡고 있습니다. `fuser -v /dev/video*`로 확인합니다. |
| RealSense serial 없음 | `./sh_scripts/00_check_robot.sh`로 연결 serial을 확인하고 `env.sh`를 수정합니다. |
| xArm error code | xArm Studio에서 error clear 후 재시도합니다. |
