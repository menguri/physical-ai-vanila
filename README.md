# xArm-project

**xArm6 (6-DOF)** 로봇을 **MuJoCo 시뮬에서 강화학습**시키고, **실제 UFactory xArm6**로 배포(sim-to-real)하는 프로젝트.

- 시뮬: MuJoCo 3.x (state-only observation)
- RL: Stable-Baselines3 — **PPO** / **SAC**
- 태스크: **Reach** (end-effector를 목표 좌표까지 이동)
- 배포: xArm-Python-SDK (`XArmAPI`) 직접 호출
- Domain randomization + Safe-zone penalty 내장

---

## 0. 한눈에 보는 흐름

```
[xArm6 URDF / MJCF]                                [실제 xArm6 컨트롤러]
       │                                                     ▲
       ▼                                                     │
[MuJoCo Gym env]   →   [PPO / SAC 학습]  →  [정책.zip]   →  [deploy_real.py]
   + DR              (Stable-Baselines3)
   + safe penalty
```

### 🎬 데모: 9-point Grid Tour (PPO baseline, 9/9 도달)

![grid tour](outputs/gif/ppo_v2_grid_tour_small.gif)

Safe zone 안에 균일한 3×3 격자(9개 target)를 정의해, 정책이 한 점씩 reach 후 home 복귀를 반복합니다. 빨강=현재 target, 초록=완료, 회색=대기. 자세한 실행 방법은 [§6.4](#64-9-point-grid-tour-데모-시각화).

---

## 1. 레포 구조

```
xArm-project/
├── assets/
│   ├── xarm6/
│   │   ├── xarm6.xml          # MuJoCo MJCF (arm + 2-finger gripper + TCP site)
│   │   ├── xarm6.urdf         # URDF (xArm6-only, Allegro 제거)
│   │   └── meshes/            # STL (base + link1..link6)
│   ├── scene_reach.xml        # Reach scene (테이블 + target sphere)
│   └── scene_pick_place.xml   # PickPlace scene (테이블 + 큐브 + target)
├── xarm_rl/
│   ├── __init__.py            # gym env 등록 (XArm6Reach-v0, XArm6PickPlace-v0)
│   └── envs/
│       ├── base_env.py        # MuJoCo wrapper + joint/gripper helpers
│       ├── reach_env.py       # XArm6Reach-v0 (DR + safe-zone penalty)
│       └── pick_place_env.py  # XArm6PickPlace-v0 (현재 미학습)
├── scripts/
│   ├── sanity_random.py       # viewer로 random rollout (display 필요)
│   ├── train.py               # PPO / SAC 학습 (--domain_rand 옵션)
│   ├── eval_headless.py       # 학습된 정책 평가 (success rate, 디스플레이 X)
│   ├── render_gif.py          # rollout을 GIF로 (MuJoCo EGL offscreen)
│   └── deploy_real.py         # 실제 xArm6 배포 (XArmAPI)
├── outputs/                   # 학습 산출물 (.gitignore)
│   ├── report.md              # 학습 결과 요약 (성공 조합 + 비교)
│   ├── gif/                   # 데모 / rollout GIF 모음 (README 임베드)
│   │   ├── ppo_v2_grid_tour.gif        (19 MB, 640x480, 풀버전)
│   │   ├── ppo_v2_grid_tour_small.gif  (7.2 MB, 480x360, README용)
│   │   ├── ppo_v2_rollout.gif          (3 episode rollout)
│   │   └── sac_v3_rollout_best.gif
│   ├── reach_ppo_v2/          # PPO baseline (DR 없음) - 86%
│   ├── reach_sac_v3/          # SAC baseline (DR 없음) - 86%
│   ├── reach_ppo_dr/          # PPO + DR (sim-to-real용 권장)
│   └── reach_sac_dr/          # SAC + DR
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 2. Safe Zone — 우리가 설정한 작업 공간

xArm6 base frame 기준, **사용자가 컨트롤러에서 등록한 안전 박스**입니다. 시뮬과 실제에서 **동일한 좌표**를 사용합니다.

```
좌표계: base frame, 단위 meter, +Z = 위쪽

    z (위)
    ▲
    │       ┌──────────────┐  z = 0.60
    │       │              │
    │       │   SAFE BOX   │
    │       │              │
    │       └──────────────┘  z = 0.18
    │
    └─────────────────────────► x (앞)
            x=0           x=0.57

x:  0.00  ~ 0.57   (앞방향 0 ~ 570 mm)
y: -0.54  ~ 0.55   (좌우 -540 ~ 550 mm)
z:  0.18  ~ 0.60   (높이 180 ~ 600 mm)
```

### 어떻게 사용되나
1. **시뮬 환경의 target sampling**: WORKSPACE box (`[0.25, -0.30, 0.30] ~ [0.55, 0.30, 0.55]`)는 safe zone 안쪽으로 더 보수적으로 잡혀 있어 학습 시 정책이 자연스럽게 박스 안에서 동작하도록 유도합니다 ([reach_env.py:18](xarm_rl/envs/reach_env.py)).
2. **시뮬 reward — safe-zone penalty**: 매 step TCP가 safe box를 벗어나면 reward에서 **-1.0** 차감 ([reach_env.py:32](xarm_rl/envs/reach_env.py)). 정책이 박스를 자발적으로 회피하도록 학습됨.
3. **실제 배포 — hard guard**: deploy_real.py 루프 매 step마다 현재 TCP가 박스 밖이면 `[SAFETY] STOPPING` 출력 후 즉시 종료 ([deploy_real.py](scripts/deploy_real.py)). 시뮬 학습이 박스 안에서 끝났더라도 sim-to-real 차이로 박스를 벗어날 수 있어 hard guard가 마지막 안전망입니다.

### 변경하려면
실제 워크스페이스 변경 시 **세 곳**을 같이 수정해야 합니다:
- [xarm_rl/envs/reach_env.py](xarm_rl/envs/reach_env.py) — `SAFE_LOW`, `SAFE_HIGH`, `WORKSPACE_LOW`, `WORKSPACE_HIGH`
- [scripts/deploy_real.py](scripts/deploy_real.py) — `SAFE_LOW_M`, `SAFE_HIGH_M`
- xArm Studio (제조사 GUI) → Settings → Safety → workspace box

---

## 3. Domain Randomization (DR)

Sim-to-real gap을 줄이기 위해 학습 시 다음을 랜덤화 합니다 (episode별 고정). `--domain_rand` 플래그로 활성화.

| 항목 | 범위 | 의도 |
|---|---|---|
| Link mass (link1..link6) | ±20 % | 실제 무게/적재 차이 대응 |
| Joint friction loss | ×0.5 ~ ×1.5 | 마찰/마모 차이 대응 |
| Actuator PD gains (kp, kv) | ±30 % | 시뮬 PD ≠ 실제 컨트롤러 PD |
| Joint position obs noise | σ = 0.5° (0.0087 rad) | 엔코더 노이즈 |
| Action latency | 0 ~ 2 step (랜덤, episode별 고정) | 통신/제어 지연 |

구현: [reach_env.py:`_apply_domain_rand`](xarm_rl/envs/reach_env.py). 매 `reset()`에서 새 파라미터 샘플링.

> ⚠️ DR을 켜면 학습이 약간 느려지고 최종 성공률이 baseline보다 살짝 낮을 수 있지만, **실제 로봇에서의 동작 안정성**이 크게 올라갑니다.

---

## 4. 환경 (Reach) 상세

| 항목 | 값 |
|---|---|
| Env ID | `XArm6Reach-v0` |
| obs (21d) | `[joint_pos(6), joint_vel(6), ee_pos(3), target_pos(3), target-ee(3)]` |
| action (6d) | joint deltas ∈ [-1, 1], scale 0.05 rad/step |
| control freq | 50 Hz (frame_skip 10 × 0.002s) |
| max_episode_steps | 100 (2초 = 50Hz × 100) |
| reward | `-dist - 0.001·‖a‖² + 10·success - 1·outside_safe_zone` |
| success | TCP↔target 거리 < 3 cm |
| target 샘플링 영역 | `x: 0.25~0.55, y: -0.30~0.30, z: 0.30~0.55 m` |

---

## 5. 설치

### 시스템 요구사항
- Linux (Ubuntu 20.04+)
- Python **3.10+** (mujoco 3.x 요구)
- (선택) NVIDIA GPU — CPU만으로도 학습 가능

### 설치 절차
```bash
git clone <repo-url> xArm-project
cd xArm-project

# Python 3.11 venv (anaconda의 python3.11이 있으면 사용)
/usr/bin/python3.11 -m venv .venv   # 경로는 환경에 맞게
source .venv/bin/activate
pip install --upgrade pip

# 학습/시뮬만
pip install -e .

# 실제 xArm6 배포까지
pip install -e ".[real]"            # xArm-Python-SDK 포함
```

### 설치 확인
```bash
python -c "import mujoco, gymnasium, stable_baselines3, xarm_rl; \
           print('mujoco', mujoco.__version__); \
           import gymnasium as gym; e=gym.make('XArm6Reach-v0'); \
           print('env OK, obs', e.reset(seed=0)[0].shape)"
```
→ `env OK, obs (21,)` 가 나오면 성공.

### SpaceMouse + RealSense + LeRobot 데이터 수집 환경

VLA fine-tuning용 LeRobot 포맷 데이터 수집은 새 스크립트
`space_mouse/space_telecontrol.py`에서 처리한다. 기존 RL 학습/배포 코드는 건드리지 않는다.

Windows에서 `uv`와 현재 `.venv`를 쓰는 경우:

```powershell
cd C:\Users\M207\Desktop\kang\code_factory\physical-ai-vanila

# 이미 venv가 있다면 activate만 하면 된다.
.\physical-ai-vanila\.venv\Scripts\activate

# xArm SDK 포함. 기존 프로젝트 real extra.
uv pip install -e '.\physical-ai-vanila[real]'

# SpaceMouse, RealSense, OpenCV, LeRobot writer.
uv pip install pyspacemouse pyrealsense2 opencv-python lerobot
```

필요 장치/패키지:

- `xArm-Python-SDK`: xArm6 연결 및 Cartesian velocity 제어
- `pyspacemouse`: SpaceMouse 입력
- `pyrealsense2`: Intel RealSense D435 wrist camera
- `opencv-python`: RealSense BGR frame을 LeRobot용 RGB로 변환
- `lerobot`: `LeRobotDataset.create()` / `add_frame()` / `save_episode()` / `finalize()`

`lerobot` repo 위치:

- inner 프로젝트 폴더(`physical-ai-vanila/physical-ai-vanila`) 안에 `lerobot` repo를 넣을 필요는 없다.
- 현재 권장 구조는 sibling layout이다.

```text
C:\Users\M207\Desktop\kang\code_factory\physical-ai-vanila\
  lerobot\                 # optional local LeRobot checkout
    src\lerobot\...
  physical-ai-vanila\
    space_mouse\
      space_telecontrol.py
```

`space_telecontrol.py`는 실행 시 `..\lerobot\src`가 있으면 그것을 먼저 import한다. 이 로컬 checkout이 없으면 `.venv`에 설치된 `lerobot` 패키지를 사용한다. LeRobot v3 writer API를 맞춰 쓰는 목적이면 현재처럼 sibling `lerobot` checkout을 유지하는 쪽이 가장 안전하다.

실행:

```powershell
cd C:\Users\M207\Desktop\kang\code_factory\physical-ai-vanila\physical-ai-vanila

# 조종만: 카메라/LeRobot writer는 켜지지 않는다.
python space_mouse/space_telecontrol.py `
  --ip 192.168.1.199 `
  --fps 10

# 조종 + LeRobot 포맷 기록: RealSense wrist RGB도 함께 연결된다.
python space_mouse/space_telecontrol.py `
  --ip 192.168.1.199 `
  --fps 10 `
  --record `
  --repo-id kangkang9412/xarm6_spacemouse_demo `
  --root ./data/xarm6_spacemouse_demo `
  --task "pick up the object"
```

`--fps 10`은 로봇 조종 속도가 아니라 데이터 저장 주기다. 기본 제어 루프는 `--control-hz 30`이라 SpaceMouse 입력과 xArm Cartesian velocity command는 초당 30회 처리되고, LeRobot frame 저장만 초당 10회 수행된다. 더 촘촘한 데이터가 필요하면 `--fps 15` 또는 `--fps 20`으로 올릴 수 있지만, 용량과 비디오 인코딩 비용도 같이 늘어난다.

`--record`가 켜졌을 때 저장되는 LeRobot feature:

```text
observation.images.wrist   RGB video, uint8, 480x640x3
observation.state          float32[7], current TCP 6D pose + gripper
action                     float32[7], target TCP 6D pose + gripper
task                       add_frame()에 같이 전달되는 task string
```

---

## 6. 학습 프로세스

### 6.1 권장 명령 (실제 배포용 — Domain Randomization ON)

```bash
# PPO + DR (안정적인 baseline, ~45분 CPU)
python scripts/train.py --task reach --algo ppo \
    --n_envs 16 --timesteps 1500000 --seed 7 --domain_rand \
    --out outputs/reach_ppo_dr

# SAC + DR (sample-efficient, ~60분 CPU)
python scripts/train.py --task reach --algo sac \
    --n_envs 1 --timesteps 250000 --seed 23 --domain_rand \
    --out outputs/reach_sac_dr
```

### 6.2 출력
```
outputs/reach_ppo_dr/
├── final_model.zip       # 최종 정책
├── ckpts/                # 중간 체크포인트 (50k 간격)
├── monitor.csv           # episode 로그
└── tb/                   # TensorBoard logs (학습 곡선)
```

학습 곡선:
```bash
tensorboard --logdir outputs/
```

### 6.3 평가 (디스플레이 X)
```bash
python scripts/eval_headless.py --task reach --algo ppo \
    --model outputs/reach_ppo_dr/final_model.zip \
    --episodes 50 --out_json outputs/reach_ppo_dr/eval.json
```
출력 JSON: `success_rate`, `mean_reward`, `mean_final_dist`, `mean_ep_len`.

**기준**: success_rate ≥ 80% 면 배포 시도 가능. 미만이면:
- (1) 다른 체크포인트 평가 (`ckpts/*.zip` 중 best 선택 — SAC는 진동이 있어 final이 best가 아닐 수 있음)
- (2) seed/timesteps 조정해 재학습

### 6.4 9-point Grid Tour 데모 (시각화)

학습된 정책의 정확도/일관성을 한눈에 보기 위해 **safe zone 안에 3×3 = 9개 균일 격자 점**을 정의하고, `home → P0 → home → P1 → … → P8 → home` 순서로 한 점씩 reach + 복귀를 반복하는 데모 GIF를 만듭니다. 매 segment마다 환경을 home으로 리셋해 학습 분포에 맞춥니다.

```bash
python scripts/demo_grid_tour.py --algo ppo \
    --model outputs/reach_ppo_v2/final_model.zip \
    --out outputs/gif/ppo_v2_grid_tour.gif
```

9개 점 좌표 (단위 m, base frame):
```
GRID_X = [0.32, 0.42, 0.52]
GRID_Y = [-0.20, 0.00, 0.20]
GRID_Z = 0.45    (고정 평면)
```

GIF 시각 요소:
- 🔴 빨강 = 현재 활성 target
- 🟢 초록 = 이미 다녀온 점
- ⚫ 회색 = 아직 안 간 점

크기 조절: `--width 480 --height 360 --render_every 4` 로 작게 (7 MB), 기본 640×480·every=2 면 19 MB.

PPO v2 baseline (86% success) 결과: **9/9 점 모두 도달**

![grid tour](outputs/gif/ppo_v2_grid_tour_small.gif)

> ❓ **왜 처음엔 1/9 만 됐다가 수정 후 9/9?**
> 처음 데모는 한 점 reach 후 환경을 reset하지 않고 다음 target만 바꿔서, 정책이 학습 시 본 적 없는 state(home에서 멀리 떨어진 자세)에 빠지면서 out-of-distribution이 됐습니다. 정책은 "home 근처 시작 → 임의 target" 분포로만 학습됐기 때문이죠. 수정 후엔 매 segment 시작 시 `env.reset()` 으로 home 복귀 → 학습 분포 일치 → 9/9 성공.
> (실제 배포에서도 매 reach 사이클을 home 근처에서 시작하면 가장 안정적입니다.)

### 6.5 단일 에피소드 GIF 만들기 (헤드리스 OK)

랜덤 target 3 에피소드를 rollout하여 짧은 GIF를 생성합니다 — 일반적인 성공률/실패 케이스 보기용.
```bash
# PPO baseline
python scripts/render_gif.py --task reach --algo ppo \
    --model outputs/reach_ppo_v2/final_model.zip \
    --out outputs/gif/ppo_v2_rollout.gif \
    --episodes 3 --width 480 --height 360 --fps 30

# SAC baseline best
python scripts/render_gif.py --task reach --algo sac \
    --model outputs/reach_sac_v3/best_model.zip \
    --out outputs/gif/sac_v3_rollout_best.gif --episodes 3
```
MuJoCo EGL backend로 GPU 디스플레이 없이 렌더링됩니다.

샘플 결과 (3 에피소드 각각):

| PPO v2 (86%) | SAC v3 best (86%) |
|---|---|
| ![ppo](outputs/gif/ppo_v2_rollout.gif) | ![sac](outputs/gif/sac_v3_rollout_best.gif) |

### 6.6 하이퍼파라미터 요약 (이미 코드에 반영)

| | PPO (in `build_ppo()`) | SAC (in `build_sac()`) |
|---|---|---|
| lr | 5e-4 | 3e-4 |
| n_steps / buffer | n_steps=256 | buffer_size=500k |
| batch | 256 | 256 |
| gamma | 0.98 | 0.95 |
| 핵심 | `ent_coef=0.005` | `ent_coef="auto_0.1"`, `target_entropy=-6.0` |
| net | [256, 256] | [256, 256] |

자세한 분석은 [outputs/report.md](outputs/report.md) 참고.

---

## 7. 배포 프로세스 (실제 xArm6, UFactory)

### 7.0 사전 점검 (10분)
- [ ] xArm6 컨트롤러 박스 부팅, **e-stop 버튼 손에 쥐기** (모든 실제 실행 동안)
- [ ] xArm Studio (제조사 GUI)로 연결 → 펌웨어 최신
- [ ] xArm Studio에서 수동 동작 (joint 각각 점프, home 이동) 확인
- [ ] **xArm Studio → Settings → Safety**: workspace box를 `x: 0~570, y: -540~550, z: 180~600 mm`로 등록
- [ ] **노트북에 정책 파일 복사**: 학습 머신의 `outputs/reach_ppo_dr/final_model.zip`를 노트북으로
- [ ] 팔 주변 충돌물 제거, 반경 1m 사람 없음 확인

**컨트롤러 기본 설정 (본 셋업)**: IP `192.168.1.199`, Modbus TCP 포트 `502` (XArmAPI 내부 기본). 배포 스크립트의 `--ip` 기본값으로 들어가 있어 보통은 생략 가능.

### 7.1 노트북 ↔ 컨트롤러 네트워크 (5분)
```bash
# 노트북 유선 LAN을 컨트롤러와 직결 (또는 같은 스위치)
# 노트북 IP를 같은 서브넷에 고정
sudo ip addr add 192.168.1.10/24 dev eth0       # Linux 예시
# Windows: 네트워크 → 어댑터 설정 → IPv4 수동 192.168.1.10 / 255.255.255.0

# 통신 확인 — 본 셋업 컨트롤러: IP 192.168.1.199, Modbus TCP 포트 502
ping 192.168.1.199
```

### 7.2 노트북 환경 (10분)
```bash
git clone <repo-url> xArm-project    # 또는 USB로 복사
cd xArm-project

# uv 사용 (Windows/Linux 공통)
uv venv .venv --python 3.11

# 활성화 — 셸별로 다름
source .venv/bin/activate            # Linux / macOS
source .venv/Scripts/activate        # Windows + Git Bash (MINGW64)
# .venv\Scripts\activate             # Windows + PowerShell/cmd

uv pip install -e ".[real]"          # mujoco + sb3 + xArm-Python-SDK 한 번에
```

### 7.3 통신 sanity — 모터 안 움직임 (2분)
```bash
# 컨트롤러 상태/에러 읽기만
python -c "from xarm.wrapper import XArmAPI; a=XArmAPI('192.168.1.199'); print('state=', a.get_state()); print('err_warn=', a.get_err_warn_code()); print('q=', a.get_servo_angle(is_radian=True)); a.disconnect()"
```
- `err_warn=(0, [0, 0])` 면 컨트롤러 정상
- `[0, 0]` 가 아니면 §7.6의 **에러 클리어** 먼저 수행

### 7.4 미세 모션 진단 — SDK 호출 sanity (1분)
SDK 호출 → 모션 체인이 실제로 동작하는지 joint1을 **0.57° (~0.01 rad)** 만 살짝 움직여 확인:
```bash
python scripts/diag_servo.py
```
- `return code : 0` + joint1이 살짝 움직이면 통신/모션 OK
- code != 0 또는 안 움직이면 §7.6 참고

### 7.5 9-point Grid Tour 실제 배포 ⭐ (안전 우선 모드)

시뮬에서 9/9 성공한 데모 시퀀스를 **실제 xArm6**에서 실행합니다. 매 target마다:
1. **position mode로 HOME_QPOS_RAD 정확 복귀** (학습 분포 보장)
2. servo mode 전환 → 정책으로 P_i reach
3. dwell → 다음 target

```bash
# Dry-run (실제 모터 X, 스크립트 sanity만)
python scripts/deploy_grid_tour.py \
    --model outputs/reach_ppo_v2/final_model.zip --dry-run

# 첫 실제 실행 — 매우 보수적 (E-stop 손에)
python scripts/deploy_grid_tour.py \
    --model outputs/reach_ppo_v2/final_model.zip \
    --home-speed 0.15 \
    --max-step-rad 0.005 \
    --dwell 2.0
```
> ⚠️ **`--max-step-rad`는 P로 가는 속도를 결정**합니다. joint 속도 = `max-step-rad × hz`. 0.005면 ~5.7°/s, TCP 6~10 cm/s. 한 사이클 6~10분. 충분히 안전한 걸 확인한 후 §7.5의 단계 표대로 점진적 증가.

**속도 손잡이 (servo mode에서 `--speed`/`--hz`만으론 안 됨 — 아래 두 개가 실제로 적용됨)**

| 인자 | 기본 | 의미 | 첫 실행 권장 |
|---|---|---|---|
| `--home-speed` | 0.15 rad/s | position mode로 HOME 이동 속도 | **0.10~0.15** (~5.7~8.6°/s) |
| `--max-step-rad` | 0.015 rad | step당 joint 변화 hard cap → joint 속도 = 값 × hz | **0.005~0.015** |
| `--collision-sensitivity` | 1 | xArm 충돌 감지 민감도 (0~5) | **1** (false trigger 회피) |
| `--dwell` | 1.0 s | 각 target 도달 후 대기 | **2.0** (관찰 시간 확보) |
| `--hz` | 20 | 정책 명령 송신 frequency | 20 |
| `--speed` | 30 | servo mode에서 **무시됨** (호환용) | 신경 X |

**한 사이클 시간**: 약 **2~3분** (9개 target, 각 home_reset + reach + dwell). 매번 position mode 복귀가 들어가 안전하지만 약간 느립니다.

**특징**
- 매 target 전 **컨트롤러가 직접** HOME_QPOS_RAD로 복귀 → drift 누적 없음
- 매 step **safe-zone hard guard**: TCP가 박스 밖이면 즉시 segment 종료
- **Ctrl+C 안전 정지**: 어디서 누르든 `set_state(4)` + `disconnect()` 보장

**속도 단계적 증가 (각 단계 1 사이클 성공 후 다음)**

| 단계 | `--home-speed` | `--max-step-rad` |
|---|---|---|
| 1차 (관찰) | 0.10 | 0.005 |
| **2차 (권장 첫 실행)** | **0.15** | **0.015** |
| 3차 (검증 후) | 0.30 | 0.030 |
| 4차 (시연 속도) | 0.50 | 0.050 |

### 7.6 문제 발생 대응

**(a) 에러/충돌 발생 시 — 컨트롤러 에러 클리어**
```bash
python -c "from xarm.wrapper import XArmAPI; import time; a=XArmAPI('192.168.1.199'); print('before:', a.get_err_warn_code(), a.get_state()); a.clean_error(); a.clean_warn(); a.motion_enable(True); a.set_mode(0); a.set_state(0); time.sleep(0.3); print('after :', a.get_err_warn_code(), a.get_state()); a.disconnect()"
```
`after: (0, [0, 0]) (0, 0)` 나오면 복구 완료. 안 사라지면 xArm Studio GUI에서 Clear 필수.

**(b) 팔이 벽/물체에 끼어있을 때**

xArm Studio → **Manual Mode 켜기** → 손으로 안전한 자세로 빼내기 → Position Mode 전환 → Clear → Enable.
모터 enable 상태에서 강제로 손으로 풀려 하면 모터 손상 가능.

**(c) 증상별 표**

| 증상 | 원인 후보 | 대응 |
|---|---|---|
| 시작 모션(HOME 이동)이 너무 빠름 | `--home-speed` 높음 | `--home-speed 0.10` |
| 정책 reach가 느리거나 못 도달 | `--max-step-rad` 너무 작음 → 정책 방향 왜곡 | `--max-step-rad 0.015 → 0.03` 단계 증가 |
| 매 target 같은 자세에서 출발 안 됨 | (현재 코드는 자동 hard reset함 — 발생 시 버그 보고) | (해당 없음) |
| collision 즉시 발동 | sensitivity 높음 + 정책 첫 step current spike | `--collision-sensitivity 1` (기본) |
| code=1 거부 spam | mode/state 깨짐 또는 joint limit boundary | §7.6 (a) 클리어 후 재실행 |
| Safe-zone 침범 즉시 중단 | 정상 (가드 작동) | target/home 좌표 확인 |
| 안 움직임 | motor disable / 에러 상태 | §7.6 (a) 클리어 |

### 7.7 좌표계 캘리브레이션 (필요시)
1. xArm Studio에서 home에 둔 뒤 `arm.get_servo_angle()` 출력 → 예: `[0, -17.2, -68.8, 0, 86.0, 0]` deg
2. [base_env.py:25](xarm_rl/envs/base_env.py) `HOME_QPOS`와 비교 (rad 단위)
3. 차이 있으면 deploy 스크립트의 `HOME_QPOS_RAD` 상수 수정 ([deploy_grid_tour.py:48](scripts/deploy_grid_tour.py#L48))

### 7.8 안전 / 운영 팁
- 노트북 ↔ 컨트롤러 **직결** (인터넷 노출 X)
- E-stop은 모든 실행 동안 손에 쥐기
- `--collision-sensitivity` 기본 1. **검증 끝나기 전엔 3 이상 금지** (false trigger로 중단되어도 OK, 충돌로 망가지는 것보단 나음)
- 매 step 로깅: `q`, TCP, action, info → 사후 분석용 (자체 추가 권장)
- 컨트롤러 자체 로그도 xArm Studio → Logs에서 별도 보관
- 첫 사이클 끝까지 성공한 뒤에만 속도 단계적 증가 (§7.5의 단계 표)

---

## 8. 트러블슈팅

| 문제 | 해결 |
|---|---|
| `mujoco.FatalError: OpenGL platform library has not been loaded` | `MUJOCO_GL=egl` 환경변수 설정 (헤드리스 렌더링) |
| `ModuleNotFoundError: mujoco` | venv 활성화 안 됨. `source .venv/bin/activate` |
| 학습이 plateau / reward 정체 | seed 변경, `--timesteps` 2배 증가, [outputs/report.md](outputs/report.md) 의 hyperparam 분석 참고 |
| GPU CUDA 에러 (driver too old) | torch가 CPU로 자동 fallback — CPU만으로도 본 태스크는 학습 가능 |
| 실제에서 jitter | action smoothing (지수 이동평균), `--hz` 30으로 ↑, action_scale ↓ |

---

## 9. 라이선스 / Credits

- xArm6 URDF/STL 메쉬: `dynamic_handover` repo에서 추출 (UFactory의 공식 xacro 기반)
- xArm Python SDK: UFactory 공식 [xArm-Python-SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
- 학습 프레임워크: [Stable-Baselines3](https://stable-baselines3.readthedocs.io/), [Gymnasium](https://gymnasium.farama.org/), [MuJoCo](https://mujoco.org/)
