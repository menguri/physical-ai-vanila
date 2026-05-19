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

### 6.4 GIF 만들기 (헤드리스 OK)
```bash
python scripts/render_gif.py --task reach --algo ppo \
    --model outputs/reach_ppo_dr/final_model.zip \
    --out outputs/reach_ppo_dr/rollout.gif \
    --episodes 3 --width 480 --height 360 --fps 30
```
MuJoCo EGL backend로 GPU 디스플레이 없이 렌더링됩니다.

### 6.5 하이퍼파라미터 요약 (이미 코드에 반영)

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
- [ ] xArm6 컨트롤러 박스 부팅, e-stop 버튼 위치 확보
- [ ] xArm Studio (제조사 GUI)로 연결 → 펌웨어 최신
- [ ] xArm Studio에서 수동 동작 (joint 각각 점프, home 이동) 확인
- [ ] **xArm Studio → Settings → Safety**: workspace box를 `x: 0~570, y: -540~550, z: 180~600 mm`로 등록
- [ ] **노트북에 정책 파일 복사**: 학습 머신의 `outputs/reach_ppo_dr/final_model.zip`를 노트북으로

### 7.1 노트북 ↔ 컨트롤러 네트워크 (5분)
```bash
# 노트북 유선 LAN을 컨트롤러와 직결 (또는 같은 스위치)
# 노트북 IP를 같은 서브넷에 고정
sudo ip addr add 192.168.1.10/24 dev eth0       # Linux 예시
# Windows: 네트워크 → 어댑터 설정 → IPv4 수동 192.168.1.10 / 255.255.255.0

# 통신 확인 (컨트롤러 기본 IP가 보통 192.168.1.185 — 실제 IP는 본인 컨트롤러 확인)
ping 192.168.1.185
```

### 7.2 노트북 환경 (10분)
```bash
git clone <repo-url> xArm-project    # 또는 USB로 복사
cd xArm-project
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[real]"             # mujoco + sb3 + xArm-Python-SDK 한 번에
```

### 7.3 Dry-run — 정책 sanity (실제 모터 X, 5분)
```bash
python scripts/deploy_real.py --task reach \
    --model outputs/reach_ppo_dr/final_model.zip \
    --ip 192.168.1.185 \
    --target 0.45 0.0 0.55 \
    --dry-run
```
출력에서 매 step `q=[..] a=[..] ee=[..]` 가 합리적인 범위로 변하면 OK.

### 7.4 실제 동작 — 보수적 시작 (15분)
```bash
# 첫 시도: 30% 속도, 20Hz control
python scripts/deploy_real.py --task reach \
    --model outputs/reach_ppo_dr/final_model.zip \
    --ip 192.168.1.185 \
    --target 0.45 0.0 0.55 \
    --speed 30 --hz 20
```

스크립트 내부 동작:
1. `XArmAPI(ip)` 연결 → `motion_enable(True)` → `set_mode(1)` (servo mode) → `set_state(0)` → `move_gohome()`
2. 매 1/hz 초:
   - `arm.get_servo_angle(is_radian=True)` → joint q
   - `arm.get_position(is_radian=True)` → TCP xyz (mm → m 변환)
   - obs 구성 → `policy.predict(obs)` → action
   - **Safe zone 가드**: TCP 박스 밖이면 즉시 중단
   - `arm.set_servo_angle_j(target_q, is_radian=True, speed=…)`
3. dist < 3 cm 도달 시 종료

### 7.5 점진적 검증 시나리오
| Step | target | 의도 |
|---|---|---|
| 1 | 5 cm 짧은 이동 | 안전 검증 |
| 2 | workspace 중앙 | 보통 케이스 |
| 3 | workspace 모서리 | 가장자리 케이스 |
| 4 | 10개 랜덤 target | success rate 측정 |

### 7.6 문제 발생 대응
| 증상 | 원인 후보 | 대응 |
|---|---|---|
| 떨림 / jitter | action_scale ↑, control rate 부족 | `--action-scale 0.03 --hz 30` |
| 안 움직임 | servo mode 실패, 모터 disable | xArm Studio에서 enable 확인 |
| 잘못된 방향 | 시뮬 ↔ 실제 좌표계 차이 | HOME_QPOS 캘리브레이션 (아래) |
| Safe-zone 침범 즉시 중단 | 정상 (가드 작동) | target 조정 |

### 7.7 좌표계 캘리브레이션 (필요시)
1. xArm Studio에서 home에 둔 뒤 `arm.get_servo_angle()` 출력 → 예: `[0, -17.2, -68.8, 0, 86.0, 0]` deg
2. [base_env.py:25](xarm_rl/envs/base_env.py) `HOME_QPOS`와 비교 (rad 단위)
3. 차이 있으면 deploy 시 offset 추가

### 7.8 안전 / 운영 팁
- 노트북 ↔ 컨트롤러 **직결** (인터넷 노출 X)
- xArm Studio에서 `set_collision_sensitivity(3)` 이상 권장 (충돌 시 즉시 정지)
- 매 step 로깅: `q`, TCP, action, info → 사후 분석용 (자체 추가 권장)
- 컨트롤러 자체 로그도 xArm Studio → Logs에서 별도 보관

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
