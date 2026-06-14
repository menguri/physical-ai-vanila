# xarm_rl

`xarm_rl`은 이 repo의 보조 기능입니다. 현재 주 목적은 real xArm6 데이터 수집과 VLA real eval이고, 이 패키지는 MuJoCo 기반 PPO/SAC reach 실험을 남겨둔 영역입니다.

## 설치 확인

```bash
cd /home/mlic/data_collection/physical-ai-vanila
python -c "import gymnasium as gym; import xarm_rl; env=gym.make('XArm6Reach-v0'); print(env.reset(seed=0)[0].shape)"
```

정상이라면 `(21,)`가 출력됩니다.

## PPO 학습

```bash
python scripts/train.py --task reach --algo ppo \
  --n_envs 16 \
  --timesteps 1500000 \
  --seed 7 \
  --out outputs/reach_ppo_v2
```

## Headless 평가

```bash
python scripts/eval_headless.py --task reach --algo ppo \
  --model outputs/reach_ppo_v2/final_model.zip \
  --episodes 50 \
  --out_json outputs/reach_ppo_v2/eval.json
```

## GIF 생성

```bash
python scripts/demo_grid_tour.py --algo ppo \
  --model outputs/reach_ppo_v2/final_model.zip \
  --out outputs/gif/ppo_v2_grid_tour.gif
```

