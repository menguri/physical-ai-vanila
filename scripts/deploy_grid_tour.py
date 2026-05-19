"""Real xArm6 grid-tour deployment.

Visits 9 evenly-spaced points in the safe zone, returning to a home pose between
each. Same 3x3 layout as the simulation demo (scripts/demo_grid_tour.py).

USAGE (start carefully):

    # 1. Dry-run: prints planned actions, NO real motion
    python scripts/deploy_grid_tour.py \
        --model outputs/reach_ppo_v2/final_model.zip \
        --ip 192.168.1.185 --dry-run

    # 2. Real run, conservative speed
    python scripts/deploy_grid_tour.py \
        --model outputs/reach_ppo_v2/final_model.zip \
        --ip 192.168.1.185 --speed 30 --hz 20

SAFETY
- E-stop must be in reach at all times.
- Hard safe-zone guard is applied every step. If the TCP exits the box, the
  loop stops immediately.
- Start with --speed 20-30, then gradually increase only after the full 9-point
  cycle has succeeded.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    XArmAPI = None


# -----------------------------------------------------------------------------
# Grid layout — MUST match scripts/demo_grid_tour.py (same coords, same order).
GRID_X = [0.32, 0.42, 0.52]
GRID_Y = [-0.20, 0.00, 0.20]
GRID_Z = 0.45
HOME_TARGET = np.array([0.42, 0.00, 0.55], dtype=np.float32)

# Home joint pose (radians) — must match xarm_rl/envs/base_env.py:HOME_QPOS
HOME_QPOS_RAD = np.array([0.0, -0.3, -1.2, 0.0, 1.5, 0.0], dtype=np.float32)

# Real xArm6 safe zone (meters, base frame). Hard guard at deploy time.
SAFE_LOW_M  = np.array([0.000, -0.540, 0.180], dtype=np.float32)
SAFE_HIGH_M = np.array([0.570,  0.550, 0.600], dtype=np.float32)

# Joint limits (rad) — same as env (xarm_rl/envs/base_env.py)
JOINT_LIMITS_LOW  = np.array([-6.283, -2.059, -3.927, -6.283, -1.693, -6.283], dtype=np.float32)
JOINT_LIMITS_HIGH = np.array([ 6.283,  2.094,  0.191,  6.283,  3.142,  6.283], dtype=np.float32)

SUCCESS_DIST = 0.03  # 3 cm
ALGO_CLS = {"ppo": PPO, "sac": SAC}


# -----------------------------------------------------------------------------
def build_targets():
    """Snake-order 3x3 grid identical to demo_grid_tour.py."""
    pts = []
    for i, y in enumerate(GRID_Y):
        xs = GRID_X if i % 2 == 0 else list(reversed(GRID_X))
        for x in xs:
            pts.append(np.array([x, y, GRID_Z], dtype=np.float32))
    return pts


def in_safe_zone(ee_pos_m: np.ndarray) -> bool:
    return bool(np.all(ee_pos_m >= SAFE_LOW_M) and np.all(ee_pos_m <= SAFE_HIGH_M))


# -----------------------------------------------------------------------------
class FakeArm:
    """Stand-in for --dry-run mode. Mirrors only the XArmAPI methods we use."""
    def __init__(self):
        self._q = HOME_QPOS_RAD.copy()
        self._tcp_mm = np.array([420.0, 0.0, 550.0], dtype=np.float32)  # m -> mm

    def motion_enable(self, *a, **kw): pass
    def set_mode(self, *a, **kw): pass
    def set_state(self, *a, **kw): pass
    def set_collision_sensitivity(self, *a, **kw): pass
    def set_self_collision_detection(self, *a, **kw): pass

    def move_gohome(self, **kw):
        self._q = HOME_QPOS_RAD.copy()
        self._tcp_mm[:] = [420, 0, 550]
        return 0

    def set_servo_angle(self, angle, **kw):
        self._q = np.array(angle, dtype=np.float32)
        return 0

    def set_servo_angle_j(self, angles, **kw):
        self._q = np.array(angles, dtype=np.float32)
        # Pretend TCP follows joint commands with a tiny Cartesian step (very rough proxy).
        # Real arm would compute FK internally — for dry-run sanity only.
        return 0

    def get_servo_angle(self, is_radian=False):
        v = self._q if is_radian else np.rad2deg(self._q)
        return 0, list(v) + [0.0]

    def get_position(self, is_radian=False):
        return 0, list(self._tcp_mm) + [0.0, 0.0, 0.0]

    def disconnect(self): pass


# -----------------------------------------------------------------------------
def build_obs(q_rad: np.ndarray, qd_rad: np.ndarray, ee_pos_m: np.ndarray,
              target_m: np.ndarray) -> np.ndarray:
    """Build the same 21-d observation the policy was trained with."""
    diff = target_m - ee_pos_m
    return np.concatenate([q_rad, qd_rad, ee_pos_m, target_m, diff]).astype(np.float32)


def read_state(arm, dry_run: bool):
    """Return (q_rad[6], ee_pos_m[3]). Works for both FakeArm and real XArmAPI."""
    _, q_list = arm.get_servo_angle(is_radian=True)
    q = np.array(q_list[:6], dtype=np.float32)

    if dry_run:
        # FakeArm: no real FK. Use a flat proxy TCP that just advances toward target
        # so safe-zone guard logic still gets exercised.
        ee = arm._tcp_mm / 1000.0
    else:
        _, tcp = arm.get_position(is_radian=True)
        ee = np.array(tcp[:3], dtype=np.float32) / 1000.0  # mm -> m
    return q, ee


def run_segment(arm, policy, target_xyz: np.ndarray, args, label: str) -> tuple[bool, float]:
    """Drive the arm with the policy toward target_xyz until success or max_steps."""
    dt = 1.0 / args.hz
    prev_q = None
    ok = False
    last_dist = float("inf")

    for step in range(args.max_steps):
        t0 = time.time()
        q, ee = read_state(arm, args.dry_run)
        qd = np.zeros(6, dtype=np.float32) if prev_q is None else (q - prev_q) / dt
        prev_q = q

        # Safe-zone hard guard
        if not in_safe_zone(ee):
            print(f"  [{label}] [SAFETY] TCP {ee} outside safe zone — STOPPING segment")
            return False, float(np.linalg.norm(target_xyz - ee))

        obs = build_obs(q, qd, ee, target_xyz)
        action, _ = policy.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)[:6]

        target_q = np.clip(q + action * args.action_scale, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

        if args.dry_run:
            arm.set_servo_angle_j(target_q.tolist(), is_radian=True)
            # Step the fake TCP toward goal so the loop terminates in dry-run
            ee_new = ee + 0.3 * (target_xyz - ee) * dt
            arm._tcp_mm = (ee_new * 1000).astype(np.float32)
            print(f"  [{label}] step {step:03d}  ee={np.round(ee, 3)}  "
                  f"d={np.linalg.norm(target_xyz - ee):.3f}m")
        else:
            arm.set_servo_angle_j(
                target_q.tolist(),
                is_radian=True,
                speed=np.deg2rad(args.speed * 3),
                mvacc=np.deg2rad(args.speed * 6),
            )

        last_dist = float(np.linalg.norm(target_xyz - ee))
        if last_dist < SUCCESS_DIST:
            ok = True
            break

        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print(f"  [{label}] reached={ok}  d={last_dist:.3f}m  steps={step+1}")
    return ok, last_dist


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained policy .zip")
    ap.add_argument("--algo", choices=list(ALGO_CLS), default="ppo")
    ap.add_argument("--ip", required=True, help="xArm controller IP")
    ap.add_argument("--speed", type=int, default=30, help="0-100, xArm servo speed")
    ap.add_argument("--hz", type=float, default=20.0, help="control loop frequency")
    ap.add_argument("--max-steps", type=int, default=200,
                    help="max policy steps per segment")
    ap.add_argument("--action-scale", type=float, default=0.05,
                    help="rad/step per joint (must match training)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dwell", type=float, default=1.0,
                    help="pause (sec) at each target before returning home")
    args = ap.parse_args()

    targets = build_targets()
    print(f"[deploy] {len(targets)} grid targets in safe zone:")
    for i, p in enumerate(targets):
        print(f"  P{i}: x={p[0]:.2f}  y={p[1]:+.2f}  z={p[2]:.2f}")

    if args.dry_run:
        arm = FakeArm()
        print("[deploy] DRY RUN — no real motion")
    else:
        if XArmAPI is None:
            raise SystemExit("xArm-Python-SDK not installed. pip install xArm-Python-SDK")
        arm = XArmAPI(args.ip, is_radian=True)
        arm.motion_enable(enable=True)
        arm.set_mode(1)            # servo motion (real-time joint streaming)
        arm.set_state(state=0)
        arm.set_collision_sensitivity(3)   # 0..5, higher = more sensitive
        arm.move_gohome(wait=True)
        time.sleep(0.5)

    policy = ALGO_CLS[args.algo].load(args.model, device="auto")

    successes = 0
    for i, p in enumerate(targets):
        print(f"\n>>> Target P{i} = {p}")
        # 1) Go to HOME (matches training distribution — see README §6.4 box)
        run_segment(arm, policy, HOME_TARGET, args, label=f"home_before_P{i}")
        # 2) Reach target
        ok, _ = run_segment(arm, policy, p, args, label=f"P{i}")
        if ok:
            successes += 1
        # 3) Optional dwell so an observer can see it landed
        if args.dwell > 0:
            time.sleep(args.dwell)

    print(f"\n>>> Returning HOME")
    run_segment(arm, policy, HOME_TARGET, args, label="home_final")

    print(f"\n[deploy] {successes}/{len(targets)} targets reached")
    if not args.dry_run:
        arm.disconnect()


if __name__ == "__main__":
    main()
