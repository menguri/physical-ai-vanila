#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# example command: LOCAL_POLICY_PORT=18080 ACTION_MODE=delta ./sh_scripts/60_vla_real_eval.sh
# Real-eval knobs. Edit these lines, or override them at the command line.
export LOCAL_POLICY_PORT="${LOCAL_POLICY_PORT:-18080}"
export REMOTE_POLICY_PORT="${REMOTE_POLICY_PORT:-8080}"
export REMOTE_LEROBOT_ROOT="${REMOTE_LEROBOT_ROOT:-/home/mlic/mingukang/lerobot}"
export REMOTE_LEROBOT_CHECKOUT="${REMOTE_LEROBOT_CHECKOUT:-spacemouse-lerobot-recording}"

export TASK="${TASK:-pick up the yellow pencil sharpener and place it on the cardboard box}"

export ACTION_MODE="${ACTION_MODE:-delta}"
export POLICY_TYPE="${POLICY_TYPE:-pi05}"
export POLICY_PATH="${POLICY_PATH:-/home/mlic/mingukang/lerobot/outputs/train/pi05_real_task1_delta_expert_task1_delta_pi05_expert_20260616_140359/checkpoints/005000/pretrained_model}"
export POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

export ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-20}"
export CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
export SERVO_SPEED="${SERVO_SPEED:-35}"
export SERVO_ACC="${SERVO_ACC:-300}"
export MAX_POS_STEP_MM="${MAX_POS_STEP_MM:-4}"
export MAX_ROT_STEP_RAD="${MAX_ROT_STEP_RAD:-0.025}"
export MAX_GRIPPER_STEP="${MAX_GRIPPER_STEP:-50}"

source "${SCRIPT_DIR}/env.sh"

cd "${PROJECT_ROOT}"

"${SCRIPT_DIR}/05_release_cameras.sh"

LOCAL_CHECKOUT="$(git rev-parse --abbrev-ref HEAD)"
LOCAL_COMMIT="$(git rev-parse --short HEAD)"

echo "Real VLA eval config:"
echo "  local repo: ${PROJECT_ROOT}"
echo "  local checkout: ${LOCAL_CHECKOUT} ${LOCAL_COMMIT}"
echo "  remote: ${REMOTE}"
echo "  remote repo: ${REMOTE_LEROBOT_ROOT}"
echo "  remote checkout: ${REMOTE_LEROBOT_CHECKOUT:-current}"
echo "  tunnel: robot PC 127.0.0.1:${LOCAL_POLICY_PORT} -> ${REMOTE} 127.0.0.1:${REMOTE_POLICY_PORT}"
echo "  policy: ${POLICY_TYPE} ${POLICY_PATH}"
echo "  policy device: ${POLICY_DEVICE}"
echo "  action mode: ${ACTION_MODE}"
echo "  fps: ${FPS}"
echo "  chunk: actions_per_chunk=${ACTIONS_PER_CHUNK}, refresh_threshold=${CHUNK_SIZE_THRESHOLD}"
echo "  step limits: pos=${MAX_POS_STEP_MM}mm, rot=${MAX_ROT_STEP_RAD}rad, gripper=${MAX_GRIPPER_STEP}"
echo "  servo: speed=${SERVO_SPEED}, acc=${SERVO_ACC}"
echo "  task: ${TASK}"

EXTRA_ARGS=()
EXTRA_ARGS+=(
  --safe-x-mm 0 570
  --safe-y-mm -540 550
  --safe-z-mm 180 600
)
if [[ "${USE_FRONT_CAMERA}" == "1" ]]; then
  EXTRA_ARGS+=(--front-camera-serial "${FRONT_CAMERA_SERIAL}" --use-front-camera)
fi

python scripts/vla_xarm_client.py \
  --server-address "127.0.0.1:${LOCAL_POLICY_PORT}" \
  --policy-type "${POLICY_TYPE}" \
  --pretrained-name-or-path "${POLICY_PATH}" \
  --policy-device "${POLICY_DEVICE}" \
  --task "${TASK}" \
  --ip "${ROBOT_IP}" \
  --action-mode "${ACTION_MODE}" \
  --fps "${FPS}" \
  --actions-per-chunk "${ACTIONS_PER_CHUNK}" \
  --chunk-size-threshold "${CHUNK_SIZE_THRESHOLD}" \
  --width 640 \
  --height 480 \
  --camera-fps 30 \
  --wrist-camera-serial "${WRIST_CAMERA_SERIAL}" \
  --servo-speed "${SERVO_SPEED}" \
  --servo-acc "${SERVO_ACC}" \
  --max-pos-step-mm "${MAX_POS_STEP_MM}" \
  --max-rot-step-rad "${MAX_ROT_STEP_RAD}" \
  --max-gripper-step "${MAX_GRIPPER_STEP}" \
  --return-home-seconds 5 \
  --return-control-hz 100 \
  --diagnostic-dir outputs/vla_diagnostics \
  --action-log-steps 10 \
  --action-timeout-s 15 \
  --demo-stats data/TASK1/meta/stats.json \
  --enable-gripper \
  --gripper-min 0 \
  --gripper-max 850 \
  --gripper-speed 5000 \
  --collision-sensitivity 1 \
  "${EXTRA_ARGS[@]}"
