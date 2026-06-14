#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${PROJECT_ROOT}"

"${SCRIPT_DIR}/05_release_cameras.sh"

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
  --actions-per-chunk 20 \
  --chunk-size-threshold 0.5 \
  --width 640 \
  --height 480 \
  --camera-fps 30 \
  --wrist-camera-serial "${WRIST_CAMERA_SERIAL}" \
  --servo-speed 35 \
  --servo-acc 300 \
  --max-pos-step-mm 4 \
  --max-rot-step-rad 0.025 \
  --max-gripper-step 50 \
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
  --diagnose \
  --no-motion \
  "${EXTRA_ARGS[@]}"
