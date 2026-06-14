#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${PROJECT_ROOT}"

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
  --wrist-camera-serial "${WRIST_CAMERA_SERIAL}" \
  --front-camera-serial "${FRONT_CAMERA_SERIAL}" \
  --servo-speed "${SERVO_SPEED}" \
  --servo-acc "${SERVO_ACC}" \
  --max-pos-step-mm "${MAX_POS_STEP_MM}" \
  --max-rot-step-rad "${MAX_ROT_STEP_RAD}" \
  --max-gripper-step "${MAX_GRIPPER_STEP}" \
  --return-home-seconds "${RETURN_HOME_SECONDS}" \
  --diagnose \
  --no-motion

