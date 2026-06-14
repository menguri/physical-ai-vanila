#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${PROJECT_ROOT}"

python joy_stick/joy_telecontrol_serial.py \
  --ip "${ROBOT_IP}" \
  --fps "${FPS}" \
  --record \
  --repo-id "${REPO_ID}" \
  --root "${DATA_ROOT}" \
  --task-id "${TASK_ID}" \
  --task "${TASK}" \
  --wrist-camera-serial "${WRIST_CAMERA_SERIAL}" \
  --front-camera-serial "${FRONT_CAMERA_SERIAL}" \
  --action-mode "${ACTION_MODE}"

