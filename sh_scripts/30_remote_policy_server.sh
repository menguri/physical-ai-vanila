#!/usr/bin/env bash
set -euo pipefail

# Run this on the robot PC. It SSHes into 10server and starts an empty
# LeRobot async policy server. The local client sends POLICY_TYPE and
# POLICY_PATH during the first handshake.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

ssh -t "${REMOTE}" \
  "cd '${REMOTE_LEROBOT_ROOT}' && ${REMOTE_SETUP} python -m lerobot.async_inference.policy_server \
    --host='${REMOTE_POLICY_HOST}' \
    --port='${REMOTE_POLICY_PORT}' \
    --fps='${FPS}' \
    --obs_queue_timeout=1"
