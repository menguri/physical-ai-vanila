#!/usr/bin/env bash
set -euo pipefail

# Run this on the robot PC. It SSHes into 10server and starts an empty
# LeRobot async policy server. The local client sends POLICY_TYPE and
# POLICY_PATH during the first handshake.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

REMOTE_CHECKOUT_CMD=""
if [[ -n "${REMOTE_LEROBOT_CHECKOUT}" ]]; then
  REMOTE_CHECKOUT_CMD="git checkout '${REMOTE_LEROBOT_CHECKOUT}' &&"
fi

echo "Starting remote policy server:"
echo "  remote: ${REMOTE}"
echo "  remote repo: ${REMOTE_LEROBOT_ROOT}"
echo "  remote checkout: ${REMOTE_LEROBOT_CHECKOUT:-current}"
echo "  tunnel: robot PC 127.0.0.1:${LOCAL_POLICY_PORT} -> ${REMOTE} 127.0.0.1:${REMOTE_POLICY_PORT}"
echo "  fps: ${FPS}"

ssh -t "${REMOTE}" \
  "cd '${REMOTE_LEROBOT_ROOT}' && \
    ${REMOTE_CHECKOUT_CMD} \
    echo '[remote] repo:' \"\$(pwd)\" && \
    echo '[remote] checkout:' \"\$(git rev-parse --abbrev-ref HEAD)\" \"\$(git rev-parse --short HEAD)\" && \
    ${REMOTE_SETUP} python -m lerobot.async_inference.policy_server \
    --host='127.0.0.1' \
    --port='${REMOTE_POLICY_PORT}' \
    --fps='${FPS}' \
    --obs_queue_timeout=1"
