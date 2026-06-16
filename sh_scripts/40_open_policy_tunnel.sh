#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "Opening policy tunnel: robot PC 127.0.0.1:${LOCAL_POLICY_PORT} -> ${REMOTE} 127.0.0.1:${REMOTE_POLICY_PORT}"

ssh -N -o ExitOnForwardFailure=yes \
  -L "${LOCAL_POLICY_PORT}:127.0.0.1:${REMOTE_POLICY_PORT}" \
  "${REMOTE}"
