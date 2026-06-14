#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

ssh -N -o ExitOnForwardFailure=yes \
  -L "${LOCAL_POLICY_PORT}:127.0.0.1:${REMOTE_POLICY_PORT}" \
  "${REMOTE}"
