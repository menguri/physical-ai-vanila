#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

LOCAL_SRC="${LOCAL_SRC:-${PROJECT_ROOT}/data}"
REMOTE_DST="${REMOTE_DATA_BASE}/$(basename "${LOCAL_SRC}")"

ssh "${REMOTE}" "mkdir -p '${REMOTE_DATA_BASE}'"
rsync -az --dry-run --itemize-changes "${LOCAL_SRC}/" "${REMOTE}:${REMOTE_DST}/"
rsync -az --human-readable --info=progress2 --partial "${LOCAL_SRC}/" "${REMOTE}:${REMOTE_DST}/"
rsync -azc --dry-run --delete --itemize-changes "${LOCAL_SRC}/" "${REMOTE}:${REMOTE_DST}/"

echo "Synced ${LOCAL_SRC} -> ${REMOTE}:${REMOTE_DST}"

