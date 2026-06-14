#!/usr/bin/env bash
set -euo pipefail

# Release RealSense devices before starting collection/eval.
# This only targets processes that currently hold /dev/video*.

if ! compgen -G "/dev/video*" >/dev/null; then
  echo "[release-cameras] no /dev/video* devices found"
  exit 0
fi

if fuser -v /dev/video* >/tmp/xarm6_camera_users.txt 2>&1; then
  echo "[release-cameras] camera devices are busy; stopping current owners"
  cat /tmp/xarm6_camera_users.txt
  fuser -k /dev/video* || true
  sleep 1
else
  echo "[release-cameras] camera devices are free"
fi

if fuser -v /dev/video* >/tmp/xarm6_camera_users_after.txt 2>&1; then
  echo "[release-cameras] WARNING: camera devices are still busy"
  cat /tmp/xarm6_camera_users_after.txt
  exit 1
fi

echo "[release-cameras] camera devices are ready"

