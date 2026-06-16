#!/usr/bin/env bash
# Experiment-level settings for xArm6 data collection and real VLA eval.
# Researchers should usually edit only this file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Robot PC / devices.
ROBOT_IP="${ROBOT_IP:-192.168.1.199}"
WRIST_CAMERA_SERIAL="${WRIST_CAMERA_SERIAL:-817512070394}"
FRONT_CAMERA_SERIAL="${FRONT_CAMERA_SERIAL:-261222078861}"
USE_FRONT_CAMERA="${USE_FRONT_CAMERA:-1}"

# Data collection.
TASK="${TASK:-pick up the yellow pencil sharpener and place it on the cardboard box}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/xarm6_delta_demo}"
REPO_ID="${REPO_ID:-kangkang9412/xarm6_delta_demo}"
TASK_ID="${TASK_ID:-TASK_DELTA}"
DATA_ACTION_MODE="${DATA_ACTION_MODE:-both}"
ACTION_MODE="${ACTION_MODE:-delta}"
FPS="${FPS:-10}"

# SSH / remote policy server.
REMOTE="${REMOTE:-10server}"
REMOTE_LEROBOT_ROOT="${REMOTE_LEROBOT_ROOT:-/home/mlic/mingukang/lerobot}"
REMOTE_LEROBOT_CHECKOUT="${REMOTE_LEROBOT_CHECKOUT:-spacemouse-lerobot-recording}"
REMOTE_DATA_BASE="${REMOTE_DATA_BASE:-/home/mlic/mingukang/lerobot/collected_demo}"
REMOTE_SETUP="${REMOTE_SETUP:-}"
REMOTE_POLICY_PORT="${REMOTE_POLICY_PORT:-8080}"
LOCAL_POLICY_PORT="${LOCAL_POLICY_PORT:-18080}"

# Inference checkpoint on the SSH server filesystem.
POLICY_TYPE="${POLICY_TYPE:-pi05}"
POLICY_PATH="${POLICY_PATH:-/home/mlic/mingukang/lerobot/outputs/train/pi05_real_full_wandb_20260612_160420/checkpoints/025000/pretrained_model}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
