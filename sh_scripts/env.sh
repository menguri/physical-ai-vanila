#!/usr/bin/env bash
# Shared local settings for xArm6 data collection and real VLA eval.
# Edit this file first, then run the other scripts from this directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Robot PC / xArm.
ROBOT_IP="${ROBOT_IP:-192.168.1.199}"
WRIST_CAMERA_SERIAL="${WRIST_CAMERA_SERIAL:-817512070394}"
FRONT_CAMERA_SERIAL="${FRONT_CAMERA_SERIAL:-261222078861}"

# Dataset.
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/xarm6_delta_demo}"
REPO_ID="${REPO_ID:-kangkang9412/xarm6_delta_demo}"
TASK_ID="${TASK_ID:-TASK_DELTA}"
TASK="${TASK:-pick up the white bottle and place it in the dark brown box}"
ACTION_MODE="${ACTION_MODE:-delta}"

# Remote server.
REMOTE="${REMOTE:-10server}"
REMOTE_LEROBOT_ROOT="${REMOTE_LEROBOT_ROOT:-/home/mlic/mingukang/lerobot}"
REMOTE_DATA_BASE="${REMOTE_DATA_BASE:-/home/mlic/mingukang/lerobot/collected_demo}"
REMOTE_SETUP="${REMOTE_SETUP:-}"
REMOTE_POLICY_HOST="${REMOTE_POLICY_HOST:-127.0.0.1}"
REMOTE_POLICY_PORT="${REMOTE_POLICY_PORT:-8080}"
LOCAL_POLICY_PORT="${LOCAL_POLICY_PORT:-8080}"

# Policy loaded by the remote LeRobot policy server after the client handshake.
POLICY_TYPE="${POLICY_TYPE:-pi05}"
POLICY_PATH="${POLICY_PATH:-/home/mlic/mingukang/lerobot/outputs/train/pi05_real_full_wandb_20260612_160420/checkpoints/025000/pretrained_model}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

# Runtime safety.
FPS="${FPS:-10}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-20}"
SERVO_SPEED="${SERVO_SPEED:-35}"
SERVO_ACC="${SERVO_ACC:-300}"
MAX_POS_STEP_MM="${MAX_POS_STEP_MM:-4}"
MAX_ROT_STEP_RAD="${MAX_ROT_STEP_RAD:-0.025}"
MAX_GRIPPER_STEP="${MAX_GRIPPER_STEP:-50}"
RETURN_HOME_SECONDS="${RETURN_HOME_SECONDS:-5}"
