#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${PROJECT_ROOT}"

python -c "from xarm.wrapper import XArmAPI; a=XArmAPI('${ROBOT_IP}'); print('state=', a.get_state()); print('err_warn=', a.get_err_warn_code()); print('q=', a.get_servo_angle(is_radian=True)); a.disconnect()"
python joy_stick/joy_telecontrol_serial.py --list-cameras

