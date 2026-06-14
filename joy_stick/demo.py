import csv
import errno
import os
import struct
import time

from xarm.wrapper import XArmAPI


# =========================
# Robot / control settings
# =========================

ROBOT_IP = "192.168.1.199"
JOYSTICK_DEVICE = "/dev/input/js0"

DT = 0.01
DEADZONE = 0.10
TRIGGER_DEADZONE = 0.05

POS_GAIN = 80.0
ROT_GAIN = 25.0
SERVO_SPEED = 100.0
SERVO_ACC = 1000

Z_MIN = 80
Z_MAX = 500

LOG_FILE = "teleop_gamepad_dataset.csv"


# =========================
# Gripper settings
# =========================

USE_GRIPPER = True
GRIPPER_OPEN_POS = 850
GRIPPER_CLOSED_POS = 0
GRIPPER_SPEED = 5000


# =========================
# Xbox 360 joydev mapping
# =========================

AXIS_LX = 0
AXIS_LY = 1
AXIS_LT = 2
AXIS_RX = 3
AXIS_RY = 4
AXIS_RT = 5
AXIS_DPAD_X = 6
AXIS_DPAD_Y = 7

BTN_A = 0
BTN_B = 1
BTN_X = 2
BTN_Y = 3
BTN_LB = 4
BTN_RB = 5
BTN_SELECT = 6
BTN_START = 7
BTN_MODE = 8
BTN_L_THUMB = 9
BTN_R_THUMB = 10


# Flip these signs if the robot moves opposite to the direction you expect.
X_SIGN = -1.0
Y_SIGN = -1.0
Z_SIGN = 1.0
ROLL_SIGN = 1.0
PITCH_SIGN = -1.0
YAW_SIGN = 1.0


# =========================
# Linux joystick reader
# =========================

JS_EVENT_FORMAT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80


def dz(v, deadzone=DEADZONE):
    if abs(v) < deadzone:
        return 0.0
    return v


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def norm_axis(value):
    return clip(value / 32767.0, -1.0, 1.0)


def trigger_value(axis_value):
    value = clip((axis_value + 1.0) * 0.5, 0.0, 1.0)
    if value < TRIGGER_DEADZONE:
        return 0.0
    return value


def joystick_name(path):
    name_path = f"/sys/class/input/{os.path.basename(path)}/device/name"
    try:
        with open(name_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return path


class Gamepad:
    def __init__(self, path):
        self.path = path
        self.name = joystick_name(path)
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.axes = [0.0] * 8
        self.buttons = [False] * 11

        # Xbox triggers rest at -1.0. Set this before init events arrive so
        # a missing init packet cannot create a false half-pressed trigger.
        self.axes[AXIS_LT] = -1.0
        self.axes[AXIS_RT] = -1.0

    def close(self):
        os.close(self.fd)

    def poll(self):
        while True:
            try:
                data = os.read(self.fd, JS_EVENT_SIZE)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                raise

            if len(data) != JS_EVENT_SIZE:
                break

            _, value, event_type, number = struct.unpack(JS_EVENT_FORMAT, data)
            is_init = bool(event_type & JS_EVENT_INIT)
            event_type &= ~JS_EVENT_INIT

            if event_type == JS_EVENT_AXIS:
                while number >= len(self.axes):
                    self.axes.append(0.0)
                axis_value = norm_axis(value)
                if is_init and number in (AXIS_LT, AXIS_RT) and abs(axis_value) < 0.001:
                    axis_value = -1.0
                self.axes[number] = axis_value
            elif event_type == JS_EVENT_BUTTON:
                while number >= len(self.buttons):
                    self.buttons.append(False)
                self.buttons[number] = bool(value)

    def axis(self, number):
        if number >= len(self.axes):
            return 0.0
        return self.axes[number]

    def button(self, number):
        if number >= len(self.buttons):
            return False
        return self.buttons[number]

    def settle(self, seconds=0.3):
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            self.poll()
            time.sleep(0.01)


def setup_arm():
    arm = XArmAPI(ROBOT_IP, is_radian=False)

    time.sleep(0.5)

    if arm.warn_code != 0:
        arm.clean_warn()

    if arm.error_code != 0:
        arm.clean_error()

    arm.motion_enable(True)
    arm.set_mode(1)
    arm.set_state(0)

    time.sleep(0.5)

    return arm


def read_arm_status(arm):
    status = {}

    for name, call in (
        ("state", arm.get_state),
        ("err_warn", arm.get_err_warn_code),
        ("position", lambda: arm.get_position(is_radian=False)),
        ("angles", lambda: arm.get_servo_angle(is_radian=False)),
    ):
        try:
            status[name] = call()
        except Exception as exc:
            status[name] = repr(exc)

    status["api_error_code"] = arm.error_code
    status["api_warn_code"] = arm.warn_code
    return status


def print_arm_status(arm, prefix):
    print(prefix)
    status = read_arm_status(arm)
    for key, value in status.items():
        print(f"  {key}: {value}")


def setup_gripper(arm):
    if not USE_GRIPPER:
        return

    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(GRIPPER_SPEED)


def set_gripper(arm, is_open):
    if not USE_GRIPPER:
        return

    position = GRIPPER_OPEN_POS if is_open else GRIPPER_CLOSED_POS
    arm.set_gripper_position(position, wait=False)


def main():
    pad = Gamepad(JOYSTICK_DEVICE)
    pad.settle()

    arm = setup_arm()
    csv_file = open(LOG_FILE, "w", newline="")
    writer = csv.writer(csv_file)

    writer.writerow([
        "timestamp",
        "x", "y", "z",
        "roll", "pitch", "yaw",
        "lx", "ly", "lt", "rx", "ry", "rt", "dpad_x", "dpad_y",
        "btn_a", "btn_b", "btn_x", "btn_y",
        "btn_lb", "btn_rb", "btn_select", "btn_start",
        "btn_mode", "btn_l_thumb", "btn_r_thumb",
        "z_input", "yaw_input", "gripper_open",
    ])

    code, pose = arm.get_position(is_radian=False)
    if code != 0:
        raise RuntimeError("Failed to get robot pose")

    target = list(pose[:6])
    gripper_open = True
    previous_a = False

    setup_gripper(arm)

    print("Start Pose")
    print(target)
    print()
    print("==========")
    print(f"Teleoperation Started: {pad.name} ({JOYSTICK_DEVICE})")
    print("Left stick: X / Y")
    print("LB / LT: Z up / down")
    print("Right stick: gripper roll / pitch")
    print("RB / RT: yaw + / -")
    print("A: toggle gripper open / close")
    print("START: emergency stop")
    print("Ctrl+C: exit")
    print("==========")
    print()

    try:
        while True:
            pad.poll()

            if pad.button(BTN_START):
                print("EMERGENCY STOP")
                arm.set_state(4)
                break

            lx = dz(pad.axis(AXIS_LX))
            ly = dz(pad.axis(AXIS_LY))
            lt = trigger_value(pad.axis(AXIS_LT))
            rx = dz(pad.axis(AXIS_RX))
            ry = dz(pad.axis(AXIS_RY))
            rt = trigger_value(pad.axis(AXIS_RT))

            z_input = dz(float(pad.button(BTN_LB)) - lt)
            yaw_input = dz(float(pad.button(BTN_RB)) - rt)

            target[0] += X_SIGN * ly * POS_GAIN * DT
            target[1] += Y_SIGN * lx * POS_GAIN * DT
            target[2] += Z_SIGN * z_input * POS_GAIN * DT
            target[2] = clip(target[2], Z_MIN, Z_MAX)

            target[3] += ROLL_SIGN * rx * ROT_GAIN * DT
            target[4] += PITCH_SIGN * ry * ROT_GAIN * DT
            target[5] += YAW_SIGN * yaw_input * ROT_GAIN * DT

            a_pressed = pad.button(BTN_A)
            if a_pressed and not previous_a:
                gripper_open = not gripper_open
                set_gripper(arm, gripper_open)
                print(f"Gripper {'open' if gripper_open else 'closed'}")
            previous_a = a_pressed

            ret = arm.set_servo_cartesian(
                target,
                speed=SERVO_SPEED,
                mvacc=SERVO_ACC,
                is_radian=False,
            )
            if ret != 0:
                print_arm_status(
                    arm,
                    f"set_servo_cartesian failed: ret={ret}, target={target}",
                )
                break

            writer.writerow([
                time.time(),
                target[0], target[1], target[2],
                target[3], target[4], target[5],
                pad.axis(AXIS_LX),
                pad.axis(AXIS_LY),
                pad.axis(AXIS_LT),
                pad.axis(AXIS_RX),
                pad.axis(AXIS_RY),
                pad.axis(AXIS_RT),
                pad.axis(AXIS_DPAD_X),
                pad.axis(AXIS_DPAD_Y),
                pad.button(BTN_A),
                pad.button(BTN_B),
                pad.button(BTN_X),
                pad.button(BTN_Y),
                pad.button(BTN_LB),
                pad.button(BTN_RB),
                pad.button(BTN_SELECT),
                pad.button(BTN_START),
                pad.button(BTN_MODE),
                pad.button(BTN_L_THUMB),
                pad.button(BTN_R_THUMB),
                z_input,
                yaw_input,
                gripper_open,
            ])

            time.sleep(DT)

    except KeyboardInterrupt:
        print()
        print("Stopped by user")
    finally:
        csv_file.close()
        pad.close()
        arm.set_state(4)
        arm.disconnect()
        print()
        print(f"Saved to {LOG_FILE}")


if __name__ == "__main__":
    main()