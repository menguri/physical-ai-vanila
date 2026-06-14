"""Teleoperate xArm6 with a Linux joystick/gamepad and optionally record LeRobot v3 data.

Examples:
    # Teleoperation only
    python joy_stick/joy_telecontrol.py --ip 192.168.1.199 --device /dev/input/js0 --fps 10

    # Teleoperation + LeRobot recording
    python joy_stick/joy_telecontrol.py --ip 192.168.1.199 --device /dev/input/js0 --fps 10 --record \
        --repo-id kangkang9412/xarm6_joystick_demo \
        --root ./data/xarm6_joystick_demo \
        --task "pick up the object"
"""
from __future__ import annotations

import argparse
import errno
import math
import os
import select
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = PROJECT_ROOT.parent
LEROBOT_SRC = MONOREPO_ROOT / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))


# =========================
# Xbox 360 / Linux joydev mapping
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

JS_EVENT_FORMAT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80


def import_required(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(f"{module_name} is not installed. {install_hint}") from exc


def move_incomplete_lerobot_root(root: Path) -> Path | None:
    """Move aside a LeRobot root created by create() before any episode was saved."""
    if not root.exists():
        return None

    info_path = root / "meta" / "info.json"
    episode_meta_dir = root / "meta" / "episodes"
    tasks_path = root / "meta" / "tasks.parquet"
    data_dir = root / "data"
    videos_dir = root / "videos"
    images_dir = root / "images"
    is_empty = not any(root.iterdir())
    has_saved_episode_metadata = episode_meta_dir.exists() or tasks_path.exists()
    has_saved_episode_payload = data_dir.exists() or videos_dir.exists()
    has_only_temporary_images = images_dir.exists() and not has_saved_episode_metadata and not has_saved_episode_payload
    is_unsaved_lerobot_root = info_path.exists() and not has_saved_episode_metadata and not has_saved_episode_payload
    if not is_empty and not is_unsaved_lerobot_root and not has_only_temporary_images:
        return None

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = root.with_name(f"{root.name}.partial-{stamp}")
    suffix = 1
    while backup.exists():
        backup = root.with_name(f"{root.name}.partial-{stamp}-{suffix}")
        suffix += 1

    shutil.move(str(root), str(backup))
    return backup


def sanitize_task_id(task_id: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in task_id.strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError("--task-id must contain at least one letter or number")
    return sanitized


def apply_task_root(args: argparse.Namespace) -> None:
    if not args.task_id:
        return

    task_id = sanitize_task_id(args.task_id)
    base_root = Path(args.root) if args.root is not None else Path("data")
    args.root = str(base_root if base_root.name == task_id else base_root / task_id)
    args.task_id = task_id


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def dz(v: float, deadzone: float) -> float:
    return 0.0 if abs(v) < deadzone else float(v)


def norm_axis(value: int) -> float:
    return clip(value / 32767.0, -1.0, 1.0)


def trigger_value(axis_value: float, trigger_deadzone: float) -> float:
    # Linux joydev exposes Xbox triggers at -1.0 when released and +1.0 when fully pressed.
    value = clip((axis_value + 1.0) * 0.5, 0.0, 1.0)
    return 0.0 if value < trigger_deadzone else value


def joystick_name(path: str) -> str:
    name_path = f"/sys/class/input/{os.path.basename(path)}/device/name"
    try:
        with open(name_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return path


class Gamepad:
    """Minimal non-blocking Linux joydev reader.

    This intentionally mirrors the previously verified demo.py reader, so the same
    /dev/input/js0 device and Xbox-style axis/button mapping can be reused.
    """

    def __init__(self, path: str):
        self.path = path
        self.name = joystick_name(path)
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.axes = [0.0] * 8
        self.buttons = [False] * 11

        # Xbox triggers rest at -1.0. Initializing them avoids a false half-press
        # before init events are received.
        self.axes[AXIS_LT] = -1.0
        self.axes[AXIS_RT] = -1.0

    def close(self) -> None:
        os.close(self.fd)

    def poll(self) -> None:
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

    def axis(self, number: int) -> float:
        if number < 0 or number >= len(self.axes):
            return 0.0
        return float(self.axes[number])

    def button(self, number: int) -> bool:
        if number < 0 or number >= len(self.buttons):
            return False
        return bool(self.buttons[number])

    def settle(self, seconds: float = 0.3) -> None:
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            self.poll()
            time.sleep(0.01)


def make_lerobot_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.action_mode == "delta":
        action_names = [
            "delta_tcp_x_mm",
            "delta_tcp_y_mm",
            "delta_tcp_z_mm",
            "delta_tcp_roll_rad",
            "delta_tcp_pitch_rad",
            "delta_tcp_yaw_rad",
            "delta_gripper_pos",
        ]
    else:
        action_names = [
            "target_tcp_x_mm",
            "target_tcp_y_mm",
            "target_tcp_z_mm",
            "target_tcp_roll_rad",
            "target_tcp_pitch_rad",
            "target_tcp_yaw_rad",
            "target_gripper_pos",
        ]

    features = {
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "tcp_x_mm",
                "tcp_y_mm",
                "tcp_z_mm",
                "tcp_roll_rad",
                "tcp_pitch_rad",
                "tcp_yaw_rad",
                "gripper_pos",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": action_names,
        },
    }

    root = Path(args.root) if args.root is not None else None
    if root is not None and root.exists():
        backup = move_incomplete_lerobot_root(root)
        if backup is not None:
            print(f"[joy-teleop] moved incomplete dataset stub to {backup}")
        else:
            dataset = LeRobotDataset(
                repo_id=args.repo_id,
                root=root,
                batch_encoding_size=1,
                streaming_encoding=args.streaming_encoding,
            )
            if args.image_writer_processes or args.image_writer_threads:
                dataset.start_image_writer(
                    num_processes=args.image_writer_processes,
                    num_threads=args.image_writer_threads,
                )
            print(
                "[joy-teleop] resuming dataset "
                f"{root} ({dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames, "
                f"next EP{dataset.meta.total_episodes + 1})"
            )
            return dataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        robot_type="xarm6",
        features=features,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        streaming_encoding=args.streaming_encoding,
    )
    print(f"[joy-teleop] created dataset {dataset.root} (next EP1)")
    return dataset


class WristRealSense:
    def __init__(self, width: int, height: int, fps: int, serial: str | None = None):
        self.rs = import_required("pyrealsense2", "Install Intel RealSense SDK and pyrealsense2.")
        self.cv2 = import_required("cv2", "pip install opencv-python")
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.pipeline = self.rs.pipeline()
        self.config = self.rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(self.rs.stream.color, width, height, self.rs.format.bgr8, fps)
        self.started = False

    def start(self) -> None:
        serial_msg = self.serial if self.serial else "auto"
        try:
            self.pipeline.start(self.config)
            self.started = True
            for _ in range(10):
                self.pipeline.wait_for_frames()
        except Exception as exc:
            if self.started:
                self.pipeline.stop()
                self.started = False
            raise RuntimeError(
                "Failed to start RealSense camera "
                f"(serial={serial_msg}, stream={self.width}x{self.height}@{self.fps}): {exc}"
            ) from exc

    def get_latest_rgb(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSense color frame is not available.")
        color_bgr = np.asanyarray(color_frame.get_data())
        return self.cv2.cvtColor(color_bgr, self.cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        if not self.started:
            return
        self.pipeline.stop()
        self.started = False


class XArm6ServoCartesian:
    """xArm wrapper for target-pose Cartesian servo teleoperation.

    It uses set_mode(1) + set_servo_cartesian(), matching the joystick demo that
    was already verified on the robot, while storing poses in the same 7D
    LeRobot schema used by space_telecontrol.py.
    """

    def __init__(self, ip: str, collision_sensitivity: int, enable_gripper: bool, gripper_speed: int):
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise SystemExit("xArm-Python-SDK is not installed. pip install xArm-Python-SDK") from exc

        self.arm = XArmAPI(ip, is_radian=True)
        time.sleep(0.5)

        if getattr(self.arm, "warn_code", 0) != 0:
            self.arm.clean_warn()
        if getattr(self.arm, "error_code", 0) != 0:
            self.arm.clean_error()

        self.arm.motion_enable(True)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.set_collision_sensitivity(collision_sensitivity)
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.5)

        if enable_gripper:
            self.setup_gripper(gripper_speed)

    def setup_gripper(self, gripper_speed: int) -> None:
        if hasattr(self.arm, "set_gripper_enable"):
            self.arm.set_gripper_enable(True)
        if hasattr(self.arm, "set_gripper_mode"):
            self.arm.set_gripper_mode(0)
        if hasattr(self.arm, "set_gripper_speed"):
            self.arm.set_gripper_speed(gripper_speed)

    def read_state(self) -> np.ndarray:
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_position failed with code={code}")
        gripper = self.read_gripper()
        return np.array([*pose[:6], gripper], dtype=np.float32)

    def read_gripper(self) -> float:
        if not hasattr(self.arm, "get_gripper_position"):
            return 0.0
        code, pos = self.arm.get_gripper_position()
        return float(pos) if code == 0 else 0.0

    def send_target_pose(self, target_pose7: np.ndarray, speed: float, acc: float) -> int:
        target6 = [float(v) for v in target_pose7[:6]]
        return int(
            self.arm.set_servo_cartesian(
                target6,
                speed=float(speed),
                mvacc=float(acc),
                is_radian=True,
            )
        )

    def command_gripper(self, target: float) -> None:
        if hasattr(self.arm, "set_gripper_position"):
            self.arm.set_gripper_position(float(target), wait=False)

    def emergency_stop(self) -> None:
        try:
            self.arm.set_state(4)
        except Exception:
            pass

    def stop(self) -> None:
        try:
            self.arm.set_state(4)
        except Exception:
            pass
        self.arm.disconnect()


def clamp_target_pose(pose7: np.ndarray, args) -> np.ndarray:
    pose7 = pose7.copy()
    pose7[0] = np.clip(pose7[0], args.safe_x_mm[0], args.safe_x_mm[1])
    pose7[1] = np.clip(pose7[1], args.safe_y_mm[0], args.safe_y_mm[1])
    pose7[2] = np.clip(pose7[2], args.safe_z_mm[0], args.safe_z_mm[1])
    pose7[3:6] = ((pose7[3:6] + math.pi) % (2.0 * math.pi)) - math.pi
    pose7[6] = np.clip(pose7[6], args.gripper_min, args.gripper_max)
    return pose7


def joystick_to_pose_delta(pad: Gamepad, args, dt: float) -> tuple[np.ndarray, dict[str, float]]:
    lx = dz(pad.axis(AXIS_LX), args.deadzone)
    ly = dz(pad.axis(AXIS_LY), args.deadzone)
    lt = trigger_value(pad.axis(AXIS_LT), args.trigger_deadzone)
    rx = dz(pad.axis(AXIS_RX), args.deadzone)
    ry = dz(pad.axis(AXIS_RY), args.deadzone)
    rt = trigger_value(pad.axis(AXIS_RT), args.trigger_deadzone)

    z_input = dz(float(pad.button(args.z_up_button)) - lt, args.deadzone)
    yaw_input = dz(float(pad.button(args.yaw_pos_button)) - rt, args.deadzone)

    delta = np.zeros(7, dtype=np.float32)
    delta[0] = args.x_sign * ly * args.pos_gain * dt
    delta[1] = args.y_sign * lx * args.pos_gain * dt
    delta[2] = args.z_sign * z_input * args.pos_gain * dt
    delta[3] = args.roll_sign * rx * args.rot_gain_rad * dt
    delta[4] = args.pitch_sign * ry * args.rot_gain_rad * dt
    delta[5] = args.yaw_sign * yaw_input * args.rot_gain_rad * dt

    debug = {
        "lx": lx,
        "ly": ly,
        "lt": lt,
        "rx": rx,
        "ry": ry,
        "rt": rt,
        "z_input": z_input,
        "yaw_input": yaw_input,
    }
    return delta, debug


def button_edge(pad: Gamepad, button: int, previous: bool) -> tuple[bool, bool]:
    current = pad.button(button)
    return current and not previous, current


def enter_pressed() -> bool:
    if not sys.stdin.isatty():
        return False
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return False
    sys.stdin.readline()
    return True


def dataset_has_pending_frames(dataset) -> bool:
    episode_buffer = getattr(dataset, "episode_buffer", None)
    if not episode_buffer:
        return False
    return int(episode_buffer.get("size", 0)) > 0


def save_pending_dataset(dataset) -> None:
    if dataset_has_pending_frames(dataset):
        dataset.save_episode()


def return_to_initial_pose(
    robot: XArm6ServoCartesian,
    current_pose7: np.ndarray,
    initial_pose7: np.ndarray,
    args,
) -> np.ndarray:
    print("[joy-teleop] returning to initial state...")
    start = current_pose7.copy()
    goal = clamp_target_pose(initial_pose7, args)
    delta = goal - start
    delta[3:6] = ((delta[3:6] + math.pi) % (2.0 * math.pi)) - math.pi

    steps = max(2, int(args.return_home_seconds * args.control_hz))
    dt = 1.0 / args.control_hz
    for i in range(1, steps + 1):
        alpha = i / steps
        pose = start + alpha * delta
        pose = clamp_target_pose(pose, args)
        ret = robot.send_target_pose(pose, speed=args.servo_speed, acc=args.servo_acc)
        if ret != 0:
            raise RuntimeError(f"return to initial state failed: ret={ret}, target={pose.tolist()}")
        time.sleep(dt)

    robot.command_gripper(float(goal[6]))
    print("[joy-teleop] returned to initial state.")
    return goal


def record_action_from_target(state7: np.ndarray, target_pose7: np.ndarray, action_mode: str) -> np.ndarray:
    if action_mode == "absolute":
        return target_pose7.astype(np.float32).copy()

    action7 = target_pose7.astype(np.float32) - state7.astype(np.float32)
    action7[3:6] = ((action7[3:6] + math.pi) % (2.0 * math.pi)) - math.pi
    return action7


def teleop_loop(args, robot: XArm6ServoCartesian, camera: WristRealSense | None, dataset) -> None:
    pad = Gamepad(args.device)
    pad.settle()

    control_dt = 1.0 / args.control_hz
    record_dt = 1.0 / args.fps
    next_record_t = time.time()

    initial_pose7 = robot.read_state()
    target_pose7 = initial_pose7.copy()
    gripper_target = float(target_pose7[6])
    previous_gripper_button = False
    discard_episode = False
    return_home_on_exit = False

    print("[joy-teleop] running")
    print(f"  device : {pad.name} ({args.device})")
    print("  left stick  : TCP X / Y")
    print("  LB / LT     : TCP Z up / down")
    print("  right stick : TCP roll / pitch")
    print("  RB / RT     : TCP yaw + / -")
    print("  A           : toggle gripper")
    print("  SELECT      : save and exit")
    print("  Enter       : save, return to initial state, and exit")
    print("  START       : emergency stop and discard episode")
    print("  Ctrl+C      : save and exit")

    try:
        while True:
            t0 = time.time()
            pad.poll()

            if pad.button(args.save_button):
                print("[joy-teleop] save button pressed.")
                break

            if enter_pressed():
                print("[joy-teleop] Enter pressed; save, return to initial state, and exit.")
                return_home_on_exit = True
                break

            if pad.button(args.emergency_button):
                print("[joy-teleop] EMERGENCY STOP button pressed.")
                discard_episode = args.discard_on_emergency
                robot.emergency_stop()
                break

            delta7, _ = joystick_to_pose_delta(pad, args, control_dt)
            target_pose7[:6] += delta7[:6]

            gripper_pressed, previous_gripper_button = button_edge(
                pad,
                args.gripper_toggle_button,
                previous_gripper_button,
            )
            if gripper_pressed:
                midpoint = 0.5 * (args.gripper_min + args.gripper_max)
                gripper_target = args.gripper_min if gripper_target > midpoint else args.gripper_max
                robot.command_gripper(gripper_target)
                print(f"[joy-teleop] gripper target: {gripper_target:.1f}")

            target_pose7[6] = gripper_target
            target_pose7 = clamp_target_pose(target_pose7, args)

            ret = robot.send_target_pose(target_pose7, speed=args.servo_speed, acc=args.servo_acc)
            if ret != 0:
                print(f"[joy-teleop] set_servo_cartesian failed: ret={ret}, target={target_pose7.tolist()}")
                discard_episode = args.discard_on_robot_error
                break

            now = time.time()
            if dataset is not None and now >= next_record_t:
                if camera is None:
                    raise RuntimeError("camera is required when dataset is enabled")
                wrist_rgb = camera.get_latest_rgb()
                state7 = robot.read_state()
                action7 = record_action_from_target(state7, target_pose7, args.action_mode)
                dataset.add_frame(
                    {
                        "observation.images.wrist": wrist_rgb,
                        "observation.state": state7.astype(np.float32),
                        "action": action7.astype(np.float32),
                        "task": args.task,
                    }
                )
                next_record_t += record_dt

            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
        if return_home_on_exit and not discard_episode:
            target_pose7 = return_to_initial_pose(robot, target_pose7, initial_pose7, args)
    finally:
        pad.close()
        if discard_episode and dataset is not None:
            dataset.clear_episode_buffer()
            print("[joy-teleop] episode buffer cleared.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.199", help="xArm controller IP")
    ap.add_argument("--device", default="/dev/input/js0", help="Linux joystick device path")
    ap.add_argument("--fps", type=int, default=10, help="LeRobot recording FPS")
    ap.add_argument("--control-hz", type=float, default=100.0, help="xArm servo Cartesian loop rate")
    ap.add_argument("--record", action="store_true", help="record one episode in LeRobot v3 format")
    ap.add_argument(
        "--action-mode",
        choices=["delta", "absolute"],
        default="delta",
        help="Action saved in dataset: delta from observation.state or absolute target TCP 7D.",
    )
    ap.add_argument("--repo-id", default=None, help="Hugging Face dataset repo id")
    ap.add_argument("--root", default=None, help="local LeRobot dataset root")
    ap.add_argument("--task-id", default=None, help="folder name under --root for this task, e.g. TASK1 -> ./data/TASK1")
    ap.add_argument("--task", default="teleoperate xarm6 with joystick", help="task string saved with each frame")

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument("--camera-serial", default=None)
    ap.add_argument("--image-writer-threads", type=int, default=4)
    ap.add_argument("--image-writer-processes", type=int, default=0)
    ap.add_argument("--streaming-encoding", action="store_true")

    ap.add_argument("--deadzone", type=float, default=0.10)
    ap.add_argument("--trigger-deadzone", type=float, default=0.05)
    ap.add_argument("--pos-gain", type=float, default=80.0, help="mm/s at full stick deflection")
    ap.add_argument("--rot-gain-deg", type=float, default=25.0, help="deg/s at full stick deflection")
    ap.add_argument("--servo-speed", type=float, default=100.0)
    ap.add_argument("--servo-acc", type=float, default=1000.0)
    ap.add_argument("--return-home-seconds", type=float, default=3.0, help="seconds used to return to the initial pose after Enter")

    ap.add_argument("--x-sign", type=float, default=-1.0)
    ap.add_argument("--y-sign", type=float, default=-1.0)
    ap.add_argument("--z-sign", type=float, default=1.0)
    ap.add_argument("--roll-sign", type=float, default=1.0)
    ap.add_argument("--pitch-sign", type=float, default=-1.0)
    ap.add_argument("--yaw-sign", type=float, default=1.0)

    ap.add_argument("--z-up-button", type=int, default=BTN_LB)
    ap.add_argument("--yaw-pos-button", type=int, default=BTN_RB)
    ap.add_argument("--gripper-toggle-button", type=int, default=BTN_A)
    ap.add_argument("--save-button", type=int, default=BTN_SELECT)
    ap.add_argument("--emergency-button", type=int, default=BTN_START)
    ap.add_argument("--discard-on-emergency", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--discard-on-robot-error", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--enable-gripper", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gripper-min", type=float, default=0.0)
    ap.add_argument("--gripper-max", type=float, default=850.0)
    ap.add_argument("--gripper-speed", type=int, default=5000)

    ap.add_argument("--collision-sensitivity", type=int, default=1)
    ap.add_argument("--safe-x-mm", nargs=2, type=float, default=[0.0, 570.0])
    ap.add_argument("--safe-y-mm", nargs=2, type=float, default=[-540.0, 550.0])
    ap.add_argument("--safe-z-mm", nargs=2, type=float, default=[180.0, 600.0])

    args = ap.parse_args()
    args.rot_gain_rad = math.radians(args.rot_gain_deg)
    try:
        apply_task_root(args)
    except ValueError as exc:
        ap.error(str(exc))

    if args.record and not args.repo_id:
        ap.error("--repo-id is required when --record is set")
    return args


def main() -> None:
    args = parse_args()
    dataset = None
    camera = None
    robot = None

    try:
        if args.record:
            dataset = make_lerobot_dataset(args)
            camera = WristRealSense(args.width, args.height, args.camera_fps, args.camera_serial)
            camera.start()

        robot = XArm6ServoCartesian(
            ip=args.ip,
            collision_sensitivity=args.collision_sensitivity,
            enable_gripper=args.enable_gripper,
            gripper_speed=args.gripper_speed,
        )
        teleop_loop(args, robot, camera, dataset)
    except KeyboardInterrupt:
        print("\n[joy-teleop] Ctrl+C received.")
    finally:
        if robot is not None:
            robot.stop()
        if camera is not None:
            camera.stop()
        if dataset is not None:
            save_pending_dataset(dataset)
            dataset.finalize()
            print(f"[joy-teleop] LeRobot dataset finalized at {dataset.root}")


if __name__ == "__main__":
    main()
