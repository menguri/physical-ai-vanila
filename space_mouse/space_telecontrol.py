"""Teleoperate xArm6 with a SpaceMouse and optionally record LeRobot v3 data.

Examples:
    python space_mouse/space_telecontrol.py --ip 192.168.1.199 --fps 10

    python space_mouse/space_telecontrol.py --ip 192.168.1.199 --fps 10 --record \
        --repo-id kangkang9412/xarm6_spacemouse_demo \
        --root ./data/xarm6_spacemouse_demo \
        --task "pick up the object"
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = PROJECT_ROOT.parent
LEROBOT_SRC = MONOREPO_ROOT / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))


def import_required(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(f"{module_name} is not installed. {install_hint}") from exc


def make_lerobot_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
            "names": [
                "target_tcp_x_mm",
                "target_tcp_y_mm",
                "target_tcp_z_mm",
                "target_tcp_roll_rad",
                "target_tcp_pitch_rad",
                "target_tcp_yaw_rad",
                "target_gripper_pos",
            ],
        },
    }

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        robot_type="xarm6",
        features=features,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        streaming_encoding=args.streaming_encoding,
    )


class WristRealSense:
    def __init__(self, width: int, height: int, fps: int, serial: str | None = None):
        self.rs = import_required("pyrealsense2", "Install Intel RealSense SDK and pyrealsense2.")
        self.cv2 = import_required("cv2", "pip install opencv-python")
        self.width = width
        self.height = height
        self.pipeline = self.rs.pipeline()
        self.config = self.rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(self.rs.stream.color, width, height, self.rs.format.bgr8, fps)

    def start(self) -> None:
        self.pipeline.start(self.config)
        for _ in range(10):
            self.pipeline.wait_for_frames()

    def get_latest_rgb(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSense color frame is not available.")
        color_bgr = np.asanyarray(color_frame.get_data())
        return self.cv2.cvtColor(color_bgr, self.cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        self.pipeline.stop()


class XArm6Cartesian:
    def __init__(self, ip: str, collision_sensitivity: int):
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise SystemExit("xArm-Python-SDK is not installed. pip install xArm-Python-SDK") from exc

        self.arm = XArmAPI(ip, is_radian=True)
        self.arm.motion_enable(enable=True)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.set_collision_sensitivity(collision_sensitivity)
        self.arm.set_mode(5)
        self.arm.set_state(state=0)
        time.sleep(0.2)

    def send_velocity(self, vel6: np.ndarray) -> None:
        command = [float(v) for v in vel6[:6]]
        if hasattr(self.arm, "vc_set_cartesian_velocity"):
            self.arm.vc_set_cartesian_velocity(command, is_radian=True)
        elif hasattr(self.arm, "set_position_aa"):
            raise RuntimeError("xArm SDK does not expose vc_set_cartesian_velocity on this install.")
        else:
            raise RuntimeError("Unsupported xArm SDK: missing Cartesian velocity API.")

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

    def command_gripper(self, target: float) -> None:
        if hasattr(self.arm, "set_gripper_position"):
            self.arm.set_gripper_position(float(target), wait=False)

    def stop(self) -> None:
        try:
            self.send_velocity(np.zeros(6, dtype=np.float32))
        except Exception:
            pass
        try:
            self.arm.set_state(state=4)
        except Exception:
            pass
        self.arm.disconnect()


def keyboard_key_pressed() -> str | None:
    if sys.platform.startswith("win"):
        import msvcrt

        if msvcrt.kbhit():
            key = msvcrt.getwch()
            return key.lower()
    return None


def clamp_target_pose(pose7: np.ndarray, args) -> np.ndarray:
    pose7 = pose7.copy()
    pose7[0] = np.clip(pose7[0], args.safe_x_mm[0], args.safe_x_mm[1])
    pose7[1] = np.clip(pose7[1], args.safe_y_mm[0], args.safe_y_mm[1])
    pose7[2] = np.clip(pose7[2], args.safe_z_mm[0], args.safe_z_mm[1])
    pose7[3:6] = ((pose7[3:6] + math.pi) % (2.0 * math.pi)) - math.pi
    pose7[6] = np.clip(pose7[6], args.gripper_min, args.gripper_max)
    return pose7


def spacemouse_axis(sample, name: str) -> float:
    return float(getattr(sample, name, 0.0) or 0.0)


def spacemouse_button(sample, index: int) -> bool:
    buttons = getattr(sample, "buttons", None)
    if buttons is None:
        return bool(getattr(sample, f"button{index}", False))
    return len(buttons) > index and bool(buttons[index])


def spacemouse_to_velocity(sample, args) -> np.ndarray:
    if sample is None:
        return np.zeros(6, dtype=np.float32)

    deadman_pressed = True
    if args.deadman_button >= 0:
        deadman_pressed = spacemouse_button(sample, args.deadman_button)
    if not deadman_pressed:
        return np.zeros(6, dtype=np.float32)

    raw = np.array(
        [
            spacemouse_axis(sample, "x"),
            spacemouse_axis(sample, "y"),
            spacemouse_axis(sample, "z"),
            spacemouse_axis(sample, "roll"),
            spacemouse_axis(sample, "pitch"),
            spacemouse_axis(sample, "yaw"),
        ],
        dtype=np.float32,
    )
    raw[np.abs(raw) < args.deadzone] = 0.0
    raw[:3] *= args.linear_scale
    raw[3:] *= args.angular_scale
    return raw


def update_target_pose(current_pose7: np.ndarray, prev_target_pose7: np.ndarray | None, vel6: np.ndarray, dt: float, args):
    target = current_pose7.copy() if prev_target_pose7 is None else prev_target_pose7.copy()
    target[:6] += vel6[:6] * dt
    return clamp_target_pose(target, args)


def maybe_update_gripper(sample, target: float, args) -> float:
    if sample is None:
        return target
    if args.open_button >= 0 and spacemouse_button(sample, args.open_button):
        target += args.gripper_step
    if args.close_button >= 0 and spacemouse_button(sample, args.close_button):
        target -= args.gripper_step
    return float(np.clip(target, args.gripper_min, args.gripper_max))


def teleop_loop(args, robot: XArm6Cartesian, camera: WristRealSense | None, dataset) -> None:
    pyspacemouse = import_required("pyspacemouse", "pip install pyspacemouse")
    if not pyspacemouse.open():
        raise RuntimeError("Could not open SpaceMouse. Check USB connection and driver state.")

    control_dt = 1.0 / args.control_hz
    record_dt = 1.0 / args.fps
    next_record_t = time.time()
    target_pose7 = robot.read_state()
    gripper_target = float(target_pose7[6])
    last_gripper_command_t = 0.0

    print("[teleop] running. Press q or Ctrl+C to save/exit.")
    try:
        while True:
            t0 = time.time()
            sample = pyspacemouse.read()

            key = keyboard_key_pressed()
            if key == "q":
                break
            if key == "\x1b":
                if dataset is not None:
                    dataset.clear_episode_buffer()
                    print("[teleop] episode buffer cleared.")
                break

            vel6 = spacemouse_to_velocity(sample, args)
            robot.send_velocity(vel6)

            current_pose7 = robot.read_state()
            gripper_target = maybe_update_gripper(sample, gripper_target, args)
            target_pose7 = update_target_pose(current_pose7, target_pose7, vel6, control_dt, args)
            target_pose7[6] = gripper_target

            now = time.time()
            if now - last_gripper_command_t >= args.gripper_command_dt:
                robot.command_gripper(gripper_target)
                last_gripper_command_t = now

            if dataset is not None and now >= next_record_t:
                wrist_rgb = camera.get_latest_rgb()
                state7 = robot.read_state()
                action7 = target_pose7.copy()
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
    finally:
        pyspacemouse.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.199", help="xArm controller IP")
    ap.add_argument("--fps", type=int, default=10, help="LeRobot recording FPS")
    ap.add_argument("--control-hz", type=float, default=30.0, help="xArm velocity control loop rate")
    ap.add_argument("--record", action="store_true", help="record one episode in LeRobot v3 format")
    ap.add_argument("--repo-id", default=None, help="Hugging Face dataset repo id")
    ap.add_argument("--root", default=None, help="local LeRobot dataset root")
    ap.add_argument("--task", default="teleoperate xarm6", help="task string saved with each frame")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument("--camera-serial", default=None)
    ap.add_argument("--image-writer-threads", type=int, default=4)
    ap.add_argument("--image-writer-processes", type=int, default=0)
    ap.add_argument("--streaming-encoding", action="store_true")
    ap.add_argument("--linear-scale", type=float, default=80.0, help="mm/s per SpaceMouse unit")
    ap.add_argument("--angular-scale", type=float, default=0.8, help="rad/s per SpaceMouse unit")
    ap.add_argument("--deadzone", type=float, default=0.05)
    ap.add_argument("--deadman-button", type=int, default=0, help="button index required for motion; -1 disables")
    ap.add_argument("--open-button", type=int, default=1, help="button index to open gripper; -1 disables")
    ap.add_argument("--close-button", type=int, default=-1, help="button index to close gripper; -1 disables")
    ap.add_argument("--gripper-min", type=float, default=0.0)
    ap.add_argument("--gripper-max", type=float, default=850.0)
    ap.add_argument("--gripper-step", type=float, default=15.0)
    ap.add_argument("--gripper-command-dt", type=float, default=0.15)
    ap.add_argument("--collision-sensitivity", type=int, default=1)
    ap.add_argument("--safe-x-mm", nargs=2, type=float, default=[0.0, 570.0])
    ap.add_argument("--safe-y-mm", nargs=2, type=float, default=[-540.0, 550.0])
    ap.add_argument("--safe-z-mm", nargs=2, type=float, default=[180.0, 600.0])
    args = ap.parse_args()

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

        robot = XArm6Cartesian(args.ip, args.collision_sensitivity)
        teleop_loop(args, robot, camera, dataset)
    except KeyboardInterrupt:
        print("\n[teleop] Ctrl+C received.")
    finally:
        if robot is not None:
            robot.stop()
        if camera is not None:
            camera.stop()
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.save_episode()
            dataset.finalize()
            print(f"[teleop] LeRobot dataset finalized at {dataset.root}")


if __name__ == "__main__":
    main()
