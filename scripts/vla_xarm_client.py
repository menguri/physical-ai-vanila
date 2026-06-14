"""Run a real xArm6 client against a remote LeRobot async policy server.

This script keeps the robot-side stack inside physical-ai-vanila:
RealSense cameras + xArm TCP state are streamed to a LeRobot policy server,
and returned 7D actions are executed with the same Cartesian servo wrapper
used by joy_stick/joy_telecontrol_serial.py.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import pickle  # nosec B403: internal LeRobot async-inference protocol
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

import grpc
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = PROJECT_ROOT.parent
LEROBOT_SRC = MONOREPO_ROOT / "lerobot-demo" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from joy_stick.joy_telecontrol_serial import (  # noqa: E402
    XArm6ServoCartesian,
    clamp_target_pose,
    configure_camera_profile,
    enter_pressed,
    make_recording_cameras,
    return_to_initial_pose,
)
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402


def install_lerobot_async_pickle_shims() -> None:
    """Provide Python 3.11-compatible classes for LeRobot's pickle protocol.

    The policy server imports the real classes from lerobot.async_inference.helpers.
    Pickle only needs matching module/name/attributes on the client side.
    """

    module_name = "lerobot.async_inference.helpers"
    shim = types.ModuleType(module_name)

    @dataclass
    class RemotePolicyConfig:
        policy_type: str
        pretrained_name_or_path: str
        lerobot_features: dict
        actions_per_chunk: int
        device: str = "cpu"
        rename_map: dict[str, str] = field(default_factory=dict)

    @dataclass
    class TimedData:
        timestamp: float
        timestep: int

        def get_timestamp(self):
            return self.timestamp

        def get_timestep(self):
            return self.timestep

    @dataclass
    class TimedAction(TimedData):
        action: object

        def get_action(self):
            return self.action

    @dataclass
    class TimedObservation(TimedData):
        observation: dict
        must_go: bool = False

        def get_observation(self):
            return self.observation

    for cls in (RemotePolicyConfig, TimedData, TimedAction, TimedObservation):
        cls.__module__ = module_name
        cls.__qualname__ = cls.__name__
        setattr(shim, cls.__name__, cls)

    sys.modules[module_name] = shim


install_lerobot_async_pickle_shims()
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation  # noqa: E402


CHUNK_SIZE = 2 * 1024 * 1024
MAX_MESSAGE_SIZE = 4 * 1024 * 1024


def grpc_channel_options(initial_backoff: str = "0.1000s"):
    service_config = {
        "methodConfig": [
            {
                "name": [{}],
                "retryPolicy": {
                    "maxAttempts": 5,
                    "initialBackoff": initial_backoff,
                    "maxBackoff": "2s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"],
                },
            }
        ]
    }
    return [
        ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ("grpc.enable_retries", 1),
        ("grpc.service_config", json.dumps(service_config)),
    ]


def send_bytes_in_chunks(buffer: bytes, message_class, log_prefix: str = "", silent: bool = True):
    del log_prefix, silent
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0, io.SEEK_END)
    size_in_bytes = bytes_buffer.tell()
    bytes_buffer.seek(0)
    sent_bytes = 0
    transfer_state_enum = services_pb2.TransferState

    while sent_bytes < size_in_bytes:
        transfer_state = transfer_state_enum.TRANSFER_MIDDLE
        if sent_bytes + CHUNK_SIZE >= size_in_bytes:
            transfer_state = transfer_state_enum.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = transfer_state_enum.TRANSFER_BEGIN

        size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
        chunk = bytes_buffer.read(size_to_read)
        yield message_class(transfer_state=transfer_state, data=chunk)
        sent_bytes += size_to_read


STATE_NAMES = [
    "tcp_x_mm",
    "tcp_y_mm",
    "tcp_z_mm",
    "tcp_roll_rad",
    "tcp_pitch_rad",
    "tcp_yaw_rad",
    "gripper_pos",
]

ACTION_NAMES = [
    "action_tcp_x_mm",
    "action_tcp_y_mm",
    "action_tcp_z_mm",
    "action_tcp_roll_rad",
    "action_tcp_pitch_rad",
    "action_tcp_yaw_rad",
    "action_gripper_pos",
]

LOCAL_IMAGE_KEYS = ("observation.images.front", "observation.images.wrist")


def renamed_image_key(camera_name: str, rename_map: dict[str, str]) -> str:
    source = f"observation.images.{camera_name}"
    target = rename_map.get(source, source)
    return target.removeprefix("observation.images.")


def server_image_feature_key(local_image_key: str, rename_map: dict[str, str]) -> str:
    return rename_map.get(local_image_key, local_image_key)


def build_lerobot_features(args: argparse.Namespace) -> dict[str, dict]:
    rename_map = json.loads(args.rename_map)
    wrist_feature_key = server_image_feature_key("observation.images.wrist", rename_map)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": STATE_NAMES,
        },
        wrist_feature_key: {
            "dtype": "image",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channels"],
        },
    }
    if args.use_front_camera:
        front_feature_key = server_image_feature_key("observation.images.front", rename_map)
        features[front_feature_key] = {
            "dtype": "image",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def local_observation_to_server_observation(local_obs: dict, rename_map: dict[str, str]) -> dict:
    """Convert dataset-style keys to the raw format expected by LeRobot's async server.

    The local robot observation intentionally matches the training dataset:
    observation.images.front / observation.images.wrist / observation.state / task.

    LeRobot async server builds a dataset frame from raw camera names, and currently
    resizes images before the rename processor runs. Therefore the raw camera keys
    sent over the wire use the post-rename names (camera1/camera2) while local
    diagnostics stay in the original dataset schema.
    """

    state = np.asarray(local_obs["observation.state"], dtype=np.float32)
    server_obs = {
        **{name: float(state[i]) for i, name in enumerate(STATE_NAMES)},
        "task": local_obs["task"],
    }
    for local_key in LOCAL_IMAGE_KEYS:
        if local_key not in local_obs:
            continue
        target_key = server_image_feature_key(local_key, rename_map)
        server_obs[target_key.removeprefix("observation.images.")] = local_obs[local_key]
    return server_obs


class XArmRealRobotAdapter:
    """Real robot adapter with dataset-aligned observation/action contracts."""

    observation_features = {
        "observation.images.front": "rgb",
        "observation.images.wrist": "rgb",
        "observation.state": STATE_NAMES,
        "task": "str",
    }
    action_features = {"action": ACTION_NAMES}

    def __init__(
        self,
        arm: XArm6ServoCartesian,
        cameras: dict,
        args: argparse.Namespace,
    ):
        self.arm = arm
        self.cameras = cameras
        self.args = args

    def get_observation(self) -> dict:
        state = self.arm.read_state().astype(np.float32)
        obs = {
            "observation.images.wrist": self.cameras["wrist"].get_latest_rgb(),
            "observation.state": state,
            "task": self.args.task,
        }
        if self.args.use_front_camera:
            obs["observation.images.front"] = self.cameras["front"].get_latest_rgb()
        return obs

    def send_action(self, action, dry_run: bool = False) -> np.ndarray:
        """Execute a policy 7D action.

        Delta mode expects:
        [delta_tcp_x_mm, delta_tcp_y_mm, delta_tcp_z_mm,
         delta_tcp_roll_rad, delta_tcp_pitch_rad, delta_tcp_yaw_rad,
         delta_gripper_pos]

        Absolute mode expects:
        [target_tcp_x_mm, target_tcp_y_mm, target_tcp_z_mm,
         target_tcp_roll_rad, target_tcp_pitch_rad, target_tcp_yaw_rad,
         target_gripper_pos]
        """

        if isinstance(action, dict):
            if "action" in action:
                raw_action = tensor_to_numpy_action(action["action"])
            else:
                raw_action = np.array([float(action[name]) for name in ACTION_NAMES], dtype=np.float32)
        else:
            raw_action = tensor_to_numpy_action(action)

        current_pose7 = self.arm.read_state()
        raw_target = action_to_target_pose(current_pose7, raw_action, self.args)
        target_pose7 = limit_action_step(current_pose7, raw_target, self.args)
        if not dry_run:
            ret = self.arm.send_target_pose(target_pose7, speed=self.args.servo_speed, acc=self.args.servo_acc)
            self.arm.command_gripper(float(target_pose7[6]))
            if ret != 0:
                raise RuntimeError(f"set_servo_cartesian failed: ret={ret}, target={target_pose7.tolist()}")
        return target_pose7


def summarize_array(value) -> str:
    if isinstance(value, np.ndarray):
        return f"shape={tuple(value.shape)} dtype={value.dtype} min={float(np.min(value)):.3f} max={float(np.max(value)):.3f}"
    return f"type={type(value).__name__} value={value!r}"


def save_diagnostic_images(local_obs: dict, out_dir: Path, cameras: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2 = next(iter(cameras.values())).cv2
    saved = {}
    for key in LOCAL_IMAGE_KEYS:
        if key not in local_obs:
            continue
        short = key.removeprefix("observation.images.")
        path = out_dir / f"{short}.png"
        rgb = local_obs[key]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"Failed to write diagnostic image: {path}")
        saved[key] = path
    return saved


def load_demo_action_stats(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats.get("action")


def project_relative_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def vector_range_report(vector: np.ndarray, stats: dict | None) -> str:
    if not stats:
        return "demo_range=unavailable"
    mins = np.asarray(stats["min"], dtype=np.float32)
    maxs = np.asarray(stats["max"], dtype=np.float32)
    q01 = np.asarray(stats.get("q01", mins), dtype=np.float32)
    q99 = np.asarray(stats.get("q99", maxs), dtype=np.float32)
    inside_minmax = bool(np.all(vector >= mins) and np.all(vector <= maxs))
    inside_q01q99 = bool(np.all(vector >= q01) and np.all(vector <= q99))
    return (
        f"inside_minmax={inside_minmax} inside_q01q99={inside_q01q99} "
        f"demo_min={np.round(mins, 3).tolist()} demo_max={np.round(maxs, 3).tolist()}"
    )


def print_observation_diagnostics(local_obs: dict, server_obs: dict, features: dict, saved_images: dict[str, Path]) -> None:
    print("[diagnose] local robot.get_observation() keys")
    for key in ("observation.images.front", "observation.images.wrist", "observation.state", "task"):
        if key not in local_obs:
            print(f"  {key}: MISSING")
            continue
        print(f"  {key}: {summarize_array(local_obs[key])}")

    print("[diagnose] saved live camera samples")
    for key, path in saved_images.items():
        print(f"  {key}: {path}")

    state = np.asarray(local_obs["observation.state"], dtype=np.float32)
    print("[diagnose] observation.state order and units")
    for name, value in zip(STATE_NAMES, state, strict=True):
        unit = "mm" if name.startswith("tcp_") and name.endswith("_mm") else "rad" if name.endswith("_rad") else "gripper_units"
        print(f"  {name}: {float(value):.6f} {unit}")

    print("[diagnose] server wire observation keys")
    for key, value in server_obs.items():
        print(f"  {key}: {summarize_array(value)}")

    print("[diagnose] RemotePolicyConfig.lerobot_features sent to server")
    for key, spec in features.items():
        if key.startswith("__"):
            continue
        print(f"  {key}: dtype={spec['dtype']} shape={tuple(spec['shape'])} names={spec['names']}")

    print("[diagnose] send_action contract")
    if features.get("__action_mode") == "absolute":
        print("  conclusion: absolute target TCP 7D pose, not joint action")
    else:
        print("  conclusion: delta TCP 7D action added to current observation.state, not joint action")
    print(f"  order: {ACTION_NAMES}")
    print("  units: x/y/z mm, roll/pitch/yaw rad, gripper raw position units")


def run_diagnostics(
    args: argparse.Namespace,
    robot: XArmRealRobotAdapter,
    client: XArmVLAClient,
    receiver: threading.Thread,
    rename_map: dict[str, str],
) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    diagnostic_root = project_relative_path(args.diagnostic_dir) or PROJECT_ROOT / "outputs" / "vla_diagnostics"
    out_dir = diagnostic_root / stamp
    local_obs = robot.get_observation()
    server_obs = local_observation_to_server_observation(local_obs, rename_map)
    saved_images = save_diagnostic_images(local_obs, out_dir, robot.cameras)
    features = build_lerobot_features(args)
    features["__action_mode"] = args.action_mode
    print_observation_diagnostics(local_obs, server_obs, features, saved_images)

    if args.action_log_steps <= 0:
        return

    action_stats = load_demo_action_stats(project_relative_path(args.demo_stats))
    receiver.start()
    client.send_observation(server_obs, timestep=0, must_go=True)
    print(f"[diagnose] waiting for {args.action_log_steps} remote action samples")

    collected = 0
    deadline = time.time() + args.action_timeout_s
    current_pose7 = np.asarray(local_obs["observation.state"], dtype=np.float32)
    while collected < args.action_log_steps and time.time() < deadline:
        timed_action = client.pop_action()
        if timed_action is None:
            time.sleep(0.02)
            continue
        raw_action = tensor_to_numpy_action(timed_action.get_action())
        raw_target = action_to_target_pose(current_pose7, raw_action, args)
        limited = limit_action_step(current_pose7, raw_target, args)
        target_delta = raw_target - current_pose7
        target_delta[3:6] = ((target_delta[3:6] + math.pi) % (2.0 * math.pi)) - math.pi
        print(
            "[diagnose] action "
            f"step={timed_action.get_timestep()} "
            f"mode={args.action_mode} "
            f"raw_action={np.round(raw_action, 4).tolist()} "
            f"raw_target={np.round(raw_target, 4).tolist()} "
            f"target_delta_from_current={np.round(target_delta, 4).tolist()} "
            f"limited_no_motion_target={np.round(limited, 4).tolist()} "
            f"{vector_range_report(raw_action, action_stats)}"
        )
        collected += 1

    if collected < args.action_log_steps:
        print(f"[diagnose] WARNING: only received {collected}/{args.action_log_steps} actions before timeout")


def tensor_to_numpy_action(action) -> np.ndarray:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 7:
        raise ValueError(f"Expected action_dim >= 7, got shape={arr.shape}")
    return arr[:7].copy()


def action_to_target_pose(current: np.ndarray, action: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32).copy()
    if args.action_mode == "absolute":
        return action

    target = current.copy()
    target[:3] += action[:3]
    target[3:6] += action[3:6]
    target[3:6] = ((target[3:6] + math.pi) % (2.0 * math.pi)) - math.pi
    target[6] += action[6]
    return target


def limit_action_step(current: np.ndarray, target: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    limited = target.copy()
    pos_delta = np.clip(
        limited[:3] - current[:3],
        -float(args.max_pos_step_mm),
        float(args.max_pos_step_mm),
    )
    rot_delta = limited[3:6] - current[3:6]
    rot_delta = ((rot_delta + math.pi) % (2.0 * math.pi)) - math.pi
    rot_delta = np.clip(
        rot_delta,
        -float(args.max_rot_step_rad),
        float(args.max_rot_step_rad),
    )
    grip_delta = np.clip(
        limited[6] - current[6],
        -float(args.max_gripper_step),
        float(args.max_gripper_step),
    )
    limited[:3] = current[:3] + pos_delta
    limited[3:6] = current[3:6] + rot_delta
    limited[6] = current[6] + grip_delta
    return clamp_target_pose(limited, args)


def infer_policy_type_from_path(pretrained_name_or_path: str) -> str | None:
    """Best-effort policy inference for local commands that pass long checkpoint paths."""

    lowered = pretrained_name_or_path.lower()
    policy_markers = (
        ("smolvla", "smolvla"),
        ("pi05", "pi05"),
        ("pi0.5", "pi05"),
        ("pi0_fast", "pi0_fast"),
        ("pi0fast", "pi0_fast"),
        ("pi0", "pi0"),
    )
    for marker, policy_type in policy_markers:
        if marker in lowered:
            return policy_type
    return None


class XArmVLAClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.channel = grpc.insecure_channel(
            args.server_address,
            grpc_channel_options(initial_backoff=f"{1.0 / args.fps:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.actions: Queue = Queue()
        self.shutdown = threading.Event()
        self.latest_action_timestep = -1
        self.action_chunk_size = max(1, args.actions_per_chunk)
        self.must_go = True

    def start(self) -> None:
        try:
            self.stub.Ready(services_pb2.Empty())
            policy_config = RemotePolicyConfig(
                policy_type=self.args.policy_type,
                pretrained_name_or_path=self.args.pretrained_name_or_path,
                lerobot_features=build_lerobot_features(self.args),
                actions_per_chunk=self.args.actions_per_chunk,
                device=self.args.policy_device,
                rename_map=json.loads(self.args.rename_map),
            )
            self.stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=pickle.dumps(policy_config))
            )
        except grpc.RpcError as exc:
            details = exc.details() if hasattr(exc, "details") else str(exc)
            code = exc.code() if hasattr(exc, "code") else None
            if code == grpc.StatusCode.UNAVAILABLE:
                raise SystemExit(
                    "[vla-client] failed to connect to remote policy server.\n"
                    f"  server_address: {self.args.server_address}\n"
                    f"  server error: {details}\n"
                    "  This is a connection/tunnel problem, not a checkpoint-load problem.\n"
                    "  Check that the SSH tunnel is still running on the robot PC:\n"
                    "    ssh -N -o ExitOnForwardFailure=yes -L 8080:127.0.0.1:8080 10server\n"
                    "  Then check that the LeRobot policy server is running on 10server and listening on 127.0.0.1:8080.\n"
                    "  If 8080 is occupied by another service, use a clean local port, for example:\n"
                    "    ssh -N -o ExitOnForwardFailure=yes -L 18080:127.0.0.1:8080 10server\n"
                    "    python scripts/vla_xarm_client.py --server-address 127.0.0.1:18080 ..."
                ) from exc
            raise SystemExit(
                "[vla-client] failed to configure remote policy server.\n"
                f"  server_address: {self.args.server_address}\n"
                f"  pretrained_name_or_path sent to server: {self.args.pretrained_name_or_path}\n"
                f"  server error: {details}\n"
                "  Check that this checkpoint path exists on the SERVER filesystem, not only on the robot PC.\n"
                "  If the policy server preloads POLICY_PATH itself, pass the exact path/name that server expects."
            ) from exc
        print(f"[vla-client] connected to {self.args.server_address}")

    def stop(self) -> None:
        self.shutdown.set()
        self.channel.close()

    def send_observation(self, observation: dict, timestep: int, must_go: bool) -> None:
        timed_obs = TimedObservation(
            timestamp=time.time(),
            timestep=max(0, timestep),
            observation=observation,
            must_go=must_go,
        )
        payload = pickle.dumps(timed_obs)
        iterator = send_bytes_in_chunks(
            payload,
            services_pb2.Observation,
            log_prefix="[vla-client] observation",
            silent=True,
        )
        self.stub.SendObservations(iterator)

    def receive_actions(self) -> None:
        while not self.shutdown.is_set():
            try:
                chunk = self.stub.GetActions(services_pb2.Empty())
                if not chunk.data:
                    continue
                timed_actions = pickle.loads(chunk.data)  # nosec B301
                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                for timed_action in timed_actions:
                    if timed_action.get_timestep() <= self.latest_action_timestep:
                        continue
                    self.actions.put(timed_action)
            except grpc.RpcError as exc:
                if not self.shutdown.is_set():
                    print(f"[vla-client] action receive error: {exc}")
                time.sleep(0.1)

    def should_send_observation(self) -> bool:
        queue_ratio = self.actions.qsize() / max(1, self.action_chunk_size)
        return self.must_go or queue_ratio <= self.args.chunk_size_threshold

    def pop_action(self):
        try:
            return self.actions.get_nowait()
        except Empty:
            return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-address", default="127.0.0.1:8080")
    ap.add_argument(
        "--policy-type",
        default=None,
        help="LeRobot policy type. If omitted, inferred from checkpoint path when possible.",
    )
    ap.add_argument("--pretrained-name-or-path", required=True)
    ap.add_argument("--policy-device", default="cuda")
    ap.add_argument(
        "--action-mode",
        choices=["delta", "absolute"],
        default="delta",
        help="Interpret policy output as delta TCP 7D or absolute target TCP 7D.",
    )
    ap.add_argument("--actions-per-chunk", type=int, default=20)
    ap.add_argument("--chunk-size-threshold", type=float, default=0.5)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--task", required=True)
    ap.add_argument(
        "--rename-map",
        default='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}',
    )

    ap.add_argument("--ip", default="192.168.1.199", help="xArm controller IP")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument("--wrist-camera-serial", required=True)
    ap.add_argument("--front-camera-serial", default=None)
    ap.add_argument("--use-front-camera", action="store_true")

    ap.add_argument("--servo-speed", type=float, default=35.0)
    ap.add_argument("--servo-acc", type=float, default=300.0)
    ap.add_argument("--return-home-seconds", type=float, default=5.0)
    ap.add_argument("--return-control-hz", dest="control_hz", type=float, default=100.0)
    ap.add_argument("--max-pos-step-mm", type=float, default=4.0)
    ap.add_argument("--max-rot-step-rad", type=float, default=0.025)
    ap.add_argument("--max-gripper-step", type=float, default=50.0)
    ap.add_argument("--no-motion", action="store_true", help="stream observations and print actions without moving")
    ap.add_argument("--diagnose", action="store_true", help="print observation/action diagnostics and exit")
    ap.add_argument("--diagnostic-dir", default="outputs/vla_diagnostics")
    ap.add_argument("--action-log-steps", type=int, default=10)
    ap.add_argument("--action-timeout-s", type=float, default=15.0)
    ap.add_argument("--demo-stats", default="data/TASK1/meta/stats.json")

    ap.add_argument("--enable-gripper", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gripper-min", type=float, default=0.0)
    ap.add_argument("--gripper-max", type=float, default=850.0)
    ap.add_argument("--gripper-speed", type=int, default=5000)
    ap.add_argument("--collision-sensitivity", type=int, default=1)
    ap.add_argument("--safe-x-mm", nargs=2, type=float, default=[0.0, 570.0])
    ap.add_argument("--safe-y-mm", nargs=2, type=float, default=[-540.0, 550.0])
    ap.add_argument("--safe-z-mm", nargs=2, type=float, default=[180.0, 600.0])

    args = ap.parse_args()
    if args.front_camera_serial:
        args.use_front_camera = True
    if args.use_front_camera and not args.front_camera_serial:
        ap.error("--front-camera-serial is required when front camera is enabled")
    if args.front_camera_serial == args.wrist_camera_serial:
        ap.error("wrist and front camera serials must be different")
    try:
        json.loads(args.rename_map)
    except json.JSONDecodeError as exc:
        ap.error(f"--rename-map must be valid JSON: {exc}")
    inferred_policy_type = infer_policy_type_from_path(args.pretrained_name_or_path)
    if args.policy_type is None:
        args.policy_type = inferred_policy_type or "smolvla"
        if inferred_policy_type:
            print(f"[vla-client] inferred --policy-type {args.policy_type} from checkpoint path")
    elif inferred_policy_type and args.policy_type != inferred_policy_type:
        print(
            "[vla-client] warning: checkpoint path looks like "
            f"{inferred_policy_type!r}, but --policy-type is {args.policy_type!r}"
        )
    args.record = True
    args.camera_format = "bgr8"
    return args


def main() -> None:
    args = parse_args()
    rename_map = json.loads(args.rename_map)
    client = XArmVLAClient(args)
    receiver = threading.Thread(target=client.receive_actions, daemon=True)

    client.start()

    configure_camera_profile(args)
    cameras = make_recording_cameras(args)
    arm = XArm6ServoCartesian(
        args.ip,
        collision_sensitivity=args.collision_sensitivity,
        enable_gripper=args.enable_gripper,
        gripper_speed=args.gripper_speed,
    )
    robot = XArmRealRobotAdapter(arm, cameras, args)
    dt = 1.0 / args.fps
    initial_pose7 = arm.read_state()
    target_pose7 = initial_pose7.copy()
    return_home_on_exit = False

    try:
        if args.diagnose:
            run_diagnostics(args, robot, client, receiver, rename_map)
            return

        receiver.start()
        print("[vla-client] running; Enter returns to initial state and exits, Ctrl+C stops")
        while True:
            t0 = time.time()

            if enter_pressed():
                print("[vla-client] Enter pressed; return to initial state and exit.")
                return_home_on_exit = True
                break

            if client.should_send_observation():
                local_obs = robot.get_observation()
                obs = local_observation_to_server_observation(local_obs, rename_map)
                client.send_observation(
                    obs,
                    timestep=max(client.latest_action_timestep, 0),
                    must_go=client.must_go,
                )
                client.must_go = False

            timed_action = client.pop_action()
            if timed_action is not None:
                raw_target = tensor_to_numpy_action(timed_action.get_action())
                target_pose7 = robot.send_action(raw_target, dry_run=args.no_motion)
                client.latest_action_timestep = timed_action.get_timestep()

                if args.no_motion:
                    print(
                        "[vla-client] action "
                        f"step={client.latest_action_timestep} target={np.round(target_pose7, 3).tolist()}"
                    )

            if client.actions.empty():
                client.must_go = True

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\n[vla-client] stopping")
    finally:
        if return_home_on_exit and not args.no_motion:
            target_pose7 = return_to_initial_pose(arm, target_pose7, initial_pose7, args)
        client.stop()
        arm.stop()
        for camera in cameras.values():
            camera.stop()
        if receiver.is_alive():
            receiver.join(timeout=1.0)


if __name__ == "__main__":
    main()
