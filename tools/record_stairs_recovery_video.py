#!/usr/bin/env python3
# Copyright (c) 2026-2027 zh
"""
内容：
    在无桌面服务器上使用指定权重录制单个 Go1 从仰卧姿态恢复并以 1 m/s 通过指定地形的 MP4 视频。
    通过 Isaac Gym 离屏相机传感器采集画面，不创建 viewer。

用法：
    python tools/record_stairs_recovery_video.py \
        --terrain stairs_up \
        --input_dir logs/go1_recover_multi_critic/<run>/model_90000.pt
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import MethodType

import numpy as np

from until import prepend_conda_lib_path

prepend_conda_lib_path()

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.math import wrap_to_pi


# ---------- 配置 ----------

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "go1_recover_multi_critic"
TERRAIN_NAMES = ("smooth_slope", "rough_slope", "stairs_up", "stairs_down", "obstacles")
TERRAIN_ROW = 6
TERRAIN_DIFFICULTY = 0.6
VIDEO_DURATION_S = 10.0
VIDEO_FPS = 25
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
CAMERA_HORIZONTAL_FOV = 75.0
POSE_HEIGHT = 0.30
SUPINE_POSE = (math.pi, 0.0, 0.0)
CAMERA_OFFSET = np.array([-1.026, -2.819, 2.4], dtype=np.float64)
CAMERA_TARGET_OFFSET = np.array([0.3, 0.0, 0.25], dtype=np.float64)


def resolve_checkpoint_path(path: Path) -> Path:
    """将权重的绝对路径或仓库相对路径解析为存在的 .pt 文件。"""
    path = path.expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if path.suffix != ".pt":
        raise ValueError(f"Checkpoint must be a .pt file: {path}")
    return path


def reset_evaluation_seed(seed: int = 1) -> None:
    """固定录制过程的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def disable_randomization(env_cfg) -> None:
    """关闭录制时的噪声、域随机化和外部扰动。"""
    env_cfg.noise.add_noise = False
    env_cfg.noise.noise_level = 0.0
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_motor_zero_offset = False
    env_cfg.domain_rand.randomize_kp = False
    env_cfg.domain_rand.randomize_kd = False
    env_cfg.domain_rand.randomize_initial_joint_pos = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.delay = False


def configure_moving_command(env_cfg) -> None:
    """将速度指令固定为沿地形前进 1 m/s。"""
    env_cfg.commands.resampling_time = 1.0e9
    env_cfg.commands.heading_command = True
    env_cfg.commands.curriculum = False
    env_cfg.commands.random_states = False
    env_cfg.commands.ranges.lin_vel_x = [1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-math.pi, math.pi]
    env_cfg.commands.ranges.heading = [0.0, 0.0]


def configure_recording_env(env_cfg, terrain_name: str) -> None:
    """配置单环境、固定难度的指定地形和离屏相机渲染。"""
    env_cfg.seed = 1
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = VIDEO_DURATION_S
    env_cfg.env.enable_camera_sensors = True
    env_cfg.env.camera_width = VIDEO_WIDTH
    env_cfg.env.camera_height = VIDEO_HEIGHT
    env_cfg.env.camera_horizontal_fov = CAMERA_HORIZONTAL_FOV
    disable_randomization(env_cfg)
    configure_moving_command(env_cfg)
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.selected = False
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = TERRAIN_ROW
    terrain_proportions = [0.0] * len(env_cfg.terrain.terrain_proportions)
    terrain_proportions[TERRAIN_NAMES.index(terrain_name)] = 1.0
    env_cfg.terrain.terrain_proportions = terrain_proportions


# ---------- 模型与环境 ----------

def load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """读取 checkpoint 中的模型参数。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no model state dictionary: {checkpoint_path}")
    return state_dict


def make_policy(env, train_cfg, args, state_dict: dict[str, torch.Tensor]):
    """创建推理策略并加载固定 checkpoint。"""
    args.resume = False
    args.resume_path = None
    train_cfg.runner.resume = False
    train_cfg.runner.resume_path = None
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
    model = runner.alg.actor_critic
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def disable_termination(env) -> None:
    """禁止运行中自动重置，保持录制连续。"""
    def check_termination_noop(self):
        self.reset_buf[:] = False
        self.time_out_buf[:] = False
        self.stuck_buf[:] = False

    env.check_termination = MethodType(check_termination_noop, env)


def destroy_env(env) -> None:
    """释放仿真器资源。"""
    try:
        env.gym.destroy_sim(env.sim)
    except Exception:
        pass


def set_terrain_origin(env, terrain_name: str) -> None:
    """将唯一环境放置到指定难度的目标地形。"""
    if not getattr(env, "custom_origins", False):
        raise ValueError("Terrain recording requires trimesh terrain origins")
    env.terrain_levels[:] = TERRAIN_ROW
    env.terrain_types[:] = 0
    env.terrain_difficulty_factor[:] = TERRAIN_DIFFICULTY
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    env.env_terrain_ids[:] = env.terrain_cols2id_tensor[env.terrain_types]
    actual_terrain_name = env.terrain_names.get(int(env.env_terrain_ids[0].item()))
    if actual_terrain_name != terrain_name:
        raise RuntimeError(f"Expected terrain '{terrain_name}', got '{actual_terrain_name}'")


def set_moving_commands(env) -> None:
    """写入固定前进速度和朝向指令。"""
    env.commands[:, 0] = 1.0
    env.commands[:, 1] = 0.0
    env.commands[:, 3] = 0.0
    forward = quat_apply(env.base_quat, env.forward_vec)
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    yaw_min, yaw_max = env.command_ranges["ang_vel_yaw"]
    env.commands[:, 2] = torch.clamp(wrap_to_pi(-heading), yaw_min, yaw_max)


def write_simulator_state(env) -> None:
    """将机器人根状态和关节状态写回仿真器。"""
    env_ids_int32 = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )


def reset_history_buffers(env) -> None:
    """重置与观测历史相关的环境缓冲区。"""
    env.episode_length_buf[:] = 0
    env.reset_buf[:] = False
    env.last_actions[:] = 0.0
    env.last_last_actions[:] = 0.0
    env.actions[:] = 0.0
    env.last_dof_vel[:] = env.dof_vel
    env.last_root_vel[:] = env.root_states[:, 7:13]
    env.stuck_time[:] = 0.0
    env.last_contacts[:] = False


def refresh_observations(env) -> torch.Tensor:
    """刷新物理状态并用当前观测填充历史缓冲区。"""
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)
    if env.cfg.terrain.measure_heights:
        env.measured_heights = env._get_heights()
    current_obs = env.get_current_obs()[:, :env.num_one_step_obs]
    env.obs_buf[:] = current_obs.repeat(1, env.history_length)
    return env.obs_buf


def euler_xyz_to_quat(roll: float, pitch: float, yaw: float, device) -> torch.Tensor:
    """将 XYZ 欧拉角转换为 Isaac Gym 四元数。"""
    half_roll, half_pitch, half_yaw = roll * 0.5, pitch * 0.5, yaw * 0.5
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return torch.tensor(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        device=device,
        dtype=torch.float,
    )


def initialize_fixed_pose(env, euler_xyz: tuple[float, float, float], height: float) -> None:
    """将机器人初始化为指定高度的固定姿态。"""
    env.root_states[:] = env.base_init_state
    env.root_states[:, :3] += env.env_origins
    env.root_states[:, 2] = env.env_origins[:, 2] + height
    env.root_states[:, 3:7] = euler_xyz_to_quat(*euler_xyz, device=env.device)
    env.root_states[:, 7:13] = 0.0
    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel[:] = 0.0
    write_simulator_state(env)
    reset_history_buffers(env)
    refresh_observations(env)


# ---------- 离屏录制 ----------

def get_camera_tensor(env) -> tuple[int, torch.Tensor]:
    """返回离屏相机及其 GPU 图像张量。"""
    camera_handle = getattr(env, "camera_sensor", None)
    if camera_handle is None or camera_handle < 0:
        raise RuntimeError("Isaac Gym camera sensor was not initialized")
    image_tensor = env.gym.get_camera_image_gpu_tensor(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
    if image_tensor is None:
        raise RuntimeError("Failed to acquire the Isaac Gym camera image tensor")
    camera_tensor = gymtorch.wrap_tensor(image_tensor)
    expected_shape = (VIDEO_HEIGHT, VIDEO_WIDTH, 4)
    if tuple(camera_tensor.shape) != expected_shape:
        raise RuntimeError(f"Unexpected camera image shape: expected {expected_shape}, got {tuple(camera_tensor.shape)}")
    return camera_handle, camera_tensor


def update_follow_camera(env, camera_handle: int) -> None:
    """更新相机世界坐标，使画面跟随机器人。"""
    root_position = env.root_states[0, :3].detach().cpu().numpy().astype(np.float64)
    camera_position = root_position + CAMERA_OFFSET
    camera_target = root_position + CAMERA_TARGET_OFFSET
    env.gym.set_camera_location(
        camera_handle,
        env.envs[0],
        gymapi.Vec3(*(float(value) for value in camera_position)),
        gymapi.Vec3(*(float(value) for value in camera_target)),
    )


def capture_frame(env, camera_handle: int, camera_tensor: torch.Tensor) -> np.ndarray:
    """渲染并返回一帧连续 RGB 画面。"""
    update_follow_camera(env, camera_handle)
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    env.gym.start_access_image_tensors(env.sim)
    try:
        return camera_tensor[:, :, :3].detach().cpu().numpy().copy()
    finally:
        env.gym.end_access_image_tensors(env.sim)


def start_video_encoder(output_path: Path) -> subprocess.Popen:
    """启动 FFmpeg，从标准输入接收 RGB 原始帧。"""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise FileNotFoundError("FFmpeg is required to encode the recorded frames")
    command = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-framerate",
        str(VIDEO_FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        process.kill()
        process.wait()
        raise RuntimeError("Failed to open the FFmpeg input pipe")
    return process


def finish_video_encoder(process: subprocess.Popen) -> None:
    """关闭 FFmpeg 输入并检查编码结果。"""
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, process.args)


def record_recovery_video(checkpoint_path: Path, terrain_name: str) -> Path:
    """在无 viewer 模式下录制恢复与通过指定地形的过程。"""
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    if terrain_name not in TERRAIN_NAMES:
        raise ValueError(f"Unsupported terrain '{terrain_name}', choose from: {', '.join(TERRAIN_NAMES)}")

    args = get_args()
    args.task = TASK_NAME
    args.headless = True
    if args.sim_device != "cuda:0" or args.rl_device != "cuda:0":
        raise ValueError("Off-screen camera recording requires cuda:0; select a physical GPU with CUDA_VISIBLE_DEVICES")
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    configure_recording_env(env_cfg, terrain_name)
    reset_evaluation_seed()
    state_dict = load_state_dict(checkpoint_path)
    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    disable_termination(env)

    try:
        model = make_policy(env, train_cfg, args, state_dict)
        env.reset()
        set_terrain_origin(env, terrain_name)
        initialize_fixed_pose(env, SUPINE_POSE, height=POSE_HEIGHT)
        set_moving_commands(env)
        obs = refresh_observations(env)
        zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs = env.step(zero_actions)[0]
        camera_handle, camera_tensor = get_camera_tensor(env)

        output_path = checkpoint_path.parent / "output" / Path(__file__).stem / f"{TASK_NAME}_{checkpoint_path.stem}_{terrain_name}_supine_recovery.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_steps = int(round(VIDEO_DURATION_S / env.dt))
        frame_stride = max(1, int(round(1.0 / (VIDEO_FPS * env.dt))))
        encoder = start_video_encoder(output_path)
        frame_index = 0
        try:
            encoder.stdin.write(capture_frame(env, camera_handle, camera_tensor).tobytes())
            frame_index += 1
            for step in range(total_steps):
                set_moving_commands(env)
                with torch.no_grad():
                    actions = model.act_inference(obs)
                obs = env.step(actions)[0]
                if (step + 1) % frame_stride == 0:
                    encoder.stdin.write(capture_frame(env, camera_handle, camera_tensor).tobytes())
                    frame_index += 1
        finally:
            finish_video_encoder(encoder)

        print(f"Recorded frames: {frame_index}")
        print(f"Video: {output_path}")
        return output_path
    finally:
        if torch.cuda.is_available() and str(env.device).startswith("cuda"):
            torch.cuda.synchronize(env.device)
        destroy_env(env)


# ---------- main ----------

def parse_recording_args() -> argparse.Namespace | None:
    """解析录制脚本参数，并将其余参数保留给 Isaac Gym。"""
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(f"Usage: {Path(sys.argv[0]).name} --input_dir MODEL_PATH --terrain TERRAIN [Isaac Gym options]")
        print("--input_dir accepts an absolute path or a path relative to the project root.")
        print(f"--terrain choices: {', '.join(TERRAIN_NAMES)}\n")
        return None

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--terrain", choices=TERRAIN_NAMES, required=True)
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]
    return args


def main() -> None:
    recording_args = parse_recording_args()
    if recording_args is None:
        get_args()
        return
    record_recovery_video(recording_args.input_dir, recording_args.terrain)
    # Isaac Gym Preview 4 的离屏 GPU 相机在 Python 解释器退出阶段会触发已知的原生资源二次释放。
    # 此时 simulator 和 FFmpeg 均已显式关闭，直接结束独立进程可避免该崩溃。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
