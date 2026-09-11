# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn.functional as F


class Symmetry:
    def __init__(self, data_augmentation_func, mirror_loss_coeff):
        if not isinstance(data_augmentation_func, list):
            raise TypeError("data_augmentation_func must be a list of symmetry functions")
        self.data_augmentation_funcs = [self._resolve(func) for func in data_augmentation_func]
        self.mirror_loss_coeff = mirror_loss_coeff

    def _resolve(self, func):
        if callable(func):
            return func
        resolved = globals().get(func)
        if not callable(resolved):
            raise ValueError(f"Unknown symmetry function: {func}")
        return resolved

    def compute_loss(self, actor_critic, obs):
        losses = [self._compute_single_loss(func, actor_critic, obs) for func in self.data_augmentation_funcs]
        return torch.stack(losses).mean()

    def _compute_single_loss(self, func, actor_critic, obs):
        mirrored_obs, _ = func(obs=obs.detach(), actions=None)
        if mirrored_obs.shape != obs.shape:
            raise ValueError("Symmetry function returned inconsistent observation shapes")
        predicted_mirrored_actions = actor_critic.get_action_mean(mirrored_obs)

        with torch.no_grad():
            target_actions = actor_critic.get_action_mean(obs.detach())
        _, target_mirrored_actions = func(obs=None, actions=target_actions)
        if predicted_mirrored_actions.shape != target_mirrored_actions.shape:
            raise ValueError("Symmetry function returned inconsistent action shapes")
        
        # Mirror consistency: pi(M(obs)) ~= M(pi(obs)).
        return F.mse_loss(predicted_mirrored_actions, target_mirrored_actions.detach())


def m20_x_axis_symmetry(obs=None, actions=None):
    """
    关节顺序：FL、FR、HL、HR 每条腿 4 个关节
    x 轴对称交换左右腿
    """
    indices = [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]
    joint_signs = [-1.0, 1.0, 1.0, 1.0] * 4

    mirrored_obs = None
    if obs is not None:
        if obs.shape[-1] % 57 != 0:
            raise ValueError("Observation size is not divisible by 57")
        mirrored = obs.reshape(obs.shape[0], -1, 57).clone()
        # 命令[0:3]
        mirrored[..., 0:3] *= mirrored.new_tensor([1.0, -1.0, -1.0])
        # 角速度[3:6]
        mirrored[..., 3:6] *= mirrored.new_tensor([-1.0, 1.0, -1.0])
        # 重力[6:9]
        mirrored[..., 6:9] *= mirrored.new_tensor([1.0, -1.0, 1.0])
        # 关节误差[9:25]、关节速度[25:41]、上一时刻动作[41:57]
        for start in (9, 25, 41):
            mirrored[..., start:start + 16] = mirrored[..., start:start + 16][..., indices] * mirrored.new_tensor(joint_signs)
        mirrored_obs = mirrored.reshape_as(obs)

    mirrored_actions = None
    if actions is not None:
        if actions.shape[-1] != 16:
            raise ValueError(f"Expected 16 actions, got {actions.shape[-1]}")
        mirrored_actions = actions[..., indices] * actions.new_tensor(joint_signs)

    return mirrored_obs, mirrored_actions


def m20_y_axis_symmetry(obs=None, actions=None):
    """
    关节顺序：FL、FR、HL、HR 每条腿 4 个关节
    y 轴对称交换前后腿
    """
    indices = [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]
    joint_signs = [1.0, -1.0, -1.0, -1.0] * 4

    mirrored_obs = None
    if obs is not None:
        if obs.shape[-1] % 57 != 0:
            raise ValueError("Observation size is not divisible by 57")
        mirrored = obs.reshape(obs.shape[0], -1, 57).clone()
        # 命令[0:3]、角速度[3:6]、重力[6:9]、
        mirrored[..., 0:3] *= mirrored.new_tensor([-1.0, 1.0, -1.0])
        # 角速度[3:6]
        mirrored[..., 3:6] *= mirrored.new_tensor([1.0, -1.0, -1.0])
        # 重力[6:9]
        mirrored[..., 6:9] *= mirrored.new_tensor([-1.0, 1.0, 1.0])
        # 关节误差[9:25]、关节速度[25:41]、上一时刻动作[41:57]
        for start in (9, 25, 41):
            mirrored[..., start:start + 16] = mirrored[..., start:start + 16][..., indices] * mirrored.new_tensor(joint_signs)
        mirrored_obs = mirrored.reshape_as(obs)

    mirrored_actions = None
    if actions is not None:
        if actions.shape[-1] != 16:
            raise ValueError(f"Expected 16 actions, got {actions.shape[-1]}")
        mirrored_actions = actions[..., indices] * actions.new_tensor(joint_signs)

    return mirrored_obs, mirrored_actions


def m20_xy_axis_symmetry(obs=None, actions=None):
    """
    关节顺序：FL、FR、HL、HR 每条腿 4 个关节
    xy 轴对称等价于水平面绕 z 轴旋转 180 度，交换对角腿
    """
    indices = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    joint_signs = [-1.0] * 16

    mirrored_obs = None
    if obs is not None:
        if obs.shape[-1] % 57 != 0:
            raise ValueError("Observation size is not divisible by 57")
        mirrored = obs.reshape(obs.shape[0], -1, 57).clone()
        # 命令[0:3]
        mirrored[..., 0:3] *= mirrored.new_tensor([-1.0, -1.0, 1.0])
        # 角速度[3:6]
        mirrored[..., 3:6] *= mirrored.new_tensor([-1.0, -1.0, 1.0])
        # 重力[6:9]
        mirrored[..., 6:9] *= mirrored.new_tensor([-1.0, -1.0, 1.0])
        # 关节误差[9:25]、关节速度[25:41]、上一时刻动作[41:57]
        for start in (9, 25, 41):
            mirrored[..., start:start + 16] = mirrored[..., start:start + 16][..., indices] * mirrored.new_tensor(joint_signs)
        mirrored_obs = mirrored.reshape_as(obs)

    mirrored_actions = None
    if actions is not None:
        if actions.shape[-1] != 16:
            raise ValueError(f"Expected 16 actions, got {actions.shape[-1]}")
        mirrored_actions = actions[..., indices] * actions.new_tensor(joint_signs)

    return mirrored_obs, mirrored_actions


def go1_x_axis_symmetry(obs=None, actions=None):
    """
    关节顺序：FR、FL、RR、RL每条腿 3 个关节
    x 轴对称交换左右腿
    """
    indices = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    joint_signs = [-1.0, 1.0, 1.0] * 4

    mirrored_obs = None
    if obs is not None:
        if obs.shape[-1] % 45 != 0:
            raise ValueError("Observation size is not divisible by 45")
        mirrored = obs.reshape(obs.shape[0], -1, 45).clone()
        # 命令[0:3]
        mirrored[..., 0:3] *= mirrored.new_tensor([1.0, -1.0, -1.0])
        # 角速度[3:6]
        mirrored[..., 3:6] *= mirrored.new_tensor([-1.0, 1.0, -1.0])
        # 重力[6:9]
        mirrored[..., 6:9] *= mirrored.new_tensor([1.0, -1.0, 1.0])
        # 关节误差[9:21]、关节速度[21:33]、上一时刻动作[33:45]
        for start in (9, 21, 33):
            mirrored[..., start:start + 12] = mirrored[..., start:start + 12][..., indices] * mirrored.new_tensor(joint_signs)
        mirrored_obs = mirrored.reshape_as(obs)

    mirrored_actions = None
    if actions is not None:
        if actions.shape[-1] != 12:
            raise ValueError(f"Expected 12 actions, got {actions.shape[-1]}")
        mirrored_actions = actions[..., indices] * actions.new_tensor(joint_signs)

    return mirrored_obs, mirrored_actions
