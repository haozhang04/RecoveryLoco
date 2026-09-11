# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go1RecoverCfg( LeggedRobotCfg ):
    class env:
        num_envs = 4096
        num_one_step_observations = 45
        num_feet = 4
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 + 187 + num_feet # additional: base_lin_vel, external_forces, scan_dots, contact 
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # terrain types: ["smooth_slope", "rough_slope", "stairs_up", "stairs_down", "obstacles"]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        class terrain_params( LeggedRobotCfg.terrain.terrain_params ):
            # slope = difficulty * slope_scale
            slope_scale = 0.64
            # step_height = step_height_base + step_height_scale * difficulty
            step_height_base = 0.05
            step_height_scale = 0.19
            # discrete_obstacles_height = discrete_obstacles_height_base + discrete_obstacles_height_scale * difficulty
            discrete_obstacles_height_base = 0.05
            discrete_obstacles_height_scale = 0.22

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,    # [rad]
            'RL_hip_joint': 0.1,    # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

        stand_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,     # [rad]
            'RL_hip_joint': 0.0,     # [rad]
            'FR_hip_joint': 0.0 ,    # [rad]
            'RR_hip_joint': 0.0,     # [rad]

            'FL_thigh_joint': 0.55,  # [rad]
            'RL_thigh_joint': 0.55,  # [rad]
            'FR_thigh_joint': 0.55,  # [rad]
            'RR_thigh_joint': 0.55,  # [rad]

            'FL_calf_joint': -1.38,  # [rad]
            'RL_calf_joint': -1.38,  # [rad]
            'FR_calf_joint': -1.38,  # [rad]
            'RR_calf_joint': -1.38,  # [rad]
        }
        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 40.0}  # [N*m/rad]
        # damping = {'joint': 1.0}     # [N*m*s/rad]
        stiffness = {'joint': 20.0}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class commands( LeggedRobotCfg.commands ):
            curriculum = True
            random_states = True
            recovery_curriculum = True
            random_states_init_tilt = 0.5 # 初始倾斜角度
            max_curriculum = 2.0
            max_curriculum_y = 1.0
            num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 10. # time before command are changed[s]
            heading_command = True # if true: compute ang vel command from heading error
            zero_command_prob = 0.05
            class ranges( LeggedRobotCfg.commands.ranges):
                lin_vel_x = [-0.5, 0.5]     # min max [m/s]
                lin_vel_y = [-0.5, 0.5]     # min max [m/s]
                ang_vel_yaw = [-3.14, 3.14] # min max [rad/s]
                heading = [-3.14, 3.14]
            
            # [smooth_slope, rough slope, stairs up, stairs down, obstacles, stepping stones, gap, pit]
            terrain_max_command_ranges = {
                "stairs_up": {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-3.14, 3.14] , 'heading': [-3.14, 3.14]},
                "stairs_down": {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-3.14, 3.14], 'heading': [-3.14, 3.14]},
                "obstacles": {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-3.14, 3.14], 'heading': [-3.14, 3.14]},
            }

    # class asset( LeggedRobotCfg.asset ):
    #     file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
    #     name = "go2"
    #     foot_name = "foot"
    #     penalize_contacts_on = ["thigh", "calf", "base"]
    #     terminate_after_contacts_on = []
    #     # privileged_contacts_on = ["base", "thigh", "calf"]
    #     self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    #     flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new_2.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        # privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
    
    class rewards( LeggedRobotCfg.rewards ):
        upright_reward_gate = True
        curriculum_rewards = [
            # {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 10.0},
            # {'reward_name': 'stand_pos', 'start_iter': 0, 'end_iter': 3000, 'start_value': 1.0, 'end_value': 0.0},
            # {'reward_name': 'foot_mirror_up', 'start_iter': 0, 'end_iter': 3000, 'start_value': 1.0, 'end_value': 0.0},
            # {'reward_name': 'feet_contact_forces', 'start_iter': 0, 'end_iter': 3000, 'start_value': 1.0, 'end_value': 0.1},
        ]

        class scales:
            # task
            termination = -200.0
            tracking_lin_vel_up = 2.0
            tracking_ang_vel_up = 1.0
            joint_power = -2e-5

            # safety:
            dof_acc = -2.5e-7
            action_rate = -0.01
            smoothness = -0.01
            collision = -1.0
            dof_pos_limits = -10.0
            dof_vel_limits = -10.0
            feet_contact_forces = -0.002

            # style:
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            # orientation = -0.2
            base_height = -1.0

            feet_air_time = 1.0
            # feet_regulation = -0.05
            # has_contact = 0.5
            # hip_pos_up = -0.3
            stand_pos = -0.03
            foot_mirror_up = -0.3

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        using_compute_terrain_reward = False
        terrain_core_relax = 0.3      # 核心 tracking 项放宽系数：地形越难，越容易拿到 tracking 奖励。
        terrain_penalty_relief = 0.3  # 惩罚减轻系数：地形越难，非核心负向项惩罚越轻。
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        base_height_target = 0.40
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.30

class Go1RecoverCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        est = 'gru'

    class runner( LeggedRobotCfgPPO.runner ):
        experiment_name = 'go1'
  
