from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go1RecoverHandstandCfg( LeggedRobotCfg ):
    class env:
        env_name = "Handstand"
        num_envs = 4096
        num_one_step_observations = 45
        num_feet = 4
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 + 187 + num_feet # additional: base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # terrain types: ["smooth_slope", "rough_slope", "stairs_up", "stairs_down", "obstacles"]
        terrain_proportions = [0.25, 0.25, 0.2, 0.2, 0.1]
        class terrain_params( LeggedRobotCfg.terrain.terrain_params ):
            # slope = difficulty * slope_scale
            slope_scale = 0.52  # max slope ~= 25 deg
            # step_height = step_height_base + step_height_scale * difficulty
            step_height_base = 0.03
            step_height_scale = 0.09  # max ~= 0.11
            # discrete_obstacles_height = discrete_obstacles_height_base + discrete_obstacles_height_scale * difficulty
            discrete_obstacles_height_base = 0.03
            discrete_obstacles_height_scale = 0.11  # max ~= 0.13

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

        # handstand: 前脚倒立时的期望关节角度
        descire_joint_angles = { 
            'FL_hip_joint': 0.,   # [rad]
            'RL_hip_joint': 0.,   # [rad]
            'FR_hip_joint': 0. ,  # [rad]
            'RR_hip_joint': 0.,   # [rad]

            'FL_thigh_joint': -0.7,   # [rad]
            'RL_thigh_joint': 0.8,    # [rad]
            'FR_thigh_joint': -0.7,   # [rad]
            'RR_thigh_joint': 0.8,    # [rad]

            'FL_calf_joint': -1.75,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.75,   # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

        # leggedstand: 后脚倒立时的期望关节角度
        # descire_joint_angles = { # = target angles [rad] when action = 0.0
        #     'FL_hip_joint': 0.0,   # [rad]
        #     'RL_hip_joint': 0.0,   # [rad]
        #     'FR_hip_joint': 0.0,   # [rad]
        #     'RR_hip_joint': 0.0,   # [rad]

        #     'FL_thigh_joint': 0.8,     # [rad]
        #     'RL_thigh_joint': 2.25,    # [rad]
        #     'FR_thigh_joint': 0.8,     # [rad]
        #     'RR_thigh_joint': 2.25,    # [rad]

        #     'FL_calf_joint': -1.5,     # [rad]
        #     'RL_calf_joint': -1.75,    # [rad]
        #     'FR_calf_joint': -1.5,     # [rad]
        #     'RR_calf_joint': -1.75,    # [rad]
        # }

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
        random_states_init_tilt = 0.5
        max_curriculum_x = 0.5
        max_curriculum_y = 0.5
        num_commands = 4
        resampling_time = 10.
        heading_command = True
        class ranges( LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1., 1.]
            heading = [-3.14, 3.14]

        terrain_max_command_ranges = {
            "stairs_up": {'lin_vel_x': [-0.5, 0.5], 'lin_vel_y': [-0.5, 0.5], 'ang_vel_yaw': [-1.0, 1.0] , 'heading': [-3.14, 3.14]},
            "stairs_down": {'lin_vel_x': [-0.5, 0.5], 'lin_vel_y': [-0.5, 0.5], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-3.14, 3.14]},
            "obstacles": {'lin_vel_x': [-0.5, 0.5], 'lin_vel_y': [-0.5, 0.5], 'ang_vel_yaw': [-1.0, 1.0], 'heading': [-3.14, 3.14]},
        }

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_new_2.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        # terminate_after_contacts_on = ["base"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        # handstand: 前脚着地撑地，后脚离地抬起，头朝下
        contact_feet_name = ['FL_foot', 'FR_foot']  # 前脚着地，支撑身体
        swing_feet_name = ['RL_foot', 'RR_foot']    # 后脚离地，奖励抬起
        target_gravity = [1.0, 0.0, 0.0]            # 头朝下
        x_lin_vel_sing = -1
        ang_vel_sing = +1
        # leggedstand: 后脚着地撑地，前脚离地抬起，头朝上
        # contact_feet_name = ['RL_foot', 'RR_foot']  # 后脚着地，支撑身体
        # swing_feet_name = ['FL_foot', 'FR_foot']    # 前脚离地，奖励抬起
        # target_gravity = [-1.0, 0.0, 0.0]           # 头朝上
        # x_lin_vel_sing = +1
        # ang_vel_sing = -1

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # handstand 奖励 (handstand_reward_scale 门控)
            stand_tracking_lin_vel = 2.0
            stand_tracking_ang_vel = 2.0

            stand_lin_vel_z = -0.5
            stand_ang_vel_xy = -0.05
            stand_orientation = -1.0
            stand_base_height = -2.0

            stand_action_rate = -0.01
            stand_smoothness = -0.01
            stand_dof_acc = -2.5e-7
            stand_joint_power = -2e-5
            dof_pos_limits = -10.0
            dof_vel_limits = -10.0
            stand_collision = -1.0
            stand_upward = 1.5

            stand_feet_air_time = 3.0
            stand_swing_feet_no_contact = 1.0
            stand_contact_feet_single_contact = 1.0

            stand_hip_pos = -0.5
            # stand_descire_joint_pos = -0.1
            stand_descire_swing_feet_pos = -0.5
            stand_descire_contact_feet_pos = -0.1

        only_positive_rewards = False
        tracking_sigma = 0.25
        base_height_target = 0.50
        max_contact_force = 200.
        cycle_time = 2.0

class Go1RecoverHandstandCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        experiment_name = 'go1_recover_handstand'
