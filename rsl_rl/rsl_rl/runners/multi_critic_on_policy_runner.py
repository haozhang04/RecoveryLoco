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

import time
import os
import inspect
from collections import deque
import statistics
import shutil
import datetime

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import MultiCriticPPO
from rsl_rl.modules import ActorMultiCritic
from rsl_rl.env import VecEnv
from rsl_rl.utils import git_store_code_state


class MultiCriticOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])
        self.policy_cfg["num_critics"] = self.env.num_reward_groups
        actor_critic: ActorMultiCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_one_step_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        if actor_critic.num_critics != self.env.num_reward_groups:
            raise ValueError(
                f"num_critics ({actor_critic.num_critics}) must match "
                f"env.num_reward_groups ({self.env.num_reward_groups})"
            )
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: MultiCriticPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        now = datetime.datetime.now()
        self.log_file_name = f'training_log_{now.strftime("%Y%m%d_%H%M%S")}.txt'
        self.log_write_counter = 0

        _, _ = self.env.reset()
    
    def _copy_config_files(self):
        """Copy files to log directory for training record"""
        if self.log_dir is None:
            return

        # copy config file based on the actual config class (not the env class)
        cfg_class = self.env.cfg.__class__
        try:
            config_source = inspect.getfile(cfg_class)
        except TypeError:
            config_source = None
        if config_source and os.path.exists(config_source):
            config_dest = os.path.join(self.log_dir, os.path.basename(config_source))
            shutil.copy2(config_source, config_dest)
            print(f"[INFO] Configuration file copied to: {config_dest}")
        else:
            print(f"[WARNING] Configuration file not found for {cfg_class}")

        # copy env python file based on the env class
        env_class = self.env.__class__
        try:
            env_source = inspect.getfile(env_class)
        except TypeError:
            env_source = None
        if env_source and os.path.exists(env_source):
            env_dest = os.path.join(self.log_dir, os.path.basename(env_source))
            shutil.copy2(env_source, env_dest)
            print(f"[INFO] Env file copied to: {env_dest}")
        else:
            print(f"[WARNING] Env file not found for {env_class}")
            
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            git_store_code_state(self.log_dir)
            self._copy_config_files()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        reward_group_buffers = [deque(maxlen=100) for _ in range(self.env.num_reward_groups)]
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, self.env.num_reward_groups, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)
                
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                        rewbuffer.extend(cur_reward_sum[new_ids].sum(dim=-1).cpu().numpy().tolist())
                        for reward_index in range(self.env.num_reward_groups):
                            reward_group_buffers[reward_index].extend(
                                cur_reward_sum[new_ids, reward_index].cpu().numpy().tolist()
                            )
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
                
            (
                mean_value_loss,
                mean_value_loss_per_critic,
                mean_surrogate_loss,
                mean_estimation_loss,
                mean_auxiliary_loss,
                mean_symmetry_loss,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                # stuck_buf_count 用求和显示总数
                if key == 'stuck_buf_count':
                    value = torch.sum(infotensor)
                    self.writer.add_scalar('Episode/' + key, value, locs['it'])
                    ep_string += f"""{f'Total episode {key}:':>{pad}} {int(value.item())}\n"""
                else:
                    value = torch.mean(infotensor)
                    self.writer.add_scalar('Episode/' + key, value, locs['it'])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        if self.alg.actor_critic.est == 'vae':
            auxiliary_loss_name = 'VAE loss'
        elif self.alg.actor_critic.est == 'him':
            auxiliary_loss_name = 'Swap loss'
        elif self.alg.actor_critic.est == 'gru':
            auxiliary_loss_name = 'GRU loss'
        elif self.alg.actor_critic.est == 'none':
            auxiliary_loss_name = 'Estimator disabled'
        else:
            raise ValueError(f"Unsupported estimator '{self.alg.actor_critic.est}'")

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        value_loss_string = ''
        for index, value_loss in enumerate(locs['mean_value_loss_per_critic']):
            self.writer.add_scalar(f'Loss/value_function_{index}', value_loss.item(), locs['it'])
            group_name = self.env.reward_group_names[index]
            value_loss_string += f"""{f'Value loss/{group_name}:':>{pad}} {value_loss.item():.4f}\n"""
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/Estimation Loss', locs['mean_estimation_loss'], locs['it'])
        self.writer.add_scalar(f'Loss/{auxiliary_loss_name}', locs['mean_auxiliary_loss'], locs['it'])
        self.writer.add_scalar('Loss/symmetry', locs['mean_symmetry_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            reward_group_string = ''
            for index, reward_buffer in enumerate(locs['reward_group_buffers']):
                if reward_buffer:
                    group_name = self.env.reward_group_names[index]
                    mean_group_reward = statistics.mean(reward_buffer)
                    self.writer.add_scalar(f'Train/mean_reward_{group_name}', mean_group_reward, locs['it'])
                    reward_group_string += f"""{f'Mean reward/{group_name}:':>{pad}} {mean_group_reward:.2f}\n"""
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
        else:
            reward_group_string = ''

        exp_name = self.cfg.get("experiment_name", "unknown_experiment")
        str = f" \033[1m [{exp_name}] Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{value_loss_string}"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{f'{auxiliary_loss_name}:':>{pad}} {locs['mean_auxiliary_loss']:.4f}\n"""
                          f"""{'Symmetry loss:':>{pad}} {locs['mean_symmetry_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{reward_group_string}"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{value_loss_string}"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{f'{auxiliary_loss_name}:':>{pad}} {locs['mean_auxiliary_loss']:.4f}\n"""
                          f"""{'Symmetry loss:':>{pad}} {locs['mean_symmetry_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        log_string += (f"""{'-' * width}\n""")
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)
        self.log_write_counter += 1
        if self.log_write_counter % 1000 == 0:
            log_file_path = os.path.join(self.log_dir, self.log_file_name)
            with open(log_file_path, 'a') as log_file:
                log_file.write(log_string)
            self.log_write_counter = 0
            
    def save(self, path, infos=None):
        checkpoint = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        if self.alg.actor_critic.estimator is not None:
            checkpoint['estimator_optimizer_state_dict'] = self.alg.actor_critic.estimator.optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            if self.alg.actor_critic.estimator is not None:
                self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
