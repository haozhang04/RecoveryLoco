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

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.estimator import HIMEstimator, VAEEstimator, GRUEstimator
from rsl_rl.utils import get_activation

class ActorMultiCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_one_step_obs,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.0,
                        est='him',
                        num_critics=1,
                        **kwargs):
        if kwargs:
            print("ActorMultiCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorMultiCritic, self).__init__()

        activation = get_activation(activation)

        self.history_size = int(num_actor_obs/num_one_step_obs)
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs
        self.num_critics = num_critics
        if num_critics < 1:
            raise ValueError(f"num_critics must be at least 1, got {num_critics}")

        # Estimator
        estimator_classes = {
            'him': HIMEstimator,
            'vae': VAEEstimator,
            'gru': GRUEstimator,
        }
        self.est = est.lower()
        if self.est == 'none':
            self.estimator = None
            mlp_input_dim_a = num_one_step_obs
        else:
            if self.est not in estimator_classes:
                available_estimators = [*estimator_classes, 'none']
                raise ValueError(f"Unsupported estimator '{self.est}'. Available estimators: {available_estimators}")
            estimator_class = estimator_classes[self.est]
            self.estimator = estimator_class(temporal_steps=self.history_size, num_one_step_obs=num_one_step_obs)
            mlp_input_dim_a = num_one_step_obs + 3 + self.estimator.num_latent
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Independent value functions
        self.critics = nn.ModuleList([self._build_critic(mlp_input_dim_c, critic_hidden_dims, activation) for _ in range(self.num_critics)])

        print(f"Actor MLP: {self.actor}")
        print(f"Independent Critic MLPs: {self.num_critics}")
        for index, critic in enumerate(self.critics):
            print(f"Critic {index} MLP: {critic}")
        if self.estimator is None:
            print('Estimator: disabled (current observation only)')
        else:
            print(f'Estimator: {self.estimator.encoder}')

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    def _build_critic(self, mlp_input_dim_c, critic_hidden_dims, activation, num_values=1):
        """Build the critic network, enabling multi-value prediction. Predict 1 value by default."""
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], num_values))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        return nn.Sequential(*critic_layers)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_action_mean(self, obs_history):
        current_obs = obs_history[:, :self.num_one_step_obs]
        if self.estimator is None:
            actor_input = current_obs
        else:
            with torch.no_grad():
                vel, latent = self.estimator(obs_history)
            actor_input = torch.cat((current_obs, vel, latent), dim=-1)
        return self.actor(actor_input)

    def update_distribution(self, obs_history):
        mean = self.get_action_mean(obs_history)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, observations=None):
        return self.get_action_mean(obs_history)

    def evaluate(self, critic_observations, **kwargs):
        if isinstance(critic_observations, list):
            if len(critic_observations) != self.num_critics:
                raise ValueError(f"Expected {self.num_critics} critic observations, got {len(critic_observations)}")
            values = [critic(observations) for critic, observations in zip(self.critics, critic_observations)]
        else:
            values = [critic(critic_observations) for critic in self.critics]
        return torch.cat(values, dim=-1)
