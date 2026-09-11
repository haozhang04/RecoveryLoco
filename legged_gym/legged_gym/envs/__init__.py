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
import os
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.utils.task_registry import task_registry

# -----------------------------------GO1---------------------------------------
# locomotion
from legged_gym.envs.go1_recover.go1_recover_config import Go1RecoverCfg, Go1RecoverCfgPPO
from legged_gym.envs.go1_recover.go1_recover import Go1Recover

# stand
from legged_gym.envs.go1_recover_stand.go1_recover_handstand_config import Go1RecoverHandstandCfg, Go1RecoverHandstandCfgPPO
from legged_gym.envs.go1_recover_stand.go1_recover_leggedstand_config import Go1RecoverLeggedstandCfg, Go1RecoverLeggedstandCfgPPO
from legged_gym.envs.go1_recover_stand.go1_recover_stand import Go1RecoverStand

# gait
from legged_gym.envs.go1_recover_gait.go1_recover_hop_config import Go1RecoverHopCfg, Go1RecoverHopCfgPPO
from legged_gym.envs.go1_recover_gait.go1_recover_bound_config import Go1RecoverBoundCfg, Go1RecoverBoundCfgPPO
from legged_gym.envs.go1_recover_gait.go1_recover_gait import Go1RecoverGait

task_registry.register( "go1", Go1Recover, Go1RecoverCfg(), Go1RecoverCfgPPO())
task_registry.register( "go1_recover_handstand", Go1RecoverStand, Go1RecoverHandstandCfg(), Go1RecoverHandstandCfgPPO())
task_registry.register( "go1_recover_leggedstand", Go1RecoverStand, Go1RecoverLeggedstandCfg(), Go1RecoverLeggedstandCfgPPO())
task_registry.register( "go1_recover_hop", Go1RecoverGait, Go1RecoverHopCfg(), Go1RecoverHopCfgPPO())
task_registry.register( "go1_recover_bound", Go1RecoverGait, Go1RecoverBoundCfg(), Go1RecoverBoundCfgPPO())

# -----------------------------------GO1MultiCritic---------------------------------------
from legged_gym.envs.go1_recover_multi_critic.go1_recover_multi_critic_config import (
    Go1RecoverMultiCriticCfg,
    Go1RecoverMultiCriticCfgPPO,
    Go1RecoverSingleCriticCfg,
    Go1RecoverSingleCriticCfgPPO,
    Go1RecoverMultiCriticCfgPPOHIM,
    Go1RecoverMultiCriticCfgPPOVAE,
    Go1RecoverMultiCriticCfgPPOWithoutSymmetry,
    Go1RecoverMultiCriticCfgWithoutCurriculum,
    Go1RecoverMultiCriticCfgPPOWithoutCurriculum,
    Go1RecoverMultiCriticCfgPPOWithoutEstimator,
    Go1RecoverMultiCriticCfgWithoutUprightGate,
    Go1RecoverMultiCriticCfgPPOWithoutUprightGate,
)
from legged_gym.envs.go1_recover_multi_critic.go1_recover_multi_critic import Go1RecoverMultiCritic
task_registry.register( "go1_recover_multi_critic", Go1RecoverMultiCritic, Go1RecoverMultiCriticCfg(), Go1RecoverMultiCriticCfgPPO())
task_registry.register( "go1_recover_single_critic", Go1RecoverMultiCritic, Go1RecoverSingleCriticCfg(), Go1RecoverSingleCriticCfgPPO())
task_registry.register( "go1_recover_multi_critic_him", Go1RecoverMultiCritic, Go1RecoverMultiCriticCfg(), Go1RecoverMultiCriticCfgPPOHIM())
task_registry.register( "go1_recover_multi_critic_vae", Go1RecoverMultiCritic, Go1RecoverMultiCriticCfg(), Go1RecoverMultiCriticCfgPPOVAE())
task_registry.register( "go1_recover_multi_critic_without_symmetry", Go1RecoverMultiCritic, Go1RecoverMultiCriticCfg(), Go1RecoverMultiCriticCfgPPOWithoutSymmetry())
task_registry.register( "go1_recover_multi_critic_without_curriculum", Go1RecoverMultiCritic, Go1RecoverMultiCriticCfgWithoutCurriculum(), Go1RecoverMultiCriticCfgPPOWithoutCurriculum())
task_registry.register( "go1_recover_multi_critic_without_estimator", Go1RecoverMultiCritic, Go1RecoverMultiCriticCfg(), Go1RecoverMultiCriticCfgPPOWithoutEstimator())
task_registry.register( "go1_recover_multi_critic_without_upright_gate", Go1RecoverMultiCritic, Go1RecoverMultiCriticCfgWithoutUprightGate(), Go1RecoverMultiCriticCfgPPOWithoutUprightGate())
