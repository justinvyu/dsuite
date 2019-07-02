# Copyright 2019 The D'Suite Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gym environment registration for DKitty environments."""

from dsuite.utils.registration import register

#===============================================================================
# Stand tasks
#===============================================================================

# Default number of steps per episode.
_STAND_EPISODE_LEN = 100  # 50*40*2.5ms = 5s

register(
    env_id='DKittyStandFixed-v0',
    class_path='dsuite.dkitty.stand:DKittyStandFixed',
    max_episode_steps=_STAND_EPISODE_LEN)

register(
    env_id='DKittyStandRandom-v0',
    class_path='dsuite.dkitty.stand:DKittyStandRandom',
    max_episode_steps=_STAND_EPISODE_LEN)

register(
    env_id='DKittyStandRandomDynamics-v0',
    class_path='dsuite.dkitty.stand:DKittyStandRandomDynamics',
    max_episode_steps=_STAND_EPISODE_LEN)

#===============================================================================
# Walk tasks
#===============================================================================

# Default number of steps per episode.
_WALK_EPISODE_LEN = 100  # 100*40*2.5ms = 10s

register(
    env_id='DKittyWalkFixed-v0',
    class_path='dsuite.dkitty.walk:DKittyWalkFixed',
    max_episode_steps=_WALK_EPISODE_LEN)

register(
    env_id='DKittyWalkRandom-v0',
    class_path='dsuite.dkitty.walk:DKittyWalkRandom',
    max_episode_steps=_WALK_EPISODE_LEN)

register(
    env_id='DKittyWalkRandomDynamics-v0',
    class_path='dsuite.dkitty.walk:DKittyWalkRandomDynamics',
    max_episode_steps=_WALK_EPISODE_LEN)
