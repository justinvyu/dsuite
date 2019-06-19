# Copyright 2019 The DSuite Authors.
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

"""Pose tasks with DClaw robots.

The DClaw is tasked to match a pose defined by the environment.
"""

import abc
import collections
from typing import Any, Dict, Optional

import numpy as np

from dsuite.controllers.robot import RobotState
from dsuite.dclaw.base_env import BaseDClawEnv
from dsuite.utils.configurable import configurable
from dsuite.utils.resources import get_asset_path

DCLAW3_ASSET_PATH = 'dsuite_scenes/dclaw/dclaw3xh.xml'


class BaseDClawPose(BaseDClawEnv, metaclass=abc.ABCMeta):
    """Shared logic for DClaw pose tasks."""

    def __init__(self,
                 asset_path: str = DCLAW3_ASSET_PATH,
                 device_path: Optional[str] = None,
                 frame_skip: int = 40,
                 **kwargs):
        """Initializes the environment.

        Args:
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            device_path: The device path to Dynamixel hardware.
            frame_skip: The number of simulation steps per environment step.
        """
        super().__init__(
            sim_model=get_asset_path(asset_path),
            robot_config=self.get_config_for_device(device_path),
            frame_skip=frame_skip,
            **kwargs)

        self._initial_pos = np.zeros(9)
        self._desired_pos = np.zeros(9)

    def _reset(self):
        """Resets the environment."""
        # Mark the target position in sim.
        self.robot.set_state({
            'dclaw': RobotState(qpos=self._initial_pos, qvel=np.zeros(9)),
        })

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({'dclaw': action})

    def get_obs_dict(self) -> Dict[str, Any]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        state = self.robot.get_state('dclaw')

        return collections.OrderedDict((
            ('qpos', state.qpos),
            ('qvel', state.qvel),
            ('qpos_error', self._desired_pos - state.qpos),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        qvel = obs_dict['qvel']

        reward_dict = collections.OrderedDict((
            ('pose_error_cost', -1 * np.linalg.norm(obs_dict['qpos_error'])),
            # Reward for low velocity.
            ('joint_vel_cost', -0.1 * np.linalg.norm(qvel[qvel >= 4.5])),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        return collections.OrderedDict((
            ('points', 1.0 + reward_dict['pose_error_cost']),
            ('success', np.max(np.abs(obs_dict['qpos_error']), axis=1) < 0.15),
        ))

    def _make_random_pose(self) -> np.ndarray:
        """Returns a random pose."""
        pos_range = self.robot.get_config('dclaw').qpos_range
        pose = self.np_random.uniform(low=pos_range[:, 0], high=pos_range[:, 1])
        # Spread middle joints outwards to avoid entanglement.
        pose[[1, 4, 7]] -= 0.75
        return pose


@configurable(pickleable=True)
class DClawPoseStatic(BaseDClawPose):
    """Track a static random initial and final pose."""

    def _reset(self):
        self._initial_pos = self._make_random_pose()
        self._desired_pos = self._make_random_pose()
        super()._reset()


@configurable(pickleable=True)
class DClawPoseDynamic(BaseDClawPose):
    """Track a dynamic pose."""

    def _reset(self):
        # Choose two poses to oscillate between.
        pose_a = self._make_random_pose()
        pose_b = self._make_random_pose()
        self._initial_pos = 0.5 * (pose_a + pose_b)
        self._dynamic_range = 0.5 * np.abs(pose_b - pose_a)

        # Initialize a random oscilliation period.
        self._period = self.np_random.uniform(
            low=-2.0,
            high=2.0,
            size=len(self.robot.get_config('dclaw').qpos_indices))

        self._update_desired_pose()
        super()._reset()

    def _update_desired_pose(self):
        self._desired_pos = (
            self._initial_pos +
            (self._dynamic_range * np.sin(self._period * self.robot.time)))
