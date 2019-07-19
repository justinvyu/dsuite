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

"""Standing tasks with D'Kitty robots.

The goal is to stand upright from an initial configuration.
"""

import abc
import collections
from typing import Dict, Optional, Sequence, Union

import numpy as np

from dsuite.dkitty.base_env import BaseDKittyEnv
from dsuite.simulation.randomize import SimRandomizer
from dsuite.utils.configurable import configurable
from dsuite.utils.resources import get_asset_path

DKITTY_ASSET_PATH = 'dsuite/dkitty/assets/dkitty_walk-v0.xml'

DEFAULT_OBSERVATION_KEYS = (
    'root_qpos',
    'kitty_qpos',
    'root_qvel',
    'kitty_qvel',
    'last_action',
    'upright',
    'pose_error',
)


class BaseDKittyStand(BaseDKittyEnv, metaclass=abc.ABCMeta):
    """Shared logic for DKitty turn tasks."""

    def __init__(self,
                 asset_path: str = DKITTY_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 device_path: Optional[str] = None,
                 torso_tracker_id: Optional[Union[str, int]] = None,
                 frame_skip: int = 40,
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            device_path: The device path to Dynamixel hardware.
            torso_tracker_id: The device index or serial of the tracking device
                for the D'Kitty torso.
            frame_skip: The number of simulation steps per environment step.
        """
        super().__init__(
            sim_model=get_asset_path(asset_path),
            robot_config=self.get_robot_config(device_path),
            tracker_config=self.get_tracker_config(torso=torso_tracker_id,),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)

        self._last_action = np.zeros(12)
        self._desired_pose = np.zeros(12)
        self._initial_pose = np.zeros(12)

    def _reset(self):
        """Resets the environment."""
        self._reset_dkitty_standing(kitty_pos=self._initial_pose,)

        # Let gravity pull the simulated robot to the ground before starting.
        if not self.has_hardware_robot:
            self.robot.step({'dkitty': self._initial_pose}, denormalize=False)
            self.sim_scene.advance(100)

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({
            'dkitty': action,
        })
        # Save the action to add to the observation.
        self._last_action = action

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        root_sim_state, robot_state = self.robot.get_state(['root', 'dkitty'])
        torso_track_state = self.tracker.get_state('torso')
        root_qpos, root_qvel = self._get_root_qpos_qvel(root_sim_state,
                                                        torso_track_state)

        # Get the alignment of the torso's z-axis with the global z-axis.
        torso_upright = torso_track_state.rot[2, 2]

        return collections.OrderedDict((
            ('root_qpos', root_qpos),
            ('root_qvel', root_qvel),
            ('kitty_qpos', robot_state.qpos),
            ('kitty_qvel', robot_state.qvel),
            ('last_action', self._last_action),
            ('upright', torso_upright),
            ('pose_error', self._desired_pose - robot_state.qpos),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        pose_mean_error = np.abs(obs_dict['pose_error']).mean(axis=1)
        upright = obs_dict['upright']

        reward_dict = collections.OrderedDict((
            # Reward for closeness to desired pose.
            ('pose_error_cost', -4 * pose_mean_error),
            # Upright - 1 @ cos(0) to 0 @ cos(25deg).
            ('upright', 2 * upright),
            # Bonus when mean error < 30deg, scaled by uprightedness.
            ('bonus_small', 5 * (pose_mean_error < (np.pi / 6)) * upright),
            # Bonus when mean error < 15deg and upright within 30deg.
            ('bonus_big',
             10 * (pose_mean_error < (np.pi / 12)) * (upright > 0.9)),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        # Normalize pose error by 60deg.
        pose_points = (1 - np.maximum(
            np.abs(obs_dict['pose_error']).mean(axis=1) / (np.pi / 3), 1))

        return collections.OrderedDict((
            ('points', pose_points * obs_dict['upright']),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))

    def get_done(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Returns whether the episode should terminate."""
        # Terminate the episode if more than ~90deg misaligned with z-axis.
        return obs_dict['upright'] < 0


@configurable(pickleable=True)
class DKittyStandFixed(BaseDKittyStand):
    """Stand up from a fixed position."""

    def _reset(self):
        """Resets the environment."""
        self._initial_pose[[0, 3, 6, 9]] = 0
        self._initial_pose[[1, 4, 7, 10]] = np.pi / 4
        self._initial_pose[[2, 5, 8, 11]] = -np.pi / 2
        super()._reset()


@configurable(pickleable=True)
class DKittyStandRandom(BaseDKittyStand):
    """Stand up from a random position."""

    def _reset(self):
        """Resets the environment."""
        limits = self.robot.get_config('dkitty').qpos_range
        self._initial_pose = self.np_random.uniform(
            low=limits[:, 0], high=limits[:, 1])
        super()._reset()


@configurable(pickleable=True)
class DKittyStandRandomDynamics(DKittyStandRandom):
    """Stand up from a random positon with randomized dynamics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._randomizer = SimRandomizer(self.sim_scene, self.np_random)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2, 4),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, 0.05),
        )
        self.sim_scene.upload_height_field(0)
        # Randomize visuals.
        self._randomizer.randomize_geoms(
            ['torso1'],
            color_range=(0.2, 0.9),
        )
        super()._reset()
