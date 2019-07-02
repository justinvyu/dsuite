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
from transforms3d.euler import mat2euler

from dsuite.controllers.tracking import TrackerState
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
    """Shared logic for DClaw turn tasks."""

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

        # Disable the constraint solver in hardware so that mimicked positions
        # do not participate in contact calculations.
        if self.has_hardware_robot:
            self.sim_scene.disable_option(constraint_solver=True)

        self._last_action = np.zeros(12)
        self._desired_pose = np.zeros(12)
        self._initial_pose = np.zeros(12)

    def _reset(self):
        """Resets the environment."""
        self._reset_dkitty_standing(kitty_pos=self._initial_pose,)

        # For hardware tracking, reset the torso tracker as the world origin
        # offset by the reset position.
        if self.has_hardware_tracker:
            self.tracker.set_state({'torso': TrackerState(pos=np.zeros(3))})

        # Disable actuation and let the simulation settle.
        if not self.has_hardware_robot:
            with self.sim_scene.disable_option_context(actuation=True):
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

        if self.has_hardware_tracker:
            # Use hardware tracking as the root position and mimic back to sim.
            root_qpos = np.concatenate(
                [torso_track_state.pos,
                 mat2euler(torso_track_state.rot)])
            self.data.qpos[:6] = root_qpos
            # TODO(michaelahn): Calculate angular velocity from tracking.
            root_qvel = np.zeros(6)
        else:
            root_qpos = root_sim_state.qpos
            root_qvel = root_sim_state.qvel

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
        xy_dist = np.linalg.norm(obs_dict['root_qpos'][:2])
        upright = obs_dict['upright']

        reward_dict = collections.OrderedDict((
            # Reward for closeness to desired pose.
            ('pose_error_cost', -4 * pose_mean_error),
            # Upright - 1 @ cos(0) to 0 @ cos(25deg).
            ('upright', upright),
            # Penalize being off-center.
            ('xy_distance_cost', -xy_dist),
            # Bonus when mean error < 20deg and upright within 30deg
            ('bonus_small', 5 * (pose_mean_error < 0.35) * (upright > 0.866)),
            # Bonus when mean error < 10deg and upright within 15deg.
            ('bonus_big', 10 * (pose_mean_error < 0.17) * (upright > 0.966)),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        points = (
            -np.abs(obs_dict['pose_error']).mean(axis=1) * obs_dict['upright'])
        return collections.OrderedDict((
            ('points', points),
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
    """Walk straight towards a random location."""

    def _reset(self):
        """Resets the environment."""
        limits = self.robot.get_config('dkitty').qpos_range
        self._initial_pose = self.np_random.uniform(
            low=limits[:, 0], high=limits[:, 1])
        super()._reset()


@configurable(pickleable=True)
class DKittyStandRandomDynamics(DKittyStandRandom):
    """Walk straight towards a random location."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._randomizer = SimRandomizer(self.sim_scene, self.np_random)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        self._randomizer.randomize_dofs(
            self._dof_indices,
            damping_range=(0.9, 1.1),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_geoms(
            ['torso1'],
            color_range=(0.2, 0.9),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, 0.05),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()
