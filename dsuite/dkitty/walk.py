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

"""Walk tasks with DKitty robots.

This is a single movement from an initial position to a target position.
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
    'heading',
    'target_error',
)


class BaseDKittyWalk(BaseDKittyEnv, metaclass=abc.ABCMeta):
    """Shared logic for DClaw turn tasks."""

    def __init__(self,
                 asset_path: str = DKITTY_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 device_path: Optional[str] = None,
                 torso_tracker_id: Optional[Union[str, int]] = None,
                 target_tracker_id: Optional[Union[str, int]] = None,
                 heading_tracker_id: Optional[Union[str, int]] = None,
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
            target_tracker_id: The device index or serial of the tracking device
                for the target location.
            heading_tracker_id: The device index or serial of the tracking
                device for the heading direction. This defaults to the target
                tracker.
            frame_skip: The number of simulation steps per environment step.
        """
        if heading_tracker_id is None:
            heading_tracker_id = target_tracker_id

        super().__init__(
            sim_model=get_asset_path(asset_path),
            robot_config=self.get_robot_config(device_path),
            tracker_config=self.get_tracker_config(
                torso=torso_tracker_id,
                target=target_tracker_id,
                heading=heading_tracker_id,
            ),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)

        # Disable the constraint solver in hardware so that mimicked positions
        # do not participate in contact calculations.
        if self.has_hardware_robot:
            self.sim_scene.disable_option(constraint_solver=True)

        self._last_action = np.zeros(12)
        self._initial_target_pos = np.zeros(3)
        self._initial_heading_pos = None

    def _reset(self):
        """Resets the environment."""
        root_reset_pos = self.robot.get_initial_state('root').qpos
        self._reset_dkitty_standing(root_pos=root_reset_pos,)

        # If no heading is provided, head towards the target.
        target_pos = self._initial_target_pos
        heading_pos = self._initial_heading_pos
        if heading_pos is None:
            heading_pos = target_pos

        tracker_states = {
            'target': TrackerState(pos=target_pos),
            'heading': TrackerState(pos=heading_pos),
        }
        # For hardware tracking, reset the torso tracker as the world origin
        # offset by the reset position.
        if self.has_hardware_tracker:
            tracker_states['torso'] = TrackerState(pos=root_reset_pos[:3])
        self.tracker.set_state(tracker_states)

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
        target_state, heading_state, torso_track_state = self.tracker.get_state(
            ['target', 'heading', 'torso'])

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

        target_xy = target_state.pos[:2]
        kitty_xy = torso_track_state.pos[:2]

        # Get the alignment of the torso's z-axis with the global z-axis.
        torso_upright = torso_track_state.rot[2, 2]

        # Get the heading of the torso (the x-axis).
        current_heading = torso_track_state.rot[:2, 0]

        # Get the direction towards the heading location.
        desired_heading = heading_state.pos[:2] - kitty_xy

        # Calculate the alignment of the heading with the desired direction.
        heading = (
            np.dot(current_heading, desired_heading) /
            (np.linalg.norm(current_heading) * np.linalg.norm(desired_heading) +
             1e-8))

        return collections.OrderedDict((
            ('root_qpos', root_qpos),
            ('root_qvel', root_qvel),
            ('kitty_qpos', robot_state.qpos),
            ('kitty_qvel', robot_state.qvel),
            ('last_action', self._last_action),
            ('upright', torso_upright),
            ('heading', heading),
            ('target_pos', target_xy),
            ('target_error', target_xy - kitty_xy),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        target_xy_dist = np.linalg.norm(obs_dict['target_error'])
        upright = obs_dict['upright']
        heading = obs_dict['heading']

        reward_dict = collections.OrderedDict((
            # Reward for proximity to the target.
            ('target_dist_cost', -4 * target_xy_dist),
            # Penalty for falling (> cos(25deg) with vertical)
            ('falling_cost', -500 * (upright < 0.9)),
            # Upright - 1 @ cos(0) to 0 @ cos(25deg).
            ('upright', (upright - 0.9) / 0.1),
            # Heading - 1 @ cos(0) to 0 @ cos(25deg).
            ('heading', 2 * (heading - 0.9) / 0.1),
            # Bonus
            ('bonus_small', 5 * (target_xy_dist < 0.5) + 5 * (heading > 0.9)),
            ('bonus_big', 10 * (target_xy_dist < 0.5) * (heading > 0.9)),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        return collections.OrderedDict((
            ('points', -np.linalg.norm(obs_dict['target_error'])),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))

    def get_done(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Returns whether the episode should terminate."""
        # Terminate the episode if more than ~25deg misaligned with z-axis.
        return obs_dict['upright'] < 0.9


@configurable(pickleable=True)
class DKittyWalkFixed(BaseDKittyWalk):
    """Walk straight towards a fixed location."""

    def _reset(self):
        """Resets the environment."""
        target_dist = 2.0
        target_theta = 0.0
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta), 0
        ])
        super()._reset()


@configurable(pickleable=True)
class DKittyWalkRandom(BaseDKittyWalk):
    """Walk straight towards a random location."""

    def _reset(self):
        """Resets the environment."""
        target_dist = self.np_random.uniform(low=1.0, high=2.0)
        target_theta = self.np_random.uniform(low=-1, high=1)
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta), 0
        ])
        super()._reset()


@configurable(pickleable=True)
class DKittyWalkRandomDynamics(DKittyWalkRandom):
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
