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

"""Turn tasks with DClaw robots.

This is a single rotation of an object from an initial angle to a target angle.
"""

import abc
import collections
from typing import Dict, Optional, Sequence

import numpy as np
from transforms3d.euler import euler2quat

from dsuite.dclaw.base_env import (BaseDClawObjectEnv,
                                   DEFAULT_CLAW_RESET_POSE)
from dsuite.utils.configurable import configurable
from dsuite.utils.resources import get_asset_path
from dsuite.utils.circle_math import circle_distance

# The observation keys that are concatenated as the environment observation.
DEFAULT_OBSERVATION_KEYS = (
    'claw_qpos',
    'object_position',
    'object_orientation_cos',
    'object_orientation_sin',
    'last_action',
    'target_orientation_cos',
    'target_orientation_sin',
    'object_to_target_relative_position',
)

DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_valve3_free.xml'


class BaseDClawTurnFreeObject(BaseDClawObjectEnv, metaclass=abc.ABCMeta):
    """Shared logic for DClaw turn tasks."""

    def __init__(self,
                 asset_path: str = DCLAW3_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 device_path: Optional[str] = None,
                 frame_skip: int = 40,
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            device_path: The device path to Dynamixel hardware.
            frame_skip: The number of simulation steps per environment step.
        """

        super().__init__(
            sim_model=get_asset_path(asset_path),
            robot_config=self.get_config_for_device(device_path, free=True),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)

        self._desired_claw_pos = DEFAULT_CLAW_RESET_POSE
        self._last_action = np.zeros(9)

        self._target_bid = self.model.body_name2id('target')

        # The following are modified (possibly every reset) by subclasses.
        self._initial_object_qpos = (0, 0, 0, 0, 0, 0)
        self._initial_object_qvel = (0, 0, 0, 0, 0, 0)
        self._set_target_object_qpos((0, 0, 0, 0, 0, 0))

    def _reset(self):
        """Resets the environment."""

        self._reset_dclaw_and_object(
            claw_pos=DEFAULT_CLAW_RESET_POSE,
            object_pos=np.atleast_1d(self._initial_object_qpos),
            object_vel=np.atleast_1d(self._initial_object_qvel),
            # guide_pos=np.atleast_1d(self._object_target_qpos))
        )

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({
            'dclaw': action,
            # 'guide': np.atleast_1d(self._object_target_qpos),
        })
        # Save the action to add to the observation.
        self._last_action = action

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """

        claw_state, object_state = self.robot.get_state(['dclaw', 'object'])

        object_position = object_state.qpos[:3]
        object_orientation = object_state.qpos[3:]

        target_position = self._object_target_position
        target_orientation = self._object_target_orientation

        object_to_target_relative_position = object_position - target_position
        object_to_target_relative_orientation = object_orientation - target_orientation
        object_to_target_circle_distance = circle_distance(
            object_orientation, target_orientation)

        return collections.OrderedDict((
            ('claw_qpos', claw_state.qpos),
            ('claw_qvel', claw_state.qvel),
            ('object_position', object_position),
            ('object_orientation', object_orientation),
            ('object_orientation_cos', np.cos(object_orientation)),
            ('object_orientation_sin', np.sin(object_orientation)),
            ('object_qvel', object_state.qvel),
            ('last_action', self._last_action),
            ('target_position', target_position),
            ('target_orientation', target_orientation),
            ('target_orientation_cos', np.cos(target_orientation)),
            ('target_orientation_sin', np.sin(target_orientation)),
            ('object_to_target_relative_position', object_to_target_relative_position),
            ('object_to_target_relative_orientation', object_to_target_relative_orientation),
            ('object_to_target_circle_distance', object_to_target_circle_distance),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        # object_to_target_relative_orientation = obs_dict['object_to_target_relative_orientation']
        object_to_target_relative_position = obs_dict['object_to_target_relative_position']
        claw_vel = obs_dict['claw_qvel']

        object_to_target_circle_distance = obs_dict['object_to_target_circle_distance']

        reward_dict = collections.OrderedDict((
            # Penalty for distance away from goal.
            ('object_to_target_distance_cost', -5 * np.linalg.norm(
                object_to_target_relative_position)),
            ('object_to_target_orientation_cost', -5 * np.linalg.norm(
                object_to_target_circle_distance)),
            # Penalty for difference with nomimal pose.
            ('pose_diff_cost',
             -1 * np.linalg.norm(obs_dict['claw_qpos'] - self._desired_claw_pos)
            ),
            # Penality for high velocities.
            ('joint_vel_cost', -1 * np.linalg.norm(claw_vel[claw_vel >= 0.5])),

            # Reward for close proximity with goal.
            ('bonus_small', 10 * (object_to_target_circle_distance[:, 2] < 0.25)),
            ('bonus_big', 50 * (object_to_target_circle_distance[:, 2] < 0.10)),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""

        return collections.OrderedDict((
            ('points', 1.0 - np.minimum(
                obs_dict['object_to_target_circle_distance'][:, 2], np.pi) / np.pi),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))

    def _set_target_object_qpos(self, target_qpos: float):
        """Sets the goal position and orientation."""
        # Modulo to [-pi, pi].
        self._object_target_position = target_qpos[:3]
        self._object_target_orientation = np.mod(
            np.array(target_qpos[3:]) + np.pi, 2 * np.pi) - np.pi

        # Mark the target position in sim.
        ## TODO: Figure out what this should be

        self.model.body_pos[self._target_bid] = self._object_target_position
        self.model.body_quat[self._target_bid] = euler2quat(
            *self._object_target_orientation)


@configurable(pickleable=True)
class DClawTurnFreeValve3Fixed(BaseDClawTurnFreeObject):
    """Turns the object with a fixed initial and fixed target position."""

    def _reset(self):
        # Turn from 0 degrees to 180 degrees.
        self._initial_object_qpos = (0, 0, 0, 0, 0, 0)
        self._set_target_object_qpos((0, 0, 0, 0, 0, np.pi))
        super()._reset()


@configurable(pickleable=True)
class DClawTurnFreeValve3ResetFree(BaseDClawTurnFreeObject):
    """Turns the object reset-free with a fixed initial and varied target positions."""

    def __init__(self,
                 swap_goal_upon_completion: bool = True,
                 reset_fingers=True,
                 **kwargs):
        self._last_claw_qpos = DEFAULT_CLAW_RESET_POSE
        self._last_object_position = np.array([0, 0, 0])
        self._last_object_orientation = np.array([0, 0, 0])
        self._reset_fingers = reset_fingers

        super().__init__(**kwargs)
        self._swap_goal_upon_completion = swap_goal_upon_completion
        self._goals = [np.pi, 0]
        self._goal_index = 1

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """

        obs_dict = super().get_obs_dict()
        self._last_claw_qpos = obs_dict['claw_qpos']
        self._last_object_position = obs_dict['object_position']
        self._last_object_orientation = obs_dict['object_orientation']
        return obs_dict

    def _get_goal_qpos(self, obs_dict):
        if self._swap_goal_upon_completion and \
           obs_dict['object_to_target_circle_distance'][2] < 0.10:
            self._goal_index = np.mod(self._goal_index + 1, 2)
        else:
            goal = np.pi

        goal = self._goals[self._goal_index]
        return (0, 0, 0, 0, 0, goal)

    def reset(self):
        obs_dict = self.get_obs_dict()
        if self._reset_fingers:
            for _ in range(15):
                self._step(DEFAULT_CLAW_RESET_POSE)
            self._set_target_object_qpos(self._get_goal_qpos(obs_dict))
        else:
            self._set_target_object_qpos(self._get_goal_qpos(obs_dict))
        return self._get_obs(self.get_obs_dict())

@configurable(pickleable=True)
class DClawTurnFreeValve3Image(BaseDClawTurnFreeObject):
    """
    Observation including the image.
    """

    def __init__(self, 
                image_shape: np.ndarray, 
                init_angle_range=(0., 0.),
                target_angle_range=(np.pi, np.pi),
                init_x_pos_range=(0., 0.),
                init_y_pos_range=(0., 0.),
                *args, **kwargs):
        self._image_shape = image_shape
        self._init_angle_range = init_angle_range
        self._target_angle_range = target_angle_range
        self._init_x_pos_range = init_x_pos_range
        self._init_y_pos_range = init_y_pos_range
        super(DClawTurnFreeValve3Image, self).__init__(*args, **kwargs)

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        width, height = self._image_shape[:2]
        obs = super(DClawTurnFreeValve3Image, self).get_obs_dict()
        image = self.render(mode='rgb_array', \
                            width=width, 
                            height=height,
                            camera_id=1).reshape(-1)
        obs['image'] = ((2.0 / 255.0) * image - 1.0) # Normalize between [-1, 1]
        return obs

    def _reset(self):
        lows, highs = list(zip(self._init_angle_range, 
                               self._target_angle_range, 
                               self._init_x_pos_range,
                               self._init_y_pos_range))
        init_angle, target_angle, x_pos, y_pos = np.random.uniform(
            low=lows, high=highs
        )
        # init_angle = np.random.uniform(
        #         low=self._init_angle_range[0], high=self._init_angle_range[1])
        # target_angle = np.random.uniform(
        #     low=self._target_angle_range[0], high=self._target_angle_range[1])

        self._initial_object_qpos = (x_pos, y_pos, 0, 0, 0, init_angle)
        self._set_target_object_qpos((0, 0, 0, 0, 0, target_angle))
        print(init_angle, target_angle)
        super()._reset()

# @configurable(pickleable=True)
# class DClawTurnFreeObjectRandom(BaseDClawTurnFreeObject):
#     """Turns the object with a random initial and random target position."""

#     def _reset(self):
#         # Initial position is +/- 60 degrees.
#         self._initial_object_pos = self.np_random.uniform(
#             low=-np.pi / 3, high=np.pi / 3)
#         # Target position is 180 +/- 60 degrees.
#         self._set_target_object_pos(
#             np.pi + self.np_random.uniform(low=-np.pi / 3, high=np.pi / 3))
#         super()._reset()


# @configurable(pickleable=True)
# class DClawTurnRandomDynamics(DClawTurnRandom):
#     """Turns the object with a random initial and random target position.

#     The dynamics of the simulation are randomized each episode.
#     """

#     def _reset(self):
#         self._randomize_claw_sim()
#         self._randomize_object_sim()
#         super()._reset()
