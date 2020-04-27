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

import os
import glob
import gzip
import pickle
import abc
import collections
from typing import Dict, Optional, Sequence

import numpy as np
from transforms3d.euler import euler2quat

from dsuite.dclaw.base_env import (BaseDClawObjectEnv,
                                   BaseDClawMultiObjectEnv,
                                   BaseDClawEnv,
                                   DEFAULT_CLAW_RESET_POSE)
from dsuite.dclaw.turn_free_object import *
from dsuite.utils.configurable import configurable
from dsuite.utils.resources import get_asset_path
from dsuite.utils.circle_math import circle_distance
from dsuite.components.robot.config import ControlMode
from dsuite.components.robot import RobotState


DEFAULT_PUCK_OBSERVATION_KEYS = (
    'claw_qpos',
    'last_action',
    'object_xy_position',
    'target_xy_position',
)

DEFAULT_MULTI_PUCK_OBSERVATION_KEYS = (
    'claw_qpos',
    'last_action',
    'object1_xy_position',
    'object2_xy_position',
    'target1_xy_position',
    'target2_xy_position',
)

class BaseDClawTranslateMultiObject(BaseDClawMultiObjectEnv, metaclass=abc.ABCMeta):
    """Shared logic for DClaw turn tasks."""

    def __init__(self,
                 asset_path: str = DCLAW3_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_MULTI_PUCK_OBSERVATION_KEYS,
                 device_path: Optional[str] = None,
                 camera_config: dict = None,
                 frame_skip: int = 40,
                 free_claw: bool = False,
                 arena_type: str = None,
                 use_bowl_arena: bool = False,
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            device_path: The device path to Dynamixel hardware.
            frame_skip: The number of simulation steps per environment step.
        """
        self._camera_config = camera_config

        asset_path = ARENA_PATHS.get(arena_type, asset_path)
        self._is_hardware = (device_path is not None)

        super().__init__(
            sim_model=get_asset_path(asset_path),
            robot_config=self.get_config_for_device(
                device_path, free_object=True, free_claw=free_claw, quat=False),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)

        self._desired_claw_pos = DEFAULT_CLAW_RESET_POSE.copy()
        self._last_action = np.zeros(9)

        self._target1_bid = self.model.body_name2id('target1')
        self._target2_bid = self.model.body_name2id('target2')

        # The following are modified (possibly every reset) by subclasses.
        self._initial_claw_qpos = DEFAULT_CLAW_RESET_POSE.copy()

        self._initial_object1_qpos = (0.1, 0.1, 0, 0, 0, 0)
        self._initial_object1_qvel = (0, 0, 0, 0, 0, 0)
        self._initial_object2_qpos = (-0.1, -0.1, 0, 0, 0, 0)
        self._initial_object2_qvel = (0, 0, 0, 0, 0, 0)

        self._set_target_object_qpos((0.1, -0.1, 0, 0, 0, 0), (-0.1, 0.1, 0, 0, 0, 0))

    def _reset(self):
        """Resets the environment."""
        self._reset_dclaw_and_object(
            claw_pos=self._initial_claw_qpos,
            object1_pos=np.atleast_1d(self._initial_object1_qpos),
            object1_vel=np.atleast_1d(self._initial_object1_qvel),
            object2_pos=np.atleast_1d(self._initial_object2_qpos),
            object2_vel=np.atleast_1d(self._initial_object2_qvel),
            # guide_pos=np.atleast_1d(self._object_target_qpos))
        )

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({
            'dclaw': action,
        })
        # Save the action to add to the observation.
        self._last_action = action

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """

        claw_state, object1_state, object2_state = self.robot.get_state(['dclaw', 'object1', 'object2'])

        object1_position = object1_state.qpos[:3].copy()
        object1_orientation = object1_state.qpos[3:].copy()
        object1_angle = np.mod(
            np.array(object1_orientation[2]) + np.pi, 2 * np.pi) - np.pi

        object2_position = object2_state.qpos[:3].copy()
        object2_orientation = object2_state.qpos[3:].copy()
        object2_angle = np.mod(
            np.array(object2_orientation[2]) + np.pi, 2 * np.pi) - np.pi

        target1_position = self._object1_target_position
        target1_orientation = self._object1_target_orientation

        target2_position = self._object2_target_position
        target2_orientation = self._object2_target_orientation

        object1_to_target_relative_position = object1_position - target1_position
        object1_to_target_relative_orientation = object1_orientation - target1_orientation
        object1_to_target_circle_distance = circle_distance(object1_orientation, target1_orientation)

        object2_to_target_relative_position = object2_position - target2_position
        object2_to_target_relative_orientation = object2_orientation - target2_orientation
        object2_to_target_circle_distance = circle_distance(object2_orientation, target2_orientation)

        return collections.OrderedDict((
            ('claw_qpos', claw_state.qpos.copy()),
            ('claw_qvel', claw_state.qvel.copy()),
            ('last_action', self._last_action),

            # === OBJECT 1 ===
            ('object1_position', object1_position),
            ('object1_qvel', object1_state.qvel),
            ('object1_xy_position', object1_position[:2]),
            ('object1_angle', object1_angle.reshape(-1)),
            ('object1_orientation', object1_orientation),
            ('object1_orientation_cos', np.cos(object1_orientation)),
            ('object1_orientation_sin', np.sin(object1_orientation)),
            ('object1_z_orientation', object1_orientation[2:]),
            ('object1_z_orientation_cos', np.cos(object1_orientation[2:])),
            ('object1_z_orientation_sin', np.sin(object1_orientation[2:])),
            
            # === OBJECT 2 ===
            ('object2_position', object2_position),
            ('object2_qvel', object2_state.qvel),
            ('object2_xy_position', object2_position[:2]),
            ('object2_angle', object2_angle.reshape(-1)),
            ('object2_orientation', object2_orientation),
            ('object2_orientation_cos', np.cos(object2_orientation)),
            ('object2_orientation_sin', np.sin(object2_orientation)),
            ('object2_z_orientation', object2_orientation[2:]),
            ('object2_z_orientation_cos', np.cos(object2_orientation[2:])),
            ('object2_z_orientation_sin', np.sin(object2_orientation[2:])),

            # === TARGET 1 ===
            ('target1_angle', target1_orientation[2].reshape(-1)),
            ('target1_orientation', target1_orientation),
            ('target1_position', target1_position),
            ('target1_xy_position', target1_position[:2]),
            ('target1_orientation_cos', np.cos(target1_orientation)),
            ('target1_orientation_sin', np.sin(target1_orientation)),

            # === TARGET 2 ===
            ('target2_angle', target2_orientation[2].reshape(-1)),
            ('target2_orientation', target2_orientation),
            ('target2_position', target2_position),
            ('target2_xy_position', target2_position[:2]),
            ('target2_orientation_cos', np.cos(target2_orientation)),
            ('target2_orientation_sin', np.sin(target2_orientation)),

            # === DISTANCE TO GOAL ===
            ('object1_to_target_relative_position', object1_to_target_relative_position),
            ('object1_to_target_relative_orientation', object1_to_target_relative_orientation),
            ('object1_to_target_position_distance', np.linalg.norm(object1_to_target_relative_position)),
            ('object1_to_target_circle_distances', object1_to_target_circle_distance),
            ('object1_to_target_circle_distance', np.linalg.norm(object1_to_target_circle_distance)),

            ('object2_to_target_relative_position', object2_to_target_relative_position),
            ('object2_to_target_relative_orientation', object2_to_target_relative_orientation),
            ('object2_to_target_position_distance', np.linalg.norm(object2_to_target_relative_position)),
            ('object2_to_target_circle_distances', object2_to_target_circle_distance),
            ('object2_to_target_circle_distance', np.linalg.norm(object2_to_target_circle_distance)),

            # ('in_corner', np.array([in_corner])),
            # ('goal_index', np.array([0])),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""

        claw_vel = obs_dict['claw_qvel']

        object1_to_target_relative_orientation = obs_dict['object1_to_target_relative_orientation']
        object1_to_target_position_distance = obs_dict['object1_to_target_position_distance']
        object1_to_target_circle_distance = obs_dict['object1_to_target_circle_distance']
        
        object2_to_target_relative_orientation = obs_dict['object2_to_target_relative_orientation']
        object2_to_target_position_distance = obs_dict['object2_to_target_position_distance']
        object2_to_target_circle_distance = obs_dict['object2_to_target_circle_distance']

        reward_dict = collections.OrderedDict((
            # Penalty for distance away from goal.
            ('object1_to_target_position_distance_log_reward',
             - np.log(20 * (object1_to_target_position_distance + 0.01))),
            ('object1_to_target_orientation_distance_log_reward',
             - np.log(1 * object1_to_target_circle_distance + 0.005)),

            ('object2_to_target_position_distance_log_reward',
             - np.log(20 * (object1_to_target_position_distance + 0.01))),
            ('object2_to_target_orientation_distance_log_reward',
             - np.log(1 * object1_to_target_circle_distance + 0.005)),

            # Penalty for difference with nomimal pose.
            ('pose_diff_cost',
             -1 * np.linalg.norm(obs_dict['claw_qpos'] - self._desired_claw_pos)),
            # Penality for high velocities.
            ('joint_vel_cost', -1 * np.linalg.norm(claw_vel[claw_vel >= 0.5])),

            # Reward for close proximity with goal.
            ('bonus_small', 10 * (object1_to_target_position_distance < 0.025
                and object2_to_target_position_distance < 0.025)),
            ('bonus_big', 50 * (object1_to_target_position_distance < 0.025
                and object2_to_target_position_distance < 0.025)),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""

        return collections.OrderedDict((
            # ('points', 1.0 - np.minimum(
            #     obs_dict['object_to_target_circle_distances'][:, 2], np.pi) / np.pi),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))

    def _set_target_object_qpos(self, target1_qpos: np.ndarray, target2_qpos: np.ndarray):
        """Sets the goal position and orientation."""
        # Modulo to [-pi, pi].
        self._object1_target_position = target1_qpos[:3]
        self._object1_target_orientation = np.mod(
            np.array(target1_qpos[3:]) + np.pi, 2 * np.pi) - np.pi

        self._object2_target_position = target2_qpos[:3]
        self._object2_target_orientation = np.mod(
            np.array(target2_qpos[3:]) + np.pi, 2 * np.pi) - np.pi

        # Mark the target position in sim.
        self.model.body_pos[self._target1_bid] = self._object1_target_position
        self.model.body_quat[self._target1_bid] = euler2quat(
            *self._object1_target_orientation)

        self.model.body_pos[self._target2_bid] = self._object2_target_position
        self.model.body_quat[self._target2_bid] = euler2quat(
            *self._object2_target_orientation)

@configurable(pickleable=True)
class DClawTranslateMultiPuckFixed(BaseDClawTranslateMultiObject):
    def __init__(
            self,
            *args,
            init_qpos_ranges: Sequence[np.ndarray]=(
                ((0.1, 0.1, 0, 0, 0, 0), (0.1, 0.1, 0, 0, 0, 0)),
                ((-0.1, -0.1, 0, 0, 0, 0), (-0.1, -0.1, 0, 0, 0, 0)),
            ),
            target_qpos_ranges: Sequence[np.ndarray]=(
                ((0.1, -0.1, 0, 0, 0, 0), (0.1, -0.1, 0, 0, 0, 0)),
                ((-0.1, 0.1, 0, 0, 0, 0), (-0.1, 0.1, 0, 0, 0, 0))
            ),
            **kwargs):
        super(DClawTranslateMultiPuckFixed, self).__init__(
            *args,
            asset_path='dsuite/dclaw/assets/dclaw3xh_puck2.xml',
            **kwargs
        )
        self._init_qpos_ranges = init_qpos_ranges
        self._target_qpos_ranges = target_qpos_ranges

    def _get_target_qpos(self, target_qpos_range):
        if isinstance(target_qpos_range, (list,)):
            if self._cycle_goals:
                if not self._let_alg_set_goals:
                    self._goal_index = (self._goal_index + 1) % self.num_goals
                target_qpos = target_qpos_range[self._goal_index]
            else:
                rand_index = np.random.randint(len(target_qpos_range))
                target_qpos = np.array(target_qpos_range[rand_index])
        elif isinstance(target_qpos_range, (tuple,)):
            target_qpos = np.random.uniform(
                low=target_qpos_range[0], high=target_qpos_range[1])    
        return target_qpos

    def _get_init_qpos(self, init_qpos_range):
        if isinstance(init_qpos_range, (list,)):
            if self._cycle_inits:
                init_index = self._init_index
                # TODO(justinvyu): One index for each object
                self._init_index = (self._init_index + 1) % len(init_qpos_range)
            else:
                init_index = np.random.randint(len(init_qpos_range))
            init_qpos = np.array(init_qpos_range[init_index])
        elif isinstance(init_qpos_range, (tuple,)):
            init_qpos = np.random.uniform(
                low=init_qpos_range[0], high=init_qpos_range[1])
        return init_qpos

    def _reset(self):
        init1_qpos_range, init2_qpos_range = self._init_qpos_ranges
        target1_qpos_range, target2_qpos_range = self._target_qpos_ranges

        self._initial_object1_qpos = self._get_init_qpos(init1_qpos_range)
        self._initial_object2_qpos = self._get_init_qpos(init2_qpos_range)

        self._set_target_object_qpos(
            target1_qpos=self._get_target_qpos(target1_qpos_range),
            target2_qpos=self._get_target_qpos(target2_qpos_range))

        super(DClawTranslateMultiPuckFixed, self)._reset()


@configurable(pickleable=True)
class DClawTranslatePuckFixed(DClawTurnFreeValve3Fixed):
    def __init__(
            self,
            target_qpos_range=(
                (-0.08, -0.08, 0, 0, 0, 0),
                (-0.08, -0.08, 0, 0, 0, 0),
            ),
            **kwargs):
        super().__init__(
            asset_path='dsuite/dclaw/assets/dclaw3xh_puck.xml',
            observation_keys=DEFAULT_PUCK_OBSERVATION_KEYS,
            target_qpos_range=target_qpos_range,
            **kwargs
        )


@configurable(pickleable=True)
class DClawTranslatePuckResetFree(DClawTurnFreeValve3ResetFree):
    def __init__(self, **kwargs):
        super().__init__(
            asset_path='dsuite/dclaw/assets/dclaw3xh_puck.xml',
            observation_keys=DEFAULT_PUCK_OBSERVATION_KEYS,
            **kwargs
        )


@configurable(pickleable=True)
class DClawTranslatePuckResetFreeSwapGoalEval(DClawTurnFreeValve3ResetFreeSwapGoalEval):
    def __init__(self, **kwargs):
        super().__init__(
            asset_path='dsuite/dclaw/assets/dclaw3xh_puck.xml',
            observation_keys=DEFAULT_PUCK_OBSERVATION_KEYS,
            **kwargs
        )
