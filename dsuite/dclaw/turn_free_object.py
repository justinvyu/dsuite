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
                                   BaseDClawEnv,
                                   DEFAULT_CLAW_RESET_POSE)
from dsuite.utils.configurable import configurable
from dsuite.utils.resources import get_asset_path
from dsuite.utils.circle_math import circle_distance
from dsuite.components.robot.config import ControlMode
from dsuite.components.robot import RobotState


# The observation keys that are concatenated as the environment observation.
DEFAULT_OBSERVATION_KEYS = (
    'claw_qpos',
    'object_position',
    'object_orientation_cos',
    'object_orientation_sin',
    'last_action',
    'target_orientation',
#    'target_orientation_cos',
#    'target_orientation_sin',
#    'object_to_target_relative_position',
#    'in_corner',
)

DEFAULT_HARDWARE_OBSERVATION_KEYS = (
    'claw_qpos',
    'last_action',
)


# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_free_valve3_in_arena.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw_valve3_in_less_tiny_box.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw_valve3_fixed_tiny_box.xml'
DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_valve3_free.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_free_cube.xml'


class BaseDClawTurnFreeObject(BaseDClawObjectEnv, metaclass=abc.ABCMeta):
    """Shared logic for DClaw turn tasks."""

    def __init__(self,
                 asset_path: str = DCLAW3_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 device_path: Optional[str] = None,
                 camera_config: dict = None,
                 frame_skip: int = 40,
                 free_claw: bool = False,
                 position_reward_weight: int = 1,
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            device_path: The device path to Dynamixel hardware.
            frame_skip: The number of simulation steps per environment step.
        """
        self._position_reward_weight = position_reward_weight
        self._camera_config = camera_config
        super().__init__(
            sim_model=get_asset_path(asset_path),
            robot_config=self.get_config_for_device(
                device_path, free_object=True, free_claw=free_claw),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)

        self._desired_claw_pos = DEFAULT_CLAW_RESET_POSE.copy()
        self._last_action = np.zeros(9)

        self._target_bid = self.model.body_name2id('target')

        # The following are modified (possibly every reset) by subclasses.
        self._initial_claw_qpos = DEFAULT_CLAW_RESET_POSE.copy()
        self._initial_object_qpos = (0, 0, 0, 0, 0, 0)
        self._initial_object_qvel = (0, 0, 0, 0, 0, 0)
        self._set_target_object_qpos((0, 0, 0, 0, 0, 0))

    def _reset(self):
        """Resets the environment."""
        self._reset_dclaw_and_object(
            claw_pos=self._initial_claw_qpos,
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

        object_position = object_state.qpos[:3].copy()
        object_orientation = object_state.qpos[3:].copy()

        target_position = self._object_target_position
        target_orientation = self._object_target_orientation

        object_to_target_relative_position = object_position - target_position
        object_to_target_relative_orientation = object_orientation - target_orientation
        object_to_target_circle_distance = circle_distance(
            object_orientation, target_orientation)

        CORNER_THRESHOLD = 0.05 #0.07
        in_corner = 0
        if np.all(object_position[:2] > CORNER_THRESHOLD):
            in_corner = 1
        elif object_position[0] < -CORNER_THRESHOLD and object_position[1] > CORNER_THRESHOLD:
            in_corner = 2
        elif object_position[0] < -CORNER_THRESHOLD and object_position[1] < -CORNER_THRESHOLD:
            in_corner = 3
        elif object_position[0] > CORNER_THRESHOLD and object_position[1] < - CORNER_THRESHOLD:
            in_corner = 4
        return collections.OrderedDict((
            ('claw_qpos', claw_state.qpos.copy()),
            ('claw_qvel', claw_state.qvel.copy()),
            ('object_position', object_position),
            ('object_orientation', object_orientation),
            ('object_orientation_cos', np.cos(object_orientation)),
            ('object_orientation_sin', np.sin(object_orientation)),
            ('object_qvel', object_state.qvel),
            ('last_action', self._last_action),
            ('target_angle', target_orientation[2]),
            ('target_position', target_position),
            ('target_orientation', target_orientation),
            ('target_orientation_cos', np.cos(target_orientation)),
            ('target_orientation_sin', np.sin(target_orientation)),
            ('object_to_target_relative_position', object_to_target_relative_position),
            ('object_to_target_relative_orientation', object_to_target_relative_orientation),
            ('object_to_target_position_distance', np.linalg.norm(object_to_target_relative_position)),
            ('object_to_target_circle_distances', object_to_target_circle_distance),
            ('object_to_target_circle_distance', np.linalg.norm(object_to_target_circle_distance)),
            ('in_corner', np.array([in_corner])),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        # object_to_target_relative_orientation = obs_dict['object_to_target_relative_orientation']

        claw_vel = obs_dict['claw_qvel']

        object_to_target_position_distance = obs_dict['object_to_target_position_distance']
        object_to_target_circle_distance = obs_dict['object_to_target_circle_distance']

        reward_dict = collections.OrderedDict((
            # Penalty for distance away from goal.
            # ('object_to_target_position_distance_cost', -5 *
            #     object_to_target_position_distance),
            # ('object_to_target_orientation_distance_cost', -5 *
            #     object_to_target_circle_distance),
            ('object_to_target_position_distance_cost', - self._position_reward_weight * \
             np.log(20 * object_to_target_position_distance + 0.005)),
            ('object_to_target_orientation_distance_cost',  - 1 * \
             np.log(1 * object_to_target_circle_distance + 0.005)),

            # Penalty for difference with nomimal pose.
            ('pose_diff_cost',
             -1 * np.linalg.norm(obs_dict['claw_qpos'] - self._desired_claw_pos)
            ),
            # Penality for high velocities.
            ('joint_vel_cost', -1 * np.linalg.norm(claw_vel[claw_vel >= 0.5])),

            # Reward for close proximity with goal.
            ('bonus_small', 10 * (object_to_target_circle_distance < 0.25)),
            ('bonus_big', 50 * (object_to_target_circle_distance < 0.10)),
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
                obs_dict['object_to_target_circle_distances'][:, 2], np.pi) / np.pi),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))

    def _set_target_object_qpos(self, target_qpos: float):
        """Sets the goal position and orientation."""
        # Modulo to [-pi, pi].
        self._object_target_position = target_qpos[:3]
        self._object_target_orientation = np.mod(
            np.array(target_qpos[3:]) + np.pi, 2 * np.pi) - np.pi

        # Mark the target position in sim.

        self.model.body_pos[self._target_bid] = self._object_target_position
        self.model.body_quat[self._target_bid] = euler2quat(
            *self._object_target_orientation)

    def replay_pool_pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    def _get_init_pool(self, initial_distribution_path):
        experiment_root = os.path.dirname(initial_distribution_path)

        experience_paths = [
            self.replay_pool_pickle_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                    os.path.join(experiment_root, 'checkpoint_*')))
        ]

        init_claw_qpos, init_object_qpos = [], []
        for experience_path in experience_paths:
            with gzip.open(experience_path, 'rb') as f:
                pool = pickle.load(f)

            init_claw_qpos.append(pool['observations']['claw_qpos'])
            object_orientation = np.arctan2(
                pool['observations']['object_orientation_sin'],
                pool['observations']['object_orientation_cos']
            )
            object_qpos = np.concatenate([
                pool['observations']['object_position'],
                object_orientation], axis=1
            )
            init_object_qpos.append(object_qpos)

        return np.concatenate(init_claw_qpos), np.concatenate(init_object_qpos)

    def render(self, *args, **kwargs):
        if self._camera_config is not None:
            return self._image_service.get_image(*args, **kwargs)

        return super().render(*args, **kwargs)


@configurable(pickleable=True)
class DClawTurnFreeValve3Hardware(BaseDClawEnv):
    def __init__(self,
                 camera_config: dict = None,
                 device_path: str = None,
                 observation_keys: Sequence[str] = DEFAULT_HARDWARE_OBSERVATION_KEYS,
                 frame_skip: int = 40,
                 **kwargs):
        super().__init__(
           sim_model=get_asset_path('dsuite-scenes/dclaw/dclaw3xh.xml'),
           robot_config=self.get_config_for_device(device_path),
           frame_skip=frame_skip,
           **kwargs)
        self._camera_config = camera_config
        if camera_config:
            from dsuite.dclaw.turn import get_image_service
            self._image_service = get_image_service(**camera_config)
        self._last_action = np.zeros(self.action_space.shape[0])

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        state = self.robot.get_state('dclaw')
        return collections.OrderedDict((
            ('claw_qpos', state.qpos),
            ('claw_qvel', state.qvel),
            ('last_action', self._last_action),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        qvel = obs_dict['claw_qvel']
        reward_dict = collections.OrderedDict({
            'joint_vel_cost': -0.1 * np.linalg.norm(qvel[np.abs(qvel) >= 4.5])
        })
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reawrd_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        return collections.OrderedDict()

    def _step(self, action: np.ndarray):
        self.robot.step({'dclaw': action})

    def _reset(self):
        self.robot.set_state({
            'dclaw': RobotState(qpos=DEFAULT_CLAW_RESET_POSE,
                                qvel=np.zeros(self.action_space.shape[0]))
        })

    def render(self, *args, **kwargs):
        if self._camera_config is not None:
            return self._image_service.get_image(*args, **kwargs)

        return super().render(*args, **kwargs)


@configurable(pickleable=True)
class DClawTurnFreeValve3Fixed(BaseDClawTurnFreeObject):
    """Turns the object with a fixed initial and fixed target position."""

    def __init__(self,
                 init_angle_range=(0, 0),
                 target_angle_range=(np.pi, np.pi),
                 init_x_pos_range=(0, 0),
                 init_y_pos_range=(0, 0),
                 *args, **kwargs):
        self._init_angle_range = init_angle_range
        self._target_angle_range = target_angle_range
        self._init_x_pos_range = init_x_pos_range
        self._init_y_pos_range = init_y_pos_range
        super().__init__(*args, **kwargs)

    def _sample_goal(self, obs_dict):
        if isinstance(self._target_angle_range, (list,)):
            target_angle = np.random.choice(self._target_angle_range)
        elif isinstance(self._target_angle_range, (tuple,)):
            target_angle = np.random.uniform(
                low=self._target_angle_range[0],
                high=self._target_angle_range[1]
            )
        return (0, 0, 0, 0, 0, target_angle)

    def _reset(self):
        lows, highs = list(zip(self._init_angle_range,
                               self._init_x_pos_range,
                               self._init_y_pos_range))
        init_angle, x_pos, y_pos = np.random.uniform(
            low=lows, high=highs
        )
        self._initial_object_qpos = (x_pos, y_pos, 0, 0, 0, init_angle)
        self._set_target_object_qpos(
            self._sample_goal(self.get_obs_dict()))
        super()._reset()


@configurable(pickleable=True)
class DClawTurnFreeValve3RandomReset(BaseDClawTurnFreeObject):
    """Turns the object with a random initial and fixed target position."""

    def __init__(self,
                 reset_from_corners=False,
                 initial_distribution_path='', #'/mnt/sda/ray_results/gym/DClaw/TurnFreeValve3ResetFree-v0/2019-06-30T18-53-06-baseline_both_push_and_turn_log_rew/id=38872574-seed=6880_2019-06-30_18-53-07whkq1aax/',#"",
                 **kwargs):
        self._reset_from_corners = reset_from_corners
        self._init_claw_qpos_dist = self._init_object_qpos_dist = None
        if initial_distribution_path:
            self._init_claw_qpos_dist, self._init_object_qpos_dist = self._get_init_pool(
                initial_distribution_path)
            self._num_init_states = self._init_claw_qpos_dist.shape[0]
        super().__init__(**kwargs)

    def _reset(self):
        # Turn from 0 degrees to 180 degrees.
        if self._reset_from_corners:
            x_ind = np.random.randint(2)
            y_ind = np.random.randint(2)
            limits = (-0.085, 0.085)
            self._initial_object_qpos = (
                limits[x_ind], limits[y_ind], 0,
                0, 0, np.random.uniform(-np.pi, np.pi)
            )
        elif self._init_claw_qpos_dist is not None:
            rand_index = np.random.randint(self._num_init_states)
            self._initial_claw_qpos = self._init_claw_qpos_dist[rand_index]
            self._initial_object_qpos = self._init_object_qpos_dist[rand_index]
        else:
            self._initial_object_qpos = np.random.uniform(
                low=(-0.07, -0.07, 0, 0, 0, -np.pi),
                high=(0.07, 0.07, 0, 0, 0, np.pi)
            )
            # self._initial_object_qpos = np.random.uniform(
            #     low=(-0.05, -0.05, 0, 0, 0, -np.pi),
            #     high=(0.05, 0.05, 0, 0, 0, np.pi)
            # )
        # self._set_target_object_qpos((-0.06, -0.08, 0, 0, 0, 0))
        self._set_target_object_qpos((0, 0, 0, 0, 0, np.pi))
        super()._reset()

    def reset(self):
        self.sim.reset()
        self.sim.forward()
        self._reset()
        if self._reset_from_corners:
            corner_index = np.random.randint(2, size=2) * 2 - 1 # -1 or 1
            self.data.qpos[-6:-4] = np.array([0.05, 0.05]) * corner_index

            open_claw_position = np.tile([0, -1.5, -1.5], 3)
            for _ in range(5):
                self.data.ctrl[:9] = open_claw_position
                self.data.qfrc_applied[-6:-4] = np.array([1, 1]) * corner_index
                self.sim_scene.advance()
            self.data.qfrc_applied[-6:-4] = 0
            self.data.qpos[:9] = DEFAULT_CLAW_RESET_POSE.copy()

        # self._set_target_object_qpos(self._sample_goal(obs_dict))
        return self._get_obs(self.get_obs_dict())

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        # object_to_target_relative_orientation = obs_dict['object_to_target_relative_orientation']
        reward_dict = super().get_reward_dict(action, obs_dict)

        object_to_target_position_distance = obs_dict['object_to_target_position_distance']

        object_to_target_circle_distance = obs_dict['object_to_target_circle_distance']

        reward_dict['object_to_target_position_distance_cost'] = - 20 * object_to_target_position_distance
        reward_dict['object_to_target_orientation_distance_cost'] = - 1 * object_to_target_circle_distance
        return reward_dict


@configurable(pickleable=True)
class DClawTurnFreeValve3ResetFree(BaseDClawTurnFreeObject):
    """Turns the object reset-free with a fixed initial and varied target positions."""

    def __init__(self,
                 swap_goal_upon_completion: bool = True,
                 reset_fingers=True,
                 position_reward_weight=1,
                 path_length: int = 50,
                 **kwargs):
        self._last_claw_qpos = DEFAULT_CLAW_RESET_POSE.copy()
        self._last_object_position = np.array([0, 0, 0])
        self._last_object_orientation = np.array([0, 0, 0])
        self._reset_fingers = reset_fingers

        super().__init__(**kwargs)
        self._swap_goal_upon_completion = swap_goal_upon_completion
        # self._goals = [(-0.06, -0.08, 0, 0, 0, 0), (-0.06, -0.08, 0, 0, 0, 0)]
        self._goals = ((0, 0, 0, 0, 0, np.pi), (0, 0, 0, 0, 0, np.pi))
        self._goal_index = 1

        self._position_reward_weight = position_reward_weight
        self._path_length = path_length
        self._step_count = 0

    def _step(self, action):
        super()._step(action)
        self._step_count += 1

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        obs_dict = super().get_obs_dict()
        obs_dict['step_count'] = self._step_count
        return obs_dict

    def _sample_goal(self, obs_dict):
        object_to_target_position_distance = obs_dict['object_to_target_position_distance']
        object_to_target_orientation_distance = obs_dict['object_to_target_circle_distance']
        if self._swap_goal_upon_completion and \
           object_to_target_orientation_distance < 0.1 and \
           object_to_target_position_distance < 0.01:
            self._goal_index = np.mod(self._goal_index + 1, 2)
        else:
            goal = (0, 0, 0, 0, 0, np.pi)

        goal = self._goals[self._goal_index]
        return goal

    def reset(self):
        obs_dict = self.get_obs_dict()
        dclaw_config = self.robot.get_config('dclaw')
        dclaw_control_mode = dclaw_config.control_mode
        dclaw_config.set_control_mode(ControlMode.JOINT_POSITION)
        if self._reset_fingers:
            reset_action = self.robot.normalize_action(
                {'dclaw': DEFAULT_CLAW_RESET_POSE.copy()})['dclaw']

            for _ in range(15):
                self._step(reset_action)
        self._set_target_object_qpos(self._sample_goal(obs_dict))
        dclaw_config.set_control_mode(dclaw_control_mode)

        self._step_count = 0
        return self._get_obs(self.get_obs_dict())


@configurable(pickleable=True)
class DClawTurnFreeValve3ResetFreeSwapGoal(DClawTurnFreeValve3ResetFree):
    """Turns the object reset-free with a target position swapped every reset."""
    def __init__(self,
                 #observation_keys=DEFAULT_OBSERVATION_KEYS,
                 **kwargs):
        super().__init__(
            #observation_keys=observation_keys + ('other_reward',),
            **kwargs)
        self._goal_index = 0
        # self._goals = [(-0.06, -0.08, 0, 0, 0, 0), (-0.06, -0.08, 0, 0, 0, 0)]
        self._goals = [
            (0.05, 0.05, 0, 0, 0, np.pi/2),
            (-0.05, -0.05, 0, 0, 0, -np.pi/2)]
            # (0.05, -0.05, 0, 0, 0, -np.pi/2),
            # (-0.05, 0.05, 0, 0, 0, np.pi/2)]
        self._goals = [
            (0.0, 0.0, 0, 0, 0, np.pi/2),
            (-0.0, -0.0, 0, 0, 0, -np.pi/2)]
            # (0.05, -0.05, 0, 0, 0, -np.pi/2),
            # (-0.05, 0.05, 0, 0, 0, np.pi/2)]
        self.n_goals = len(self._goals)

    def get_obs_dict(self):
        obs_dict = super().get_obs_dict()

        self._set_target_object_qpos(self._sample_goal(None))
        swapped_goal_obs_dict = super().get_obs_dict()
        self._set_target_object_qpos(self._sample_goal(None))

        obs_dict['other_reward'] = [self._get_total_reward(
            self.get_reward_dict(None, swapped_goal_obs_dict))]
        return obs_dict

    def relabel_path(self, path):
        observations = path['observations']
        goal_angle = np.arctan2(
            observations['target_orientation_sin'][0][2],
            observations['target_orientation_cos'][0][2]
        )

        if goal_angle == self._goals[0][5]:
            path['observations']['target_orientation_sin'][:, 2] = np.sin(self._goals[1][5])
            path['observations']['target_orientation_cos'][:, 2] = np.cos(self._goals[1][5])
        elif goal_angle == self._goals[1][5]:
            path['observations']['target_orientation_sin'][:, 2] = np.sin(self._goals[0][5])
            path['observations']['target_orientation_cos'][:, 2] = np.cos(self._goals[0][5])
        path['rewards'] = path['observations']['other_reward']
        return path

    def _sample_goal(self, obs_dict):
        self._goal_index = (self._goal_index + 1) % self.n_goals
        return self._goals[self._goal_index]

    # def get_done(self, obs_dict, rew_dict):
    #     dones = obs_dict['step_count'] == self._path_length
    #     return dones


@configurable(pickleable=True)
class DClawTurnFreeValve3ResetFreeSwapGoalEval(DClawTurnFreeValve3Fixed):
    """Turns the object reset-free with a target position swapped every reset."""
    def __init__(self,
                 #observation_keys=DEFAULT_OBSERVATION_KEYS,
                 **kwargs):
        super().__init__(
            #observation_keys=observation_keys + ('other_reward',),
            **kwargs)
        self._goal_index = 0
        self._goals = [
            (0.0, 0.0, 0, 0, 0, np.pi/2),
            (-0.0, -0.0, 0, 0, 0, -np.pi/2)]
            # (0.05, -0.05, 0, 0, 0, -np.pi/2),
            # (-0.05, 0.05, 0, 0, 0, np.pi/2)]
        self.n_goals = len(self._goals)

    def _sample_goal(self, obs_dict):
        self._goal_index = (self._goal_index + 1) % self.n_goals
        return self._goals[self._goal_index]

    def _reset(self):
        self._initial_object_qpos = self._goals[(self._goal_index + 1) % 2]
        super()._reset()


@configurable(pickleable=True)
class DClawTurnFreeValve3ResetFreeRandomGoal(DClawTurnFreeValve3ResetFree):
    """Turns the object reset-free with a target position swapped every reset."""
    def __init__(self,
                 goal_range=((-0.08, -0.08, 0, 0, 0, -np.pi), (0.08, 0.08, 0, 0, 0, np.pi)),
                 **kwargs):
        self._goal_range = goal_range
        super().__init__(
            **kwargs)

    def _sample_goal(self, obs_dict):
        return np.random.uniform(low=self._goal_range[0], high=self._goal_range[1])


@configurable(pickleable=True)
class DClawTurnFreeValve3FixedResetSwapGoal(DClawTurnFreeValve3Fixed):
    """Turns the object reset-free with a fixed initial and varied target positions."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_angle_range = (-np.pi, np.pi)
        self._init_x_pos_range = [-0.05, 0.05]
        self._init_y_pos_range = [-0.05, 0.05]
        self._goals = [
            (0.05, 0.05, 0, 0, 0, np.pi/2),
            (-0.05, -0.05, 0, 0, 0, -np.pi/2)]
            # (0.05, -0.05, 0, 0, 0, -np.pi/2),
            # (-0.05, 0.05, 0, 0, 0, np.pi/2)]
        self._init_x_pos_range = [-0.03, 0.03]
        self._init_y_pos_range = [-0.03, 0.03]

        self._goals = [
            (0.0, 0.0, 0, 0, 0, np.pi/2),
            (-0.0, -0.0, 0, 0, 0, -np.pi/2)]
            # (0.05, -0.05, 0, 0, 0, -np.pi/2),
            # (-0.05, 0.05, 0, 0, 0, np.pi/2)]

        self.n_goals = len(self._goals)
        self._goal_index = 0

    def _sample_goal(self, obs_dict):
        self._goal_index = (self._goal_index + 1) % self.n_goals
        return self._goals[self._goal_index]


@configurable(pickleable=True)
class DClawTurnFreeValve3ResetFreeCurriculum(DClawTurnFreeValve3ResetFree):
    """Turns the object reset-free with a target position swapped every reset."""
    def __init__(self,
                 **kwargs):
        super().__init__(
            **kwargs)
        self._goal_index = 1
        self._goals = [(-0.06, -0.08, 0, 0, 0, 0), (0, 0, 0, 0, 0, np.pi)]
        self.n_goals = len(self._goals)
        self._switch_to_reset_controller_threshold = 0.01
        self._switch_to_forward_controller_threshold = 0.02
        self._first_step = True
        self._reset_counter = 0

    def get_obs_dict(self):
        # current_goal = self._goals[self._goal_index]
        obs_dict = super().get_obs_dict()
        obs_dict['switch_to_reset_controller_threshold'] = self._switch_to_reset_controller_threshold
        obs_dict['switch_to_forward_controller_threshold'] = self._switch_to_forward_controller_threshold
        return obs_dict

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        # object_to_target_relative_orientation = obs_dict['object_to_target_relative_orientation']
        reward_dict = super().get_reward_dict(action, obs_dict)

        object_to_target_position_distance = obs_dict['object_to_target_position_distance']

        object_to_target_circle_distance = obs_dict['object_to_target_circle_distance']

        # reward_dict['object_to_target_position_distance_cost'] = -0 * \
        #     object_to_target_position_distance
        # reward_dict['object_to_target_orientation_distance_cost'] = -1 * \
        #     object_to_target_circle_distance

        reward_dict['object_to_target_position_distance_cost'] = - 1 * \
            20 * object_to_target_position_distance
        reward_dict['object_to_target_orientation_distance_cost'] = - 1 * \
            object_to_target_circle_distance

        if self._goal_index == 0:
            reward_dict['object_to_target_orientation_distance_cost'] *= 0

        # swap goals according to curriculum
        dist_to_corner = np.linalg.norm(
            obs_dict['object_position'] - np.array(self._goals[0][:3])
        )

        if self._goal_index == 1 and \
           dist_to_corner > self._switch_to_reset_controller_threshold:
            # ran forward controller past threshold
            self._goal_index = 0
            self._set_target_object_qpos(self._goals[self._goal_index])
            print('swap to reset')
        if self._goal_index == 0 and dist_to_corner < self._switch_to_forward_controller_threshold:
            # ran reset controller for long enough
            self._goal_index = 1
            self._set_target_object_qpos(self._goals[self._goal_index])
            print('swap to forward: ', self._switch_to_reset_controller_threshold)
            self._switch_to_reset_controller_threshold += 0.0002

        return reward_dict

    def _sample_goal(self, obs_dict):
        return self._goals[self._goal_index]

    def reset(self):
        if self._first_step:
            self.data.qpos[-6:] = (-0.06, -0.08, 0, 0, 0, 0)
            self._first_step = False
        return super().reset()


@configurable(pickleable=True)
class DClawTurnFreeValve3ResetFreeCurriculumEval(DClawTurnFreeValve3Fixed):
    def _sample_goal(self, obs_dict):
        """Sets the goal position and orientation."""
        # Modulo to [-pi, pi].
        return (0, 0, 0, 0, 0, np.pi)

    def _reset(self):
        self._initial_object_qpos = (-0.06, -0.08, 0, 0, 0, 0)
        self._set_target_object_qpos(
            self._sample_goal(self.get_obs_dict()))
        self._reset_dclaw_and_object(
            claw_pos=self._initial_claw_qpos,
            object_pos=np.atleast_1d(self._initial_object_qpos),
            object_vel=np.atleast_1d(self._initial_object_qvel),
            # guide_pos=np.atleast_1d(self._object_target_qpos))
        )



        self._set_target_object_qpos(self._sample_goal_qpos(obs_dict))
        return self._get_obs(self.get_obs_dict())

@configurable(pickleable=True)
class DClawTurnFreeValve3Image(DClawTurnFreeValve3Fixed):
    """
    Observation including the image.
    """

    def __init__(self,
                image_shape: np.ndarray,
                *args, **kwargs):
        self._image_shape = image_shape
        super(DClawTurnFreeValve3Image, self).__init__(*args, **kwargs)

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        width, height = self._image_shape[:2]
        obs = super().get_obs_dict()
        image = self.render(mode='rgb_array', \
                            width=width,
                            height=height,
                            camera_id=-1).reshape(-1)
        obs['image'] = ((2.0 / 255.0) * image - 1.0) # Normalize between [-1, 1]
        return obs


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
