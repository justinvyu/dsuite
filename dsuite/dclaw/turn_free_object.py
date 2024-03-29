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

INTERMEDIATE_CLAW_RESET_POSE_0 = np.array([np.pi / 3, -np.pi / 3, np.pi / 2] * 3)
INTERMEDIATE_CLAW_RESET_POSE_1 = np.array([np.pi / 3, np.pi / 5, np.pi / 2] * 3)
INTERMEDIATE_CLAW_RESET_POSE_2 = np.array([np.pi / 4, -np.pi / 5, np.pi / 2] * 3)

# The observation keys that are concatenated as the environment observation.
DEFAULT_OBSERVATION_KEYS = (
    'claw_qpos',
    'object_xy_position',
    'object_z_orientation_cos',
    'object_z_orientation_sin',
    'last_action',
    'target_xy_position',
    'target_z_orientation_cos',
    'target_z_orientation_sin',
)

DEFAULT_HARDWARE_OBSERVATION_KEYS = (
    'claw_qpos',
    'last_action',
    'goal_index',
    'target_xy_position',
    'target_z_orientation_cos',
    'target_z_orientation_sin',
)


# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_free_valve3_in_arena.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw_valve3_in_tiny_box.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw_valve3_in_less_tiny_box.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw_valve3_fixed_tiny_box.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_valve3_free.xml'
DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw_valve3_in_larger_box.xml'

ARENA_PATHS = {
    'box_tiny': 'dsuite/dclaw/assets/dclaw_valve3_in_tiny_box.xml',
    'box_less_tiny': 'dsuite/dclaw/assets/dclaw_valve3_in_less_tiny_box.xml',
    'box_regular': 'dsuite/dclaw/assets/dclaw3xh_valve3_free.xml',      # 15
    'box_large': 'dsuite/dclaw/assets/dclaw_valve3_in_larger_box.xml',  # 17.5
    'bowl': 'dsuite/dclaw/assets/dclaw3xh_valve4_bowl.xml',
}

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
        self._position_reward_weight = position_reward_weight
        self._camera_config = camera_config
        self._use_bowl_arena = use_bowl_arena

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
        object_angle = np.mod(
            np.array(object_orientation[2]) + np.pi, 2 * np.pi) - np.pi

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
            ('object_xy_position', object_position[:2]),
            ('object_orientation', object_orientation),
            ('object_orientation_cos', np.cos(object_orientation)),
            ('object_orientation_sin', np.sin(object_orientation)),
            ('object_z_orientation', object_orientation[2:]),
            ('object_z_orientation_cos', np.cos(object_orientation[2:])),
            ('object_z_orientation_sin', np.sin(object_orientation[2:])),
            ('object_angle', object_angle.reshape(-1)),
            ('object_qvel', object_state.qvel),
            ('last_action', self._last_action),
            ('target_angle', target_orientation[2].reshape(-1)),
            ('target_orientation', target_orientation),
            ('target_position', target_position),
            ('target_xy_position',
                np.repeat(target_position[:2], 5)
            ),
            ('target_orientation_cos', np.cos(target_orientation)),
            ('target_orientation_sin', np.sin(target_orientation)),
            ('target_z_orientation_cos',
                np.repeat(np.cos(target_orientation[2]), 5)
            ),
            ('target_z_orientation_sin',
                np.repeat(np.sin(target_orientation[2]), 5)
            ),
            ('object_to_target_relative_position', object_to_target_relative_position),
            ('object_to_target_relative_orientation', object_to_target_relative_orientation),
            ('object_to_target_position_distance', np.linalg.norm(object_to_target_relative_position)),
            ('object_to_target_circle_distances', object_to_target_circle_distance),
            ('object_to_target_circle_distance', np.linalg.norm(object_to_target_circle_distance)),
            ('in_corner', np.array([in_corner])),
            ('goal_index', np.array([0])),
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
            # ('object_to_target_position_distance_cost', self._position_reward_weight *
            #     -20 * object_to_target_position_distance),
            # ('object_to_target_orientation_distance_cost',
            #     -1 * object_to_target_circle_distance),
            ('object_to_target_position_distance_reward',
             - np.log(20 * (object_to_target_position_distance + 0.01))),
            ('object_to_target_orientation_distance_reward',
             - np.log(1 * object_to_target_circle_distance + 0.005)),

            # Penalty for difference with nomimal pose.
            ('pose_diff_cost',
             -1 * np.linalg.norm(obs_dict['claw_qpos'] - self._desired_claw_pos)),
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
                 num_goals: int = 2,
                 # goals: np.ndarray = ((0, 0, 0, 0, 0, np.pi)),
                 **kwargs):
        # if num_goals > 1:
        #     observation_keys = observation_keys + ('goal_index', )

        super().__init__(
           sim_model=get_asset_path('dsuite-scenes/dclaw/dclaw3xh.xml'),
           robot_config=self.get_config_for_device(device_path),
           frame_skip=frame_skip,
           observation_keys=observation_keys,
           **kwargs)
        self._camera_config = camera_config
        if camera_config:
            from dsuite.dclaw.turn import get_image_service
            self._image_service = get_image_service(**camera_config)
        self._last_action = np.zeros(self.action_space.shape[0])
        self._num_goals = num_goals
        # self._goal_index = 0
        # Goals need to be specified with 9-dim vector (same shape as qpos)
        # self._goals = goals
        # assert num_goals == len(goals)

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
        self._last_action = action

    def _reset(self):
        self.robot.set_state({
            'dclaw': RobotState(qpos=INTERMEDIATE_CLAW_RESET_POSE_0,
                                qvel=np.zeros(self.action_space.shape[0]))
        })
        self.robot.set_state({
            'dclaw': RobotState(qpos=INTERMEDIATE_CLAW_RESET_POSE_1,
                                qvel=np.zeros(self.action_space.shape[0]))
        })
        self.robot.set_state({
            'dclaw': RobotState(qpos=INTERMEDIATE_CLAW_RESET_POSE_0,
                                qvel=np.zeros(self.action_space.shape[0]))
        })
        DEFAULT_CLAW_RESET_POSE[[1, 4, 7]] = -np.pi/2
        self.robot.set_state({
            'dclaw': RobotState(qpos=DEFAULT_CLAW_RESET_POSE,
                                qvel=np.zeros(self.action_space.shape[0]))
        })

        self._last_action = np.zeros(self.action_space.shape[0])
        # Set the new goal every episode
        # self._goal_index = self._sample_goal()

    def _sample_goal(self):
        if self._num_goals >= 2:
            other_indices = [index for index in range(self._num_goals)
                if index != self._goal_index]
            sampled_goal = np.random.choice(other_indices)
        else:
            sampled_goal = self._goal_index
        return sampled_goal

    def render(self, *args, **kwargs):
        if self._camera_config is not None:
            image = self._image_service.get_image(*args, **kwargs)
            return image

        return super().render(*args, **kwargs)

    @property
    def num_goals(self):
        return self._num_goals

    def set_goal(self, goal_index):
        """Allow outside algorithms to alter goals."""
        self._goal_index = goal_index
        self._cycle_goals = True
        self._let_alg_set_goals = True


@configurable(pickleable=True)
class DClawTurnFreeValve3Fixed(BaseDClawTurnFreeObject):
    """Turns the object with a fixed initial and fixed target position."""

    def __init__(self,
                 target_qpos_range=((0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)),
                 init_qpos_range=((0, 0, 0, 0, 0, -np.pi/2), (0, 0, 0, 0, 0, -np.pi/2)),
                 reset_policy_checkpoint_path='',
                 cycle_inits=False,
                 cycle_goals=False,
                 *args,
                 **kwargs):
        self._init_qpos_range = init_qpos_range
        self._target_qpos_range = target_qpos_range
        super().__init__(*args, **kwargs)
        self._policy = None
        self._cycle_goals = cycle_goals
        self._cycle_inits = cycle_inits
        self._goal_index = 0
        self._init_index = 0
        if reset_policy_checkpoint_path:
            self._load_policy(reset_policy_checkpoint_path)

    def _load_policy(self, checkpoint_path):
        import pickle
        from softlearning.policies.utils import get_policy_from_variant
        checkpoint_path = checkpoint_path.rstrip('/')
        experiment_path = os.path.dirname(checkpoint_path)

        variant_path = os.path.join(experiment_path, 'params.pkl')
        with open(variant_path, 'rb') as f:
            variant = pickle.load(f)

        policy_weights_path = os.path.join(checkpoint_path, 'policy_params.pkl')
        with open(policy_weights_path, 'rb') as f:
            policy_weights = pickle.load(f)

        from softlearning.environments.adapters.gym_adapter import GymAdapter
        from softlearning.environments.gym.wrappers import (
            NormalizeActionWrapper)

        env = GymAdapter(None, None, env=NormalizeActionWrapper(self))

        self._policy = (
            get_policy_from_variant(variant, env))
        self._policy.set_weights(policy_weights)
        self._reset_horizon = variant['sampler_params']['kwargs']['max_path_length']

        self._reset_target_qpos_range = variant['environment_params']['training']['kwargs']['target_qpos_range']

    @property
    def num_goals(self):
        if isinstance(self._target_qpos_range, (list,)):
            return len(self._target_qpos_range)
        else:
            raise Exception("infinite goals")

    def set_goal(self, goal_index):
        """Allow outside algorithms to alter goals."""
        self._goal_index = goal_index
        self._cycle_goals = True
        self._let_alg_set_goals = True

    def _sample_goal(self, obs_dict):
        if isinstance(self._target_qpos_range, (list,)):
            if self._cycle_goals:
                if not self._let_alg_set_goals:
                    self._goal_index = (self._goal_index + 1) % self.num_goals
                target_qpos = self._target_qpos_range[self._goal_index]
            else:
                rand_index = np.random.randint(len(self._target_qpos_range))
                target_qpos = np.array(self._target_qpos_range[rand_index])
        elif isinstance(self._target_qpos_range, (tuple,)):
            target_qpos = np.random.uniform(
                low=self._target_qpos_range[0],
                high=self._target_qpos_range[1]
            )
        return target_qpos

    def _reset(self):
        if isinstance(self._init_qpos_range, (list,)):
            if self._cycle_inits:
                init_index = self._init_index
                self._init_index = (self._init_index + 1) % len(self._init_qpos_range)
            else:
                init_index = np.random.randint(len(self._init_qpos_range))
            self._initial_object_qpos = np.array(self._init_qpos_range[init_index])
        elif isinstance(self._init_qpos_range, (tuple,)):
            self._initial_object_qpos = np.random.uniform(
                low=self._init_qpos_range[0], high=self._init_qpos_range[1]
            )
        self._set_target_object_qpos(
            self._sample_goal(self.get_obs_dict()))
        super()._reset()

    def get_policy_input(self):
        from softlearning.models.utils import flatten_input_structure
        obs_dict = self.get_obs_dict()
        observation = flatten_input_structure({
            key: obs_dict[key][None, ...]
            for key in self._policy.observation_keys
        })
        return observation


@configurable(pickleable=True)
class DClawTurnFreeValve3ResetFree(DClawTurnFreeValve3Fixed):
    """Turns the object reset-free with a fixed initial and varied target positions."""

    def __init__(self,
                 swap_goal_upon_completion: bool = True,
                 reset_fingers=True,
                 path_length: int = 50,
                 reset_frequency: int = 0,
                 target_qpos_range=[
                     (0.04, -0.04, 0, 0, 0, 0),
                     (-0.04, 0.04, 0, 0, 0, 0),
                     (0, 0, 0, 0, 0, 0),
                     (-0.04, -0.04, 0, 0, 0, 0),
                     (0.04, 0.04, 0, 0, 0, 0)
                 ],
                 # target_qpos_range=(
                 #     (-0.04, -0.04, 0, 0, 0, 0),
                 #     (0.04, 0.04, 0, 0, 0, 0)
                 # ),
                 init_qpos_range=[(0, 0, 0, 0, 0, 0)],
                 swap_goal_frequency=1,
                 take_random_actions_for=0,
                 reset_policy_checkpoint_path='', #'/mnt/sda/ray_results/gym/DClaw/TurnFreeValve3ResetFree-v0/2019-08-22T12-37-40-random_translate_centered_around_origin/id=4de1a720-seed=779_2019-08-22_12-37-41qqs0v4da/checkpoint_200/',
                 **kwargs):
        self._last_claw_qpos = DEFAULT_CLAW_RESET_POSE.copy()
        self._last_object_position = np.array([0, 0, 0])
        self._last_object_orientation = np.array([0, 0, 0])
        self._reset_fingers = reset_fingers
        self._reset_frequency = reset_frequency
        self._reset_counter = 0
        self._take_random_actions_for = take_random_actions_for

        self._path_length = path_length
        self._step_count = 0
        super().__init__(
            reset_policy_checkpoint_path=reset_policy_checkpoint_path,
            **kwargs
        )
        self._swap_goal_upon_completion = swap_goal_upon_completion
        self._target_qpos_range = target_qpos_range
        self._init_qpos_range = init_qpos_range
        self._swap_goal_frequency = swap_goal_frequency
        self._traj_count = 0

    def _step(self, action):
        super()._step(action)
        self._step_count += 1

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        obs_dict = super().get_obs_dict()
        obs_dict['step_count'] = self._step_count
        return obs_dict

    def reset(self):
        self._step_count = 0
        self._traj_count += 1

        self._reset_counter += 1
        if self._reset_frequency \
           and self._reset_counter % self._reset_frequency == 0:
            self._reset_counter = 0
            return super().reset()

        dclaw_config = self.robot.get_config('dclaw')
        dclaw_control_mode = dclaw_config.control_mode
        dclaw_config.set_control_mode(ControlMode.JOINT_POSITION)
        for _ in range(self._take_random_actions_for):
            rand_action = np.random.uniform(low=-1, high=1, size=(9,))
            self._step(rand_action)

        if self._reset_fingers:
            if self._use_bowl_arena:
                reset_action = self.robot.normalize_action(
                    {'dclaw': INTERMEDIATE_CLAW_RESET_POSE_0})['dclaw']
                for _ in range(10):
                    self._step(reset_action)
                    reset_action = self.robot.normalize_action(
                        {'dclaw': INTERMEDIATE_CLAW_RESET_POSE_1})['dclaw']
                for _ in range(10):
                    self._step(reset_action)
                    reset_action = self.robot.normalize_action(
                        {'dclaw': INTERMEDIATE_CLAW_RESET_POSE_2})['dclaw']
                for _ in range(10):
                    self._step(reset_action)
                    reset_action = self.robot.normalize_action(
                        {'dclaw': DEFAULT_CLAW_RESET_POSE.copy()})['dclaw']
                for _ in range(10):
                    self._step(reset_action)
            else:
                reset_action = self.robot.normalize_action(
                    {'dclaw': DEFAULT_CLAW_RESET_POSE.copy()})['dclaw']
                for _ in range(15):
                    self._step(reset_action)

        dclaw_config.set_control_mode(dclaw_control_mode)

        if self._policy:
            target_qpos_range = self._target_qpos_range
            self._target_qpos_range = self._reset_target_qpos_range
            self._set_target_object_qpos(
                super()._sample_goal(self.get_obs_dict()))

            for _ in range(self._reset_horizon):
                policy_input = self.get_policy_input()
                action = self._policy.actions_np(policy_input)[0]
                self.step(action)

            self._target_qpos_range = target_qpos_range
            self._set_target_object_qpos(
                super()._sample_goal(self.get_obs_dict()))
        else:
            if self._traj_count % self._swap_goal_frequency == 0:
                self._set_target_object_qpos(
                    self._sample_goal(self.get_obs_dict()))

        return self._get_obs(self.get_obs_dict())


@configurable(pickleable=True)
class DClawTurnFreeValve3ResetFreeSwapGoal(DClawTurnFreeValve3ResetFree):
    """Turns the object reset-free with a target position swapped every reset."""
    def __init__(self,
                 goals=[(0.01, 0.01, 0, 0, 0, np.pi / 2),
                        (-0.01, -0.01, 0, 0, 0, -np.pi / 2)],
                 # goals=[(0.01, 0.01, 0, 0, 0, np.pi / 2),
                 #        (-0.01, -0.01, 0, 0, 0, -np.pi / 2),
                 #        (-0.01, 0.01, 0, 0, 0, np.pi),
                 #        (0.01, -0.01, 0, 0, 0, 0)],

                 #observation_keys=DEFAULT_OBSERVATION_KEYS,
                 choose_furthest_goal=False,
                 **kwargs):
        super().__init__(
            #observation_keys=observation_keys + ('other_reward',),
            **kwargs)
        self._goal_index = 0
        self._goals = np.array(goals)
        self.n_goals = len(self._goals)
        self._reset_target_qpos_range = [(0, 0, 0, 0, 0, 0)]
        self._choose_furthest_goal = choose_furthest_goal

    def get_obs_dict(self):
        obs_dict = super().get_obs_dict()

        obs_dict['goal_index'] = np.array([self._goal_index])
        return obs_dict

    def relabel_path(self, path):
        observations = path['observations']
        goal_angle = np.arctan2(
            observations['target_orientation_sin'][0][2],
            observations['target_orientation_cos'][0][2]
        )

        if goal_angle == self._goals[0][5]:
            path['observations']['target_orientation_sin'][:, 2] = (
                np.sin(self._goals[1][5]))
            path['observations']['target_orientation_cos'][:, 2] = (
                np.cos(self._goals[1][5]))
        elif goal_angle == self._goals[1][5]:
            path['observations']['target_orientation_sin'][:, 2] = (
                np.sin(self._goals[0][5]))
            path['observations']['target_orientation_cos'][:, 2] = (
                np.cos(self._goals[0][5]))
        path['rewards'] = path['observations']['other_reward']
        return path

    def _sample_goal(self, obs_dict):
        if self._choose_furthest_goal:
            goal_rewards = []
            for goal_index in range(self.n_goals):
                self._set_target_object_qpos(self._goals[goal_index])
                goal_reward = self._get_total_reward(self.get_reward_dict(None, self.get_obs_dict()))
                goal_rewards.append(goal_reward)
            print(goal_rewards)
            goal_rewards = np.array(goal_rewards)
            self._goal_index = np.argmin(goal_rewards)
        else:
            other_goal_inds = [i for i in range(self.n_goals) if i != self._goal_index]
            self._goal_index = np.random.choice(other_goal_inds)
            # self._goal_index = (self._goal_index + 1) % self.n_goals
        return self._goals[self._goal_index]

    def _reset(self):
        """ For manual resetting. """
        other_goal_inds = [i for i in range(self.n_goals) if i != self._goal_index]
        other_goal_ind = np.random.choice(other_goal_inds)
        self._set_target_object_qpos(
            self._sample_goal(self.get_obs_dict()))
        other_goal_inds = [i for i in range(self.n_goals) if i != self._goal_index]
        other_goal_ind = np.random.choice(other_goal_inds)
        self._initial_object_qpos = self._goals[other_goal_ind]
        self._reset_dclaw_and_object(
            claw_pos=self._initial_claw_qpos,
            object_pos=np.atleast_1d(self._initial_object_qpos),
            object_vel=np.atleast_1d(self._initial_object_qvel),
            # guide_pos=np.atleast_1d(self._object_target_qpos))
        )


@configurable(pickleable=True)
class DClawTurnFreeValve3ResetFreeSwapGoalEval(DClawTurnFreeValve3Fixed):
    """Turns the object reset-free with a target position swapped every reset."""
    def __init__(self,
                 # goals=[(0.01, 0.01, 0, 0, 0, np.pi / 2),
                 #        (-0.01, -0.01, 0, 0, 0, -np.pi / 2)],
                 goals=[(0.01, 0.01, 0, 0, 0, np.pi / 2),
                        (-0.01, -0.01, 0, 0, 0, -np.pi / 2),
                        (-0.01, 0.01, 0, 0, 0, np.pi),
                        (0.01, -0.01, 0, 0, 0, 0)],
                 cycle_goals=False,
                 **kwargs):
        super().__init__(
            #observation_keys=observation_keys + ('other_reward',),
            **kwargs)
        self._goal_index = 0
        self._goals = np.array(goals)
        self.n_goals = len(self._goals)
        self._cycle_goals = cycle_goals
        if self._cycle_goals:
            self._goal_index = -1
        self._set_goal = False

    def _sample_goal(self, obs_dict):
        if self._cycle_goals:
            self._goal_index = (self._goal_index + 1) % self.n_goals
        else:
            # Sample another goal randomly
            other_goal_inds = [i for i in range(self.n_goals) if i != self._goal_index]
            self._goal_index = np.random.choice(other_goal_inds)
        return self._goals[self._goal_index]

    def set_goal(self, goal_index):
        self._goal_index = goal_index
        self._set_target_object_qpos(self._goals[self._goal_index])
        self._set_goal = True

    def get_obs_dict(self):
        obs_dict = super().get_obs_dict()

        obs_dict['goal_index'] = np.array([self._goal_index])
        return obs_dict

    def _reset(self):
        if not self._set_goal:
            self._set_target_object_qpos(
                self._sample_goal(self.get_obs_dict()))
        if self._cycle_goals:
            other_goal_ind = (self._goal_index - 1) % self.n_goals
        else:
            # Sample init position from one of the other goals
            other_goal_inds = [i for i in range(self.n_goals) if i != self._goal_index]
            other_goal_ind = np.random.choice(other_goal_inds)
        self._initial_object_qpos = self._goals[other_goal_ind]
        self._reset_dclaw_and_object(
            claw_pos=self._initial_claw_qpos,
            object_pos=np.atleast_1d(self._initial_object_qpos),
            object_vel=np.atleast_1d(self._initial_object_qvel),
            # guide_pos=np.atleast_1d(self._object_target_qpos))
        )


@configurable(pickleable=True)
class DClawTurnFreeValve3ResetFreeComposedGoals(DClawTurnFreeValve3ResetFree):
    """ Multistage task consisting of translating then turning. """
    def __init__(self,
                 goals=[
                     (0, 0, 0, 0, 0, 0),
                     (0, 0, 0, 0, 0, np.pi/2),
                     (0, 0, 0, 0, 0, -np.pi/2)
                 ],
                 **kwargs):
        super().__init__(
            **kwargs)
        self._goal_index = 0
        self._goals = np.array(goals)
        self.n_goals = len(self._goals)

    @property
    def num_goals(self):
        return len(self._goals)

    def set_goal(self, goal_index):
        """Allow outside algorithms to alter goals."""
        self._goal_index = goal_index

    def _sample_goal(self, obs_dict):
        return self._goals[self._goal_index]

    def get_reward_dict(self, action, obs_dict):
        """ Alter rewards based on goal. """
        reward_dict = super().get_reward_dict(action, obs_dict)
        if self._goal_index == 0:
            reward_dict['object_to_target_orientation_distance_reward'] *= 0
        else:
            reward_dict['object_to_target_position_distance_reward'] *= 0.1
        return reward_dict


@configurable(pickleable=True)
class DClawTurnFreeValve3FixedResetSwapGoal(DClawTurnFreeValve3Fixed):
    """Turns the object reset-free with a fixed initial and varied target positions."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_angle_range = (-np.pi, np.pi)
        self._init_x_pos_range = [-0.05, 0.05]
        self._init_y_pos_range = [-0.05, 0.05]
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


DEFAULT_PUCK_OBSERVATION_KEYS = (
    'claw_qpos',
    'object_xy_position',
    'last_action',
    'target_xy_position',
    'goal_index',
)


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
