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
import glob, gzip, pickle
import abc
import collections
from typing import Dict, Optional, Sequence

import numpy as np
from transforms3d.euler import euler2quat

from dsuite.dclaw.base_env import (
    BaseDClawObjectEnv,
    BaseDClawEnv,
    DEFAULT_CLAW_RESET_POSE)
from dsuite.utils.configurable import configurable
from dsuite.utils.resources import get_asset_path
from dsuite.utils.circle_math import circle_distance, quat_distance
from dsuite.components.robot.config import ControlMode
from dsuite.components.robot import RobotState
from dsuite.dclaw.turn_free_object import (
    INTERMEDIATE_CLAW_RESET_POSE_0,
    INTERMEDIATE_CLAW_RESET_POSE_1,
    INTERMEDIATE_CLAW_RESET_POSE_2
)

from dsuite.dclaw.config import get_dclaw_beads_config


# The observation keys that are concatenated as the environment observation.
DEFAULT_OBSERVATION_KEYS = (
    'claw_qpos',
    'objects_positions',
    'last_action',
    'objects_target_positions',
    'goal_index',
)

DEFAULT_HARDWARE_OBSERVATION_KEYS = (
    'claw_qpos',
    'last_action',
)

DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_beads.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_beads_4.xml'


class BaseDClawSlideFreeObject(BaseDClawObjectEnv, metaclass=abc.ABCMeta):
    """Shared logic for DClaw turn tasks."""
    def __init__(
            self,
            asset_path: str = DCLAW3_ASSET_PATH,
            observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
            device_path: Optional[str] = None,
            camera_config: dict = None,
            frame_skip: int = 40,
            free_claw: bool = False,
            position_reward_weight: int = 1,
            num_objects: int = 9,
            **kwargs
    ):
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
        self._initial_claw_qpos = DEFAULT_CLAW_RESET_POSE.copy()

        self._num_objects = num_objects
        if self._num_objects == 2:
            self._objects_offsets = np.array(
                [-0.0175, 0.0175]
            )
            self._initial_objects_qpos = [0, 0]
            self._initial_objects_qvel = [0, 0] #(0, 0, 0, 0, 0, 0)
            asset_path = 'dsuite/dclaw/assets/dclaw3xh_beads_2.xml'
            self._objects_target_positions = [0, 0]
        elif self._num_objects == 4:
            self._objects_offsets = np.array(
                [-0.0525, -0.0175, 0.0175, 0.0525]
            )
            self._initial_objects_qpos = [0, 0, 0, 0]
            self._initial_objects_qvel = [0, 0, 0, 0]
            asset_path = 'dsuite/dclaw/assets/dclaw3xh_beads_4.xml'
            self._objects_target_positions = [0, 0, 0, 0]
        elif self._num_objects == 9:
            self._objects_offsets = np.array(
                [-0.035, 0, 0.035]*3
            )
            self._initial_objects_qpos = [0, 0, 0]*3
            self._initial_objects_qvel = [0, 0, 0]*3
            asset_path = 'dsuite/dclaw/assets/dclaw3xh_beads_9.xml'
            self._objects_target_positions = [0, 0, 0]*3
        self._desired_claw_pos = DEFAULT_CLAW_RESET_POSE.copy()
        self._last_action = np.zeros(9)

        super().__init__(
            sim_model=get_asset_path(asset_path),
            robot_config=get_dclaw_beads_config(num_objects),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)

        self._target_bids, i = [], 0
        for i in range(self._num_objects):
            self._target_bids.append(self.model.body_name2id(f'target_{i}'))
            i += 1

    def _reset(self):
        """Resets the environment."""
        claw_init_state = self.robot.get_initial_state(
            ['dclaw'])[0]

        initial_state = {
            'dclaw': RobotState(qpos=self._initial_claw_qpos, qvel=claw_init_state.qvel),
        }

        i = 0
        for i in range(self._num_objects):
            initial_state.update({
                f'object_{i}': RobotState(
                    qpos=self._initial_objects_qpos[i],
                    qvel=self._initial_objects_qvel[i]
                )
            })
            i += 1
        self.robot.set_state(initial_state)

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        # action  = self.robot.normalize_action(
        #             {'dclaw': DEFAULT_CLAW_RESET_POSE.copy()})['dclaw']
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

        claw_state = self.robot.get_state(['dclaw'])[0]

        objects_states, i = [], 0

        for i in range(self._num_objects):
            objects_states.append(self.robot.get_state(f'object_{i}'))
            i += 1

        objects_positions = np.concatenate(
            [object_state.qpos[:3].copy() for object_state in objects_states]
        )
        objects_vels = np.concatenate(
            [object_state.qvel[:3].copy() for object_state in objects_states]
        )

        objects_target_positions = self._objects_target_positions
        objects_to_targets_relative_positions = objects_positions - objects_target_positions
        objects_to_targets_distances = np.abs(objects_to_targets_relative_positions)
        return collections.OrderedDict((
            ('claw_qpos', claw_state.qpos.copy()),
            ('claw_qvel', claw_state.qvel.copy()),
            ('objects_positions', objects_positions),
            ('object_qvel', objects_vels),
            ('last_action', self._last_action),
            ('objects_target_positions', objects_target_positions),
            ('objects_to_targets_relative_positions', objects_to_targets_relative_positions),
            ('objects_to_targets_distances', objects_to_targets_distances),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        # object_to_target_relative_orientation = obs_dict['object_to_target_relative_orientation']

        claw_vel = obs_dict['claw_qvel']

        objects_to_targets_distances = obs_dict['objects_to_targets_distances']
        objects_to_targets_mean_distance = np.mean(objects_to_targets_distances)

        reward_dict = collections.OrderedDict((
            ('objects_to_targets_mean_distance_reward',
             - np.log(10 * objects_to_targets_mean_distance + 0.01)),
            # Penalty for difference with nomimal pose.
            ('pose_diff_cost',
             -1 * np.linalg.norm(obs_dict['claw_qpos'] - self._desired_claw_pos)),
            # Penality for high velocities.
            ('joint_vel_cost', -1 * np.linalg.norm(claw_vel[claw_vel >= 0.5])),
            # Reward for close proximity with goal.
            ('bonus_small', 10 * (objects_to_targets_mean_distance < 0.5)),
            ('bonus_big', 50 * (objects_to_targets_mean_distance < 0.20)),
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
                np.mean(np.abs(obs_dict['objects_to_targets_relative_positions'])), 0.2) / 0.2),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))

    def _set_target_objects_qpos(self, target_qpos: float):
        raise NotImplementedError()
        # # Mark the target position in sim.
        # self._objects_target_positions = []
        # for i, target_pos in enumerate(target_qpos):
        #     self._objects_target_positions.append(target_pos)
        #     self.model.body_pos[self._target_bids[i]] = target_pos + self._objects_offsets[i]
        # self._objects_target_positions = np.array(self._objects_target_positions)

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
            from dsuite.dclaw.turn import get_image_service
            return self._image_service.get_image(*args, **kwargs)

        return super().render(*args, **kwargs)


@configurable(pickleable=True)
class DClawSlideBeadsFixed(BaseDClawSlideFreeObject):
    """Slide beads to desired positions."""

    def __init__(self,
                 # init_qpos_range=[
                 #     (0, 0)
                 # ],
                 # target_qpos_range=[  # target pos relative to init
                 #     (-0.0825, 0.0825),
                 #     (-0.04, 0.04),
                 #     (0.0825, 0.0825),
                 # ],
                 # init_qpos_range=[(0, 0, 0, 0)],
                 # target_qpos_range=[
                 #     (-0.0475, -0.0475, 0.0475, 0.0475)
                 # ],
                 init_qpos_range=[[0, 0, 0]*3],
                 target_qpos_range=[
                     [-0.0875, 0, 0.0875]*3
                 ],
                 reset_policy_checkpoint_path='', #'/mnt/sda/ray_results/gym/DClaw/LiftDDFixed-v0/2019-08-01T18-06-55-just_lift_single_goal/id=3ac8c6e0-seed=5285_2019-08-01_18-06-565pn01_gq/checkpoint_1500/',
                 cycle_goals=False,
                 *args, **kwargs):
        self._init_qpos_range = init_qpos_range
        self._target_qpos_range = target_qpos_range
        super().__init__(*args, **kwargs)
        self._cycle_goals = cycle_goals
        self._goal_index = 0
        self._let_alg_set_goals = False
        # self._reset_horizon = 0
        # self._policy = None
        # if reset_policy_checkpoint_path:
        #     self._load_policy(reset_policy_checkpoint_path)

    def get_obs_dict(self):
        obs_dict = super().get_obs_dict()

        obs_dict['goal_index'] = np.array([self._goal_index])
        return obs_dict

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

    def _sample_goals(self, obs_dict):
        if isinstance(self._target_qpos_range, (list,)):
            rand_index = np.random.randint(len(self._target_qpos_range))
            target_qpos = np.array(self._target_qpos_range[rand_index])
            if self._cycle_goals:
                if not self._let_alg_set_goals:
                    self._goal_index = (self._goal_index + 1) % self.num_goals
                target_qpos = self._target_qpos_range[self._goal_index]
        elif isinstance(self._target_qpos_range, (tuple,)):
            target_qpos = np.random.uniform(
                low=self._target_qpos_range[0],
                high=self._target_qpos_range[1]
            )
        return target_qpos

    def _set_target_objects_qpos(self, target_qpos: float):
        # Mark the target position in sim.
        self._objects_target_positions = []
        for i, x_pos in enumerate(target_qpos):
            self._objects_target_positions.append(x_pos)
            self.model.body_pos[self._target_bids[i]][0] = x_pos + self._objects_offsets[i]
        self._objects_target_positions = np.array(self._objects_target_positions)

    def _reset(self):
        if isinstance(self._init_qpos_range, (list,)):
            rand_index = np.random.randint(len(self._init_qpos_range))
            self._initial_objects_qpos = np.array(self._init_qpos_range[rand_index])
        elif isinstance(self._init_qpos_range, (tuple,)):
            self._initial_object_qpos = np.random.uniform(
                low=self._init_qpos_range[0], high=self._init_qpos_range[1]
            )

        self._set_target_objects_qpos(
            self._sample_goals(self.get_obs_dict()))
        super()._reset()

    def get_policy_input(self):
        from softlearning.models.utils import flatten_input_structure
        obs_dict = self.get_obs_dict()
        observation = flatten_input_structure({
            key: obs_dict[key][None, ...]
            for key in self._policy.observation_keys
        })
        return observation

    def reset(self):
        self.sim.reset()
        self.sim.forward()
        self._reset()

        # if self._policy:
        #     target_qpos_range = self._target_qpos_range
        #     self._target_qpos_range = self._reset_target_qpos_range
        #     self._set_target_object_qpos(
        #         self._sample_goals(self.get_obs_dict()))

        #     for _ in range(self._reset_horizon):
        #         policy_input = self.get_policy_input()
        #         action = self._policy.actions_np(policy_input)[0]
        #         self.step(action)

        #     self._target_qpos_range = target_qpos_range
        # self._set_target_objects_qpos(
        #     self._sample_goals(self.get_obs_dict()))

        obs_dict = self.get_obs_dict()
        self.last_obs_dict = obs_dict
        self.last_reward_dict = None
        self.last_score_dict = None
        self.is_done = False

        return self._get_obs(obs_dict)


@configurable(pickleable=True)
class DClawSlideBeadsResetFree(DClawSlideBeadsFixed):
    """Slide the objects reset-free with a fixed initial and varied target positions."""

    def __init__(self,
                 swap_goal_upon_completion: bool = True,
                 reset_fingers=True,
                 position_reward_weight=1,
                 # target_qpos_range=(
                 #     (-0.1, -0.1, 0.0, 0, 0, 0),
                 #     (0.1, 0.1, 0.0, 0, 0, 0), # bgreen side up
                 # ),
                 #  target pos relative to init
                 target_qpos_range = [
                     (-0.0825, 0.0825),
                     (0, 0),
                 ],
                 init_qpos_range = [(0, 0)],
                 reset_frequency: int = 0,
                 reset_policy_checkpoint_path='', # '/home/abhigupta/ray_results/gym/DClaw/LiftDDResetFree-v0/2019-08-12T22-28-02-random_translate/id=1efced72-seed=3335_2019-08-12_22-28-03bqyu82da/checkpoint_1500/',
                 **kwargs):
        self._last_claw_qpos = DEFAULT_CLAW_RESET_POSE.copy()
        self._last_objects_positions = np.array([0, 0])
        # self._last_object_orientation = np.array([0, 0, 0])
        self._reset_fingers = reset_fingers
        self._reset_frequency = reset_frequency
        self._reset_counter = 0

        super().__init__(reset_policy_checkpoint_path=reset_policy_checkpoint_path, **kwargs)
        self._target_qpos_range = target_qpos_range
        self._init_qpos_range = init_qpos_range
        self._swap_goal_upon_completion = swap_goal_upon_completion
        self._position_reward_weight = position_reward_weight
        self._reset_target_qpos_range = [(0, 0, 0.041, 0, 0, 0)]

    def reset(self):
        self._reset_counter += 1
        if self._reset_frequency \
           and self._reset_counter % self._reset_frequency == 0:
            self._reset_counter = 0
            return super().reset()

        obs_dict = self.get_obs_dict()
        dclaw_config = self.robot.get_config('dclaw')
        dclaw_control_mode = dclaw_config.control_mode
        dclaw_config.set_control_mode(ControlMode.JOINT_POSITION)

        if self._reset_fingers:
            reset_action = self.robot.normalize_action(
                {'dclaw': DEFAULT_CLAW_RESET_POSE.copy()})['dclaw']
            for _ in range(10):
                self._step(reset_action)

        self._set_target_objects_qpos(self._sample_goals(obs_dict))
        dclaw_config.set_control_mode(dclaw_control_mode)

        # if self._policy:
        #     target_qpos_range = self._target_qpos_range
        #     self._target_qpos_range = self._reset_target_qpos_range
        #     self._set_target_objects_qpos(
        #         self._sample_goals(self.get_obs_dict()))

        #     for _ in range(self._reset_horizon):
        #         policy_input = self.get_policy_input()
        #         action = self._policy.actions_np(policy_input)[0]
        #         self.step(action)

        #     self._target_qpos_range = target_qpos_range
        #     self._set_target_object_qpos(
        #         self._sample_goals(self.get_obs_dict()))

        return self._get_obs(self.get_obs_dict())


@configurable(pickleable=True)
class DClawSlideBeadsResetFreeEval(DClawSlideBeadsFixed):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cycle_goals = True

    def _reset(self):
        self._set_target_objects_qpos(
            self._sample_goals(self.get_obs_dict()))
        reset_index = (self._goal_index - 1) % self.num_goals
        self._initial_objects_qpos = np.array(self._target_qpos_range[reset_index])
        super(DClawSlideBeadsFixed, self)._reset()


# @configurable(pickleable=True)
# class DClawLiftDDResetFreeComposedGoals(DClawLiftDDResetFree):
#     """ Multistage of translating to origin, lifting, then reorienting. """

#     def __init__(self,
#                  goals=[
#                      (0, 0, 0, 0, 0, 0),
#                      (0, 0, 0.05, 0, 0, 0),
#                      (0, 0, 0, np.pi, 0, 0)
#                  ],
#                  **kwargs):
#         super().__init__(
#             **kwargs)
#         self._goal_index = 0
#         self._goals = np.array(goals)
#         self.n_goals = len(self._goals)

#     @property
#     def num_goals(self):
#         return len(self._goals)

#     def set_goal(self, goal_index):
#         """Allow outside algorithms to alter goals."""
#         self._goal_index = goal_index

#     def _sample_goals(self, obs_dict):
#         return self._goals[self._goal_index]

#     def get_reward_dict(self, action, obs_dict):
#         """ Alter rewards based on goal. """
#         reward_dict = super().get_reward_dict(action, obs_dict)
#         if self._goal_index == 0:
#             reward_dict['object_to_target_z_position_distance_reward'] *= 0
#             reward_dict['object_to_target_orientation_distance_reward'] *= 0
#         elif self._goal_index == 1:
#             reward_dict['object_to_target_xy_position_distance_reward'] *= 0
#             reward_dict['object_to_target_orientation_distance_reward'] *= 0
#         elif self._goal_index == 2:
#             reward_dict['object_to_target_z_position_distance_reward'] *= 0
#             reward_dict['object_to_target_xy_position_distance_reward'] *= 0
#         return reward_dict
