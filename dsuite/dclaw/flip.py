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
from transforms3d.euler import euler2quat, quat2euler

from dsuite.dclaw.base_env import (BaseDClawObjectEnv,
                                   BaseDClawEnv,
                                   DEFAULT_CLAW_RESET_POSE)
from dsuite.utils.configurable import configurable
from dsuite.utils.resources import get_asset_path
from dsuite.utils.circle_math import circle_distance, quat_distance
from dsuite.components.robot.config import ControlMode
from dsuite.components.robot import RobotState


# The observation keys that are concatenated as the environment observation.
DEFAULT_OBSERVATION_KEYS = (
    'claw_qpos',
    'object_position',
    'object_quaternion',
    'last_action',
    'target_quaternion',
    'target_position',
)

DEFAULT_HARDWARE_OBSERVATION_KEYS = (
    'claw_qpos',
    'last_action',
)


# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_free_valve3_in_arena.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw_valve3_in_less_tiny_box.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw_valve3_fixed_tiny_box.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_valve3_free.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_free_cube.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw_x2_catch.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_octahedron.xml'
# DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_dodecahedron.xml'
DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_eraser_sloped_arena.xml'
DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_eraser.xml'


class BaseDClawFlipFreeObject(BaseDClawObjectEnv, metaclass=abc.ABCMeta):
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
        self._target_offset = np.array([0, 0, 0.0175]) # get from dodecahedron.xml, object['pos']

        super().__init__(
            sim_model=get_asset_path(asset_path),
            robot_config=self.get_config_for_device(
                device_path, free_object=True, free_claw=free_claw, quat=True),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)

        self._desired_claw_pos = DEFAULT_CLAW_RESET_POSE.copy()
        self._last_action = np.zeros(9)

        self._target_bid = self.model.body_name2id('target')

        # The following are modified (possibly every reset) by subclasses.
        self._set_target_object_qpos((0, 0, 0, 0, 0, 0))
        self._initial_claw_qpos = DEFAULT_CLAW_RESET_POSE.copy()
        self._initial_object_qpos = (0, 0, 0, 0, 0, 0)
        self._initial_object_qvel = (0, 0, 0, 0, 0, 0, 0)

    def _reset(self):
        """Resets the environment."""
        # print(self._initial_object_qpos[3:])
        # init_object_pos = np.concatenate([
        #     self._initial_object_qpos[:3],
        #     euler2quat(*self._initial_object_qpos[3:])
        # ])
        self._reset_dclaw_and_object(
            claw_pos=self._initial_claw_qpos,
            object_pos=np.atleast_1d(self._initial_object_qpos),
            # object_vel=np.atleast_1d(self._initial_object_qvel),
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
        object_quaternion = object_state.qpos[3:] #euler2quat(*object_state.qpos[3:])

        if object_quaternion[0] < 0: # avoid double cover
            object_quaternion = -object_quaternion

        target_position = self._object_target_position
        target_quaternion = self._target_quaternion

        object_to_target_relative_position = object_position - target_position
        object_to_target_sphere_distance = quat_distance(
            object_quaternion, target_quaternion)

        # CORNER_THRESHOLD = 0.05 #0.07
        # in_corner = 0
        # if np.all(object_position[:2] > CORNER_THRESHOLD):
        #     in_corner = 1
        # elif object_position[0] < -CORNER_THRESHOLD and object_position[1] > CORNER_THRESHOLD:
        #     in_corner = 2
        # elif object_position[0] < -CORNER_THRESHOLD and object_position[1] < -CORNER_THRESHOLD:
        #     in_corner = 3
        # elif object_position[0] > CORNER_THRESHOLD and object_position[1] < - CORNER_THRESHOLD:
        #     in_corner = 4
        return collections.OrderedDict((
            ('claw_qpos', claw_state.qpos.copy()),
            ('claw_qvel', claw_state.qvel.copy()),
            ('object_position', object_position),
            ('object_xy_position', object_position[:2]),
            ('object_quaternion', object_quaternion),
            # ('object_orientation_cos', np.cos(object_orientation)),
            # ('object_orientation_sin', np.sin(object_orientation)),
            ('object_qvel', object_state.qvel),
            ('last_action', self._last_action),
            # ('target_angle', target_orientation[2]),
            ('target_position', target_position),
            ('target_quaternion', target_quaternion),
            # ('target_orientation', target_orientation),
            # ('target_orientation_cos', np.cos(target_orientation)),
            # ('target_orientation_sin', np.sin(target_orientation)),
            ('object_to_target_relative_position', object_to_target_relative_position),
            # ('object_to_target_relative_orientation', object_to_target_relative_orientation),
            ('object_to_target_position_distance', np.linalg.norm(object_to_target_relative_position[:2])),
            # ('object_to_target_z_position_distance', np.linalg.norm(object_to_target_relative_position[2])),
            ('object_to_target_sphere_distance', object_to_target_sphere_distance),
            # ('object_to_target_circle_distance', np.linalg.norm(object_to_target_circle_distance)),
            # ('in_corner', np.array([in_corner])),
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
        # object_to_target_z_position_distance = obs_dict['object_to_target_z_position_distance']
        object_to_target_sphere_distance = obs_dict['object_to_target_sphere_distance']

        reward_dict = collections.OrderedDict((
            # Penalty for distance away from goal.
            # ('object_to_target_position_distance_cost', -5 *
            #     object_to_target_position_distance),
            # ('object_to_target_orientation_distance_cost', -5 *
            #     object_to_target_circle_distance),
            ('object_to_target_position_distance_reward',
             - np.log(10 * (object_to_target_position_distance + 0.001))),
            # ('object_to_target_z_position_distance_reward',
            #  - np.log(30 * (object_to_target_z_position_distance + 0.005))),
            ('object_to_target_orientation_distance_reward',
             - np.log(0.5 * (object_to_target_sphere_distance + 0.01))),

            # Penalty for difference with nomimal pose.
            ('pose_diff_cost',
             -1 * np.linalg.norm(obs_dict['claw_qpos'] - self._desired_claw_pos)),
            # Penality for high velocities.
            ('joint_vel_cost', -1 * np.linalg.norm(claw_vel[claw_vel >= 0.5])),

            # Reward for close proximity with goal.
            ('bonus_small', 10 * (object_to_target_sphere_distance < 0.5)),
            ('bonus_big', 50 * (object_to_target_sphere_distance < 0.20)),
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
                obs_dict['object_to_target_sphere_distance'], 0.2) / 0.2),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))

    def _set_target_object_qpos(self, target_qpos: float):
        """Sets the goal position and orientation."""
        # Modulo to [-pi, pi].
        self._object_target_position = target_qpos[:3]
        self._object_target_orientation = np.mod(
            np.array(target_qpos[3:]) + np.pi, 2 * np.pi) - np.pi

        # Mark the target position in sim.
        self.model.body_pos[self._target_bid] = self._object_target_position + self._target_offset
        quat = euler2quat(*self._object_target_orientation)
        if quat[0] < 0: # avoid double cover
            quat = -quat
        self.model.body_quat[self._target_bid] = self._target_quaternion = quat

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
class DClawFlipEraserFixed(BaseDClawFlipFreeObject):
    """Turns the dodecahedron with a fixed initial and fixed target position."""

    def __init__(self,
                 init_qpos_range=(
                     (-0.08, -0.08, 0.03, 0, 0, -np.pi),
                     (0.08, 0.08, 0.03, 0, 0, np.pi)
                 ),
                 target_qpos_range=[(0, 0, 0, np.pi, 0, 0), (0, 0, 0, np.pi, 0, 0)],
                 reset_from_corners=False,
                 reset_policy_checkpoint_path='/home/abhigupta/ray_results/gym/DClaw/FlipEraserResetFree-v0/2019-08-16T17-14-22-random_translate_slippery/id=f710abae-seed=2754_2019-08-16_17-14-224agh2l_j/checkpoint_950/',
                 *args, **kwargs):
        self._init_qpos_range = init_qpos_range
        self._target_qpos_range = target_qpos_range
        self._reset_from_corners = reset_from_corners
        super().__init__(*args, **kwargs)
        self._policy = None
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

    def get_policy_input(self):
        from softlearning.models.utils import flatten_input_structure
        obs_dict = self.get_obs_dict()
        observation = flatten_input_structure({
            key: obs_dict[key][None, ...]
            for key in self._policy.observation_keys
        })
        return observation

    def _sample_goal(self, obs_dict):
        if isinstance(self._target_qpos_range, (list,)):
            rand_index = np.random.randint(len(self._target_qpos_range))
            target_qpos = np.array(self._target_qpos_range[rand_index])
        elif isinstance(self._target_qpos_range, (tuple,)):
            target_qpos = np.random.uniform(
                low=self._target_qpos_range[0],
                high=self._target_qpos_range[1]
            )
        return target_qpos

    def _rollout_reset_policy(self):
        target_qpos_range = self._target_qpos_range
        self._target_qpos_range = self._reset_target_qpos_range
        self._set_target_object_qpos(
            self._sample_goal(self.get_obs_dict()))

        for _ in range(self._reset_horizon):
            policy_input = self.get_policy_input()
            action = self._policy.actions_np(policy_input)[0]
            self.step(action)

        self._target_qpos_range = target_qpos_range
        self._set_target_object_qpos(
            self._sample_goal(self.get_obs_dict()))

    def _reset(self):
        if isinstance(self._init_qpos_range, (list,)):
            rand_index = np.random.randint(len(self._init_qpos_range))
            self._initial_object_qpos = np.array(self._init_qpos_range[rand_index])
        elif isinstance(self._init_qpos_range, (tuple,)):
            self._initial_object_qpos = np.random.uniform(
                low=self._init_qpos_range[0], high=self._init_qpos_range[1]
            )
        self._initial_object_qpos = np.concatenate([
            self._initial_object_qpos[:3],
            euler2quat(*self._initial_object_qpos[3:])
        ])
        self._set_target_object_qpos(
            self._sample_goal(self.get_obs_dict()))
        super()._reset()

        # Reset eraser to random corner
        if self._reset_from_corners:
            corner_index = np.random.randint(2, size=2) * 2 - 1 # -1 or 1
            self.data.qpos[-7:-5] = np.array([0.05, 0.05]) * corner_index

            open_claw_position = np.tile([0, -1.5, -1.5], 3)
            for _ in range(5):
                self.data.ctrl[:9] = open_claw_position
                self.data.qfrc_applied[-7:-5] = np.array([1, 1]) * corner_index
                self.sim_scene.advance()
            self.data.qfrc_applied[-7:-5] = 0
            self.data.qpos[:9] = DEFAULT_CLAW_RESET_POSE.copy()

        # Use reset policy
        if self._policy:
            self._rollout_reset_policy()

@configurable(pickleable=True)
class DClawFlipEraserResetFree(DClawFlipEraserFixed):
    """Turns the object reset-free with a fixed initial and varied target positions."""

    def __init__(self,
                 swap_goal_upon_completion: bool = True,
                 reset_fingers=True,
                 target_qpos_range=((-0.08, -0.08, 0, 0, 0, 0), (0.08, 0.08, 0, 0, 0, 0)),
                 init_qpos_range=[(0, 0, 0.03, 0, 0, 0)],
                 position_reward_weight=1,
                 reset_frequency: int = 0,
                 reset_policy_checkpoint_path='/home/abhigupta/ray_results/gym/DClaw/FlipEraserResetFree-v0/2019-08-16T17-14-22-random_translate_slippery/id=f710abae-seed=2754_2019-08-16_17-14-224agh2l_j/checkpoint_950/',
                 **kwargs):
        self._last_claw_qpos = DEFAULT_CLAW_RESET_POSE.copy()
        self._last_object_position = np.array([0, 0, 0])
        self._last_object_orientation = np.array([0, 0, 0])
        self._reset_fingers = reset_fingers
        self._reset_frequency = reset_frequency
        self._reset_counter = 0

        super().__init__(
            reset_policy_checkpoint_path=reset_policy_checkpoint_path,
            **kwargs
        )
        self._target_qpos_range = target_qpos_range
        self._init_qpos_range = init_qpos_range
        self._swap_goal_upon_completion = swap_goal_upon_completion
        self._position_reward_weight = position_reward_weight

    def _sample_goal(self, obs_dict):
        if isinstance(self._target_qpos_range, (list,)):
            rand_index = np.random.randint(len(self._target_qpos_range))
            target_qpos = np.array(self._target_qpos_range[rand_index])
        elif isinstance(self._target_qpos_range, (tuple,)):
            target_qpos = np.random.uniform(
                low=self._target_qpos_range[0],
                high=self._target_qpos_range[1]
            )
        return target_qpos

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
            for _ in range(15):
                self._step(reset_action)
        self._set_target_object_qpos(self._sample_goal(obs_dict))
        dclaw_config.set_control_mode(dclaw_control_mode)

        # Use reset policy
        if self._policy:
            self._reset_target_qpos_range = [(0, 0, 0, 0, 0, 0)]
            self._rollout_reset_policy()

        return self._get_obs(self.get_obs_dict())


@configurable(pickleable=True)
class DClawFlipEraserResetFreeSwapGoal(DClawFlipEraserResetFree):
    """Turns the object reset-free with a target position swapped every reset."""
    def __init__(self,
                 goals = ((0, 0, 0, np.pi, 0, 0), (0, 0, 0, 0, 0, 0)),
                 **kwargs):
        super().__init__(
            **kwargs)
        self._goal_index = 0
        self._goals = np.array(goals)
        self.n_goals = len(self._goals)
        self._init_qpos_range = [(0, 0, 0.03, np.pi, 0, 0), (0, 0, 0.03, 0, 0, 0)]

    def _sample_goal(self, obs_dict):
        self._goal_index = (self._goal_index + 1) % self.n_goals
        return self._goals[self._goal_index]


@configurable(pickleable=True)
class DClawFlipEraserResetFreeSwapGoalEval(DClawFlipEraserFixed):
    """Turns the object reset-free with a target position swapped every reset."""
    def __init__(self,
                 goals = ((0, 0, 0, np.pi, 0, 0), (0, 0, 0, 0, 0, 0)),
                 **kwargs):
        super().__init__(
            **kwargs)
        self._goal_index = 0
        self._goals = np.array(goals)
        self.n_goals = len(self._goals)
        self._set_goal = False

    def _sample_goal(self, obs_dict):
        self._goal_index = (self._goal_index + 1) % self.n_goals
        return self._goals[self._goal_index]

    def set_goal(self, goal_index):
        self._goal_index = goal_index
        self._set_target_object_qpos(self._goals[self._goal_index])
        self._set_goal = True

    def _reset(self):
        if not self._set_goal:
            self._set_target_object_qpos(
                self._sample_goal(self.get_obs_dict()))
        self._initial_object_qpos = self._goals[(self._goal_index - 1) % 2]

        self._initial_object_qpos = np.concatenate([
            self._initial_object_qpos[:3],
            euler2quat(*self._initial_object_qpos[3:])
        ])

        self._reset_dclaw_and_object(
            claw_pos=self._initial_claw_qpos,
            object_pos=np.atleast_1d(self._initial_object_qpos),
            object_vel=np.atleast_1d(self._initial_object_qvel),
        )


@configurable(pickleable=True)
class DClawFlipEraserResetFreeComposedGoals(DClawFlipEraserResetFree):
    """ Multistage task consisting of translating then turning. """
    def __init__(self,
                 goals=[
                     (0, 0, 0, 0, 0, 0),
                     (0, 0, 0, np.pi, 0, 0),
                     (0, 0, 0, np.pi, 0, 0),
                     (0, 0, 0, 0, 0, 0),
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
            reward_dict['object_to_target_orientation_distance_reward'] *= 0.1
        elif self._goal_index == 1:
            reward_dict['object_to_target_position_distance_reward'] *= 0.1
        elif self._goal_index == 2:
            reward_dict['object_to_target_orientation_distance_reward'] *= 0.1
        elif self._goal_index == 3:
            reward_dict['object_to_target_position_distance_reward'] *= 0.1
        return reward_dict
