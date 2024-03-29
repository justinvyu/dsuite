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

"""Turn tasks with DClaw robots.

This is a single rotation of an object from an initial angle to a target angle.
"""

import abc
import collections
from typing import Dict, Optional, Sequence

import numpy as np
from transforms3d.euler import euler2quat

from dsuite.components.robot.dynamixel_robot import DynamixelRobotState
from dsuite.dclaw.base_env import BaseDClawObjectEnv
from dsuite.simulation.randomize import SimRandomizer
from dsuite.utils.configurable import configurable
from dsuite.utils.resources import get_asset_path
from dsuite.utils.circle_math import circle_distance
from dsuite.components.robot.config import ControlMode
from dsuite.components.robot import RobotState
import pickle

IMAGE_SERVICE = None

def get_image_service(*args, topic, **kwargs):
    try:
        from dsuite.utils import kinect_image_service
    except ImportError as e:
        if "cannot import name 'kinect_image_service'" not in e.msg:
            raise

    try:
        from dsuite.utils import logitech_image_service
    except ImportError as e:
        if "cannot import name 'logitech_image_service'" not in e.msg:
            raise

    global IMAGE_SERVICE
    if IMAGE_SERVICE is None:
        print("CREATING NEW IMAGE_SERVICE")
        # IMAGE_SERVICE = logitech_image_service.LogitechImageService(
        #     *args, topic=topic, **kwargs)
        IMAGE_SERVICE = kinect_image_service.KinectImageService(
            *args, topic=topic, **kwargs)

    return IMAGE_SERVICE


# The observation keys that are concatenated as the environment observation.
DEFAULT_OBSERVATION_KEYS = (
    'claw_qpos',
    'object_angle_cos',
    'object_angle_sin',
    'last_action',
    # 'object_to_target_angle_distance',
    'target_angle_cos',
    'target_angle_sin',
)

# Reset pose for the claw joints.
DEFAULT_CLAW_RESET_POSE = np.array([0., -np.pi / 3, np.pi / 3] * 3)

DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_valve3_v0.xml'

# Threshold near the joint limits at which we consider to be unsafe.
SAFETY_POS_THRESHOLD = 5 * np.pi / 180  # 5 degrees

SAFETY_VEL_THRESHOLD = 1.0  # 1rad/s

# Current threshold above which we consider as unsafe.
SAFETY_CURRENT_THRESHOLD = 200  # mA


class BaseDClawTurn(BaseDClawObjectEnv, metaclass=abc.ABCMeta):
    """Shared logic for DClaw turn tasks."""

    def __init__(self,
                 asset_path: str = DCLAW3_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 device_path: Optional[str] = None,
                 frame_skip: int = 40,
                 camera_config=None,
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
            robot_config=self.get_config_for_device(device_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)

        self._camera_config = camera_config
        if self._camera_config is not None:
            self.image_service = get_image_service(**camera_config)
        self._desired_claw_pos = DEFAULT_CLAW_RESET_POSE
        self._last_action = np.zeros(9)

        self._target_bid = self.model.body_name2id('target')

        # The following are modified (possibly every reset) by subclasses.
        self._initial_object_pos = 0
        self._initial_object_vel = 0
        self._set_target_object_pos(0)

    def _reset(self):
        """Resets the environment."""
        self._reset_dclaw_and_object(
            claw_pos=DEFAULT_CLAW_RESET_POSE,
            object_pos=self._initial_object_pos,
            object_vel=self._initial_object_vel,
            guide_pos=self._target_object_pos)

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({
            'dclaw': action,
            # 'guide': np.atleast_1d(self._target_object_pos),
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

        # Calculate the signed angle difference to the target in [-pi, pi].
        object_angle = object_state.qpos.copy()
        object_to_target_angle_dist = circle_distance(
            self._target_object_pos, object_angle)
        # target_error = np.mod(target_error + np.pi, 2 * np.pi) - np.pi
        obs_dict = collections.OrderedDict((
            ('claw_qpos', claw_state.qpos.copy()),
            ('claw_qvel', claw_state.qvel.copy()),
            ('object_angle_cos', np.cos(object_state.qpos)),
            ('object_angle_sin', np.sin(object_state.qpos)),
            ('object_rotational_vel', object_state.qvel.copy()),
            ('last_action', self._last_action.copy()),
            # ('target_error', target_error),
            ('target_angle', self._target_object_pos[None].copy()),
            ('target_angle_cos', np.cos(self._target_object_pos)[None]),
            ('target_angle_sin', np.sin(self._target_object_pos)[None]),
            ('object_to_target_angle_distance', object_to_target_angle_dist),
        ))

        # Add hardware-specific state if present.
        if isinstance(claw_state, DynamixelRobotState):
            obs_dict['claw_current'] = claw_state.current

        return obs_dict

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        object_to_target_angle_dist = obs_dict['object_to_target_angle_distance']
        claw_vel = obs_dict['claw_qvel']

        reward_dict = collections.OrderedDict((
            # Penalty for distance away from goal.
            ('object_to_target_angle_distance_reward', - np.log(object_to_target_angle_dist + 1e-10)),
            # Penalty for difference with nomimal pose.
            ('pose_diff_cost',
             -1 * np.linalg.norm(obs_dict['claw_qpos'] - self._desired_claw_pos)
            ),
            # Penality for high velocities.
            ('joint_vel_cost',
             -1 * np.linalg.norm(claw_vel[np.abs(claw_vel) >= 0.5])),

            # Reward for close proximity with goal.
            ('bonus_small', 10 * (object_to_target_angle_dist < 0.25)),
            ('bonus_big', 50 * (object_to_target_angle_dist < 0.10)),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        qpos = obs_dict['claw_qpos']
        # Calculate if the claw positions are near the edges of the joint range.
        qpos_range = self.robot.get_config('dclaw').qpos_range
        near_lower_limit = (
            np.abs(qpos_range[:, 0] - qpos) < SAFETY_POS_THRESHOLD)
        near_upper_limit = (
            np.abs(qpos_range[:, 1] - qpos) < SAFETY_POS_THRESHOLD)
        near_pos_limit = np.sum(near_lower_limit | near_upper_limit, axis=1)

        above_vel_limit = np.sum(
            np.abs(obs_dict['claw_qvel']) > SAFETY_VEL_THRESHOLD, axis=1)

        score_dict = collections.OrderedDict((
            ('points', 1.0 - np.minimum(
                obs_dict['object_to_target_angle_distance'], np.pi) / np.pi),
            ('success', reward_dict['bonus_big'] > 0.0),
            ('safety_pos_violation', near_pos_limit),
            ('safety_vel_violation', above_vel_limit),
        ))

        # Add hardware-specific scores.
        if 'claw_current' in obs_dict:
            above_current_limit = (
                np.abs(obs_dict['claw_current']) > SAFETY_CURRENT_THRESHOLD)
            score_dict['safety_current_violation'] = np.sum(
                above_current_limit, axis=1)

        return score_dict

    def _set_target_object_pos(self, target_pos: float):
        """Sets the goal angle to the given position."""
        # Modulo to [-pi, pi].
        target_pos = np.mod(target_pos + np.pi, 2 * np.pi) - np.pi
        self._target_object_pos = np.array(target_pos, dtype=np.float32)

        # Mark the target position in sim.
        self.model.body_quat[self._target_bid] = euler2quat(0, 0, target_pos)

    def render(self, *args, **kwargs):
        if self._camera_config is not None:
            return self.image_service.get_image(*args, **kwargs)

        return super(BaseDClawTurn, self).render(*args, **kwargs)


@configurable(pickleable=True)
class DClawTurnFixed(BaseDClawTurn):
    """Turns the object with a fixed initial and fixed target position."""

    def __init__(self,
                 *args,
                 init_pos_range=(-np.pi, np.pi),
                 target_pos_range=(-np.pi, np.pi),
                 cycle_goals=False,
                 cycle_inits=False,
                 **kwargs):
        self._init_pos_range = init_pos_range
        self._target_pos_range = target_pos_range
        self._cycle_goals = cycle_goals
        self._cycle_inits = cycle_inits
        self._let_alg_set_goals = False
        self._goal_index = 0
        self._init_index = 0

        super().__init__(*args, **kwargs)

    @property
    def num_goals(self):
        if isinstance(self._target_pos_range, (list,)):
            return len(self._target_pos_range)
        else:
            raise Exception("infinite goals")

    def set_goal(self, goal_index):
        """Allow outside algorithms to alter goals."""
        self._goal_index = goal_index
        self._cycle_goals = True
        self._let_alg_set_goals = True

    def _sample_goal(self, obs_dict):
        if isinstance(self._target_pos_range, (list,)):

            if self._cycle_goals:
                if not self._let_alg_set_goals:
                    self._goal_index = (self._goal_index + 1) % self.num_goals
                target_pos = self._target_pos_range[self._goal_index]
            else:
                rand_index = np.random.randint(len(self._target_pos_range))
                target_pos = np.array(self._target_pos_range[rand_index])
        elif isinstance(self._target_pos_range, (tuple,)):
            target_pos = np.random.uniform(
                low=self._target_pos_range[0],
                high=self._target_pos_range[1]
            )
        return target_pos

    def _reset(self):
        if isinstance(self._init_pos_range, (list,)):
            if self._cycle_inits:
                init_index = self._init_index
                self._init_index = (self._init_index + 1) % len(self._init_pos_range)
            else:
                init_index = np.random.randint(len(self._init_pos_range))
            self._initial_object_pos = np.array(self._init_pos_range[init_index])
        elif isinstance(self._init_pos_range, (tuple,)):
            self._initial_object_pos = np.random.uniform(
                low=self._init_pos_range[0],
                high=self._init_pos_range[1]
            )
        self._set_target_object_pos(
            self._sample_goal(self.get_obs_dict()))
        super()._reset()


# @configurable(pickleable=True)
# class DClawTurnRandom(BaseDClawTurn):
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
# class DClawTurnRandomResetSingleGoal(BaseDClawTurn):
#     """Turns the object with a random initial and random target position."""
#     def __init__(self,
#                  *args,
#                  initial_object_pos_range=(-np.pi, np.pi),
#                  camera_config=None,
#                  **kwargs):
#         self._initial_object_pos_range = initial_object_pos_range
#         self._camera_config = camera_config
#         if self._camera_config is not None:
#             self.image_service = get_image_service(**camera_config)
#         return super(DClawTurnRandomResetSingleGoal, self).__init__(
#             *args, **kwargs)

#     def _reset(self):
#         # Initial position is +/- 180 degrees.
#         low, high = self._initial_object_pos_range
#         self._initial_object_pos = self.np_random.uniform(low=low, high=high)
#         # Target position is at 0 degrees.
#         self._set_target_object_pos(0)
#         super()._reset()

#     def render(self, *args, **kwargs):
#         if self._camera_config is not None:
#             return self.image_service.get_image(*args, **kwargs)

#         raise ValueError(args, kwargs)
#         return super(DClawTurnRandomResetSingleGoal, self).render(
#             *args, **kwargs)


# @configurable(pickleable=True)
# class DClawTurnRandomDynamics(DClawTurnRandom):
#     """Turns the object with a random initial and random target position.

#     The dynamics of the simulation are randomized each episode.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._randomizer = SimRandomizer(self.sim_scene, self.np_random)
#         self._dof_indices = (
#             self.robot.get_config('dclaw').qvel_indices.tolist() +
#             self.robot.get_config('object').qvel_indices.tolist())

#     def _reset(self):
#         # Randomize joint dynamics.
#         self._randomizer.randomize_dofs(
#             self._dof_indices,
#             damping_range=(0.005, 0.1),
#             friction_loss_range=(0.001, 0.005),
#         )
#         self._randomizer.randomize_actuators(
#             all_same=True,
#             kp_range=(1, 3),
#         )
#         # Randomize friction on all geoms in the scene.
#         self._randomizer.randomize_geoms(
#             all_same=True,
#             friction_slide_range=(0.8, 1.2),
#             friction_spin_range=(0.003, 0.007),
#             friction_roll_range=(0.00005, 0.00015),
#         )
#         self._randomizer.randomize_bodies(
#             ['mount'],
#             position_perturb_range=(-0.01, 0.01),
#         )
#         self._randomizer.randomize_geoms(
#             ['mount'],
#             color_range=(0.2, 0.9),
#         )
#         self._randomizer.randomize_geoms(
#             parent_body_names=['valve'],
#             color_range=(0.2, 0.9),
#             size_perturb_range=(-0.003, 0.003),
#         )
#         super()._reset()


# @configurable(pickleable=True)
# class DClawTurnImage(DClawTurnFixed):
#     """
#     Observation including the image.
#     """

#     def __init__(self,
#                  image_shape: np.ndarray,
#                  goal_completion_threshold: bool = 0.15,
#                  *args, **kwargs):
#         self._image_shape = image_shape
#         self._goal_completion_threshold = goal_completion_threshold
#         super().__init__(*args, **kwargs)

#     def get_obs_dict(self) -> Dict[str, np.ndarray]:
#         width, height = self._image_shape[:2]
#         obs_dict = super(DClawTurnImage, self).get_obs_dict()
#         image = self.render(mode='rgb_array',
#                             width=width,
#                             height=height,
#                             camera_id=-1).reshape(-1)
#         obs_dict['image'] = ((2.0 / 255.0) * image - 1.0) # Normalize between [-1, 1]
#         angle_dist = obs_dict['object_to_target_angle_dist']
#         obs_dict['is_goal'] = angle_dist < self._goal_completion_threshold
#         return obs_dict

@configurable(pickleable=True)
class DClawTurnResetFree(DClawTurnFixed):
    def __init__(self, reset_fingers=True, hardware=True, **kwargs):
        super().__init__(**kwargs)
        self._reset_fingers = reset_fingers
        self._hardware = hardware
    # def _reset(self):
    #     self._initial_object_pos = np.random.uniform(
    #             low=self._init_angle_range[0], high=self._init_angle_range[1])
    #     print("ANGLE RANGE:", self._init_angle_range, "OBJECT POS:", self._initial_object_pos)
    #     self._set_target_object_pos(
    #             np.random.uniform(low=self._target_angle_range[0],
    #                 high=self._target_angle_range[1]))
    #     super()._reset()

    def reset(self):
        if self._hardware:
            if self._reset_fingers:
                self.robot.set_motors_engaged('dclaw', True)
                self.robot.set_state({'dclaw': RobotState(qpos=DEFAULT_CLAW_RESET_POSE)})
                self.robot.set_motors_engaged('object', False)
            else:
                print("not resetting fingers")
                self.robot.set_motors_engaged('object', False)
            self.robot.reset_time()


        else:
            dclaw_config = self.robot.get_config('dclaw')
            dclaw_control_mode = dclaw_config.control_mode
            dclaw_config.set_control_mode(ControlMode.JOINT_POSITION)
            if self._reset_fingers:
                reset_action = self.robot.normalize_action(
                    {'dclaw': DEFAULT_CLAW_RESET_POSE.copy()})['dclaw']

            for _ in range(15):
                self._step(reset_action)
            dclaw_config.set_control_mode(dclaw_control_mode)

        self._set_target_object_pos(self._sample_goal(self.get_obs_dict()))
        return self._get_obs(self.get_obs_dict())

    # def _sample_goal(self, obs_dict):
    #     return np.pi


# @configurable(pickleable=True)
# class DClawTurnResetFreeSwapGoal(DClawTurnResetFree):
#     def __init__(self,
#                  **kwargs):
#         super().__init__(
#             **kwargs)
#         self._goal_index = 0
#         # self._goals = [(-0.06, -0.08, 0, 0, 0, 0), (-0.06, -0.08, 0, 0, 0, 0)]
#         self._goals = [np.pi/2, -np.pi/2]
#         self.n_goals = len(self._goals)

#     def _sample_goal(self, obs_dict):
#         self._goal_index = (self._goal_index + 1) % self.n_goals
#         return self._goals[self._goal_index]


# @configurable(pickleable=True)
# class DClawTurnResetFreeRandomGoal(DClawTurnResetFree):

#     def _current_goal(self):
#         return self._goal

#     def _sample_goal(self, obs_dict=None):
#         return np.random.uniform(-np.pi, np.pi)


# @configurable(pickleable=True)
# class DClawTurnImageResetFree(DClawTurnImage):
#     """
#     Resets do not move the screw back to its original position.
#     """

#     def _reset(self):
#         # Only reset the target position. Keep the object where it is.
#         self._set_target_object_pos(np.random.uniform(
#             low=self._target_pos_range[0],
#             high=self._target_pos_range[1]))

#     def reset(self):
#         obs_dict = self.get_obs_dict()
#         for _ in range(15):
#             self._step(DEFAULT_CLAW_RESET_POSE)
#         self._reset()
#         return self._get_obs(obs_dict)

class BaseMultiGoalEnv(metaclass=abc.ABCMeta):
    # TODO: Put common abstractions in this class
    pass

@configurable(pickleable=True)
class DClawTurnMultiGoal(DClawTurnFixed):
    """
    :param: goals
        This is a list of the goal angles.

    ...

    :param use_concatenated_goal
        If this is True, the rendered image will contain a sampled goal
        image concatenated by the channel.
        If False, the rendered image will just be the current obs image
        and there will be no `_goal_image` sampled from a goal pool.
    :param swap_goals_upon_completion
        If this is True, the goal will be switched only when it is reached.
        If False, the goal will be switched randomly at every.
    """
    # TODO: Finish the param specification above.

    def __init__(self,
                 goals,
                 *args,
                 goal_image_pools_path: str = None,
                 goal_completion_threshold: float = 0.15,
                 initial_goal_index: int = 0,
                 use_concatenated_goal: bool = False,
                 swap_goals_upon_completion: bool = True,
                 reset_claw: bool = True,
                 reset_free: bool = False,
                 observation_keys = DEFAULT_OBSERVATION_KEYS + ('goal_index',),
                 goal_collection: bool = False,
                 random_goal_sampling: bool = False,
                 one_hot_goal_index: bool = False,
                 random_reset: bool = False,
                 **kwargs):

        if one_hot_goal_index:
            observation_keys += ('one_hot_goal_index', )
        super().__init__(*args, observation_keys=observation_keys, **kwargs)

        self._goals = goals
        if goal_image_pools_path is None:
            assert not use_concatenated_goal, 'Cannot use concatenated' \
               + ' observation images without goal pools'
        else:
            # `goal_image_pools` is an array of dicts, where each
            # index i corresponds to the ith set of goal images.
            with open(goal_image_pools_path, 'rb') as file:
                goal_image_pools = pickle.load(file)
            self._goal_image_pools = goal_image_pools

        self._num_goals = len(goals)
        self._goal_index = initial_goal_index
        assert self._goal_index >= 0 and self._goal_index < self._num_goals, \
           "Initial goal cannot be outside the range 0-{}".format(self._num_goals - 1)

        # Initialize goal params
        self._goal_completion_threshold = goal_completion_threshold
        self._use_concatenated_goal = use_concatenated_goal
        self._swap_goals_upon_completion = swap_goals_upon_completion
        if self._use_concatenated_goal:
            self._goal_image = self.sample_goal_image()

        self._reset_claw = reset_claw
        self._reset_free = reset_free
        # reset to the _init_object_pos_range on the first reset
        self._initial_reset = False
        self._goal_collection = goal_collection
        self._random_goal_sampling = random_goal_sampling
        self._one_hot_goal_index = one_hot_goal_index
        self._random_reset = random_reset
        self._reset()

        self._num_goal_switches = 0

    def get_obs_dict(self):
        obs_dict = super().get_obs_dict()
        obs_dict['num_goal_switches'] = np.array([self._num_goal_switches])
        if self._one_hot_goal_index:
            one_hot = np.zeros(self._num_goals).astype(np.float32)
            one_hot[self._goal_index] = 1
            obs_dict['one_hot_goal_index'] = one_hot
        goal_index_obs = np.array([self._goal_index])
        obs_dict['goal_index'] = goal_index_obs
        return obs_dict

    def render(self, mode='rgb_array', **kwargs):
        if mode == 'human':
            return super().render(mode=mode, **kwargs)
        elif mode == 'rgb_array':
            img_obs = super().render(
                    mode=mode,
                    **kwargs)
            if self._use_concatenated_goal:
                # Concatenated by the channels.
                img_obs = np.concatenate([normalized, self._goal_image], axis=2)
            return img_obs
        else:
            raise NotImplementedError

    def _reset(self):
        if self._goal_collection:
            self._set_target_object_pos(self._goals[self._goal_index])
            super()._reset()
        elif self._reset_free and self._initial_reset:
            self._set_target_object_pos(self._goals[self._goal_index])
        else:
            self._initial_reset = True
            # If multigoal with resets, change the init
            target = self._goals[self._goal_index]
            if self._random_reset:
                # Sample uniformly for resets
                init = np.random.uniform(-np.pi, np.pi)
            else:
                # Sample randomly from the other goals for reset pos
                other_goal_inds = [
                    i for i in range(self._num_goals)
                    if i != self._goal_index
                ]
                init = np.random.choice(other_goal_inds)
            self._init_object_pos_range = (init, init)
            self._target_pos_range = (target, target)
            super()._reset()

    def reset(self):
        obs_dict = self.get_obs_dict()
        if self._reset_claw:
            for _ in range(15):
                self._step(DEFAULT_CLAW_RESET_POSE)
        # Check if the goal has been completed heuristically.
        object_target_angle_dist = obs_dict['object_to_target_angle_dist']
        if self._swap_goals_upon_completion:
            if object_target_angle_dist < self._goal_completion_threshold:
                self.switch_goal()
            # else:
            #     self.sample_goal_image()
        else:
            # Sample new goal at every reset if multigoal with resets.
            self.switch_goal(random=self._random_goal_sampling)
        self._reset()
        return self._get_obs(obs_dict)

    """
    def _get_obs(self, obs_dict=None):
        obs_dict = self.get_obs_dict()
        img_obs = obs_dict['image']
        if self._use_concatenated_goal:
            img_obs = np.concatenate([img_obs, self._goal_image])
        return img_obs
    """

    def set_goal(self):
        self._set_target_object_pos(self._goals[self._goal_index])
        if self._use_concatenated_goal:
            self._goal_image = self.sample_goal_image()

    def switch_goal(self, random=False):
        if random:
            # Pick randomly out of the other goals
            other_goal_inds = [
                i for i in range(self._num_goals)
                if i != self._goal_index
            ]
            self._goal_index = np.random.choice(other_goal_inds)
        else:
            # Cycle to the next goal
            self._goal_index = np.mod(self._goal_index + 1, self._num_goals)
        self._num_goal_switches += 1
        self.set_goal()

    """
    Goal example pools functions
    """

    def sample_goal_image(self):
        # Get the pool of goal images from the dictionary at the correct goal index.
        goal_images = self._goal_image_pools[self._goal_index]['image_desired_goal']
        rand_img_idx = np.random.randint(0, goal_images.shape[0])
        return goal_images[rand_img_idx]

@configurable(pickleable=True)
class DClawTurnMultiGoalResetFree(DClawTurnMultiGoal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reset_free=True, **kwargs)
