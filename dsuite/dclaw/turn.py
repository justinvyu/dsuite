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

from dsuite.dclaw.base_env import (BaseDClawObjectEnv,
                                   DEFAULT_CLAW_RESET_POSE)
from dsuite.utils.configurable import configurable
from dsuite.utils.resources import get_asset_path
from dsuite.utils.circle_math import circle_distance

# The observation keys that are concatenated as the environment observation.
DEFAULT_OBSERVATION_KEYS = (
    'claw_qpos',
    'object_angle_cos',
    'object_angle_sin',
    'last_action',
    'object_to_target_angle_dist',
)

DCLAW3_ASSET_PATH = 'dsuite/dclaw/assets/dclaw3xh_valve3_v0.xml'


class BaseDClawTurn(BaseDClawObjectEnv, metaclass=abc.ABCMeta):
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
            robot_config=self.get_config_for_device(device_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)

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
            object_pos=np.atleast_1d(self._initial_object_pos),
            object_vel=np.atleast_1d(self._initial_object_vel),
            guide_pos=np.atleast_1d(self._target_object_pos))

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({
            'dclaw': action,
            'guide': np.atleast_1d(self._target_object_pos),
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
        object_angle = object_state.qpos
        object_to_target_angle_dist = circle_distance(
            self._target_object_pos, object_angle)
        # target_error = np.mod(target_error + np.pi, 2 * np.pi) - np.pi

        return collections.OrderedDict((
            ('claw_qpos', claw_state.qpos),
            ('claw_qvel', claw_state.qvel),
            ('object_angle_cos', np.cos(object_state.qpos)),
            ('object_angle_sin', np.sin(object_state.qpos)),
            ('object_rotational_vel', object_state.qvel),
            ('last_action', self._last_action),
            # ('target_error', target_error),
            ('object_to_target_angle_dist', object_to_target_angle_dist),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        object_to_target_angle_dist = obs_dict['object_to_target_angle_dist']
        claw_vel = obs_dict['claw_qvel']

        reward_dict = collections.OrderedDict((
            # Penalty for distance away from goal.
            ('object_to_target_angle_dist_cost', -5 * object_to_target_angle_dist),
            # Penalty for difference with nomimal pose.
            ('pose_diff_cost',
             -1 * np.linalg.norm(obs_dict['claw_qpos'] - self._desired_claw_pos)
            ),
            # Penality for high velocities.
            ('joint_vel_cost', -1 * np.linalg.norm(claw_vel[claw_vel >= 0.5])),

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

        return collections.OrderedDict((
            ('points', 1.0 - np.minimum(
                obs_dict['object_to_target_angle_dist'], np.pi) / np.pi),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))

    def _set_target_object_pos(self, target_pos: float):
        """Sets the goal angle to the given position."""
        # Modulo to [-pi, pi].
        target_pos = np.mod(target_pos + np.pi, 2 * np.pi) - np.pi
        self._target_object_pos = np.array(target_pos, dtype=np.float32)

        # Mark the target position in sim.
        self.model.body_quat[self._target_bid] = euler2quat(0, 0, target_pos)


@configurable(pickleable=True)
class DClawTurnFixed(BaseDClawTurn):
    """Turns the object with a fixed initial and fixed target position."""

    def __init__(self,
                 *args,
                 init_object_pos_range=(0., 0.),
                 target_pos_range=(np.pi, np.pi),
                 **kwargs):
        self._init_object_pos_range = init_object_pos_range
        self._target_pos_range = target_pos_range
        super().__init__(*args, **kwargs)

    def _reset(self):
        self._initial_object_pos = np.random.uniform(
                low=self._init_object_pos_range[0],
                high=self._init_object_pos_range[1])
        self._set_target_object_pos(np.random.uniform(
            low=self._target_pos_range[0],
            high=self._target_pos_range[1]))

        super()._reset()


@configurable(pickleable=True)
class DClawTurnRandom(BaseDClawTurn):
    """Turns the object with a random initial and random target position."""

    def _reset(self):
        # Initial position is +/- 60 degrees.
        self._initial_object_pos = self.np_random.uniform(
            low=-np.pi / 3, high=np.pi / 3)
        # Target position is 180 +/- 60 degrees.
        self._set_target_object_pos(
            np.pi + self.np_random.uniform(low=-np.pi / 3, high=np.pi / 3))
        super()._reset()


@configurable(pickleable=True)
class DClawTurnRandomResetSingleGoal(BaseDClawTurn):
    """Turns the object with a random initial and random target position."""
    def __init__(self,
                 *args,
                 initial_object_pos_range=(-np.pi, np.pi),
                 **kwargs):
        self._initial_object_pos_range = initial_object_pos_range
        return super(DClawTurnRandomResetSingleGoal, self).__init__(
            *args, **kwargs)

    def _reset(self):
        # Initial position is +/- 180 degrees.
        low, high = self._initial_object_pos_range
        self._initial_object_pos = self.np_random.uniform(low=low, high=high)
        # Target position is at 0 degrees.
        self._set_target_object_pos(0)
        super()._reset()


@configurable(pickleable=True)
class DClawTurnRandomDynamics(DClawTurnRandom):
    """Turns the object with a random initial and random target position.

    The dynamics of the simulation are randomized each episode.
    """

    def _reset(self):
        self._randomize_claw_sim()
        self._randomize_object_sim()
        super()._reset()

@configurable(pickleable=True)
class DClawTurnImage(DClawTurnFixed):
    """
    Observation including the image.
    """

    def __init__(self, 
                 image_shape: np.ndarray, 
                 *args, 
                 **kwargs):
        self.image_shape = image_shape
        super().__init__(*args, **kwargs)

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        width, height = self.image_shape[:2]
        obs = super(DClawTurnImage, self).get_obs_dict()
        image = self.render(mode='rgb_array', \
                            width=width,
                            height=height,
                            camera_id=-1).reshape(-1)
        obs['image'] = ((2.0 / 255.0) * image - 1.0) # Normalize between [-1, 1]
        return obs

@configurable(pickleable=True)
class DClawTurnResetFree(DClawTurnFixed):
    def _reset(self):
        self._set_target_object_pos(np.random.uniform(
            low=self._target_pos_range[0],
            high=self._target_pos_range[1]))

    def reset(self):
        obs_dict = self.get_obs_dict()
        for _ in range(15):
            self._step(DEFAULT_CLAW_RESET_POSE)
        self._reset()
        return self._get_obs(obs_dict)

@configurable(pickleable=True)
class DClawTurnImageResetFree(DClawTurnImage):
    """
    Resets do not move the screw back to its original position.
    """

    def _reset(self):
        # Only reset the target position. Keep the object where it is.
        self._set_target_object_pos(np.random.uniform(
            low=self._target_pos_range[0],
            high=self._target_pos_range[1]))

    def reset(self):
        obs_dict = self.get_obs_dict()
        for _ in range(15):
            self._step(DEFAULT_CLAW_RESET_POSE)
        self._reset()
        return self._get_obs(obs_dict)

@configurable(pickleable=True)
class DClawTurnImageMultiGoal(DClawTurnFixed):
    def __init__(self,
                 goal_image_pools,
                 *args,               
                 goal_completion_threshold: float = 0.15,
                 initial_goal_index: int = 1,
                 use_concatenated_goal: bool = True,
                 swap_goals_upon_completion: bool = True,
                 reset_claw: bool = True,
                 reset_free: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # `goal_image_pools` is an array of dicts, where each
        # index i corresponds to the ith set of goal images.
        self._goal_image_pools = goal_image_pools
        self.num_goals = len(goal_image_pools)
        
        self._goal_index = initial_goal_index
        assert self._goal_index >= 0 and self._goal_index < self.num_goals, \
            "Initial goal cannot be outside the range 0-{}".format(self.num_goals - 1)

        # Initialize goal params
        self._goal_image = self.sample_goal_image()
        self._goals = [np.pi, 0.]
        self._goal_completion_threshold = goal_completion_threshold
        self._use_concatenated_goal = use_concatenated_goal
        self._swap_goals_upon_completion = swap_goals_upon_completion

        self._reset_claw = reset_claw
        self._reset_free = reset_free
        self._reset()

        self._num_goal_switches = 0

    def get_obs_dict(self):
        obs_dict = super().get_obs_dict()

         # Log some other metrics with multigoal
        obs_dict['num_goal_switches'] = self._num_goal_switches
        obs_dict['current_goal'] = self._goal_index

        return obs_dict

    def _reset(self):
        if self._reset_free:
            self._set_target_object_pos(self._goals[self._goal_index])
        else:
            # If multigoal with resets, change the init 
            target = self._goals[self._goal_index]
            init = self._goals[1 - self._goal_index]
            self._init_object_pos_range = (init, init)
            self._target_pos_range = (target, target)
            super()._reset()

    def render(self, mode='human', **kwargs):
        if mode == 'human':
            return super().render(mode=mode, **kwargs)
        elif mode == 'rgb_array':
            img_obs = super().render(
                    mode=mode,
                    **kwargs)
            print(img_obs.shape)
            # TODO: Move normalization into PixelObservationWrapper
            normalized = ((2.0 / 255.0) * img_obs - 1.0)
            # Concatenated by the channels.
            concatenated = np.concatenate([normalized, self._goal_image], axis=2)
            return concatenated
        else:
            raise NotImplementedError

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
        else:
            # Sample new goal at every reset if multigoal with resets.
            self.switch_goal(random=True)
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
        self._goal_image = self.sample_goal_image()

    def switch_goal(self, random=False):
        # For now, just increment by one and mod by # of goals.
        if random:
            self._goal_index = np.random.randint(low=0, high=self.num_goals)
        else:
            self._goal_index = np.mod(self._goal_index + 1, self.num_goals)
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
class DClawTurnImageMultiGoalResetFree(DClawTurnImageMultiGoal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, reset_free=True, **kwargs)

