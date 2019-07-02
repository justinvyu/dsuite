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

"""Shared logic for all DKitty environments."""

import abc
from typing import Any, Dict, Optional, Sequence

import gym
import numpy as np

from dsuite.controllers.robot import DynamixelRobotController, RobotState
from dsuite.controllers.tracking import VrTrackerController
from dsuite.dkitty.config import (
    DEFAULT_DKITTY_CALIBRATION_MAP, DKITTY_SIM_CONFIG, DKITTY_HARDWARE_CONFIG,
    TRACKER_SIM_CONFIG, TRACKER_HARDWARE_CONFIG)
from dsuite.dkitty import hardware_reset
from dsuite.robot_env import make_box_space, RobotEnv


class BaseDKittyEnv(RobotEnv, metaclass=abc.ABCMeta):
    """Base environment for all DKitty robot tasks."""

    @classmethod
    def get_robot_config(cls,
                         device_path: Optional[str] = None) -> Dict[str, Any]:
        """Returns the robot configuration for the given device path."""
        if device_path is not None:
            config = DKITTY_HARDWARE_CONFIG.copy()
            config['device_path'] = device_path
            hardware_reset.add_groups_for_reset(config['groups'])
            # Calibrate the configuration groups.
            DEFAULT_DKITTY_CALIBRATION_MAP.update_group_configs(config)
        else:
            config = DKITTY_SIM_CONFIG
        return config

    @classmethod
    def get_tracker_config(cls, **device_identifiers) -> Dict[str, Any]:
        """Returns the robot configuration for the given device path."""
        # Filter out None entries.
        device_identifiers = {
            name: device_id
            for name, device_id in device_identifiers.items()
            if device_id is not None
        }
        if device_identifiers:
            config = TRACKER_HARDWARE_CONFIG.copy()
            for name, device_id in device_identifiers.items():
                config['groups'][name]['device_identifier'] = device_id
        else:
            config = TRACKER_SIM_CONFIG
        return config

    def __init__(self, *args, robot_config: Dict[str, Any],
                 tracker_config: Dict[str, Any], **kwargs):
        """Initializes the environment.

        Args:
            robot_config: A dictionary of keyword arguments to pass to
                RobotController.
            tracker_config: A dictionary of keyword arguments to pass to
                TrackerController.
        """
        super().__init__(*args, **kwargs)
        self.robot = self._add_controller(**robot_config)
        self.tracker = self._add_controller(**tracker_config)

    @property
    def has_hardware_robot(self) -> bool:
        """Returns true if the environment is using a hardware robot."""
        return isinstance(self.robot, DynamixelRobotController)

    @property
    def has_hardware_tracker(self) -> bool:
        """Returns true if the environment is using a hardware tracker."""
        return isinstance(self.tracker, VrTrackerController)

    def initialize_action_space(self) -> gym.Space:
        """Returns the observation space to use for this environment."""
        qpos_indices = self.robot.get_config('dkitty').qpos_indices
        return make_box_space(-1.0, 1.0, shape=(qpos_indices.size,))

    def _reset_dkitty_standing(self,
                               root_pos: Optional[Sequence[float]] = None,
                               root_vel: Optional[Sequence[float]] = None,
                               kitty_pos: Optional[Sequence[float]] = None,
                               kitty_vel: Optional[Sequence[float]] = None):
        """Resets the D'Kitty to a standing position.

        Args:
            root_pos: The root position (x, y, z, rx, ry, rz) of the
                D'Kitty. (x, y, z) are in meters, rx, ry, rz are in radians.
            root_vel: The root velocity of the D'Kitty.
            kitty_pos: The joint positions (radians).
            kitty_vel: The joint velocities (radians/second).
        """
        # Set defaults if parameters are not given.
        root_init_state, kitty_init_state = self.robot.get_initial_state(
            ['root', 'dkitty'])
        root_pos = (
            root_init_state.qpos if root_pos is None else np.asarray(root_pos))
        root_vel = (
            root_init_state.qvel if root_vel is None else np.asarray(root_vel))
        kitty_pos = (
            kitty_init_state.qpos
            if kitty_pos is None else np.asarray(kitty_pos))
        kitty_vel = (
            kitty_init_state.qvel
            if kitty_vel is None else np.asarray(kitty_vel))

        if not isinstance(self.robot, DynamixelRobotController):
            self.robot.set_state({
                'root': RobotState(qpos=root_pos, qvel=root_vel),
                'dkitty': RobotState(qpos=kitty_pos, qvel=kitty_vel),
            })
        else:
            # Reset to a standing position.
            hardware_reset.reset_standup(self.robot)

            # Move to the desired position.
            self.robot.set_state({
                'dkitty': RobotState(qpos=kitty_pos, qvel=kitty_vel),
            })
            self.robot.reset_time()
