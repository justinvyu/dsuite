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
from typing import Any, Dict, Optional, Sequence, Tuple

import gym
import numpy as np
from transforms3d.euler import euler2mat, mat2euler

from dsuite.components.robot import DynamixelRobotComponent, RobotState
from dsuite.components.tracking import VrTrackerComponent, TrackerState
from dsuite.dkitty.config import (
    DEFAULT_DKITTY_CALIBRATION_MAP, DKITTY_SIM_CONFIG, DKITTY_HARDWARE_CONFIG,
    TRACKER_SIM_CONFIG, TRACKER_HARDWARE_CONFIG)
from dsuite.dkitty import scripted_reset
from dsuite.robot_env import make_box_space, RobotEnv

# The position offset for tracking in hardware.
KITTY_HW_TRACKER_OFFSET = np.array([0, 0, 0.35])


class BaseDKittyEnv(RobotEnv, metaclass=abc.ABCMeta):
    """Base environment for all DKitty robot tasks."""

    @classmethod
    def get_robot_config(cls,
                         device_path: Optional[str] = None) -> Dict[str, Any]:
        """Returns the robot configuration for the given device path."""
        if device_path is not None:
            config = DKITTY_HARDWARE_CONFIG.copy()
            config['device_path'] = device_path
            scripted_reset.add_groups_for_reset(config['groups'])
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

    def __init__(self,
                 *args,
                 robot_config: Dict[str, Any],
                 tracker_config: Dict[str, Any],
                 manual_reset: bool = False,
                 **kwargs):
        """Initializes the environment.

        Args:
            robot_config: A dictionary of keyword arguments to pass to
                RobotComponent.
            tracker_config: A dictionary of keyword arguments to pass to
                TrackerComponent.
            manual_reset: If True, waits for the user to reset the robot
                instead of performing the automatic reset procedure.
        """
        super().__init__(*args, **kwargs)
        self.robot = self._add_component(**robot_config)
        self.tracker = self._add_component(**tracker_config)
        self.manual_reset = manual_reset

        # Disable the constraint solver in hardware so that mimicked positions
        # do not participate in contact calculations.
        if self.has_hardware_robot:
            self.sim_scene.disable_option(constraint_solver=True)

    @property
    def has_hardware_robot(self) -> bool:
        """Returns true if the environment is using a hardware robot."""
        return isinstance(self.robot, DynamixelRobotComponent)

    @property
    def has_hardware_tracker(self) -> bool:
        """Returns true if the environment is using a hardware tracker."""
        return isinstance(self.tracker, VrTrackerComponent)

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

        # For simulation, simply set the state.
        if not isinstance(self.robot, DynamixelRobotComponent):
            self.robot.set_state({
                'root': RobotState(qpos=root_pos, qvel=root_vel),
                'dkitty': RobotState(qpos=kitty_pos, qvel=kitty_vel),
            })
            return

        # Perform the scripted reset if we're not doing manual resets.
        if not self.manual_reset:
            scripted_reset.reset_standup(self.robot)

        # Move to the desired position.
        self.robot.set_state({
            'dkitty': RobotState(qpos=kitty_pos, qvel=kitty_vel),
        })
        if self.manual_reset:
            # Prompt the user to start the episode.
            input('Press Enter to start the episode...')

        # Reset the hardware tracking position to consider the current world
        # position of the D'Kitty as the desired reset position.
        if self.has_hardware_tracker:
            self.tracker.set_state({
                'torso': TrackerState(
                    pos=root_pos[:3] + KITTY_HW_TRACKER_OFFSET,
                    rot=euler2mat(*root_pos[3:6]),
                )
            },)
        self.robot.reset_time()

    def _get_root_qpos_qvel(
            self,
            root_robot_state: RobotState,
            torso_tracker_state: TrackerState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the root position and velocity of the robot.

        This is needed because we use a free joint to track the D'Kitty position
        for simulation, but we use a site tracker for hardware.
        """
        if self.has_hardware_tracker:
            # Use hardware tracking as the root position and mimic back to sim.
            root_qpos = np.concatenate([
                torso_tracker_state.pos - KITTY_HW_TRACKER_OFFSET,
                mat2euler(torso_tracker_state.rot),
            ])
            self.data.qpos[:6] = root_qpos
            # TODO(michaelahn): Calculate angular velocity from tracking.
            root_qvel = np.zeros(6)
        else:
            root_qpos = root_robot_state.qpos
            root_qvel = root_robot_state.qvel
        return root_qpos, root_qvel
