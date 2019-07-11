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

"""Shared logic for all DClaw environments."""

import abc
from typing import Any, Dict, Optional, Sequence

import gym
import numpy as np

from dsuite.controllers.robot import DynamixelRobotController, RobotState
from dsuite.dclaw.config import (
    DCLAW_SIM_CONFIG, DCLAW_HARDWARE_CONFIG, DCLAW_OBJECT_SIM_CONFIG,
    DCLAW_FREE_OBJECT_SIM_CONFIG,
    DCLAW_OBJECT_HARDWARE_CONFIG,
    DCLAW_OBJECT_GUIDE_HARDWARE_CONFIG,
    DEFAULT_DCLAW_CALIBRATION_MAP)
from dsuite.robot_env import make_box_space, RobotEnv
DEFAULT_CLAW_RESET_POSE = np.array([0, -np.pi/3, np.pi/3]*3)

class BaseDClawEnv(RobotEnv, metaclass=abc.ABCMeta):
    """Base environment for all DClaw robot tasks."""

    @classmethod
    def get_config_for_device(
            cls, device_path: Optional[str] = None) -> Dict[str, Any]:
        """Returns the configuration for the given device path."""
        if device_path is not None:
            config = DCLAW_HARDWARE_CONFIG.copy()
            config['device_path'] = device_path
            # Calibrate the configuration groups.
            DEFAULT_DCLAW_CALIBRATION_MAP.update_group_configs(config)
        else:
            config = DCLAW_SIM_CONFIG
        return config

    def __init__(self, *args, robot_config: Dict[str, Any], **kwargs):
        """Initializes the environment.

        Args:
            robot_config: A dictionary of keyword arguments to pass to
                RobotController.
        """
        super().__init__(*args, **kwargs)
        self.robot = self._add_controller(**robot_config)

    def initialize_action_space(self) -> gym.Space:
        """Returns the observation space to use for this environment."""
        qpos_indices = self.robot.get_config('dclaw').qpos_indices
        return make_box_space(-1.0, 1.0, shape=(qpos_indices.size,))


class BaseDClawObjectEnv(BaseDClawEnv, metaclass=abc.ABCMeta):
    """Base environment for all DClaw robot tasks with objects."""

    @classmethod
    def get_config_for_device(
            cls,
            device_path: Optional[str] = None,
            free_object: bool = False,
            free_claw: bool = False,
    ) -> Dict[str, Any]:
        """Returns the configuration for the given device path."""
        if device_path is not None:
            if 'dlantern' in device_path:
                config = DCLAW_OBJECT_GUIDE_HARDWARE_CONFIG.copy()
            else:
                config = DCLAW_OBJECT_HARDWARE_CONFIG.copy()
            config['device_path'] = device_path
            # Calibrate the configuration groups.
            DEFAULT_DCLAW_CALIBRATION_MAP.update_group_configs(config)
        else:
            if free_object:
                if free_claw:
                    config = FREE_DCLAW_FREE_OBJECT_SIM_CONFIG
                else:
                    config = DCLAW_FREE_OBJECT_SIM_CONFIG
            else:
                config = DCLAW_OBJECT_SIM_CONFIG
        return config

    def __init__(self, **kwargs):
        """Initializes the environment."""
        super().__init__(**kwargs)

        # Make a copy of the model to store initial values.
        self._nominal_model = self.sim_scene.copy_model()

        # Get handles to commonly referenced elements.
        self._mount_bid = self.model.body_name2id('mount')
        self._mount_gid = self.model.geom_name2id('mount')
        self._object_bid = self.model.body_name2id('object')

    def _reset_dclaw_and_object(
            self,
            claw_pos: Optional[Sequence[float]] = None,
            claw_vel: Optional[Sequence[float]] = None,
            object_pos: Optional[Sequence[float]] = None,
            object_vel: Optional[Sequence[float]] = None,
            guide_pos: Optional[Sequence[float]] = None,
    ):
        """Reset procedure for DClaw robots that manipulate objects.

        Args:
            claw_pos: The joint positions for the claw (radians).
            claw_vel: The joint velocities for the claw (radians/second). This
                is ignored on hardware.
            object_pos: The joint position for the object (radians).
            object_vel: The joint velocity for the object (radians/second). This
                is ignored on hardware.
            guide_pos: The joint position for the guide motor (radians). The
                guide motor is optional for marking the desired position.
        """
        # Set defaults if parameters are not given.
        claw_init_state, object_init_state = self.robot.get_initial_state(
            ['dclaw', 'object'])
        claw_pos = (
            claw_init_state.qpos if claw_pos is None else np.asarray(claw_pos))
        claw_vel = (
            claw_init_state.qvel if claw_vel is None else np.asarray(claw_vel))
        object_pos = (
            object_init_state.qpos
            if object_pos is None else np.atleast_1d(object_pos))
        object_vel = (
            object_init_state.qvel
            if object_vel is None else np.atleast_1d(object_vel))

        if not isinstance(self.robot, DynamixelRobotController):
            self.robot.set_state({
                'dclaw': RobotState(qpos=claw_pos, qvel=claw_vel),
                'object': RobotState(qpos=object_pos, qvel=object_vel),
            })
        else:
            # Multi-stage reset procedure for hardware.
            # Initially attempt to reset a subset of the motors.
            self.robot.set_motors_engaged('disable_in_reset', False)
            self.robot.set_state({'dclaw': RobotState(qpos=claw_pos)},
                                 block=False)

            self.robot.set_motors_engaged('object', True)
            self.robot.set_state({
                'object': RobotState(qpos=object_pos),
                'guide': RobotState(qpos=guide_pos)
            })

            self.robot.set_motors_engaged('dclaw', True)
            self.robot.set_state({'dclaw': RobotState(qpos=claw_pos)})

            # Start the episode with the object disengaged.
            self.robot.set_motors_engaged('object', False)
            self.robot.reset_time()
