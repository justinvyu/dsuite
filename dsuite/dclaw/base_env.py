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

"""Shared logic for all DClaw environments."""

import abc
from typing import Any, Dict, Optional, Sequence, Tuple

import gym
import numpy as np

from dsuite.controllers.robot import DynamixelRobotController, RobotState
from dsuite.dclaw.config import (
    DCLAW_SIM_CONFIG, DCLAW_HARDWARE_CONFIG, DCLAW_OBJECT_SIM_CONFIG,
    DCLAW_OBJECT_HARDWARE_CONFIG, DCLAW_OBJECT_GUIDE_HARDWARE_CONFIG)
from dsuite.robot_env import make_box_space, RobotEnv

DEFAULT_CLAW_RESET_POSE = np.array(
    [0, -np.pi / 3, np.pi / 3] * 3, dtype=np.float32)


class BaseDClawEnv(RobotEnv, metaclass=abc.ABCMeta):
    """Base environment for all DClaw robot tasks."""

    @classmethod
    def get_config_for_device(
            cls, device_path: Optional[str] = None) -> Dict[str, Any]:
        """Returns the configuration for the given device path."""
        if device_path is not None:
            config = DCLAW_HARDWARE_CONFIG.copy()
            config['device_path'] = device_path
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
        self.robot = self._add_controller(
            random_state=self.np_random, **robot_config)

    def initialize_action_space(self) -> gym.Space:
        """Returns the observation space to use for this environment."""
        qpos_indices = self.robot.get_config('dclaw').qpos_indices
        return make_box_space(-1.0, 1.0, shape=(qpos_indices.size,))


class BaseDClawObjectEnv(BaseDClawEnv, metaclass=abc.ABCMeta):
    """Base environment for all DClaw robot tasks with objects."""

    @classmethod
    def get_config_for_device(
            cls, device_path: Optional[str] = None) -> Dict[str, Any]:
        """Returns the configuration for the given device path."""
        if device_path is not None:
            if 'dlantern' in device_path:
                config = DCLAW_OBJECT_GUIDE_HARDWARE_CONFIG.copy()
            else:
                config = DCLAW_OBJECT_HARDWARE_CONFIG.copy()
            config['device_path'] = device_path
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
        self._object_bid = self.model.body_name2id('valve')

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
            claw_pos: The joint positions to move the claw joints to.
            claw_vel: The joint velocities to set the claw joints to. This is
                ignored on hardware.
            object_pos: The joint position to move the object joints to.
            object_vel: The joint position to move the object velocity to. This
                is ignored on hardware.
            guide_pos: The joint position to move the guide motor to. This is
                only used in hardware.
        """
        # Set defaults if parameters are not given.
        if claw_pos is None:
            claw_pos = DEFAULT_CLAW_RESET_POSE
        if claw_vel is None:
            claw_vel = np.zeros(
                self.robot.get_config('dclaw').qvel_indices.size)
        if object_pos is None:
            object_pos = np.zeros(
                self.robot.get_config('object').qpos_indices.size)
        if object_vel is None:
            object_vel = np.zeros(
                self.robot.get_config('object').qvel_indices.size)

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

    def _randomize_claw_sim(
            self,
            mount_position_range: Tuple[float, float] = (-0.01, 0.01),
            mount_color_range: Tuple[float, float] = (0.2, 0.9),
            damping_range: Tuple[float, float] = (0.1, 0.5),
            friction_loss_range: Tuple[float, float] = (0.001, 0.005),
    ):
        """Randomizes the DClaw simulation.

        Args:
            mount_position_range: The range to randomize the mount position.
            mount_color_range: The range to randomize the color of the mount.
            damping_range: The range to randomize the damping per DoF.
            friction_loss_range: The range to randomize the friction per DoF.
        """
        self.model.body_pos[self._mount_bid] = (
            self._nominal_model.body_pos[self._mount_bid] +
            self.np_random.uniform(*mount_position_range, size=3))

        self.model.geom_rgba[self._mount_gid][:3] = self.np_random.uniform(
            *mount_color_range, size=3)

        claw_dof_indices = self.robot.get_config('dclaw').qvel_indices
        self.model.dof_damping[claw_dof_indices] = self.np_random.uniform(
            *damping_range)
        self.model.dof_frictionloss[claw_dof_indices] = self.np_random.uniform(
            *friction_loss_range)

    def _randomize_object_sim(
            self,
            size_range: Tuple[float, float] = (-0.003, 0.003),
            color_range: Tuple[float, float] = (0.2, 0.9),
            damping_range: Tuple[float, float] = (0.1, 0.5),
            friction_loss_range: Tuple[float, float] = (0.001, 0.005),
    ):
        """Randomizes the object in the simulation.

        Args:
            mount_position_range: The range to randomize the sizes of the object
                components.
            mount_color_range: The range to randomize the color of the object
                components.
            damping_range: The range to randomize the damping per DoF.
            friction_loss_range: The range to randomize the friction per DoF.
        """
        # Randomize the components of the object.
        for i in range(self.model.body_geomnum[self._object_bid]):
            geom_id = self.model.body_geomadr[self._object_bid] + i
            self.model.geom_rgba[geom_id][:3] = self.np_random.uniform(
                *color_range, size=3)

            self.model.geom_size[geom_id] = (
                self._nominal_model.geom_size[geom_id] + self.np_random.uniform(
                    *size_range, size=3))

        obj_dof_indices = self.robot.get_config('object').qvel_indices
        self.model.dof_damping[obj_dof_indices] = self.np_random.uniform(
            *damping_range)
        self.model.dof_frictionloss[obj_dof_indices] = self.np_random.uniform(
            *friction_loss_range)
