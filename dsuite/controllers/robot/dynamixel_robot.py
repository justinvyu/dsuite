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

"""Implementation of HardwareRobotController using the DynamixelSDK."""

from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from dsuite.controllers.robot import HardwareRobotController, RobotState
from dsuite.controllers.robot.dynamixel_client import DynamixelClient
from dsuite.controllers.robot.hardware_robot import HardwareRobotGroupConfig
from dsuite.controllers.robot.config import ControlMode


class DynamixelGroupConfig(HardwareRobotGroupConfig):
    """Stores group configuration for a DynamixelRobotController."""

    def __init__(self,
                 *args,
                 motor_ids: Optional[Iterable[int]] = None,
                 **kwargs):
        """Initializes a new configuration for a HardwareRobotController group.

        Args:
            motor_ids: The Dynamixel motor identifiers to associate with this
                group.
        """
        super().__init__(*args, **kwargs)

        self.motor_ids = None
        if motor_ids is not None:
            self.motor_ids = np.array(motor_ids, dtype=int)

            if self.calib_scale is not None:
                assert self.motor_ids.shape == self.calib_scale.shape
            if self.calib_offset is not None:
                assert self.motor_ids.shape == self.calib_offset.shape

        self.motor_id_indices = None

    def set_all_motor_ids(self, all_motor_ids: Sequence[int]):
        """Sets this group's motor ID mask from the given total list of IDs."""
        assert np.all(np.diff(all_motor_ids) > 0), \
            'all_motor_ids must be sorted.'
        assert np.all(np.isin(self.motor_ids, all_motor_ids))
        self.motor_id_indices = np.searchsorted(all_motor_ids, self.motor_ids)


class DynamixelRobotController(HardwareRobotController):
    """Controller for hardware robots using Dynamixel motors."""

    # Cache dynamixel_py instances by device path.
    DEVICE_CLIENTS = {}

    def __init__(self, *args, groups: Dict[str, Dict], device_path: str,
                 **kwargs):
        """Initializes the controller.

        Args:
            groups: Group configurations for reading/writing state.
            device_path: The path to the Dynamixel device to open.
        """
        self._combined_motor_ids = set()
        super().__init__(*args, groups=groups, **kwargs)
        self._all_motor_ids = np.array(
            sorted(self._combined_motor_ids), dtype=int)

        for group_config in self.groups.values():
            if group_config.motor_ids is not None:
                group_config.set_all_motor_ids(self._all_motor_ids)

        if device_path not in self.DEVICE_CLIENTS:
            hardware = DynamixelClient(
                self._all_motor_ids, port=device_path, lazy_connect=True)
            self.DEVICE_CLIENTS[device_path] = hardware

        self._hardware = self.DEVICE_CLIENTS[device_path]

    def _process_group(self, **config_kwargs) -> DynamixelGroupConfig:
        """Processes the configuration for a group."""
        config = DynamixelGroupConfig(self.sim_scene, **config_kwargs)

        if config.motor_ids is not None:
            self._combined_motor_ids.update(config.motor_ids)

        return config

    def set_motors_engaged(self, groups: Union[str, Sequence[str], None],
                           engaged: bool):
        """Enables the motors in the given group name."""
        # Interpret None as all motors.
        if groups is None:
            self._hardware.set_torque_enabled(self._all_motor_ids, engaged)
            return

        if isinstance(groups, str):
            group_configs = [self.get_config(groups)]
        else:
            group_configs = [self.get_config(name) for name in groups]

        total_motor_id_mask = np.zeros_like(self._all_motor_ids, dtype=bool)
        for config in group_configs:
            if config.motor_ids is not None:
                total_motor_id_mask[config.motor_id_indices] = True

        self._hardware.set_torque_enabled(
            self._all_motor_ids[total_motor_id_mask], engaged)

    def _get_group_states(
            self,
            configs: Sequence[DynamixelGroupConfig],
    ) -> Sequence[RobotState]:
        """Returns the states for the given group configurations."""
        # Make one call to the hardware to get all of the positions/velocities,
        # and extract each individual groups' subset from them.
        all_qpos, all_qvel = self._hardware.read_pos_vel()

        states = []
        for config in configs:
            state = RobotState()
            # Return a blank state if this is a sim-only group.
            if config.motor_ids is None:
                states.append(state)
                continue

            state.qpos = all_qpos[config.motor_id_indices]
            state.qvel = all_qvel[config.motor_id_indices]

            self._calibrate_state(state, config)

            states.append(state)

        self._copy_to_simulation_state(zip(configs, states))
        return states

    def _set_group_states(
            self,
            group_states: Sequence[Tuple[DynamixelGroupConfig, RobotState]],
            block: bool = True):
        """Sets the robot joints to the given states.

        Args:
            group_states: The states to set for each group.
            block: If True, blocks the current thread until completion.
        """
        # Filter out sim-only groups.
        group_states = [
            (config, state)
            for config, state in group_states
            if config.motor_ids is not None and state.qpos is not None
        ]
        # Only write the qpos for the state.
        group_control = [(config, state.qpos) for config, state in group_states]
        self._set_hardware_control(group_control)

        # Block until we've reached the given states.
        if block:
            self._wait_for_desired_states(group_states)

        # Reset the step time.
        self.reset_time()

    def _perform_timestep(
            self,
            group_controls: Sequence[Tuple[DynamixelGroupConfig, np.ndarray]]):
        """Applies the given control values to the robot."""
        self._set_hardware_control(group_controls)
        self._synchronize_timestep()

    def _set_hardware_control(
            self,
            group_control: Sequence[Tuple[DynamixelGroupConfig, np.ndarray]]):
        """Sets the desired hardware positions.

        Args:
            group_control: A list of (group config, control) pairs to write to
                the hardware.
        """
        total_motor_id_mask = np.zeros_like(self._all_motor_ids, dtype=bool)
        total_qpos = np.zeros_like(self._all_motor_ids, dtype=np.float32)

        for config, control in group_control:
            if config.motor_ids is None:
                continue
            if control is not None:
                # TODO(michaelahn): Consider if other control modes need
                # decalibration.
                if config.control_mode == ControlMode.JOINT_POSITION:
                    control = self._decalibrate_qpos(control, config)

                total_motor_id_mask[config.motor_id_indices] = True
                total_qpos[config.motor_id_indices] = control

        if np.any(total_motor_id_mask):
            # TODO(michaeahn): Need to switch control mode if we're not in joint
            # position control.
            self._hardware.write_desired_pos(
                self._all_motor_ids[total_motor_id_mask],
                total_qpos[total_motor_id_mask])
