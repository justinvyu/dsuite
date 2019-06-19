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

"""Base logic for hardware robots."""

import abc
import logging
import time
from typing import Iterable, Optional, Tuple

import numpy as np

from dsuite.controllers.robot import RobotController, RobotState
from dsuite.controllers.robot.config import RobotGroupConfig


class HardwareRobotGroupConfig(RobotGroupConfig):
    """Stores group configuration for a HardwareRobotController."""

    def __init__(self,
                 *args,
                 calib_scale: Optional[Iterable[float]] = None,
                 calib_offset: Optional[Iterable[float]] = None,
                 **kwargs):
        """Initializes a new configuration for a HardwareRobotController group.

        Args:
            calib_scale: A scaling factor that is multipled with state to
                convert from controller state space to hardware state space,
                and divides control to convert from hardware control space to
                controller control space.
            calib_offset: An offset that is added to state to convert from
                controller state space to hardware state space, and subtracted
                from control to convert from hardware control space to
                controller control space.
        """
        super().__init__(*args, **kwargs)

        self.calib_scale = None
        if calib_scale is not None:
            self.calib_scale = np.array(calib_scale, dtype=np.float32)

        self.calib_offset = None
        if calib_offset is not None:
            self.calib_offset = np.array(calib_offset, dtype=np.float32)


class HardwareRobotController(RobotController, metaclass=abc.ABCMeta):
    """Base controller for hardware robots."""

    def __init__(self, *args, **kwargs):
        """Initializes the controller."""
        super().__init__(*args, **kwargs)
        self.reset_time()

    @property
    def time(self) -> float:
        """Returns the time (total sum of timesteps) since the last reset."""
        return self._time

    def reset_time(self):
        """Resets the timer for the controller."""
        self._last_reset_time = time.time()
        self._time = 0

    def _process_group(self, **config_kwargs) -> HardwareRobotGroupConfig:
        """Processes the configuration for a group."""
        return HardwareRobotGroupConfig(self.sim_scene, **config_kwargs)

    def _calibrate_state(self, state: RobotState,
                         group_config: HardwareRobotGroupConfig):
        """Converts the given state from hardware space to controller space."""
        # Calculate qpos' = qpos * scale + offset, and qvel' = qvel * scale.
        if group_config.calib_scale is not None:
            assert state.qpos.shape == group_config.calib_scale.shape
            assert state.qvel.shape == group_config.calib_scale.shape
            state.qpos *= group_config.calib_scale
            state.qvel *= group_config.calib_scale
        if group_config.calib_offset is not None:
            assert state.qpos.shape == group_config.calib_offset.shape
            # Only apply the offset to positions.
            state.qpos += group_config.calib_offset

    def _decalibrate_qpos(self, qpos: np.ndarray,
                          group_config: HardwareRobotGroupConfig) -> np.ndarray:
        """Converts the given position from controller to hardware space."""
        # Calculate qpos' = (qpos - offset) / scale.
        if group_config.calib_offset is not None:
            assert qpos.shape == group_config.calib_offset.shape
            qpos = qpos - group_config.calib_offset
        if group_config.calib_scale is not None:
            assert qpos.shape == group_config.calib_scale.shape
            qpos = qpos / group_config.calib_scale
        return qpos

    def _synchronize_timestep(self, minimum_sleep: float = 1e-4):
        """Waits for one timestep to elapse."""
        # Block the thread such that we've waited at least `step_duration` time
        # since the last call to `_synchronize_timestep`.
        time_since_reset = time.time() - self._last_reset_time
        elapsed_time = time_since_reset - self._time
        remaining_step_time = self.sim_scene.step_duration - elapsed_time
        if remaining_step_time > minimum_sleep:
            time.sleep(remaining_step_time)
        elif remaining_step_time < 0:
            logging.warning('Exceeded timestep by %0.4fs', -remaining_step_time)

        # Update the current time, relative to the last reset time.
        self._time = time.time() - self._last_reset_time

    def _wait_for_desired_states(
            self,
            desired_states: Iterable[Tuple[RobotGroupConfig, RobotState]],
            error_tolerance: float = 1e-2,
            timeout: float = 3.0,
            poll_interval: float = 0.5,
    ):
        """Polls the current state until it reaches the desired state."""
        # Poll for the hardware move command to complete.
        configs, desired_states = zip(*desired_states)
        start_time = time.time()
        while True:
            cur_states = self._get_group_states(configs)
            all_complete = True
            for cur_state, des_state in zip(cur_states, desired_states):
                logging.debug('Waiting for reset (Err: %1.4f)',
                              np.linalg.norm(cur_state.qpos - des_state.qpos))
                if not np.allclose(
                        cur_state.qpos, des_state.qpos, atol=error_tolerance):
                    all_complete = False
                    break
            if all_complete:
                return
            if time.time() - start_time > timeout:
                logging.warning('Reset timed out after %1.1fs', timeout)
                return
            time.sleep(poll_interval)

    def _copy_to_simulation_state(
            self, group_states: Iterable[Tuple[RobotGroupConfig, RobotState]]):
        """Copies the given states to the simulation."""
        for config, state in group_states:
            # Skip if this is a hardware-only group.
            if config.qpos_indices is None:
                continue
            if state.qpos is not None:
                self.sim_scene.data.qpos[config.qpos_indices] = state.qpos
            if state.qvel is not None:
                self.sim_scene.data.qvel[config.qvel_indices] = state.qvel

        # Recalculate forward dynamics.
        self.sim_scene.sim.forward()
        self.sim_scene.renderer.refresh_window()
