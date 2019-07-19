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

"""Component implementation for interfacing with a Tracker."""

import logging
from typing import Dict, Optional, Sequence

import numpy as np
from transforms3d.quaternions import mat2quat

from dsuite.components.base import BaseComponent
from dsuite.simulation.sim_scene import SimScene


class TrackerState:
    """Data class that represents the state of the tracker."""

    def __init__(self,
                 pos: Optional[np.ndarray] = None,
                 rot: Optional[np.ndarray] = None):
        """Initializes a new state object.

        Args:
            pos: The (x, y, z) position of the tracked position. The z-axis
                points upwards.
            rot: The (3x3) rotation matrix of the tracked position.
        """
        self.pos = pos
        self.rot = rot


class TrackerGroupConfig:
    """Group configuration for a TrackerComponent."""

    def __init__(self,
                 sim_scene: SimScene,
                 site_name: Optional[str] = None,
                 sim_observation_noise: Optional[float] = None):
        """Initializes a group configuration for a TrackerComponent.

        Args:
            sim_scene: The simulation, used for validation purposes.
            site_name: The site to use for tracking in simulation.
            sim_observation_noise: The range of the observation noise (in
                meters) to apply to the state in simulation.
        """
        self.site_id = None
        if site_name is not None:
            self.site_id = sim_scene.model.site_name2id(site_name)

        self.sim_observation_noise = sim_observation_noise


class TrackerComponent(BaseComponent):
    """Component for reading tracking data."""

    def _process_group(self, **config_kwargs):
        """Processes the configuration for a group."""
        return TrackerGroupConfig(self.sim_scene, **config_kwargs)

    def set_state(self, state_groups: Dict[str, TrackerState]):
        """Sets the tracker to the given initial state.

        Args:
            state_groups: A mapping of control group name to desired position
                and velocity.
        """
        changed_sites = []
        for group_name, state in state_groups.items():
            config = self.get_config(group_name)
            if self._set_site_state(state, config):
                changed_sites.append((config.site_id, state.pos))

        if changed_sites:
            self.sim_scene.sim.forward()
        # Verify that changes occured.
        for site_id, pos in changed_sites:
            if not np.allclose(self.sim_scene.data.site_xpos[site_id, :], pos):
                logging.error(
                    'Site #%d is immutable (modify the XML with a non-zero '
                    'starting position).', site_id)

    def _get_group_states(
            self,
            configs: Sequence[TrackerGroupConfig],
    ) -> Sequence[TrackerState]:
        """Returns the TrackerState for the given groups.

        Args:
            configs: The group configurations to retrieve the states for.

        Returns:
            A list of TrackerState(timestamp, pos, quat, euler).
        """
        return [self._get_site_state(config) for config in configs]

    def _get_site_state(self, config: TrackerGroupConfig) -> TrackerState:
        """Returns the simulation site state for the given group config."""
        state = TrackerState()
        if config.site_id is None:
            return state
        state.pos = self.sim_scene.data.site_xpos[config.site_id]
        state.rot = self.sim_scene.data.site_xmat[config.site_id].reshape((3,
                                                                           3))

        if (config.sim_observation_noise is not None
                and self.random_state is not None):
            amplitude = config.sim_observation_noise / 2.0
            state.pos += self.random_state.uniform(
                low=-amplitude, high=amplitude, size=state.pos.shape)
        return state

    def _set_site_state(self,
                        state: TrackerState,
                        config: TrackerGroupConfig,
                        ignore_z_axis: bool = False) -> bool:
        """Sets the simulation state for the given site."""
        changed = False
        if config.site_id is None:
            return changed
        model = self.sim_scene.model
        if state.pos is not None:
            if ignore_z_axis:
                model.site_pos[config.site_id, :2] = state.pos[:2]
            else:
                model.site_pos[config.site_id, :] = state.pos
            changed = True
        if state.rot is not None:
            rot_quat = mat2quat(state.rot)
            model.site_quat[config.site_id, :] = rot_quat
            changed = True
        return changed
