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

"""Implementation of TrackerController using OpenVR to track devices."""

import logging
from typing import Dict, Optional, Sequence, Union

from transforms3d.quaternions import mat2quat

from dsuite.controllers.tracking import (TrackerController, TrackerState)
from dsuite.controllers.tracking.tracker import TrackerGroupConfig


class VrTrackerGroupConfig(TrackerGroupConfig):
    """Stores group configuration for a VrTrackerController."""

    def __init__(self,
                 *args,
                 device_identifier: Optional[Union[int, str]] = None,
                 is_origin: bool = False,
                 mimic_in_sim: bool = False,
                 mimic_ignore_z_axis: bool = False,
                 mimic_ignore_rotation: bool = False,
                 **kwargs):
        """Initializes a new configuration for a VrTrackerController group.

        Args:
            device_identifier: The device index or device serial string of the
                tracking device.
            is_origin: If True, the (0, 0, 0) world position is inferred to be
                at this group's location.
            mimic_in_sim: If True, updates the simulation sites with the tracked
                positions.
            mimic_ignore_z_axis: If True, the simulation site is only updated
                with the x and y position.
            mimic_ignore_rotation: If True, the simulation site is not updated
                with the rotation.
        """
        super().__init__(*args, **kwargs)
        self.device_identifier = device_identifier
        self.is_origin = is_origin
        self.mimic_in_sim = mimic_in_sim
        self.mimic_ignore_z_axis = mimic_ignore_z_axis
        self.mimic_ignore_rotation = mimic_ignore_rotation


class VrTrackerController(TrackerController):
    """Controller for reading tracking data from a HTC Vive."""

    # Cached VR client that is shared for the application lifetime.
    _VR_CLIENT = None

    def __init__(self, *args, **kwargs):
        """Initializes a ViveTrackerController."""
        super().__init__(*args, **kwargs)
        if self._VR_CLIENT is None:
            from dsuite.controllers.tracking.virtual_reality import VrClient
            self._VR_CLIENT = VrClient()

        # Check that all devices exist.
        for name, group in self.groups.items():
            if group.device_identifier is None:
                if group.site_id is not None:
                    print('Using simulation site for group "{}"'.format(name))
                continue
            device = self._VR_CLIENT.get_device(group.device_identifier)
            print('Assigning group "{}" with device: {}'.format(
                name, device.get_summary()))

    def _process_group(self, **config_kwargs):
        """Processes the configuration for a group."""
        return VrTrackerGroupConfig(self.sim_scene, **config_kwargs)

    def _get_group_states(
            self,
            configs: Sequence[VrTrackerGroupConfig],
    ) -> Sequence[TrackerState]:
        """Returns the TrackerState for the given groups.

        Args:
            configs: The group configurations to retrieve the states for.

        Returns:
            A list of TrackerState(timestamp, pos, quat, euler).
        """
        pose_batch = self._VR_CLIENT.get_poses()
        simulation_changed = False
        states = []
        for config in configs:
            if config.device_identifier is None:
                # Fall back to simulation site.
                states.append(self._get_site_state(config))
                continue
            device = self._VR_CLIENT.get_device(config.device_identifier)
            pos, rot = pose_batch.get_pos_rot(device)
            state = TrackerState(pos, rot)

            # Mimic the site in simulation if needed.
            if config.mimic_in_sim:
                mimic_state = TrackerState(
                    state.pos,
                    None if config.mimic_ignore_rotation else state.rot)
                simulation_changed |= self._set_site_state(
                    mimic_state,
                    config,
                    ignore_z_axis=config.mimic_ignore_z_axis)

            states.append(state)

        if simulation_changed:
            self.sim_scene.sim.forward()
        return states

    def set_state(self, state_groups: Dict[str, TrackerState]):
        """Sets the tracker to the given initial state.

        Args:
            state_groups: A mapping of control group name to desired position
                and velocity.
        """
        origin_device = None
        origin_offset = None
        device_rotations = {}
        ignored_group_positions = []

        for group_name, state in state_groups.items():
            config = self.get_config(group_name)
            if config.device_identifier is None:
                continue
            device = self._VR_CLIENT.get_device(config.device_identifier)

            # Only respect the set position for the origin device.
            if state.pos is not None:
                assert state.pos.shape == (3,), state.pos
                if config.is_origin:
                    if origin_device is not None:
                        raise ValueError('Cannot have more than one origin.')
                    origin_device = device
                    origin_offset = state.pos
                else:
                    ignored_group_positions.append((group_name, device,
                                                    state.pos))

            # If a rotation if not given, the device's rotation will be set to
            # (0, 0, 0).
            device_rotations[device] = None
            if state.rot is not None:
                assert state.rot.shape == (3, 3), state.rot
                device_rotations[device] = mat2quat(state.rot)

        # Set the coordinate system.
        self._VR_CLIENT.update_coordinate_system(origin_device, origin_offset,
                                                 device_rotations)

        # Log any ignored positions.
        if ignored_group_positions:
            pose_batch = self._VR_CLIENT.get_poses()
            for group_name, device, desired_pos in ignored_group_positions:
                actual_pos, _ = pose_batch.get_pos_rot(device)
                logging.warning(
                    'Ignored setting position for group: "%s" - '
                    'Desired position: %s, Actual position: %s', group_name,
                    str(desired_pos.tolist()), str(actual_pos.tolist()))
