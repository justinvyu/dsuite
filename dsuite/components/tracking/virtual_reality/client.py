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

"""Client to communicate with a VR device using OpenVR.

Example usage:
>>> client = VrClient()
>>> client.set_devices({'tracker': 1})
"""

import collections
import logging
from typing import Dict, Optional, List, Sequence, Union

import numpy as np
import openvr

from dsuite.components.tracking.virtual_reality.device import VrDevice
from dsuite.components.tracking.virtual_reality.poses import (
    VrPoseBatch, VrCoordinateSystem)
from dsuite.utils.math_utils import average_quaternions


class VrClient:
    """Communicates with a VR device."""

    def __init__(self):
        self._vr_system = None
        self._devices = {}
        self._device_serial_lookup = {}
        self._device_index_lookup = {}
        self._coord_system = VrCoordinateSystem()

        # Attempt to start OpenVR.
        if not openvr.isRuntimeInstalled():
            raise OSError('OpenVR runtime not installed.')

        self._vr_system = openvr.init(openvr.VRApplication_Other)

    def close(self):
        """Cleans up any resources used by the client."""
        if self._vr_system is not None:
            openvr.shutdown()
            self._vr_system = None

    def get_device(self, identifier: Union[int, str]) -> VrDevice:
        """Returns the device with the given name."""
        identifier = str(identifier)
        if identifier in self._device_index_lookup:
            return self._device_index_lookup[identifier]
        if identifier in self._device_serial_lookup:
            return self._device_serial_lookup[identifier]

        self.discover_devices()
        if (identifier not in self._device_index_lookup
                and identifier not in self._device_serial_lookup):
            raise ValueError(
                'Could not find device with name or index: {} (Available: {})'
                .format(identifier, list(self._device_serial_lookup.keys())))

        if identifier in self._device_index_lookup:
            return self._device_index_lookup[identifier]
        return self._device_serial_lookup[identifier]

    def discover_devices(self) -> List[VrDevice]:
        """Returns and caches all connected devices."""
        self._device_index_lookup.clear()
        self._device_serial_lookup.clear()
        devices = []
        for device_index in range(openvr.k_unMaxTrackedDeviceCount):
            device = VrDevice(self._vr_system, device_index)
            if not device.is_connected():
                continue
            devices.append(device)
            self._device_index_lookup[str(device.index)] = device
            self._device_serial_lookup[device.get_serial()] = device
        return devices

    def get_poses(self, time_from_now: float = 0.0) -> VrPoseBatch:
        """Returns a batch of poses that can be queried per device.

        Args:
            time_from_now: The seconds into the future to read poses.
        """
        return VrPoseBatch(self._vr_system, self._coord_system, time_from_now)

    def update_coordinate_system(
            self,
            origin_device: Optional[VrDevice] = None,
            origin_offset: Optional[Sequence[float]] = None,
            device_rotations: Optional[
                Dict[VrDevice, Optional[Sequence[float]]]] = None,
            num_samples: int = 10):
        """Configures the position and rotation of the devices."""
        pos_samples = []
        quat_samples = collections.defaultdict(list)
        device_rotations = device_rotations or {}

        # Collect samples for the devices.
        for _ in range(num_samples):
            pose_batch = self.get_poses()
            if origin_device:
                pos, _ = pose_batch.get_pos_rot(origin_device, raw=True)
                pos_samples.append(pos)
            for device, _ in device_rotations.items():
                _, quat = pose_batch.get_pos_quat(device, raw=True)
                quat_samples[device].append(quat)

        # Set the origin from the mean position.
        if pos_samples:
            pos_mean = np.mean(pos_samples, axis=0)
            pos_std = np.std(pos_samples, axis=0)
            logging.info(
                'Setting device %s as origin.\n> pos-mean: %s\n> pos-std: %s',
                origin_device.get_serial(), str(pos_mean), str(pos_std))
            if origin_offset is not None:
                origin_offset = np.array(origin_offset)
            self._coord_system.set_user_origin(pos_mean, origin_offset)

        # Set the device rotations.
        for device, rot_offset in device_rotations.items():
            offset_quat = None
            if rot_offset is not None:
                offset_quat = np.array(rot_offset)
                assert offset_quat.shape == (4,), \
                    'Rotation offset must be a quaternion.'
            quat_mean = average_quaternions(quat_samples[device])
            self._coord_system.set_user_local_rotation(device.index, quat_mean,
                                                       offset_quat)

    def __enter__(self):
        """Enables use as a context manager."""
        return self

    def __exit__(self, *args):
        """Enables use as a context manager."""
        self.close()

    def __del__(self):
        """Automatically disconnect on destruction."""
        self.close()
