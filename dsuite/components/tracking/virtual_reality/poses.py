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

"""Pose-related logic for OpenVR devices."""

import collections
from typing import Optional, Tuple

import numpy as np
import openvr
from transforms3d.euler import mat2euler
from transforms3d.quaternions import mat2quat, qconjugate, qmult, quat2mat

from dsuite.components.tracking.virtual_reality.device import VrDevice


class VrCoordinateSystem:
    """Stores the most recently returned values for device poses."""

    # Transform from OpenVR space (y upwards) to simulation space (z upwards).
    GLOBAL_TRANSFORM = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                dtype=np.float32)
    # Locally rotate to preserve original orientation.
    LOCAL_TRANSFORM = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                               dtype=np.float32)

    def __init__(self):
        # Per-device cached position and rotation matrix.
        self._cached_pos = collections.defaultdict(lambda: np.zeros(
            3, dtype=np.float32))
        self._cached_rot = collections.defaultdict(lambda: np.identity(
            3, dtype=np.float32))

        self._user_translation = np.zeros(3, dtype=np.float32)
        self._user_local_rotations = collections.defaultdict(
            lambda: np.identity(3, dtype=np.float32))

    def get_cached_pos_rot(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the cached program space position and rotation."""
        return self._cached_pos[index].copy(), self._cached_rot[index].copy()

    def process_from_vr(self, index: int, pos: np.ndarray,
                        rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms from VR space to program space."""
        assert pos.shape == (3,)
        assert rot.shape == (3, 3)
        # If all of the translations are 0, get from the cache.
        if not pos.any():
            return self.get_cached_pos_rot(index)
        # Apply the constant transforms.
        pos = np.matmul(self.GLOBAL_TRANSFORM, pos)
        rot = np.matmul(rot, self.LOCAL_TRANSFORM)
        rot = np.matmul(self.GLOBAL_TRANSFORM, rot)
        # Update the cache.
        self._cached_pos[index] = pos
        self._cached_rot[index] = rot
        return pos.copy(), rot.copy()

    def to_user_space(self, index: int, pos: np.ndarray, rot: np.ndarray):
        """Transforms from program space to user space."""
        pos = pos + self._user_translation
        rot = np.matmul(rot, self._user_local_rotations[index])
        return pos, rot

    def set_user_origin(self,
                        position: np.ndarray,
                        offset: Optional[np.ndarray] = None):
        """Sets the user translation shared between all devices."""
        assert position.shape == (3,), position
        if offset is not None:
            assert offset.shape == (3,), offset
            position = position - offset
        self._user_translation = -position

    def set_user_local_rotation(self,
                                index: int,
                                rot_quat: np.ndarray,
                                offset: Optional[np.ndarray] = None):
        """Sets the device local rotation."""
        assert rot_quat.shape == (4,), "Rotation must be a quaternion."
        if offset is not None:
            assert offset.shape == (4,), offset
            rot_quat = qmult(qconjugate(offset), rot_quat)
        self._user_local_rotations[index] = quat2mat(qconjugate(rot_quat))


class VrPoseBatch:
    """Represents a batch of poses calculated by the OpenVR system."""

    def __init__(self,
                 vr_system,
                 coord_system: VrCoordinateSystem,
                 time_from_now: float = 0.0):
        """Initializes a new pose batch."""
        self._vr_system = vr_system
        self._coord_system = coord_system
        # Query poses for all devices.
        self.poses = self._vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, time_from_now,
            openvr.k_unMaxTrackedDeviceCount)

    def get_pos_rot(self, device: VrDevice,
                    raw: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the 4x4 pose matrix of the given device."""
        if not self.poses[device.index].bPoseIsValid:
            pos, rot = self._coord_system.get_cached_pos_rot(device.index)
        else:
            vr_pose = np.ctypeslib.as_array(
                self.poses[device.index].mDeviceToAbsoluteTracking[:],
                shape=(3, 4))
            # Check that the pose is valid.
            # If all of the translations are 0, get from the cache.
            assert vr_pose.shape == (3, 4)
            pos, rot = self._coord_system.process_from_vr(
                device.index, vr_pose[:, 3], vr_pose[:, :3])
        if not raw:
            pos, rot = self._coord_system.to_user_space(device.index, pos, rot)
        return pos, rot

    def get_pos_euler(
            self,
            device: VrDevice,
            raw: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the translation and euler rotation of the given device."""
        pos, rot = self.get_pos_rot(device, raw)
        ai, aj, ak = mat2euler(rot)
        return pos, np.array([ai, aj, ak], dtype=np.float32)

    def get_pos_quat(
            self,
            device: VrDevice,
            raw: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the translation quaternion rotation of the given device."""
        pos, rot = self.get_pos_rot(device, raw)
        return pos, mat2quat(rot).astype(np.float32)
