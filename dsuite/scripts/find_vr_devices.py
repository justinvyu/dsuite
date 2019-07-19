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

"""Allows the user to interact with the OpenVR client."""

import cmd
import logging

from transforms3d.euler import euler2quat

from dsuite.components.tracking.virtual_reality import VrClient

INTRODUCTION = """Interactive shell for using the OpenVR client.

Type `help` or `?` to list commands.
"""


class VrDeviceShell(cmd.Cmd):
    """Implements a command-line interface for using the OpenVR client."""

    intro = INTRODUCTION
    prompt = '>>> '

    def __init__(self, client: VrClient):
        super().__init__()
        self.client = client

    def do_list(self, unused_arg):
        """Lists the available devices on the machine."""
        devices = self.client.discover_devices()
        if not devices:
            print('No devices found!')
            return
        for device in devices:
            print(device.get_summary())

    def do_pose(self, args):
        """Prints the pose for the given device."""
        names = args.split()
        devices = [self.client.get_device(name) for name in names]

        pose_batch = self.client.get_poses()
        for device in devices:
            pos, euler = pose_batch.get_pos_euler(device)
            print(device.get_summary())
            print('> Pos: [{:.3f} {:.3f} {:.3f}]'.format(*pos))
            print('> Rot: [{:.3f} {:.3f} {:.3f}]'.format(*euler))

    def do_origin(self, args):
        """Sets the given device number as the origin."""
        components = args.strip().split()
        if not components or not components[0]:
            print('Must provide device number.')
            return
        device_id = components[0]
        position_offsets = None
        if len(components) >= 4:
            position_offsets = list(map(float, components[1:4]))

        device = self.client.get_device(device_id)
        print('Setting device {} as the origin.'.format(
            device.get_model_name()))

        self.client.update_coordinate_system(
            origin_device=device, origin_offset=position_offsets)

    def do_rotation(self, args):
        """Sets the rotation for the given device number."""
        components = args.strip().split()
        if not components or not components[0]:
            print('Must provide device number.')
            return
        device_id = components[0]
        rotation_offsets = None
        if len(components) >= 4:
            rotation_offsets = list(map(float, components[1:4]))
            rotation_offsets = euler2quat(*rotation_offsets)

        device = self.client.get_device(device_id)
        print('Setting rotation for device {}.'.format(device.get_model_name()))

        self.client.update_coordinate_system(
            device_rotations={device: rotation_offsets})

    def emptyline(self):
        """Overrides behavior when an empty line is sent."""


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with VrClient() as vr_client:
        repl = VrDeviceShell(vr_client)
        repl.cmdloop()
